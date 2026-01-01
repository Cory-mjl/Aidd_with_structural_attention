import os
import torch

from .config import (
    MODEL_PATH,
    DATA_ROOT,
    SPLIT_PATH,
    CROSSDOCK_SUBDIR,
    HID_DIM,
    N_HEADS,
    MAX_LEN,
    KG_TRIPLES_PATH,
    KG_NODE_IDS_PATH,
    KG_NODE_EMB_PATH,
    KG_TOPK,
    KG_DEG_WEIGHT,
)
from .data import CrossDockedPocket10Dataset
from .models import Pocket2SmilesModel
from .kg_rerank import load_kg_triples, load_embedding_index, build_degree, smiles_to_morgan, KGReranker
from .utils import top_k_top_p_filtering, smiles_valid


@torch.no_grad()
def sample_smiles(model, pocket_feats, pocket_coords, pocket_mask, max_len=128, top_k=40, top_p=0.9):
    model.eval()
    device = pocket_feats.device
    tokenizer = model.decoder.tokenizer

    pocket_mem = model.pocket_enc(pocket_feats, pocket_coords, mask=pocket_mask)
    input_ids = torch.full((pocket_feats.size(0), 1), tokenizer.bos_token_id or tokenizer.eos_token_id, device=device)
    attention_mask = torch.ones_like(input_ids)

    for _ in range(max_len - 1):
        logits = model.decoder(input_ids, attention_mask, pocket_mem, pocket_mask=pocket_mask)
        next_logits = logits[:, -1, :]
        next_logits = top_k_top_p_filtering(next_logits, top_k=top_k, top_p=top_p)
        probs = torch.softmax(next_logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_id], dim=1)
        attention_mask = torch.ones_like(input_ids)

    smiles = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
    return smiles


def infer_one(index=0, checkpoint_path=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = CrossDockedPocket10Dataset(
        DATA_ROOT,
        split_path=SPLIT_PATH,
        subset="test",
        subdir=CROSSDOCK_SUBDIR,
    )
    feat_dim = dataset[0][0].size(-1)

    model = Pocket2SmilesModel(feat_dim=feat_dim, hid_dim=HID_DIM, model_path=MODEL_PATH, n_heads=N_HEADS).to(device)
    if checkpoint_path:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    # use dataset sample to build pocket input
    pocket_feats, pocket_coords, _, _, _ = dataset[index]
    pocket_feats = pocket_feats.unsqueeze(0).to(device)
    pocket_coords = pocket_coords.unsqueeze(0).to(device)
    pocket_mask = torch.ones(pocket_feats.size()[:2], dtype=torch.bool, device=device)

    smiles = sample_smiles(model, pocket_feats, pocket_coords, pocket_mask, max_len=MAX_LEN)
    smiles = [s for s in smiles if smiles_valid(s)]

    if os.path.exists(KG_NODE_IDS_PATH) and os.path.exists(KG_NODE_EMB_PATH):
        ids, id_to_idx, embs = load_embedding_index(KG_NODE_IDS_PATH, KG_NODE_EMB_PATH)
        deg = None
        if os.path.exists(KG_TRIPLES_PATH):
            triples = load_kg_triples(KG_TRIPLES_PATH)
            deg = build_degree(triples, id_to_idx)
        reranker = KGReranker(ids, id_to_idx, embs, deg=deg, deg_weight=KG_DEG_WEIGHT)
        query_embs = smiles_to_morgan(smiles)
        if query_embs is not None:
            scored = reranker.rerank(smiles, query_embs, top_k=KG_TOPK)
            smiles = [s["smiles"] for s in scored]

    print(smiles[:5])


if __name__ == "__main__":
    infer_one()
