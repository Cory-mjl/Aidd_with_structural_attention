import os
import torch
from torch.utils.data import DataLoader

from .config import (
    MODEL_PATH,
    DATA_ROOT,
    SPLIT_PATH,
    CROSSDOCK_SUBDIR,
    VALID_LAST_N,
    HID_DIM,
    N_HEADS,
    MAX_LEN,
    BATCH_SIZE,
    LR,
    WEIGHT_DECAY,
    EPOCHS,
    SEED,
)
from .data import CrossDockedPocket10Dataset, collate_fn
from .models import Pocket2SmilesModel
from .utils import info_nce, set_seed


def masked_mean(x, mask):
    mask = mask.unsqueeze(-1)
    denom = mask.sum(dim=1).clamp_min(1)
    return (x * mask).sum(dim=1) / denom


def build_model(feat_dim):
    return Pocket2SmilesModel(
        feat_dim=feat_dim,
        hid_dim=HID_DIM,
        model_path=MODEL_PATH,
        n_heads=N_HEADS,
    )


def train():
    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset = CrossDockedPocket10Dataset(
        DATA_ROOT,
        split_path=SPLIT_PATH,
        subset="train",
        subdir=CROSSDOCK_SUBDIR,
    )
    if len(train_dataset) == 0:
        raise RuntimeError("No train samples found. Check DATA_ROOT and SPLIT_PATH.")

    if VALID_LAST_N > 0 and len(train_dataset.samples) > VALID_LAST_N:
        train_dataset.samples = train_dataset.samples[:-VALID_LAST_N]
        valid_dataset = CrossDockedPocket10Dataset(
            DATA_ROOT,
            split_path=SPLIT_PATH,
            subset="train",
            subdir=CROSSDOCK_SUBDIR,
        )
        valid_dataset.samples = valid_dataset.samples[-VALID_LAST_N:]
    else:
        valid_dataset = None
    if len(dataset) == 0:
        raise RuntimeError("No samples found. Check DATA_ROOT and file structure.")

    feat_dim = train_dataset[0][0].size(-1)
    model = build_model(feat_dim).to(device)
    tokenizer = model.decoder.tokenizer

    loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda b: collate_fn(b, tokenizer, max_len=MAX_LEN),
    )
    valid_loader = None
    if valid_dataset is not None:
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            collate_fn=lambda b: collate_fn(b, tokenizer, max_len=MAX_LEN),
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    model.train()
    for epoch in range(EPOCHS):
        for step, batch in enumerate(loader):
            batch = [t.to(device) if torch.is_tensor(t) else t for t in batch]
            (
                pocket_feats,
                pocket_coords,
                pocket_mask,
                drug_feats,
                drug_coords,
                drug_mask,
                input_ids,
                attention_mask,
                labels,
            ) = batch

            logits, pocket_mem, drug_mem = model(
                pocket_feats,
                pocket_coords,
                pocket_mask,
                drug_feats,
                drug_coords,
                drug_mask,
                input_ids,
                attention_mask,
            )

            vocab_size = logits.size(-1)
            gen_loss = torch.nn.functional.cross_entropy(
                logits.view(-1, vocab_size), labels.view(-1)
            )

            pocket_repr = masked_mean(pocket_mem, pocket_mask)
            drug_repr = masked_mean(drug_mem, drug_mask)
            cl_loss = info_nce(pocket_repr, drug_repr)

            loss = gen_loss + 0.1 * cl_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if step % 50 == 0:
                print(f"epoch {epoch} step {step} loss {loss.item():.4f} gen {gen_loss.item():.4f} cl {cl_loss.item():.4f}")

        if valid_loader is not None:
            model.eval()
            total = 0.0
            count = 0
            with torch.no_grad():
                for batch in valid_loader:
                    batch = [t.to(device) if torch.is_tensor(t) else t for t in batch]
                    (
                        pocket_feats,
                        pocket_coords,
                        pocket_mask,
                        drug_feats,
                        drug_coords,
                        drug_mask,
                        input_ids,
                        attention_mask,
                        labels,
                    ) = batch
                    logits, pocket_mem, drug_mem = model(
                        pocket_feats,
                        pocket_coords,
                        pocket_mask,
                        drug_feats,
                        drug_coords,
                        drug_mask,
                        input_ids,
                        attention_mask,
                    )
                    vocab_size = logits.size(-1)
                    gen_loss = torch.nn.functional.cross_entropy(
                        logits.view(-1, vocab_size), labels.view(-1)
                    )
                    total += gen_loss.item()
                    count += 1
            print(f"epoch {epoch} valid_gen_loss {total / max(count, 1):.4f}")
            model.train()
    save_path = os.path.join(DATA_ROOT, "pocket2smiles.pt")
    torch.save(model.state_dict(), save_path)
    print(f"Saved to {save_path}")


if __name__ == "__main__":
    train()
