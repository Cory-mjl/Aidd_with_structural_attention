import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem


def load_kg_triples(path):
    triples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            h, r, t = line.split()
            triples.append((h, r, t))
    return triples


def load_embedding_index(ids_path, emb_path):
    # Load node ids + embeddings, normalize for cosine similarity.
    with open(ids_path, "r", encoding="utf-8") as f:
        ids = [line.strip() for line in f if line.strip()]
    embs = np.load(emb_path)
    if embs.shape[0] != len(ids):
        raise ValueError("Node ids and embeddings size mismatch.")
    embs = torch.from_numpy(embs).float()
    embs = torch.nn.functional.normalize(embs, dim=-1)
    id_to_idx = {k: i for i, k in enumerate(ids)}
    return ids, id_to_idx, embs


def build_degree(triples, id_to_idx):
    # Simple undirected degree count as a KG prior.
    deg = torch.zeros(len(id_to_idx), dtype=torch.float)
    for h, _, t in triples:
        if h in id_to_idx:
            deg[id_to_idx[h]] += 1
        if t in id_to_idx:
            deg[id_to_idx[t]] += 1
    return deg


def smiles_to_morgan(smiles_list, n_bits=2048, radius=2):
    # Fallback query embedding if KG embeddings are fingerprint-based.
    embs = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            embs.append(None)
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        arr = np.zeros((n_bits,), dtype=np.float32)
        Chem.DataStructs.ConvertToNumpyArray(fp, arr)
        embs.append(arr)
    embs = [e for e in embs if e is not None]
    if not embs:
        return None
    embs = torch.from_numpy(np.stack(embs)).float()
    embs = torch.nn.functional.normalize(embs, dim=-1)
    return embs


class KGReranker:
    def __init__(self, ids, id_to_idx, embs, deg=None, deg_weight=0.1):
        self.ids = ids
        self.id_to_idx = id_to_idx
        self.embs = embs
        self.deg = deg if deg is not None else torch.zeros(len(ids))
        self.deg_weight = deg_weight

    def score(self, query_embs, top_k=20):
        sims = query_embs @ self.embs.t()
        topk_sim, topk_idx = torch.topk(sims, k=min(top_k, sims.size(1)), dim=1)
        deg = self.deg[topk_idx]
        score = topk_sim.mean(dim=1) + self.deg_weight * torch.log1p(deg).mean(dim=1)
        return score, topk_idx, topk_sim

    def rerank(self, smiles_list, smiles_embs, top_k=20):
        score, topk_idx, topk_sim = self.score(smiles_embs, top_k=top_k)
        scored = []
        for i, smi in enumerate(smiles_list):
            neighbors = [self.ids[j] for j in topk_idx[i].tolist()]
            scored.append(
                {
                    "smiles": smi,
                    "kg_score": float(score[i]),
                    "neighbors": neighbors,
                    "neighbor_sims": topk_sim[i].tolist(),
                }
            )
        scored.sort(key=lambda x: x["kg_score"], reverse=True)
        return scored
