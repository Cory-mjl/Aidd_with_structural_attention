import glob
import os

import torch
from rdkit import Chem
from torch.utils.data import Dataset
from Bio.PDB import PDBParser


AA_3TO1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}
AA_LIST = list("ARNDCQEGHILKMFPSTWYV")
UNK_AA = "X"


class ProteinTokenizer:
    def __init__(self):
        # Minimal AA vocabulary with pad + UNK.
        vocab = [UNK_AA] + AA_LIST
        self.pad_token = "<pad>"
        self.pad_idx = 0
        self.vocab = {self.pad_token: self.pad_idx}
        for aa in vocab:
            self.vocab[aa] = len(self.vocab)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def encode(self, seq):
        ids = []
        for aa in seq:
            if aa == " ":
                continue
            ids.append(self.vocab.get(aa, self.vocab[UNK_AA]))
        return torch.tensor(ids, dtype=torch.long)

    @property
    def vocab_size(self):
        return len(self.vocab)


def read_pocket_pdb(pdb_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("pocket", pdb_path)
    try:
        first_model = next(structure.get_models())
    except StopIteration:
        raise ValueError(f"No models found in {pdb_path}")

    residues = []
    for residue in first_model.get_residues():
        res_type, res_id, _ = residue.get_id()
        if res_type != " ":
            continue
        coords = [atom.coord for atom in residue.get_atoms()]
        if not coords:
            continue
        # Residue center as the structural coordinate.
        coord = torch.tensor(coords, dtype=torch.float).mean(dim=0)
        residues.append((residue.get_parent().id, res_id, residue.get_resname(), coord))

    # Enforce deterministic ordering across chains and residue ids.
    residues.sort(key=lambda x: (x[0], x[1]))
    coords = [r[3] for r in residues]
    seq = "".join(AA_3TO1.get(r[2], UNK_AA) for r in residues)

    if not coords:
        raise ValueError(f"No residues parsed from {pdb_path}")

    return seq, torch.stack(coords, dim=0)


def atom_features_from_atomic_num(atomic_num, degree=0, formal_charge=0, aromatic=0):
    return torch.tensor([atomic_num, degree, formal_charge, aromatic], dtype=torch.float)


def read_ligand_sdf(sdf_path):
    mol = Chem.MolFromMolFile(sdf_path, removeHs=False)
    if mol is None:
        return None, None, None
    conf = mol.GetConformer()
    coords = []
    feats = []
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        coords.append([pos.x, pos.y, pos.z])
        # Simple atom-level features; extend as needed.
        feats.append(
            atom_features_from_atomic_num(
                atom.GetAtomicNum(),
                degree=atom.GetDegree(),
                formal_charge=atom.GetFormalCharge(),
                aromatic=int(atom.GetIsAromatic()),
            )
        )
    smiles = Chem.MolToSmiles(mol)
    return torch.tensor(coords, dtype=torch.float), torch.stack(feats, dim=0), smiles


def _load_split_entries(split_path, subset):
    split = torch.load(split_path)
    if subset not in split:
        raise KeyError(f"Subset {subset} not in split file.")
    return split[subset]


class CrossDockedPocket10Dataset(Dataset):
    def __init__(self, root_dir, prefer_min=True, split_path=None, subset=None, subdir="crossdocked_pocket10"):
        self.samples = []
        if split_path and subset:
            entries = _load_split_entries(split_path, subset)
            for pocket_fn, ligand_fn in entries:
                # split_by_name.pt stores relative paths under crossdocked_pocket10
                pocket_pdb = os.path.join(root_dir, subdir, pocket_fn)
                ligand_sdf = os.path.join(root_dir, subdir, ligand_fn)
                if os.path.exists(pocket_pdb) and os.path.exists(ligand_sdf):
                    self.samples.append((pocket_pdb, ligand_sdf))
        else:
            min_files = glob.glob(
                os.path.join(root_dir, "**/*_lig_tt_min_0_pocket10.pdb"), recursive=True
            )
            docked_files = glob.glob(
                os.path.join(root_dir, "**/*_lig_tt_docked_*_pocket10.pdb"),
                recursive=True,
            )

            def add_sample(pocket_pdb):
                sdf = pocket_pdb.replace("_pocket10.pdb", ".sdf")
                if not os.path.exists(sdf):
                    return
                self.samples.append((pocket_pdb, sdf))

            if prefer_min:
                for p in min_files:
                    add_sample(p)
                min_bases = {p.replace("_pocket10.pdb", "") for p in min_files}
                for p in docked_files:
                    base = p.replace("_pocket10.pdb", "")
                    if base in min_bases:
                        continue
                    add_sample(p)
            else:
                for p in docked_files:
                    add_sample(p)
                for p in min_files:
                    add_sample(p)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pocket_pdb, sdf_path = self.samples[idx]
        pocket_seq, pocket_coords = read_pocket_pdb(pocket_pdb)
        drug_coords, drug_feats, smiles = read_ligand_sdf(sdf_path)
        if smiles is None:
            raise ValueError(f"Failed to read ligand from {sdf_path}")
        pocket_seq_spaced = " ".join(list(pocket_seq))
        return pocket_seq_spaced, pocket_coords, drug_feats, drug_coords, smiles


def pad_stack(tensors, pad_value=0.0):
    max_len = max(t.size(0) for t in tensors)
    if tensors[0].dim() == 1:
        out = tensors[0].new_full((len(tensors), max_len), pad_value)
    else:
        feat_dim = tensors[0].size(1)
        out = tensors[0].new_full((len(tensors), max_len, feat_dim), pad_value)
    mask = torch.zeros(len(tensors), max_len, dtype=torch.bool)
    for i, t in enumerate(tensors):
        out[i, : t.size(0)] = t
        mask[i, : t.size(0)] = 1
    return out, mask


def collate_fn(batch, tokenizer, max_len=128, protein_tokenizer=None):
    pocket_seq, pocket_coords, drug_feats, drug_coords, smiles = zip(*batch)

    if protein_tokenizer is None:
        protein_tokenizer = ProteinTokenizer()
    # Protein tokens from space-separated AA sequence.
    pocket_tokens = [protein_tokenizer.encode(seq) for seq in pocket_seq]
    pocket_tokens, pocket_mask = pad_stack(pocket_tokens, pad_value=protein_tokenizer.pad_idx)
    pocket_coords, _ = pad_stack(pocket_coords)
    drug_feats, drug_mask = pad_stack(drug_feats)
    drug_coords, _ = pad_stack(drug_coords)

    tok = tokenizer(list(smiles), padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    input_ids = tok["input_ids"]
    attention_mask = tok["attention_mask"]
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100

    return (
        pocket_tokens,
        pocket_coords,
        pocket_mask,
        drug_feats,
        drug_coords,
        drug_mask,
        input_ids,
        attention_mask,
        labels,
    )
