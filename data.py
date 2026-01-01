import glob
import os

import torch
from rdkit import Chem
from torch.utils.data import Dataset
from Bio.PDB import PDBParser


# 三字母残基名 -> 一字母
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

# 20 种标准氨基酸
AA_LIST = list("ARNDCQEGHILKMFPSTWYV")
UNK_AA = "X"


class ProteinTokenizer:
    """
    将蛋白氨基酸序列（如 'ARND...' 或 'A R N D ...'）编码为 token id
    vocab: <pad>=0, 其余为 X + 20AA
    """
    def __init__(self):
        vocab = [UNK_AA] + AA_LIST
        self.pad_token = "<pad>"
        self.pad_idx = 0
        self.vocab = {self.pad_token: self.pad_idx}
        for aa in vocab:
            self.vocab[aa] = len(self.vocab)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def encode(self, seq: str) -> torch.Tensor:
        """
        支持两种输入：
        1) 'ARNDC...'
        2) 'A R N D C ...'（带空格）
        """
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
    """
    读取 pocket pdb：
    - 取第一个 model
    - 仅保留标准残基(res_type == ' ')
    - 每个残基坐标 = 残基内所有原子坐标均值
    - seq 与 coords 顺序严格一致
    返回：
        seq: str, 例如 'ARND...'
        coords: Tensor [N_res, 3]
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("pocket", pdb_path)
    try:
        first_model = next(structure.get_models())
    except StopIteration:
        raise ValueError(f"No models found in {pdb_path}")

    residues = []
    for residue in first_model.get_residues():
        res_type, res_id, icode = residue.get_id()
        if res_type != " ":
            continue

        coords = [atom.coord for atom in residue.get_atoms()]
        if not coords:
            continue

        coord = torch.tensor(coords, dtype=torch.float).mean(dim=0)

        # 修改点1：把 insertion code 也保存/排序，避免 100A/100B 顺序错乱
        chain_id = residue.get_parent().id
        residues.append((chain_id, res_id, icode, residue.get_resname(), coord))

    # 修改点1：排序加入 insertion code
    residues.sort(key=lambda x: (x[0], x[1], str(x[2])))

    coords = [r[4] for r in residues]
    seq = "".join(AA_3TO1.get(r[3], UNK_AA) for r in residues)

    if not coords:
        raise ValueError(f"No residues parsed from {pdb_path}")

    return seq, torch.stack(coords, dim=0)


def atom_features_from_atomic_num(atomic_num, degree=0, formal_charge=0, aromatic=0):
    """简单原子特征: [原子序数, 度, 形式电荷, 是否芳香]"""
    return torch.tensor([atomic_num, degree, formal_charge, aromatic], dtype=torch.float)


def read_ligand_sdf(sdf_path):
    """
    读取配体 SDF：
    返回：
        coords: Tensor [N_atom, 3]
        feats : Tensor [N_atom, 4]
        smiles: str
    """
    mol = Chem.MolFromMolFile(sdf_path, removeHs=False)
    if mol is None:
        return None, None, None

    # 修改点2：更稳健地处理 conformer 缺失
    if mol.GetNumConformers() == 0:
        return None, None, None

    conf = mol.GetConformer()
    coords = []
    feats = []
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        coords.append([pos.x, pos.y, pos.z])
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
    """从 torch.save 的 split 文件中读取 train/val/test 子集条目"""
    split = torch.load(split_path)
    if subset not in split:
        raise KeyError(f"Subset {subset} not in split file.")
    return split[subset]


class CrossDockedPocket10Dataset(Dataset):
    """
    每个样本返回：
        pocket_seq_spaced (str) : 'A R N ...'
        pocket_coords (Tensor)  : [N_res, 3]
        drug_feats (Tensor)     : [N_atom, 4]
        drug_coords (Tensor)    : [N_atom, 3]
        smiles (str)
    """
    def __init__(self, root_dir, prefer_min=True, split_path=None, subset=None, subdir="crossdocked_pocket10"):
        self.samples = []

        if split_path and subset:
            entries = _load_split_entries(split_path, subset)
            for pocket_fn, ligand_fn in entries:
                pocket_pdb = os.path.join(root_dir, subdir, pocket_fn)
                ligand_sdf = os.path.join(root_dir, subdir, ligand_fn)
                if os.path.exists(pocket_pdb) and os.path.exists(ligand_sdf):
                    self.samples.append((pocket_pdb, ligand_sdf))
        else:
            min_files = glob.glob(os.path.join(root_dir, "**/*_lig_tt_min_0_pocket10.pdb"), recursive=True)
            docked_files = glob.glob(os.path.join(root_dir, "**/*_lig_tt_docked_*_pocket10.pdb"), recursive=True)

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

        # 仍保持你原来的“空格分隔序列”格式
        pocket_seq_spaced = " ".join(list(pocket_seq))
        return pocket_seq_spaced, pocket_coords, drug_feats, drug_coords, smiles


def pad_stack(tensors, pad_value=0.0):
    """
    对可变长度序列做 padding
    - 若 tensor 为 1D: [L] -> [B, Lmax]
    - 若 tensor 为 2D: [L, D] -> [B, Lmax, D]
    同时返回 mask: [B, Lmax]，True 表示真实位置
    """
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
    """
    DataLoader 用的 collate_fn：
    - pocket: 序列 -> token ids (pad) + coords (pad) + mask
    - ligand: atom feats/coords (pad) + mask
    - smiles: tokenizer -> input_ids/attention_mask/labels
    """
    pocket_seq, pocket_coords, drug_feats, drug_coords, smiles = zip(*batch)

    # 修改点3：建议外部传入 protein_tokenizer，避免每个 batch 重建
    if protein_tokenizer is None:
        protein_tokenizer = ProteinTokenizer()

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
