# Pocket2Smiles 代码骨架（中文说明）

这是一个“口袋结构 → SMILES”de novo 生成的最小可运行骨架，包含：
- 口袋/配体结构编码（结构感知注意力：距离 × 相似度）
- MolGPT 解码（cross-attention adapter）
- 训练期对比学习（口袋–配体 InfoNCE）
- 推理期 KG 相似邻域 rerank（embedding 检索）

## 目录结构
```
pocket2smiles/
  config.py
  data.py
  gninatypes.py
  kg_rerank.py
  models.py
  train.py
  infer.py
  utils.py
```

## 依赖
```
pip install torch transformers rdkit-pypi numpy biopython
```

## 数据准备（CrossDocked2020 / pocket10）
当前数据格式示例：
- 口袋：`*_lig_tt_min_0_pocket10.pdb`
- 配体：`*_lig_tt_min_0.sdf`
还需要官方划分文件：
- `split_by_name.pt`

代码会自动：
- 优先使用 `tt_min`
- 若缺失则回退到 `tt_docked`

## 模型路径与参数
编辑 `config.py`：
```
MODEL_PATH = "~/MolGpt"
DATA_ROOT = "~/CrossDocked2020_v1.3"
SPLIT_PATH = "/Users/corymou/Downloads/TamGen-main/data/crossdocked/raw/split_by_name.pt"
CROSSDOCK_SUBDIR = "crossdocked_pocket10"
VALID_LAST_N = 100
```
本版本使用 `pocket10.pdb` 和 `sdf`，不再依赖 gninatypes。

## 训练
```
python -m pocket2smiles.train
```
默认保存：
```
<DATA_ROOT>/pocket2smiles.pt
```

## 推理
```
python -m pocket2smiles.infer
```
默认会从数据集中取一个口袋进行采样。

## KG 相似邻域 rerank（可选）
在 `config.py` 中配置：
```
KG_NODE_IDS_PATH = "~/kg_node_ids.txt"
KG_NODE_EMB_PATH = "~/kg_node_embs.npy"
KG_TRIPLES_PATH = "~/kg_triples.tsv"  # 可选
```
格式要求：
- `kg_node_ids.txt`：每行一个节点 ID
- `kg_node_embs.npy`：形状 `[N, D]`，与 ID 顺序一致
- `kg_triples.tsv`：每行 `head relation tail`

注意：当前实现使用 **Morgan 指纹** 作为生成分子的查询 embedding。  
如果你的 KG embedding 来自其他模型（图模型/语言模型），需要用相同方式生成 SMILES embedding 才能匹配。

## 重要说明
- 口袋编码使用“残基一字母序列 + 残基坐标”：序列按链与残基号排序；坐标为残基原子平均位置。
- 序列在数据加载时会被转成“空格分隔字符序列”，并在模型端做 token 化。
- MolGPT 本身没有 cross-attention，这里通过 adapter 注入条件。
- 训练和推理时条件一致（仅 protein/口袋），避免“训练-推理不匹配”。
- 本骨架是最小示例，适合你继续扩展：采样策略、ADMET、Docking、关系评分等。
