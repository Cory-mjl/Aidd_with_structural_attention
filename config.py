import os

MODEL_PATH = os.path.expanduser("~/MolGpt")
DATA_ROOT = os.path.expanduser("~/CrossDocked2020_v1.3")

SPLIT_PATH = os.path.expanduser(
    "/Users/corymou/Downloads/TamGen-main/data/crossdocked/raw/split_by_name.pt"
)
CROSSDOCK_SUBDIR = "crossdocked_pocket10"
VALID_LAST_N = 100

HID_DIM = 256
N_HEADS = 4
MAX_LEN = 128
BATCH_SIZE = 4
LR = 2e-5
WEIGHT_DECAY = 0.01
EPOCHS = 1
SEED = 42

DEVICE = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"

# KG rerank (optional)
KG_TRIPLES_PATH = os.path.expanduser("~/kg_triples.tsv")
KG_NODE_IDS_PATH = os.path.expanduser("~/kg_node_ids.txt")
KG_NODE_EMB_PATH = os.path.expanduser("~/kg_node_embs.npy")
KG_TOPK = 20
KG_DEG_WEIGHT = 0.1
