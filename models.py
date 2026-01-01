import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


class StructAttnBlock(nn.Module):
    def __init__(self, in_dim, hid_dim, n_heads=4):
        super().__init__()
        self.q = nn.Linear(in_dim, hid_dim)
        self.k = nn.Linear(in_dim, hid_dim)
        self.v = nn.Linear(in_dim, hid_dim)
        self.out = nn.Linear(hid_dim, hid_dim)
        self.n_heads = n_heads
        self.hid_dim = hid_dim

    def forward(self, feats, coords, mask=None):
        # feats: [B,N,D], coords: [B,N,3], mask: [B,N] (1=valid)
        B, N, _ = feats.shape
        H = self.n_heads
        dh = self.hid_dim // H
        q = self.q(feats).view(B, N, H, dh).transpose(1, 2)
        k = self.k(feats).view(B, N, H, dh).transpose(1, 2)
        v = self.v(feats).view(B, N, H, dh).transpose(1, 2)

        sim = (q @ k.transpose(-2, -1)) / (dh ** 0.5)
        # Distance-weighted attention for structure awareness.
        dist = torch.cdist(coords, coords)
        dist_w = torch.exp(-dist ** 2 / 10.0).unsqueeze(1)

        logits = sim * dist_w
        if mask is not None:
            key_mask = ~mask[:, None, None, :]
            logits = logits.masked_fill(key_mask, -1e9)

        attn = F.softmax(logits, dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, N, -1)
        return self.out(out)


class PocketEncoder(nn.Module):
    def __init__(self, vocab_size, hid_dim, n_heads=4, pad_idx=0):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, hid_dim, padding_idx=pad_idx)
        self.block = StructAttnBlock(hid_dim, hid_dim, n_heads=n_heads)

    def forward(self, token_ids, coords, mask=None):
        feats = self.emb(token_ids)
        return self.block(feats, coords, mask=mask)


class DrugEncoder(nn.Module):
    def __init__(self, in_dim, hid_dim, n_heads=4):
        super().__init__()
        self.block = StructAttnBlock(in_dim, hid_dim, n_heads=n_heads)

    def forward(self, feats, coords, mask=None):
        return self.block(feats, coords, mask=mask)


class CrossAttnAdapter(nn.Module):
    def __init__(self, hid_dim, n_heads=4):
        super().__init__()
        self.ln = nn.LayerNorm(hid_dim)
        self.cross = nn.MultiheadAttention(hid_dim, n_heads, batch_first=True)
        # Gated residual to avoid overpowering the base LM.
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, h, memory, memory_mask=None):
        h_norm = self.ln(h)
        key_padding_mask = None
        if memory_mask is not None:
            key_padding_mask = ~memory_mask
        ca, _ = self.cross(h_norm, memory, memory, key_padding_mask=key_padding_mask)
        return h + self.alpha * ca


class MolGPTWithCrossAttn(nn.Module):
    def __init__(self, model_path, n_heads=4):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.lm = AutoModelForCausalLM.from_pretrained(model_path)
        emb_dim = self.lm.config.n_embd
        self.adapter = CrossAttnAdapter(emb_dim, n_heads=n_heads)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.lm.resize_token_embeddings(len(self.tokenizer))

    def forward(self, input_ids, attention_mask, pocket_mem, pocket_mask=None):
        if hasattr(self.lm, "transformer"):
            # GPT2-style forward to get hidden states for adapter injection.
            h = self.lm.transformer(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            h = self.adapter(h, pocket_mem, memory_mask=pocket_mask)
            logits = self.lm.lm_head(h)
        else:
            outputs = self.lm(input_ids, attention_mask=attention_mask, output_hidden_states=True)
            h = outputs.hidden_states[-1]
            h = self.adapter(h, pocket_mem, memory_mask=pocket_mask)
            logits = self.lm.lm_head(h)
        return logits


class Pocket2SmilesModel(nn.Module):
    def __init__(self, drug_feat_dim, hid_dim, model_path, n_heads=4, protein_vocab_size=0, protein_pad_idx=0):
        super().__init__()
        self.pocket_enc = PocketEncoder(protein_vocab_size, hid_dim, n_heads=n_heads, pad_idx=protein_pad_idx)
        self.drug_enc = DrugEncoder(drug_feat_dim, hid_dim, n_heads=n_heads)
        self.decoder = MolGPTWithCrossAttn(model_path, n_heads=n_heads)

    def forward(
        self,
        pocket_tokens,
        pocket_coords,
        pocket_mask,
        drug_feats,
        drug_coords,
        drug_mask,
        input_ids,
        attention_mask,
    ):
        pocket_mem = self.pocket_enc(pocket_tokens, pocket_coords, mask=pocket_mask)
        drug_mem = self.drug_enc(drug_feats, drug_coords, mask=drug_mask)
        logits = self.decoder(input_ids, attention_mask, pocket_mem, pocket_mask=pocket_mask)
        return logits, pocket_mem, drug_mem
