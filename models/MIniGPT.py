from Attention import CausalMultiHeadAttention, FeedForward
from Attention import GPTBlock
from torch import nn
import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass

@dataclass
class GPTConfig:
    vocab_size: int
    max_seq_len: int = 128
    d_model: int = 128
    num_heads: int = 4
    num_layers: int = 4
    d_ff: int = 512
    dropout: float = 0.1
    weight_tying: bool = True

class MiniGPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.d_model = config.d_model
        self.max_seq_len = config.max_seq_len
        self.token_emb = nn.Embedding(self.vocab_size, self.d_model)
        self.pos_emb = nn.Embedding(self.max_seq_len, self.d_model)
        self.dropout = nn.Dropout(config.dropout)

        self.layers = nn.ModuleList([GPTBlock(config.d_model, config.num_heads, config.d_ff, config.max_seq_len, config.dropout) for _ in range(config.num_layers)])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, self.vocab_size)

    def forward(self,idx:torch.Tensor,targets : torch.Tensor = None,return_attn = False):
        B,T = idx.shape
        assert T <=self.max_seq_len, f"Sequence length {T} exceeds maximum {self.max_seq_len}"
        tok = self.token_emb(idx)  # [B, T, D]
        pos = torch.arange(T, device=idx.device).unsqueeze(0)  # [B, T]
        pos = self.pos_emb(pos)  # [B, T, D]
        x = self.dropout(tok + pos)

        all_attn = []
        for block in self.layers:
            x, last_attn_weights = block(x)
            all_attn.append(last_attn_weights)
        x = self.ln_f(x)
        logits = self.head(x)  # [B, T, V]

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))

        if return_attn:
            return logits, loss, all_attn
        else:
            return logits, loss

    @torch.no_grad()
    def generate(
    self,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int = None,
    ):
        self.eval()

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.max_seq_len:]

            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]   # [B, V]

        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        logits = logits / temperature

        if top_k is not None:
            values, _ = torch.topk(logits, top_k)
            min_values = values[:, -1].unsqueeze(-1)
            logits = torch.where(
                logits < min_values,
                torch.full_like(logits, float("-inf")),
                logits
            )

        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        idx = torch.cat([idx, next_token], dim=1)

        return idx
