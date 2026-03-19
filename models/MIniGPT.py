from Attention import CausalMultiHeadAttention
from Attention import GPTBlock
from torch import nn
import torch
import torch.nn.functional as F
import numpy as np

class MiniGPT(nn.Module):
    def __init__(self,vocab_size :int,max_seq_len:int,d_model:int,num_heads:int ,num_layers :int,d_ff : int,dropout:float):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.token_emb = nn.Embedding(vocab_size,d_model)
        self.pos_emb = nn.Embedding(max_seq_len,d_model)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([GPTBlock(d_model,num_heads,d_ff,dropout) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model,vocab_size)

    def forward(self,idx:torch.Tensor,targets : torch.Tensor = None):
        B,T = idx.shape
        assert T <=self.max_seq_len, f"Sequence length {T} exceeds maximum {self.max_seq_len}"
        tok = self.token_emb(idx)  # [B, T, D]
        pos = torch.arange(T, device=idx.device).unsqueeze(0)  # [B, T]
        pos = self.pos_emb(pos)  # [B, T, D]
        x = self.dropout(tok + pos)

        last_attn_weights = None
        for block in self.layers:
            x, last_attn_weights = block(x)
        x = self.ln_f(x)
        logits = self.head(x)  # [B, T, V]

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))

        return logits, loss, last_attn_weights
    
    @torch.no_grad()
    def generate(self,idx:torch.Tensor,max_new_tokens:int):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.max_seq_len:]  # [B, T]
            logits, _, _ = self(idx_cond)  # [B, T, V]
            logits = logits[:, -1, :]  # [B, V]
            prob = F.softmax(logits, dim=-1)  # [B, V]

            next_token = torch.multinomial(prob, num_samples=1)  # [B, 1]
            idx = torch.cat((idx, next_token), dim=1)  # [B, T+1]
        return idx
