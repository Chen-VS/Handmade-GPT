import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalMultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # 提前注册 causal mask，避免每次 forward 重建
        mask = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool()
        self.register_buffer("causal_mask", mask)

    def forward(self, x: torch.Tensor):
        """
        x: [B, T, D]
        return:
            output: [B, T, D]
            attn_weights: [B, H, T, T]
        """
        B, T, D = x.shape

        Q = self.Wq(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # [B,H,T,Hd]
        K = self.Wk(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.Wv(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)     # [B,H,T,T]

        mask = self.causal_mask[:T, :T]
        scores = scores.masked_fill(mask, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        output = torch.matmul(attn_weights, V)  # [B,H,T,Hd]
        output = output.transpose(1, 2).contiguous().view(B, T, D)
        output = self.Wo(output)
        output = self.resid_dropout(output)

        return output, attn_weights


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),   # 比 ReLU 更贴近 GPT
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)


class GPTBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = CausalMultiHeadAttention(d_model, num_heads, max_seq_len, dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor):
        # Pre-LN 写法，更常见也更稳
        attn_out, attn_weights = self.attn(self.norm1(x))
        x = x + attn_out

        ff_out = self.ff(self.norm2(x))
        x = x + ff_out

        return x, attn_weights