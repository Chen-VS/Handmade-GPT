from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

from attention import GPTBlock


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

        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

        self.layers = nn.ModuleList([
            GPTBlock(
                d_model=config.d_model,
                num_heads=config.num_heads,
                d_ff=config.d_ff,
                max_seq_len=config.max_seq_len,
                dropout=config.dropout,
            )
            for _ in range(config.num_layers)
        ])

        self.ln_f = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # 权重共享：更贴近真实语言模型
        if config.weight_tying:
            self.head.weight = self.token_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None, return_attn: bool = False):
        """
        idx: [B, T]
        targets: [B, T]
        """
        B, T = idx.shape
        assert T <= self.config.max_seq_len, (
            f"Sequence length {T} exceeds maximum {self.config.max_seq_len}"
        )

        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)   # [1, T]

        tok = self.token_emb(idx)      # [B, T, D]
        pos = self.pos_emb(pos)        # [1, T, D]
        x = self.dropout(tok + pos)

        attn_list = []
        for block in self.layers:
            x, attn = block(x)
            if return_attn:
                attn_list.append(attn)

        x = self.ln_f(x)
        logits = self.head(x)          # [B, T, V]

        loss = None
        if targets is not None:
            B, T, V = logits.shape
            loss = F.cross_entropy(
                logits.reshape(B * T, V),
                targets.reshape(B * T)
            )

        if return_attn:
            return logits, loss, attn_list
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

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)   # [B,1]

            idx = torch.cat([idx, next_token], dim=1)

        return idx