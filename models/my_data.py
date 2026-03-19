import torch
from torch.utils.data import Dataset


class CharTokenizer:
    def __init__(self, text: str):
        chars = sorted(list(set(text)))
        self.chars = chars
        self.vocab_size = len(chars)

        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

    def encode(self, s: str):
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        return "".join([self.itos[i] for i in ids])


class CharDataset(Dataset):
    def __init__(self, text: str, block_size: int, tokenizer: CharTokenizer):
        self.block_size = block_size
        self.tokenizer = tokenizer
        self.data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        chunk = self.data[idx: idx + self.block_size + 1]
        x = chunk[:-1]   # 输入
        y = chunk[1:]    # 目标右移一位
        return x, y