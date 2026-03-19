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
        # block_size 表示“上下文窗口长度”，即模型一次看到多少个字符
        self.block_size = block_size
        # 保存 tokenizer，后续调试或可视化时可能会用到
        self.tokenizer = tokenizer
        # 先把原始文本编码成 token id，再转成 PyTorch 的 long 张量
        # 形状: [文本总长度]
        self.data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    def __len__(self):
        # 可取到的样本数。
        # 每个样本需要 block_size + 1 个字符（前 block_size 作为输入，后 block_size 作为标签）
        # 因此起点 idx 最大只能到 len(data) - block_size - 1，样本总数就是 len(data) - block_size
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        # 取一个长度为 block_size + 1 的连续片段
        # 例如 block_size=4，拿到 [t0,t1,t2,t3,t4]
        chunk = self.data[idx: idx + self.block_size + 1]
        # 输入 x: 前 block_size 个 token -> [t0,t1,t2,t3]
        x = chunk[:-1]
        # 标签 y: 后 block_size 个 token -> [t1,t2,t3,t4]
        # 即“下一个字符预测”任务：用当前位置预测下一位置
        y = chunk[1:]
        return x, y


def main():
    text = "你好世界！这是一个测试文本。"
    tokenizer = CharTokenizer(text)
    dataset = CharDataset(text, block_size=5, tokenizer=tokenizer)
    print(tokenizer.chars)
    print(tokenizer.vocab_size)
    print(tokenizer.stoi)
    print(tokenizer.itos)

    ids = tokenizer.encode("你好世界")
    print(ids)

    s = tokenizer.decode(ids)
    print(s)
    
if __name__ == "__main__":
    main()