import math
import torch
from torch.utils.data import DataLoader, random_split

from my_data import CharTokenizer, CharDataset
from minigpt import MiniGPT, GPTConfig


def train_one_epoch(dataloader, model, optimizer, device, grad_clip=1.0):
    model.train()
    total_loss = 0.0

    for step, (x, y) in enumerate(dataloader):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        _, loss = model(x, y)
        loss.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        total_loss += loss.item()

        if step % 100 == 0:
            print(f"step {step:4d} | train loss {loss.item():.4f}")

    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate(dataloader, model, device):
    model.eval()
    total_loss = 0.0

    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)

        _, loss = model(x, y)
        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")
    return avg_loss, ppl


def main():
    # 你可以先换成自己的文本文件
    with open("input.txt", "r", encoding="utf-8") as f:
        text = f.read()

    block_size = 64
    batch_size = 32
    epochs = 10
    lr = 3e-4

    tokenizer = CharTokenizer(text)
    dataset = CharDataset(text, block_size=block_size, tokenizer=tokenizer)

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, drop_last=True)

    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=block_size,
        d_model=128,
        num_heads=4,
        num_layers=4,
        d_ff=512,
        dropout=0.1,
        weight_tying=True,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    model = MiniGPT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    for epoch in range(epochs):
        train_loss = train_one_epoch(train_loader, model, optimizer, device)
        val_loss, val_ppl = evaluate(val_loader, model, device)

        print(
            f"epoch {epoch+1:02d} | "
            f"train loss {train_loss:.4f} | "
            f"val loss {val_loss:.4f} | "
            f"val ppl {val_ppl:.4f}"
        )

        # 每轮都生成一点看看效果
        prompt = "你好"
        # 若 prompt 中有训练集未出现字符，会报错，所以开始建议用训练文本里一定存在的字符
        try:
            start_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
        except KeyError:
            start_ids = torch.tensor([[0]], dtype=torch.long, device=device)

        out = model.generate(
            start_ids,
            max_new_tokens=100,
            temperature=0.8,
            top_k=20
        )

        text_out = tokenizer.decode(out[0].tolist())
        print("sample:")
        print(text_out)
        print("-" * 60)

        # 保存 checkpoint
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config.__dict__,
            "tokenizer_chars": tokenizer.chars,
            "epoch": epoch + 1,
        }, "minigpt_checkpoint.pt")


if __name__ == "__main__":
    main()