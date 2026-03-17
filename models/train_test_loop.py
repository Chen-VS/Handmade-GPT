import torch
import torch.nn as nn
import torch.nn.functional as F

def train_loop(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    
    # 这里的 X 是输入 token，y 是目标 token (通常是 X 右移一位)
    for batch, (X, y) in enumerate(dataloader):
        # 1. 搬运到 GPU/CPU
        X, y = X.to(device), y.to(device)

        # 2. 前向传播
        # 注意：GPT 的输出通常是 (Batch, Time, Vocab)
        logits, _ , _ = model(X) 
        
        # 3. 维度重塑 (Flattening)
        # CrossEntropyLoss 要求预测值是 [B*T, V], 目标值是 [B*T]
        B, T, V = logits.shape
        loss = loss_fn(logits.view(B*T, V), y.view(B*T))

        # 4. 反向传播三部曲
        optimizer.zero_grad() # 建议先清零，再计算，更符合习惯
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss_val = loss.item()
            print(f"loss: {loss_val:>7f}  [{batch * len(X):>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn, device):
    model.eval()
    num_batches = len(dataloader)
    test_loss = 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            logits, _, _ = model(X)
            
            # 同样需要重塑维度来计算损失
            B, T, V = logits.shape
            test_loss += loss_fn(logits.view(B*T, V), y.view(B*T)).item()

    test_loss /= num_batches
    # 对于 GPT，我们更关注 Loss 而不是分类准确率 Accuracy
    print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")