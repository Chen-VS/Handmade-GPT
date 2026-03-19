# 🚀 Handmade-MiniGPT Pro: 工业级手搓 GPT 实践

这是一个从零开始、完全基于 PyTorch 实现的生成式 Transformer (GPT) 框架。本项目不仅复刻了 GPT-2 的核心架构，还集成了现代大模型训练中的多种性能优化手段。

## 🌟 核心硬核特性

### 1. 结构优化 (Architectural Refinement)
- **Pre-LN 架构**: 不同于原始 Transformer，我们将 LayerNorm 置于残差连接内部，有效解决了深层网络中的梯度消失/爆炸问题。
- **GELU 激活函数**: 使用 $GELU(x) = x\Phi(x)$ 替代传统的 ReLU，提供更平滑的非线性变换。
- **权重共享 (Weight Tying)**: 实现了输入 `Embedding` 层与输出 `Linear` 层的权重对齐，大幅减少了参数量。

### 2. 精准生成引擎 (Sampling Strategy)
内置了比基础版更强大的生成算法：
- **Temperature 控制**: 调节生成文本的随机性。
- **Top-K 过滤**: 自动剔除低概率噪声 Token，显著提升语义连贯性。

### 3. 稳健训练流 (Robust Training)
- **梯度裁剪**: 在 `train.py` 中集成了 `clip_grad_norm_`，确保训练过程平滑。
- **权重初始化**: 手动实现了针对 Linear 和 Embedding 层的 `_init_weights` 逻辑（均值为 0，标准差为 0.02）。

## 📁 模块化文件树

```text
Handmade-MiniGPT/
├── models/
│   ├── attention.py       # CausalMultiHeadAttention, FeedForward (GELU), GPTBlock (Pre-LN)
│   ├── MIniGPT.py         # 核心架构：GPTConfig 配置类, MiniGPT 主类, 权重初始化逻辑
│   └── my_data.py         # 数据流：CharTokenizer (字符分词器), CharDataset (滑窗数据集)
├── train.py               # 炼丹炉：支持梯度裁剪、自动生成采样、模型保存 (Checkpoints)
├── input.txt              # 训练语料（如莎士比亚、武侠小说等）
├── .gitignore             # 排除 .pt 权重与缓存
└── README.md              # 你正在看的这份文档
