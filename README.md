# 🚀 Handmade-MiniGPT: 从零组装的生成式预训练模型

就像搭建一套精密的机械传动系统一样，本项目拒绝直接调用高度封装的高阶 API，而是使用原生 PyTorch 从零开始“手搓”了 GPT (Generative Pre-trained Transformer) 的核心架构。

本项目旨在透彻理解大语言模型底层的运作逻辑，从多头注意力机制的矩阵乘法，到自回归循环的文本生成，每一行代码都清晰可见。

## 🛠️ 核心架构与特性

本仓库包含了一个完整的基础 GPT 模型运转流程，主要特性包括：

* **因果多头注意力 (Causal Multi-Head Attention)**: 
    纯手工实现了带掩码（Mask）的注意力机制。通过构建下三角布尔矩阵 $M$，严格阻断了模型在训练时对“未来 Token”的注意力，确保自回归逻辑的严谨性。
    $$Attention(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$$
* **标准的 Transformer 模块 (GPTBlock)**: 
    集成了 `LayerNorm`、残差连接 (Residual Connections) 以及前馈神经网络 (FeedForward)，保证了深层网络梯度传播的稳定性。
* **位置编码 (Positional Embedding)**: 
    使用可学习的绝对位置编码 `nn.Embedding(max_seq_len, d_model)`，赋予模型理解序列时序顺序的能力。
* **自回归生成引擎 (Generate)**: 
    内置 `generate` 方法，完美复刻了模型在推理阶段“基于历史上下文，预测下一个 Token”的循环过程。

## 📁 目录结构

项目采用了清晰的模块化设计，核心组件与外围配置分离：

```text
Handmade-MiniGPT/
├── models/                   # 核心“发动机”零件库
│   ├── Attention.py          # 底层机制：多头因果注意力、前馈神经网络、Transformer Block
│   ├── MIniGPT.py            # 架构总成：词嵌入、位置编码及核心生成逻辑 (generate)
│   └── train_test_loop.py    # 炼丹炉：标准的模型训练与测试评估循环逻辑
│
├── .gitignore                # Git 忽略配置文件（防止不小心上传权重文件）
├── LICENSE                   # 开源许可协议
└── README.md                 # 项目说明文档
