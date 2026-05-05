# Transformer架构详解

## 背景

Transformer架构由Google在2017年论文《Attention Is All You Need》中提出，彻底改变了NLP领域。它完全基于注意力机制，摒弃了循环和卷积结构，实现了高度并行化和长距离依赖建模能力。Transformer已成为现代LLM的基础架构。

## 核心机制：自注意力（Self-Attention）

### Q、K、V的含义
- **Q（Query，查询）**：当前要查询的目标向量，代表"我要找什么"
- **K（Key，键）**：每个位置的索引向量，代表"每个位置有什么特征标签"
- **V（Value，值）**：每个位置的实际内容向量，代表"每个位置存储什么信息"

### 计算过程
```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V
```
1. 用Q与所有K计算点积相似度得分
2. 除以√d_k进行缩放，防止点积过大导致Softmax梯度消失
3. 通过Softmax将得分转换为概率分布（注意力权重）
4. 用注意力权重对V加权求和得到最终输出

### 多头注意力（Multi-Head Attention）
将Q、K、V分别投影到多个子空间，在每个子空间独立计算注意力，最后拼接结果。每个注意力头可以关注不同类型的模式（句法、语义、位置等）。

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) × W^O
其中 head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

## 架构组件

### 位置编码（Positional Encoding）
由于Transformer不包含循环或卷积结构，需要显式编码位置信息：
- **正弦位置编码**（原始论文）：使用sin/cos函数生成固定位置向量
- **可学习位置编码**（BERT、GPT）：将位置编码作为可训练的参数
- **旋转位置编码（RoPE）**：通过旋转矩阵编码相对位置，在LLaMA等模型中使用
- **ALiBi**：通过注意力偏置引入位置信息，外推能力更强

### 前馈神经网络（Feed-Forward Network）
每个注意力层后接两层线性变换加ReLU激活：
```
FFN(x) = W_2 × ReLU(W_1 × x + b_1) + b_2
```
FFN引入了非线性，使模型能够学习更复杂的特征变换。现代架构常使用SwiGLU等门控激活函数替代ReLU。

### 残差连接与层归一化
- **残差连接（Residual Connection）**：将输入直接加到子层输出上，缓解深层网络退化问题
- **层归一化（Layer Normalization）**：对每个样本的特征维度进行归一化，加速训练
- **Post-LN vs Pre-LN**：原始的Post-LN在残差连接后归一化；Pre-LN在输入前归一化，训练更稳定，在现代架构中更常见

## Encoder-Decoder vs Decoder-Only

### Encoder-Decoder（如原始Transformer、T5）
- Encoder：双向注意力，编码输入序列
- Decoder：单向（因果）注意力 + 交叉注意力连接Encoder
- 适合：机器翻译、文本摘要

### Encoder-Only（如BERT）
- 使用双向注意力，可以看到上下文全部信息
- 通过掩码语言模型（MLM）预训练
- 适合：文本分类、命名实体识别、问答等理解任务

### Decoder-Only（如GPT系列）
- 使用因果（单向）注意力，只能看到已生成的部分
- 通过自回归语言模型预训练
- 适合：文本生成、对话、代码生成
- 优势：架构简单，扩展性好，成为当前LLM的主流选择

## Scaling法则

### Kaplan等人的发现
- 模型性能与模型参数量、训练数据量、计算量呈幂律关系
- 在计算预算固定的情况下，大模型 + 少数据优于小模型 + 多数据

### Chinchilla法则（DeepMind 2022）
- 对于给定的计算预算，模型大小和训练数据量应等比例增长
- 训练token数应约为模型参数量的20倍
- 意味着许多大模型实际上训练不足（undertrained）

## 现代Transformer变体

- **混合专家（MoE）**：使用多个"专家"子网络，每次激活部分参数（如Mixtral）
- **分组查询注意力（GQA）**：多个Query头共享Key/Value头，减少KV缓存
- **FlashAttention**：通过分块计算和重计算减少显存占用，实现更快的注意力计算
- **State Space Models**：如Mamba，尝试用线性时间复杂度的状态空间模型替代注意力
