# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在本仓库中工作时提供指导。

## 项目概述

nanoGPT 是一个极简、高性能的中型 GPT 语言模型训练和微调仓库。代码库优先考虑简洁性和可读性，仅包含两个核心文件（每个约 300 行）：
- `train.py`：支持 DDP 的训练循环
- `model.py`：GPT 模型定义，可选加载 GPT-2 检查点

代码库已汉化，注释和文档均为中文。

## 架构

### 核心组件

**model.py** - GPT 模型实现：
- `GPTConfig`：模型配置数据类（n_layer、n_head、n_embd、block_size、dropout、bias）
- `GPT`：包含 Transformer 块的主模型类
  - `CausalSelfAttention`：多头自注意力机制，支持 Flash Attention（PyTorch >= 2.0）
  - `MLP`：使用 GELU 激活的前馈网络
  - `Block`：结合注意力 + MLP 和残差连接的 Transformer 块
  - `LayerNorm`：自定义层归一化，支持可选偏置
- `from_pretrained()`：加载 OpenAI GPT-2 检查点（gpt2、gpt2-medium、gpt2-large、gpt2-xl）

**train.py** - 训练脚本，支持三种模式：
1. 单 GPU：`python train.py [config] [--args]`
2. 单节点 DDP：`torchrun --standalone --nproc_per_node=N train.py`
3. 多节点 DDP：`torchrun --nproc_per_node=N --nnodes=M --node_rank=R --master_addr=IP --master_port=PORT train.py`

**configurator.py** - 配置系统：
- 解析配置文件和命令行参数
- 配置文件覆盖默认值，命令行参数覆盖配置文件
- 用法：`python train.py config/file.py --key=value`

**sample.py** - 推理/采样脚本：
- 支持从训练的检查点或预训练的 GPT-2 模型采样
- 可使用文件提示词：`--start=FILE:prompt.txt`

### 数据管道

每个数据集都有一个 `prepare.py` 脚本，用于创建 `train.bin` 和 `val.bin` 文件：
- `data/shakespeare_char/`：字符级莎士比亚数据集
- `data/shakespeare/`：使用 GPT-2 BPE 分词器的词元级莎士比亚数据集
- `data/openwebtext/`：用于复现 GPT-2 的 OpenWebText 数据集

数据以表示词元 ID 的原始 uint16 字节存储。

## 常用命令

### 训练

**快速开始（莎士比亚字符级）**：
```bash
# 准备数据
python data/shakespeare_char/prepare.py

# 在 GPU 上训练
python train.py config/train_shakespeare_char.py

# 在 CPU 上训练（macOS/笔记本）
python train.py config/train_shakespeare_char.py --device=cpu --compile=False --eval_iters=20 --log_interval=1 --block_size=64 --batch_size=12 --n_layer=4 --n_head=4 --n_embd=128 --max_iters=2000 --lr_decay_iters=2000 --dropout=0.0

# 在 Apple Silicon 上训练（M1/M2/M3）
python train.py config/train_shakespeare_char.py --device=mps
```

**复现 GPT-2 (124M)**：
```bash
# 准备 OpenWebText 数据集
python data/openwebtext/prepare.py

# 在 8x A100 40GB 上训练（需要约 4 天）
torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py
```

**多节点训练**：
```bash
# 主节点（IP: 123.456.123.456）
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py

# 工作节点
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py

# 对于没有 Infiniband 的集群，在前面加上：NCCL_IB_DISABLE=1
```

### 微调

**在莎士比亚数据集上微调 GPT-2**：
```bash
# 准备数据（使用 GPT-2 BPE 分词器）
python data/shakespeare/prepare.py

# 微调（加载预训练 GPT-2，使用小学习率训练）
python train.py config/finetune_shakespeare.py
```

### 推理/采样

**从训练的模型采样**：
```bash
python sample.py --out_dir=out-shakespeare-char
```

**从预训练 GPT-2 采样**：
```bash
python sample.py --init_from=gpt2-xl --start="What is the answer to life, the universe, and everything?" --num_samples=5 --max_new_tokens=100
```

**使用文件提示词采样**：
```bash
python sample.py --start=FILE:prompt.txt --out_dir=out-shakespeare
```

### 评估

**在 OpenWebText 上评估 OpenAI GPT-2 基线**：
```bash
python train.py config/eval_gpt2.py         # 124M 参数
python train.py config/eval_gpt2_medium.py  # 350M 参数
python train.py config/eval_gpt2_large.py   # 774M 参数
python train.py config/eval_gpt2_xl.py      # 1558M 参数
```

### 基准测试

```bash
python bench.py  # 在不包含完整训练复杂性的情况下分析训练循环
```

## 配置系统

配置系统使用独特的方法，配置文件是覆盖全局变量的 Python 文件：

1. `train.py` 全局变量中定义的默认值
2. 配置文件（如 `config/train_shakespeare_char.py`）覆盖默认值
3. 命令行参数（`--key=value`）覆盖配置文件

**关键配置参数**：
- 模型：`n_layer`、`n_head`、`n_embd`、`block_size`、`dropout`、`bias`
- 训练：`batch_size`、`gradient_accumulation_steps`、`max_iters`、`learning_rate`
- 数据：`dataset`（必须与 `data/` 中的目录名匹配）
- 系统：`device`（'cuda'/'cpu'/'mps'）、`dtype`（'float32'/'bfloat16'/'float16'）、`compile`（PyTorch 2.0）
- DDP：`backend`（'nccl'/'gloo'）
- 输出：`out_dir`、`eval_interval`、`always_save_checkpoint`
- 日志：`wandb_log`、`wandb_project`、`wandb_run_name`

## 模型初始化选项

`init_from` 参数控制模型初始化：
- `'scratch'`：随机初始化（默认）
- `'resume'`：从 `out_dir` 中的检查点恢复
- `'gpt2'`：加载 OpenAI GPT-2 124M
- `'gpt2-medium'`：加载 GPT-2 350M
- `'gpt2-large'`：加载 GPT-2 774M
- `'gpt2-xl'`：加载 GPT-2 1558M

## 性能优化

- **PyTorch 2.0 编译**：设置 `compile=True` 可提速约 2 倍（需要 PyTorch >= 2.0）
- **Flash Attention**：PyTorch >= 2.0 时自动使用（GPU 注意力计算更快）
- **混合精度**：使用 `dtype='bfloat16'` 或 `dtype='float16'` 加速训练
- **梯度累积**：通过 `gradient_accumulation_steps` 模拟更大的批次大小
- **DDP**：在多个 GPU/节点上分布式训练

**平台特定说明**：
- Windows：添加 `--compile=False`（PyTorch 2.0 编译尚不支持）
- Apple Silicon：使用 `--device=mps` 可比 CPU 快 2-3 倍
- 无 Infiniband：在多节点命令前加上 `NCCL_IB_DISABLE=1`

## 检查点和输出

检查点保存到 `out_dir`（默认：`out/`）：
- `ckpt.pt`：包含模型状态、优化器状态、配置、迭代次数的最新检查点
- 检查点频率由 `eval_interval` 和 `always_save_checkpoint` 控制

## 数据集要求

`data/` 下的每个数据集目录必须包含：
- `prepare.py`：下载和分词数据的脚本
- `train.bin`：训练数据（原始 uint16 词元 ID）
- `val.bin`：验证数据（原始 uint16 词元 ID）

`dataset` 配置参数必须与目录名匹配。

## 代码风格

本代码库有意避免过度抽象。修改时：
- 保持代码扁平易读（避免深层类继承）
- 优先显式而非隐式（最小化魔法）
- 注释和文档使用中文（代码库已汉化）
- 尽可能保持核心文件约 300 行的限制
