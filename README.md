# PCVRHyFormer - 点击后转化率预测

用于点击后转化率预测的混合Transformer模型，采用多序列HyFormer块和RankMixer架构。

## 项目结构

```
TAAC2026/
├── train.py          # 训练入口，支持大量超参数配置
├── trainer.py        # PCVRHyFormerRankingTrainer (BCE/Focal Loss, AUC监控, 早停)
├── model.py          # 模型定义: HyFormer + MultiSeqQueryGenerator + RankMixerBlock
├── dataset.py        # PCVRParquetDataset，高效读取Parquet数据
├── utils.py          # 日志、早停、随机种子、Focal Loss工具
├── run.sh            # 默认启动脚本 (RankMixer模式)
└── ns_groups.json    # 特征分组配置示例
```

## 模型架构

### 核心组件

- **MultiSeqHyFormerBlock**: 多序列混合Transformer块，独立处理每个序列域
- **MultiSeqQueryGenerator**: 基于全局信息为每个序列域独立生成Query token
- **RankMixerBlock**: Query Boosting模块，三种模式:
  - `full`: Token混合 + Per-token FFN（要求 d_model 能被 T 整除）
  - `ffn_only`: 仅Per-token FFN
  - `none`: 恒等映射

### 序列编码器

通过 `--seq_encoder_type` 选择三种变体:

- `swiglu`: 高效无注意力编码器 (SwiGLU激活)
- `transformer`: 标准自注意力 + 可选RoPE
- `longer`: Top-K压缩编码器（长序列用交叉注意力，短序列用自注意力）

### NS Tokenizer

两种变体，通过 `--ns_tokenizer_type` 选择:

- `group`: 每个特征组映射到一个NS token（需要 `ns_groups.json`）
- `rankmixer`: 所有embedding拼接后分块（token数量可调）

## 训练

### 快速开始

```bash
# 默认配置 (RankMixer模式)
bash run.sh

# 自定义参数
python train.py --num_epochs 10 --batch_size 256 --lr 1e-4
```

### 关键超参数

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `--batch_size` | 256 | 训练/验证批次大小 |
| `--lr` | 1e-4 | 密集参数学习率 (AdamW) |
| `--num_epochs` | 999 | 最大轮数（通常被早停提前终止） |
| `--patience` | 5 | 早停耐心值 |
| `--d_model` | 64 | 主干隐藏层维度 |
| `--emb_dim` | 64 | 每个Embedding的维度 |
| `--num_queries` | 1 | 每个序列域的Query token数 |
| `--num_hyformer_blocks` | 2 | MultiSeqHyFormerBlock堆叠层数 |
| `--num_heads` | 4 | 注意力头数 |
| `--seq_encoder_type` | transformer | 序列编码器变体 |
| `--ns_tokenizer_type` | rankmixer | NS tokenizer变体 |
| `--loss_type` | bce | 损失类型: `bce` 或 `focal` |

### 稀疏参数处理

- **稀疏优化器**: Adagrad用于Embedding层
- **高基数重初始化**: `--reinit_sparse_after_epoch` 启用冷启动技巧减少过拟合
- **Embedding跳过**: `--emb_skip_threshold` 跳过超高基数特征的embedding创建

## 数据格式

期望Parquet文件配合 `schema.json` 描述特征布局:

- `user_int` / `item_int`: 离散特征（标量或多热）
- `user_dense` / `item_dense`: 稠密浮点特征
- `seq_*`: 按域分组的序列特征 (seq_a, seq_b, seq_c, seq_d)
- `timestamp`: 用于时间桶计算的全局时间戳
- `label_type`: 标签推导 (label=2 表示正样本)

## 默认配置 (run.sh)

```bash
python train.py \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 5 \
    --item_ns_tokens 2 \
    --num_queries 2 \
    --ns_groups_json "" \
    --emb_skip_threshold 1000000 \
    --num_workers 8
```

## 评估

训练监控指标:
- **Binary AUC**（主要指标）
- **Binary LogLoss**

基于AUC improvement的早停机制。
