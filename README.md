# PCVRHyFormer - Post-Click Conversion Rate Prediction

A hybrid transformer model for post-click conversion rate prediction, featuring Multi-Sequence HyFormer blocks with RankMixer architecture.

## Project Structure

```
TAAC2026/
├── train.py          # Training entry point with extensive hyperparameter support
├── trainer.py        # PCVRHyFormerRankingTrainer (BCE/Focal Loss, AUC monitoring, early stopping)
├── model.py          # Model definition: HyFormer + MultiSeqQueryGenerator + RankMixerBlock
├── dataset.py        # PCVRParquetDataset for efficient Parquet data loading
├── utils.py          # Logging, early stopping, random seed, Focal Loss utilities
├── run.sh            # Default launch script (RankMixer mode)
└── ns_groups.json    # Feature grouping configuration (example)
```

## Model Architecture

### Core Components

- **MultiSeqHyFormerBlock**: Multi-sequence hybrid transformer block that independently processes each sequence domain
- **MultiSeqQueryGenerator**: Generates query tokens independently per sequence domain using global information
- **RankMixerBlock**: Query Boosting module with three modes:
  - `full`: Token mixing + per-token FFN (requires d_model divisible by T)
  - `ffn_only`: Per-token FFN only
  - `none`: Identity passthrough

### Sequence Encoders

Three variants supported via `--seq_encoder_type`:

- `swiglu`: Efficient attention-free encoder (SwiGLU activation)
- `transformer`: Standard self-attention with optional RoPE
- `longer`: Top-K compressed encoder (cross-attention for long sequences, self-attention for short)

### NS Tokenizer

Two variants via `--ns_tokenizer_type`:

- `group`: Projects each feature group to one NS token (requires `ns_groups.json`)
- `rankmixer`: Concatenates all embeddings, splits into equal chunks (token count tunable)

## Training

### Quick Start

```bash
# Default configuration (RankMixer mode)
bash run.sh

# With custom arguments
python train.py --num_epochs 10 --batch_size 256 --lr 1e-4
```

### Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--batch_size` | 256 | Batch size for training/validation |
| `--lr` | 1e-4 | Learning rate for dense parameters (AdamW) |
| `--num_epochs` | 999 | Maximum epochs (typically stopped earlier by early stopping) |
| `--patience` | 5 | Early stopping patience |
| `--d_model` | 64 | Backbone hidden dimension |
| `--emb_dim` | 64 | Per-embedding dimension |
| `--num_queries` | 1 | Query tokens per sequence domain |
| `--num_hyformer_blocks` | 2 | Number of stacked MultiSeqHyFormerBlock layers |
| `--num_heads` | 4 | Attention heads |
| `--seq_encoder_type` | transformer | Sequence encoder variant |
| `--ns_tokenizer_type` | rankmixer | NS tokenizer variant |
| `--loss_type` | bce | Loss type: `bce` or `focal` |

### Sparse Parameter Handling

- **Sparse Optimizer**: Adagrad for Embedding layers
- **High-Cardinality Reinitialization**: `--reinit_sparse_after_epoch` enables cold-restart trick to reduce overfitting
- **Embedding Skipping**: `--emb_skip_threshold` skips embedding creation for ultra-high-cardinality features

## Data Format

Expects Parquet files with a `schema.json` describing feature layouts:

- `user_int` / `item_int`: Discrete features (scalar or multi-hot)
- `user_dense` / `item_dense`: Dense float features
- `seq_*`: Sequence features grouped by domain (seq_a, seq_b, seq_c, seq_d)
- `timestamp`: Global timestamp for time-bucket computation
- `label_type`: Label derivation (label=2 indicates positive)

## Default Configuration (run.sh)

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

## Evaluation

Training monitors:
- **Binary AUC** (primary metric)
- **Binary LogLoss**

Early stopping based on AUC improvement.
