# Lhotse-based Data Loading in MELT

This document explains how Lhotse-based data loading works in the MELT training pipeline, covering both training and evaluation.

## Overview

The data loading pipeline uses [Lhotse](https://github.com/lhotse-speech/lhotse) for efficient speech data handling. The key components are:

```
CutSet → Sampler → Dataset → DataLoader → Trainer
```

| Component | Role |
|-----------|------|
| **CutSet** | Lhotse's data structure containing audio references, supervisions, and metadata |
| **Sampler** | Batches cuts by duration constraints (e.g., `max_duration`, `max_cuts`) |
| **Dataset** | Processes CutSets into model inputs via `MELTProcessor` |
| **DataLoader** | PyTorch DataLoader that feeds batches to the trainer |

## Key Classes and Functions

### Dataset (`dataset.py`)

**`SpeechToTextDataset`**: The core dataset class used for both training and evaluation.

- Receives a **CutSet** (not an index) in `__getitem__`
- Loads audio, extracts text from supervisions or custom fields
- Processes through `MELTProcessor` to produce model inputs
- Supports task/language tags for multi-task training

**`FallbackDataset`**: Wrapper that provides fault tolerance by returning the previous successful batch if loading fails.

### Samplers (`dataloader.py`)

Lhotse provides two main samplers:

| Sampler | Use Case |
|---------|----------|
| `DynamicCutSampler` | Simple batching by duration/count constraints |
| `DynamicBucketingSampler` | Groups similar-duration utterances for efficient padding |

Both samplers support:
- Duration-based batching (`batch_duration`)
- Count-based batching (`max_cuts` / `batch_size`)
- Distributed training (`rank`, `world_size`)

### DataLoader Functions

| Function | Purpose |
|----------|---------|
| `get_train_dataloader_from_config` | Creates infinite dataloader for training |
| `get_eval_dataloader_from_config` | Creates finite dataloader for evaluation |
| `get_finite_dataloader_from_config` | Low-level function for finite iteration |

## Training vs Evaluation

### Training Data Loading

```
CutSet (lazy) → DynamicBucketingSampler → IterableDatasetWrapper → DataLoader
                     ↑                              ↑
               shuffle=True                    loops forever
               rank/world_size                 (infinite iteration)
```

Key characteristics:
- **Infinite iteration**: `IterableDatasetWrapper` loops forever; trainer controls epochs via `max_steps`
- **Shuffling**: Enabled for training randomization
- **Bucketing**: Groups similar-length utterances to minimize padding
- **Multi-worker**: Uses `make_worker_init_fn` for proper sharding

### Evaluation Data Loading

```
CutSet (lazy) → DynamicBucketingSampler → FiniteIterableDatasetWrapper → DataLoader
                     ↑                              ↑
               shuffle=False                   single epoch
               rank/world_size                 has __len__
```

Key characteristics:
- **Finite iteration**: `FiniteIterableDatasetWrapper` iterates exactly once
- **Deterministic**: `shuffle=False` for reproducible evaluation
- **Progress bars**: `__len__` enables proper progress tracking
- **Bucketing**: Still used for efficiency (groups by duration)

## Configuration

Data loading is configured via `DataConfig` with `train_ds` and `validation_ds` sections:

```yaml
data:
  sample_rate: 16000
  
  train_ds:
    input_cfg:
      - type: lhotse_shar
        shar_path: /path/to/train/shar
    batch_duration: 60.0     # Max seconds per batch
    batch_size: 8            # Max cuts per batch
    lhotse_sampler_type: dynamic_bucketing   # DynamicBucketingSampler
    num_buckets: 30          # Number of duration buckets
    shuffle: true
    num_workers: 4
    
  validation_ds:
    input_cfg:
      - type: lhotse_shar
        shar_path: /path/to/val/shar
    batch_duration: 60.0
    batch_size: 8
    num_workers: 2
```

## Distributed Training and Evaluation

### Training (Infinite Iteration)

For **training** with shar/tarred data:
- Sampler uses `rank=0, world_size=1` (no GPU sharding at sampler level)
- Lhotse's `make_worker_init_fn` handles sharding across both GPUs and workers
- This works because training loops infinitely and workers coordinate via the init function

### Evaluation (Finite Iteration)

For **evaluation**, a two-level sharding approach is used:

1. **GPU-level sharding** (via sampler):
   - Sampler uses `rank=global_rank, world_size=world_size`
   - Each GPU processes 1/world_size of the dataset
   
2. **Worker-level sharding** (via round-robin):
   - `FiniteIterableDatasetWrapper` assigns batches to workers via `batch_idx % num_workers == worker_id`
   - This ensures each batch is processed exactly once within a GPU

**Example with 4 GPUs and 4 workers per GPU:**
```
Total batches: 160

GPU 0 (rank=0): batches 0, 4, 8, ... (40 batches)
  └─ Worker 0: batches 0, 16, 32, ...
  └─ Worker 1: batches 4, 20, 36, ...
  └─ Worker 2: batches 8, 24, 40, ...
  └─ Worker 3: batches 12, 28, 44, ...

GPU 1 (rank=1): batches 1, 5, 9, ... (40 batches)
  ...etc
```

## Multi-Worker Data Loading

**Training** (`num_workers > 0`):
- Lhotse's `make_worker_init_fn` is passed to the DataLoader
- Each worker is configured with a different random seed
- Workers coordinate for infinite iteration

**Evaluation** (`num_workers > 0`):
- No `worker_init_fn` needed
- `FiniteIterableDatasetWrapper` handles round-robin assignment
- Each worker processes only batches where `batch_idx % num_workers == worker_id`

## MELTTrainer Integration

The `MELTTrainer` class:

1. **`__init__`**: Creates `SpeechToTextDataset` for eval, wrapped in `FallbackDataset`
2. **`get_train_dataloader`**: Uses `get_train_dataloader_from_config` (infinite)
3. **`get_eval_dataloader`**: Uses `get_eval_dataloader_from_config` (finite)

## Epoch Estimation

For infinite training dataloaders, epochs are estimated from dataset duration:

```
steps_per_epoch = total_duration / (batch_duration × world_size × grad_accum)
```

This is computed via `estimate_steps_per_epoch()` and used to convert between `num_train_epochs` and `max_steps`.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "BucketingSampler does not support lazy CutSet" | Use `DynamicBucketingSampler` instead |
| Progress bar shows wrong count | Ensure dataloader has `__len__` (use `FiniteIterableDatasetWrapper` for eval) |
| Duplicate samples in eval | Check `num_workers` settings and `worker_init_fn` |
| Infinite eval loop | Use `get_eval_dataloader_from_config`, not `get_lhotse_dataloader_from_config` |
