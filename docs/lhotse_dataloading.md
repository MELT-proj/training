# How Lhotse Dynamic Bucketing Sampler Works

Let me break down the data flow from disk to batches, focusing on shuffling, shard loading, and bucketing behavior.

## Overview: Shard Structure

First, clarify a key point: **Shards do NOT contain pre-sorted audio by length**. When you convert datasets using your scripts (e.g., `infra/setup/lhotse/librispeech.py`), the `SharWriter`:

1. Writes cuts in **the order they appear** in the source dataset
2. Groups them into shards based on `--shard-size` (e.g., 25,000 cuts)
3. **No sorting by duration happens during conversion**

So each shard contains a **random mix** of short and long utterances as they occurred in the original dataset splits.

---

## Step-by-Step: Loading and Shuffling Pipeline

### 1. **Dataset Initialization** (happens once per epoch)

```python
# In src/data/audio/lhotse/dataloader.py
cuts = CutSet.from_shar(
    in_dir="/path/to/shar/train/",
    # Lazy loading: reads shard manifests (JSON metadata), NOT audio bytes
)
```

**What happens:**
- Lhotse scans the SHAR directory and reads all `.jsonl.gz` manifest files
- Builds an in-memory index: `{cut_id: (shard_number, byte_offset)}`
- **Audio bytes stay on disk** at this point
- Memory usage: ~few hundred MB for metadata (cut IDs, durations, text, etc.)

---

### 2. **Sampler Creation** (per epoch)

```python
# Dynamic bucketing sampler
sampler = DynamicBucketingSampler(
    cuts,
    max_duration=300.0,  # Max total audio seconds per batch
    shuffle=True,
    drop_last=True,
    buffer_size=10000,   # Key parameter!
)
```

**What happens:**
- The sampler creates an **internal shuffle buffer** (size = `buffer_size`, e.g., 10,000 cuts)
- This buffer will hold cut metadata (not audio bytes) during iteration

---

### 3. **Iteration Starts** (each training step)

When the DataLoader worker calls `next(sampler)`, here's the flow:

#### **Step 3a: Fill the shuffle buffer**

```
┌─────────────────────────────────────────┐
│ Disk: SHAR shards (shard-000000.tar)   │
│   ├─ cut_0001.flac (metadata in .jsonl)│
│   ├─ cut_0002.flac                      │
│   └─ ...                                │
└─────────────────────────────────────────┘
              ↓ (sequential read of manifests)
┌─────────────────────────────────────────┐
│ Shuffle Buffer (in RAM, metadata only) │
│   [cut_0001, cut_0045, cut_0123, ...]  │  ← 10,000 random cuts
│   sorted by duration                    │
└─────────────────────────────────────────┘
```

**Details:**
- Lhotse reads cuts **sequentially from shards** until the buffer has `buffer_size` cuts
- **Which shard?** It reads shards in order: `shard-000000.tar`, then `shard-000001.tar`, etc.
- **Shuffling:** Once the buffer is full, Lhotse **shuffles the buffer randomly**
- **Sorting:** After shuffling, cuts are **sorted by duration** (ascending or descending based on config)

#### **Step 3b: Create a batch (bucketing)**

```
Buffer (sorted by duration):
[2.3s, 2.5s, 2.8s, ... 18.1s, 18.5s, 19.2s]
              ↓
Pick cuts greedily until max_duration reached:
Batch 1: [2.3s, 2.5s, 2.8s, 3.1s, ...]  → total ~300s
              ↓
Return batch of cut IDs to DataLoader worker
```

**Bucketing logic:**
- Start from the **shortest** cut in the buffer
- Keep adding cuts until `sum(durations) >= max_duration` (e.g., 300s)
- This creates **batches with similar-length utterances** (natural bucketing)
- Remove those cuts from the buffer

#### **Step 3c: Refill buffer** (streaming behavior)

After emitting a batch:
- Buffer now has fewer cuts (e.g., 9,800 left)
- Lhotse **refills** by reading the next cuts from disk (continuing sequentially through shards)
- Shuffles the **new cuts only**, then merges with remaining buffer
- Re-sorts by duration

---

### 4. **Audio Loading** (lazy, happens in DataLoader worker)

**Key point:** Audio bytes are loaded **AFTER** the sampler returns cut IDs.

```python
# In the DataLoader collate function
for cut_id in batch:
    audio_bytes = read_from_shar(cut_id)  # Seeks to byte offset in .tar
    audio = decode_flac(audio_bytes)      # Decode to waveform
    features = processor(audio)           # Feature extraction
```

**What happens:**
- The DataLoader worker **seeks directly** to the byte offset in the relevant `.tar` file (from the metadata index)
- Reads compressed audio bytes (FLAC/WAV)
- Decodes in memory
- **No entire shard is loaded**—only the specific cuts needed for this batch

---

## Answering Your Specific Questions

### Q: How does the loader decide which shard to load to memory?

**A:** It doesn't load entire shards into memory. Instead:
1. **Manifests** (`.jsonl.gz`) are read sequentially to fill the shuffle buffer with metadata
2. **Audio bytes** are loaded **on-demand** per-cut when building batches
3. The sampler reads shards in **sequential order** (shard-000000, shard-000001, ...) to fill the buffer, but audio loading is **random-access** based on batch composition

### Q: Does each shard contain only examples of similar length?

**A:** No. Shards contain cuts in the order they appeared in the source dataset (e.g., LibriSpeech's original ordering). Bucketing happens **dynamically at sampling time**, not during conversion.

---

## Visual Summary: Full Pipeline

```
Epoch Start
    ↓
┌──────────────────────────────────────────────────────────┐
│ 1. Read shard manifests sequentially                     │
│    shard-000000.jsonl.gz → shard-000001.jsonl.gz → ...   │
│    (metadata only, ~10k cuts)                            │
└──────────────────────────────────────────────────────────┘
    ↓
┌──────────────────────────────────────────────────────────┐
│ 2. Fill shuffle buffer (10k cuts metadata in RAM)       │
│    Shuffle randomly                                      │
│    Sort by duration                                      │
└──────────────────────────────────────────────────────────┘
    ↓
┌──────────────────────────────────────────────────────────┐
│ 3. Emit batch (greedy bucketing by duration)            │
│    Batch: [cut_001, cut_045, cut_123, ...]              │
│    (similar durations, ~300s total)                      │
└──────────────────────────────────────────────────────────┘
    ↓
┌──────────────────────────────────────────────────────────┐
│ 4. DataLoader worker loads audio bytes (lazy)           │
│    For each cut_id:                                      │
│      - Seek to byte offset in .tar                      │
│      - Read FLAC bytes                                   │
│      - Decode to waveform                               │
│      - Extract features                                  │
└──────────────────────────────────────────────────────────┘
    ↓
Training step processes batch
    ↓
Next batch: refill buffer from next cuts in shards, repeat
```

---

## Key Parameters in Your Config

From `src/config.py` and `src/data/audio/lhotse/dataloader.py`:

```python
@dataclass
class DatasetConfig:
    # ...
    max_duration: float = 300.0        # Max seconds per batch
    buffer_size: int = 10000   # Buffer size (affects randomness vs memory)
    num_workers: int = 8               # DataLoader workers (parallel audio loading)
```

### Tuning `buffer_size`:
- **Larger** (e.g., 50k): Better shuffle quality, more memory
- **Smaller** (e.g., 5k): Less memory, slightly less random (but usually fine)
- **Rule of thumb**: 10k-20k is good for most cases

---

## Indexed Shar: how ranks and workers get different data

A Shar source can be read two ways, chosen by `indexed` in the dataset config.

**Streaming** (no `.idx` sidecars). Nothing partitions the corpus, so every DP rank
and every DataLoader worker iterates *all* of it. Separation is statistical: each
process walks the shards in its own random order. Because independent streams sample
the same corpus with replacement, one nominal epoch covers about `1 - e⁻¹` ≈ **63.2%**
of the data, the rest being repeats.

**Indexed** (`.idx` sidecars present). Lhotse partitions by *sample index* across the
whole `rank × worker` pool, so every cut is produced **exactly once** and an epoch is
100% of the data. Measured on `cv22_sidon/it/train` (172,828 cuts) at `world_size=4`:

| | per rank | coverage | cross-rank duplicates |
|---|---|---|---|
| indexed | 4 × 43,207 | 172,828 / 172,828 | **0** |
| streaming | 4 × 172,828 | 172,828 / 172,828 | 518,484 |

```yaml
train_ds:
  indexed: null   # null = auto-detect per source; true = require; false = force streaming
```

Two consequences worth knowing:

- **`shard_seed: randomized` does not apply to indexed sources.** Partitioning already
  separates the streams, so shard order should be identical everywhere — and lhotse
  refuses a randomized seed under a multiplexer over indexed sources. The loader falls
  back to `seed` and logs a warning. Set `shard_seed` to an integer to silence it.
- **`num_workers` must be ≥ 1.** Partitioning is armed by `make_worker_init_fn`, which
  only runs inside a DataLoader worker subprocess. At `num_workers: 0` the partition
  collapses and every rank reads everything; the loader raises rather than allow it.

To convert a collection, see `infra/index_shar.py`. The `.gz` manifests are replaced by
plain `.jsonl` permanently — the index stores byte offsets into them — so the migration
is not reversible by re-compressing.

---

## Why This Design?

1. **Memory efficiency**: Only metadata in RAM, audio stays on disk until needed
2. **I/O efficiency**: Sequential manifest reads + random-access audio loads (good for HDDs and network FS)
3. **Dynamic batching**: Bucketing happens at sampling time, so you get efficient batches without pre-sorting the entire dataset
4. **Shuffle quality**: The buffer provides good randomness while keeping memory bounded

This is why Lhotse works well for large-scale speech training on HPC systems like yours (shared FS, multi-GPU, long utterances).
