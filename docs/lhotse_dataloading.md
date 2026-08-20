# How Lhotse builds batches

This walks the data path from disk to a training batch: what each component does,
where randomness enters, and which knobs trade against which.

Source of truth for the MELT side is `melt/training/data/audio/lhotse/dataloader.py`;
the sampler internals described here are lhotse's
`dataset/sampling/dynamic_bucketing.py`.

> **Lhotse 2 changed the on-disk contract.** Shar collections are now *indexed*, and
> that is not merely a partitioning detail — it changes what a cut holds in memory and
> how audio is read. See [Indexed Shar](#indexed-shar-the-lhotse-2-change) below.

---

## The unit: a cut

A **cut** is one training example: a pointer to a span of audio, plus its supervisions
(transcript, language, speaker) and any custom metadata.

A cut is *metadata*. It describes where the audio lives; whether it also carries the
audio itself depends on the reader, which is the crux of the indexed change.

---

## 1. Shards on disk

Shar stores a dataset as numbered pairs of files:

```
cuts.000000.jsonl        ← metadata for a group of cuts
recording.000000.tar     ← the matching audio blobs
cuts.000001.jsonl
recording.000001.tar
```

A **shard** is one such pair. Sharding means readers stream sequentially through a tar
instead of opening millions of small files.

**Shards are not sorted by duration.** `SharWriter` writes cuts in the order they appear
in the source dataset, so every shard holds a random mix of short and long utterances.
All length grouping happens at sampling time (§6), never during conversion.

---

## 2. Shards become a lazy stream of cuts

`CutSet.from_shar()` returns a **lazy** CutSet — an iterator, not a list. Nothing is
loaded up front; pulling from it yields cuts one at a time, shard by shard
(`dataloader.py:585`):

```python
cuts = CutSet.from_shar(
    in_dir=shar_path,
    shuffle_shards=shuffle,
    seed=shard_seed,
    stateful_shuffle=shuffle,
    indexed=indexed,
)
```

- **`shuffle_shards`** permutes the *order shards are visited*. Coarse-grained: it
  decides which region of the corpus you meet when, not the order within a region.
- **`stateful_shuffle`** makes that permutation resumable, so a restarted job continues
  the traversal instead of replaying from the top.

For training the stream is also wrapped in `repeat()`, making it infinite — the epoch
boundary becomes a bookkeeping convention rather than the end of the iterator.

---

## 3. Many datasets become one stream

Training runs over many sources, so `CutSet.mux()` interleaves them into a single
stream, drawing from each by weight (`dataloader.py:633`):

```python
return CutSet.mux(*cutsets, weights=weights, seed=shard_seed), use_iterable, total_cuts
```

**This is the layer that controls language and task balance**, and it sits *upstream* of
everything below. Nothing further down can change how much German you see — in
particular, `buffer_size` cannot. See `docs/mixture_weights.md` for how weights resolve.

---

## 4. The stream splits across workers

Each DP rank runs a DataLoader with some number of worker processes. Every worker gets
its **own private copy of the whole pipeline** — its own mux, its own sampler, its own
buffer — plus a partition identifier saying which slice of the data is its job.

`make_worker_init_fn` gives worker `w` of rank `r` the partition
`(rank = r * num_workers + w, world_size = world_size * num_workers)`. Workers never
coordinate; they produce batches independently and the DataLoader interleaves them.

Note the consequence: **`buffer_size` is per worker, per rank.** Total resident cuts is
`buffer_size × world_size × num_workers`.

---

## 5. The buffer: a reservoir for shuffling

The sampler cannot shuffle a stream it has not read. So before yielding anything it
pulls `buffer_size` cuts out of the muxed stream and holds them
(`dataloader.py:850`, default `10000`). That pool is the window it gets to be random
within.

Two properties matter:

- **The initial fill is blocking.** `_collect_cuts_in_buckets(self.buffer_size)` runs
  synchronously before the first batch, so nothing trains until the whole buffer is
  read. This is a one-time startup cost that grows with the buffer — and grows
  *superlinearly* in practice, since a wider window draws from more shards at once.
  (Lhotse has a background-producer mode, but `concurrent` defaults to `False` and MELT
  does not pass it.)
- **Refill is incremental.** After each batch the sampler pulls exactly as many new cuts
  as the batch consumed, so occupancy stays constant and the window slides along the
  stream.

---

## 6. The buffer is organised into buckets

The pool is not one flat pile. It is **`num_buckets` FIFO queues split by duration**,
with boundaries given by `bucket_duration_bins` — bucket 0 holds the shortest cuts, the
last the longest. As each cut arrives from the stream its duration decides which bucket
it is filed into.

The point is **padding efficiency**. Batching a 3-second clip with a 30-second clip
means padding the short one out and burning most of the compute on silence. Drawing a
batch from a single bucket keeps its members similar in length.

`num_buckets` sets how finely you slice: more buckets means tighter length matching, but
each bucket holds proportionally fewer cuts to choose from.

Use `infra/estimate_bucket_bins.py` to measure bins against a real mixture rather than
guessing them.

---

## 7. Building one batch

Per batch, `DynamicBucketer.__iter__` does the following.

1. **Pick a bucket** (`_select_bucket`). It finds buckets that are *ready* — `_is_ready`
   accumulates durations until the constraint is close to exceeding, i.e. "can this
   bucket fill a whole batch?" — and chooses among them. If none is ready it either
   stops or emits a partial batch, per `drop_last`.
2. **Shuffle that bucket** (`pick_at_random`) and draw cuts in the shuffled order.
3. **Add cuts until a constraint trips** (`DurationBatcher`):
   - `batch_duration` caps total seconds of audio per batch,
   - `batch_size` caps the cut count,
   - `quadratic_duration` adds an extra penalty for long utterances, because attention
     cost grows faster than linearly with sequence length — so a batch of long clips is
     closed earlier than raw duration alone would suggest.
4. **Remove the used cuts** from the bucket and **refill** the same number from the
   stream.

So a batch is neither a contiguous slice of the data nor a draw from the whole dataset:
it is a random draw from **one duration bucket of the current buffer**.

> **Naming trap.** In the MELT config, `batch_duration` is what becomes lhotse's
> `max_duration` sampler argument (`DynamicBucketingSampler(cuts, max_duration=batch_duration, …)`).
> The config's own `max_duration` is something else entirely — a **per-cut filter** that
> drops utterances longer than the threshold before they ever reach the sampler. Do not
> read one for the other.

---

## 8. Audio is read last

Everything above moved metadata only. Once a batch of cuts is chosen, the **Dataset**
does the real work: for each cut it goes to the recording tar, reads the audio, decodes,
extracts features, tokenises the text, pads, and stacks into tensors.

The I/O pattern here is therefore determined by *which cuts ended up together in a
batch* — which is decided by the buffer and the buckets upstream.

---

## Indexed Shar: the lhotse 2 change

A Shar source can be read two ways, chosen by `indexed` in the dataset config. Lhotse 2
made the indexed path the one we use, and it differs on **three** axes, not just the
partitioning one that gets discussed most.

```yaml
train_ds:
  indexed: null   # null = auto-detect per source; true = require; false = force streaming
```

### (a) Coverage — every cut exactly once

**Streaming** (no `.idx` sidecars). Nothing partitions the corpus, so every DP rank and
every DataLoader worker iterates *all* of it. Separation is statistical: each process
walks the shards in its own random order. Because independent streams sample the same
corpus with replacement, one nominal epoch covers about `1 - e⁻¹` ≈ **63.2%** of the
data, the rest being repeats.

**Indexed** (`.idx` sidecars present). Lhotse partitions by *sample index* across the
whole `rank × worker` pool, so every cut is produced **exactly once** and an epoch is
100% of the data. Measured on `cv22_sidon/it/train` (172,828 cuts) at `world_size=4`:

| | per rank | coverage | cross-rank duplicates |
|---|---|---|---|
| indexed | 4 × 43,207 | 172,828 / 172,828 | **0** |
| streaming | 4 × 172,828 | 172,828 / 172,828 | 518,484 |

### (b) Memory — offsets instead of audio bytes

This is the part that is easy to miss. The two readers fill a cut's audio placeholder
differently:

- **Streaming** → `shar/readers/tar.py` → `fill_shar_placeholder`, which does
  `sources[0].type = "memory"; sources[0].source = data`. The **encoded audio bytes are
  attached to the cut** and stay resident for as long as that cut sits in the sampler
  buffer.
- **Indexed** → `shar/readers/indexed.py` → `fill_shar_placeholder_lazy`, which stores
  `(tar_path, offset, end_offset)`. The cut stays lightweight; the audio is fetched by
  seeking at collate time (§8).

So under lhotse 2 the buffer holds manifests and file offsets, and its memory cost is
roughly independent of audio length. Under streaming, buffer memory scaled with the
audio itself — which is why large buffers used to be far more dangerous than they are
now.

### (c) Read pattern — seeks instead of sequential scans

The flip side of (b): indexed reads are **random-access seeks** into tars at collate
time, where streaming reads were sequential scans. This is what couples batch diversity
to I/O cost (see the tradeoff below), and it is more pronounced on a shared or network
filesystem than on local scratch.

### Two operational consequences

- **`shard_seed: randomized` does not apply to indexed sources.** Partitioning already
  separates the streams, so shard order should be identical everywhere — and lhotse
  refuses a randomized seed under a multiplexer over indexed sources. The loader falls
  back to `seed` and logs a warning. Set `shard_seed` to an integer to silence it.
- **`num_workers` must be ≥ 1.** Partitioning is armed by `make_worker_init_fn`, which
  only runs inside a DataLoader worker subprocess. At `num_workers: 0` the partition
  collapses and every rank reads everything; the loader raises rather than allow it.

To convert a collection, see `data-utils/index_shar.py` in MELT-proj/preprocessing. The
`.gz` manifests are replaced by plain `.jsonl` permanently — the index stores byte
offsets into them — so the migration is not reversible by re-compressing.

---

## Where randomness comes from

Three independent levels. Only the third is `buffer_size`.

| level | what it randomises | controlled by |
|---|---|---|
| shard order | which region of a source you meet when | `shuffle_shards` |
| source interleaving | the dataset / language mixture | `mux` weights |
| within-buffer | which specific cuts share a batch | `buffer_size` |

---

## The `buffer_size` tradeoff

A **bigger** buffer means each bucket holds a wider sample of the stream, so the cuts
that end up in a batch were drawn from further apart — more decorrelated batches.

A **smaller** buffer means a batch's members were neighbours in the stream, so they are
likelier to share a source, a speaker, or a recording session.

The costs on the other side:

1. **A longer blocking fill** before the first step (§5).
2. **More memory held** — though under indexed Shar this is manifests only (§b), so it
   is far cheaper than it was under streaming.
3. **More scattered reads.** Because a batch's cuts came from further apart in the
   stream, their audio lives at more scattered offsets across more tar files, so the
   collate step does more seeking and less sequential reading.

Point 3 is the structural one: **batch diversity and read locality are the same dial
pointing in opposite directions.** The very property that makes a batch diverse is that
its members came from far-apart places on disk.

What `buffer_size` does *not* affect: mixture balance (fixed by mux, §3), shard
shuffling (§2), or batch fullness (bucketing keeps batches packed regardless, §6).

### Sizing it

Think in terms of **pool per bucket** — `buffer_size / num_buckets` — measured against
the number of cuts in a batch. That ratio, not the raw buffer number, is what determines
how much freedom the sampler has when composing a batch.

Note also that resident cuts scale as `buffer_size × world_size × num_workers` (§4), so
a value that is fine on one GPU is multiplied across a run.

---

## Visual summary

```
  shards on disk (cuts.NNNNNN.jsonl + recording.NNNNNN.tar + .idx)
        │  shuffle_shards: permute shard visit order
        ▼
  lazy CutSet per source  ──repeat()──▶ infinite stream
        │
        ▼
  CutSet.mux(weights)          ← language / dataset balance decided HERE
        │
        ▼
  per (rank × worker) partition by sample index      [indexed only]
        │
        ▼
  ┌─────────── sampler buffer (buffer_size cuts) ───────────┐
  │  filed on arrival into num_buckets duration queues      │
  │   [bucket 0: short] … [bucket N: long]                  │
  └─────────────────────────────────────────────────────────┘
        │  pick a ready bucket → shuffle it → draw until
        │  batch_duration / batch_size / quadratic_duration trips
        ▼
  batch of cuts (metadata only)
        │
        ▼
  Dataset: seek to (tar, offset) → decode → features → tokenise → pad → stack
        │
        ▼
  training step
```

---

## Why this design

1. **Memory efficiency** — only metadata is resident; audio stays on disk until needed.
2. **Bounded shuffling** — the buffer gives randomness without materialising the corpus.
3. **Dynamic batching** — length grouping happens at sampling time, so no pre-sorting
   pass over the dataset is required.
4. **Exact coverage** — indexed partitioning makes an epoch mean what it says.

Which is why Lhotse suits large-scale speech training on shared-filesystem HPC: the
expensive resource is random I/O, and every layer above is arranged to keep metadata
cheap and defer audio reads to the last possible moment.
