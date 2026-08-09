# lhotse 2.x validation campaign

Three MN5 tests that gate merging `feat/lhotse-2-indexed-shar` into `main`.
Two properties — no batch overlap, exact resume — were already proven at
1 node × 4 GPUs. These cover the three places the branch is most likely to
break in production and were not exercised before: a source too small to
partition, multi-node resume, and a long two-task run.

All three read the indexed collection at
`/gpfs/projects/epor48/melt-data/shar-indexed/` (793 sources, plain `.jsonl` +
`.idx`, `.tar` hard-linked from `shar/`). The shared `shar/` tree keeps its
`.gz` and carries no `.idx`, so lhotse 1.32 consumers are unaffected.

**Budget: the campaign must not exceed 500 GPUh.** Accounting follows *elapsed*
time, not the requested wall — `sacct` reports `CPUTimeRAW` equal to
`Elapsed x AllocCPUS` on jobs that finished well inside their limit. What you are
charged for regardless of use is the whole node: MN5 acc nodes allocate all 80
CPUs and 4 GPUs, so a one-GPU job still bills four. Asking for a generous wall is
therefore cheap; asking for more nodes is not.

## Test 0 — local pre-flight (0 GPUh)

Run before anything is queued; each catches a failure that would otherwise cost
a full allocation.

| Check | Result |
|---|---|
| `LazyRepeater` on a starved partition | Returns rather than looping — it has an explicit `if not at_least_once: return` guard. No hang is possible. |
| `CutSet.mux` on an exhausted source | `stop_early` defaults to `False` and is never overridden, so the source is dropped and the stream continues. |
| Eval path on an indexed-only tree | Works. `materialize_cuts_for_eval` never passes `indexed=`, but lhotse auto-detects and reads the `.jsonl`/`.idx` pair with no `.gz` present. |
| Configs load against real data | All three; language codes, task tags and ST target text verified against real manifests. |

A CPU repro at the same shape as Test 1 (3-cut source, 8 partitions) showed
partitions 0–2 receiving the source and 3–7 receiving none of it, with
disjointness intact and no stream ending early.

## Test 1 — starved partitions (1 node × 4 GPUs, ~4 GPUh)

```bash
tests/integration/lhotse2_campaign/run_test1_mn5.sh
# then, once it finishes:
python tests/integration/lhotse2_campaign/check_test1.py \
    --cut-ids-dir  <OUTPUT_DIR>/<exp>/debug_cut_ids \
    --shar-root    /gpfs/projects/epor48/melt-data/shar-indexed
```

The mixture holds `voxpopuli/lt/validation` (**3 cuts**, fewer than the 8
partitions) and `voxpopuli/lt/test` (**42 cuts**, ~5 per partition) alongside
two bulk sources. Weights are explicit — automatic weighting is
cut-proportional, so a 3-cut source among 160k would never be drawn and the
test would prove nothing.

`check_test1.py` attributes every emitted cut ID to its source by reading the
manifests, so the per-partition composition is exact rather than inferred.

Pass: reaches step 20, all 8 streams emit, zero shared cut IDs across all 28
pairs. The per-source reach table is the finding — a source that reaches fewer
than 8 partitions has its effective mixture weight silently scaled by
`min(cuts, partitions) / partitions`.

## Test 2 — two-node resume (2 nodes × 4 GPUs, ~16 GPUh)

```bash
tests/integration/lhotse2_campaign/run_test2_mn5.sh run1     # wait for it
tests/integration/lhotse2_campaign/run_test2_mn5.sh prune
tests/integration/lhotse2_campaign/run_test2_mn5.sh run2     # wait for it
tests/integration/lhotse2_campaign/run_test2_mn5.sh compare
```

8 ranks × `num_workers: 2` = **16 streams**. `world_size` has never varied in
any previous resume test, and `PartitionedIndexedIterator` refuses to restore
under a changed topology — this checks it restores correctly under an unchanged
one. run1 → prune to `checkpoint-5` → run2 → `compare_cut_ids.py`, in one
allocation.

Unlike `sampler_resume/run_test_2node_fsdp2.sbatch`, which prints `PASSED`
unconditionally, this driver accounts for each phase's status and exits
non-zero on failure.

Pass: zero shared IDs across all 16 streams, and `run1[split:] == run2[:]` on
every stream.

## Test 3 — long mixed-task soak (2 nodes × 4 GPUs, ~325 GPUh)

```bash
tests/integration/lhotse2_campaign/run_test3_mn5.sh smoke   # 40 min, 5.3 GPUh
tests/integration/lhotse2_campaign/run_test3_mn5.sh legA    # 20 h, 160 GPUh
tests/integration/lhotse2_campaign/run_test3_mn5.sh legB    # 20 h, 160 GPUh
```

ASR (VoxPopuli de/es/nl) mixed 60/40 with ST (CoVoST2 nl→en, it→en, pt→en).
`nl` appears under both tasks, which is the sharpest available check that task
tagging and prompt selection actually separate them — same input language, two
different targets.

Validation is split into six **named** sets (`asr_de`, `asr_es`, `asr_nl`,
`st_nl_en`, `st_it_en`, `st_pt_en`), which HF reports as `eval_<name>_loss`.
Per-task figures are the mean over the `asr_*` and `st_*` sets.

`legB` resumes from `legA`, so the soak test doubles as a full-scale resume
check at real duration. Both legs must keep 2 nodes × 4 GPUs × 2 workers or the
sampler will refuse to restore.

Pass: no discontinuity in the loss curve at the legA/legB seam; no NaN, OOM or
stall; all six eval curves below their `eval_on_start` baseline.

## Notes carried by these configs

- **Eval formatting.** The eval collator resolves `apply_chat_template` from
  `validation_ds`, not from `data`, so it defaults to `False` while training
  uses the chat template — a validation loss over a different sequence format
  than the training loss is not comparable to it. Test 3's `validation_ds` sets
  `apply_chat_template`, `prompt_template_selection` and `chat_template_config`
  explicitly. The inconsistent default is a separate bug.
- **Flat `input_cfg`, not `type: group`.** Group tags are applied after the
  children's and replace `cut.tags` wholesale, so a group carrying `task: st`
  would clobber each leaf's `src_lang`/`tgt_lang`.
- **No `evaluation:` section.** Its mere presence enables `compute_metrics`.
  WER is meaningless on ST, and the per-language WER split is mis-attributed
  under DDP because the langs buffer is rank-local.
- **CoVoST2 target text.** `supervisions[0].text` is the translation; the
  source-language transcript is in `custom.sentence`. The default
  `text_field: text` is already correct for ST.
