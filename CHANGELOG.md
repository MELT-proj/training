# Changelog

Notable changes to MELT, by release. This is a research codebase with a small
team and long-lived branches, so a release here means "the commit that bumped
`pyproject.toml`'s `version`", not a tagged, published artifact — there are no
git tags. Dates are the bump commit's own commit date.

The entries from v0.1.0 through v0.4.0 were reconstructed after the fact from
commit history and merged-PR titles (2026-09-09), since no changelog was kept
at the time. Some editorial judgment was used to group commits; treat those
entries as approximate. From v0.5.0 onward, PR numbers and titles are quoted
directly from GitHub.

## [Unreleased]

Targeting v0.8.0, pending merge of both:

- [#107](https://github.com/MELT-proj/training/pull/107) — a per-encoder spec
  table (`melt/modeling/encoder_specs.py`) describing each supported speech
  encoder's I/O contract (feature layout, frame rate, fixed vs. variable input
  window, attention-mask handling), and `openai/whisper-large-v3` as a new
  encoder option built on it.
- [#122](https://github.com/MELT-proj/training/pull/122) — `facebook/mms-1b`
  as a new speech encoder (the `wav2vec2` family, raw-waveform input via its
  own feature extractor), and an `attn_implementation` config path for the
  audio encoder (`model.encoder.attn_implementation`), matching the one the
  decoder already had. Measured on MN5: `flash_attention_2` is 1.6x faster
  than `sdpa` on MMS, and matches w2v-BERT-2.0's step time despite MMS being
  1.7x larger.
- Fixed `utils/sync_wandb.sh` silently dropping the eval hypotheses
  `wandb.Table` on sync (#115): wandb bakes the container-internal staging
  path into the offline run log, which never resolves once the run is
  rsynced off MN5. A new `--staging-path` mirrors that directory too and runs
  `wandb sync` inside the run's own Singularity image so the path resolves.

## [0.6.2] - 2026-09-08

- [#109](https://github.com/MELT-proj/training/pull/109) — upgraded
  `transformers` 4.57.1 -> 5.16.1 (unblocks Qwen 3.5 support).
- Fixed an `AttributeError` that broke every DDP and FSDP run.
- [#103](https://github.com/MELT-proj/training/pull/103),
  [#110](https://github.com/MELT-proj/training/pull/110) — declarative,
  parameterised ablation-campaign launchers (data/architecture/optimisation
  axes as a grid), a status view, an enforced run ledger, and a
  `campaign.py run --time` override for per-submission wall-clock budgets.
- Let MELT models accept gradient checkpointing.
- [#119](https://github.com/MELT-proj/training/pull/119) — let inference
  choose its attention implementation, and bucket batches by length.
- [#111](https://github.com/MELT-proj/training/pull/111),
  [#112](https://github.com/MELT-proj/training/pull/112) — fixed a
  Qwen3.5-2B OOM on the MA-700 arm (freed eval memory, added an FSDP2 config),
  and added `skip_text_field_mismatch` to tolerate rare bad cuts instead of
  crashing training.
- IFT-700: checkpoint roughly every 100h per language, plus a final
  checkpoint, with an overridable topology.
- Added multi-cluster HPC guidance to `AGENTS.md`, including a documented
  CPU-only SLURM test job on Artemis.

## [0.6.1] - 2026-08-31

- [#81](https://github.com/MELT-proj/training/pull/81),
  [#86](https://github.com/MELT-proj/training/pull/86),
  [#87](https://github.com/MELT-proj/training/pull/87),
  [#89](https://github.com/MELT-proj/training/pull/89),
  [#90](https://github.com/MELT-proj/training/pull/90),
  [#98](https://github.com/MELT-proj/training/pull/98),
  [#100](https://github.com/MELT-proj/training/pull/100) — hardened
  generation-based eval: fixed eval references dropping chat-template
  scaffolding, collapsed redundant eval breakdowns, applied the configured
  `attn_implementation` when loading from a checkpoint (previously silently
  fell back and made generation ~12x slower), verified saved weights, and
  fixed unquoted braced prompt-template CLI overrides.
- [#84](https://github.com/MELT-proj/training/pull/84) — bounded the number
  of open Lhotse Shar shard files (`MELT_SHAR_OPEN_SHARDS`) and raised the
  open-file limit so indexed-Shar runs survive.
- Added an opt-in host-memory tracer ("memtrace") for the MN5 host-RAM wall:
  per-process memory/fd tracking, optional worker `tracemalloc`, checkpointing
  before the host-RAM limit is hit.
- [#83](https://github.com/MELT-proj/training/pull/83),
  [#79](https://github.com/MELT-proj/training/pull/79) — per-task/language
  training-hours accounting, and derived `MASTER_PORT` from the job ID so it
  actually reaches the rendezvous (fixed a collision on two-rank Artemis
  runs).
- Fixed silent job-submission failure on MN5 when `MELT_PARTITION` is unset;
  added a cluster reference doc for Sardine and MareNostrum5.
- Ran multiple new training campaigns (MA-Llama3.2-1B-Instruct,
  ABL-IFT-125, IFT-700-arm, MA-700-bins-bd150, MA-700-chat-template) and
  added a `wandb` sync script.

## [0.6.0] - 2026-08-19

- [#80](https://github.com/MELT-proj/training/pull/80) — replaced
  teacher-forced eval with `generate()`-based greedy decoding; required
  unsharding FSDP params around `generate()` and casting audio features to
  the encoder dtype.
- [#72](https://github.com/MELT-proj/training/pull/72) — added a
  configuration consistency checker (`infra/check_training_config.py`) for
  training configs, and fixed the bucket-estimation script.
- [#71](https://github.com/MELT-proj/training/pull/71) — SFT v1.4.0 training
  support for the Yodas AST/ASR mix.
- [#74](https://github.com/MELT-proj/training/pull/74) — fixed an
  epoch-length bias tied to `num_workers`, and a race where `pad_token_id`
  could exceed vocab size when tokens are added.
- Dropped unused attention matrices/hidden states from forward passes to
  reduce memory.
- Set `seed=42` everywhere after moving to index-based data partitioning
  (removed prior randomization).
- [#69](https://github.com/MELT-proj/training/pull/69),
  [#70](https://github.com/MELT-proj/training/pull/70) — added campaign
  artifacts and a checked ABL-MA-125 config; retuned ablation batching knobs
  against measured throughput and narrowed the ablation campaign to the
  ASR-only 125h render.
- Dropped the "noreshard" accelerate config (doesn't survive larger batch
  sizes); passed the HF token and `WANDB_DATA_DIR` through to the container.

## [0.5.3] - 2026-08-17

- [#69](https://github.com/MELT-proj/training/pull/69) — fixed
  `strict_text_field` handling to skip textless cuts instead of raising.

## [0.5.2] - 2026-08-12

- [#62](https://github.com/MELT-proj/training/pull/62),
  [#64](https://github.com/MELT-proj/training/pull/64) — fixed eval
  per-language metric alignment, and the dataloader teardown exit code (a
  finished run now exits 0).
- Made the experiment tracker configurable/pluggable instead of hardwired to
  W&B.
- Promoted the Lhotse-2 container image to the default `.sif`.
- Brought the HPC runbook back in line with the cluster, and added a
  3-language MA (multi-adapter) training config.
- [#65](https://github.com/MELT-proj/training/pull/65) — set up
  worktree-based ablation-campaign infra; made `sync_wandb.sh` generally
  usable and fixed its venv activation.

## [0.5.1] - 2026-08-10

- [#61](https://github.com/MELT-proj/training/pull/61) — fixed eval to
  inherit the same chat-template formatting keys used in training (could
  previously diverge).

## [0.5.0] - 2026-08-10

- [#57](https://github.com/MELT-proj/training/pull/57) — migrated to Lhotse
  2.0.0a3 indexed Shar (added a migration tool), validated on MN5; indexed
  Shar reads prevent ranks/workers from duplicating data.
- Changed dataloader resume to snapshot per-worker state instead of replaying
  a step count; fixed the shard-traversal seed and pre-model-init seeding.
- Reported validation loss broken out per named eval set.
- Retired `use_bucketing` from defaults — now explicitly refused
  (**breaking**).
- [#38](https://github.com/MELT-proj/training/pull/38),
  [#40](https://github.com/MELT-proj/training/pull/40),
  [#42](https://github.com/MELT-proj/training/pull/42),
  [#43](https://github.com/MELT-proj/training/pull/43),
  [#49](https://github.com/MELT-proj/training/pull/49) — fixed the default
  eval-dataloader worker count and container OMP/CPU pinning.
- Made MN5 wall-clock/QoS configurable per submission; corrected SLURM
  accounting to use elapsed, not requested, wall time.
- Adjusted CI to run once per PR plus on demand via `/test`.

## [0.4.0] - 2026-08-04

- [#40](https://github.com/MELT-proj/training/pull/40) — fixed containerized
  runs pinning every rank to a single CPU core; made per-rank
  `OMP_NUM_THREADS` actually apply, and logged CPU affinity at startup.
- [#36](https://github.com/MELT-proj/training/pull/36) — simplified and
  consolidated the training launcher scripts.
- [#8](https://github.com/MELT-proj/training/pull/8),
  [#19](https://github.com/MELT-proj/training/pull/19),
  [#24](https://github.com/MELT-proj/training/pull/24),
  [#30](https://github.com/MELT-proj/training/pull/30) — merged several
  months of accumulated `dev`-branch work into `main`.

## [0.3.0] - 2026-04-17

- [#27](https://github.com/MELT-proj/training/pull/27) — end-to-end support
  for IWSLT 2026 metric training: dataset-to-Shar converter, training/
  inference config, custom eval loop, README.
- Added sampler-resume support across dataloader configs, with integration
  tests and deferred sampler loading for determinism under multiple workers.
- Vectorized `_inject_tensor` for multimodal embedding injection
  (performance).
- Added a preallocation/warmup training step to surface OOMs ahead of time,
  plus memory-profiling dumps on OOM and GPU memory monitoring.
- Enabled activation checkpointing in the FSDP config.
- Added English number/text normalizers and evaluation metrics.
- Various dataloader/config/checkpoint-resume bugfixes.

## [0.2.0] - 2026-02-12

- [#19](https://github.com/MELT-proj/training/pull/19) — Lhotse-based data
  loading pipeline integrated with the HF `Trainer` (`SpeechToTextDataset`,
  custom dataloader/sampler wiring), and improvements to the webdataset
  converter script plus new runners.
- Added qformer and conformer adapter support to the MELT architecture.
- Refactored configuration handling onto OmegaConf.
- Added multi-node training infra: FSDP2 accelerate config, DeepSpeed
  multi-node config, SLURM + Singularity support, and MareNostrum5 (MN5)
  runners.
- Added dataset-to-Shar converters for LibriSpeech, People's Speech,
  VoxPopuli, FLEURS, CommonVoice22, and Sidon webdatasets.
- Added a bucket-duration estimation script for dynamic batching.
- [#8](https://github.com/MELT-proj/training/pull/8),
  [#24](https://github.com/MELT-proj/training/pull/24) — fixed label
  construction/padding after audio resampling, and `pad_token_id`/
  `loss_ignore_index` handling for correct HF loss scaling; assorted
  distributed-training bugfixes.
- Added Lhotse data-loading docs and a GitHub Actions CI workflow.

## [0.1.0] - 2025-12-05

Initial commit — repo scaffolding for the MELT speech-LM architecture:

- `SpeechLMForConditionalGeneration` model.
- MELT configuration classes for the projector and the main model.
- `MELTProcessor` for combined audio/text inputs.
- Initial audio dataset loading utilities.
- Initial unit tests, `AGENTS.md`, `pyproject.toml`.
