# Cost axis of the audio-stack crossing

What each crossing run cost, read from SLURM accounting and the training logs (`plan/03-audio-stack.md` §4). First pass,
2026-10-15: the 16 scored cells and the 11 runs discarded on the way. The seed-43 replicates and the QK-norm Conformer cells
join when they finish.

`crossing-2100-cost-2026-10-15.tsv` has one row per run. `final` is the run the cell's row renders to (the one the selection
metric scored); `discarded` is every other crossing submission in `arms.tsv`. `gpu_h` is SLURM elapsed time x allocated GPUs,
so startup, evaluations and checkpoints are in it. `audio_h` is `train_hours/total` of the last metrics line, and
`wall_s_per_audio_h` is wall seconds per hour of audio trained, which does not depend on the GPU count. `racks` are the racks of
the job's nodes. Job ids stay in `arms.tsv`; none are written here.

Regenerate (nothing on MN5 is written):

```bash
python3 crossing_job_list.py <main checkout> > joblist.tsv    # carries job ids: keep it out of the repo
ssh mn5 "bash -s -- $(awk -F'\t' '{printf "%s:%s:%s ", $1, $2, $3}' joblist.tsv)" < collect_cost.sh > cost_raw.tsv
python3 assemble_cost.py joblist.tsv cost_raw.tsv <scores file with the ranking> > table.tsv
```

## What the numbers show (2026-10-15)

- All 16 cells trained the same audio (10,410-10,440 h) and cost 2,194 GPU-h. The 11 discarded runs cost another 883 GPU-h,
  29% of the crossing's 3,077.
- GPU-h across cells does not compare stacks. The node count was set per cell from the planner's estimates (PI, 2026-09-28),
  and the wall time did not shrink with it: Whisper and MMS-1b took 24.2 h on 1 node and 24.0 h on 2; mHuBERT 23.0 h on 1,
  14.1-14.9 h on 2 and 14.1 h on 4; w2v-BERT 28.5 h on 1 and 20.7-26.9 h on 2. In GPU-h the 2- and 4-node cells cost 1.2-2.4x
  the 1-node cell of the same encoder.
- Run time does not look like model compute. On 1 node MLP and MoE took the same wall time on every encoder (Whisper 24.18 h
  and 24.18 h, MMS-1b 24.19 and 24.19, w2v-BERT 28.51 and 28.50) although the MoE trains 7x the adapter parameters, and the
  Whisper and MMS-1b MLP cells agree to 0.04% on different racks. Whisper and MMS-1b sit at 8.3-8.7 s per audio hour on 4 and on
  8 GPUs, mHuBERT at 4.9-5.1 on 8 and 16. Something shared sets the pace; the data path is the first suspect, untested.
- Three runs were 1.7-1.9x slower than their peers at the same topology and config: mms1b-conformer (14.2 s per audio hour
  against 8.4 for the MMS-1b MLP) and, still running, the two seed-43 replicates (15-16 against 8.4-8.7 for their originals).
  mms1b-conformer ran on a rack where an earlier run had a normal pace; both replicates are on a rack (as03r3) that no earlier
  crossing run used. Cause not found.

## Still to do

- A compute-only measure, since the SLURM figure is the cost of the run as it happened: FLOPs per audio second from the configs
  (encoder forward, adapter, decoder), or a timing on synthetic batches.
- Whisper's window-padding ratio per corpus (`plan/03-audio-stack.md` §3), and eval throughput at batch 16 sorted by duration.
- Active parameters of the MoE; the table lists trainable parameters.
