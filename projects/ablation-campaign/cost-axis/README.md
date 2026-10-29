# Cost axis of the audio-stack crossing

What each crossing run cost, read from SLURM accounting and the training logs (`plan/03-audio-stack.md` §4). Second pass,
2026-10-17: the 16 scored cells, the five runs of the 10-14/10-15 batch (the seed-43 replicates of whisper-conformer and
whisper-moe and the QK-norm Conformer cells of whisper, mhubert and w2vb, labelled `-s43` and `-qkn`) and the 11 runs discarded
on the way.

`crossing-2100-cost-2026-10-17.tsv` has one row per run. `final` is the run the cell's row renders to (the one the selection
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

## What the numbers show (2026-10-17)

- All 21 final runs trained the same audio (10,410-10,440 h) and cost 2,906 GPU-h: 2,194 for the 16 cells and 712 for the five
  runs of the 10-14/10-15 batch. The 11 discarded runs cost another 883 GPU-h, 23% of the 3,790 in all.
- GPU-h across cells does not compare stacks. The node count was set per cell from the planner's estimates (PI, 2026-09-28),
  and the wall time did not shrink with it: Whisper and MMS-1b took 24.2 h on 1 node and 24.0 h on 2; mHuBERT 23.0 h on 1,
  14.1-14.9 h on 2 and 14.1 h on 4; w2v-BERT 28.5 h on 1 and 20.7-26.9 h on 2. In GPU-h the 2- and 4-node cells cost 1.2-2.4x
  the 1-node cell of the same encoder.
- Run time does not look like model compute. On 1 node MLP and MoE took the same wall time on every encoder (Whisper 24.18 h
  and 24.18 h, MMS-1b 24.19 and 24.19, w2v-BERT 28.51 and 28.50) although the MoE trains 7x the adapter parameters, and the
  Whisper and MMS-1b MLP cells agree to 0.04% on different racks. Whisper and MMS-1b sit at 8.3-8.7 s per audio hour on 4 and on
  8 GPUs, mHuBERT at 4.9-5.1 on 8 and 16. Something shared sets the pace; the data path is the first suspect, untested.
- Three runs were 1.4-1.7x slower than their peers at the same topology and config: mms1b-conformer (14.2 s per audio hour
  against 8.4 for the MMS-1b MLP) and the two seed-43 replicates (12.2 and 12.6 against 8.7 and 8.4 for their originals; their
  step rate had recovered from 0.85-1.0 to about 1.4 per second part-way through, on the same nodes). The QK-norm Conformers
  of Whisper (10.9 against 8.7 for its original) and mHuBERT (6.0 against 4.9) were 1.2x slower, w2v-BERT's (6.9 against 7.2)
  was not. mms1b-conformer ran on a rack where an earlier run had a normal pace; both replicates are on a rack (as03r3) that no
  earlier crossing run used. Cause not found. The wall time of one config can differ by 1.4-1.5x between two submissions (the
  replicate pairs), which is more than the gap between adapters.

## Still to do

- A compute-only measure, since the SLURM figure is the cost of the run as it happened: FLOPs per audio second from the configs
  (encoder forward, adapter, decoder), or a timing on synthetic batches.
- A dataloader-only benchmark (no model, CPU) at 4, 8 and 16 ranks, to test the data-path suspicion behind the shared pace.
- Whisper's window-padding ratio per corpus (`plan/03-audio-stack.md` §3), and eval throughput at batch 16 sorted by duration.
- Active parameters of the MoE; the table lists trainable parameters.
