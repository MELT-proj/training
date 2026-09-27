# Leftovers

Disposable artifacts a session created and left behind on purpose (a debug
run's output directory, a scratch backup, a probe checkpoint) that the PI can
delete once they no longer matter. Append-only, newest first; the PI removes
an entry (and the files it names) when it's actually deleted. See
`plan/agent-protocol.md` §0 for the rule that puts entries here.

This is not for anything already covered by an existing cleanup path: a
finished campaign run's checkpoints are handled by the off-boarding plan
(`plan/06-fondue.md` §7), not by this file.

Entry template:

```
## YYYY-MM-DD — <who> — <one-line reason>
- <host>:<path> (<rough size if known>) — <what it is, why it's safe to delete>
```

---

## 2026-09-25/27 — Claude (worker, fondue-dry-run) — Fondue MA dry-run outputs, superseded checkpoints and scratch staging

Context: timeline week 3 Track B, Fondue dry run (config resolution, throughput probes at three
`batch_duration` points). Every item below is disposable: none is a campaign arm, none is referenced
by `arms.tsv` as a kept result, and none is needed once the numbers are read into `06-fondue.md` §2 and
the board (already done for the first two probes as of 2026-09-26).

- mn5:/gpfs/scratch/epor48/outputs/MA-FondueMAdryrun-whisperlargeF-qwen35_2bInsF-mlpT-elr6e6-dlr2e5-lr1e3-s42-32g (9.7 GB) — 600-step probe at `batch_duration 30 x accum 4`, job 46618634. Numbers already recorded.
- mn5:/gpfs/scratch/epor48/outputs/MA-FondueMAdryrun-whisperlargeF-qwen35_2bInsF-mlpT-elr6e6-dlr2e5-lr1e3-s42-32g.attempt1-no-total-cuts (65 KB) — first submission, died at startup (`total_cuts` missing from the render).
- mn5:/gpfs/scratch/epor48/outputs/MA-FondueMAdryrun-whisperlargeF-qwen35_2bInsF-mlpT-elr6e6-dlr2e5-lr1e3-s42-32g.attempt2-people-speech (65 KB) — second submission, died at the first batch (People's Speech unreadable through lhotse's indexed reader).
- mn5:/gpfs/scratch/epor48/outputs/MA-FondueMAdryrun-whisperlargeF-qwen35_2bInsF-mlpT-bd120-ga1-elr6e6-dlr2e5-lr1e3-s42-32g (9.7 GB) — 300-step probe at `batch_duration 120 x accum 1`, job 46649405. Numbers already recorded.
- mn5:/gpfs/scratch/epor48/outputs/MA-FondueMAdryrun-whisperlargeF-qwen35_2bInsF-mlpT-bd200-ga1-elr6e6-dlr2e5-lr1e3-s42-32g (small — OOM'd at step 70) — `batch_duration 200 x accum 1` probe, job 46667595, failed on real GPU OOM (not just the warmup-pass warning). Numbers recorded once written up.
- mn5:~/training-fondue-dryrun — dedicated repo checkout used to keep this dry run off the other in-flight session's MN5 checkout (`infrastructure.md` §3 reconciliation). Every ledger row from it has been copied into `arms.tsv` on `main` and verified byte-identical; the checkout itself has no further use once the dry-run phase is closed.
- nyx:/mnt/scratch-nyx/giuseppe/melt/pnc_backfill/backup_pre_ca_pnc/ — pre-truecase backup of `fleurs/ca_es/train`'s manifest and idx files, taken before a merge that was ultimately never run (option 1 was chosen instead: ca FLEURS stays on its original text). Safe to delete once the PI confirms ca stays raw for good.
- nyx:/mnt/scratch-nyx/giuseppe/melt/pnc_backfill/out_fleurs_ca/ — the staged Qwen sidecar for the same un-run merge.
- nyx:/mnt/scratch-nyx/giuseppe/melt/pnc_backfill/fleurs_ca_leaf.txt — the one-line leaf list for the same un-run merge.
- artemis:/mnt/scratch-artemis/giuseppe/pnc_out/fleurs/{ga_ie,ca_es}/train/ — the two truecase sidecar outputs (`cuts.*.pnc.jsonl`) from jobs 333476/333477. Kept as the source for the edit-rate numbers in the PNC trust rule (`06-fondue.md` §3); safe to delete once that rule is considered settled and no one needs to re-derive or spot-check the numbers.
- artemis:/mnt/scratch-artemis/giuseppe/pnc_prompts_ca_ga/ — the ca/ga prompt YAMLs (not reviewed by a native speaker) plus a copy of the other 25 prompts, used only to run the two truecase jobs above. Delete alongside them.
