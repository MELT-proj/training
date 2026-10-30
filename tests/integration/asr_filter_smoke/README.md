# ASR score band smoke

Runs `data.train_ds.cut_filter` with one `array` rule (`asr_wer`, matching `task: asr`); it predates the generic rules and was run on MN5 under the earlier `asr_filter` key.

One node, acc_debug. A short run whose ASR sources have *different* bands and phases, next to ST sources and an exempt ASR source (`smoke.yaml`):

| source | task | filter |
|---|---|---|
| `voxpopuli/de/train` | asr | default phases: WER [0.1, 0.4], then [0, 0.1] from step 25 |
| `cv22_sidon/de/train` | asr | own phases: WER [0.01, 0.3], then exactly 0 from step 25 |
| `mls_sidon/dutch/train` | asr | static WER [0, 0.05] |
| `voxpopuli/es/train` | asr | `cut_filter: false` |
| `covost2/{nl,it}_en/train` | st | none |

`run_smoke_mn5.sh train` runs 40 steps (`buffer_size` 300 so the old band drains in a few steps); `run_smoke_mn5.sh resume` resumes from `checkpoint-20`, so the switch at step 25 happens after a restore. `check_smoke.py` reads the per-batch cut-id logs (which carry the step the worker saw) and prints, per source and step window, the share of delivered cuts inside each phase's band.
