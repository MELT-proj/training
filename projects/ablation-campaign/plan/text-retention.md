# Text retention after IFT

Decision-rule item 3 of `02-backbones.md` §4: how much of the decoder's
pre-training language knowledge survives IFT. Measured with exactly the
instrument of §3.7 — teacher-forced NLL / bits-per-character of the FLEURS
references under the decoder, raw and conditioned on the task instruction,
on the same frozen sets — applied to each arm's IFT checkpoint instead of the
bare backbone. Lower BPC is better (the decoder assigns more probability to
the reference); a positive delta means the decoder retained less.

The cascade oracle (row 5) is **not** part of this measurement; the exporter
below is what unblocks it later.

## Instrument

- Tool: `melteval text-prior` (melt-eval, PR #14), unchanged.
- Settings per `02-backbones.md` §3.1/§3.3: assistant-masked conditioned
  scoring with `llama3` for the Llama arms, `chatml` for the Qwen arms; the
  first (canonical) instruction of each task's prompt pool; raw scoring is
  the tokenizer's own encoding of the reference with no template.
- Checkpoints are MELT checkpoints, so each one was first exported to a plain
  `AutoModelForCausalLM` directory with `projects/ablation-campaign/text_prior_export.py`
  (new, standalone; `text_decoder.*` kept, `audio_stack.*` dropped).
- Container: `/gpfs/scratch/epor48/melt_eval_cuda126_spbleu.sif`
  (transformers 5.19.0, torch 2.9.1+cu126), with `PYTHONPATH=~/eval-main`
  shadowing the image's baked `melteval`. Qwen3.5 needs transformers 5.x;
  this image has it, so no arm needed the training image.

## Inputs

Frozen sets (identical bytes on artemis and MN5; md5 of the sorted
`lang|task|target` records in `verify/`):

| set | samples | languages | md5 |
|---|---|---|---|
| `fleurs24-asr-dev` | 2,400 | 24 | `07de0fd2c3b494ab55106f870d820e3c` |
| `fleurs24-st-xen-dev` | 2,300 | 1 (`en`; 23 source locales) | `bba11359d9bef6b96bc4c0fb856a56fe` |

MN5 paths: `/gpfs/scratch/epor48/eval-sets-spbleu/fleurs24-{asr,st-xen}-dev`.
The ASR set was staged from artemis scratch on 2026-10-22 (it was missing on
MN5); the ST copy was already there and matches artemis byte for byte.

Checkpoints: `<run>/checkpoint-6310` under
`/gpfs/scratch/epor48/outputs/`, where `<run>` is
`IFT-700backbone-whisperlargeF-<arm>-moeT-sk5-hz10-...-s1337-{4,8}g`
(`llama1bBaseT`, `llama1bInsT`, `qwen35_2bBaseT`, `qwen35_2bInsT`).

## Export verification

Every export was checked before scoring. Per arm: every `text_decoder.*`
tensor of the source exists in the export with the same shape and the same
sha256 of its raw bytes; the export loads with `AutoModelForCausalLM` and one
teacher-forced forward gives finite logits of shape `(1, seq, vocab)`; the
exported tokenizer produces byte-identical token ids to the pre-IFT tokenizer
on six references, and renders the conditioned prompt byte-identically to it
on ten.

| arm | decoder keys compared | audio_stack dropped | class | vocab | tied | tokenizer ids = base |
|---|---|---|---|---|---|---|
| llama1b-base | 146 / 146 | 515 | LlamaForCausalLM | 128,264 | yes | yes |
| llama1b-ins | 146 / 146 | 515 | LlamaForCausalLM | 128,264 | yes | yes |
| qwen35-2b-base | 320 / 320 | 515 | Qwen3_5ForCausalLM | 248,320 | yes | yes |
| qwen35-2b-ins | 320 / 320 | 515 | Qwen3_5ForCausalLM | 248,320 | yes | yes |

Verification artifacts: `/gpfs/scratch/epor48/text-prior/verify/`.

## Control: the pipeline reproduces §3.7

Before any IFT number is read, the bare backbones were re-scored in the same
container on the same frozen sets.

All 24 control numbers (six backbones × two sets × raw/conditioned) agree
with §3.7 to **< 0.001 bits/char** except the two Llama-3.2-1B-Instruct
conditioned cells: ASR cond 1.884 → 1.859, ST cond
1.250 → 1.242. The cause is the Llama-3.2 chat template, which embeds
`Today Date: <strftime_now()>` in the system turn. §3.7 ran on 2026-09-16,
so it rendered `16 Sep 2026`; today renders `22 Oct 2026`. A direct A/B on
300 ASR samples (en/de/mt) with the same model and tokenizer gives
**1.9129** with the September date vs **1.8754** with today's — the whole
gap, and only on the instruction-tuned arm (the base decoder does not
condition on the date). Token counts are identical under both renders.

Consequence: **raw rows are compared against the published §3.7 values**
(no template, reproduced exactly); **conditioned rows are compared against
the same-day control**, not the September number. The published-versus-
control column below makes the size of that correction visible.

| arm | ASR raw §3.7 / control | ASR cond §3.7 / control | ST raw §3.7 / control | ST cond §3.7 / control |
|---|---|---|---|---|
| Llama-3.2-1B (base) | 1.571 / 1.571 | 2.207 / 2.206 | 1.007 / 1.007 | 1.679 / 1.679 |
| Llama-3.2-1B-Instruct | 1.940 / 1.940 | 1.884 / 1.859 | 1.094 / 1.094 | 1.250 / 1.242 |
| Qwen3.5-2B-Base | 1.324 / 1.324 | 1.651 / 1.651 | 0.959 / 0.959 | 1.464 / 1.463 |
| Qwen3.5-2B (instruct) | 1.390 / 1.390 | 1.428 / 1.428 | 0.986 / 0.986 | 1.158 / 1.158 |
| EuroLLM-1.7B (base) | 1.190 / 1.190 | 1.856 / 1.856 | 1.023 / 1.023 | 1.801 / 1.800 |
| EuroLLM-1.7B-Instruct | 1.200 / 1.200 | 1.700 / 1.701 | 1.039 / 1.040 | 1.572 / 1.572 |

## Results

Delta = post-IFT − pre-IFT, bits per character; positive means the decoder
retained less. Raw deltas are against §3.7; conditioned deltas are against
the same-day control (see above). The headline is the conditioned pair,
because it matches the eval condition — with one caveat for the two base
arms: their pre-IFT conditioned number is dominated by template unfamiliarity
(§2's confound), so a negative conditioned delta there means "learned the
chat format", not "kept more of the language". The raw delta is the clean
retention signal for a base arm; for an instruct arm both rows point the
same way.

### Pooled

| arm | ASR raw pre → post | Δ raw | ASR cond control → post | Δ cond | ST raw pre → post | Δ raw | ST cond control → post | Δ cond |
|---|---|---|---|---|---|---|---|---|
| Llama-3.2-1B (base) | 1.571 → 2.242 | +0.671 | 2.206 → 2.148 | -0.058 | 1.007 → 1.218 | +0.211 | 1.679 → 1.363 | -0.315 |
| Llama-3.2-1B-Instruct | 1.940 → 2.719 | +0.779 | 1.859 → 2.536 | +0.677 | 1.094 → 1.287 | +0.194 | 1.242 → 1.366 | +0.123 |
| Qwen3.5-2B-Base | 1.324 → 2.368 | +1.044 | 1.651 → 2.279 | +0.628 | 0.959 → 1.280 | +0.321 | 1.463 → 1.472 | +0.009 |
| Qwen3.5-2B (instruct) | 1.390 → 2.523 | +1.133 | 1.428 → 2.433 | +1.006 | 0.986 → 1.313 | +0.326 | 1.158 → 1.490 | +0.332 |
| EuroLLM-1.7B (base) | pending | pending | pending | pending | pending | pending | pending | pending |
| EuroLLM-1.7B-Instruct | pending | pending | pending | pending | pending | pending | pending | pending |

### The Llama-Instruct correction, made visible

| arm | ASR cond published (§3.7) | ASR cond same-day control | ASR cond post | Δ vs published | ST cond published | ST cond control | ST cond post | Δ vs published |
|---|---|---|---|---|---|---|---|---|
| Llama-3.2-1B (base) | 2.207 | 2.206 | 2.148 | -0.058 | 1.679 | 1.679 | 1.363 | -0.316 |
| Llama-3.2-1B-Instruct | 1.884 | 1.859 | 2.536 | +0.652 | 1.250 | 1.242 | 1.366 | +0.115 |
| Qwen3.5-2B-Base | 1.651 | 1.651 | 2.279 | +0.628 | 1.464 | 1.463 | 1.472 | +0.009 |
| Qwen3.5-2B (instruct) | 1.428 | 1.428 | 2.433 | +1.006 | 1.158 | 1.158 | 1.490 | +0.332 |
| EuroLLM-1.7B (base) | pending | pending | pending | pending | pending | pending | pending | pending |
| EuroLLM-1.7B-Instruct | pending | pending | pending | pending | pending | pending | pending | pending |

### Per language

ST's frozen set carries a single `lang` (`en`), so its per-language block
is the `en` row; the other rows are ASR-only. `pre` for conditioned ASR is
the same-day control; `pre` for raw ASR is §3.7.

### Llama-3.2-1B (base)

| lang | ASR raw pre → post | Δ | ASR cond control → post | Δ | ST raw pre → post | Δ | ST cond control → post | Δ |
|---|---|---|---|---|---|---|---|---|
| bg | 1.538 → 2.449 | +0.911 | 2.166 → 2.374 | +0.208 | — | — | — | — |
| cs | 1.533 → 2.239 | +0.706 | 2.259 → 2.172 | -0.087 | — | — | — | — |
| da | 1.555 → 2.220 | +0.665 | 2.247 → 2.059 | -0.189 | — | — | — | — |
| de | 1.142 → 1.785 | +0.642 | 1.732 → 1.861 | +0.129 | — | — | — | — |
| el | 1.328 → 2.277 | +0.949 | 1.933 → 2.263 | +0.329 | — | — | — | — |
| en | 0.987 → 1.179 | +0.192 | 1.627 → 1.294 | -0.333 | 1.007 → 1.218 | +0.211 | 1.679 → 1.363 | -0.315 |
| es | 1.198 → 1.426 | +0.229 | 1.753 → 1.423 | -0.330 | — | — | — | — |
| et | 2.155 → 3.082 | +0.927 | 2.806 → 2.959 | +0.153 | — | — | — | — |
| fi | 1.626 → 2.287 | +0.661 | 2.230 → 2.154 | -0.076 | — | — | — | — |
| fr | 1.053 → 1.313 | +0.260 | 1.659 → 1.260 | -0.400 | — | — | — | — |
| ga | 2.263 → 2.804 | +0.541 | 2.989 → 2.778 | -0.211 | — | — | — | — |
| hr | 1.696 → 2.410 | +0.713 | 2.364 → 2.250 | -0.114 | — | — | — | — |
| hu | 1.503 → 2.310 | +0.807 | 2.144 → 2.164 | +0.020 | — | — | — | — |
| it | 1.177 → 1.484 | +0.307 | 1.757 → 1.373 | -0.384 | — | — | — | — |
| lt | 2.146 → 2.996 | +0.850 | 2.790 → 2.853 | +0.063 | — | — | — | — |
| lv | 2.224 → 3.154 | +0.930 | 2.919 → 3.041 | +0.122 | — | — | — | — |
| mt | 2.392 → 3.536 | +1.144 | 3.013 → 3.338 | +0.325 | — | — | — | — |
| nl | 1.274 → 2.052 | +0.778 | 1.916 → 1.926 | +0.010 | — | — | — | — |
| pl | 1.370 → 2.080 | +0.710 | 2.031 → 1.982 | -0.049 | — | — | — | — |
| pt | 1.256 → 1.692 | +0.436 | 1.842 → 1.597 | -0.245 | — | — | — | — |
| ro | 1.373 → 1.982 | +0.609 | 1.960 → 1.806 | -0.153 | — | — | — | — |
| sk | 1.880 → 2.739 | +0.859 | 2.515 → 2.626 | +0.111 | — | — | — | — |
| sl | 1.876 → 2.623 | +0.747 | 2.556 → 2.387 | -0.169 | — | — | — | — |
| sv | 1.449 → 2.079 | +0.630 | 2.088 → 1.962 | -0.126 | — | — | — | — |

### Llama-3.2-1B-Instruct

| lang | ASR raw pre → post | Δ | ASR cond control → post | Δ | ST raw pre → post | Δ | ST cond control → post | Δ |
|---|---|---|---|---|---|---|---|---|
| bg | 2.310 → 3.687 | +1.377 | 2.112 → 3.559 | +1.447 | — | — | — | — |
| cs | 2.096 → 3.066 | +0.970 | 1.927 → 2.793 | +0.866 | — | — | — | — |
| da | 1.906 → 2.582 | +0.677 | 1.759 → 2.324 | +0.565 | — | — | — | — |
| de | 1.251 → 1.881 | +0.629 | 1.307 → 1.940 | +0.633 | — | — | — | — |
| el | 1.857 → 4.025 | +2.169 | 1.851 → 3.728 | +1.877 | — | — | — | — |
| en | 1.072 → 1.245 | +0.173 | 1.250 → 1.358 | +0.108 | 1.094 → 1.287 | +0.194 | 1.242 → 1.366 | +0.123 |
| es | 1.284 → 1.538 | +0.255 | 1.324 → 1.502 | +0.178 | — | — | — | — |
| et | 2.756 → 3.636 | +0.880 | 2.431 → 3.302 | +0.871 | — | — | — | — |
| fi | 1.853 → 2.551 | +0.698 | 1.725 → 2.367 | +0.641 | — | — | — | — |
| fr | 1.157 → 1.436 | +0.279 | 1.156 → 1.345 | +0.189 | — | — | — | — |
| ga | 2.753 → 3.296 | +0.544 | 2.807 → 3.054 | +0.247 | — | — | — | — |
| hr | 2.156 → 2.874 | +0.718 | 2.022 → 2.667 | +0.645 | — | — | — | — |
| hu | 1.815 → 2.730 | +0.915 | 1.718 → 2.463 | +0.746 | — | — | — | — |
| it | 1.333 → 1.628 | +0.295 | 1.325 → 1.520 | +0.195 | — | — | — | — |
| lt | 2.746 → 3.608 | +0.863 | 2.552 → 3.364 | +0.811 | — | — | — | — |
| lv | 2.847 → 3.643 | +0.796 | 2.785 → 3.462 | +0.677 | — | — | — | — |
| mt | 3.066 → 4.158 | +1.092 | 3.029 → 3.870 | +0.840 | — | — | — | — |
| nl | 1.484 → 2.251 | +0.767 | 1.437 → 2.057 | +0.620 | — | — | — | — |
| pl | 1.668 → 2.622 | +0.953 | 1.574 → 2.368 | +0.795 | — | — | — | — |
| pt | 1.354 → 1.830 | +0.476 | 1.361 → 1.683 | +0.322 | — | — | — | — |
| ro | 1.588 → 2.312 | +0.724 | 1.439 → 2.011 | +0.571 | — | — | — | — |
| sk | 2.511 → 3.437 | +0.926 | 2.280 → 3.311 | +1.031 | — | — | — | — |
| sl | 2.422 → 3.164 | +0.743 | 2.193 → 2.942 | +0.749 | — | — | — | — |
| sv | 1.693 → 2.381 | +0.688 | 1.579 → 2.155 | +0.576 | — | — | — | — |

### Qwen3.5-2B-Base

| lang | ASR raw pre → post | Δ | ASR cond control → post | Δ | ST raw pre → post | Δ | ST cond control → post | Δ |
|---|---|---|---|---|---|---|---|---|
| bg | 1.282 → 2.837 | +1.556 | 1.557 → 2.560 | +1.003 | — | — | — | — |
| cs | 1.372 → 2.653 | +1.281 | 1.774 → 2.582 | +0.807 | — | — | — | — |
| da | 1.335 → 2.265 | +0.930 | 1.698 → 2.145 | +0.448 | — | — | — | — |
| de | 0.988 → 1.694 | +0.706 | 1.353 → 1.860 | +0.507 | — | — | — | — |
| el | 1.097 → 3.087 | +1.989 | 1.387 → 2.458 | +1.070 | — | — | — | — |
| en | 0.948 → 1.219 | +0.272 | 1.424 → 1.406 | -0.018 | 0.959 → 1.280 | +0.321 | 1.463 → 1.472 | +0.009 |
| es | 1.042 → 1.348 | +0.306 | 1.407 → 1.434 | +0.027 | — | — | — | — |
| et | 1.666 → 3.072 | +1.406 | 1.955 → 2.793 | +0.838 | — | — | — | — |
| fi | 1.406 → 2.671 | +1.265 | 1.736 → 2.602 | +0.866 | — | — | — | — |
| fr | 0.894 → 1.198 | +0.305 | 1.257 → 1.293 | +0.036 | — | — | — | — |
| ga | 2.104 → 2.962 | +0.858 | 2.354 → 2.822 | +0.468 | — | — | — | — |
| hr | 1.383 → 2.458 | +1.076 | 1.698 → 2.423 | +0.724 | — | — | — | — |
| hu | 1.400 → 2.588 | +1.188 | 1.695 → 2.475 | +0.780 | — | — | — | — |
| it | 0.991 → 1.367 | +0.375 | 1.356 → 1.415 | +0.059 | — | — | — | — |
| lt | 1.535 → 2.782 | +1.247 | 1.849 → 2.690 | +0.842 | — | — | — | — |
| lv | 1.632 → 3.015 | +1.383 | 1.923 → 2.953 | +1.030 | — | — | — | — |
| mt | 1.971 → 4.071 | +2.101 | 2.126 → 3.719 | +1.594 | — | — | — | — |
| nl | 1.152 → 2.027 | +0.875 | 1.487 → 2.017 | +0.530 | — | — | — | — |
| pl | 1.178 → 2.323 | +1.145 | 1.553 → 2.339 | +0.785 | — | — | — | — |
| pt | 1.079 → 1.657 | +0.578 | 1.472 → 1.700 | +0.228 | — | — | — | — |
| ro | 1.203 → 2.151 | +0.948 | 1.473 → 2.063 | +0.589 | — | — | — | — |
| sk | 1.443 → 2.739 | +1.295 | 1.784 → 2.514 | +0.730 | — | — | — | — |
| sl | 1.468 → 2.643 | +1.176 | 1.784 → 2.519 | +0.735 | — | — | — | — |
| sv | 1.380 → 2.261 | +0.881 | 1.725 → 2.209 | +0.484 | — | — | — | — |

### Qwen3.5-2B (instruct)

| lang | ASR raw pre → post | Δ | ASR cond control → post | Δ | ST raw pre → post | Δ | ST cond control → post | Δ |
|---|---|---|---|---|---|---|---|---|
| bg | 1.344 → 3.212 | +1.868 | 1.372 → 2.813 | +1.440 | — | — | — | — |
| cs | 1.440 → 2.829 | +1.389 | 1.478 → 2.739 | +1.260 | — | — | — | — |
| da | 1.404 → 2.421 | +1.017 | 1.450 → 2.316 | +0.866 | — | — | — | — |
| de | 1.027 → 1.740 | +0.712 | 1.118 → 1.916 | +0.798 | — | — | — | — |
| el | 1.146 → 3.333 | +2.187 | 1.236 → 2.696 | +1.460 | — | — | — | — |
| en | 0.974 → 1.248 | +0.275 | 1.141 → 1.453 | +0.312 | 0.986 → 1.313 | +0.326 | 1.158 → 1.490 | +0.332 |
| es | 1.075 → 1.395 | +0.320 | 1.167 → 1.491 | +0.323 | — | — | — | — |
| et | 1.788 → 3.199 | +1.412 | 1.703 → 2.965 | +1.261 | — | — | — | — |
| fi | 1.480 → 2.832 | +1.352 | 1.496 → 2.811 | +1.315 | — | — | — | — |
| fr | 0.926 → 1.251 | +0.326 | 1.011 → 1.375 | +0.364 | — | — | — | — |
| ga | 2.214 → 3.086 | +0.872 | 2.264 → 2.957 | +0.694 | — | — | — | — |
| hr | 1.434 → 2.616 | +1.182 | 1.475 → 2.550 | +1.074 | — | — | — | — |
| hu | 1.466 → 2.835 | +1.370 | 1.472 → 2.669 | +1.196 | — | — | — | — |
| it | 1.039 → 1.419 | +0.380 | 1.120 → 1.470 | +0.350 | — | — | — | — |
| lt | 1.600 → 3.000 | +1.400 | 1.597 → 2.853 | +1.256 | — | — | — | — |
| lv | 1.741 → 3.230 | +1.490 | 1.700 → 3.138 | +1.438 | — | — | — | — |
| mt | 2.079 → 4.139 | +2.060 | 2.002 → 4.000 | +1.999 | — | — | — | — |
| nl | 1.216 → 2.195 | +0.979 | 1.272 → 2.151 | +0.880 | — | — | — | — |
| pl | 1.247 → 2.565 | +1.319 | 1.318 → 2.568 | +1.250 | — | — | — | — |
| pt | 1.126 → 1.773 | +0.647 | 1.207 → 1.812 | +0.605 | — | — | — | — |
| ro | 1.262 → 2.328 | +1.066 | 1.271 → 2.228 | +0.957 | — | — | — | — |
| sk | 1.522 → 2.934 | +1.412 | 1.524 → 2.717 | +1.192 | — | — | — | — |
| sl | 1.537 → 2.796 | +1.259 | 1.538 → 2.655 | +1.117 | — | — | — | — |
| sv | 1.453 → 2.432 | +0.979 | 1.482 → 2.357 | +0.875 | — | — | — | — |

## Notes

- **EuroLLM is pending.** Its two IFT runs are queued (`47783768`,
  `47783769`, 36 h wall); at submission time only `checkpoint-3608` exists,
  against the 6,310-step step-2 checkpoint the other four arms use. It is
  measured with the same procedure once the runs reach `checkpoint-6310`.
- The ST frozen set's only `lang` is `en` (the target), so its per-language
  breakdown is a single cell; this is how §3.7 scored it too, and the
  per-language deltas that matter for ST are not available from this set.
- `ga`'s target text has no `pnc_text` (§3.5); its BPC is not on the same
  textual basis as the rest, as in §3.7.
- The exported tokenizer keeps MELT's padded vocabulary (Llama 128,264 rows
  for a 128,259-token tokenizer; Qwen 248,320 for 248,081). No reference
  token reaches the padding, and ids were checked equal to the base
  tokenizer, so BPC stays comparable.

## Reproducing

```bash
# export + verify (CPU is enough; the container is only needed for transformers)
python text_prior_export.py RUN/checkpoint-6310 OUT_DIR

# measurement, one call per (arm, set)
python -m melteval.cli text-prior FROZEN_DIR --model OUT_DIR \
  --chat-template-config {llama3,chatml} --device cuda -o RESULT.json
```

## Jobs

All on MN5, account `epor48`, QOS `acc_debug`, one whole `acc` node.

| job | what | outcome |
|---|---|---|
| 47790852 | export + verification, four arms | COMPLETED, 2:15 |
| 47790938 | Llama controls, first attempt | FAILED at start: `singularity exec` ran without `--nv`, so torch saw no GPU (environment, not code) |
| 47790967 | Llama controls (llama1b-base/-ins, both sets) | COMPLETED, rc=0 |
| 47791038 | date A/B on Llama-Instruct (300 ASR samples) | COMPLETED |
| 47791474 | Qwen and EuroLLM controls, both sets | COMPLETED, rc=0 |
| 47791831 | Llama IFT measurements, both sets | COMPLETED, rc=0 |
| 47792179 | Qwen IFT measurements, both sets | COMPLETED, rc=0 |

Logs: `/gpfs/scratch/epor48/text-prior/logs/`. Per-arm results:
`/gpfs/scratch/epor48/text-prior/results/`; exports under
`/gpfs/scratch/epor48/text-prior/exports/`; verification JSONs under
`/gpfs/scratch/epor48/text-prior/verify/`.
