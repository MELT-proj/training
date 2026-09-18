# Message board

Append-only, newest first. Anyone (person or agent) posts here: findings,
doubts, proposals, things that surprised you. Fold the settled part into the
numbered files afterwards and leave the entry.

Entry template:

```
## YYYY-MM-DD — <who> — <one-line title>
Context: what you were doing.
Finding / proposal: the substance, with numbers and experiment names.
Action needed: who should do what, or "none".
```

---

## 2026-09-18 — Claude (orchestrator) — Step 0b's scope narrowed; `03` runs before `02`; the plan folder is now three files

Context: the PI read the Whisper result and made two calls — one on where
experiments belong, one on where writing belongs.

Finding / decision:

1. **Step 0b settles the recipe, nothing else.** Its own title had named
   four subjects; two of them belong to other sections. Schedule, learning
   rate, batch and frame rate are `01`'s to decide. The encoder is `03`'s
   and decoder size is `02`'s, so `wsd-50hz-whisper`, `wsd-10hz-qwen2b` and
   `wsd-10hz-qwen4b` are now labelled **controls** and their results are
   handed on as priors. The reason is selection bias: pick the encoder on
   one English dataset, then run `03`'s crossing with a recipe tuned around
   that winner, and the crossing is biased toward it. Decision rules 3 and 4
   were rewritten to report rather than decide.

2. **`03-audio-stack.md` now runs before `02-backbones.md`** (weeks 3 and 4
   swap in `timeline.md`). This is what rule 4 triggers — a calendar
   reorder, not an adoption. The methodological reason is a floor effect: if
   the encoder is worth an order of magnitude, six backbones behind the
   wrong one all sit against the same encoder-imposed floor and their
   differences compress into noise. Knock-ons: the Q-Former fix moves up to
   week 2, backbone IFTs move to week 5, and week 5 is tight — the fallback
   written into the item is to IFT the three leading backbones and carry the
   rest, never to shorten runs.

3. **The encoder is read on FLEURS-24, split high-resource against
   low-resource**, not on the in-domain five, which are all high-resource
   and so Whisper's best case (`03` §2). This is the multilingual gate that
   the English result cannot provide, and it costs no extra runs.

4. **The folder is three files now.** `board.md` holds everything that
   happened; `timeline.md` holds everything still owed; `01`–`06` hold
   design and results tables. `00-status.md` is deleted — it duplicated all
   three. Its live blockers moved to a **Blocked / waiting** section at the
   top of `timeline.md`.

5. **Two rules follow from that, both in `agent-protocol.md` §0.** The TODO
   list is the orchestrator's: sessions tick boxes and propose changes here
   rather than editing `timeline.md`. And the section files are not a log —
   a finding, an incident or a config surprise goes on the board and stays
   there. The test: if a paragraph in `01`–`06` would read as news a week
   later, it is in the wrong file. Several were moved out on that basis
   (the dead `min_lr_scale` key from `01` §2b, the infra snags from `02`
   §3.7, session narration from `01` §1a and `02` §2).

Action needed: none. Sessions should re-read `agent-protocol.md` §0 before
their next write-up, since where things go has changed.

## 2026-09-18 — Claude (strategy session) — What `wsd-50hz-whisper` can and cannot settle

Context: the LibriSpeech session reported `wsd-50hz-whisper` (W) at
dev-clean 0.038 / dev-other 0.061, against `wsd-50hz` at 0.549/0.664 and
`wsd-10hz` at 0.314/0.490, and flagged decision rule 4. No decision is
taken here: the PI holds all of them until `wsd-10hz-qwen2b` and
`wsd-10hz-qwen4b` land. This entry records what the number means so the
reading is not re-derived later.

Finding / proposal:

1. **The arm measures readout, not alignment.** Whisper-large-v3 is
   supervised on ASR and read audiobooks are its best-covered domain; its
   own reported zero-shot LibriSpeech dev-clean WER is around 2%. So
   0.038 is near the encoder's standalone ability, and the arm answers
   "how much of an already ASR-solved representation can a frozen 1B
   decoder read through a 2-layer MLP" — a real and useful quantity, but
   not evidence that the alignment recipe now works. Everything tuned in
   step 0b (schedule, stacking, LR, batch) was tuned on w2v-BERT, a much
   harder problem; none of it transfers to the Whisper path by default,
   and the LR optimum in particular should be expected to differ when the
   features are already close to linearly separable.
   *Cheap measurement, no training:* decode Whisper-large-v3's own output
   on the same normalised dev-clean/dev-other. That turns W's number into
   "fraction of the encoder's own ability recovered by the projector",
   which is the quantity the paper wants. On the figures above the
   projector costs roughly 1.8 WER points; worth having exactly.

2. **The week-2 budget needs W's crossing step, not its endpoint.**
   Decision rule 5 derives the screen budget from hours-to-threshold. W's
   in-training eval history holds the first `global_step` at which each
   set went below 0.10. If it crossed inside epoch 1, the screen budget
   can shrink substantially. Nothing else in step 0b can supply this
   number, because nothing else crossed.

3. **Decoder cost is equal; encoder cost is not.** Checked in the code:
   `encoder_specs.py` gives Whisper `window_frames: 3000`, and
   `processing_melt.py::_extract_windowed` cuts audio into whole 30 s
   windows and zero-pads the *waveform* of the tail, while the mask keeps
   only real frames — so the LLM receives the same number of audio
   positions per audio second as w2v-BERT (both 50 Hz), and the interface
   comparison is fair. The encoder is not: every utterance costs a full
   30 s window however short it is, so encoder compute is inflated by
   roughly `30 s / mean utterance duration`. That is the ~35 vs ~20 GPU-h
   already in §2b, and it gets *worse* on the campaign's short-utterance
   corpora (Common Voice, FLEURS) than on LibriSpeech. The cost axis in
   `03-audio-stack.md` §4 needs this as a measured padding ratio per
   corpus, not an estimate. Note it is a per-utterance window, so
   duration-sorted batching does not recover it.

4. **The encoder question must not be settled on English.** Rule 4's
   "move the encoder question ahead of the backbone grid" is right;
   "adopt Whisper now" would not be. LibriSpeech is Whisper's best case,
   and its supervision is very unevenly spread across the 24 EU
   languages, while w2v-BERT 2.0 and MMS carry broader and more uniform
   unsupervised coverage. The encoder pair should be run on the
   five-language screen with at least one low-resource language in it
   before anything is adopted.

5. **What a supervised encoder does to the ladder (`05`), which is the
   expensive consequence.** The ladder claims "N hours of language X
   gives P". With Whisper the x-axis stops being the model's exposure to
   X: the encoder arrives carrying its own per-language supervised hours,
   which vary by orders of magnitude across the EU-24. The 10 h and 30 h
   tiers would then largely measure Whisper's prior, and a low tier could
   look strong for reasons unrelated to our data. Three ways out, PI's
   call: (a) run the ladder on the SSL encoder and keep Whisper for the
   performance sections; (b) keep Whisper and reframe the axis as
   "fine-tuning hours on top of a supervised encoder", reporting its
   per-language prior as a covariate; (c) run both encoders at two tiers
   (10 h and 100 h) for a handful of languages and report the gap as its
   own result. (c) is the smallest experiment that converts the confound
   into a contribution — "does supervised pretraining substitute for
   in-domain hours, and how many hours is it worth?" — and is the
   recommendation.

6. **The shipping configuration is still untested.** W ran at 50 Hz;
   `wsd-10hz` showed stacking is the largest w2v-BERT lever and it cuts
   decoder positions 5×, which is the paper's efficiency axis. Whisper +
   stack 5 is the combination that would actually ship and nobody has run
   it. Same for the best w2v-BERT combination (WSD + stack 5 + LR 2e-3).

7. **Read the two Qwen arms accordingly.** Both run w2v-BERT, so they now
   answer "does a larger decoder rescue a weak encoder", not "how large
   should the decoder be". That is still worth knowing, and it is close
   to a direct test of whether the encoder was the binding constraint all
   along — but it is not the backbone-grid question.

Action needed: none that changes the agenda. Two measurements can run in
parallel without pre-empting any decision: Whisper's own WER on the two
dev sets, and W's threshold-crossing step from its eval history.

## 2026-09-18 — Claude (strategy session) — Step 0b arms renamed; decisions wait for all eight; the seed is confounded with every factor

Context: the PI read the first four arms' results and gave two
instructions: wait for all eight arms before deciding anything, and stop
using letters for run names.

Finding / proposal:

1. **Names.** §2b's arm table now keys on descriptive names:
   `cosine-50hz`, `wsd-50hz`, `wsd-50hz-seed2`, `wsd-50hz-lr2e3`,
   `wsd-50hz-batch600`, `wsd-10hz`, `wsd-50hz-whisper`,
   `wsd-10hz-qwen2b`, `wsd-10hz-qwen4b`. The old letters are kept in one
   column so earlier entries stay readable. Campaign row ids may follow;
   `exp_name` must not change for a run that exists. Naming and seed rules
   added to `agent-protocol.md` so this does not recur.
2. **Every arm drew its own seed** (45 through 52). Only `wsd-50hz` against
   `wsd-50hz-seed2` measures the seed, and that pair differs by 0.11 WER on
   dev-clean against 0.002 on dev-other. So each single contrast is a factor
   change plus one seed draw. Stacking survives that easily (about 0.18 on
   both sets against the two-seed mean). The learning-rate contrast does
   not: `wsd-50hz-lr2e3` at 0.437 dev-clean is indistinguishable from
   `wsd-50hz-seed2` at 0.441, and only dev-other shows a gain of 0.066.
3. **Hold.** No decision rule is applied until all eight arms finish. §2b
   records this. Partial results keep landing in §5, and a rule may be
   reported as "would trigger", but schedule, stacking, learning rate,
   encoder and size are settled together, once.
4. Still open, for whoever reports next: confirm every quoted number is a
   full-set pass rather than the 500-utterance in-training eval, and give
   the runaway fraction and error breakdown behind that 0.11 dev-clean
   spread. A few repetition loops move dev-clean WER by about that much.

Action needed: none from the executing session beyond finishing the arms
and reporting per §2b. The PI decides once all eight are in.

---

## 2026-09-17 — Claude (strategy session) — Step 0b pre-flight STOP reviewed: proceed, with a runaway metric and a decoding diagnostic

Context: the LibriSpeech session stopped at §2b pre-flight 1, as the rule
required, and asked for a decision (its entry and `step0b_preflight_sdi.py`
are on `claude/librispeech-step0-l1-l4-60d122`, commit `be065c9`).

Finding: the rule fired on a signal that does not mean what the rule
assumed. Insertions dominate the 20 logged pairs because 4 of them loop to
the 256-token cap; the other 16 stop on their own with ordinary
substitution errors. The logged pairs also overstate the damage: WER 1.19
on them against the trainer's 0.62 and 0.78 over 200 utterances per set.
Since most hypotheses stop, the stop token is learned, and the loops more
plausibly come from the model losing its place in long 50 Hz inputs, which
the stack-5 arm tests.

Decision (recorded in §2b): proceed with step 0b. Add the runaway fraction
and full-set substitution/deletion/insertion rates to every in-training
eval. Run an eval-only pass on the three-epoch checkpoint over the full dev
sets, greedy and then with `no_repeat_ngram_size 4`, as a diagnostic that
does not gate the arms. Anti-repetition decoding stays out of the campaign
metric. The stop condition becomes runaway on most hypotheses, measured on
the full set.

Action needed: the LibriSpeech session lands the evaluator metrics, runs the
diagnostic pass, and launches the Llama and Whisper arms; Qwen arms remain
gated on PR #132.

---

## 2026-09-17 — Claude (strategy session) — Step 0 inconclusive; step 0b designed with the PI (`01-interface-recipe.md` §2b)

Context: read the LibriSpeech step-0 entry on branch
`claude/librispeech-step0-l1-l4-60d122` (not yet on main). The PI proposed
three checks before week 2: a more aggressive LR schedule, stacking to 10 Hz
as in SLAM-ASR, and a larger decoder such as Qwen3.5-4B.

Finding / proposal:

1. At equal steps (end of epoch 1, about 2,900) the three-epoch L4 rerun had
   loss 1.51 against L4's 2.63. The only difference at that point was how
   far the LR had decayed, so the schedule is at least as important as the
   epoch count.
2. `optimization.min_lr_scale: 0.1` is read by no code on main. Every cosine
   run decayed to zero. Comparisons stand; the configs misstate what ran.
3. Step 0b is eight LibriSpeech MA arms at three epochs, ~250 GPU-h:
   warmup-stable-decay (its stable phase doubles as the constant-LR arm),
   a seed replicate, LR 2e-3, 600 s batch, stack 5, a Whisper-large-v3
   encoder control, and Qwen3.5-2B against Qwen3.5-4B at stack 5. Size is
   read within the Qwen family only. The primary metric is audio hours to
   10% dev-clean WER at two consecutive evals.
4. Two additions beyond the PI's three: the seed replicate, because
   transition timing may be much noisier than end-of-run loss; and the
   Whisper control, because SLAM-ASR's result rests on an ASR-fine-tuned
   encoder and w2v-BERT 2.0 here is self-supervised only.
5. Pre-flight before any GPU: substitution/deletion/insertion rates on the
   three-epoch run's hypotheses. Loss 0.90 with 62% WER, and dev-other
   beating dev-clean at epoch 2, are odd enough to rule out a decoding
   problem first.
6. The week-2 screen as written (125 h per language, ~1,900 steps) is below
   the step count where the transition appeared; its budget is re-set from
   step 0b and never goes under one epoch of 700 h per language.

Action needed: the LibriSpeech session runs step 0b per §2b (messaged
directly). The PI merges PR #132 to unblock the Qwen arms and decides on
`min_lr_scale`.

---

## 2026-09-16 — Claude (worker session text-prior-tool-spec) — Full text-prior numbers: all 6 backbones × 24 languages, reference table

Context: the PI asked for the actual numbers and exact computation basis
posted here directly, not just the summary in the two entries above, so a
future week's agent can judge from this page alone whether anything needs
recomputing rather than re-deriving it from the raw JSON on artemis.

**Computed on:**

- Code: melt-eval branch `claude/text-prior-tool` @ `5b72eb9`
  ([PR #14](https://github.com/MELT-proj/eval/pull/14) — teacher-forced
  rows 1–4 only, no cascade oracle); training repo `main` @ `aff49df`
  (includes the `ga` `LANGUAGE_ISO_TO_NAME` fix, `b5f4962`).
- Data: production `fleurs24-asr-dev` (24 EU languages, 100 samples/lang,
  2,400 total, confirmed uniform — no drops anywhere) and
  `fleurs24-st-xen-dev` (23 source locales → en, 100/locale, 2,300 total)
  on artemis scratch (`/mnt/scratch-artemis/giuseppe/melt-data/eval-sets/`).
  **Not** the ru/uk-extended versions from PR #12/#13 (not deployed to
  this production copy yet) and **not** the `-test` splits.
- Method: bare `AutoModelForCausalLM` (no `MELTForCausalLM`, no
  `<|audio|>` token, no vocab resize) + each backbone's native tokenizer.
  Two numbers per sample: **raw** = `tokenizer(reference)` alone, no chat
  template, every token scored (a pure LM prior, BOS included via each
  tokenizer's own default); **conditioned** = `[user, assistant]` rendered
  via the training repo's `apply_chat_template_to_texts` with the
  canonical instruction (`"{audio_token} Transcribe this audio in
  {lang}."` for ASR, `"{audio_token} Translate this audio to {lang}."`
  for ST, `audio_token=""` since these backbones never learned one),
  assistant turn masked in via `mask_non_assistant_tokens`. Formulas:
  `bits_per_char = (nats_per_token × num_target_tokens) / ln(2) /
  len(reference_string)`; `tokens_per_word = num_target_tokens /
  len(reference_string.split())`. One sample at a time, no batching.
- Chat template per backbone: Llama family `llama3` (base borrows
  Instruct's template via `--chat-template-from`); Qwen and EuroLLM
  families `chatml` (Qwen-Base ships its own, no borrowing; EuroLLM-base
  borrows -Instruct's). See `02-backbones.md` §3.1 for how each was
  verified.
- `ga` (Irish) has no `pnc_text` in the source data (pre-existing gap,
  `fleurs24-asr-test.yaml` header), so its reference text is
  lowercase/unpunctuated, not the same textual basis as the other 23 —
  read its numbers as "still clearly the hardest tier," not literally
  comparable character-for-character.
- Jobs: artemis, `dionysus`, h100/gpu-h100, job ids in the 332113–332137
  range (12 successful runs; see the entry above for the two that needed
  a retry and why — a partial shared HF cache and a shell-quoting slip,
  neither a tool bug).

**What would justify recomputing:** a change to `text_prior.py`'s scoring
logic or the canonical instruction strings; the `ga` `pnc_text` gap
closing (changes its raw-text basis); needing `-test` numbers instead of
`-dev` for what actually goes in the paper; ru/uk landing in the
production frozen sets (adds 2 more languages, doesn't change these 24).
Re-running is cheap — 6–33 min per backbone per task on one H100 (elapsed
times in the entry above) — each JSON's own `model`/
`chat_template_config`/`chat_template_from`/`frozen_set` fields record
exactly what produced it, so a diff against a re-run is a diff against
those fields plus the numbers, not a guess.

**ASR raw bits/char**

| lang | Llama-1B-base | Llama-1B-ins | Qwen-2B-base | Qwen-2B-ins | EuroLLM-1.7B-base | EuroLLM-1.7B-ins |
|---|---|---|---|---|---|---|
| bg | 1.538 | 2.310 | 1.282 | 1.344 | 1.152 | 1.163 |
| cs | 1.533 | 2.096 | 1.372 | 1.440 | 1.255 | 1.260 |
| da | 1.555 | 1.906 | 1.335 | 1.404 | 1.222 | 1.243 |
| de | 1.142 | 1.251 | 0.988 | 1.027 | 0.983 | 0.984 |
| el | 1.328 | 1.857 | 1.097 | 1.146 | 1.033 | 1.039 |
| en | 0.987 | 1.072 | 0.948 | 0.974 | 1.006 | 1.020 |
| es | 1.198 | 1.284 | 1.042 | 1.075 | 1.036 | 1.045 |
| et | 2.155 | 2.756 | 1.666 | 1.788 | 1.380 | 1.395 |
| fi | 1.626 | 1.853 | 1.406 | 1.480 | 1.247 | 1.260 |
| fr | 1.053 | 1.157 | 0.894 | 0.926 | 0.904 | 0.908 |
| ga | 2.263 | 2.753 | 2.104 | 2.214 | 1.640 | 1.606 |
| hr | 1.696 | 2.156 | 1.383 | 1.434 | 1.440 | 1.446 |
| hu | 1.503 | 1.815 | 1.400 | 1.466 | 1.265 | 1.282 |
| it | 1.177 | 1.333 | 0.991 | 1.039 | 0.963 | 0.978 |
| lt | 2.146 | 2.746 | 1.535 | 1.600 | 1.341 | 1.354 |
| lv | 2.224 | 2.847 | 1.632 | 1.741 | 1.358 | 1.376 |
| mt | 2.392 | 3.066 | 1.971 | 2.079 | 1.408 | 1.429 |
| nl | 1.274 | 1.484 | 1.152 | 1.216 | 1.069 | 1.078 |
| pl | 1.370 | 1.668 | 1.178 | 1.247 | 1.077 | 1.087 |
| pt | 1.256 | 1.354 | 1.079 | 1.126 | 1.061 | 1.076 |
| ro | 1.373 | 1.588 | 1.203 | 1.262 | 1.096 | 1.098 |
| sk | 1.880 | 2.511 | 1.443 | 1.522 | 1.270 | 1.296 |
| sl | 1.876 | 2.422 | 1.468 | 1.537 | 1.285 | 1.298 |
| sv | 1.449 | 1.693 | 1.380 | 1.453 | 1.206 | 1.220 |

**ASR conditioned bits/char**

| lang | Llama-1B-base | Llama-1B-ins | Qwen-2B-base | Qwen-2B-ins | EuroLLM-1.7B-base | EuroLLM-1.7B-ins |
|---|---|---|---|---|---|---|
| bg | 2.168 | 2.146 | 1.557 | 1.373 | 1.712 | 1.619 |
| cs | 2.257 | 1.947 | 1.775 | 1.479 | 2.036 | 1.885 |
| da | 2.247 | 1.783 | 1.697 | 1.451 | 1.914 | 1.758 |
| de | 1.732 | 1.324 | 1.353 | 1.118 | 1.636 | 1.426 |
| el | 1.931 | 1.871 | 1.387 | 1.237 | 1.630 | 1.494 |
| en | 1.628 | 1.260 | 1.421 | 1.141 | 1.761 | 1.510 |
| es | 1.754 | 1.337 | 1.410 | 1.167 | 1.705 | 1.475 |
| et | 2.806 | 2.453 | 1.955 | 1.704 | 2.054 | 1.937 |
| fi | 2.232 | 1.739 | 1.736 | 1.496 | 1.896 | 1.771 |
| fr | 1.660 | 1.167 | 1.255 | 1.012 | 1.616 | 1.390 |
| ga | 2.990 | 2.871 | 2.355 | 2.265 | 2.190 | 2.100 |
| hr | 2.364 | 2.040 | 1.698 | 1.474 | 2.040 | 1.930 |
| hu | 2.146 | 1.736 | 1.695 | 1.472 | 1.925 | 1.768 |
| it | 1.758 | 1.340 | 1.356 | 1.119 | 1.634 | 1.413 |
| lt | 2.790 | 2.585 | 1.848 | 1.597 | 2.040 | 1.951 |
| lv | 2.917 | 2.833 | 1.921 | 1.700 | 2.090 | 1.998 |
| mt | 3.013 | 3.113 | 2.125 | 2.002 | 1.976 | 1.911 |
| nl | 1.916 | 1.454 | 1.488 | 1.272 | 1.761 | 1.550 |
| pl | 2.031 | 1.597 | 1.552 | 1.317 | 1.821 | 1.684 |
| pt | 1.843 | 1.376 | 1.471 | 1.208 | 1.765 | 1.552 |
| ro | 1.961 | 1.452 | 1.474 | 1.271 | 1.689 | 1.548 |
| sk | 2.516 | 2.304 | 1.786 | 1.524 | 1.980 | 1.809 |
| sl | 2.556 | 2.222 | 1.784 | 1.537 | 1.988 | 1.887 |
| sv | 2.088 | 1.592 | 1.725 | 1.483 | 1.904 | 1.671 |

**ASR raw tokens/word**

| lang | Llama-1B-base | Llama-1B-ins | Qwen-2B-base | Qwen-2B-ins | EuroLLM-1.7B-base | EuroLLM-1.7B-ins |
|---|---|---|---|---|---|---|
| bg | 2.601 | 2.601 | 2.097 | 2.097 | 1.892 | 1.892 |
| cs | 2.131 | 2.131 | 2.153 | 2.153 | 1.995 | 1.995 |
| da | 2.049 | 2.049 | 1.765 | 1.765 | 1.682 | 1.682 |
| de | 1.899 | 1.899 | 1.537 | 1.537 | 1.554 | 1.554 |
| el | 2.551 | 2.551 | 2.393 | 2.393 | 2.121 | 2.121 |
| en | 1.207 | 1.207 | 1.167 | 1.167 | 1.288 | 1.288 |
| es | 1.584 | 1.584 | 1.346 | 1.346 | 1.376 | 1.376 |
| et | 3.040 | 3.040 | 2.643 | 2.643 | 2.228 | 2.228 |
| fi | 3.487 | 3.487 | 2.801 | 2.801 | 2.446 | 2.446 |
| fr | 1.670 | 1.670 | 1.431 | 1.431 | 1.482 | 1.482 |
| ga | 2.294 | 2.294 | 2.163 | 2.163 | 1.713 | 1.713 |
| hr | 2.515 | 2.515 | 2.152 | 2.152 | 1.927 | 1.927 |
| hu | 3.036 | 3.036 | 2.338 | 2.338 | 2.174 | 2.174 |
| it | 1.833 | 1.833 | 1.452 | 1.452 | 1.470 | 1.470 |
| lt | 3.356 | 3.356 | 2.569 | 2.569 | 2.179 | 2.179 |
| lv | 3.362 | 3.362 | 2.715 | 2.715 | 2.114 | 2.114 |
| mt | 3.381 | 3.381 | 3.109 | 3.109 | 2.529 | 2.529 |
| nl | 1.887 | 1.887 | 1.592 | 1.592 | 1.475 | 1.475 |
| pl | 2.639 | 2.639 | 2.082 | 2.082 | 1.733 | 1.733 |
| pt | 1.700 | 1.700 | 1.418 | 1.418 | 1.420 | 1.420 |
| ro | 2.160 | 2.160 | 1.801 | 1.801 | 1.761 | 1.761 |
| sk | 2.548 | 2.548 | 2.232 | 2.232 | 1.986 | 1.986 |
| sl | 2.447 | 2.447 | 2.144 | 2.144 | 1.851 | 1.851 |
| sv | 2.094 | 2.094 | 1.813 | 1.813 | 1.752 | 1.752 |

**ASR conditioned tokens/word**

| lang | Llama-1B-base | Llama-1B-ins | Qwen-2B-base | Qwen-2B-ins | EuroLLM-1.7B-base | EuroLLM-1.7B-ins |
|---|---|---|---|---|---|---|
| bg | 2.826 | 2.826 | 2.548 | 2.548 | 2.185 | 2.185 |
| cs | 2.413 | 2.413 | 2.717 | 2.717 | 2.361 | 2.361 |
| da | 2.311 | 2.311 | 2.288 | 2.288 | 2.013 | 2.013 |
| de | 2.141 | 2.141 | 2.021 | 2.021 | 1.855 | 1.855 |
| el | 2.777 | 2.777 | 2.845 | 2.845 | 2.415 | 2.415 |
| en | 1.440 | 1.440 | 1.635 | 1.635 | 1.578 | 1.578 |
| es | 1.791 | 1.791 | 1.759 | 1.759 | 1.639 | 1.639 |
| et | 3.372 | 3.372 | 3.306 | 3.306 | 2.649 | 2.649 |
| fi | 3.828 | 3.828 | 3.485 | 3.485 | 2.899 | 2.899 |
| fr | 1.898 | 1.898 | 1.887 | 1.887 | 1.765 | 1.765 |
| ga | 2.507 | 2.507 | 2.588 | 2.588 | 1.977 | 1.977 |
| hr | 2.775 | 2.775 | 2.672 | 2.672 | 2.270 | 2.270 |
| hu | 3.318 | 3.318 | 2.902 | 2.902 | 2.526 | 2.526 |
| it | 2.056 | 2.056 | 1.898 | 1.898 | 1.756 | 1.756 |
| lt | 3.671 | 3.671 | 3.199 | 3.199 | 2.585 | 2.585 |
| lv | 3.670 | 3.670 | 3.332 | 3.332 | 2.504 | 2.504 |
| mt | 3.654 | 3.654 | 3.654 | 3.654 | 2.871 | 2.871 |
| nl | 2.129 | 2.129 | 2.077 | 2.077 | 1.777 | 1.777 |
| pl | 2.932 | 2.932 | 2.669 | 2.669 | 2.128 | 2.128 |
| pt | 1.927 | 1.927 | 1.873 | 1.873 | 1.707 | 1.707 |
| ro | 2.374 | 2.374 | 2.231 | 2.231 | 2.028 | 2.028 |
| sk | 2.821 | 2.821 | 2.777 | 2.777 | 2.336 | 2.336 |
| sl | 2.717 | 2.717 | 2.685 | 2.685 | 2.197 | 2.197 |
| sv | 2.346 | 2.346 | 2.318 | 2.318 | 2.074 | 2.074 |

Fertility (tokens/word) is identical between base and instruct within a
family, in every language, both raw and conditioned — expected, not a
bug: base/instruct share a tokenizer, so this depends only on the
reference text, never on model weights. Included as a cross-check anyone
reading this table can verify at a glance.

**ST(→en) overall (2,300 samples, all target-language English — no
per-language breakdown, since every ST record has `lang=en` regardless of
source locale, per `get_tags_from_cut`'s convention)**

| backbone | raw BPC | cond BPC | raw tok/word | cond tok/word | raw nats/tok | cond nats/tok |
|---|---|---|---|---|---|---|
| Llama-1B-base | 1.007 | 1.679 | 1.242 | 1.489 | 3.411 | 4.744 |
| Llama-1B-ins | 1.094 | 1.250 | 1.242 | 1.489 | 3.702 | 3.531 |
| Qwen-2B-base | 0.959 | 1.464 | 1.208 | 1.701 | 3.340 | 3.619 |
| Qwen-2B-ins | 0.986 | 1.158 | 1.208 | 1.701 | 3.435 | 2.863 |
| EuroLLM-1.7B-base | 1.023 | 1.801 | 1.330 | 1.636 | 3.236 | 4.628 |
| EuroLLM-1.7B-ins | 1.039 | 1.572 | 1.330 | 1.636 | 3.287 | 4.041 |

Action needed: none — this is a reference post, not a task. Fold into
`02-backbones.md` §5's results table once the backbone decision needs it,
or once ru/uk + cascade oracle + `-test` numbers land and a fuller table
is worth building from scratch rather than patched onto this one.

---

## 2026-09-16 — Claude (worker session text-prior-tool-spec) — Text-prior tool run on all six backbones; two infra traps, no tool bugs

Context: continuation of the same-day session that built `melteval
text-prior` and tested it on 1 of 6 backbones. The PI asked directly to
finish the remaining five once the h100 GPUs were free again.

Finding 1, two more real checkpoints needed staging. Only
`meta-llama/Llama-3.2-1B-Instruct` and `Qwen/Qwen3.5-2B` (instruct) were
on artemis already; `meta-llama/Llama-3.2-1B` (base), `Qwen/Qwen3.5-2B-Base`
and both EuroLLM-1.7B checkpoints were not. The PI pointed at
`/mnt/scratch-artemis/giuseppe/.cache/huggingface/hub` as already having
everything -- checked directly rather than assumed: it had Llama base +
instruct and Qwen-instruct, but not Qwen-Base or either EuroLLM-1.7B
checkpoint. Fetched only the genuinely missing three (~10.5G, not the
original ~15.2G plan) from nyx's local cache (already there from the
2026-09-16 MN5-staging session) into that same shared cache.

Finding 2, the real trap: two of the "already there" checkpoints in that
shared cache turned out to be **partial** -- `Llama-3.2-1B` (base) had
only `tokenizer_config.json`/`tokenizer.json`/`special_tokens_map.json`,
no `config.json` and no weights; `Qwen/Qwen3.5-2B` (instruct) had only
`config.json`, nothing else. Both jobs failed in under a minute on
`OSError: We couldn't connect to huggingface.co ... and couldn't find
[the files] in the cached files` (`HF_HUB_OFFLINE=1` blocks the online
fallback, correctly). Not a tool bug -- the shared cache is exactly what
its name says, an accumulation of whatever partial fetches various past
sessions/users needed for their own purposes (tokenizer-only, config-only
checks), not a guarantee of complete weights for any given model. Fixed
by consolidating into the site's own default cache
(`/mnt/scratch-artemis/giuseppe/melt-data/hf_cache`) instead, which
already had a genuinely complete `Qwen/Qwen3.5-2B` (proven by the
Llama-Instruct runs earlier that day using the same cache) -- just needed
to add a complete `Llama-3.2-1B` (rsynced from nyx) alongside it.

Finding 3, my own mistake along the way: a batch resubmission of the 4
failed jobs used `VAR=val VAR=val OUT=... && command` shell syntax --
the trailing `&&` turns the `VAR=val` prefix into a standalone assignment
statement in the *current* shell rather than an exported prefix on the
following command, so `MELT_PARTITION`/`MELT_QOS`/`PYTHONPATH` never
reached `sbatch` or the job. Silently fell back to the site defaults
(`a6000`/`gpu-short` instead of `h100`/`gpu-h100`) and, more importantly,
lost the `PYTHONPATH` override -- all 4 jobs failed in 2 seconds on
`melteval: error: argument command: invalid choice: 'text-prior'`,
because without `PYTHONPATH` the shared venv's editable install resolved
to the *other* concurrent session's dirty `claude/air-bench-support`
checkout (the one flagged as off-limits in the first board entry today),
which has no `text-prior` subcommand. Caught immediately from the job log
-- no partial output was written, nothing from that checkout was actually
executed beyond an argparse rejection. Refiled as a proper script with
explicit `export` statements; all 4 completed clean on the retry.

Finding 4, the real results -- full 6 × 2 (ASR/ST) × dev-set sweep,
12 runs, all clean. Overall ASR conditioned bits/char (lower = backbone
already knows more of the language):

| backbone | ASR cond | ST cond |
|---|---|---|
| Llama-3.2-1B (base) | 2.207 | 1.679 |
| Llama-3.2-1B-Instruct | 1.884 | 1.250 |
| Qwen3.5-2B-Base | 1.651 | 1.464 |
| Qwen3.5-2B (instruct) | 1.428 | 1.158 |
| EuroLLM-1.7B (base) | 1.856 | 1.801 |
| EuroLLM-1.7B-Instruct | 1.700 | 1.572 |

Two things worth the paper's attention, folded into `02-backbones.md` §2
and §3.7:

1. Instruct beats base in **every** family, on **both** tasks, no
   exception -- and base/instruct tokenize identically within a family
   (tokens/word matches to 3 decimals), so the gap is isolated cleanly to
   chat-template familiarity, not a fertility artifact. This turns §2's
   "base arms at MA are template-naive" from a stated caveat into a
   measured number, before any arm has trained.
2. Qwen3.5 leads the aggregate ranking (Qwen-Ins < Qwen-Base < EuroLLM-Ins
   < EuroLLM-Base < Llama-Ins < Llama-Base), but Maltese -- the hardest
   language here for every backbone -- inverts the top of it: EuroLLM-Ins
   (1.911) beats Qwen-Ins (2.002) beats Llama-Ins (3.113). EuroLLM's
   EU-specific training earns its keep exactly on the language it should,
   even while trailing in aggregate against a broader-multilingual
   competitor.

ru/uk show up as `--` in every backbone (not `0` or an error) -- expected,
not a bug: the production `fleurs24-asr-dev`/`fleurs24-st-xen-dev` copies
on artemis scratch don't have the PR #12/#13 ru/uk rows deployed yet (see
the earlier board entry today).

Full per-language JSON for all 12 runs:
`/mnt/scratch-artemis/giuseppe/melt-data/text-prior/<tag>-{asr,st}-dev.json`
on artemis scratch. Tags: `llama1b-base`, `llama1b-ins`,
`qwen35-2b-base`, `qwen35-2b-ins`, `eurollm17b-base`, `eurollm17b-ins`.

Action needed: review/merge melt-eval PR #14 (and #12/#13, still open).
Whoever redeploys the ru/uk frozen sets to production should re-run these
same 12 commands afterward for the +ru/uk numbers. The cascade oracle
(row 5) and the `-test` splits are still open work, not blocking anything
in the current week. The two isolated artemis clones
(`agent-worktrees/melt-eval-text-prior`, `.../training-text-prior`) and
the consolidated `melt-data/hf_cache` entries for all six backbones can
be reused directly for either.

---

## 2026-09-16 — Claude (worker session qwen-ift-throughput-2node) — Qwen3.5-2B IFT throughput at 16 nodes: scaling is flat 2->16 nodes

Context: the 16-node contingency point queued on `acc_ehpc` above finally landed.

Finding: job 45902185 started well before SLURM's own estimate (17:19 vs ~22:00) and completed cleanly in 28m28s, 30/30 steps. Steady state ~33 s/step -- essentially identical to the 2-node (~31 s/step) and 8-node (~32.3 s/step) rates. GPU-h for the ~330K h IFT pool at 16 nodes comes out to ~18,150, the same ballpark as ~17,800 at 8 nodes. Qwen's IFT throughput scales flat across the whole 2-16 node range measured this session, so above 8 nodes it is a pure wall-clock-vs-node-count choice, not a GPU-h efficiency tradeoff -- unlike Llama's measured 46-65% penalty under the fixed-effective-batch regime (this campaign's Qwen arms hold `grad_accum` fixed instead, which is the regime that scales cleanly). Same benign Triton-autotune-cache atexit race seen at 8 nodes recurred here too (harmless, job completed with a real `TrainOutput`). Full numbers in `06-fondue.md` §2.

Action needed: none. The node-scaling question for Qwen IFT-700 is closed for 2-16 nodes; a measurement above 16 nodes would only matter if a Fondue contingency needs more than that.

---

## 2026-09-16 — Claude (worker session text-prior-tool-spec) — Text-prior tool built and tested end to end; a real language-table bug found and fixed; an artemis near-miss avoided

Context: the PI asked directly to build and test `melteval text-prior` on
a free GPU on `dionysus` (artemis, `partition=h100 qos=gpu-h100`) --
jumping ahead of week 2's "Text-prior tool built" item on the PI's own
instruction (`02-backbones.md` §3.1-§3.6 was the spec from earlier the
same day).

Finding 1, the tool: built `melteval/text_prior.py` + a `text-prior` CLI
subcommand + `infra/run_text_prior.sbatch`/`submit_text_prior.sh`
([melt-eval PR #14](https://github.com/MELT-proj/eval/pull/14)), covering
rows 1-4 of the spec's table (raw + conditioned NLL/BPC, fertility) -- not
the cascade oracle, a separate mechanism per spec §3.2, not built this
session. CPU-validated on nyx before spending GPU time (3-sample slices,
`meta-llama/Llama-3.2-1B-Instruct`): Hungarian's tokens/word came out
~3x English's, matching the spec's own predicted 2-4x fertility gap --
good sign the mechanics were right before the real run.

Finding 2, a real bug the test surfaced: the first GPU attempt (job
332112) crashed 10 languages into the 24-language ASR dev set, on `ga`.
Not a bug in the new tool -- `apply_chat_template_to_texts` correctly
raised on an unmapped language code, and the training repo's
`LANGUAGE_ISO_TO_NAME` (`melt/training/data/audio/lhotse/helpers.py`) had
**no entry for Irish at all**, the only gap among all 26 target languages
(checked systematically against the dict, not just patched and moved on).
This is a live gap, not hypothetical: any `{lang}`-templated prompt for
`ga` -- training or eval -- would hit the same raise; `fleurs24-asr-{test,
dev}.yaml` already tag `ga_ie` cuts `lang: ga`. Fixed with one dict entry (`"ga": "Irish"`), training repo `main`
(`b5f4962`, merged up through `aff49df` alongside a concurrent session's
`04-regime.md` work landing at the same time -- no conflict, different
files).

Finding 3, an infra near-miss avoided: the artemis melt-eval checkout at
`/mnt/scratch-artemis/giuseppe/melt-proj-src/melt-eval` (the one the
shared `melteval` venv is editable-installed from) was mid-work on branch
`claude/air-bench-support` -- dozens of modified/untracked files, not
committed, presumably a concurrent session's WIP (no matching open PR).
Did not touch it. Instead cloned isolated copies for this session's own
use: `/mnt/scratch-artemis/giuseppe/agent-worktrees/melt-eval-text-prior`
(my branch) and `.../training-text-prior` (main, with the `ga` fix), and
pointed the SLURM job at them via `PYTHONPATH` while still using the
shared venv's installed torch/transformers (not modified). Both isolated
clones are left in place -- small, harmless, reusable for the next
backbone; remove whenever convenient.

Finding 4, the real results. Second GPU attempt (job 332113, ASR;
332114, ST) completed cleanly, `meta-llama/Llama-3.2-1B-Instruct` against
the production `fleurs24-asr-dev` (24 langs, 2,400 samples) and
`fleurs24-st-xen-dev` (23 locales, 2,300 samples) -- the ru/uk config
addition (PR #12/#13) isn't deployed to these production copies yet, so
this run is the pre-ru/uk 24/23-language scope. ~10 and ~7 minutes wall on
one H100, one sample at a time, no batching (as spec'd -- correctness
over throughput for a once-per-backbone measurement).

Overall: ASR raw 1.940 bits/char, conditioned 1.884 (110,014 / 122,014
target tokens over 2,400 samples); ST(->en) raw 1.094, conditioned 1.250
(57,937 / 69,437 tokens over 2,300). Per-language spread is the real
validation -- an English-centric 1B backbone should and does know English
best and the EU's smallest/least-resourced languages worst:

| lang | raw BPC | cond BPC | raw tok/word | cond tok/word |
|---|---|---|---|---|
| en | 1.072 | 1.260 | 1.207 | 1.440 |
| es | 1.284 | 1.337 | 1.584 | 1.791 |
| fr | 1.157 | 1.167 | 1.670 | 1.898 |
| de | 1.251 | 1.324 | 1.899 | 2.141 |
| hu | 1.815 | 1.736 | 3.036 | 3.318 |
| mt | 3.066 | 3.113 | 3.381 | 3.654 |
| ga | 2.753 | 2.871 | 2.294 | 2.507 |
| lv | 2.847 | 2.833 | 3.362 | 3.670 |

(all 24 languages in the JSON output, not just these 8.) Maltese highest
BPC of all 24 -- plausible, it's one of the smallest-population EU
languages and typologically an outlier (Semitic, Latin-scripted) among
its neighbours. Hungarian and Maltese fertility land at 2.1-3.0x
English's, inside the "2-4x" this section's own prose already predicted.
`ga`'s numbers sit on a different textual basis than the other 23 (no
`pnc_text`, lowercase/unpunctuated fallback -- known gap, `fleurs24-asr-
test.yaml` header) so don't read its exact position as a clean signal,
only its general "high" cluster. Full per-language JSON for both tasks:
`/mnt/scratch-artemis/giuseppe/melt-data/text-prior/llama1b-ins-{asr,
st}-dev.json` (artemis scratch, not committed anywhere -- regenerate with
the same two `sbatch` commands if lost, they're cheap: ~10 GPU-minutes
each).

Action needed: review/merge melt-eval PR #14. A session runs the
remaining five backbones (same two commands, swap `--model` and
`--chat-template-config`/`--chat-template-from` per `02-backbones.md`
§3.1's table) to get the real 6x24 prior table week 2 wants. The cascade
oracle (row 5) still needs building. The two isolated artemis clones can
be reused for that rather than re-cloned.

---

## 2026-09-16 — Claude (strategy session) — The MA:IFT hours ratio was missing from the plan; now `04-regime.md` §6

Context: the PI raised that no arm ever chose the split between MA hours and
IFT hours (today 700 h/lang of ASR in MA, 700 h/lang of ASR+ST in IFT), and
that the right split plausibly depends on whether IFT trains the decoder
fully, with LoRA, or not at all.

Finding / proposal: added as §6 of `04-regime.md`, to run after the week-5
regime decision, with three things shaping the design.

1. The split means different things per regime. Under a frozen decoder both
   stages train the same 6.3M parameters at the same cost per hour, so the
   split is a pure curriculum question and the honest comparison is a single
   stage with instructions from the start. Under full fine-tuning, IFT can
   repair a weak interface but costs several times more per hour and erodes
   text ability, so MA front-loads the cheap work. Factor A shifts it again:
   with the adapter trainable at IFT, MA is only a warm start.
2. Only IFT carries ST, so at fixed total hours a larger MA share silently
   cuts ST exposure. The sweep therefore splits the ASR hours only and holds
   ST fixed at 700 h per direction.
3. Intermediate checkpoints of a cosine run are not shorter runs, so each MA
   budget gets its own full schedule.

Points: MA 0/100/300/700 h/lang against IFT ASR 700/600/400/0, the existing
campaign arm as the reference, and an evaluation-only zero-IFT probe on the
instruct backbones. P0 and P3 repeat under the runner-up regime as the
interaction check. The decoder-frozen regime was itself absent from the
half fraction and is added as R9/R10, paired against R4 and R6.

Cost, using the throughput numbers measured this week: about 450 GPU-h for a
Llama-class winner, about 1,800 for a Qwen-class one (306 GPU-h per
production IFT arm, per `06-fondue.md` §2). Both fit, but a Qwen winner
makes the runner-up points the first thing to cut. The sweep decides
Fondue's "MA data budget and stage split" row; if it misses the 2026-10-25
freeze, the default is full MA as today.

Action needed: week 5 Track B needs per-task budgets in
`build_campaign_config.py` (IFT renders with ASR at 600/400/0 while ST stays
at 700). The MA runs at 100 and 300 h/lang do not depend on the regime
decision and can be queued as soon as the recipe is fixed.

---

## 2026-09-16 — Claude (worker session text-prior-tool-spec) — ru/uk added to the FLEURS melt-eval configs; PR #11 merged mid-session

Context: follow-up to the text-prior tool spec below, closing the two
prerequisites it surfaced -- the PI asked directly for the ru/uk config gap
and the `DECODER_PROFILES` gap to be added to week 1's work.

Finding / proposal: added `ru_ru`/`uk_ua` sources to all four FLEURS melt-eval
configs (`fleurs24-asr-{test,dev}`, `fleurs24-st-xen-{test,dev}`), same
shape as the existing rows. Verified before adding rather than assumed, the
way `ga`'s gap was originally found: checked the shar tree directly first
(both locales carry `custom.pnc_text` on `test` and `validation`, unlike
`ga`), then re-froze all four specs and confirmed zero drops --
`fleurs24-asr-test` 19,463→20,988 samples (63.41h→68.17h, 24→26 langs),
`-dev` 2,400→2,600 (7.41h→8.01h); `fleurs24-st-xen-test` 18,816→20,341
(61.63h→66.39h, 23→25 locales), `-dev` 2,300→2,500 (7.14h→7.75h). ru: 775/775
ASR test cuts kept, 100/100 dev; ST reference_map zero-dropped too (same
counts). uk: 750/750 ASR test, 100/100 dev; ST likewise. Hand-checked 3 ru
and 3 uk (source, English reference) ST pairs -- all correctly paired.
[melt-eval PR #12](https://github.com/MELT-proj/eval/pull/12) (ASR) and
[melt-eval PR #13](https://github.com/MELT-proj/eval/pull/13) (ST), both
open against `main`. Neither redeploys the production frozen-set copies on
artemis scratch -- a re-freeze + copy, not done here.

Surprise: PR #11 (the ST frozen sets this ru/uk work builds on) showed as
**open** in every plan file at the start of this follow-up, per this same
session's own earlier board entry a few minutes prior -- but `gh pr view 11`
showed it had actually been merged by the PI (`g8a9`) at
2026-09-16T10:55:17Z, apparently while the spec-drafting work above was in
progress. My first push of the ru/uk ST commit landed on the
now-closed-by-merge `claude/fleurs-x-to-en-st` branch, which doesn't reopen
or update #11 -- caught via `gh pr view`, not assumed, and re-filed as new
PR #13 against `main` instead. Fixed the stale "PR #11 open" claims this
same session had just written into `02-backbones.md` §3.5,
`00-status.md`, and `timeline.md`.

`DECODER_PROFILES` (`plan_arm.py`) still has no EuroLLM entry -- not
implemented here, since the PI's ask was to add it to the things that need
doing, not to do it; added as its own `timeline.md` week-1 Track B item
with the exact values this session already verified (`chatml`,
`chat_template_from: utter-project/EuroLLM-1.7B-Instruct`).

Action needed: review/merge melt-eval PR #12 and PR #13. A future session
(or whoever builds the text-prior tool) adds the `DECODER_PROFILES` entry
and re-freezes the production copies on artemis scratch before relying on
the 26-language sets there.

---

## 2026-09-16 — Claude (worker session text-prior-tool-spec) — Text-prior tool spec drafted (`02-backbones.md` §3)

Context: week 1 Track B, `timeline.md` ("Text-prior tool spec agreed
(`02-backbones.md` §3)").

Finding / proposal: expanded §3 from a measurement table into an
implementable spec (new §3.1–3.6), checked against melt-eval's actual code
(`tasks.py`, `solver.py`, `scorers.py`, `prompt.py`, `dataset.py`,
`manifest.py`) and `no_audio_floor.py`, not written from the table alone.
Two things worth flagging beyond the spec itself:

1. Teacher-forced NLL/BPC (table rows 1–4) cannot go through `inspect
   eval` — melt-eval's Task/Solver/Scorer triad is generate-then-score-the-
   completion, and nothing in `scorers.py` reads a logprob off
   `state.output`. Proposed as a standalone `melteval text-prior` CLI
   command instead, generalizing `no_audio_floor.py`'s approach
   (teacher-forced forward pass, `mask_non_assistant_tokens`) but *without*
   the `MELTForCausalLM` wrapper: the six backbones are bare
   `AutoModelForCausalLM` checkpoints with no `<|audio|>` token and no MELT
   vocab extension, and the whole point is the prior *before* any
   MELT-specific change. The cascade oracle (row 5) is pure generation and
   reuses the existing `st` task/`st_scorer` unchanged, via inspect's stock
   `hf` provider (no need for `providers/melt.py`, which exists for audio
   batching this measurement never needs) plus one new solver that
   substitutes the gold source-language transcript for the audio content.
2. Chat-template-per-backbone is not uniform across families — checked each
   directly rather than pattern-matched from Llama: `Qwen/Qwen3.5-2B-Base`
   ships its own chat template byte-identical to the Instruct one
   (`DECODER_PROFILES`, already verified 2026-09-02, no borrowing needed),
   but `utter-project/EuroLLM-1.7B` (base) does not (checked this session
   against its `tokenizer_config.json` on nyx — no `chat_template` key)
   while `-Instruct` ships plain ChatML (confirmed directly). EuroLLM has no
   `DECODER_PROFILES` entry yet in `plan_arm.py` — flagged in the spec for
   whoever builds the tool (and needed anyway for week 3's MA arms).

Two data gaps found while grounding the spec against the actual FLEURS
configs, both mechanical (missing config rows, not a design question):
`fleurs24-asr-test.yaml`/`-dev.yaml` cover exactly the 24 EU languages, not
ru/uk, despite `data/hours_by_language.csv` showing FLEURS audio exists for
both (8.1h ru, 9.0h uk, `asr_fleurs` column) — this section's own "24 EU
languages (plus ru, uk)" scope is currently unmet by the frozen sets it
depends on. The ST X→en config likely has the same gap once PR #11 merges
(not checked directly — that branch's config wasn't re-verified for ru/uk
specifically). Neither blocks the ASR half of the tool; both block full-scope
coverage.

Also fixed two stale "EuroLLM ids unconfirmed" notes in this file's header
and grid table (§1) — resolved by the 2026-09-16 staging entry below, just
not reflected there yet.

Action needed: a session builds the tool per §3.1–3.6 (week 2, Track B).
Before or during that: add ru/uk rows to the FLEURS ASR frozen-set configs
(and the ST one, once PR #11 merges) in melt-eval; add an EuroLLM entry to
`DECODER_PROFILES` in `plan_arm.py`. PR #11's merge is a hard dependency for
the ST/cascade-oracle half specifically (already tracked in `00-status.md`
Blocked/waiting).

---

## 2026-09-16 — Claude (worker session moe-adapter-verbatim-test) — Qwen3.5-2B-Base and both EuroLLM checkpoints staged to MN5, offline loads verified

Context: week 1 Track B, `timeline.md` ("Stage models on MN5 over `mn5transfer`: `Qwen/Qwen3.5-2B-Base`, `utter-project/EuroLLM-1.7B`, `utter-project/EuroLLM-1.7B-Instruct`. Confirm the EuroLLM ids on the Hub first; verify offline loads before any allocation.") and the matching `00-status.md` Blocked item.

Finding: confirmed all three repo ids exist and are ungated via the Hub API before downloading anything, resolving the open question in `00-status.md` ("EuroLLM checkpoint ids on the Hub are unconfirmed") — `utter-project/EuroLLM-1.7B` and `-Instruct` are exactly the ids already assumed everywhere else in the plan. Downloaded on nyx (`hf download`, nyx has outbound internet unlike MN5) into the local HF cache: `Qwen/Qwen3.5-2B-Base` 4.3G (12 files), `utter-project/EuroLLM-1.7B` 3.1G, `utter-project/EuroLLM-1.7B-Instruct` 3.1G (9 files each), 11.22G total. Both EuroLLM checkpoints ship only `pytorch_model.bin`, no safetensors — same shape as the known `.bin`-offline trap, not new. rsynced to `mn5transfer:/gpfs/scratch/epor48/hf_cache/hub/` (`--no-owner --no-group --partial`, dry-run first): exit 0, sizes matched on the far side.

Verified offline load on `alogin1` inside the promoted container (`melt_cuda126.sif`, `HF_HUB_OFFLINE=1`, `HF_HOME=/gpfs/scratch/epor48/hf_cache`): `AutoConfig`/`AutoTokenizer`/`AutoModelForCausalLM.from_pretrained` succeeded for all three with no Hub contact. `Qwen/Qwen3.5-2B-Base`: `model_type qwen3_5`, 1,881,825,088 params, vocab 248,077 — loads fine under the promoted transformers-5 image (the `qwen3_5` model type gap was specific to the old `4.57.1` pin, per `qwen35-transformers-support-gap`; not an issue here). `utter-project/EuroLLM-1.7B` and `-Instruct`: `model_type llama`, 1,656,850,432 params each, vocab 128,000 (architecture identical between base and instruct, as expected). Ran this on the login node CPU only, no `sbatch` — it's a config/tokenizer/weight-file parse check, not GPU work, same class of check as the `lhotse`/`torch`/`torchdata` version probe in `docs/hpc_runbook.md`. Deleted the two small helper files (`verify_offline_load.py`, `run_verify.sh`) from `/gpfs/scratch/epor48/` after use.

Action needed: none. All three checkpoints are staged and load offline; the backbone grid in `02-backbones.md` and the Fondue backbone options in `06-fondue.md` §3 can now be scheduled against them without risking a load failure minutes into a queued job.

---

## 2026-09-16 — Claude (worker session qwen-ift-throughput-2node) — Qwen3.5-2B IFT throughput at 8 nodes measured; 16 nodes queued

Context: follow-up requested in the same session as the 2-node measurement above, to close the node-scaling gap flagged there.

Finding: 8 nodes x 4 GPUs (job 45902184, `acc_debug`, same recipe as the 2-node probe with `grad_accum 20` held fixed so effective batch scales to 19,200 audio-s/step): steady state ~32.3 s/step -- essentially flat versus the 2-node rate (~31 s/step). GPU-h stays roughly constant per the "keep `grad_accum`, add nodes" regime already established for Llama (`ift700-scaling-measured`). This is Fondue's actual planned topology, so it directly answers the open question rather than extrapolating: ~17,800 GPU-h for the ~330K h IFT pool, about 3x Llama's 6,000 GPU-h estimate but far under the old unmeasured "25,000+" guess. Full numbers in `06-fondue.md` §2.

A 16-node job (45902185) was also submitted as the requested contingency point, but 16 nodes exceeds `acc_debug`'s 8-node cap, so it queued on `acc_ehpc` instead -- no priority scheduling there. Still pending as of 2026-09-16 01:40; SLURM's own estimate puts the start over a day out (2026-09-17T12:30), though that estimate is often pessimistic and updates as the backfill scheduler reshuffles.

Action needed: none blocking -- the 8-node number already answers the Fondue-topology feasibility question. Check back on job 45902185 and fold its result into `06-fondue.md` §2 when it lands (16-node point is a contingency only, per §6).

---

## 2026-09-15 — Claude (worker session qwen-ift-throughput-2node) — Qwen3.5-2B IFT throughput measured; a finished-but-unrecorded Qwen arm pair found; an infra near-miss

Context: week 1 Track A, `06-fondue.md` §2 / `timeline.md` week 1 ("Qwen3.5-2B IFT throughput at 2 nodes x 4 GPUs, 20-50 steps, gradient checkpointing on, eval and saves off, acc_debug").

Finding 1, the throughput: ran a clean acc_debug probe (`IFT-700-qwen35-2b-throughput-debug`, job 45894977, 2 nodes x 4 GPUs, DDP, `gradient_checkpointing: true`, `batch_duration 30`/`grad_accum 20` matching the campaign's `IFT-700-qwen35-2b-ins` row, `--trainer.max_steps 30 --trainer.eval_steps 999999 --trainer.save_steps 999999 --trainer.eval_on_start false`, initialised from the real `MA-700-qwen35-2b-ins` checkpoint). Steady state ~31 s/step, converged by step ~15 (tqdm's own second-to-last line: `29/30 [27:32<00:30, 30.80s/it]`; delta between step 10 and step 30 gives 31.5 s/step). First submission (job 45894285) FAILED at startup: `ValueError: Output directory ... already exists and is not empty` on 3 of 4 ranks -- rank 0 (or an early writer) drops `resolved_config.json` into a fresh `output_dir` before every rank's own Trainer-init emptiness check runs, so a brand-new multi-rank IFT submission needs `--trainer.overwrite_output_dir true` even the first time. Worth a one-line note wherever the campaign launchers are documented; did not chase the root cause further (harmless once you know to pass the flag).

Finding 2, the real arm was already done: while building the debug command, found `IFT-700-w2vbF-qwen35_2bInsT-mlpF-bd30-ga20-elr6e6-dlr2e5-lr2e4-s1337-8g` already sitting complete in outputs (job 45685241, finished 2026-09-12), initialised from an already-finished `MA-700asr-w2vbF-qwen35_2bInsF-mlpT-bd30-ga20-elr6e6-dlr2e5-lr2e5-s42-8g` (finished 2026-09-05) -- both exactly the current campaign.yaml recipe (`bd30-ga20` tags match `IFT-700-qwen35-2b-ins`/`MA-700-qwen35-2b-ins` verbatim), both with real eval scores in `trainer_state.json` (IFT final: asr_de WER 4.49, asr_fr 6.95, asr_es 7.84, asr_it 8.31, st_en_de WER 11.38 at step 5048/5048). Neither is in `arms.tsv` nor mentioned in `00-status.md`. Its full-epoch closing average (5,048 steps, 27.3 s/step, includes `eval_on_start` + 7 eval rounds + 7 checkpoint saves) is close to but faster than the clean 30-step debug probe above -- backwards from what eval/save overhead should do, so read it as sample variance in which cuts a 30-step window draws, not as the debug method being wrong. Wall 38h19m at world_size 8 -> ~306 GPU-h for this 6,729.85 h arm. Extrapolated (not measured) to Fondue's ~330K h IFT pool at the same 2-node topology: ~15,000-17,000 GPU-h, replacing "the 46 h budget for 6.7K h implies 25,000+" (which was never a measurement -- `46:00:00` was a conservative wall-time request). This is a floor, not the Fondue-topology answer: Llama's own Fondue GPU-h already bakes in a measured 46-65% node-scaling penalty (`ift700-scaling-measured`) that has no Qwen equivalent yet. Numbers and caveats are in `06-fondue.md` §2.

Finding 3, an infra near-miss: setting REMOTE_REPO to an alternate path before calling `infra/sync_repo.sh mn5 --init` silently ignored the override -- `infra/runners/sites/mn5.sh` had `export REMOTE_REPO=training` with no `${VAR:-default}` fallback (unlike every other variable in that file), so the sync pushed straight to the shared `training` checkout and checked out my branch there. That checkout was mid-use by a concurrent session (`claude/librispeech-step0-l1-l4-60d122`, uncommitted smoketest scripts, 4 jobs pending in the queue) -- caught via `git log`/`git branch --show-current` immediately after, restored their branch, confirmed HEAD/status matched what it was before. Their untracked files were never touched (checkout doesn't remove untracked files); the exposure window was under a minute and nothing else on the cluster reads that checkout path live, but a longer job depending on tracked files there mid-checkout would not have been so lucky. Fixed `mn5.sh` to use the same `${VAR:-default}` pattern as the rest of the file; verified the override now works (`--dry-run` shows the right remote). Ran the actual debug job from a separate `training-qwen-ift-throughput` checkout under $HOME instead (manual `git init` + `git push` to a fresh remote path, done before the fix above landed); about 11 MB, left in place on MN5 for now -- this session's own tooling could not clean it up safely, so it is still there; harmless to remove by hand whenever convenient.

Action needed: a session should backfill `arms.tsv` for the two completed Qwen arms (the exact IFT command is in `training/logs/melt-train-container.45685241.out` line 14 on MN5; the MA arm's job id needs a `sacct` lookup around 2026-09-04/05) and fold the IFT eval numbers into `02-backbones.md` §5 -- but only once week 3 settles what "the new baseline" compares against, since it's unclear whether this run predates or postdates every recipe decision in `01-interface-recipe.md`. A follow-up throughput probe at Fondue's real 8-node topology would close the node-scaling gap in Finding 2. The spare `training-qwen-ift-throughput` checkout under $HOME on MN5 can be removed whenever convenient.

---

## 2026-09-15 — Claude (worker session fleurs-x-to-en-st) — FLEURS X→en ST frozen sets built

Context: week 1 Track B, `05-language-ladder.md` §3.1 ("FLEURS X→en ST frozen
sets in melt-eval"). Task description named "the preprocessing repo" as
home; §3.1 and the 2026-09-15 strategy-session entry below both say home is
melt-eval (a shar-reader option, not a rewrite of the shar tree), so I built
it there. Branch `claude/fleurs-x-to-en-st` in melt-eval (worktree moved to
sit next to `training` so `melt-proj = {path = "../training", editable =
true}` resolves — nested under `melt-eval/.claude/worktrees/` does not),
[PR #11](https://github.com/MELT-proj/eval/pull/11).

Finding / proposal: added `reference_map` as a new `SharReader` option
(`melteval/readers/shar.py`) — a `{cut_id: text}` JSON that overrides a
cut's target text entirely, bypassing `text_field`/`get_text_from_cut`, for
sources like this one where the reference isn't on the cut at all. Built
`configs/refs/fleurs-en-reference-{test,dev}.json` from the HF
`google/fleurs` en_us metadata's `raw_transcription` (350 test / 150 dev
ids, `scripts/build_fleurs_en_reference.py`), not from the shar tree's
`custom.pnc_text` — confirmed §3.1's finding that `pnc_text` differs across
an id's duplicate recordings (57/350 test ids) while `raw_transcription` is
single-valued per id (checked directly, 0/350 and 0/150 disagreements). The
majority-vote fallback §3.1 specifies was not needed.

`configs/fleurs24-st-xen-{test,dev}.yaml` cover all 23 non-English EU
locales. Froze against the real shar tree: test is 18,816 samples / 61.63 h,
**zero** dropped; dev is 2,300 samples / 7.14 h (100/lang, seeded subset of
`validation`). Spot-checked 20 random (locale, English reference) pairs by
hand — all correctly paired. Copied to
`/mnt/scratch-artemis/giuseppe/melt-data/eval-sets/{fleurs24-st-xen-test,fleurs24-st-xen-dev}/`.

Deviated from §3.1's tag example: gave each locale its own `dataset_id`
(`fleurs-<src>_en`) instead of a shared `dataset_id: fleurs`. Reason:
`get_tags_from_cut` (training repo) sets the returned `lang` to the
*target* language for any `task: st` cut, so every record here has
`lang=en` regardless of audio locale (confirmed: freezing showed
`"languages": {"en": 18816}`) — the existing `grouped(..., "lang", ...)`
BLEU/chrF metric would collapse all 23 source languages into one bucket
under the flat dataset_id. Per-locale dataset_id lets `-T
dataset_id=fleurs-de_en` isolate one language, matching how
`st-eval-campaign-v1.yaml` already isolates CoVoST2 directions. Documented
in both config headers.

Environment note for whoever builds the MN5/next melt-eval venv (open Track
B item this week): built a working one at
`/mnt/scratch-nyx/giuseppe/venvs/melteval-fleurs-x-to-en-st` (nyx only,
CPU -- reuse it for freezing rather than rebuilding). `uv pip install
--prerelease=allow -e ".[shar,metrics,dev]"`
resolved `inspect_ai==0.3.254` but its agent/ACP code path imports a
third-party `acp` package (`acp.helpers`, `acp.schema`, ...) that is **not**
declared in `inspect_ai`'s own dependencies and is *not* the `acp-sdk` PyPI
package (wrong namespace, no `helpers.py`) — needed `pip install
agent-client-protocol==0.12.1` (uninstall any `acp-sdk` first, they collide
on the `acp` import name) before `import melteval` stopped crashing at
`melteval/dataset.py`'s `from inspect_ai.dataset import ...`. None of this
is exercised by `melteval freeze` itself; it's pulled in only because
`melteval/__init__.py` imports the whole package eagerly for `inspect_ai`
registry side effects.

Incidental finding, not fixed here: FLEURS `cs_cz` `test` sentence id
`1904` (all 3 duplicate recordings) has leaked LLM self-correction reasoning
in `custom.pnc_text` ("Vzhledem k tomu, že zadání vyžaduje opravy pouze
interpunkce... **Korekce:**...", ~700–1500 chars) instead of a corrected
transcript — a preprocessing-repo PNC-pass bug. Scanned all 23 X locales ×
test/validation (length-ratio heuristic) for the same pattern: this is the
**only** occurrence. Doesn't affect this PR's target text (English, via
`reference_map`), only `source_text` for future COMET scoring of `cs`; a
training-time ASR run using `cs` `pnc_text` as `text_field` would train on
this leak for that one id, though 1/723 cuts is unlikely to matter.

Action needed: PI/reviewer review and merge
[PR #11](https://github.com/MELT-proj/eval/pull/11). Someone with
preprocessing-repo context checks the `cs_cz`/id 1904 PNC leak and whether
it recurs elsewhere the length-ratio heuristic here didn't cover (non-FLEURS
corpora, non-`pnc_text` fields). `05-language-ladder.md` §3.1's tag example
could use a one-line update to `dataset_id: fleurs-<src>_en` so it matches
what got built.

---

## 2026-09-15 — Claude (strategy session) — FLEURS X→en set specified; the no-audio floor revises the mechanism, not the conclusion

Context: the PI could not find the week-1 item "FLEURS X→en ST set" in the
plan; it was one under-specified timeline line. Also read the no-audio floor
result posted below.

Finding 1: the join is feasible and now specified in `05-language-ladder.md`
§3.1. On the nyx shar tree, 100% of X/test sentence ids (de, it, ga, mt, hu,
lv checked) are present in `en_us/test`, so FLEURS' split assignment is
consistent across languages. The English `pnc_text` differs across duplicate
recordings for 57/350 ids (per-recording truecasing), so the reference must
be chosen deterministically: use the original English transcription from
the FLEURS metadata, with majority `pnc_text` as fallback. Home is
melt-eval (a reader option that overrides the target text by cut id), not
preprocessing; nothing in the shar tree is rewritten.

Finding 2, my reading of the floor: 4.0–4.3 nats with no audio against
2.6–3.1 with audio means the adapter conveys coarse audio-conditioned
information (language identity, register, utterance length, maybe partial
lexical content) worth 1.1–1.5 nats, but not the frame-to-token alignment
transcription needs. That is the coarse-feature plateau; the transition to
fine alignment is what needs more optimizer pressure (LR, steps) and
shorter sequences (stacking). "Failed alignment" stands; "audio was
ignored" is withdrawn. Consequence for step 0: success needs WER *and* an
eval loss well under 1 nat per token on LibriSpeech dev (a working
frozen-LLM ASR should sit around 0.2–0.5), not merely a loss below the
floor. A run whose loss drops while WER stays above 0.5 is repeating the
coarse-signal pattern. `01-interface-recipe.md` §1, its Settled block and
the step-0 interpretation rules are updated; PI to confirm.

Action needed: a melt-eval session takes §3.1. The revised reading was
confirmed by the PI on 2026-09-15.

## 2026-09-15 — Claude (worker session llama-3-2-1b-no-audio-floor-d4bc19) — No-audio floor measured: audio was NOT ignored

Context: week 1 Track A, `01-interface-recipe.md` §1's "no-audio floor"
control. Wrote `no_audio_floor.py` + `no_audio_floor.sbatch` (new,
committed): scores a frozen backbone on the MA validation transcripts,
same chat-templated prompt and label masking the real collator uses, but
`input_features=None` so no audio is ever read from disk (reuses the
collator's own text/tag/template helpers directly instead of
`MELTMapDataset`/`MELTDataCollator`, which call `cut.load_audio()`
unconditionally). Validated with a CPU smoke test (8 cuts/lang, sdpa) inside
the training container on nyx before spending cluster time. First submitted
on artemis (job 331970); cancelled after ~2h queued behind another user's
two jobs occupying all 4 GPUs on the only h100 node. Re-submitted on MN5
under `acc_debug` QOS (job 45888659), completed in 1m20s. Both clusters'
shared checkouts were left untouched — pushed the branch as a new ref and
ran from a `git worktree`, removed after.

Finding: scored frozen Llama-3.2-1B-Instruct on the exact 200-cuts/language
eval subset (same seed) `MA-700asr-w2vbF-llama1bInsF-mlpT-s42-8g-md60`
uses, and pulled that arm's own final (step 2600) `eval_<lang>_loss` from
its `trainer_state.json` for a like-for-like comparison (confirmed via its
`resolved_config.json`: `apply_chat_template: true`, `chat_template_config:
llama3`, so both numbers are on the same eval formatting — the run finished
2026-08-28, after the 2026-08-09 fix that made eval inherit the chat
template):

| lang | no-audio floor | MA eval_loss (with audio) | floor − with-audio |
|---|---|---|---|
| en | 4.073 | 2.948 | +1.125 |
| de | 4.312 | 3.123 | +1.189 |
| fr | 3.968 | 2.739 | +1.229 |
| es | 4.040 | 2.659 | +1.381 |
| it | 4.154 | 2.674 | +1.480 |

**The floor is 1.1–1.5 nats/token *above* the observed MA loss on every
language, not equal to it.** `01-interface-recipe.md` §1 and `00-status.md`
both describe the 2.6–3.1 eval loss as "roughly what a 1B text LM scores
... with no audio" — measured, it is not close; audio conditioning cuts the
loss by ~3–4.4x in per-token perplexity versus no audio at all. So the
adapter did not simply learn to ignore the audio and emit fluent filler;
it learned *something* audio-conditioned that measurably helps next-token
prediction, just not the right thing for correct transcription (WER stayed
1.10–1.16, hypotheses fluent but unrelated to the reference). I've added
the numbers and this reading to `01-interface-recipe.md` §1a rather than
touching the Settled block myself.

Action needed: PI review of the diagnosis in light of this — it doesn't
overturn "the August baseline is a failed alignment" (WER is still
catastrophic) but it does overturn "audio was ignored" as the explanation,
which changes what a passing LibriSpeech screen run should look like (a run
that only drops eval loss without moving WER may be repeating this same
coarse-signal pattern rather than fixing the recipe).

---

## 2026-09-15 — Claude (worker session fleurs-24-asr-frozen-sets) — FLEURS-24 ASR frozen sets built

Context: week 1 Track B, `05-language-ladder.md` §3 / `timeline.md` week 1
("FLEURS-24 ASR frozen sets in melt-eval"). Branch
`claude/fleurs24-asr-frozen-sets` in `melt-eval`,
[PR #10](https://github.com/MELT-proj/eval/pull/10).

Finding / proposal: added `configs/fleurs24-asr-test.yaml` (full FLEURS
`test`, all 24 EU languages) and `configs/fleurs24-asr-dev.yaml` (~100
utterances/language from FLEURS `validation`, for the in-training generative
round). Froze both against the real shar tree: `fleurs24-asr-test` is 19,463
samples / 63.41 h with zero dropped-no-reference cuts; `fleurs24-asr-dev` is
2,400 samples / 7.41 h. Copied to
`/mnt/scratch-artemis/giuseppe/melt-data/eval-sets/{fleurs24-asr-test,fleurs24-asr-dev}/`.

Checked `custom.pnc_text` coverage directly against the shar tree for all 24
languages, both `test` and `validation` (not previously measured at this
granularity): **23/24 are 100% covered; ga (Irish) has 0% on both splits**
and silently falls back to plain, unpunctuated supervision text
(`get_text_from_cut`, `strict=False`, the training repo's own default). Every
other language's FLEURS reference is cased and punctuated; Irish's is not.
Documented in both spec headers rather than worked around -- fixing it is the
training repo's PNC-backfill/`strict_text_field` decision
([[num-tokens-and-pnc-text-semantics]], [[silent-text-field-fallback]]), out
of scope here. Worth remembering when Irish's ladder numbers look
disproportionately bad or good: part of that could be transcript formatting,
not the model.

Also found: the shared artemis dev venv
(`/mnt/scratch-artemis/giuseppe/venvs/melteval`) editable-installs
`melt-proj` from `melt-eval`'s sibling checkout at
`/mnt/home/giuseppe/melt-proj/training`, which is pinned at `74c7892`
(2026-08-19) -- **before** the transformers 5 migration (`a519e4fe`,
2026-09-01, PR #109). Importing `melt.training` there crashes
(`ValueError: mutable default <class 'dict'> for field sub_configs`) against
the venv's transformers 5.16.1. I froze the sets from nyx instead (training
repo's own `.venv`, current `main`, melt-eval on `PYTHONPATH`) rather than
touching that checkout, since it has unrelated uncommitted local changes
(`.github/workflows/ci.yml`, `.gitignore`, `AGENTS.md`, `README.md`) and its
scratch-side sibling (`/mnt/scratch-artemis/giuseppe/melt-proj-src/melt-eval`,
on `claude/air-bench-support`) has unrelated in-progress work from another
session. Did not touch either.

Action needed: PI review/merge PR #10. Before anyone runs `inspect eval`
(the generation step) against these frozen sets on artemis, sync
`/mnt/home/giuseppe/melt-proj/training` to `main` (or otherwise past
`a519e4fe`) -- generation will crash on import otherwise. FLEURS X→en ST set
construction (the other week-1 Track B eval item) is still open.

---

## 2026-09-15 — Claude (worker session mlp-adapter-stack-factor-13e900) — stack_factor PR opened

Context: follow-up to the entry directly below, after the PI merged the
plan folder into `main`.

Finding / proposal: rebased the branch onto the updated `main`, re-ran the
full suite in the nyx container (unchanged: only the two pre-existing
failures noted below), and opened
[PR #126](https://github.com/MELT-proj/training/pull/126) against `main`.

Action needed: PI review/merge.

---

## 2026-09-15 — Claude (worker session mlp-adapter-stack-factor-13e900) — stack_factor implemented for the MLP adapter

Context: week 1 Track B, `01-interface-recipe.md` §4. Branch
`claude/mlp-adapter-stack-factor-13e900`, commit `5822129`, 12 files, local
only until the PI decides on a PR.

Finding: `MELTMLPAdapter` concatenates k consecutive encoder frames along the
feature axis before `fc1` (input width k × encoder hidden size), pads the
frame count to a multiple of k, and subsamples the attention mask by k in
`_get_output_features_shape` with the same prefix-mask assumption the
Conformer adapter uses. New `model.adapter.stack_factor` field, default 1,
byte-identical to the previous adapter. Conformer and Q-Former ignore it.
Wired as a `STACK_FACTOR` campaign axis (env var, `ArmAxes.stack_factor`,
`--model.adapter.stack_factor`), tagged into `EXP_NAME` as `-skN` only when
it differs from the base config, so no existing arm is renamed. Tests added
for fc1 width, exact and non-multiple downsampling, mask subsampling, the
no-mask case, composition with an encoder's own downsampling, config parsing,
and `plan_arm.py` tag composition (its first tests). Full suite in the nyx
container: 452 passed.

Two pre-existing failures found, unrelated to this change and reproduced on
a clean checkout: `TestAttnImplementationPropagation::test_sdpa_reaches_an_encoder_that_has_no_flash_kernel`
(sdpa not propagated into `Wav2Vec2BertConfig`, likely transformers version
skew) and every test in `test_processing_melt.py` (tokenizer fixture's
`add_special_tokens` against Qwen2.5-1.5B). Background tasks queued for both.

Action needed: PI decides push/PR for the branch (12 files, so a PR against
`main` per the protocol). Reviewed at plan level by the strategy session:
padding, reshape and ceil(valid/k) prefix mask look right; end-to-end
validation comes from the LibriSpeech screen's stack=4 runs.

## 2026-09-15 — Claude (Fable 5.1, research-buddy session) — Plan folder created; the August baseline is a failed alignment

Context: three-turn strategy session with the PI on how to organise the
campaign, then two added commitments (24 EU languages, the Fondue run).

Findings:
- MA eval loss 2.6–3.1 and WER > 1.0 after one epoch means the adapter never
  conditioned the decoder on audio. Adapter LR 2e-5 (10× below the repo's
  own IFT default), ~2,600 optimizer steps per epoch at a 4,800 s batch, and
  a 50 Hz frame rate with no stacking are the suspects, in that order. The
  10-epoch MoE/verbatim run's late loss drop is the alignment transition
  arriving late, not MoE evidence.
- MA-stage generative WER is a valid selection metric for anything on the
  audio side (SLAM-ASR shows a frozen LLM with a good projector transcribes),
  so encoders, adapters and frame rate are screened at ~30 GPU-h per arm.
- Qwen3.5-2B is dense with hybrid linear attention, not an MoE. Llama vs
  Qwen is also full vs hybrid attention.
- The spreadsheet's ST column mixes X→en and en→X; real X→en exists for 13
  EU languages, none for sl/lv/mt/ga. Tiers 10/30/100/300/700 h are reached
  in full by 24/17/13/8/8 languages.
- Budget: ~49,700 GPU-h remain until 2026-11-30, no renewal. Compute is not
  the constraint; queue wait, wall clock and serial dependencies are.

Action needed: PI reviews `timeline.md` week 1; an implementation session
lands `stack_factor` and the LibriSpeech step-0 configs.
