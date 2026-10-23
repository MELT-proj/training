# 02 — Backbones: the 2×3 grid and the text prior

**Settled (2026-09-15):** the grid is Llama-3.2-1B, Qwen3.5-2B, EuroLLM-1.7B,
each base and instruct; backbones are judged on the five in-domain languages
*and* FLEURS-24 zero-shot; a text-only prior is measured for every backbone
and language before training and reported against post-training adaptability.
**Open:** the decision rule's weighting between in-domain and zero-shot;
whether a ~4B point joins the grid (decided by step 0b's two size controls,
`01-interface-recipe.md` §2b rule 3).
**Owner:** PI decides the backbone at the week-5 gate.
**Order (2026-09-18):** this section now runs **after** `03-audio-stack.md`,
so the six decoders are compared on a chosen audio stack rather than an
assumed one. If the encoder is worth an order of magnitude, six backbones
behind the wrong one all sit against the same floor and their differences
compress into noise. Rationale in `03-audio-stack.md` §0.

## 1. The grid

| family | base | instruct | facts to report |
|---|---|---|---|
| Llama 3.2 | `meta-llama/Llama-3.2-1B` | `meta-llama/Llama-3.2-1B-Instruct` | 16 layers, hidden 2048, vocab 128K; official coverage of 8 languages; the chat template injects a dated system message |
| Qwen 3.5 | `Qwen/Qwen3.5-2B-Base` | `Qwen/Qwen3.5-2B` | 24 layers, hidden 2048, vocab 248K; **dense**, hybrid attention: 18 Gated-DeltaNet linear-attention layers and 6 full-attention layers in a 3:1 pattern; the chat template opens every assistant turn with an empty think block |
| EuroLLM 1.7B | `utter-project/EuroLLM-1.7B` | `utter-project/EuroLLM-1.7B-Instruct` | trained on all 24 EU languages; the natural fit for the coverage commitment; ids confirmed and staged on MN5 (2026-09-16) |

Every arm: same MA config (ASR-only, 700 h/lang, the recipe from
`01-interface-recipe.md`), same IFT config (ASR+ST, 700 h/lang), same
encoder and adapter, same seeds. Base arms borrow the instruct sibling's chat
template (`chat_template_from`) so the prompt format is identical.

## 2. Confounds to state in the paper

- **Size is not matched.** 1B vs 2B vs 1.7B. A Qwen advantage cannot be
  attributed to the family. Llama-3.2-3B would bracket Qwen from above if a
  size control is wanted; not in the grid today.
- **Qwen3.5-2B is not an MoE.** Its config has no experts. Llama vs Qwen is
  also full attention vs hybrid linear attention, and the linear layers hold
  a fixed-size recurrent state that must summarise the audio prefix. Frame
  stacking matters more for it, not less.
- **Base arms at MA are template-naive.** The frozen base decoder has never
  seen the chat-template tokens the MA stage wraps around the audio, so the
  MA-stage base-vs-instruct contrast includes template familiarity. IFT
  trains the decoder and washes most of it out; report the MA-stage contrast
  with that caveat. Measured directly, not just predicted, by the text
  prior (§3.7): instruct beats base in every one of the three families on
  both tasks, with tokenization held identical, so the gap is isolated to
  template-wrapping — Llama 1.884 vs 2.207 conditioned ASR bits/char,
  Qwen 1.428 vs 1.651, EuroLLM 1.700 vs 1.856.
- **Llama's dated system prompt** and **Qwen's think block** are structural
  input differences; melt-eval renders the generation prompt with the same
  template and `enable_thinking=False`, and its tests assert the eval prompt
  is a strict prefix of the training render.
- **Text ability after IFT.** Full fine-tuning of a 1–2B decoder on 6,500 h
  of speech tasks erodes text ability. Measure it (§3) for every IFT arm;
  it is the real content of the base-vs-instruct and LoRA questions.

## 3. The text prior: "how much of the language does the decoder already know?"

For each of the six backbones and each of the 24 EU languages (plus ru, uk),
on the FLEURS references used by the speech eval (dev subset for the first
pass, §3.6):

| measurement | on what | why |
|---|---|---|
| NLL per token and **bits per character** | ASR transcript, raw text | BPC is tokenizer-independent, so it compares across backbones |
| the same, conditioned on the exact IFT instruction (`Transcribe this audio in {lang}.`) | ASR transcript | matches the eval condition |
| the same on the English ST reference, conditioned on `Translate this audio to English.` | ST reference | the ST prior |
| tokens per word and tokens per character | transcript | fertility: Maltese and Hungarian cost 2–4× more decoder tokens per word than English on English-heavy tokenizers, which changes the `max_tokens` filter, the effective batch in tokens, and learning speed |
| **cascade oracle**: gold transcript as text + the translation instruction → chrF/COMET | ST | the text-only upper bound on translation; separates "cannot hear" from "cannot translate" |

Adaptability is the same backbone's CER (ASR) and chrF/COMET (ST) on the
same references after MA + IFT. The paper figure is BPC (x) against CER (y),
one point per backbone × language, with family markers. The cascade oracle
also runs on the *trained* decoder, so the ST gap decomposes into audio
interface vs translation ability lost or never present.

It lives in melt-eval, as a text-only measurement over the same frozen sets,
so references and prompts are byte-identical to the speech eval. Needs one
GPU for an afternoon on an internal machine; no MN5 time. It is two
mechanisms, not one tool (§3.2); the frozen sets now cover the full
24+ru+uk scope (§3.5), and one prerequisite remains before it can run on
all six backbones — an EuroLLM entry in `DECODER_PROFILES` (§3.1,
`timeline.md` week 1). Design below, checked against melt-eval's actual
code (`~/melt-proj/melt-eval`) and the training repo's `no_audio_floor.py`,
not written from the table alone.

### 3.1 What is scored: bare backbones, not MELT checkpoints

The six backbones are scored as published —
`AutoModelForCausalLM.from_pretrained(hub_id, dtype=bfloat16)` plus their own
`AutoTokenizer` — never wrapped in `MELTForCausalLM`. No `<|audio|>` token is
added, no embedding table is resized, and none of `DECODER_PROFILES`'
eos/pad-token remapping (`plan_arm.py`) applies: that remapping exists for
MELT's own vocabulary extension, and this measurement is specifically the
prior *before* any MELT-specific change touches the model.
`no_audio_floor.py` (Track A, week 1) is the closest existing precedent —
teacher-forced NLL of a frozen decoder with `input_features=None` — but it
scores the decoder *inside* an already-constructed `MELTForCausalLM`; this
tool skips that construction and is simpler for it.

The conditioned rows (§3.3) need a chat template per backbone. Checked
directly rather than assumed — the pattern is not uniform across families,
so guessing it from one family would mislead on the others:

| backbone | `chat_template_config` | base checkpoint's own template |
|---|---|---|
| Llama-3.2-1B(-Instruct) | `llama3` | none — Base borrows Instruct's (`DECODER_PROFILES["meta-llama/Llama-3.2-1B"]["chat_template_from"]`) |
| Qwen3.5-2B(-Base) | `chatml` | **has its own**, byte-identical to Instruct's (`DECODER_PROFILES`, verified 2026-09-02) — no borrowing |
| EuroLLM-1.7B(-Instruct) | `chatml` | none — checked this session (`tokenizer_config.json` on nyx has no `chat_template` key); Instruct's is plain ChatML (`<\|im_start\|>{role}\n...<\|im_end\|>\n`, confirmed directly, not assumed) |

Llama and Qwen come straight from `DECODER_PROFILES`
(`projects/ablation-campaign/plan_arm.py`); EuroLLM has no entry there yet
and needs one (`chatml`, `chat_template_from:
utter-project/EuroLLM-1.7B-Instruct`) — a `timeline.md` item.

**Every row above is read from the checkpoint, never inferred from a
sibling.** Qwen3.5-2B-Base ships its own template, which disproves the "a
base checkpoint never ships one" pattern the Llama row would suggest, and a
wrong template corrupts a whole backbone's rows silently (MELT-proj/training
#58).

### 3.2 Two mechanisms

Rows 1–4 score the probability the model already assigns to a *given*
reference string — teacher forcing, no sampling. Row 5 (cascade oracle)
scores a *generated* completion against a reference. These do not fit the
same code path:

- **Rows 1–4 cannot go through `inspect eval`.** melt-eval's `Task`/`Solver`/
  `Scorer` triad (`tasks.py`, `solver.py`, `scorers.py`) is built around
  `generate()` — sample a completion, score the sampled text. Nothing in
  `scorers.py` reads a logprob off `state.output`; every scorer there
  (`asr_scorer`, `st_scorer`) works from `state.output.completion` text. The
  one existing precedent for teacher-forced scoring in this project,
  `no_audio_floor.py`, is a standalone script for exactly this reason, not an
  inspect task — forcing rows 1–4 into the generate-then-score shape would
  fight the abstraction, not use it. Same shape as that script: load a model,
  render `[user, assistant]`, mask everything but the assistant span, forward
  pass, read `out.loss`.
- **Row 5 fits `inspect eval` exactly as it already exists.** It is
  generation, scored by the existing `st_scorer` (chrF/BLEU, COMET later via
  `melteval rescore`) — `tgt_lang` is already uniform (`en`) across the ST
  frozen set (board entry 2026-09-15), so `st_scorer`'s single-tokenizer
  requirement is satisfied with no change. The only new piece is a solver
  that swaps the audio content for the gold source-language transcript.

Proposed home for rows 1–4: a new `melteval/text_prior.py` plus a
`melteval text-prior` CLI subcommand, extending `cli.py`'s `freeze`/`show`/
`rescore` pattern — inside melt-eval, not a second copy of `no_audio_floor.py`
in the training repo, so a backbone's prior and its post-training
adaptability (§4, §5) come from the same frozen-set manifests and the same
repo. One call per backbone:

```
melteval text-prior <frozen_set_dir> --model <hub_id> \
  --chat-template-config {llama3,chatml} [--chat-template-from <hub_id>] \
  -o results/<backbone>.json
```

Row 5 needs no new provider: these are bare HF causal LMs with no audio path,
so inspect's own `hf` provider (`--model hf/<hub_id>`) generates directly —
`providers/melt.py`'s batching and audio resolution exist for a *speech*
model mid-training and buy nothing here. Add one task, `cascade_st` in
`tasks.py`, and one solver, `cascade_prompt` in `solver.py` (§3.3).

### 3.3 Prompt construction: one render path, three uses

All three prompt-bearing rows (conditioned ASR NLL, conditioned ST NLL,
cascade oracle) are one call to
`melt.training.data.audio.lhotse.helpers.apply_chat_template_to_texts` with
`prompt_template_selection="custom"` and

```python
prompt_template = {
    "asr": "{audio_token} Transcribe this audio in {lang}.",
    "st":  "{audio_token} Translate this audio to {lang}.",
}
```

— the first (canonical) entry of each `TASK_TEMPLATES` pool, pinned rather
than randomly drawn, matching this section's "exact IFT instruction" wording
literally instead of approximately. `audio_token` is what varies between the
three uses:

| row | `audio_token` value | target (assistant turn / thing scored) |
|---|---|---|
| conditioned ASR NLL | `""` — no encoder, no audio, and a placeholder token these tokenizers never learned | FLEURS ASR transcript |
| conditioned ST NLL | `""` | FLEURS-en ST reference |
| cascade oracle | the sample's own source-language transcript (`record.source_text`) | generated text, scored by `st_scorer` |

Substituting the transcript for the placeholder — rather than appending it
before or after the instruction — is what makes "gold transcript as text"
operationally precise: the model reads the same instruction wording as the
real audio eval, with the one substitution that turns "audio it cannot yet
hear" into "text it already knows how to read." It also means one rendering
call serves both mechanisms in §3.2 — the standalone scorer and the
`cascade_prompt` solver differ only in what they pass as `audio_token`.

The **raw** row (unconditioned NLL/BPC) does not go through this at all:
`tokenizer(reference_text, return_tensors="pt")` with each tokenizer's own
default special-token behavior, unmodified — no chat template, no
instruction, every token scored. This is the one row that is a pure
language-model prior rather than an instruction-following prior; keeping it
separate from the conditioned row is what lets §2's "base arms are
template-naive at MA" confound be measured directly, as a number, before any
arm trains — the gap between raw and conditioned BPC for a base backbone is
a prediction of exactly that confound.

Masking for the conditioned rows: render `[user, assistant]` with
`add_generation_prompt=False` (training's convention, not generation's — see
`melteval/prompt.py`'s `apply_generation_format` docstring), tokenize, then
`mask_non_assistant_tokens` against `CHAT_TEMPLATE_CONFIGS[chat_template_config]`'s
`assistant_start`/`assistant_end` markers (§3.1's table) — the same two
training-repo helpers `no_audio_floor.py` already validated end to end.
Boundaries are inclusive (`chat_templates.py`), so the counted span includes
a handful of structural tokens (e.g. `<|eot_id|>`) beyond the reference
text's own tokens — negligible nats, since they are near-deterministic given
the template, and irrelevant to BPC (§3.4 divides by the *character* count
of the raw reference string, not token count) — but it does mean row 4's
tokens-per-word/char is a token or two higher per sample than a bare
tokenizer count of the reference alone would give. Not worth correcting for
six backbones' worth of numbers; worth knowing if a fertility number looks
higher than expected.

### 3.4 Formulas

Per batch `b`, `*ForCausalLM.forward(..., labels=...)` returns a mean loss
over its own non-masked (`!= -100`) positions. Accumulate the way
`no_audio_floor.py` does — weight each batch's mean back up to a sum before
dividing by the grand total, since a straight mean-of-means would weight a
short batch and a long one equally, which is wrong for a corpus-level number:

```
total_nats   = Σ_b  loss_b × valid_b        # valid_b = unmasked target positions in batch b
total_tokens = Σ_b  valid_b
nats_per_token  = total_nats / total_tokens
bits_per_token  = nats_per_token / ln(2)
bits_per_char   = (total_nats / ln 2) / Σ len(reference_string)   # raw string, codepoints
tokens_per_word = total_tokens / Σ len(reference_string.split())
tokens_per_char = total_tokens / Σ len(reference_string)
```

`reference_string` is the untouched frozen-set `target` field — not
WER-normalized, since normalization (lowercasing, punctuation stripping)
would change the character count the "tokenizer-independent" claim rests on,
and FLEURS `pnc_text` casing/punctuation is what IFT actually trains the
decoder to produce.

### 3.5 Inputs: reuse the FLEURS-24 frozen sets

No new freezing for the ASR rows: `fleurs24-asr-{test,dev}` (melt-eval
[PR #10](https://github.com/MELT-proj/eval/pull/10), **merged**) already
carries the transcript in `target`, byte-identical to what the language
ladder scores against. For the ST rows and the cascade oracle:
`fleurs24-st-xen-{test,dev}` ([PR #11](https://github.com/MELT-proj/eval/pull/11),
**merged** 2026-09-16) carries the English reference in `target` and, load-bearing
for the cascade oracle specifically, the source-language transcript in
`source_text` (`source_text_field: custom.pnc_text` in that config) — no new
manifest field needed, `record.source_text` is already read into
`metadata["source_text"]` by `melteval/dataset.py`.

ru/uk are now in all four configs too
([PR #12](https://github.com/MELT-proj/eval/pull/12) ASR,
[PR #13](https://github.com/MELT-proj/eval/pull/13) ST, both open against
`main` as of 2026-09-16) — they were missing when this spec was first
drafted (checked against `data/hours_by_language.csv`, which shows FLEURS
audio exists for both: `asr_fleurs` 8.1 h ru, 9.0 h uk), closing the gap
between this section's stated scope ("24 EU languages (plus ru, uk)") and
what the frozen sets actually covered. Checked directly before adding,
the way `ga`'s gap (next paragraph) was found: both locales carry
`custom.pnc_text` on `test` and `validation` (unlike `ga`), and the ST
`reference_map` — built from all `en_us` sentence ids, not
per-source-language — covers ru/uk's ids with zero drops, same as the
other 23. Test grows to 20,988 ASR / 20,341 ST samples (26 / 25 languages);
dev to 2,600 ASR / 2,500 ST (100 per language throughout). Not yet
redeployed to the production frozen-set copies on artemis scratch — whoever
first runs this tool for real should re-run `melteval freeze` and copy over
rather than trust an already-materialized directory.

**`ga` has no `pnc_text`** (documented already, `fleurs24-asr-test.yaml`
header): its `target` falls back to lowercase, unpunctuated supervision
text. Its raw/conditioned BPC is therefore not on the same textual basis
as the other languages — expect it to look different and do not read that
as a backbone signal.

The manifest schema is shared between `dev` and `test`, so running on `test`
later needs no rework, only a different `--frozen-set` path.

### 3.6 Scale and where it lives

Dev subsets (`fleurs24-asr-dev`: 2,600 samples / 26 langs; `fleurs24-st-xen-dev`:
2,500 / 25 langs) are the right size for this tool's first pass: six backbones
× roughly 5,100 forward-pass samples (raw + conditioned, ASR + ST) plus
generation for the cascade oracle is the "one GPU afternoon, no MN5 time"
this section already promises, and dev is roughly 8× cheaper than test for
the same per-language coverage. Re-run on `fleurs24-asr-test`/
`fleurs24-st-xen-test` for the number that goes in the paper figure ("BPC (x)
against CER (y)", above) once the backbone decision (§4) narrows six
backbones to one — the manifest schema does not change between them, only
the sample count. Lives in melt-eval, run on an internal GPU (artemis, via
`sbatch`, per its own rules) — not MN5, since this measurement has nothing to
do with a checkpoint the campaign trained.

### 3.7 Built and tested: all six backbones, dev sets (rows 1–4 only)

Built 2026-09-16 as `melteval text-prior`
([melt-eval PR #14](https://github.com/MELT-proj/eval/pull/14)), per
§3.1–§3.4 above. The cascade oracle (row 5) is not built yet.

Ran on artemis (`dionysus`, h100/gpu-h100) against all six backbones (§1)
on the production `fleurs24-asr-dev` (24 languages, 2,400 samples — ru/uk
not deployed to this copy yet, §3.5) and `fleurs24-st-xen-dev` (23
locales, 2,300 samples). All twelve jobs completed cleanly.

**Overall** (all languages pooled; lower bits/char = the backbone already
knows more of the language):

| backbone | ASR raw | ASR cond | ST raw | ST cond |
|---|---|---|---|---|
| Llama-3.2-1B (base) | 1.571 | 2.207 | 1.007 | 1.679 |
| Llama-3.2-1B-Instruct | 1.940 | 1.884 | 1.094 | 1.250 |
| Qwen3.5-2B-Base | 1.324 | 1.651 | 0.959 | 1.464 |
| Qwen3.5-2B (instruct) | 1.390 | 1.428 | 0.986 | 1.158 |
| EuroLLM-1.7B (base) | 1.190 | 1.856 | 1.023 | 1.801 |
| EuroLLM-1.7B-Instruct | 1.200 | 1.700 | 1.039 | 1.572 |

Two findings worth carrying into the paper:

1. **Instruct beats base, in every family, on both tasks, no exception.**
   Llama 1.884 vs 2.207, Qwen 1.428 vs 1.651, EuroLLM 1.700 vs 1.856 (ASR
   conditioned; ST tells the same story). This is §2's "base arms at MA
   are template-naive" confound, now a measured number instead of a
   prediction — the base decoder is worse specifically at the
   conditioned-on-a-chat-template task, before any training touches it.
   Base and instruct within a family tokenize identically (shared
   vocabulary — confirmed directly, their tokens/word figures match to
   three decimals), so the gap isolated by the raw-vs-conditioned split
   (§3.3) is purely the chat-template-wrapping effect, not a fertility
   artifact.
2. **Qwen3.5 leads overall; EuroLLM specifically wins on the hardest
   language.** Ranked by ASR conditioned BPC: Qwen-Ins (1.428) <
   Qwen-Base (1.651) < EuroLLM-Ins (1.700) < EuroLLM-Base (1.856) <
   Llama-Ins (1.884) < Llama-Base (2.207). Maltese (mt), the hardest
   language in this set for every backbone, inverts the family ranking:
   EuroLLM-Instruct (1.911) beats Qwen-Instruct (2.002) beats
   Llama-Instruct (3.113) — EuroLLM's EU-specific training shows up
   exactly where it should, even though its broader-aggregate number
   trails Qwen's.

Selected per-language ASR conditioned BPC (all 24 in the JSON; en/mt
bracket the range for every backbone):

| backbone | en | hu | mt | ga |
|---|---|---|---|---|
| Llama-3.2-1B (base) | 1.628 | 2.146 | 3.013 | 2.990 |
| Llama-3.2-1B-Instruct | 1.260 | 1.736 | 3.113 | 2.871 |
| Qwen3.5-2B-Base | 1.421 | 1.695 | 2.125 | 2.355 |
| Qwen3.5-2B (instruct) | 1.141 | 1.472 | 2.002 | 2.265 |
| EuroLLM-1.7B (base) | 1.761 | 1.925 | 1.976 | 2.190 |
| EuroLLM-1.7B-Instruct | 1.510 | 1.768 | 1.911 | 2.100 |

Full per-language JSON for all twelve runs (6 backbones × 2 tasks):
`/mnt/scratch-artemis/giuseppe/melt-data/text-prior/<tag>-{asr,st}-dev.json`
on artemis scratch (not committed — see the board entry for the tag
naming and the retry/infra details).

Not yet done: ru/uk (frozen sets not redeployed to production, §3.5), the
`-test` splits, and the cascade oracle (row 5).

Not yet run: the other five backbones, the cascade oracle, and anything
on `fleurs24-*-test` (only `-dev` so far).

## 4. Decision rule for the backbone (week-5 gate)

Ranked, in this order, with the noise estimate from the seed replicates:

1. mean CER over the five in-domain languages after IFT, and mean FLEURS-24
   zero-shot CER, both must be within noise of the best or better;
2. X→en chrF/COMET in-domain, and the cascade-oracle gap;
3. text-ability retention after IFT;
4. Fondue feasibility: measured IFT throughput at 2 nodes × 4 GPUs (Qwen's
   is unverified as of 2026-09-15 and is a week-1 measurement). A backbone
   whose IFT cannot reach 500K h before 11-30 either gets more nodes, a
   smaller Fondue, or is not the Fondue backbone.

## 5. Results

| arm | MA-stage WER (5-lang mean, %) | FLEURS-24 CER (MA) | IFT CER (5-lang) | FLEURS-24 CER (IFT) | sel (ASR) | sel (ST) | sel (ASR+ST)³ | text retention | notes |
|---|---|---|---|---|---|---|---|---|---|
| llama1b-base | runaway¹ | runaway¹ | .0500 | .3571 | 0.0614 | 0.4009 | 0.2311 | −0.058 / −0.315 † | MA unusable; IFT recovers it |
| llama1b-ins | 11.52 | .6680 | .0477 | .4448 | 0.0851 | 0.3956 | 0.2403 | +0.677 / +0.123 |  |
| qwen35-2b-base | 10.28 | .6191 | **.0431** | .4350 | 0.0858 | 0.3780 | 0.2319 | +0.628 / +0.009 | 2 nodes, bd 30 |
| qwen35-2b-ins | 9.87 | .6547 | .0473 | .4219 | 0.1067 | 0.3811 | 0.2439 | +1.006 / +0.332 | 2 nodes, bd 30 |
| eurollm1.7b-base | not scored² | — | not entered⁴ | .365 | 0.0646 | 0.3849 | 0.2247 | −0.365 / −0.455 † | resumed IFT, `checkpoint-6310` |
| eurollm1.7b-ins | 10.88 | .6661 | not entered⁴ | **.335** | 0.0563 | 0.3800 | 0.2182 | −0.264 / −0.236 | resumed IFT, `checkpoint-6310` |

Read the columns as follows (all numbers from melt-eval on the saved checkpoint,
never from the in-training history). The columns left of `sel (ASR)` describe
the runs; the comparison is the three `sel` columns, which score only the
languages and directions the IFT mix trains and the related ones the metric
already weights.

- **WER and CER means** are unweighted means over the five training languages, on
  the frozen selection sets (600 in-domain clips per language), not the medians the
  selection metric uses. WER and CER differ by the reference length; the selection
  metric's per-language quantity is CER.
- **FLEURS-24 CER** is the mean over the 24 EU languages of `fleurs26-dev200`, so
  ru and uk are excluded here; including them raises every IFT figure by 0.055–0.073.
- **sel (ASR)** is the selection metric of `selection-metric/README.md`, unchanged:
  `(1.0·med(ID) + 1.5·med(OOD-train) + 0.6·med(OOD-related)) / 3.1` over clipped
  per-language CER on the frozen selection sets. Post-IFT for all six arms. Raw
  output (per-language CER, group medians, runaways) in
  `selection-metric/scores/backbones-ma-and-ift-2026-10-22.txt`.
- **sel (ST)** and **sel (ASR+ST)** extend it with speech translation over the same
  language groups; defined in "Selection score with ST" below, next to the
  per-direction chrF they are built from.
- **Text retention** is "ASR cond Δ / ST cond Δ" in bits per character,
  post-IFT minus pre-IFT (positive = less retained), measured with the §3.7
  instrument on `checkpoint-6310` after exporting the decoder to a bare causal
  LM (`text_prior_export.py`). Full tables, per-language breakdown, same-day
  controls and job IDs in [`text-retention.md`](text-retention.md). EuroLLM was
  measured last (job 47880299, 2026-10-23). eurollm1.7b-ins is the one
  instruct arm with negative conditioned deltas (the other two are +0.123 to
  +1.006), and the cause was not investigated; eurollm1.7b-base is negative
  like llama1b-base (†). The raw deltas (+0.250 / +0.186 ins, +0.261 / +0.214
  base) are positive like everyone else's, and much smaller.
- † The base arms' conditioned delta is confounded by learning the chat template
  (their pre-IFT conditioned number is dominated by template unfamiliarity,
  §2). Their raw deltas (+0.671 ASR / +0.211 ST for llama1b-base) are the clean
  retention numbers.
- ¹ 2,555 of 3,000 in-domain generations run away (per-sample CER > 1): the Llama-Base
  MA weights do not transcribe, so both MA columns are meaningless for that arm.
- ² EuroLLM-Base MA ran away on 4–7 of the WER scale and was never scored; the IFT
  stage starts from a staged checkpoint-47730 of that arm and repairs its ChatML rows.
- ⁴ Not entered: the EuroLLM IFT CER (5-lang) and the per-language ASR and X→en
  cells were not part of the hand-over this table was filled from (X→en logs:
  `/gpfs/scratch/epor48/eval-logs/ift-st/ift-700-backbone-eurollm-{ins,base}/`).
  The FLEURS-24 CER (IFT) values .335 / .365 are the hand-over's 33.5 / 36.5.

### Selection score with ST

³ This section defines `sel (ST)` and `sel (ASR+ST)`. They compare the arms
only on the languages and directions the IFT mix actually trains: the
selection metric's own groups, with speech translation added on the same
languages.

- **ASR — unchanged** (the pre-registered metric): ID `en,de,es,fr,it`
  in-domain; OOD-train `en,de,es,fr,it` on FLEURS; OOD-related
  `pt,ro,nl,da,sv` on FLEURS; clipped per-language CER, group medians,
  weights 1.0 / 1.5 / 0.6.
- **ST**: per direction `s_l = 1 − chrF_l/100`, on X→en, lowercase plain
  chrF, 100 clips per source:
  - ST-train `de,es,fr,it` → en (the four directions the mix trains, YODAS
    Granary `ast`, 4.7–10.5% of group draws each), weight **1.5**;
  - ST-related `pt,ro,nl,da,sv` → en (zero-shot, same genealogy rule as the
    ASR group), weight **0.6**.

  `sel (ST) = (1.5·med(ST-train) + 0.6·med(ST-related)) / 2.1`.
- **Headline**: `sel (ASR+ST) = (sel (ASR) + sel (ST)) / 2`, lower is better.

The two halves are weighted equally because the ASR score spans three groups
and ST two: adding the ST groups to the same sum would weight ASR 60/40 on no
principled basis. Folding them in that way instead —
`(3.1·sel(ASR) + 1.5·ST-train + 0.6·ST-related)/5.2` — gives the same order
(.1985 < .2038 < .2105 < .2175), so the choice is not load-bearing here.
Neither is the related group: restricting ST to the four trained directions
also leaves the order, at .2271 / .2276 / .2382 / .2406.

| arm | de→en | es→en | fr→en | it→en | med (trained) | pt→en | ro→en | nl→en | da→en | sv→en | med (related) | en→de | en→X, 23 targets |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| llama1b-base | 63.1 | 56.7 | 62.1 | 59.3 | 60.7 | 65.0 | 57.5 | 53.1 | 57.9 | 61.7 | 57.9 | 56.5 | 23.9 |
| llama1b-ins | 63.1 | 56.9 | 61.5 | 60.2 | 60.9 | 65.7 | 58.6 | 53.2 | 59.4 | 61.3 | 59.4 | 57.3 | 24.5 |
| qwen35-2b-base | 65.2 | 58.5 | 64.2 | 61.9 | 63.1 | 66.0 | 60.0 | 53.7 | 59.3 | 62.0 | 60.0 | 61.5 | 25.9 |
| qwen35-2b-ins | 65.3 | 58.7 | 63.4 | 61.7 | 62.5 | 65.5 | 59.3 | 53.4 | 60.2 | 62.1 | 60.2 | 59.5 | 25.1 |
| eurollm1.7b-base | —⁴ | —⁴ | —⁴ | —⁴ | —⁴ | —⁴ | —⁴ | —⁴ | —⁴ | —⁴ | —⁴ | 59.5 | 37.6 |
| eurollm1.7b-ins | —⁴ | —⁴ | —⁴ | —⁴ | —⁴ | —⁴ | —⁴ | —⁴ | —⁴ | —⁴ | —⁴ | 59.9 | 42.9 |

Lowercase plain chrF throughout (`melteval st-metrics --lowercase`); cased
macro-means run 3.2–3.3 points lower and are in
`selection-metric/scores/backbones-st-xen-2026-10-22.txt`. The median of
`1 − chrF/100` is `1 − med(chrF)/100`, so the score can be recomputed from
this table to within rounding:
`sel (ST) = 1 − (1.5·med(trained) + 0.6·med(related))/210`.

- **What the score resolves.** The leading pair (llama1b-base .2311,
  qwen35-2b-base .2319) is a tie at 0.0008, and the pair swaps with
  llama1b-ins relative to the ASR-only score (0.0851 vs 0.0858 for
  llama1b-ins and qwen35-2b-base; the ST part, 0.3956 vs 0.3780, more than
  reverses it). Both gaps are inside the 0.002–0.010 eval noise the metric's
  README records, and the ST directions add 100-clip noise on top of it. The
  only gaps that clear its 0.010 upper bound are the leading pair against
  qwen35-2b-ins (0.0128 and 0.0120).
- **No ST in-domain twin.** `en` has no X→en direction, so ST-train is four
  languages where ASR's OOD-train is five, and no in-domain X→en set exists
  for these arms: the ST half has no ID group to weight at 1.0.
- **en→de is reported, not scored.** It is the only trained en→X direction
  (CoVoST2, 429.9 h, 9.6% of group draws). Scoring it would count German
  twice in a group of five and rest the result on a single 100-clip
  direction; folding it in as a fifth ST-train direction leaves the order
  unchanged (the leading pair swaps by 0.00004). Every other en→X target is
  zero-shot — the older four arms mostly answer in German or English rather
  than the language asked for, and EuroLLM mostly does not ("What the en→X
  column measures", below) — so the 23-target column is there for continuity,
  not as a comparison of these arms. That set exists only on artemis (MN5's eval copy
  has neither the spec nor a frozen set), so those runs were made there,
  jobs 338939–338942.
- **Exploratory ordering.** The metric was frozen 2026-09-23, before any of
  these numbers existed, precisely so a ranking could not be tuned to them.
  This extension was defined 2026-10-23, after four arms had been scored. The
  ASR-only score is the pre-registered one and is kept in the table beside
  the extended one so the difference stays visible.
- The Tier-1 joint score of the MA-utility study (`joint =
  (100 − asr24 + X→en + en→X)/3`, `projects/self-distill/eval/tier1_joint.py`,
  adopted 2026-10-21) is not used here: it averages over all 24 ASR languages
  and all 23 directions each way, and these arms train X→en for four sources
  and en→X for one, so most of its mass is zero-shot.

EuroLLM's rows were added 2026-10-23 once its resumed IFT reached
`checkpoint-6310`: sel (ASR) 0.0563 / 0.0646, sel (ST) 0.3800 / 0.3849, joint
0.2182 / 0.2247 (ins / base). These figures were handed to this document as
established under the definitions above; they were not re-derived here, and
the per-direction X→en cells behind them are not entered (⁴).

### Per language

Languages on rows, arms on columns: **L-B** Llama-3.2-1B (base), **L-I**
Llama-3.2-1B-Instruct, **Q-B** Qwen3.5-2B-Base, **Q-I** Qwen3.5-2B (instruct).
EuroLLM is in the en→X table only (**E-B**, **E-I**); its ASR and X→en
per-language cells are not entered here (⁴).

These tables are context, not the comparison. Only en, de, es, fr, it (ID and
OOD-train) and pt, ro, nl, da, sv (OOD-related) enter the scores above; the
remaining rows are reported so a reader can see what the untrained languages
do, and are not part of any ranking.

**ASR — FLEURS-24 dev CER (%), 200 clips per language.** These are the eval
logs' own per-language CERs, unclipped: a language where the model runs away
sits above 100% instead of being capped, which is what makes the `mean` row
equal to the FLEURS-24 CER column above. Bold marks the best arm in the row.

| lang | L-B | L-I | Q-B | Q-I |
|---|---|---|---|---|
| en | **3.3** | 3.7 | 4.2 | 4.2 |
| de | 3.0 | 2.7 | **2.4** | 2.9 |
| es | 1.7 | 1.8 | **1.6** | 1.7 |
| fr | 3.2 | 3.1 | **2.4** | 2.6 |
| it | 1.6 | 1.6 | **1.5** | 1.6 |
| pt | **8.6** | 9.6 | 11.4 | 18.4 |
| ro | **13.0** | 18.5 | 31.0 | 43.0 |
| nl | **24.7** | 29.2 | 40.9 | 37.5 |
| da | **22.3** | 37.3 | 35.3 | 50.1 |
| sv | **15.7** | 30.1 | 24.5 | 40.8 |
| cs | 50.7 | 58.3 | 58.2 | **48.8** |
| sk | 40.9 | 53.9 | 35.3 | **32.9** |
| pl | **32.8** | 66.1 | 76.8 | 65.5 |
| hr | **16.8** | 31.6 | 32.4 | 31.2 |
| sl | **21.4** | 35.0 | 42.6 | 27.9 |
| hu | **50.4** | 64.2 | 84.3 | 75.0 |
| fi | **33.0** | 49.9 | 40.6 | 55.6 |
| et | 18.8 | 30.2 | 34.5 | **17.5** |
| lt | **38.2** | 49.1 | 51.3 | 41.8 |
| lv | 34.7 | 35.8 | 34.5 | **26.2** |
| ga | 94.1 | 93.7 | 93.3 | **92.2** |
| mt | **68.6** | 73.7 | 81.4 | 74.4 |
| bg | **105.8** | 146.3 | 106.7 | 110.7 |
| el | 153.6 | 142.3 | 116.9 | **110.1** |
| **mean** | 35.7 | 44.5 | 43.5 | 42.2 |

**ST X→en — lowercase chrF, 100 clips per source.** The rows that enter
`sel (ST)` are de, es, fr, it (trained) and pt, ro, nl, da, sv (related).
`ga` is the outlier for every arm (18–19 against 39–66 elsewhere): the
outputs are degenerate English repetitions ("...the church of the church of
the church..."), not translations, so that row says nothing about the arms'
relative quality.

| src | L-B | L-I | Q-B | Q-I |
|---|---|---|---|---|
| de | 63.1 | 63.1 | 65.2 | **65.3** |
| es | 56.7 | 56.9 | 58.5 | **58.7** |
| fr | 62.1 | 61.5 | **64.2** | 63.4 |
| it | 59.3 | 60.2 | **61.9** | 61.7 |
| pt | 65.0 | 65.7 | **66.0** | 65.5 |
| ro | 57.5 | 58.6 | **60.0** | 59.3 |
| nl | 53.1 | 53.2 | **53.7** | 53.4 |
| da | 57.9 | 59.4 | 59.3 | **60.2** |
| sv | 61.7 | 61.3 | 62.0 | **62.1** |
| cs | 52.7 | 53.7 | **53.8** | 53.6 |
| sk | 54.0 | **54.6** | 53.6 | 53.3 |
| pl | **51.4** | 51.2 | 50.2 | 50.2 |
| hr | 52.6 | 52.9 | **54.2** | 52.8 |
| sl | 44.8 | **46.1** | 44.3 | 44.9 |
| hu | 43.5 | **44.5** | 41.0 | 39.7 |
| fi | 44.2 | **45.2** | 40.0 | 40.8 |
| et | 39.7 | **42.5** | 37.0 | 39.4 |
| lt | **40.5** | 39.1 | 37.3 | 38.3 |
| lv | **40.3** | 39.5 | 37.8 | 38.4 |
| ga | 18.3 | 18.5 | **18.9** | 15.8 |
| mt | 39.0 | 39.7 | **39.8** | 38.5 |
| bg | 54.4 | 52.7 | **55.0** | 54.7 |
| el | **49.0** | 48.8 | 47.5 | 47.4 |
| **mean** | 50.5 | 50.8 | 50.5 | 50.3 |

**ST en→X — lowercase chrF, 100 clips per target.** **E-B** EuroLLM-1.7B
(base), **E-I** EuroLLM-1.7B-Instruct, added 2026-10-23; the other four columns
are unchanged. Only `de` was trained: `build_campaign_config.py` keeps a single
en→de probe (CoVoST2, 429.9 h — 9.6% of the mixture's group draws) because that
is the only en→X direction among the five training languages; the CoVoST2 en→X
sets staged for et/lv/sl/sv are outside the grid and are not in the render.
Every other target is zero-shot. The older four arms mostly answer in German
(the trained direction) or in English; "What the en→X column measures" below
quantifies it. The near-zero bg/el cells are non-Latin targets answered in a
Latin script, where chrF has no n-gram overlap to find.

| tgt | L-B | L-I | Q-B | Q-I | E-B | E-I |
|---|---|---|---|---|---|---|
| de | 56.5 | 57.3 | **61.5** | 59.5 | 59.5 | 59.9 |
| es | 44.6 | 45.3 | 50.4 | 49.9 | 50.3 | **51.2** |
| fr | 45.2 | 49.8 | 57.4 | 57.2 | 60.3 | 60.3 |
| it | 41.4 | 41.7 | 48.4 | 48.8 | **50.9** | 50.7 |
| pt | 41.1 | 46.3 | 48.3 | 46.6 | 56.0 | **56.4** |
| ro | 24.7 | 30.7 | 27.5 | 23.0 | 44.4 | **53.4** |
| nl | 26.6 | 27.5 | 27.1 | 26.9 | **47.8** | 46.4 |
| da | 27.0 | 27.3 | 27.4 | 27.4 | **55.9** | 54.8 |
| sv | 26.8 | 26.5 | 26.9 | 26.7 | **55.7** | 55.1 |
| cs | 17.1 | 17.1 | 18.5 | 17.2 | 42.9 | **44.9** |
| sk | 17.8 | 17.5 | 19.2 | 17.8 | 35.2 | **47.6** |
| pl | 17.7 | 16.7 | 17.3 | 16.9 | **42.4** | 42.2 |
| hr | 19.7 | 18.8 | 22.2 | 19.4 | 19.2 | **35.6** |
| sl | 18.7 | 18.4 | 21.1 | 21.2 | 38.0 | **45.0** |
| hu | 15.7 | 15.4 | 15.6 | 15.9 | 39.1 | **39.3** |
| fi | 17.6 | 17.3 | 17.6 | 17.6 | 44.5 | **44.6** |
| et | 20.4 | 20.5 | 18.9 | 18.3 | 42.1 | **43.6** |
| lt | 17.2 | 16.5 | 15.7 | 14.0 | 19.0 | **43.6** |
| lv | 16.8 | 15.8 | 16.2 | 16.2 | 23.4 | **41.6** |
| ga | 16.5 | 16.3 | 16.7 | 15.8 | 17.0 | **19.5** |
| mt | 18.6 | 18.5 | 19.5 | 19.9 | 19.2 | 19.9 |
| bg | 0.7 | 0.7 | 0.7 | 0.7 | 0.6 | **17.6** |
| el | 0.6 | 0.6 | 0.6 | 0.5 | 0.8 | **13.7** |
| **mean** | 23.9 | 24.5 | 25.9 | 25.1 | 37.6 | 42.9 |

### What the en→X column measures

Added 2026-10-23 after EuroLLM scored 42.9 / 37.6 on the 23-target column
against 23.9–25.9 for the other four. Two explanations were put to the data:
(H1) the instruct arms prepend boilerplate ("Sure, here is the translation:")
that chrF charges as errors; (H2) EuroLLM translates better out of English.
All numbers below are from the same six logs the column is computed from
(`/mnt/scratch-artemis/giuseppe/melt-data/eval-logs/ift-st-enx/`, 2,300
samples each, `--no-score` runs), re-derived with a local chrF that reproduces
all six published macros to the first decimal.

**H1, boilerplate: not present, so no normalisation was applied.** The
prompt is the bare tag (`[st es 7.9s]`) and every arm emits a single line of
text (lowercase, apart from 54 EuroLLM samples with a capitalised proper noun),
13,800 generations in all: none empty, none multi-line, none
carrying an intro sentence or a language label. Leading "clause:" openings
exist in 0.7–1.4% of generations (15–32 per arm) and every one inspected is
sentence-internal ("miller disse: …"); the reference texts have that shape at
a comparable rate (40 of 2,300, 1.7%). A deliberately over-eager rule (strip any
leading clause up to a colon, no phrase list) lowers every arm by
0.05–0.08 chrF and changes no rank. A stock-phrase search hit 2–10 samples
per arm, all of them content words ("aquí" inside a Spanish translation, "it
certainly wasn't fair"), none genuine. Paired bootstrap over clips (300
resamples): ins − base +5.36 [+4.85, +5.84]; EuroLLM-base − qwen35-2b-base
+11.71 [+11.18, +12.18]. The order of the older four is unchanged by the strip
rule and their adjacent gaps are 0.6–0.75 chrF; the intervals resample clips
only.

**What the gap is made of: whether the output is in the language asked for.**
Each generation was labelled with a character-n-gram language identifier
trained on the FLEURS references (5-fold accuracy 98.5%; the weak spots are
cs/sk/sl/hr, 92–94%). Share of the 2,300 samples:

| arm | asked language | German | English | other wrong language | degenerate |
|---|---|---|---|---|---|
| llama1b-base | 21.2% | 58.2% | 19.7% | 0.5% | 0.4% |
| llama1b-ins | 24.0% | 64.6% | 9.3% | 0.9% | 1.2% |
| qwen35-2b-base | 31.5% | 32.3% | 24.4% | 6.8% | 4.9% |
| qwen35-2b-ins | 25.1% | 51.0% | 12.0% | 8.0% | 3.8% |
| eurollm-base | 66.7% | 0.8% | 32.1% | 0.0% | 0.3% |
| eurollm-ins | 83.5% | 2.3% | 13.5% | 0.2% | 0.5% |

("Degenerate" is a compression-ratio or length-blowup rule: runs like
"je je je …".) The four older arms produce the asked language mainly for the
targets whose language the mix trains as speech output or that sit next to it
(de, es, fr, it, pt); EuroLLM does it for most of the other 18 as well
(`eurollm-ins`: 92–100% on cs, da, et, fi, hu, lt, lv, nl, pl, pt, ro, sk, sl,
sv; 80% hr; 33% bg, 25% el, 16% ga, 3% mt).

Macro chrF by slice shows where the 17 points sit:

| arm | de | de, es, fr, it, pt (5) | other 18 targets |
|---|---|---|---|
| llama1b-base | 56.5 | 45.7 | 17.8 |
| llama1b-ins | 57.3 | 48.1 | 17.9 |
| qwen35-2b-base | 61.5 | 53.2 | 18.3 |
| qwen35-2b-ins | 59.5 | 52.4 | 17.5 |
| eurollm-base | 59.5 | 55.4 | 32.6 |
| eurollm-ins | 59.9 | 55.7 | 39.4 |

On the trained direction the six arms are within 5 points and EuroLLM does
not lead. Where every arm writes the asked language, EuroLLM's lead over the
Qwen arms is about 2–3 chrF, over Llama about 8–10. On the other 18 targets the four
older arms sit at 17.5–18.3, which is what a faithful output in the wrong
language scores against the reference (Latin-script n-gram overlap), and
EuroLLM sits at 32.6 and 39.4.

**H2, translation or fluent nonsense: translation, of mixed fidelity.**
Content survives the wrong language. The older arms' German answers to
non-German targets score 56.6–61.5 chrF against the German reference of the
same source sentence, and their English answers 79.3–86.1 against the source
transcript (EuroLLM-base 88.4: it transcribes). So the audio is understood and
the language tag is not obeyed. When the language is right, sentence chrF
averages 43.7–46.7 for the older arms and 48.6–49.2 for EuroLLM. Aligned
clips (all six arms, `en→hr` unless noted; sample ids
12-000022 and 12-000024 for hr, 01-000010 and 01-000011 for fr):

- Source "The East African islands are in the Indian Ocean off the eastern
  coast of Africa."; reference "Istočnoafrički otoci smješteni su u Indijskom
  oceanu, uz istočnu obalu Afrike." llama1b-base, llama1b-ins, qwen35-2b-base
  and qwen35-2b-ins: "die ostafrikanischen inseln liegen im indischen ozean
  vor der ostküste afrikas." eurollm-base: "the east african islands are in the
  indian ocean off the east coast of africa." eurollm-ins: "ostale afričke
  otoke nalaze se u indijskom oceanu, u blizini istočnog kontinenta." — Croatian
  and on topic, with errors ("ostale", "istočnog kontinenta").
- Same target, a longer clip about the Portuguese rebuilding a town:
  llama1b-ins "portugals je je je je …" and qwen35-2b-base "portugals uni je
  uni je …" run away; eurollm-ins writes Croatian that drifts into invention
  ("… opustošen i prebačen u novo ime u potragu za novim gradom u 17. stol…").
  Fluent-but-unrelated text does occur for EuroLLM on the hard languages; it is
  the minority outcome there.
- `fr`: on the clips read, all six arms translate. EuroLLM renders "tear gas" as "gaz
  lacrymogène" and "land of Canaan" as "pays de Canaan" where the others leave
  "tear gas" / "land of" in English. One
  eurollm-ins sample is English-contaminated: "free more bombs explose in
  government buildings in a period of two hours."

**What this shows and does not show.** The column is dominated by whether an arm
writes the asked language on targets the mix does not train. On the trained
direction the six arms span 56.5–61.5 (en→de: qwen35-2b-base 61.5 highest,
EuroLLM-ins 59.9, EuroLLM-base 59.5). It does not show why EuroLLM follows the
tag: a stronger multilingual decoder prior is consistent with the data (it
follows the tag for languages the mix never trains as speech output) but was not
isolated here. It says nothing about X→en. What to conclude from it belongs to
the PI.

Method and caveats: language labels come from a classifier built here, not from
a library; near-misses inside the cs/sk/sl/hr group (6–8%) are possible, and it
counts garbled-but-Croatian-looking text as "asked language". The bootstrap
resamples clips only (not systems or seeds), and the same source sentences recur
across targets. The analysis scripts and the labelled dump live in the session
scratch directory and are not committed.
