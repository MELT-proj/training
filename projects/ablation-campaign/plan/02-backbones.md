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

| arm | MA-stage WER (5-lang mean, %) | FLEURS-24 CER (MA) | IFT CER (5-lang) | FLEURS-24 CER (IFT) | X→en chrF | en→X chrF | joint³ | text retention | notes |
|---|---|---|---|---|---|---|---|---|---|
| llama1b-base | runaway¹ | runaway¹ | .0500 | **.3571** | 47.3 | 22.5 | 51.7 | −0.058 / −0.315 † | MA unusable; IFT recovers it (selection .0614) |
| llama1b-ins | 11.52 | .6680 | .0477 | .4448 | 47.6 | 23.1 | 50.4 | +0.677 / +0.123 | selection .0850 |
| qwen35-2b-base | 10.28 | .6191 | **.0431** | .4350 | 47.2 | 24.4 | 50.6 | +0.628 / +0.009 | selection .0858; 2 nodes, bd 30 |
| qwen35-2b-ins | 9.87 | .6547 | .0473 | .4219 | 47.1 | 23.7 | 50.3 | +1.006 / +0.332 | selection .1066; 2 nodes, bd 30 |
| eurollm1.7b-base | not scored² | — | pending | pending | — | — | pending | pending | IFT resuming (47783769) from step 3,608 |
| eurollm1.7b-ins | 10.88 | .6661 | pending | pending | — | — | pending | pending | selection .1407 (MA); IFT resuming (47783768) |

Read the columns as follows (all numbers from melt-eval on the saved checkpoint,
never from the in-training history):

- **WER and CER means** are unweighted means over the five training languages, on
  the frozen selection sets (600 in-domain clips per language), not the medians the
  selection metric uses. WER and CER differ by the reference length; the selection
  metric's per-language quantity is CER.
- **FLEURS-24 CER** is the mean over the 24 EU languages of `fleurs26-dev200`, so
  ru and uk are excluded here; including them raises every IFT figure by 0.055–0.073.
- **X→en chrF** is cased, macro-mean over the 23 directions of `fleurs24-st-xen-dev`
  (100 clips each). Lowercase chrF++ runs 48.2–48.6 for all four scored arms; the
  full BLEU/spBLEU/chrF/chrF++ rows are in
  `selection-metric/scores/backbones-st-xen-2026-10-22.txt`.
- **en→X chrF** is cased, macro-mean over the 23 targets of `fleurs24-st-enx-dev`
  (100 clips each). That set exists only on artemis — MN5's eval copy has neither
  the spec nor a frozen set — so these four runs were made there (jobs 338939–338942);
  the full rows are in `selection-metric/scores/backbones-st-enx-2026-10-22.txt`.
  Lowercase values and the per-target rows are in "Per language" below.
- **Selection scores** (a different, median-based aggregate): .0614, .0850, .0858,
  .1066 for the four scored arms, .1478 / .1311 / .1216 / .1407 at MA; raw output
  in `selection-metric/scores/backbones-ma-and-ift-2026-10-22.txt`.
- **Text retention** is "ASR cond Δ / ST cond Δ" in bits per character,
  post-IFT minus pre-IFT (positive = less retained), measured with the §3.7
  instrument on `checkpoint-6310` after exporting the decoder to a bare causal
  LM (`text_prior_export.py`). Full tables, per-language breakdown, same-day
  controls and job IDs in [`text-retention.md`](text-retention.md). EuroLLM is
  pending: its resumed runs are queued and only step 3,608 exists.
- † The base arms' conditioned delta is confounded by learning the chat template
  (their pre-IFT conditioned number is dominated by template unfamiliarity,
  §2). Their raw deltas (+0.671 ASR / +0.211 ST for llama1b-base) are the clean
  retention numbers.
- ¹ 2,555 of 3,000 in-domain generations run away (per-sample CER > 1): the Llama-Base
  MA weights do not transcribe, so both MA columns are meaningless for that arm.
- ² EuroLLM-Base MA ran away on 4–7 of the WER scale and was never scored; the IFT
  stage starts from a staged checkpoint-47730 of that arm and repairs its ChatML rows.

### Joint ASR + ST score

³ This section defines the `joint` column of the table above. It is the
MA-utility study's headline score (defined in
`projects/self-distill/eval/tier1_joint.py`, branch `ma-utility-scoring`;
adopted 2026-10-21):

`joint = (100 − asr24 + X→en chrF + en→X chrF) / 3`, higher is better, where
`asr24` is the geometric mean over the 24 EU languages of min(CER, 1), in
percent, and both chrF terms are lowercase plain chrF, macro-means over the
23 directions. Equal thirds: over steps 718 to 2,872 the study's arms gain
about as many points on each part (1.7 / 1.9 / 1.5), so one point of each
stands for about the same amount of fine-tuning.

| arm | asr24 (%) | X→en chrF | en→X chrF | joint |
|---|---|---|---|---|
| llama1b-base | 19.09 | 50.5 | 23.9 | 51.7 |
| llama1b-ins | 24.02 | 50.8 | 24.5 | 50.4 |
| qwen35-2b-base | 24.64 | 50.5 | 25.9 | 50.6 |
| qwen35-2b-ins | 24.63 | 50.3 | 25.1 | 50.3 |

Two instrument differences from the study's own table, both improving this
one: `asr24` here is measured on `fleurs26-dev200` (200 clips per language)
rather than the study's 100, and the ST terms are 100 clips per direction on
the frozen ST dev sets rather than the step-5 battery. Absolute values are
therefore not interchangeable with Tier 1's; the ranking within this table is.

EuroLLM is pending (its IFT is resuming); its MA-stage selection score is
.1407, the worst of the six, so its joint score is not expected to lead.

### Per language

Languages on rows, arms on columns: **L-B** Llama-3.2-1B (base), **L-I**
Llama-3.2-1B-Instruct, **Q-B** Qwen3.5-2B-Base, **Q-I** Qwen3.5-2B (instruct).
EuroLLM is not in these tables: its IFT is still resuming, so there is no
`checkpoint-6310` to score.

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

**ST X→en — lowercase chrF, 100 clips per source.** `ga` is the outlier for
every arm (18–19 against 39–66 elsewhere): the outputs are degenerate
English repetitions ("...the church of the church of the church..."), not
translations, so that row says nothing about the arms' relative quality.

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

**ST en→X — lowercase chrF, 100 clips per target.** Only `de` was trained
(the 429.9 h en→de probe in `ABL-IFT-700.yaml`); every other target is
zero-shot, and the model usually answers in English or another language
than the one asked for (bg and el come out as English, mt as Italian,
el often unrelated to the audio) — hence the near-zero cells. The
intermediate values (es 44–50, fr 45–57, it 41–49) are untrained
directions too; why they sit far above bg/el was not investigated.

| tgt | L-B | L-I | Q-B | Q-I |
|---|---|---|---|---|
| de | 56.5 | 57.3 | **61.5** | 59.5 |
| es | 44.6 | 45.3 | **50.4** | 49.9 |
| fr | 45.2 | 49.8 | **57.4** | 57.2 |
| it | 41.4 | 41.7 | 48.4 | **48.8** |
| pt | 41.1 | 46.3 | **48.3** | 46.6 |
| ro | 24.7 | **30.7** | 27.5 | 23.0 |
| nl | 26.6 | **27.5** | 27.1 | 26.9 |
| da | 27.0 | 27.3 | 27.4 | 27.4 |
| sv | 26.8 | 26.5 | **26.9** | 26.7 |
| cs | 17.1 | 17.1 | **18.5** | 17.2 |
| sk | 17.8 | 17.5 | **19.2** | 17.8 |
| pl | **17.7** | 16.7 | 17.3 | 16.9 |
| hr | 19.7 | 18.8 | **22.2** | 19.4 |
| sl | 18.7 | 18.4 | 21.1 | **21.2** |
| hu | 15.7 | 15.4 | 15.6 | **15.9** |
| fi | 17.6 | 17.3 | 17.6 | 17.6 |
| et | 20.4 | **20.5** | 18.9 | 18.3 |
| lt | **17.2** | 16.5 | 15.7 | 14.0 |
| lv | **16.8** | 15.8 | 16.2 | 16.2 |
| ga | 16.5 | 16.3 | **16.7** | 15.8 |
| mt | 18.6 | 18.5 | 19.5 | **19.9** |
| bg | 0.7 | 0.7 | 0.7 | 0.7 |
| el | 0.6 | 0.6 | 0.6 | 0.5 |
| **mean** | 23.9 | 24.5 | 25.9 | 25.1 |
