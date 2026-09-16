# 02 — Backbones: the 2×3 grid and the text prior

**Settled (2026-09-15):** the grid is Llama-3.2-1B, Qwen3.5-2B, EuroLLM-1.7B,
each base and instruct; backbones are judged on the five in-domain languages
*and* FLEURS-24 zero-shot; a text-only prior is measured for every backbone
and language before training and reported against post-training adaptability.
**Open:** the decision rule's weighting between in-domain and zero-shot.
**Owner:** PI decides the backbone at the week-5 gate.

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
  with that caveat.
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
(`projects/ablation-campaign/plan_arm.py`). EuroLLM has no entry there yet —
no MA/IFT arm has used it — so this session verified both checkpoints'
`tokenizer_config.json` directly instead of extrapolating from the Llama
pattern; good thing it did, since Qwen3.5-2B-Base disproves the "base never
ships a template" assumption a reader might otherwise bring from the Llama
row alone. **Whoever builds this should add the EuroLLM entry to
`DECODER_PROFILES` while at it** (`chatml`, `chat_template_from:
utter-project/EuroLLM-1.7B-Instruct`): MA/IFT need it in week 3 regardless,
and an unverified guess here would silently corrupt every EuroLLM row of the
prior table, the same way a wrong `chat_template_config` silently corrupted
training before eval caught it (`melteval/prompt.py`'s framing,
MELT-proj/training#58).

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

| arm | MA-stage WER (5-lang mean) | FLEURS-24 CER (MA) | IFT CER (5-lang) | FLEURS-24 CER (IFT) | X→en chrF | text retention | notes |
|---|---|---|---|---|---|---|---|
| llama1b-base | | | | | | | |
| llama1b-ins | | | | | | | |
| qwen35-2b-base | | | | | | | |
| qwen35-2b-ins | | | | | | | |
| eurollm1.7b-base | | | | | | | |
| eurollm1.7b-ins | | | | | | | |
