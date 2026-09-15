# 02 — Backbones: the 2×3 grid and the text prior

**Settled (2026-09-15):** the grid is Llama-3.2-1B, Qwen3.5-2B, EuroLLM-1.7B,
each base and instruct; backbones are judged on the five in-domain languages
*and* FLEURS-24 zero-shot; a text-only prior is measured for every backbone
and language before training and reported against post-training adaptability.
**Open:** EuroLLM checkpoint ids (assumed `utter-project/EuroLLM-1.7B` and
`-Instruct`; confirm on the Hub); the decision rule's weighting between
in-domain and zero-shot.
**Owner:** PI decides the backbone at the week-5 gate.

## 1. The grid

| family | base | instruct | facts to report |
|---|---|---|---|
| Llama 3.2 | `meta-llama/Llama-3.2-1B` | `meta-llama/Llama-3.2-1B-Instruct` | 16 layers, hidden 2048, vocab 128K; official coverage of 8 languages; the chat template injects a dated system message |
| Qwen 3.5 | `Qwen/Qwen3.5-2B-Base` | `Qwen/Qwen3.5-2B` | 24 layers, hidden 2048, vocab 248K; **dense**, hybrid attention: 18 Gated-DeltaNet linear-attention layers and 6 full-attention layers in a 3:1 pattern; the chat template opens every assistant turn with an empty think block |
| EuroLLM 1.7B | `utter-project/EuroLLM-1.7B` | `utter-project/EuroLLM-1.7B-Instruct` | trained on all 24 EU languages; the natural fit for the coverage commitment; ids unconfirmed |

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
on the FLEURS test references used by the speech eval:

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

Where it lives: melt-eval, as a text-only task over the same frozen sets, so
references and prompts are byte-identical to the speech eval. Needs one GPU
for an afternoon on an internal machine; no MN5 time.

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
