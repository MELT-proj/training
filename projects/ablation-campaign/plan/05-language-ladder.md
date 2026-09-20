# 05 — The language ladder: "N hours of language X gives P"

**Settled (2026-09-15):** an equal-per-language-budget ladder of multilingual
models at five log-spaced tiers, each run as MA + IFT with the winning
recipe, backbone and stack; every run is evaluated on all 24 EU languages;
N means unique hours at one epoch; CER is the cross-language ASR metric,
chrF/COMET for ST; Russian and Ukrainian stay in the data and Russian's
effect on low-resource Slavic languages is a named probe.
**Open:** which high-resource language per family is dropped in each
mixed-tier run; the exact language of each probe; whether Russian sits in the
base tiers or only in the probe (proposal: only in the probe, so the base
ladder is EU-24 only); **whether the ladder runs on a supervised encoder at
all** (added 2026-09-18, see §4.1).
**Owner:** PI, when the ladder configs are built (week 3–5).

## 1. The data, and what it forces

Unique ASR training hours per language, train splits only, after
de-duplicating leaves (`data/hours_by_language.csv`, from the shar
statistics spreadsheet of 2026-09-15):

| tier | languages that reach it in full | count |
|---|---|---|
| 700 h | en es fr de it nl pt pl | 8 |
| 300 h | the same eight (hu has 274 h) | 8 |
| 100 h | + hu fi cs ro sv | 13 |
| 30 h | + hr sk bg lv | 17 |
| 10 h | + da lt el et sl ga mt | 24 |

Each tier run includes all 24 languages, each at min(tier, available). The
eleven languages under 100 h sit at their full amount from tier 30 or 100
upwards; their curve truncates, which is information, and their transfer
environment keeps growing (see §3).

Three facts about the tail that the paper must state:

- **ST X→en with X audio** (Granary `ast` plus CoVoST2 X→en) exists at 30 h
  or more for 13 languages and is essentially zero for sl, lv, mt and ga.
  en→X exists only for de, et, lv, sl, sv (CoVoST2, 430 h each). The ST
  curve covers about half the languages.
- **Granary `ast` leaves are both ASR and ST** (transcript and translation of
  the same audio). For the tail that is most of their data, so ASR and ST
  share audio there, unlike the five-language design. The exposure audit
  counts both.
- **In-domain and FLEURS coincide for the tail.** For ga, mt, el, bg, da, sv
  the training data is mostly FLEURS train plus a little CommonVoice, so
  FLEURS test is in-domain and CV22 test is the out-of-domain check, the
  reverse of the high-resource languages.

Also: VoxPopuli outside de/en/es/fr/it went through a generic converter and
lacks the `custom` block (no `num_tokens`, colliding ids, unfiltered empty
transcripts; hr is ~47% textless). Those leaves feed cs, hu, ro, sk, hr, fi,
pl, sl, lt, et, nl and must be checked before the tail tiers are trusted.

Families (for pooling and for held-out choices):

| family | EU-24 members | outside donors |
|---|---|---|
| Germanic | en de nl sv da | |
| Romance | fr es it pt ro | ca |
| Slavic | pl cs sk sl hr bg | ru uk |
| Baltic | lt lv | |
| Uralic | fi hu et | |
| Hellenic | el | |
| Celtic | ga | |
| Semitic | mt | |

Greek, Irish and Maltese have no relatives in the set, and Irish and Maltese
are the scarcest. They are where the question matters most and where the
answer is data-bound.

## 2. The runs

| run | what | cost (Llama-class, extrapolated) |
|---|---|---|
| T10, T30, T100, T300, T700 | equal-budget ladder, all 24 languages, MA (ASR) + IFT (ASR+ST) | ≈ 100 GPU-h MA + 400 GPU-h IFT for all five |
| M1, M2 | mixed-tier: at the 700 budget, one high-resource language per family dropped to 10 h, rotating which | ≈ 2 × 120 GPU-h |
| P-rep | repetition probe: 100 h × 4 epochs vs 400 h × 1 epoch, on two or three languages inside an otherwise T100 world | ≈ 2 × 80 GPU-h |
| P-ru | Russian probe: T100 with and without Russian at 700 h (and Ukrainian at its 647 h) | ≈ 2 × 100 GPU-h |

The tail's transfer estimate comes from the ladder itself: Irish sits at its
12.7 h in every tier while the world grows from ~240 h to ~6,800 h. The
mixed-tier runs supply the same estimate for the eight high-resource
languages. The Russian probe answers a Fondue design question directly.

The config builder needs a per-language "min(tier, available)" budget and
the ability to pin one language to a different tier; both are small changes
to `build_campaign_config.py` (week 3).

## 3. Evaluation

| task | set | languages | metric |
|---|---|---|---|
| ASR in-domain / zero-shot | FLEURS test | 24 | CER, WER |
| ASR out-of-domain | CV22 test | 23 (no hr) | CER, WER |
| ASR | VoxPopuli test | 15 | CER, WER |
| ST X→en | FLEURS X→en, built per §3.1 | 23 (all but en) | chrF, COMET, BLEU |
| ST X→en | CoVoST2 X→en test | 10, several tiny | chrF, COMET, BLEU |

Both hypothesis and reference go through the same multilingual normaliser
(casing, punctuation, apostrophe variants) before scoring; the models train
on truecased text and FLEURS references are cased. Score corpus-level.
Anchor every table with published Whisper-large-v3 and SeamlessM4T-v2
per-language FLEURS numbers so P is interpretable.

### 3.1 Building the FLEURS X→en set (week 1, Track B; home: melt-eval)

**Why it is possible.** FLEURS is sentence-parallel and every shar cut's
`id` is the FLEURS sentence id, shared across locales. The X→en reference
for a cut is therefore the English text of the same id. Measured on the nyx
shar tree on 2026-09-15: for `de_de`, `it_it`, `ga_ie`, `mt_mt`, `hu_hu`
and `lv_lv` test, **100% of sentence ids are present in `en_us/test`**
(343–348 ids and 842–926 cuts per language against 350 English ids), so the
split assignment is consistent across languages and no cross-split lookup
is needed.

**The reference is not unique per id after truecasing.** Across the
duplicate recordings of one English sentence, the lowercase supervision text
is identical (0 of 350 ids differ) but `custom.pnc_text` differs for 57 of
350 ids, because the PNC pass rewrote each recording separately (for
example "and which was made famous" against "which was made famous"). The
reference must be chosen deterministically per id.

**Design.**

- Reference text: FLEURS' original cased English transcription for the
  sentence id (`raw_transcription` in the HF `google/fleurs` en_us metadata,
  350 test and 150 validation sentences, text only), fetched once on a
  machine with internet and stored as a small `{id: text}` JSON beside the
  frozen sets or under melt-eval `configs/refs/`. Fallback if that cannot be
  obtained: the majority `pnc_text` among the id's `en_us` recordings, ties
  broken by the lowest recording index. Never the lowercase supervision
  text, which would make chrF/COMET incomparable with published numbers.
- Mechanism (built 2026-09-15, `melt-eval` PR #11): `reference_map` on
  `SharReader` (`<path to the json>`) overrides the target text by cut id,
  with tags `task: st`, `src_lang: <x>`, `tgt_lang: en`, `dataset_id:
  fleurs-<x>_en` -- per-locale, not a shared `dataset_id: fleurs`, because
  `get_tags_from_cut` returns the *target* language as `lang` for any
  `task: st` cut, so a shared dataset_id would collapse all locales into
  one `lang=en` bucket under the `grouped(..., "lang", ...)` BLEU/chrF
  metric. The audio stays a locator into `fleurs/<locale>/test`, zero-copy.
  Do not rewrite shar manifests: that invalidates the `.idx` files, and
  this set is an evaluation artefact, not training data.
- Specs: `fleurs24-st-xen-test` (23 locales, English excluded) and a
  `fleurs24-st-xen-dev` subset of about 100 utterances per language from
  `validation`, sharing sentence ids with the ASR dev subset where possible
  so both tasks score the same audio.
- Prompt: the IFT ST prompt (`Translate this audio to English.`) rendered
  through melt-eval's parity layer.
- Scoring: chrF and COMET on the raw reference; BLEU with sacrebleu's
  default tokenizer. The ASR normaliser is not applied to ST.
- Verification before use: sample counts (23 locales × about 350 ids ×
  about 2.5 recordings, so roughly 20K), zero dropped-no-reference, and a
  manual spot check of 20 random (audio locale, English reference) pairs.
- Optional, same recipe: en→X sets for de, et, lv, sl, sv (English audio,
  X reference), matching the CoVoST2 en→X training directions.

Locales: `bg_bg cs_cz da_dk de_de el_gr en_us es_419 et_ee fi_fi fr_fr ga_ie
hr_hr hu_hu it_it lt_lt lv_lv mt_mt nl_nl pl_pl pt_br ro_ro sk_sk sl_si sv_se`.

## 4. Analysis

Fit, per language, a saturating power law on CER: CER(N) = c + a·N^(−b),
with the family-level exponent pooled for languages with fewer than three
tiers. Report "hours to reach CER ≤ 10%" per language with a confidence
band from the seed replicates, and the transfer effect per family from
§2. Fondue supplies one out-of-sample point per language; plot predicted vs
achieved.

### 4.1 What a supervised encoder does to the x-axis (open, 2026-09-18)

Raised after `wsd-50hz-whisper` crossed the step-0b threshold by an order of
magnitude (`01-interface-recipe.md` §5, board 2026-09-18). If the winning
stack uses Whisper, N stops being the model's exposure to language X: the
encoder arrives with its own per-language supervised hours, unevenly spread
across the EU-24 by orders of magnitude. The 10 h and 30 h tiers would then
largely measure that prior, and a low tier could look strong for reasons
unrelated to our data. Three options, PI's call before the ladder configs
are built:

| option | what the ladder then claims | cost |
|---|---|---|
| (a) run the ladder on the SSL encoder, keep the supervised one for the performance sections | "N hours gives P", clean, but not the shipping system | no extra runs |
| (b) keep the supervised encoder, reframe the axis as fine-tuning hours on top of it, and report its per-language pretraining hours as a covariate | honest, and matches the shipping system, but the curve is no longer a data-scaling law | no extra runs; needs the encoder's published per-language hours as a table |
| (c) run both encoders at two tiers (10 h, 100 h) for a handful of languages spanning the coverage range | adds "does supervised pretraining substitute for in-domain hours, and for how many hours?" | 4 extra MA+IFT runs at two tiers, on a language subset |

(c) is the recommendation: it is the smallest experiment that turns the
confound into a result, and the substitution rate it measures is the kind of
number a reader of the ladder section will want anyway.

## 5. Results

| tier | exp_name (MA) | exp_name (IFT) | status |
|---|---|---|---|
| T10 | | | |
| T30 | | | |
| T100 | | | |
| T300 | | | |
| T700 | | | |
| M1 | | | |
| M2 | | | |
| P-rep | | | |
| P-ru | | | |
