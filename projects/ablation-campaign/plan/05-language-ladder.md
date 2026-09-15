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
ladder is EU-24 only).
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
| ST X→en | FLEURS X→en, built from sentence ids | 24 | chrF, COMET, BLEU |
| ST X→en | CoVoST2 X→en test | 10, several tiny | chrF, COMET, BLEU |

Both hypothesis and reference go through the same multilingual normaliser
(casing, punctuation, apostrophe variants) before scoring; the models train
on truecased text and FLEURS references are cased. Score corpus-level.
Anchor every table with published Whisper-large-v3 and SeamlessM4T-v2
per-language FLEURS numbers so P is interpretable.

## 4. Analysis

Fit, per language, a saturating power law on CER: CER(N) = c + a·N^(−b),
with the family-level exponent pooled for languages with fewer than three
tiers. Report "hours to reach CER ≤ 10%" per language with a confidence
band from the seed replicates, and the transfer effect per family from
§2. Fondue supplies one out-of-sample point per language; plot predicted vs
achieved.

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
