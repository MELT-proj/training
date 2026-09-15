# data/

## `hours_by_language.csv`

Generated 2026-09-15 from the `by_leaf` tab of the shar statistics
spreadsheet (`melt_shar_stats.xlsx`, Google Drive, PI's account), after
removing duplicated leaf rows in the export (275 rows appeared twice).
`check_all_splits` reproduces the spreadsheet's `by_language_task` ASR total
exactly for every language, which is the proof the parse is right.

| column | meaning |
|---|---|
| `asr_train_hours` | unique ASR hours in train splits: CV22 + MLS + VoxPopuli + FLEURS train + (LibriSpeech, People's Speech for en) + Granary `asr_only` + Granary `ast` |
| `asr_cv22`, `asr_mls`, `asr_voxpopuli`, `asr_fleurs`, `asr_other` | the non-Granary parts |
| `granary_asr_only` | Granary leaves with transcript only |
| `granary_ast` | Granary leaves with transcript **and** English translation; counted as both ASR and ST hours |
| `st_x_to_en_train_hours` | Granary `ast` + CoVoST2 X→en train: translation data with audio in X |
| `covost_x_to_en`, `covost_en_to_x` | CoVoST2 train, by direction; en→X has **English** audio |
| `max_full_tier` | largest of 10/30/100/300/700 that `asr_train_hours` reaches |
| `check_all_splits`, `sheet_by_language_task_asr` | the cross-check against the spreadsheet |

Rows: the 24 EU languages, then ru, ca, uk.

## Eval-set inventory (test/dev splits present in the shar tree, 2026-09-15)

| set | languages | task |
|---|---|---|
| FLEURS test + validation | all 24 EU, plus 69 others | ASR |
| CV22 test + validation | 23 EU (no hr) | ASR |
| VoxPopuli test + validation | cs de en es et fi fr hr hu it lt nl pl ro sk sl | ASR (only de/en/es/fr/it have the full `custom` block) |
| MLS dev + test | en de fr es it nl pl pt | ASR |
| LibriSpeech dev + test, People's Speech test | en | ASR |
| CoVoST2 X→en dev + test | de es fr it nl pt sv sl lv et (+ ca ru and non-EU) | ST |
| CoVoST2 en→X dev + test | de et lv sl sv (+ non-EU) | ST, English audio |
| FLEURS X→en | **does not exist yet**; build by joining sentence ids | ST |
