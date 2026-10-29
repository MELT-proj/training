# Sharing `results_master.csv` across sessions

Every number a chart, a table or a message quotes lives in one file,
`projects/ablation-campaign/charts/results_master.csv`: **one row per model, one column per measurement**.
Several sessions (agents and people) read and write it. This protocol is what keeps it one trustworthy table.
Read it once before your first write; reading needs none of it beyond §3.

It sits beside `plan/agent-protocol.md` and never overrides it (never scan the filesystem, GPU work only
through SLURM, no self-healing, ask the PI when a rule does not cover the case).

## 0. The rules on one screen

1. **One table.** Charts and tables read this file and nothing else. A number that is not in it is not yet a result.
2. **Read freely, write only through `master.py`** (or an `ingest_*.py` that calls it). Never edit the CSV by hand,
   in a spreadsheet, or with pandas `to_csv`: the tool takes the lock, checks the schema, refuses silent
   overwrites and writes the changelog; nothing else does.
3. **Measured or cited, nothing else.** A cell is a number that melt-eval (or the scorer the column names) produced
   on a saved checkpoint, or a published number copied from a table you cite. No in-training eval numbers, no
   estimates, no interpolation, no figure retyped from a chat message or rounded in a README sentence.
4. **A name is a promise.** The same `set`, `fmt` and `metric` mean the same clips, prompt and scorer. A different
   subset, split, prompt, normaliser or scorer gets a new name (§4: register the set, add the metric); a number
   scored differently never goes under an existing name, not even "close enough".
5. **A filled cell stays.** A different value for a filled cell is refused. `--overwrite --reason "<what was
   wrong>"` replaces it and logs the old value. If a re-run of the same eval comes out different, that is a
   finding for `board.md` before it is an overwrite: the second number is not automatically the right one.
6. **Rows are never deleted or renamed.** Figure scripts and other sessions' notes refer to the keys. A wrong or
   discarded model gets `status=deprecated` (overwrite plus reason); charts skip deprecated rows.
7. **Write, validate, publish as one short cycle** (§4, §5). Never leave the master modified and uncommitted
   overnight, never push a file that `master.py validate` rejects.

## 1. The files

| file | holds | written by |
|---|---|---|
| `results_master.csv` | the table | `master.py` and the ingest scripts only |
| `results_sets.csv` | registry of evaluation sets: what each `set.fmt` is, clips per unit, which metrics it may carry | `master.py add-set` |
| `results_changelog.csv` | append-only audit log, one line per model per write: when, who, how many cells, which prefixes, the note, the reason, old values of overwritten cells | `master.py` (never by hand) |
| `master.py` | the only writer, plus `show`, `columns`, `validate`, `export-long`, the merge driver | reviewed code |
| `ingest_eu24_report.py` | an `eu24_report.json` (FLEURS dev, one prompt format) into the master | reviewed code |
| `ingest_published.py`, `published/` | hand-copied published tables into the master | reviewed code |
| `fig_*.py`, `figures/` | one script per figure, and its output | anyone, §3 |

## 2. Schema

**Rows.** `model` is a stable key (letters, digits, `.`, `_`, `-`). Metadata columns, in this order: `label` (the
text a legend shows), `kind` (`ours`, `baseline`, `reference`), `family` (decoder or model family), `recipe`
(`start`, `full-ft`, `adapter-ce`, `adapter-sd`, `adapter-sd-sift`, `public`), `run` (the training run, the
melt-eval `sft_run`), `status` (`active`, `deprecated`), `order` (position in bars and legends, lower first;
baselines 1-9, Qwen checkpoints 10-19, EuroLLM 20-29), `source` (where the row's numbers come from), `notes`.

**Columns.** Every other column is one measurement, named `<set>.<fmt>.<metric>.<unit>`:

| part | is | examples |
|---|---|---|
| `set` | the frozen evaluation set; its name carries the split | `fleurs24-asr-dev`, `fleurs24-st-xen-dev`, `selection-id`, `fleurs-reported` (published) |
| `fmt` | the prompt format the model ran under | `repeatfirst`, `sft-random`, `native` (published) |
| `metric` | one of `METRICS` in `master.py` | `wer`, `cer`, `chrfpp`, `chrfpp-lc`, `spbleu-lc`, `nocaps`, `runaway`, `score`, `median` |
| `unit` | a language code, a group, or `all` | `en`, `ga`, `ood-latin`, `all` |

For X->en sets the unit is the source language, for en->X sets the target language. `-lc` metrics are scored on
lowercased text (the primary ST scoring, because standard-training models write without capitals).

**Values.** WER, CER, `nocaps`, `runaway` in percent (7.5 means 7.5 %), MT scores on 0-100, and only the selection
metric's `score` and `median` as fractions, the way the campaign quotes them. Up to 4 decimals (6 for fractions).
A blank cell is "not measured": never `0`, `NaN`, `n/a`. UTF-8, LF line ends; the tool writes columns and rows in
a canonical order, so two writers produce the same bytes for the same content.

Find columns with `python3 master.py columns 'fleurs24-asr-dev.repeatfirst.wer.*'` and the sets with
`python3 master.py columns --sets`. The definition of every metric is in `METRICS`; of every set, in
`results_sets.csv`.

## 3. Reading

- No lock and no coordination needed: the file is replaced atomically, so a reader sees a whole file, old or new.
- It may be **stale**: see §5 for the latest. A decision based on a number needs the latest.
- Python without dependencies: `import master; table = master.load()` (`table.rows[model][column]`). pandas:
  `pd.read_csv(path).set_index("model")`. Tidy form for pandas or R: `python3 master.py export-long --out x.csv`.
- Quoting a number in prose, a README or a slide: take it from the master
  (`python3 master.py show -m SD24 -c 'fleurs24-asr-dev.repeatfirst.wer.en'`) and name its set and `fmt`.
- A **figure script** reads only the master: everything else (languages, which models, clipping) is a command-line
  option, the output goes to `figures/`, the script prints the table it drew and a caption with the comparability
  caveats, and the output file carries the master's SHA-256 (`fig_fleurs_asr_vs_baselines.py` is the pattern).
  It needs only the standard library, numpy and matplotlib, so it runs anywhere and can be copied into a poster
  repository.

## 4. Writing

**Once per clone** (nyx, artemis; the setting is shared by every worktree of the clone):
`python3 projects/ablation-campaign/charts/master.py install-git-driver`. It registers the cell-level merge
driver §5 relies on; `validate`, `where` and `upsert` print a warning while it is missing.

**Adding a scored checkpoint** (the usual case; commands from the repository root, `C=projects/ablation-campaign/charts`):

1. Get the latest master (§5).
2. Score the saved checkpoint with melt-eval (`projects/self-distill/eval/eu24_jobs.py` and `eval_queue.py`) and
   build the report with `projects/self-distill/eval/eu24_report.py`, which writes an `eu24_report.json` per
   prompt format. If the model is new, add its key to `MODELS` in `eu24_jobs.py`, `eu24_report.py` and
   `$C/ingest_eu24_report.py` (label, family, recipe, run, order).
3. Dry run: `python3 $C/ingest_eu24_report.py REPORT.json --fmt repeatfirst --who "<you>" --dry-run`. It must show
   your model's cells as added and **0 changed**. Anything changed means someone else's numbers are about to be
   replaced: stop and read rule 5.
4. The same command without `--dry-run`. Repeat with `--fmt sft-random` for the second prompt format.
5. `python3 $C/master.py validate` must report 0 problems.
6. Publish (§5).

**Adding other numbers by hand** (your own scorer, a new set): write a CSV `cells.csv` with the header
`model,column,value`, and for a new model a `meta.csv` with `model,label,kind,source` (plus `family`, `recipe`,
`run`, `order`), then
`python3 $C/master.py upsert --cells cells.csv --meta meta.csv --who "<you>" --note "<run, eval logs or report, date>" --dry-run`
and again without `--dry-run`. The note is the provenance: a stranger must be able to find the logs from it.

**A new evaluation set** must be registered before a column of it can be written:
`python3 $C/master.py add-set --set <name> --fmt <fmt> --task <asr|st|aggregate> --split <dev|test> --n-per-unit <N> --metrics "wer cer" --description "<the frozen set, how its clips were drawn, the prompt, the normaliser>" --who "<you>"`.
The description must let a stranger redo the scoring. **A new metric** is a reviewed edit of `METRICS` in
`master.py`: its name, scale and definition. Do not reuse a name for a different scorer.

**Published numbers**: copy the table by hand into `published/<source>.csv` (`#` header lines: the citation, the
section, who supplied it and when; digits exactly as printed, one row per language), describe it in `TABLES` of
`ingest_published.py`, run that script. Do not average, round or convert. Say in `notes` what the source does not
state (split, normaliser): the figure's caption repeats it.

**Retiring a model**: a `retire.csv` holding `model,status` and `<key>,deprecated`, then
`python3 $C/master.py upsert --meta retire.csv --who "<you>" --note "<why>" --overwrite --reason "<why>"`.
The row and its numbers stay.

## 5. Git: how the copies stay one table

Each clone and worktree holds its own copy of the three data files; **git is the only transport** and there is no
live shared file. The branch that carries this folder and that sessions publish results to is the **results
branch**: `main`. (Until 2026-10-19 it was `self-distill-whisper`; that branch was merged into `main` that day.)

- **Latest, read-only:** `git fetch origin <results-branch>`, then
  `git show origin/<results-branch>:projects/ablation-campaign/charts/results_master.csv > <scratch>/results_master.csv`
  and point a figure script at it with `--master`. Your working tree is untouched.
- **Latest, to write on top of:** merge or rebase your branch onto `origin/<results-branch>` first. If your branch
  is a different line of work, check out the results branch in its own worktree for the write.
- **Publish:** `git add results_master.csv results_sets.csv results_changelog.csv` (those three, nothing else in the
  commit), commit with a message like `results: SD24-EuroLLM fleurs24-asr-dev repeatfirst`, `git pull --rebase`,
  `master.py validate`, `git push`. If the push is rejected, pull again and repeat; do not force.
- **Concurrent writers merge by themselves.** `.gitattributes` maps the master to the cell-level merge driver
  (`master.py merge-driver`): edits to different cells, new rows and new columns combine, and the changelog is a
  `union` merge. The merge stops only when two sessions set **the same cell to different values**; it names the
  cell and both values. Decide which one is right, usually by finding out that one of them was scored under a
  different protocol and needs a new name (rule 4), and write the winner with `--overwrite --reason`. Plain git
  cannot do this: two branches that each add a column conflict on the whole file.
- **Two processes in one checkout** are serialised by `.results_master.lock` (an exclusive-create file) and each
  write is read-modify-write inside it, so none is lost. A lock whose holder died is never broken by the tool:
  the error names the holder (host, pid, who, since); check that the process is gone, remove the file yourself,
  and say so in your next note.

## 6. When something is wrong

| you see | do |
|---|---|
| `<<<<<<<` in the master, or "N fields but the header has M", after a merge or rebase | the driver was missing. `git merge --abort` (or `git rebase --abort`), `master.py install-git-driver`, redo it. Never repair the CSV by hand. |
| the merge stops on a cell conflict | §5: decide, then write the winner with `--overwrite --reason`; `git add` and continue |
| `validate` fails on a file you just pulled | do not write on top of it. Find the commit (`git log -p -- results_master.csv`), tell the PI and the writer named in `results_changelog.csv`, revert that commit if it is clearly bad |
| you wrote a wrong number | `--overwrite --reason "<what was wrong>"`; the changelog keeps the old value |
| you wrote a wrong or discarded model | `status=deprecated`, as in §4 |
| you hand-edited the file | `git checkout -- results_master.csv`, then redo it with the tool; the changelog shows what went in |
| `LockTimeout` | §5, last bullet |

## 7. What does not go in

Per-sample outputs and logs (they stay with the eval logs; the changelog `note` names them). Aggregates a script
can recompute from the cells (tier means, macro averages): scripts compute them, except `selection-metric.*`, whose
definition is fixed in `projects/ablation-campaign/selection-metric/` and which the campaign quotes. Costs, step
times and training losses. Anything that is not a number: put it in `notes` (one line) or on the board.

## 8. Not in the master yet (2026-10-16)

The FLEURS **test** sets (never run; when they are, register `fleurs24-asr-test` and the ST test sets, then ingest
with `eu24_report.py --split test`), the earlier self-distillation steps and the HR5 runs, the audio-stack
crossing cells and their selection scores, COMET, and the text-only translation ceilings (scored with plain chrF,
not `chrfpp-lc`: a different metric, so they need their own name). The owner of each adds them by §4.

## 9. Changing this protocol

The PI decides schema and protocol changes. A session that needs one posts it on `board.md` and carries on within
the rules; it does not edit `METRICS`, `META_COLUMNS` or this file on its own authority. Version: 1, 2026-10-16.
