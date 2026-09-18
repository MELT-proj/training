# Agent protocol — how any session works on this campaign

Read this once at the start of every session, after `README.md` and the
current week of `timeline.md`, and before touching any machine. It applies
to every week, not just the first.

## 0. Three places, and only three

Everything this campaign records goes to exactly one of these. If you are
unsure where something belongs, it belongs on the board.

| | holds | who writes it |
|---|---|---|
| `board.md` | everything that happened: findings, numbers, surprises, incidents, dead ends, doubts, proposals | any session, append-only, newest first |
| `timeline.md` | the TODO list, week by week, with tick boxes | **only the orchestrator adds or removes items**; sessions tick the boxes they complete |
| `01`–`06` | the *design* of each experiment, its Settled decisions, and its results tables | any session, but only for design changes and measured results |

**The section files are not a log.** A finding, an incident, a config
surprise or a "found this while doing that" note goes on the board and
stays there. What may enter `01`–`06` is a design decision, a Settled item,
or a number in a results table. If a paragraph in a section file would read
as news in a week's time, it is board material and it is in the wrong file.

**You do not edit the TODO list.** If work needs adding, dropping or
moving between weeks, post it on the board and the orchestrator folds it
into `timeline.md`. This keeps one person's view of what the campaign owes
itself, and keeps two sessions from inventing overlapping work. Ticking a
box you finished is always yours to do.

## 1. Where things are

- Branch: `claude/speech-llm-ablation-research-46d735` until it is merged;
  check `git log -1` on `main` first, the plan may have moved there.
- The plan: `projects/ablation-campaign/plan/`. Section files `01`–`06`
  hold each experiment's design and its results tables. Their **Settled**
  blocks are decisions already made with the PI: do not reopen them in a
  session. If you think one is wrong, post to `board.md` and carry on with
  the task as specified.
- Machines, repos, tooling, measured costs and known traps:
  `infrastructure.md`. Do not re-derive any of it; if it is stale, fix the
  file and say so on the board.
- Data: `data/hours_by_language.csv` and `data/README.md`.
- **Never scan the filesystem to find something** (PI, 2026-09-18). No
  `find /`, no `grep` from `/` or `$HOME`, no walking the tree hoping a name
  turns up. It is slow, it fills your context with noise, and on a shared
  machine it reads other people's data. Look in this repo, in
  `infrastructure.md` §2, and in the sibling repos that section lists — that
  is the whole search space.
- **If it is still not there, ask the PI.** Do not hunt, do not guess a
  path, and do not substitute a file you are not certain is the right one.
  A one-line question costs less than a filesystem walk, and the answer is
  then written into `infrastructure.md` so the next session never has to
  ask it again.

## 2. Rules on machines

- GPU work runs only through SLURM: `sbatch`, or `campaign.py run` on MN5.
  Never load a model on a GPU interactively, not even for a smoke test.
- MN5 has no internet on any node. Stage models over `mn5transfer`, push
  the repo with `infra/sync_repo.sh mn5`, regenerate rendered `ABL-*.yaml`
  configs on MN5 rather than syncing copies.
- Always pass `MELT_GPUS_PER_NODE=4` on MN5 and confirm `world_size` in the
  `[run_train] starting` log line.
- Nothing sizeable under `/mnt/home` on artemis; results, venvs, frozen
  sets and caches go to scratch. Check `df` before writing audio anywhere.
- Never delete or overwrite a checkpoint, a `.sif` image or an output
  directory without asking the PI first. Rename out of the way instead.
- Cleanup after a killed process is the operator's job: report, do not
  auto-fix.

## 3. Rules on experiments

- One epoch with derived steps. Never pin `max_steps` per arm.
- The effective batch (`batch_duration × grad_accum × world_size`) is a
  campaign constant per stage. Changing it is a recipe change, not a
  convenience, and must be written into `01-interface-recipe.md`.
- Anything identical across arms (`max_duration`, `max_tokens`,
  `quadratic_duration`, eval `max_samples`) lives in the base YAML, never
  in a per-arm override.
- `EXP_NAME` is composed by the tooling, never typed by hand. Every
  submission goes through `campaign.py run` so it lands in `arms.tsv`;
  a run submitted any other way is off the ledger and does not count.
- **Name arms descriptively, never by letter or number** (PI, 2026-09-18).
  `wsd-10hz-qwen4b` is readable a week later; `R-k5`, `A0` and `L4` are
  not. The name carries the setting that differs from the reference arm.
  This applies to campaign row ids and to every table in `plan/`. A row id
  is only the grid key, so renaming one is safe; `exp_name` is not, because
  output directories, `arms.tsv` and W&B all key on it.
- **Hold the seed fixed across arms unless the seed is the variable.** One
  seed per arm makes every contrast a factor change plus a seed draw, which
  is how a 0.11 WER difference in step 0b turned out to be within the seed
  spread (`01-interface-recipe.md` §2b).
- Turn on `run.memory_preallocation` whenever what is trainable changes.
- Read steady-state throughput from the second-to-last tqdm line, never
  from `train_runtime` or the closing average.
- A run is finished when it has generative scores and a passing exposure
  audit. A training loss is a health check, not a result.
- Compare batched WER only at the same batch width, with
  `flash_attention_2` and duration-sorted batches.
- Stay inside the current week of `timeline.md`. Do not start Fondue,
  Raclette, or a later week's arms unless the PI says so in the session.

## 4. Rules on code

- Small change: show the diff in chat, then commit to the working branch.
  Multi-part change: a PR against `main`.
- Tests run in the container on nyx (see `infrastructure.md`); do not
  create a venv for them.
- No self-healing logic in training code.
- Do not implement anything a plan file lists as "Open" or "decided by the
  PI" without asking.

## 5. Recording what you did

Every session ends with both of these, in this order:

1. **`board.md`**: one entry at the top, using the template in the file.
   Findings with numbers and experiment names; doubts; what surprised you;
   anything blocked and what it blocks; what the next person should do.
   Job ids belong in `arms.tsv`, not here.
2. **`timeline.md`**: tick the boxes you completed in the current week.
   Do not add, delete or reword an item, and do not move one between weeks
   — say so on the board and the orchestrator does it. A box you could not
   finish stays unticked with a board entry explaining why.

Then, only if you produced a measured result or changed a design: update
the relevant section file's results table or design text. Measured numbers
say "measured" and name the experiment; extrapolations say "extrapolated".
Numbers in prose are kept out of the section files; use the tables.

## 6. When to stop and ask

Stop and ask the PI, rather than guess, when: a Settled decision looks
wrong; a run needs more than the week's budget; a checkpoint or output
directory must be removed; a result contradicts the section file's
interpretation rules; or the task as written is impossible with what is
staged on the cluster.
