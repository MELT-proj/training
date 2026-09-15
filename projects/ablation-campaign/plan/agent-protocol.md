# Agent protocol — how any session works on this campaign

Read this once at the start of every session, after `README.md`,
`00-status.md` and the current week of `timeline.md`, and before touching
any machine. It applies to every week, not just the first.

## 1. Where things are

- Branch: `claude/speech-llm-ablation-research-46d735` until it is merged;
  check `git log -1` on `main` first, the plan may have moved there.
- The plan: `projects/ablation-campaign/plan/`. Section files `01`–`06`
  hold each experiment's design. Their **Settled** blocks are decisions
  already made with the PI: do not reopen them in a session. If you think
  one is wrong, post to `board.md` and carry on with the task as specified.
- Machines, repos, tooling, measured costs and known traps:
  `infrastructure.md`. Do not re-derive any of it; if it is stale, fix the
  file and say so on the board.
- Data: `data/hours_by_language.csv` and `data/README.md`.

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

Every session ends with all three of these, in this order:

1. **`board.md`**: one entry at the top, using the template in the file.
   Findings with numbers and experiment names; doubts; what surprised you;
   what the next person should do. Job ids belong in `arms.tsv`, not here.
2. **`00-status.md`**: update Running / Done / Blocked and the next
   decisions. Keep it a snapshot, not a log.
3. **`timeline.md`**: tick the boxes you completed in the current week.
   If something slipped, move it forward into the next week explicitly;
   never leave a silent unticked box in a past week.

Measured numbers go in the results table of the relevant section file and
say "measured" with the experiment name; extrapolations say
"extrapolated". Numbers in prose are kept out of the plan files; use the
tables.

## 6. When to stop and ask

Stop and ask the PI, rather than guess, when: a Settled decision looks
wrong; a run needs more than the week's budget; a checkpoint or output
directory must be removed; a result contradicts the section file's
interpretation rules; or the task as written is impossible with what is
staged on the cluster.
