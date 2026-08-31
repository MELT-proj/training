# AGENTS.md Guide for MELT Project

This AGENTS.md file provides guidance for code agents working with this codebase.

## Core Project Structure

- `/melt`: Main source code for the training library
  - `/config.py`: OmegaConf-based configuration (YAML + CLI overrides)
  - `/logging_utils.py`: Central logging utilities (only global-rank-0 logs)
  - `/data/audio/`: Speech data loading utilities
    - `/lhotse/`: Lhotse-based data loading for speech datasets
  - `/modeling`: Core model components (configuration, modeling, processing)
  - `/evaluation`: Evaluation utilities and text normalizers
- `/tests`: Unit tests for the project
- `/bash`: Shell scripts for training and data preparation
- `/infra`: Infrastructure scripts for environment setup and synchronization
- `./venv`: Virtual environment to be activate for running tests.

## Coding Conventions

- Code style is enforced using `ruff`.
- PRs should be focused and minimal. Bugfix PRs should be as brief as possible.
- When writing tests, add them to existing test files when appropriate.
- Use type hints for function signatures.
- Document public functions and classes with docstrings.
- Do not use the `typing` module as it is deprecated starting from Python 3.10.
- I use `uv` for package management, so `pip` calls should take that into account.
- The virtual environment is assumed to live in `./venv`.
  - In container runs, the expected venv path is typically `/workspace/venv`.

## Dependencies

- The project depends on `transformers` for model implementations.
- Speech data loading uses `lhotse` for efficient audio processing.
- Training is managed with `accelerate` and optionally `deepspeed` for distributed training.
- Experiment tracking uses `wandb`.

## Testing

Run tests with:
```bash
pytest tests/
```

To run specific test files:
```bash
pytest tests/test_librispeech_lhotse.py
pytest tests/test_peoples_speech_lhotse.py
```

Tests that need something from the HuggingFace Hub carry `@pytest.mark.hub`.
GitHub CI deselects them with `-m "not hub"` and runs with `HF_HUB_OFFLINE=1`,
so **a new test that reaches the Hub without the marker fails CI** with a
"couldn't connect" error. Add the marker rather than removing the guard.


## Development Setup

1. Create a virtual environment and install the package in editable mode:
   ```bash
  uv venv
  uv pip install -e ".[dev]"
   ```

2. Install pre-commit hooks (optional but recommended):
   ```bash
   pre-commit install
   ```

## Key Components

### Training launch scripts
- `bash/run_train.sh`: THE launcher for all contexts (local / SLURM native / container); activates the venv and runs `accelerate launch python -m melt.training.train`.
- `bash/run_train_singularity.sbatch`: thin SLURM + Singularity/Apptainer shim that runs `run_train.sh` inside the image (bind-mounts datasets/cache/outputs).
- `infra/runners/`: per-site submit wrappers — `submit-native.sh` / `submit-container.sh <site> ...` sourcing `sites/<site>.sh`. See `docs/run_training.md`.

## Environment Variables

- `WANDB_PROJECT`: W&B project name for experiment tracking
- `HF_HOME`: Hugging Face cache directory
- `HF_HUB_CACHE`: Hugging Face hub cache directory (optional; if set, should be consistent with `HF_HOME`)
- `HF_HUB_OFFLINE`: Set to `1` to force HF Hub offline mode
- `TRANSFORMERS_OFFLINE`: Set to `1` to force Transformers offline mode
- `CUDA_VISIBLE_DEVICES`: GPU device selection

Common launch variables:
- `VENV_PATH`: Path to the venv activation script used by `bash/run_train.sh`
- `LOCAL_DATASETS_DIR`: Where SHAR datasets live (host path; bind-mounted in containers)
- `OUTPUT_DIR`: Output directory for checkpoints/logs (host path; bind-mounted to a writable container path)
- `SINGULARITY_IMG`: Path to the `.sif` image for container runs
- `TMPDIR_HOST`: Host tmp directory to bind into container

## Multi-cluster HPC guidance

Three machines are available for building, testing, and running training jobs.
Full topology (partitions, QOS limits, per-node hardware, storage quotas) lives
in the private cross-project reference `g8a9/agents-info`, not here —
`/home/giuseppe/agents-info/HPCs/sardine-and-marenostrum5.md` is the source of
truth; read it before submitting anything non-trivial. This section is
deliberately just a quick "which one do I want" summary. This project's own
HPC procedures (image builds, syncing the repo, launcher conventions, dataset
staging) live in `docs/hpc_runbook.md`.

| | nyx | artemis (sardine) | marenostrum5 (mn5) |
|---|---|---|---|
| role | dev/build box | interactive GPU debugging, small-scale validation | production campaign runs |
| GPUs | none (not a SLURM node) | 7x A6000, 4x H100, 8x H200 via `srun`/`sbatch` | 4x H100 per `acc` node, allocated whole-node only |
| filesystem | local disk | NFS | GPFS |
| internet | yes | yes | **no, none at all** |
| typical queue wait | none (not SLURM) | seconds to minutes | `acc_debug`: fast; `acc_ehpc`: 12-13+ hours observed |

**nyx** — storage box, not a SLURM node (no `sinfo`/`sbatch`/GPUs). Use it for
CPU-heavy prep and the dev/test loop (see the testing section below). Runs
`vm.overcommit_memory=2` (strict commit accounting): an `OSError: Cannot
allocate memory` here is usually the commit ledger, not a real leak — check
`Committed_AS` vs `CommitLimit` in `/proc/meminfo` before assuming a code bug.

**artemis** — real GPUs for interactive debugging and small-scale validation
before committing an MN5 allocation. QOS tiers that matter day to day:
- `gpu-debug`: 1 h wall, 1 job/user — quick sanity checks.
- `gpu-h100` / `gpu-h200`: up to 2 days, 4 GPUs, 2 jobs/user — real debugging
  runs, e.g. reproducing a bug at controlled scale before chasing it on mn5.

Always pass `--partition` and `--qos` explicitly — artemis has no default for
either.

**marenostrum5** — where campaign-scale runs actually happen (multi-node,
whole `acc` nodes, GPFS). No outbound internet on any node, ever: code goes
over via `infra/sync_repo.sh mn5`, never `git pull` on mn5 itself; datasets
and HF checkpoints must be pre-staged too. Two QOS tiers:
- `acc_debug`: 2 h wall, 1 job / 1 submission at a time, but very fast to get
  since few people use it — the right tool for confirming a fix survives
  MN5's actual filesystem/memory semantics (GPFS behaves very differently
  from a dev box for some workloads) before committing to a long `acc_ehpc`
  allocation.
- `acc_ehpc`: the real campaign QOS, up to 3 days / 100 nodes, but much lower
  scheduling priority than `acc_debug` — queue waits of 12-13+ hours have
  been observed. Request the shortest walltime that covers the work: MN5's
  backfill scheduler starts short jobs sooner, and billing is by elapsed
  time, not requested time, so over-requesting only costs queue position.

Rule of thumb: reproduce and fix on artemis (GPUs in seconds, has internet,
small scale), confirm on mn5 via `acc_debug` (same filesystem/memory
semantics as the real run, cheap and fast to get), then commit to `acc_ehpc`
for the actual campaign work.

## artemis- or nyx- specific commands

### Python Environments

Depending on the result of the command `hostname`, you can find specific venvs to be used for testing and debugging purposes. If hostname is `nyx` or `artemis` you can find:

- `/mnt/scratch-artemis/giuseppe/venvs/melt-312/`: fully functional python 3.12 venv, used for lhotse 1.*
- `/mnt/scratch-nyx/giuseppe/venvs/lhotse2`: minimal venv to be used with lhotse 2.*

### Testing

To run the test suite on `nyx` or `artemis`, you can use:

```
cd ~/melt-proj/training
singularity exec --bind /mnt/scratch-nyx,/mnt/scratch-artemis \
  /mnt/scratch-artemis/giuseppe/melt-data/melt_cuda126_lhotse2_td.sif \
  bash -c 'source /workspace/venv/bin/activate
    export PYTHONPATH=/mnt/scratch-nyx/giuseppe/container-extras:$PYTHONPATH
    export HF_HOME=/mnt/scratch-artemis/giuseppe/.cache/huggingface
    export NUMBA_CACHE_DIR=/tmp/numba
    python -m pytest tests/ -v'
```

`NUMBA_CACHE_DIR` is not optional. Without it `tests/test_processing_melt.py`
raises 18 errors of the form `RuntimeError: cannot cache function '__o_fold':
no locator available for file`, because numba tries to write its cache next to
a module inside the read-only image. With it, that file passes.

### When tests fail for reasons that are not your code

**Read this before concluding a test failure is a bug.** These hosts produce
two failure modes that look like code defects and are not, and both have cost
real debugging time.

#### 1. The commit ledger, not free memory

nyx runs `vm.overcommit_memory=2` — **strict commit accounting**. The kernel
refuses allocations against a ledger (`Committed_AS` vs `CommitLimit`, ~141 GB),
not against free RAM. The symptom is scattered failures that move around
between runs:

```
OSError: [Errno 12] Cannot allocate memory
MemoryError
```

typically in `test_check_training_config.py`, `test_lhotse_dataloader.py` and
`test_trainer.py` — the files that fork subprocesses. One has been seen as a
`MemoryError` raised inside pytest's own traceback formatter, which no test
assertion can produce.

**`free -g` and `MemAvailable` are misleading here** and will happily show
100+ GB available while every fork fails. They measure resident pages; the
ledger charges *committed address space at mmap time*, touched or not. Check
the right thing:

```bash
grep -E 'CommitLimit|Committed_AS' /proc/meminfo
```

If `Committed_AS` is near `CommitLimit`, the box is the problem. Measured: a
single pytest process holds **~5.0 GB of VmData against ~0.6 GB of RSS**, an 8x
gap, from torch's arenas, CUDA address reservations, glibc per-thread malloc
arenas (up to 8 x 64 cores here) and 8 MB thread stacks. Eight workers charge
~40 GB for ~5 GB actually used.

#### 2. Orphaned DataLoader workers, which cause (1)

torch DataLoader workers are `daemon=True`, which only covers a **clean**
interpreter shutdown. Nothing runs when pytest is SIGKILLed — including by your
own `timeout -s KILL` — so the workers are reparented and stay resident
forever, each holding its share of the ledger. They then cause more failures,
which cause more timeouts, which leak more workers.

Reaping them has been measured to drop `Committed_AS` from **131 GB to 96 GB**
in one step. Note this is deliberately *not* fixed in the training code: see
issue #97, closed as not planned — cleaning up after a killed process is the
operator's job, not the library's.

#### Best practices

1. **Reap after every pytest run**, unconditionally:
   `pkill -9 -f "python -m pytest"`. Workers do not exit on their own.
2. **Check the ledger before blaming code**, and again before re-running.
   Above ~110 GB, clean up and wait rather than interpreting results.
3. **Never wrap pytest in `timeout -s KILL`** without reaping afterwards. That
   is what created the problem in the first place.
4. **Run one pytest invocation at a time**, and prefer file-by-file over the
   whole suite when the box is shared. Clean up between files.
5. **Look for orphans before starting**:
   `ps -eo pid,etime,rss,cmd | grep "[p]ython -m pytest"`.
6. **Classify failures before reporting them.** Only `OSError: [Errno 12]` and
   `MemoryError` are environmental. **Anything else — `AttributeError`,
   assertion failures, `ValueError` — is a real failure and must be
   investigated**, not waved away as flakiness. When a run mixes both, say
   which is which.
7. You are probably the biggest consumer. When attributing ledger pressure,
   sum **VmData** (from `/proc/<pid>/status`) per user, not RSS — RSS makes
   other users' editors look dominant while your own pytest workers, which
   actually hold the ledger, look small.

## Staleness Warning

This codebase is an academic-driven effort. Hence, we will likely stop updating the files related to past projects. For those files, avoid updating the code and adapting to new conventions. The following list of projects is now stale:

- `projects/iwslt26-metric/`

## Rules

- Avoid running `find` on the file system root "\". If you need to find a file, a venv, or else, just a path to who issued the command.
