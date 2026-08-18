# Site config template. Copy this to sites/<name>.sh and fill in the values.
#
# A site file is sourced by submit-native.sh / submit-container.sh. It contains
# ONLY `export`s (inherited by the SLURM job) and one SBATCH_ARGS array (passed
# to sbatch). No logic, no experiment args — those live on the CLI / in
# config/train/*.yaml.
#
# Which vars matter depends on the mode:
#   native    -> VENV_PATH, HF_HOME, OUTPUT_DIR, LOCAL_DATASETS_DIR
#   container -> SINGULARITY_IMG, HF_HOME, OUTPUT_DIR, LOCAL_DATASETS_DIR, TMPDIR_HOST
# (setting all of them is fine; each mode ignores what it doesn't use.)

# --- storage (host paths) -------------------------------------------------
# Write these as "${VAR:-default}" so a value exported by the caller wins:
#   OUTPUT_DIR=/my/outputs infra/runners/submit-container.sh <site> …
# On a cluster shared by several accounts that is how each person gets their own
# writable OUTPUT_DIR and TMPDIR_HOST without editing (and desyncing) this file.
export HF_HOME="${HF_HOME:-/path/to/melt-data/hf_cache}"                    # HF cache; bind-mounted to /workspace/hf_cache in container
export OUTPUT_DIR="${OUTPUT_DIR:-/path/to/melt-data/outputs}"               # checkpoints/logs; train YAMLs read ${oc.env:OUTPUT_DIR}
export LOCAL_DATASETS_DIR="${LOCAL_DATASETS_DIR:-/path/to/melt-data/shar}"  # SHAR datasets; train YAMLs read ${oc.env:LOCAL_DATASETS_DIR}
export TMPDIR_HOST="${TMPDIR_HOST:-/tmp}"                                   # host tmp; bind-mounted to /workspace/tmp (container mode)

# --- container mode -------------------------------------------------------
export SINGULARITY_IMG=/path/to/melt-data/melt_cuda126.sif
# export SINGULARITY_BIN=singularity                  # optional; apptainer/singularity autodetected otherwise

# --- native mode ----------------------------------------------------------
export VENV_PATH=/path/to/venvs/melt/bin/activate     # python virtualenv activate script

# --- code sync (infra/sync_repo.sh) ---------------------------------------
# Only needed for clusters you push code TO (e.g. air-gapped ones that cannot
# `git pull`). Use an ssh ALIAS, not user@host, so each person can point it at
# their own account via ~/.ssh/config.
# export REMOTE_SSH=<ssh-alias>                       # e.g. mn5
# export REMOTE_REPO=training                         # relative to the remote $HOME

# --- misc -----------------------------------------------------------------
export WANDB_MODE=online                              # use offline on air-gapped sites (+ export HF_HUB_OFFLINE=1)
export MASTER_PORT=60001                              # rendezvous port; bump if it clashes on the node
export MELT_SEED="${MELT_SEED:-42}"                   # forwarded by run_train.sh as an explicit shard_seed override

# --- scheduler ------------------------------------------------------------
# Passed verbatim to sbatch. Set partition/QoS/account/time/nodes/gpus here.
SBATCH_ARGS=(--time=01:00:00 --nodes=1 --gpus-per-node=2 --qos=gpu-h100 --partition=h100)
