# HPC clusters: partitions, QOS, resources, limits

Reference for the two SLURM clusters reachable from this account. Nothing here
is specific to any project — it describes the machines, how to ask them for
resources, and the ways each one differs from a textbook SLURM site.

All figures were read from the clusters themselves (`scontrol`, `sacctmgr`,
`sinfo`, `nvidia-smi`) rather than from vendor documentation. Where a number
could not be verified it says so.

## Which cluster

| | **sardine** | **marenostrum5** |
|---|---|---|
| ssh alias | `artemis` | `mn5` (data: `mn5transfer`) |
| Scale | 4 nodes, 27 GPUs | 1120 GPU nodes, 4480 GPUs |
| Longest job | 7 days (2 GPUs) | 3 days (up to 100 nodes) |
| Internet on nodes | yes | **no, none at all** |
| Interactive work | yes, `srun` straight in | discouraged; 2 h cap |
| Best for | iteration, debugging, single-node jobs, anything needing the network | large multi-node runs, long batch jobs |

Rule of thumb: develop on sardine because you can reach the internet and get a
GPU in seconds; run big or long things on marenostrum5 because it has 1120 GPU
nodes and you will never queue behind yourself.

---

# sardine

`ClusterName=sardine-cluster`, SLURM 21.08.5. Log in with `ssh artemis`; the
login node is also a compute node, so **do not run heavy work on the shell** —
submit it.

Note `nyx` is part of the same environment but is **not** a SLURM node (no
`sinfo`, no `sbatch`). It is a storage/dev box that serves `/mnt/scratch-nyx`.
Use it for CPU-heavy prep work that would otherwise clog a login node.

## Hardware

| Node | Partition | CPUs | RAM | GPUs |
|---|---|---|---|---|
| `artemis` | `a6000` | 112 | ~1.0 TB | 7 × RTX A6000, **49140 MiB** each |
| `poseidon` | `a6000` | 112 | ~1.0 TB | 8 × RTX A6000, 49140 MiB each |
| `dionysus` | `h100` | 96 | ~1.0 TB | 4 × H100 |
| `hades` | `h200` | 192 | ~2.0 TB | 8 × H200 |

A6000 memory was measured with `nvidia-smi`. H100/H200 capacities are **not
verified here** — both nodes were fully allocated at the time of writing. Do not
assume the nominal SKU figure: marenostrum5's H100s are the 64 GB variant, not
80 GB, so check with `nvidia-smi` inside your first job rather than guessing.

## Partitions

Three, one per GPU type: `a6000`, `h100`, `h200`. **There is no default
partition and no default time limit** — every partition reports
`Default=NO`, `MaxTime=UNLIMITED`, `DefaultTime=NONE`. Always pass
`--partition`.

## QOS — this is where time limits come from

Because the partitions impose no limits, the QOS is the only thing bounding your
job. There is also **no default QOS on the association**, so pass `--qos`
explicitly.

| QOS | Priority | Max wall | Max GPUs/user | Max jobs/user |
|---|---|---|---|---|
| `gpu-debug` | 20 | 1 h | 6 (A6000 only) | 1 |
| `gpu-short` | 10 | 4 h | 4 | 4 |
| `gpu-medium` | 5 | 2 days | 4 | 3 |
| `gpu-long` | 2 | 7 days | 2 | 2 |
| `gpu-h100` | 10 | 2 days | 4 | 2 |
| `gpu-h200` | 10 | 2 days | 4 | 4 |
| `cpu` | 10 | unlimited | 0 | 8 |
| `cpu-debug` | 11 | 1 h | 0 | 4 |

**Priority runs inverse to length** — 20 for a 1 h debug job down to 2 for a
7 day one. Asking for less time genuinely starts sooner. Pick the shortest QOS
that fits and plan to resume rather than reserving a week up front.

`gpu-h100` / `gpu-h200` are separate from the generic tiers: to use those nodes
you need the matching QOS, and both cap you at 2 days.

`normal` and `admin` exist on the cluster but are not granted to this account.

## Submitting

```bash
# Interactive GPU shell, 1 hour
srun --partition=a6000 --qos=gpu-debug --gres=gpu:1 \
     --cpus-per-task=8 --mem=64G --time=01:00:00 --pty bash

# Batch, 4 × H100 for a day
sbatch --partition=h100 --qos=gpu-h100 --gres=gpu:4 \
       --cpus-per-task=32 --mem=256G --time=1-00:00:00 job.sh
```

Request GPUs with `--gres=gpu:N`, optionally by model (`--gres=gpu:a6000:2`).
Nodes are shared, so ask for the CPUs and memory you need — you will not be
given the whole node.

## Storage

| Path | Size | Notes |
|---|---|---|
| `/mnt/home` | 1.6 TB | **GlusterFS — slow.** Code and configs only; never write datasets, checkpoints or logs here |
| `/mnt/scratch-nyx` | 196 TB | The big one. Served by `nyx` over NFS |
| `/mnt/scratch-artemis` | 56 TB | Local to `artemis`. Often near-full |
| `/mnt/scratch-hades` | 18 TB | Local to `hades` |
| `/mnt/scratch-dionysus` | 1.8 TB | Local to `dionysus` |
| `/mnt/data-{artemis,poseidon,hades}` | 3.5–7 TB | Per-node data volumes |

Every `scratch`/`data` mount is exported from one machine and NFS-mounted on the
others, so a node reading its *own* scratch is much faster than reading a peer's.
For I/O-bound jobs, put the data on the node you will run on. Check `df -h`
before writing anything large — these fill up.

---

# marenostrum5

`ClusterName=marenostrum5`, at BSC. Log in with `ssh mn5`; move bulk data
through `mn5transfer`.

## Hardware (`acc` partition)

Per GPU node:

| | |
|---|---|
| GPUs | 4 × NVIDIA H100, **64 GB** (63.29 GiB usable to CUDA) |
| CPUs | 160 logical = 80 physical cores, hyperthreaded |
| RAM | ~500 GB |

`gpp` nodes are CPU-only with 224 logical CPUs each.

## Partitions

| Partition | Nodes | CPUs/node | Purpose |
|---|---|---|---|
| `gpp` | 6408 | 224 | General purpose, CPU only. **Cluster default** |
| `acc` | 1120 | 160 | Accelerated, 4 GPUs/node |
| `hbm` | 72 | 224 | High-bandwidth-memory CPU nodes |
| `gpdata` | 10 | 224 | Data staging |
| `gpinteractive` | 1 | 224 | Interactive CPU |
| `accinteractive` | 1 | 160 | Interactive GPU |

**You normally do not pass `--partition` here — the QOS selects it.** Submitting
with `--qos=acc_ehpc` and no `--partition` lands the job on `acc`. This matters
because the *default* partition is `gpp` (CPU-only): a GPU job that forgets its
QOS silently goes somewhere with no GPUs.

## QOS

| QOS | Priority | Max wall | Max jobs/user | Max submit/user | Max per job |
|---|---|---|---|---|---|
| `acc_ehpc` | 100 | **3 days** | — | 366 | 100 nodes / 16000 CPUs |
| `acc_debug` | 10000 | 2 h | 1 | 1 | 8 nodes / 1280 CPUs |
| `acc_interactive` | 100 | 2 h | 1 | 366 | 1 node / 80 CPUs |
| `gp_ehpc` | 100 | **3 days** | — | 366 | 800 nodes / 179200 CPUs |
| `gp_debug` | 10000 | 2 h | 1 | 366 | 32 nodes / 7168 CPUs |
| `gp_interactive` | 100 | 2 h | 1 | 366 | 1 node / 64 CPUs |

The **default QOS is `gp_ehpc`**, i.e. CPU-only. Every GPU job must pass
`--qos=acc_ehpc` (or `acc_debug`).

The `*_debug` tiers carry priority 10000 against 100, so they jump the queue —
but `acc_debug` allows exactly **one job submitted at a time** (`MaxSubmitPU=1`),
so it is for a single quick check, not a sweep.

## Submitting

```bash
# 2 nodes × 4 H100, 6 hours
sbatch --qos=acc_ehpc --account=<project> \
       --nodes=2 --gpus-per-node=4 --cpus-per-task=80 \
       --time=6:00:00 job.sh
```

`--account` is required and is the project code you are charged against; a user
may belong to several.

**Shorter walltime requests start sooner.** The backfill scheduler fills gaps
ahead of large reservations, and 6 h requests have consistently started within
minutes where longer ones waited. Request what you need plus margin, checkpoint,
and resume — do not reserve 3 days to avoid resuming.

## Accounting

Billing is by **elapsed** time, not requested time. A job that asks for 6 h and
finishes in 1 h is charged for 1 h, so over-requesting walltime costs queue
position, not budget.

```
CPUTimeRAW      = Elapsed × AllocCPUS          # AllocCPUS counts hyperthreads
physical core-h = CPUTimeRAW / 2
1 GPU-hour      = 20 physical core-hours       # 80 cores / 4 GPUs on an acc node
```

Check the balance with `bsc_acct`, and quotas with `bsc_quota`.

## Storage

| Filesystem | Scope | Typical quota |
|---|---|---|
| `gpfs_home` | per user | 80 GB |
| `gpfs_projects` | per group | ~19.5 TB |
| `gpfs_scratch` | per group | ~19.5 TB |

Home is small — put outputs on scratch and datasets on projects.

## Things that will surprise you

- **No internet, anywhere.** Neither login nor compute nodes can reach the
  outside. Anything that fetches at runtime (model hubs, experiment trackers,
  package installs) must be pre-staged, and clients must be told to work
  offline. Verified: an HTTPS request from the login node times out.
- **`acc` nodes are allocated whole.** Asking for fewer CPUs still allocates —
  and bills — all 160 per node. Relatedly, `SLURM_GPUS_ON_NODE` reports the
  node's full GPU count regardless of what `--gpus-per-node` requested, so a
  script that derives world size from it will get the wrong answer. Pin the GPU
  count explicitly if a restart has to match the original.
- **SLURM visibility is restricted.** `sinfo` and `scontrol show node` both
  return `Access/permission denied`. `PrivateData=jobs` means `squeue` shows
  only *your* jobs — so an empty `squeue` says nothing about how busy the
  cluster is, and a pending job's `ReqNodeNotAvail` node list is the partition's
  entire node set, not a list of broken nodes. Use `scontrol show partition`,
  which does work.
- **Containers** come from a module: `module load singularity` (3.11.5).

---

# Applies to both

- **Match the QOS to the job, shortest first.** Both clusters reward it, for
  different reasons: sardine gives short QOS higher priority outright,
  marenostrum5 backfills short jobs into gaps.
- **Checkpoint and resume.** Neither cluster will run a job to completion out of
  goodwill: nodes fail, collectives time out, and wall clocks expire. Assume any
  long run takes several allocations.
- **Verify GPU memory in the job, once.** `nvidia-smi --query-gpu=name,memory.total
  --format=csv` costs nothing and prevents sizing work against the wrong number.
- **Never assume a partition or QOS default.** sardine has neither; marenostrum5
  has both and they are the CPU-only ones.
- Useful, portable: `scontrol show partition`, `sacctmgr show qos`,
  `sacctmgr show assoc where user=$USER` (what you are actually allowed),
  `sacct -j <id> -X --format=State,ExitCode,Elapsed,AllocTRES` (what a finished
  job really consumed).
