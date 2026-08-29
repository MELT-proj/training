#!/bin/bash
#
# Periodic host-memory tracer. Opt-in, one instance per node, run on the HOST
# (outside the container) by run_train_singularity.sbatch when MELT_MEMTRACE is
# set to a sampling interval in seconds.
#
# Why this exists
# ---------------
# On MN5 every 2-node run that lives past ~45 min saturates at 486-489 GB of
# resident memory against a 500 GB/node cgroup and then dies. It never reports a
# clean OOM, because the site sets:
#
#   ConstrainRAMSpace  = yes                 (cgroup caps the job at ReqMem)
#   JobAcctGatherParams = UsePSS,NoOverMemoryKill
#
# NoOverMemoryKill means SLURM will not kill a job that exceeds its memory, so
# instead of a kill the job simply runs out of allocatable pages and whatever
# asks for memory next fails. Observed faces of the same wall, all on different
# nodes: a DataLoader worker taking SIGBUS on a /dev/shm page ("insufficient
# shared memory"), munged timing out and failing step auth, and a bare
# NODE_FAIL. sacct only records the peak, which is why five NODE_FAILs across
# the MA arm read as unrelated infrastructure noise for five days.
#
# sacct's MaxRSS is a peak with no shape and no attribution: it cannot say
# whether the memory is anonymous or page cache, whether it grows or plateaus,
# or which process holds it. This samples all three.
#
# Output is one TSV per node under logs/, plus a per-process breakdown, so the
# growth curve can be plotted against the training step from the main log.
set -u

interval="${1:-30}"
out="${2:-/dev/stderr}"

kb_field() { awk -v k="$1:" '$1==k {print $2; exit}' /proc/meminfo; }

# The step's cgroup, if it can be found: it is the only way to separate what
# this job holds from what else is resident on a shared node. Path differs
# between cgroup v1 and v2, so probe rather than assume.
cg=""
if [[ -r /proc/self/cgroup ]]; then
    rel=$(awk -F: '$1=="0"{print $3; exit}' /proc/self/cgroup 2>/dev/null)
    [[ -n "${rel}" && -r "/sys/fs/cgroup${rel}/memory.stat" ]] && cg="/sys/fs/cgroup${rel}"
fi

cg_field() {
    [[ -n "${cg}" && -r "${cg}/memory.stat" ]] || { echo ""; return; }
    awk -v k="$1" '$1==k {print $2; exit}' "${cg}/memory.stat"
}

{
    echo "# memtrace on $(hostname -s), interval ${interval}s, cgroup=${cg:-<none>}"
    echo "# sizes GiB except top_rss, which is per-process MiB, largest first."
    echo "# used = MemTotal-MemAvailable."
    printf 'time\tused\tavail\tcached\tshmem\tdevshm\tcg_anon\tcg_file\tcg_shmem\tnproc\ttop_rss\n'
} >>"${out}"

g() { awk -v v="${1:-0}" 'BEGIN{printf "%.1f", v/1048576}'; }   # KiB -> GiB
gb() { awk -v v="${1:-0}" 'BEGIN{printf "%.1f", v/1073741824}'; } # bytes -> GiB

while :; do
    total=$(kb_field MemTotal)
    avail=$(kb_field MemAvailable)
    cached=$(kb_field Cached)
    shmem=$(kb_field Shmem)
    devshm=$(df -k /dev/shm 2>/dev/null | awk 'NR==2{print $3}')

    # Resident set of every python process, largest first: this is what says
    # whether the memory sits in the 4 training ranks or in their DataLoader
    # workers, which points at model/optimizer state vs the input pipeline.
    top=$(ps -eo rss=,comm= 2>/dev/null | awk '$2 ~ /python|pt_|pt_data/ {print $1}' \
          | sort -rn | head -8 | awk '{printf "%.0f ", $1/1024}')
    nproc=$(pgrep -c python 2>/dev/null || echo 0)

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$(date +%H:%M:%S)" \
        "$(g $((total - avail)))" "$(g "${avail}")" "$(g "${cached}")" \
        "$(g "${shmem}")" "$(g "${devshm:-0}")" \
        "$(gb "$(cg_field anon)")" "$(gb "$(cg_field file)")" \
        "$(gb "$(cg_field shmem)")" \
        "${nproc}" "${top:-none}" >>"${out}"

    sleep "${interval}"
done
