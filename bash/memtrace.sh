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
procout="${out%.tsv}.proc.tsv"

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
    echo "# job_anon/job_nproc cover ONLY this job's own process tree. Node-wide"
    echo "# columns and the cgroup columns are unreliable where the node is shared"
    echo "# (artemis hades) or where the cgroup path does not resolve, so job_anon"
    echo "# is the column to trust for attributing growth to this run."
    printf 'time\tused\tavail\tcached\tshmem\tdevshm\tcg_anon\tcg_file\tcg_shmem\tnproc\tjob_nproc\tjob_anon\ttop_rss\n'
} >>"${out}"

printf 'time\tpid\tppid\trole\tanon_gb\tfile_gb\tshmem_gb\n' >>"${procout}"

g() { awk -v v="${1:-0}" 'BEGIN{printf "%.1f", v/1048576}'; }   # KiB -> GiB
gb() { awk -v v="${1:-0}" 'BEGIN{printf "%.1f", v/1073741824}'; } # bytes -> GiB

# This job's own processes: the training ranks (matched on the module they were
# launched with) plus their DataLoader workers. Needed because a shared node
# (artemis hades runs several users' jobs at once) makes both /proc/meminfo and
# a list of every python process on the box useless for attribution -- most of
# what they report belongs to somebody else.
job_pids() {
    local roots kids
    roots=$(pgrep -f "melt.training.train" 2>/dev/null)
    [[ -z "${roots}" ]] && return
    kids=""
    for r in ${roots}; do
        kids="${kids} $(pgrep -P "${r}" 2>/dev/null)"
    done
    echo ${roots} ${kids} | tr ' ' '\n' | sort -u | grep -E '^[0-9]+$'
}

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

    # Per-process detail beside the totals. `top_rss` alone cannot say which
    # process is which -- a training rank and its DataLoader worker are both
    # "python" with the same cmdline -- and the answer decides where to look:
    # a rank growing implicates the model/optimizer or pinned batches, a worker
    # growing implicates the Lhotse input pipeline. ppid resolves it (a worker's
    # parent is a rank), and RssAnon/RssFile/RssShmem say what kind of memory it
    # is, which /proc/meminfo can only report node-wide.
    now=$(date +%H:%M:%S)
    job_anon_kb=0
    job_nproc=0
    for pid in $(job_pids); do
        st="/proc/${pid}/status"
        [[ -r "${st}" ]] || continue
        read -r ppid anon file shm < <(awk '
            $1=="PPid:"     {ppid=$2}
            $1=="RssAnon:"  {anon=$2}
            $1=="RssFile:"  {file=$2}
            $1=="RssShmem:" {shm=$2}
            END {print ppid+0, anon+0, file+0, shm+0}
        ' "${st}" 2>/dev/null)
        [[ -z "${anon:-}" ]] && continue
        job_anon_kb=$((job_anon_kb + anon))
        job_nproc=$((job_nproc + 1))
        # `is_worker` is the distinction top_rss cannot make: a training rank and
        # its DataLoader worker are both "python" with the same cmdline, and which
        # one grows decides where to look -- a rank implicates the model, optimizer
        # or pinned batches; a worker implicates the Lhotse input pipeline.
        if pgrep -f "melt.training.train" 2>/dev/null | grep -qx "${pid}"; then
            role="rank"
        else
            role="worker"
        fi
        printf '%s\t%s\t%s\t%s\t%.1f\t%.1f\t%.1f\n' \
            "${now}" "${pid}" "${ppid}" "${role}" \
            "$(awk -v v="${anon}" 'BEGIN{print v/1048576}')" \
            "$(awk -v v="${file}" 'BEGIN{print v/1048576}')" \
            "$(awk -v v="${shm}" 'BEGIN{print v/1048576}')" >>"${procout}"
    done

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${now}" \
        "$(g $((total - avail)))" "$(g "${avail}")" "$(g "${cached}")" \
        "$(g "${shmem}")" "$(g "${devshm:-0}")" \
        "$(gb "$(cg_field anon)")" "$(gb "$(cg_field file)")" \
        "$(gb "$(cg_field shmem)")" \
        "${nproc}" "${job_nproc}" "$(g "${job_anon_kb}")" \
        "${top:-none}" >>"${out}"

    sleep "${interval}"
done
