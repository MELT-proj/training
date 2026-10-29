#!/bin/bash
# Runs on MN5, read-only. Cost facts of training runs from SLURM accounting and the tail/head of their logs.
#   ssh mn5 'bash -s -- cell:role:jobid ...' < collect_cost.sh      one argument per run
# One TSV line per run: cell role state nodes gpus elapsed_h gpu_h steps bar_h train_hours peak_gb trainable_params racks
#   gpu_h        = SLURM elapsed x allocated GPUs: everything the run was billed for (startup, evals, checkpoints included)
#   steps/bar_h  = the last tqdm bar of the log: optimizer steps done and the bar's own clock in hours
#   train_hours  = audio hours the run consumed (last `train_hours/total` of the metrics lines)
#   racks        = the racks (asNNrM) of the job's nodes
# Only the head (parameter count) and the tail (final metrics and bar) of each log are read.
printf 'cell\trole\tstate\tnodes\tgpus\telapsed_h\tgpu_h\tsteps\tbar_h\ttrain_hours\tpeak_gb\ttrainable_params\tracks\n'
for arg in "$@"; do
  IFS=: read -r cell role J <<<"$arg"
  [ -n "$J" ] || continue
  L=$HOME/training/logs/melt-train-container.$J.out
  acc=$(sacct -j "$J" -X -n -P -o State,ElapsedRaw,NNodes,AllocTRES%120,NodeList%80 2>/dev/null | head -1)
  state=$(cut -d'|' -f1 <<<"$acc" | awk '{print $1}')
  elapsed=$(cut -d'|' -f2 <<<"$acc")
  nodes=$(cut -d'|' -f3 <<<"$acc")
  gpus=$(grep -o 'gres/gpu=[0-9]*' <<<"$acc" | head -1 | cut -d= -f2)
  racks=$(cut -d'|' -f5 <<<"$acc" | grep -o 'as[0-9]*r[0-9]*' | sort -u | paste -sd, -)
  params=NA; steps=NA; barh=NA; hours=NA; peak=NA
  if [ -f "$L" ]; then
    params=$(grep -a -m1 'Number of trainable parameters' "$L" | grep -o '[0-9][0-9,]*$' | tr -d ',')
    lastm=$(tail -c 4000000 "$L" | tr '\r' '\n' | grep -a 'train_hours/total' | tail -1)
    hours=$(grep -o "'train_hours/total': '[^']*'" <<<"$lastm" | cut -d"'" -f4)
    peak=$(grep -o "'gpu_peak_gb': '[^']*'" <<<"$lastm" | cut -d"'" -f4)
    # the training bar has max_steps (>= 10000) as its total; evaluation bars are short
    bar=$(tail -c 3000000 "$L" | tr '\r' '\n' | grep -a -o -E '[0-9]+/[0-9]{5,} \[[0-9:]+<' | tail -1)
    [ -n "$bar" ] || bar=$(tr '\r' '\n' < "$L" | grep -a -o -E '[0-9]+/[0-9]{5,} \[[0-9:]+<' | tail -1)
    steps=$(cut -d/ -f1 <<<"$bar")
    barclock=$(sed -E 's/^[0-9]+\/[0-9]+ \[//; s/<$//' <<<"$bar")
    barh=$(awk -v t="$barclock" 'BEGIN { n = split(t, p, ":"); s = 0; for (i = 1; i <= n; i++) s = s * 60 + p[i]; printf "%.3f", s / 3600 }')
  fi
  awk -v cell="$cell" -v role="$role" -v st="$state" -v nodes="$nodes" -v g="$gpus" -v el="$elapsed" -v steps="${steps:-NA}" -v barh="${barh:-NA}" \
      -v hours="${hours:-NA}" -v peak="${peak:-NA}" -v params="${params:-NA}" -v racks="${racks:-NA}" 'BEGIN {
    eh = (el + 0) / 3600; gh = eh * (g + 0)
    printf "%s\t%s\t%s\t%s\t%s\t%.3f\t%.1f\t%s\t%s\t%s\t%s\t%s\t%s\n", cell, role, st, nodes, g, eh, gh, steps, barh, hours, peak, params, racks
  }'
done
