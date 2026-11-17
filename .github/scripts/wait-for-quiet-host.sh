#!/usr/bin/env bash
# Wait until the benchmark machine is quiet: no Python process on the GPU, and
# both the 1- and 5-minute load averages below a quarter of the cores. The host
# is shared, and other users' CPU-only jobs leave the GPU idle while slowing
# every host-side benchmark (model construction, compilation, dispatch).
set -euo pipefail
limit=$(( $(nproc) / 4 ))
while true; do
  read -r load1 load5 _ < /proc/loadavg
  if nvidia-smi --query-compute-apps=process_name --format=csv,noheader \
     | grep -qi python; then
    echo "$(date): GPU in use by a Python process, retrying in 60s..."
  elif awk -v a="$load1" -v b="$load5" -v l="$limit" 'BEGIN {exit !(a >= l || b >= l)}'; then
    echo "$(date): load $load1 / $load5 (limit $limit), retrying in 60s..."
  else
    echo "Host is quiet: GPU free, load $load1 / $load5 (limit $limit)."
    exit 0
  fi
  sleep 60
done
