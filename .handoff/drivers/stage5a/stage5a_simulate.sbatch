#!/bin/bash
#SBATCH --account=ag_iame_gaudecker
#SBATCH --partition=sgpu_short
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --job-name=pytask-pylcm-stage5a-simulate
#SBATCH --output=%x-%j.out
# Stage 5A grouped vs ungrouped simulate on reduced3 ACA (cold, warm, panel parity).
# Required: PYLCM_DIR (checkout of the pushed branch), COMMIT (full 40-hex sha),
# ACA_SLURM_SRC, OUT_ROOT. Optional: PRECISION (64|32, default 64), ORDER.
set -euo pipefail
export PATH=$HOME/.pixi/bin:$PATH
cd "$PYLCM_DIR"
head=$(git rev-parse HEAD)
echo "host $(hostname) job $SLURM_JOB_ID head $head dirty $(git status --porcelain --untracked-files=no | wc -l)"
if [ "$head" != "$COMMIT" ]; then echo "HEAD DOES NOT MATCH REQUESTED COMMIT"; exit 98; fi
flag=""; [ "${PRECISION:-64}" = "32" ] && flag="--fp32"
out="$OUT_ROOT/stage5a-fp${PRECISION:-64}-${COMMIT:0:8}-$SLURM_JOB_ID"
pixi run --frozen -e benchmarks-cuda12 python \
  .task-evidence/invariant-state/stage5a/drivers/stage5a_simulate.py \
  --out "$out" --aca-slurm-src "$ACA_SLURM_SRC" --order "${ORDER:-grouped-first}" \
  --end-to-end $flag
rm -rf "/tmp/jax-cache-$SLURM_JOB_ID-stage5a"
