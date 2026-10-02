#!/bin/bash
#SBATCH --account=ag_iame_gaudecker
#SBATCH --partition=sgpu_short
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --job-name=pytask-pylcm-inv-5b-lifetime
#SBATCH --output=%x-%j.out
# Stage 5B arms on reduced3 ACA, unbudgeted, one process per arm, ARM_ORDER rotated
# across repeats. Required: PYLCM_DIR, COMMIT (40-hex), ACA_SLURM_SRC, OUT_ROOT,
# DRIVER_DIR. Optional: PRECISION (64|32), ARM_ORDER (space-separated).
set -euo pipefail
export PATH=$HOME/.pixi/bin:$PATH
cd "$PYLCM_DIR"
head=$(git rev-parse HEAD)
echo "host $(hostname) job $SLURM_JOB_ID head $head dirty $(git status --porcelain --untracked-files=no | wc -l)"
if [ "$head" != "$COMMIT" ]; then echo "HEAD DOES NOT MATCH REQUESTED COMMIT"; exit 98; fi
flag=""; [ "${PRECISION:-64}" = "32" ] && flag="--fp32"
base="$OUT_ROOT/stage5b-fp${PRECISION:-64}-${COMMIT:0:8}-$SLURM_JOB_ID"
dirs=()
for arm in ${ARM_ORDER:-block_major period_major unblocked}; do
  pixi run --frozen -e benchmarks-cuda12 python "$DRIVER_DIR/stage5b_lifetime.py" \
    --arm "$arm" --out "$base/$arm" --aca-slurm-src "$ACA_SLURM_SRC" $flag || echo "arm $arm exit $?"
  dirs+=("$base/$arm")
  rm -rf "/tmp/jax-cache-$SLURM_JOB_ID-stage5b-$arm"
done
pixi run --frozen -e benchmarks-cuda12 python "$DRIVER_DIR/compare_stage5b.py" "${dirs[@]}"
