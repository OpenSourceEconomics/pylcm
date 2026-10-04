#!/bin/bash
#SBATCH --account=ag_iame_gaudecker
#SBATCH --partition=mlgpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --job-name=pytask-pylcm-inv-8a-reference
#SBATCH --output=%x-%j.out
# Owner-admission draft: local publication checks pass; native allocation is pending.
set -euo pipefail
export PATH="$HOME/.pixi/bin:$PATH"
: "${PYLCM_DIR:?}" "${PYLCM_COMMIT:?}" "${ACA_MODEL_DIR:?}" "${ACA_MODEL_COMMIT:?}"
: "${ACA_SLURM_DIR:?}" "${ACA_SLURM_COMMIT:?}" "${PIXI_LOCK_SHA256:?}"
: "${DRIVER_SHA256:?}" "${BENCHMARK_HELPER_SHA256:?}" "${OUT_ROOT:?}"
[[ "$PYLCM_DIR" == /* && "$ACA_MODEL_DIR" == /* && "$ACA_SLURM_DIR" == /* ]] || {
  echo 'PYLCM_DIR, ACA_MODEL_DIR and ACA_SLURM_DIR must be absolute paths' >&2
  exit 98
}
[[ "$OUT_ROOT" == /* && -d "$OUT_ROOT" ]] || {
  echo 'OUT_ROOT must be an absolute path to an existing owner Lustre workspace' >&2
  exit 98
}
[[ "$(findmnt -n -o FSTYPE --target "$OUT_ROOT" --types lustre)" == lustre ]] || {
  echo 'OUT_ROOT must be an absolute path to an existing owner Lustre workspace' >&2
  exit 98
}
cd "$PYLCM_DIR"
driver="$PYLCM_DIR/.handoff/drivers/stage8a/stage8a_production.py"
helper="$PYLCM_DIR/.handoff/drivers/stage5b/stage3_arms.py"
assert_source() {
  local directory="$1" expected="$2"
  [[ "$expected" =~ ^[0-9a-f]{40}$ ]] || return 98
  [[ "$(git -C "$directory" rev-parse HEAD)" == "$expected" ]] || return 98
  [[ -z "$(git -C "$directory" status --porcelain)" ]] || return 98
}
assert_inputs() {
  assert_source "$PYLCM_DIR" "$PYLCM_COMMIT"
  assert_source "$ACA_MODEL_DIR" "$ACA_MODEL_COMMIT"
  assert_source "$ACA_SLURM_DIR" "$ACA_SLURM_COMMIT"
  [[ "$ACA_MODEL_COMMIT" == ad38653696ec366e318ac61b9a81b597a4ecb700 ]] || return 98
  printf '%s  %s\n' "$PIXI_LOCK_SHA256" "$PYLCM_DIR/pixi.lock" | sha256sum --check
  printf '%s  %s\n' "$DRIVER_SHA256" "$driver" | sha256sum --check
  printf '%s  %s\n' "$BENCHMARK_HELPER_SHA256" "$helper" | sha256sum --check
}
assert_inputs
export PYTHONPATH="$PYLCM_DIR/src:$PYLCM_DIR:$ACA_MODEL_DIR/src:$ACA_SLURM_DIR/src"
export JAX_ENABLE_X64=false ACA_JAX_ENABLE_X64=false
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export XLA_PYTHON_CLIENT_ALLOCATOR=bfc
export JAX_COMPILATION_CACHE_DIR="/tmp/jax-cache-$SLURM_JOB_ID-stage8a-reference"
base="$OUT_ROOT/stage8a-reference-fp32-${PYLCM_COMMIT:0:8}-$SLURM_JOB_ID"
[[ ! -e "$base" ]] || exit 98
mkdir -p "$base"
assert_inputs > "$base/identity-start.log"
date -u +%FT%TZ >> "$base/identity-start.log"
scontrol show job "$SLURM_JOB_ID" > "$base/allocation.txt"
findmnt -n -o SOURCE,FSTYPE,TARGET --target "$OUT_ROOT" > "$base/output-filesystem.txt"
srun --nodes=1 --ntasks=1 --gpus-per-node=8 --cpus-per-task=96 \
  --kill-on-bad-exit=1 \
  pixi run --as-is -e benchmarks-cuda12 python "$driver" reference \
  --plan-directory "$base/reference-plan" --aca-slurm-src "$ACA_SLURM_DIR/src" \
  --out "$base/reference"
assert_inputs > "$base/identity-end.log"
date -u +%FT%TZ >> "$base/identity-end.log"
[[ -f "$base/reference/receipt.json" ]] || exit 98
sha256sum "$base/reference/receipt.json" > "$base/reference-receipt.sha256"
