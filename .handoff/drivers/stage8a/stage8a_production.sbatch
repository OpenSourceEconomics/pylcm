#!/bin/bash
#SBATCH --account=ag_iame_gaudecker
#SBATCH --partition=mlgpu_short
#SBATCH --nodes=3
#SBATCH --ntasks=3
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --job-name=pytask-pylcm-inv-8a-production
#SBATCH --output=%x-%j.out
# Preparation draft: the driver and native acceptance are not complete.
set -euo pipefail
export PATH="$HOME/.pixi/bin:$PATH"
: "${PYLCM_DIR:?}" "${PYLCM_COMMIT:?}" "${ACA_MODEL_DIR:?}" "${ACA_MODEL_COMMIT:?}"
: "${ACA_SLURM_DIR:?}" "${ACA_SLURM_COMMIT:?}" "${PIXI_LOCK_SHA256:?}"
: "${DRIVER_SHA256:?}" "${BENCHMARK_HELPER_SHA256:?}" "${OUT_ROOT:?}"
: "${REFERENCE_DIR:?}" "${REFERENCE_RECEIPT_SHA256:?}"
[[ "$REFERENCE_DIR" == /* ]] || {
  echo 'REFERENCE_DIR must be an absolute path' >&2
  exit 98
}
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

assert_source() {
  local directory="$1" expected="$2"
  [[ "$expected" =~ ^[0-9a-f]{40}$ ]] || return 98
  [[ "$(git -C "$directory" rev-parse HEAD)" == "$expected" ]] || return 98
  [[ -z "$(git -C "$directory" status --porcelain)" ]] || return 98
}
cd "$PYLCM_DIR"
driver="$PYLCM_DIR/.handoff/drivers/stage8a/stage8a_production.py"
helper="$PYLCM_DIR/.handoff/drivers/stage5b/stage3_arms.py"
assert_inputs() {
  assert_source "$PYLCM_DIR" "$PYLCM_COMMIT"
  assert_source "$ACA_MODEL_DIR" "$ACA_MODEL_COMMIT"
  assert_source "$ACA_SLURM_DIR" "$ACA_SLURM_COMMIT"
  [[ "$ACA_MODEL_COMMIT" == ad38653696ec366e318ac61b9a81b597a4ecb700 ]] || return 98
  printf '%s  %s\n' "$PIXI_LOCK_SHA256" "$PYLCM_DIR/pixi.lock" | sha256sum --check
  printf '%s  %s\n' "$DRIVER_SHA256" "$driver" | sha256sum --check
  printf '%s  %s\n' "$BENCHMARK_HELPER_SHA256" "$helper" | sha256sum --check
  printf '%s  %s\n' "$REFERENCE_RECEIPT_SHA256" "$REFERENCE_DIR/receipt.json" | sha256sum --check
}
assert_inputs
export PYTHONPATH="$PYLCM_DIR/src:$PYLCM_DIR:$ACA_MODEL_DIR/src:$ACA_SLURM_DIR/src"
export JAX_ENABLE_X64=false ACA_JAX_ENABLE_X64=false
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export XLA_PYTHON_CLIENT_ALLOCATOR=bfc
mapfile -t nodes < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
[[ "${#nodes[@]}" == 3 && "${nodes[0]}" != "${nodes[1]}" && \
   "${nodes[0]}" != "${nodes[2]}" && "${nodes[1]}" != "${nodes[2]}" ]] || exit 98
base="$OUT_ROOT/stage8a-production-fp32-${PYLCM_COMMIT:0:8}-$SLURM_JOB_ID"
[[ ! -e "$base" ]] || exit 98
mkdir -p "$base"
printf '%s\n' "${nodes[@]}" > "$base/allocation-hosts.txt"
scontrol show job "$SLURM_JOB_ID" > "$base/allocation.txt"
findmnt -n -o SOURCE,FSTYPE,TARGET --target "$OUT_ROOT" > "$base/output-filesystem.txt"
common=(--plan-directory "$base/plan" --aca-slurm-src "$ACA_SLURM_DIR/src")

# The immutable matched reference is produced by the separate one-node job.
printf '%s  %s\n' "$REFERENCE_RECEIPT_SHA256" "$REFERENCE_DIR/receipt.json" \
  > "$base/reference-receipt.sha256"
assert_inputs > "$base/identity-start.log"
date -u +%FT%TZ >> "$base/identity-start.log"
export JAX_COMPILATION_CACHE_DIR="/tmp/jax-cache-$SLURM_JOB_ID-stage8a-plan"
srun --nodes=1 --ntasks=1 --nodelist="${nodes[0]}" --gpus-per-node=8 \
  --cpus-per-task=96 --kill-on-bad-exit=1 \
  pixi run --as-is -e benchmarks-cuda12 python "$driver" plan \
  "${common[@]}" --out "$base/planning"
assert_inputs > "$base/identity-after-plan.log"
date -u +%FT%TZ >> "$base/identity-after-plan.log"
planning_receipt="$base/planning/receipt.json"
planning_sha256="$(sha256sum "$planning_receipt")"
planning_sha256="${planning_sha256%% *}"
planning=(--planning-receipt "$planning_receipt" --planning-receipt-sha256 "$planning_sha256")

# One parallel step owns all three nodes. No array, MPI or distributed JAX.
unset JAX_COMPILATION_CACHE_DIR
srun --nodes=3 --ntasks=3 --ntasks-per-node=1 --gpus-per-node=8 \
  --cpus-per-task=96 --kill-on-bad-exit=1 \
  bash -c 'export JAX_COMPILATION_CACHE_DIR="/tmp/jax-cache-$SLURM_JOB_ID-stage8a-$SLURM_PROCID"; exec pixi run --as-is -e benchmarks-cuda12 python "$@"' \
  _ "$driver" run \
  "${common[@]}" "${planning[@]}" --job-from-slurm-procid --out "$base/workers"

# set -e prevents collection after any failed worker step.
assert_inputs > "$base/identity-before-collect.log"
date -u +%FT%TZ >> "$base/identity-before-collect.log"
export JAX_COMPILATION_CACHE_DIR="/tmp/jax-cache-$SLURM_JOB_ID-stage8a-collect"
srun --nodes=1 --ntasks=1 --nodelist="${nodes[0]}" --gpus-per-node=8 \
  --cpus-per-task=96 --kill-on-bad-exit=1 \
  pixi run --as-is -e benchmarks-cuda12 python "$driver" collect \
  "${common[@]}" "${planning[@]}" --reference "$REFERENCE_DIR" \
  --worker-receipts "$base/workers" \
  --reference-receipt-sha256 "$REFERENCE_RECEIPT_SHA256" --out "$base/collection"
assert_inputs > "$base/identity-end.log"
date -u +%FT%TZ >> "$base/identity-end.log"
