#!/bin/bash
#SBATCH --account=ag_iame_gaudecker
#SBATCH --partition=intelsr_devel,intelsr_short
#SBATCH --time=00:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=300G
#SBATCH --output=/home/hmg308_hpc/marvin-jobs/pylcm-inv-5a/%x-%j.out
# Stage 5A gate battery, adapted from ~/marvin-jobs/pylcm-437/battery.sbatch.
# Usage: sbatch --job-name=pytask-pylcm-inv-5a-<set>-fp<P> \
#   --export=ALL,SET=<main|topo>,PRECISION=<64|32>,COMMIT=<40-hex sha> battery5a.sbatch
# main: tests/simulation, invariant blocking, related execution/certificate files, -n 96.
# topo: 8-device and 2-device files at -n 0 under the forced host topology, then the
#       four-device files (which pin their own topology) one at a time at -n 0.
set -u
export PATH=$HOME/.pixi/bin:$PATH
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1
cd $HOME/pylcm-inv-5a || exit 97
echo "host=$(hostname) date=$(date -Is) job=$SLURM_JOB_ID set=$SET precision=$PRECISION"
echo "head=$(git rev-parse HEAD) requested=$COMMIT dirty=$(git status --short | wc -l)"
if [ "$(git rev-parse HEAD)" != "$COMMIT" ]; then echo "HEAD DOES NOT MATCH REQUESTED COMMIT"; exit 98; fi
pixi run --frozen -e tests-cpu python -c "import lcm, jax; print('lcm.__file__', lcm.__file__); print(jax.__version__, jax.devices())"
OUT=$HOME/marvin-jobs/pylcm-inv-5a/${COMMIT:0:8}
mkdir -p $OUT
G=tests/simulation/test_type_grouped_simulation.py
REL="tests/test_random.py tests/test_simulation_candidate_program_certificate.py tests/execution/test_value_views.py tests/execution/test_device_memory_budget_resolution.py tests/solution/test_solution_consumption.py tests/regime_building/test_invariant_components.py tests/test_grid_search_candidate_certificate.py tests/test_source_certificate_portability.py tests/candidate_certificate"
SIM8="tests/simulation/test_independent_outer_cohorts.py tests/simulation/test_subject_parallel.py tests/simulation/test_subject_parallel_runtime.py tests/simulation/test_subject_parallel_model.py tests/test_distributed_simulation_eight_devices.py $G"
SHARD8="tests/test_continuous_assets_sharding.py tests/test_continuous_transfer_admission.py"
FOUR="tests/test_distributed_simulation_value_reads.py tests/test_distributed_entry_allocations.py tests/test_distributed_taste_stream.py tests/test_distributed_lifetime.py"
status=0
run() {
  name=$1; shift
  "$@" -v --junitxml=$OUT/$name.xml > $OUT/$name.log 2>&1
  rc=$?; echo "$name exit $rc"; [ $rc -ne 0 ] && status=1
  grep -o "<testsuite [^>]*>" $OUT/$name.xml || echo "$name: NO JUNIT XML"
}
P=$PRECISION
if [ "$SET" = main ]; then
  run main_p$P pixi run --frozen -e tests-cpu pytest tests/simulation tests/solution/test_invariant_blocking.py $REL --precision=$P -n 96
else
  E8="JAX_NUM_CPU_DEVICES=8 JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=8"
  E2="JAX_NUM_CPU_DEVICES=2 JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=2"
  for p in 64 32; do
    run sim8_p$p env $E8 pixi run --frozen -e tests-cpu pytest $SIM8 --precision=$p -n 0
    run shard8_p$p env $E8 pixi run --frozen -e tests-cpu pytest $SHARD8 --precision=$p -n 0
    run grouped2_p$p env $E2 pixi run --frozen -e tests-cpu pytest $G --precision=$p -n 0
  done
  for f in $FOUR; do
    run four_$(basename $f .py)_p64 pixi run --frozen -e tests-cpu pytest $f --precision=64 -n 0
  done
fi
exit $status
