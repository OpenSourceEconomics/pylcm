---
title: Run invariant components as independent jobs
---

# Run invariant components as independent jobs

`lcm.component_jobs` divides a block-major solve by the original codes of its invariant
state. Each job runs the existing engine on its assigned codes through every period.
Jobs can run in separate processes or on separate nodes without communicating during the
solve. They publish fragments in a shared directory; the collector verifies them and
returns one complete solution and, optionally, one complete simulation.

The functions manage the plan and results. Your launcher starts the worker processes,
supplies their job indices, waits for them to succeed, and starts the collector. They do
not allocate nodes or create a distributed JAX runtime.

## Build the same block-major model in every process

Continue the model definition in
[Solve, simulate and release one code at a time](tuning.md#solve-simulate-and-release-one-code-at-a-time).
Reuse its complete economic declarations, grids and parameters:

```python
from lcm import ExecutionConfig, InvariantBlockSchedule, Model

model = Model(
    regimes=regimes,
    ages=ages,
    regime_id_class=RegimeId,
    initial_regimes=initial_regimes,
    execution_config=ExecutionConfig(
        invariant_block_widths={"pref_type": 1},
        invariant_block_schedule=InvariantBlockSchedule.BLOCK_MAJOR,
        device_memory_bytes=None,
    ),
)
```

This route requires one invariant discrete state, width one, carried by every regime,
and the `BLOCK_MAJOR` schedule. Combined solve/simulation also requires the forward
phase to group subjects by that state and to run without a device memory budget. The
other admission restrictions in the linked tuning guide still apply.
`device_memory_bytes=None` disables memory-budget admission; it does not establish that
the workload fits the available hardware.

Use the same model, parameters, precision and pylcm build for planning, workers and
collection. Workers and the collector must also match their recorded execution choices:
log level, JIT mode, action partitions, JAX/JAXlib versions, `XLA_FLAGS`, installed
native-build fingerprint, backend, device kind and count, and device-memory budget.
These execution records are separate from the mathematical solution identity. Device IDs
are local to each process. Also match the ambient JAX settings `jax_disable_jit`,
`jax_default_prng_impl`, `jax_random_seed_offset` and `jax_threefry_partitionable`
across workers, collector and reference.

(plan-once)=

## Plan once

This example assumes the declared `pref_type` grid has the original codes `0`, `1`, and
`2`. It assigns one code to each job, without rebuilding or renumbering the grid:

```python
from pathlib import Path

from lcm.component_jobs import ComponentJobPlan, plan_component_jobs

directory = Path("component-campaign")
seed = 7

plan: ComponentJobPlan = plan_component_jobs(
    model=model,
    params=params,
    directory=directory,
    assignment=((0,), (1,), (2,)),
    initial_conditions=initial_conditions,
    seed=seed,
)
```

The directory must be absent or empty. Supply exactly one of `assignment` and `n_jobs`;
`n_jobs=3` partitions the grid codes into three contiguous groups, with larger groups
first. An explicit assignment must cover every original code exactly once in nonempty
jobs. `plan.codes` reports the full grid order and `plan.jobs` the code tuple for each
zero-based job index.

Give both `initial_conditions` and `seed` to solve and simulate, or omit both for a
solve-only campaign. Every worker receives the **same full population in the same row
order**, including rows belonging to other jobs. The engine selects the assigned
subjects itself, retaining each subject's global row and random keys from the
full-population seed. Do not split the input frame or seed workers independently. A code
with no subjects still contributes all of its solution values. The full population must
be nonempty.

The population digest covers column names, dtypes, shapes and contents; it does not
authenticate a DataFrame's index. If an application uses external subject IDs, retain
their mapping separately and use the same canonical row order in the campaign and its
single-process reference.

(run-one-worker-per-job)=

## Run one worker per job

Rebuild the same model and parameters in each worker, load the shared plan, and use the
job index supplied by your launcher. For example, the worker for job zero runs:

```python
from lcm.component_jobs import load_component_job_plan, run_component_job

plan = load_component_job_plan(directory=directory)
fragment = run_component_job(
    model=model,
    params=params,
    directory=directory,
    job=0,
    initial_conditions=initial_conditions,
    log_level="off",
)
```

Run the same call with `job=1` and `job=2` in the other workers. Omit
`initial_conditions` for a solve-only plan. Workers read the simulation seed from the
plan. `max_compilation_workers` optionally caps parallel compilation threads on workers
and the collector.

The job is the retry unit. Its fragment is published atomically after success; rerunning
that job replaces its fragment. A failure during solve, simulation or publication is
re-raised and a failure record is written when possible. A successful retry removes that
record. Wait for all workers to finish before collecting, and avoid simultaneous retries
of the same job.

(collect-the-complete-result)=

## Collect the complete result

On matching hardware and with the same model, parameters and execution choices:

```python
from lcm.component_jobs import CollectedComponentJobs, collect_component_jobs

collected: CollectedComponentJobs = collect_component_jobs(
    model=model,
    params=params,
    directory=directory,
    log_level="off",
)
solution = collected.solution
simulation = collected.simulation
```

Collection has no `initial_conditions` argument. It reconstructs the population from the
verified fragments, restores global row order, and returns `simulation=None` for a
solve-only plan. For a simulating plan, `simulation.raw_results` and
`simulation.to_dataframe()` expose the usual complete results. `solution.values`
contains every period, regime and code, including codes with empty subject groups.
Values remain retained on the host and are assembled on access, as in a single
block-major solve.

The collector refuses missing, failed, duplicated, stale or corrupt fragments, incorrect
code coverage, conflicting execution records, invalid value schemas, and incomplete or
duplicated simulation addresses. It does not return a partial result. Retry the named
jobs or make a new plan when campaign inputs change.

Save the complete solution through the existing
[solution persistence API](../reference/runtime_and_results.md#api-standalone-persistence).
If saving a simulation too, save the solution first: `SimulationResult.save` releases
its live solution storage, and a loaded simulation does not carry a `SolutionResult`.
Host memory and shared-storage capacity must accommodate the full retained result.

## Compare with a matched single-process run

The reference uses the same block-major model, complete population, parameters, seed and
execution choices:

```python
reference = model.simulate(
    params=params,
    initial_conditions=initial_conditions,
    seed=seed,
    log_level="off",
)
```

Component collection preserves the full single-process block-major result: compare every
value's coordinates, dtype, shape and bytes, every raw simulation field, and the panel's
schema, index and row order. This equivalence concerns the same blocked engine; the
separate blocked/unblocked value tolerance is described in the tuning guide. Local
correctness checks do not establish GPU production capacity, multi-node timing or
speedup. Validate those on the actual workload and target hardware with a matched
reference.
