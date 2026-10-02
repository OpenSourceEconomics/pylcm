# Stage 8A design: independent node-local component jobs

Worktree `/home/user/pylcm-8a`, branch `feat/invariant-node-local-jobs`, base
`8691e767a9c26cc89e4cd0308c98988bacea79f8` (Stage 5B head). Authoritative plan:
`.handoff/plan.md` §1, §8 (5B), §11 (8A), §12. Builds on
`.handoff/reports/stage5b-design.md` §3, §4 and §7.

Reading limitation (inherited from 5B, re-checked): `agent-guide/execution-and-audits.md`
routes ownership/lifetime work to `/home/hmg/sciebo/pro-audits/...` documents 01-19.
That path does not exist in this container. `plan.md` is the authoritative plan the
parent supplied; documents 01-19 were not read.

## 0. In one paragraph

A *task manifest* (the plan) splits the codes of the blocked state into disjoint jobs and
pins the identity every job must reproduce. A *job* is an ordinary Python call that a
Slurm array task (or a test, in-process) makes with the model and parameters the user's
script builds. It runs exactly the 5B engine restricted to its codes: per code,
`ComponentSchedule.solve_component` -> host retention; with simulation, the 5B combined
route simulates that code's subjects while its values are resident, on the
full-population subject plan. The job writes one checksummed HDF5 *fragment* per job by
atomic rename. The *collector* verifies every fragment against the plan, then feeds the
host blocks into the same `RetainedComponentValues` a single-process block-major solve
builds and returns the same lazy `SolutionResult`; per-job panels are scattered back to
original subject rows. Same per-code programs, same bytes: collected values and panels
are compared to single-process 5B **bitwise**.

## 1. Public surface (opt-in, keyword-only)

New public submodule `lcm.component_jobs` (not re-exported from `lcm`, like
`lcm.persistence`):

```python
from lcm.component_jobs import (
    ComponentJobPlan, CollectedComponentJobs,
    plan_component_jobs, run_component_job, collect_component_jobs, load_component_job_plan,
)

plan = plan_component_jobs(model=model, params=params, directory=Path("jobs"), n_jobs=3,
                           initial_conditions=initial, seed=7)        # launcher, once
run_component_job(model=model, params=params, directory=Path("jobs"), job=i,
                  initial_conditions=initial, log_level="off")        # each array task
collected = collect_component_jobs(model=model, params=params, directory=Path("jobs"),
                                   log_level="off")                   # collector
collected.solution   # complete lazy SolutionResult (5B store)
collected.simulation # complete SimulationResult or None
```

- `plan_component_jobs` takes `n_jobs` *or* `assignment` (tuple of code tuples), not
  both. `n_jobs` splits the codes in grid order into contiguous, near-equal groups
  (`np.array_split` sizes, larger groups first). Simulation is planned by passing
  `initial_conditions` and `seed` together (both or neither).
- Every function refuses with `ExecutionPlanningError` naming the failed condition and
  the remedy (§7). Corrupt or partial fragment bytes raise `SolutionIntegrityError`
  (the archive's integrity vocabulary).
- The model must already be a block-major model
  (`ExecutionConfig(invariant_block_widths={...: 1},
  invariant_block_schedule=BLOCK_MAJOR)`); simulation additionally needs what 5B needs
  (grouping certified, `device_memory_bytes=None`).

Alternatives rejected: `Model.solve(jobs=...)`/new `Model` methods (grows the certified
`Model` surface; jobs are a launcher concern); top-level exports (adds index rows for a
distributed-only feature); a CLI entry point (the job must load the model the way the
user script does, so the user script *is* the entry point; the driver calls the Python
function).

## 2. Task manifest (`<directory>/plan.json`)

Written once by `plan_component_jobs` through temp + fsync + rename (+ directory fsync).
The directory must be empty or absent (refused otherwise: remedy, use a fresh
directory), so a plan never adopts fragments of another plan. Schema
(`format = "pylcm-component-job-plan"`, `format_version = 1`):

| Field | Meaning |
|---|---|
| `plan_id` | random UUID4; distinguishes two plans with otherwise equal content |
| `identity.model_fingerprint` | `Model._model_fingerprint(flat_params)` (durable, cross-process) |
| `identity.params_fingerprint` | solution-relevant canonical params (`Model._params_fingerprint`) |
| `identity.flat_params_sha256` | every canonical param (`fingerprint_flat_params`), so simulate-only params are pinned too |
| `identity.program_fingerprint` | `Model._program_fingerprint` (lowered-program facts) |
| `identity.precision` | `"float64"`/`"float32"` from `jax_enable_x64` |
| `identity.pylcm_version` | `_lcm.version.__version__` (carries the commit for dev builds) |
| `identity.pylcm_source_sha256` | digest of every `*.py` of the imported `lcm` and `_lcm` packages (path + bytes), so a dirty or different checkout is caught even when the version string is stale |
| `identity.solve_config` | device-independent execution choices: blocked state + width, schedule, `axis_widths`, `axis_widths_by_regime`, `sharded_states`, `covered_axes`, `simulation_sharding`, `donate_buffers` |
| `state_name`, `codes` | blocked state and every code in grid order |
| `jobs` | list of code lists; disjoint, complete, nonempty, each in grid order |
| `simulation` | `null`, or `{seed, n_subjects, initial_conditions_sha256}` |

The plan digest is the SHA-256 of the file bytes; every fragment records it.

Device-dependent facts (backend, device kind/count, resolved budget) are *not* in the
plan: the launcher usually runs on a login node. They are recorded per fragment (§3)
and must agree across fragments.

## 3. Fragment format and writer (`<directory>/fragments/job-NNNN.h5`)

One HDF5 file per job, reusing the archive's conventions (JSON manifest dataset with a
`sha256` attribute; one dataset per leaf; per-leaf SHA-256 framed with the leaf's
logical identity, shape, dtype and C-order bytes). Contents:

- `manifest` (uint8 JSON, attr `sha256`): `format = "pylcm-component-fragment"`,
  `format_version`, `plan_sha256`, `plan_id`, `job`, `codes`, `identity` (recomputed by
  the job; equals the plan's), `execution` (`backend`, `device_kinds`, `n_devices`,
  `device_memory_bytes`), `coordinates` (the solved `(period, regime)` order, the 5B
  retention's publication order), `values` (per code × coordinate: dataset, shape, dtype,
  sha256), `panel` (below or `null`).
- `values/NNNNNN`: exact host bytes of each component block (state axis length 1), as
  5B's `RetainedComponentValues.retain` copies them (`jax.device_get`).
- `panel` (when the plan simulates): `n_subjects` (real population), `subject_batch_size`
  (the chunk width the job dispatched, equal across jobs), `rows` (original rows the job
  simulated, ascending), `regimes` (ordered `[regime, [periods...]]`, the raw-result tree
  order), and one leaf per `(regime, period, field, key)` of `PeriodRegimeSimulationData`
  in field order (`key` names an `actions`/`states` entry), rows in the order of `rows`.

Writer: build the whole file at `.job-NNNN.h5.<random>.tmp` in the same directory, flush,
`fsync`, `os.replace` onto `job-NNNN.h5`, `fsync` the directory (same sequence as
`save_solution_archive`). A rerun replaces the file atomically. On success the job then
removes its own stale `job-NNNN.failed.json`.

Failure: any exception inside the job writes `job-NNNN.failed.json` (plan digest, job,
codes, exception type and message) atomically, removes its temp file, and re-raises. No
fragment is published.

## 4. Job runner

`run_component_job(model=, params=, directory=, job=, initial_conditions=None,
log_level=, max_compilation_workers=None) -> Path`:

1. Read and parse the plan; check `job` is in range.
2. Recompute the identity from `model` + `params` (same calls as `Model.solve`:
   `_process_params`, `validate_transitions`, `_prepare_solution`) and refuse any
   difference, naming the differing keys.
3. Solve-only plan: `ComponentSchedule` (via `Model._component_schedule`, the exact
   object 5B builds) solves each of the job's codes in grid order with
   `solve_component` + `retain_component` (host copy, device buffers deleted).
4. Simulating plan: require `initial_conditions` whose digest equals the plan's, then
   call the public `model.simulate(solution=None, seed=plan.seed, ...)` inside
   `selected_components(codes=job codes)`. The selection is a scoped `ContextVar` that
   `Model._component_schedule` reads, so the 5B schedule built by `simulate` solves
   only these codes, and `simulate` (5B combined route) simulates only these codes'
   subjects (`subject_codes`), on the plan built from the *full* population, then
   returns their rows in original order without publishing a solution.
5. Write the fragment (§3).

No model is reconstructed and no Bellman code is added: the job is the 5B per-code route
with a subset of codes. Several jobs run sequentially in one process (tests): each call
builds its own schedule, executable cache and retention; nothing is cached between
calls except what `Model` already caches for any repeated solve.

Contract extensions (flagged, owner: parent):
- `block_major.ComponentSchedule(codes=...)`: subset of components, in grid order;
  `RetainedComponentValues.retain_host(...)` (host blocks from a fragment, same shape
  checks) and `.host_blocks(code=...)`; `SolvingComponentValues.subject_codes`.
- `block_major.selected_components(codes=...)` context manager and
  `active_component_selection()`.
- `subject_groups.plan_subject_groups(..., selected=...)` and
  `SubjectGroupPlan.rows`; `ComponentValueSource.subject_codes` (protocol).
- `simulate.simulate`: with `component_values.subject_codes` set, simulate only those
  codes' subjects, publish no value store (`{}`), and record the original rows on the
  result (`result._subject_rows`). Certified source: corridor re-pin required.
- `Model._component_schedule` reads the selection; `Model.simulate` does not publish a
  solution when the schedule is a selection. Certified source: re-pin required.

## 5. Collector

`collect_component_jobs(model=, params=, directory=, log_level=) -> CollectedComponentJobs`:

1. Plan: parse; recompute identity from `model` + `params`; refuse a mismatch (a
   collector holding other params/model/pylcm/precision is refused, not trusted).
2. Fragment listing: `job-NNNN.h5` and `job-NNNN.failed.json` names; temp files
   (`.job-*.tmp`, interrupted writes) are ignored. Refusals, all collected and named at
   once:
   - **failed**: a failure record for a job of this plan;
   - **missing**: a job with no fragment (names job and codes);
   - **duplicated**: a fragment and a failure record for one job, a file whose content
     names another job, or a code covered twice (`ComponentCoverage.with_code`);
   - **stale**: a fragment of another plan (`plan_sha256`/`plan_id`), or whose identity
     differs;
   - **mixed execution**: fragments whose `execution` records differ;
   - **schema/partial/corrupt** (`SolutionIntegrityError`): unreadable HDF5 (truncated
     copy), manifest checksum mismatch, unknown format/version, extra or missing
     datasets, a leaf checksum mismatch, a shape/dtype that disagrees with the
     collector's own layout.
3. Values: build the 5B `ComponentSchedule` in the collector (layouts from the
   collector's canonical regimes and devices), `retain_host` every code's verified
   blocks in grid order with the fragments' common `coordinates`, then
   `Model._finish_solution(component_values=retained)`: the result is the 5B
   `SolutionResult` (lazy `_ComponentValueEntry`s, `stored_codes`, save, simulate split
   via `UploadedComponentValues`, materialization admission) bound to the collector's
   model instance.
4. Panel (simulating plan): every fragment has a panel with the plan's `n_subjects` and
   one `subject_batch_size`; rows disjoint and covering `range(n_subjects)`; the same
   tree (`regimes`, leaf keys/dtypes/trailing shapes) in every fragment with rows. Each
   leaf is scattered into a full host array by row and placed with `jax.device_put`;
   `SimulationResult` is built with the collector's canonical regimes, flat params,
   `simulation_output_dtypes`, the collected value store and `_solution`.
5. Interrupted/rerun jobs: an interrupted job leaves only a temp file (ignored) or a
   failure record (refused); a rerun replaces both atomically. The collector never
   deletes anything; directories are caller-owned.

Everything is verified before the result is exposed; the collector holds all value
blocks on the host, exactly as 5B does after a single-process block-major solve.

## 6. RNG and ordering

- 5A grouped simulation draws per-subject keys for the **full** population and gathers
  them by original row (`subject_slice=SubjectRows(rows)`, `n_subjects` = full padded
  population). A job receives the full initial conditions and the plan's seed, so each
  of its subjects reads the same keys; `subject_codes` only drops other codes' chunks.
- The chunk width comes from `Model.simulate` (`grouped_extent` of **all** group sizes),
  identical in every job and in the single-process run, so each chunk is the same
  program on the same rows.
- Gated by the end-to-end test: collected panel bytes == single-process 5B combined
  panel bytes, plus a positive control (changed seed changes the panel).

## 7. Aggregation

`src` contains no likelihood or moment aggregation (`grep -i likelihood src` matches
nothing; "moment" matches unrelated numerical helpers). The contract that applies is the
panel's: rows in original subject order, every leaf's bytes unchanged, and the raw-result
tree order (regimes in model order, periods ascending) equal to the single-process
result. A user's moment computation over `collected.simulation.to_dataframe()` then sees
the single-process frame. Throughput (jobs in parallel) is to be measured separately
from one solve's latency (§9).

## 8. Refused and deferred

Refused (`ExecutionPlanningError`, condition + remedy):
- non-block-major model; `n_jobs`/`assignment` both or neither; `n_jobs` < 1 or > codes;
  an assignment with an unknown, repeated, missing code or an empty job;
- planning into a non-empty directory; only one of `initial_conditions`/`seed`;
- a job index outside the plan; identity mismatch (job or collector); a simulating plan
  run without initial conditions or with different ones; initial conditions given to a
  solve-only plan;
- missing / failed / duplicated / stale / mixed-execution fragments (collector).

Deferred:
- several codes per *compiled* block (width > 1) and several blocked states (5B limit);
- type-free regimes shared across components (5B limit);
- budgeted per-job simulation (5B limit);
- lazy disk-backed collected entries (the collector holds blocks on the host, as 5B);
- replay artifacts / policies in fragments (5B's block-major route publishes values only);
- resuming a half-finished job (a job is the unit of retry);
- a single component across nodes (removed from the plan);
- automatic launching (the driver submits; pylcm does not call Slurm).

## 9. Drivers (Marvin; run by the parent)

- `.handoff/drivers/stage8a/cpu_array_smoke.py` + `cpu_array_smoke.sbatch`: the life
  cycle test model; `plan` step, then a 3-task Slurm array (one code per task, separate
  processes on CPU), then a dependent `collect` step that compares bitwise with an
  in-process single 5B solve+simulate and writes a JSON receipt.
- `.handoff/drivers/stage8a/reduced3_gpu_jobs.py` + `reduced3_gpu_jobs.sbatch`: the
  reduced3 ACA workload (same builder as `drivers/stage5b`), one node (4×A100) per
  preference type; plan on the submit host, array of 3 GPU jobs, dependent collect +
  comparison against the stored single-node 5B run. Records per-job wall, compile,
  fragment bytes, write and collect times.

## 10. Test plan (red first)

New `tests/solution/test_component_jobs.py` (registered in the CI manifest), fp64 and
fp32:

| Test | Reference |
|---|---|
| plan splits codes contiguously; explicit assignment kept; refusals (both/neither, n_jobs out of range, unknown/repeated/missing code, empty job, non-block-major, non-empty dir, seed without initial conditions) | unit |
| plan file round-trips and pins identity | unit |
| job refuses an index outside the plan, changed params, initial conditions differing from the plan | unit |
| solve-only: N jobs in-process, collected values == single-process 5B values, bytes and order; positive control (other params differ) | 5B solve |
| simulate: N jobs in-process, collected panel and values == single-process 5B combined, bytes; populations unbalanced/empty, one-type, balanced; uneven assignment (2 jobs over 3 codes) | 5B combined simulate |
| positive control: a job run with another seed is refused; a panel with a changed seed differs | comparator |
| collector refusals: missing, failed, duplicated (fragment + failure record; renamed file), stale plan, identity mismatch, checksum-corrupt leaf, truncated (partial) file, mixed execution | unit, real fragments |
| an interrupted write leaves a temp file the collector ignores, and a rerun completes the set | unit |
| collected result outlives the jobs and saves/loads like a 5B result; split simulate of the collected result == 5B | 5B |
| one job in a separate Python subprocess, others in-process, collected == 5B | subprocess |
| precision mismatch between plan and job refused | subprocess at the other precision is out of scope locally; unit via identity mutation |

Every test that solves lives in `tests/solution/` (no `slow` marker needed).
