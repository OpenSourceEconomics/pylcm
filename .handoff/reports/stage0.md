# Stage 0 report — baselines and attribution

(Written by the Stage 0 subagent; saved verbatim by the parent because subagents cannot write `.md` reports. The parent pushed 4c39d7ef and submitted the production diagnostics job 28021011; see "Parent follow-up" at the end.)

Plan: `.task-evidence/invariant-state/plan.md` (Stage 0, §3 and §12). No algorithm change.

## Source identity
- Base: `f2b97fbb96dac55782adb940cc9f669a5b10d259` (#482 on main), tree-identical to `4674c038`. Evidence taken at 4674c038 therefore counts for f2b97fbb: the reduced2 A1 arm and production job 28020582.
- Stage 0 commit: `4c39d7ef564f4df2fb994db8b3dcf478a61dfbc6`.
- `git diff --stat f2b97fbb 4c39d7ef`: 11 files, 1025 insertions, 23 deletions.

  | File | Lines changed |
  |---|---|
  | `benchmarks/perf_loop.py` | 17 |
  | `benchmarks/warm_solve_phases.py` | 62 |
  | `docs/user_guide/debugging.md` | 23 |
  | `src/_lcm/execution/execution_plan.py` | +233 |
  | `src/_lcm/solution/backward_induction.py` | +78 |
  | `tests/candidate_certificate/direct_flow.py` | 18 |
  | `tests/candidate_certificate/sources.json` | 12 |
  | `tests/ci/ci-workloads.json` | +14 |
  | `tests/execution/test_core_plan_record.py` | +231 |
  | `tests/solution/test_independent_types_oracle.py` | +127 |
  | `tests/test_models/independent_types.py` | +233 |

- The plan record is the only production source change. `src/_lcm/regime_building/` and `fixed_components.py` are untouched.
- Certificate handling:
  - repinned with `repin_corridors.py --changed-source`;
  - resealed with `check_seals.py --fix`;
  - `verify.py` and `verify.py --self-test` both pass.

## Environment
- **Local** (hmg-office):
  - Python 3.14.7 (conda-forge); jax and jaxlib 0.11.1.
  - pixi envs `tests-cpu` and `benchmarks`, on the unchanged #482 lock.
  - `lcm.__file__` = `/home/hmg/econ/dev-pylcm/pylcm-invariant/src/lcm/__init__.py`, recorded in `local/provenance-tests-cpu.txt` and in the perf_loop JSON.
- **Marvin, software:**
  - Clone `~/pylcm-inv`, run with `pixi run --frozen -e benchmarks-cuda12`.
  - jax, jaxlib, jax_cuda12_plugin and pjrt all at 0.11.1.
  - `lcm.__file__` = `/home/hmg308_hpc/pylcm-inv/src/lcm/__init__.py`, with `pylcm_dirty` empty.
  - Every slurm log checks HEAD against the requested SHA.
- **Marvin, hardware:**
  - One A100-SXM4-80GB per reduced job; driver 580.65.06, CUDA 13.0.
  - Topology from `nvidia-smi topo -m`: GPU0 to NIC is NODE, NUMA node 7. Stored in `result.json` under `topology`.
- **Marvin, run settings:**
  - Allocator: default, with PREALLOCATE=true and MEM_FRACTION=0.90.
  - XLA_FLAGS: `--xla_gpu_autotune_level=0 --xla_gpu_enable_command_buffer=`.
  - A fresh compilation cache directory per arm.
  - Precision: fp32 for the A arms (production precision), fp64 for the C64 arm.
  - Retention: `ResultRetention.VALUES`.
  - Logging: `progress` for timed arms; `debug` only in untimed diagnostic runs.
- **Model inputs:**
  - aca-model `ad38653696ec366e318ac61b9a81b597a4ecb700`.
  - The aca-slurm `src` is copied from `~/dense-aca/production6/aca-slurm-b650981`. Its git commit can't be verified, so the content is pinned by `marvin/aca-slurm-src.sha256`.
  - The aca-data bld is pinned by `marvin/aca-data-bld.sha256`.
- **Reduced grids:** `GridConfig(assets=3, aime=3, consumption_dollars=5, wage_res=3, hcc_persistent=3, hcc_transitory=3, savings=200, nbegm certified/one_sided)`.
- **Execution config:** the `ExecutionConfig()` defaults: `device_memory_bytes='device'`, headroom 0.15, exhaustive width search.
- **Params:**
  - reduced2 uses the benchmark params.
  - reduced3 uses the benchmark params plus the production 3-type `consumption_weights`, `discount_factor_by_type` and `coefficients_rra`, with shapes asserted to be (3,).
  - `params_sha256` is recorded per arm.
- **Seeds:**
  - Initial conditions: `n_subjects=4096`, seed 0.
  - reduced3 `pref_type`: rng seed 1.
  - Simulation: seed 20260903.

## Workloads
1. **Tiny independent-type oracle:** `tests/test_models/independent_types.py`. Cake eating with 3 fixed preference types, 11 wealth points and 3 periods, checked against exact enumeration. It is registered in perf_loop and warm_solve_phases.
2. **Reduced ACA:** reduced2 and reduced3, run through `drivers/stage0_arms.py`. The driver extends the #482 `dense_arms_aca.py` and `parse_phase_records`; it is not a new harness.
3. **Production ACA, unchanged:** 4×A100 at fp32. Pending.

## Commands and exit status
Local runs were capped, with the cap.slice state logged per run in `junit/battery-summary.txt`. Below, `R=/home/hmg/econ/dev-pylcm/pylcm-invariant`.

| Local step | Exit |
|---|---|
| Red: `zsh -ic "cd $R && PYTHONPATH=$R/src cap pixi run -e tests-cpu pytest tests/execution/test_core_plan_record.py -v --junitxml=..."` | 2, collection ImportError on `CorePlanRecord` (expected) |
| green1 / green2 / green3 | 1 / 1 / 0 |
| `junit/battery.sh` b1–b4 (pytest `-v -p no:cacheprovider --junitxml`) | 0 |
| b5: `verify.py --repo-root . --self-test` | 0, with `all_controls_sensitive=true` |
| Seal, repin and verify (logs in `local/`) | clean final state |
| `generate_ci_workloads` | 0 |
| `prek run --files <11 files>` | 0 |
| perf_loop `independent_types`, A1 and A2 | 0 |
| warm_solve_phases `independent_types`, with and without `--plan-records` | 0 |

Marvin jobs:

| Job | Name / content | Pin | Node | Outcome |
|---|---|---|---|---|
| 28020581 | s0-reduced, driver v1 | 4674c038 | sgpu011 | reduced2 A1, A2 and C64 exited 0. The reduced3 arms exited 1 on the expected derived-categorical conflict and are excluded. |
| 28020721 | s0-reduced3, driver v3 | f2b97fbb | sgpu026 | reduced3 A1, A2 and C64 exited 0. |
| 28020711 | nsys, driver v2 | — | — | Cancelled by explicit ID; superseded by v3. |
| 28020722 | nsys reduced3, driver v3 | f2b97fbb | sgpu028 | Exited 0. |
| 28020582 | production A1 and A2, fp32, 4×A100 | 4674c038 | — | PENDING. |

Analysis:
- `analyze_stage0.py`; its comparator self-check (equal, 1-ULP and signed-zero cases) passes.
- `drivers/relerr.py` writes `marvin/fp32-fp64-relative.json`.

## Junit counts (tests / failures / errors / skipped)

| Run | Counts |
|---|---|
| red | 1/0/1/0 |
| green1 | 13/10/0/0 |
| green2 | 13/9/0/0 |
| green3 | 13/0/0/0 |
| b1: new test files, fp64 | 22/0/0/0 |
| b2: new test files, fp32 | 22/0/0/0 |
| b3: release_log, solve_phase_records, execution_config, workspace_planning, bounded_workspace_planning, ci_workloads_manifest | 211/0/0/0 |
| b4: grid_search_candidate_certificate, source_certificate_portability, tests/candidate_certificate | 102/0/0/0 |

No failing nodeids. The full suite was not run locally, per house rules.

## Plan item status
1. **Environment record.** Implemented and validated for local runs and for the reduced and Nsight runs on Marvin. The production record comes with job 28020582.
2. **Workloads.**
   - 2a, tiny oracle: implemented and validated. 9 tests at fp64 and fp32 cover values, the simulated policy and bitwise type independence; a perturbed oracle is rejected, as it should be.
   - 2b, reduced2 and reduced3: implemented and validated.
   - 2c, production: driver implemented, not yet measured.
3. **Phase separation.** Validated for the reduced and tiny workloads. The phases are:
   - construction;
   - params_validation, structural_resolution, compilation_waves (with trace, lowering and compile counts) and backward_induction;
   - chunk_planning, simulation_inputs and simulation_chunk;
   - to_dataframe, feather and savez.

   reduced3 was profiled with Nsight outside the timed runs. Production Nsight needs the push.
4. **CorePlanRecord.** Implemented and red-green tested; 13 tests; 12 records on tiny in `local/tiny-plan-records.jsonl`.
   - It is built only under `logger.isEnabledFor(DEBUG)`, from host plan objects and the text of the already-compiled executable, and adds no synchronization.
   - Cost: workspace_selection grows from 0.66 ms to about 70 ms on tiny, so it is diagnostic only.
   - Production plan records need the push.
5. **A/A noise.** Validated on the reduced workloads. No A/B was run.
6. **Exit gate.** Met for the reduced and tiny workloads; open for production.

## Time attribution (reduced, 1×A100, fp32, A1 / A2, h:mm:ss)

| Call | reduced2 | reduced3 |
|---|---|---|
| Construction | 0:00:38 / 0:00:37 | 0:00:34 / 0:00:34 |
| Solve, cold | 0:04:42 / 0:04:36 | 0:04:05 / 0:04:06 |
| Solve, cold: compilation_waves | 0:04:17 / 0:04:12 (182 compiles) | 0:03:45 (181 compiles) |
| Solve, cold: params_validation | 0:00:17 | 0:00:13 |
| Solve, cold: backward_induction | 0:00:07 | 0:00:05 |
| Solve, warm_same_1 | 0:00:11.2 / 0:00:11.4 | 0:00:09.6 |
| Solve, warm_same_1: params_validation | 0:00:05 | 0:00:04.3 |
| Solve, warm_same_1: backward_induction | 0:00:04.4–4.9 | 0:00:03.9 |
| Solve, warm_same_1: compilation_waves | 0:00:00.8, **25 compiles** | 25 compiles |
| Solve, warm_same_2/3 | 0:00:09.0–0:00:10.6 | 0:00:08.1–8.4 |
| Solve, warm_same_2/3: backward_induction | 0:00:03.9–4.8 | 0:00:03.9 |
| Solve, warm_same_2/3: params_validation | 0:00:03.3–4.2 | 0:00:02.8–3.1 |
| Solve, warm_same_2/3: compiles | 0 | 0 |
| Solve, warm_changed | 0:00:10.2 / 0:00:11.1, 0 compiles | 0:00:09.0 / 0:00:08.9, 0 compiles |
| Simulate, cold | 0:03:27 / 0:03:23 | 0:02:51 / 0:02:52 |
| Simulate, cold: chunk_planning | 0:02:48 / 0:02:45 (266 compiles) | 0:02:20 |
| Simulate, warm | 0:00:24.3 | 0:00:19.9 |
| Simulate, warm: simulation_chunk | 0:00:13 | 0:00:12.5 |
| Simulate, warm: params_validation | 0:00:06 | 0:00:02.8 |
| Simulate, warm: simulation_inputs | 0:00:04.7 | 0:00:04.5 |
| Output I/O, warm | to_dataframe 0:00:01.0; feather 0:00:00.02; savez 0:00:00.07–0.3 | to_dataframe 0:00:00.9; feather 0:00:00.02 |

Findings:
- **Cold runs are compile-bound.** Compilation takes about 91% of cold solve time and about 81% of cold simulate time.
- **Warm runs are host-bound.**
  - Every warm solve makes 37 trace requests.
  - The first warm call also makes 25 lowering/compile requests. That is the 1.5–2 s first-warm penalty, and worth investigating.
  - params_validation takes 30–45% of a warm solve.
- **Nsight, reduced3, warm_same_1** (ages 95–51, a window of about 7.8 s out of the 10.5 s call):
  - GPU kernel time totals 2.19 s, so the GPU is mostly idle.
  - `input_reduce_fusion` and `input_reduce_fusion_1`, the Q max-reduction, account for 97.6% of kernel time, with 409 launches each.
  - Device-to-device copies: 44.3 MB over 8665 copies, 14.7 ms. Host-to-device and device-to-host traffic is negligible.
- **2 vs 3 types.** The two workloads ran on different nodes, so the comparison is confounded and is not a scaling result.
- **Tiny model, local CPU:**
  - Warm medians are 0:00:00.207 (A1) and 0:00:00.221 (A2).
  - Cold fingerprints are equal.
  - A changed-params solve is bitwise equal to a fresh solve and differs from the base.
  - backward_induction is 86.8% of warm time.

## Memory attribution (per device)
- **Allocator `peak_bytes_in_use`:**

  | Workload | fp32, solves | fp32, after warm_changed and simulate | fp64 |
  |---|---|---|---|
  | reduced2 | 0.76 GB | 0.81 GB | 0.45–0.54 GB |
  | reduced3 | 0.80 GB | 0.87 GB | 0.60–0.73 GB |

  The fp64 peaks are *lower* than fp32. That is consistent with different widths being selected, but it is not attributed per core, because plan records were off in the timed runs.
- **NVML** reads 73,517 MiB on every arm. That is the preallocated pool, not a demand metric.
- **Per-core breakdown** (resident, replica, gathered, workspace and compiler bytes) is what the new plan record reports. It is validated on tiny; reduced and production debug runs are deferred.

## A/A noise and comparators
- **Values.** A1 and A2 are bitwise equal: 181 of 181 arrays, max ULP 0, for cold, warm_same_1 and warm_changed, on both reduced2 and reduced3.
  - For reduced2 the two arms come from 4674c038 and f2b97fbb, which are tree-identical, so the pair counts as A/A.
  - Within one process, cold equals warm_same_1 bitwise.
- **Panels.** The sha256 of the simulation DataFrames is equal across A1 and A2, for both cold and warm.
- **Positive control.** With bequest_shifter ×1.01, all 181 of 181 arrays differ in every arm.
- **Wall-time noise.**
  - Same call compared across A1 and A2: the relative range is 0.2–11.7% (reduced2) and 0–0.9% (reduced3).
  - Pooled over warm_same calls: 23.7% and 18.9%, dominated by the first-warm recompile.
  - Exclude warm_same_1 from steady-state comparisons.
- **fp32 vs fp64 control** (fp64 rounded to fp32):
  - reduced2: max abs 3.97e-4, max ULP 939,214, max relative error (floor 1) 1.27e-4.
  - reduced3: max abs 720, max ULP 2.2e9, max relative error **5.03**, in `5__tied_nomc_inelig_canwork`, where |V| reaches 1.29e8.
  - In reduced3, 7,440 of 16.9M cells (0.044%), all in periods 0–18, exceed a relative error of 1e-3. Type-1 RRA is 0.99908, and fp32 loses accuracy there.
  - **Consequence:** parity checks for Stage 1 and later with 3 types must be bitwise against the workload's own fp32 off arm.

## Unsupported historical configurations
- The aca benchmark builder rejects a `pref_type` override passed through `derived_categoricals` ("conflicts with model grid"). The v1 reduced3 arms record this (rc=1). The v3 driver builds through `create_model` instead.
- The benchmark params carry 2-entry type vectors, and JAX would silently clamp a third type index. The runs substitute the production 3-type vectors and assert their shapes.
- The #482 S-arm production numbers are inherited context only, not re-measured: cold about 0:29:36 and warm about 0:22:06 on 4×A100, with ages 62–63 at about 0:03:48 each.

## Deferred
1. Production A/A: job 28020582, pending.
2. Production plan records and Nsight: need 4c39d7ef pushed, then `run_diag.sbatch` run in `~/pylcm-inv-diag`.
3. `LCM_CAPTURE_PERIOD` capture/replay: not used.
4. Per-regime kernel attribution in Nsight: blocked. NVTX ranges are named `jit__unnamed_function,program_id=…`, so this needs named jits or a map from program_id to regime.

## Evidence
Everything is under `.task-evidence/invariant-state/stage0/`, which git excludes. `SHA256SUMS` covers 150 files and verifies with `sha256sum -c`.
- `drivers/`: run drivers, sbatch scripts, `analyze_stage0.py`, `relerr.py`.
- `junit/`: XML and logs, `battery.sh`, `battery-summary.txt`.
- `local/`: seal, repin, verify and prek logs; perf_loop JSON; phase logs; `tiny-plan-records.jsonl`; `provenance-tests-cpu.txt`.
- `marvin/`: slurm logs, `result.json` and memstats; `analysis/reduced-summary.json`; `fp32-fp64-relative.json`; `nsys-28020722/*.csv`; aca manifests.
- Large files (npz, nsys-rep, sqlite, feather) stay on Marvin in `~/pylcm-inv-jobs/stage0/`. They are indexed in `marvin/large-files.sha256` and `.sizes`, 34 files.

## Parent follow-up (2026-10-01)
- 4c39d7ef is pushed to `origin/feat/invariant-state-execution`.
- Fresh clone `~/pylcm-inv-diag` on Marvin. Submitted job **28021011** (`pytask-pylcm-inv-s0-diag`) on sgpu_short with 4×A100, 32 CPUs and 240G, `COMMIT=4c39d7ef564f4df2fb994db8b3dcf478a61dfbc6`.
- Production A/A job 28020582 is still pending.
