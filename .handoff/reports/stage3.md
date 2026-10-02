# Stage 3 report (merge + opt-in type-local GridSearch)

The Stage 3 subagent wrote this report as its hand-back message. The parent saved it here, condensed but complete in substance. Subagents cannot write `.md` files.

## Identity
- Branch `feat/invariant-state-execution`, pushed by the parent on 2026-10-02.
- Step-0 merge `bd99623a648cf9a82f2936a00a8042a1d8ad215c` = Stage 0 4c39d7ef + `origin/feat/invariant-state-views` bb179c5d (Stage 1 80c638b7, Stage 2).
- Stage 3 `1650805e8b97b7df9840f22236cef978664a1546`: 18 files, +1629/−124. `diff.sha256` = `dd1931e2…f21e4`.
- Every process imported `/home/hmg/econ/dev-pylcm/pylcm-invariant/src/lcm/__init__.py`.

## What Stage 3 does
- **Interface.** `ExecutionConfig(invariant_block_widths={"pref_type": 1})` is keyword-only. The default empty `MappingProxyType` leaves the existing route unchanged. Validation:
  - keys must be non-empty strings (TypeError otherwise);
  - widths must be positive exact ints (TypeError or ValueError; beartype rejects floats).
- **Construction refusals,** in `src/_lcm/regime_building/invariant_blocking.py`. Every failing reason is listed. It refuses:
  - more than one state;
  - a width other than 1;
  - a sharded state;
  - an undeclared state;
  - a non-terminal carrier that is not GridSearch, or that has taste shocks, stakeholders, gated edges, same-period refs, or a non-discrete grid for the state.
- **Analysis gate.** After the build, Stage 1's `analyze_invariant_components` and `fail_if_invariant_blocking_is_unsafe(phase="solve")` run. `processing.py` also refuses folded processes and edge-reference reads. Terminal regimes are solved whole.
- **Programs.**
  - `_bound_programs` builds one bound program per code, named `main[pref_type=<code>]`.
  - Each carries `InvariantBinding(state_name, start, code, family="main")`, keeps `main`'s function and argument builder, and has its reads rekeyed via `rekeyed_value_reads`.
  - The bound state has extent 1 on the cell axis.
- **Q_and_F.** The solve-phase Q_and_F drops the bound state's continuation coordinate, using the existing co-map mechanism. Simulate and diagnostics are untouched.
- **Engine.**
  - Bound programs materialise on the block state space and publish a block-shaped output layout.
  - Reads of a stored value that carries the state become Stage 2 selected views (`keep_axis=False`). The required layout is the stored layout minus that axis on the same mesh, otherwise the planned replica or copy destination.
  - The lowering identity uses the family name: one executable for all codes, with the code as a runtime operand.
  - The five transfer-key sites use `transfer_result_key`. The period cache uses a per-solve uuid generation (parent decision).
  - The solve loop runs a unit's bound programs in code order. It assembles the full array with a jitted repeat plus a donated `dynamic_update_slice` pinned to the template sharding, and checks the result against the template.
  - Liveness, ledger and commits stay per regime dispatch.
  - Plan records set `selected_block`.
  - Capture or replay of a blocked regime-period is refused.

## Validation
### Red runs
- `stage3/red/red.{log,xml}`: rc=1. 20 failures, from the missing kwarg or attribute; the 3 width-type tests passed.
- `stage3/red/multi_red.*`: the 8-device tests against bd99623a, rc=1, 5/5 TypeError.
- Three red-file tests were wrong and were corrected:
  - a float width is refused by beartype;
  - `independent_types` carrier periods are 0–2;
  - the sector-model terminal is period 3.

### Green battery
Serial, `cap pixi run --as-is -e tests-cpu pytest … -v --junitxml`. Summary in `stage3/junit/SUMMARY.txt`.

| Run | rc | Tests | Fail | Err | Skip |
|---|---|---|---|---|---|
| `test_invariant_blocking.py` p64 / p32 | 0 / 0 | 25 / 25 | 0 | 0 | 0 |
| `test_continuous_assets_sharding.py` 8-dev p64 / p32 (first run) | 1 / 1 | 18 / 18 | 1 | 0 | 0 |
| same, after the test-side fix | 0 / 0 | 18 / 18 | 0 | 0 | 0 |
| `test_continuous_transfer_admission.py` 8-dev p64 | 0 | 3 | 0 | 0 | 0 |
| execution + regime_building set (16 files) p64 | 0 | 396 | 0 | 0 | 0 |
| solution set (~190 files) + `test_solvers` p64 | 0 | 1776 | 0 | 0 | 1 xfail |
| known flake files alone, `-n 0` | 0 | 102 | 0 | 0 | 0 |
| `test_distributed` / `_placement` / `_lifetime` / `test_transfer_catalogue` | 0 | 51 / 72 / 23 / 27 | 0 | 0 | 0 |
| certificate test files | 0 | 102 | 0 | 0 | 0 |
| `verify.py --self-test` | 0 | 1148 mutations, all rejected | | | |

- **The first-run failure** was `test_shared_native_all_gather_releases_after_both_consumers_are_ready`: `[] == [False, True]`.
  - Cause: the probe used the old `(target, sharding)` key, which the per-solve generation changes by design.
  - Fix (test side): track `transfer_result_key(transfer, generation=cache._generation)`.
  - Rerun results: `junit/sharding8_fixed_p{64,32}.*`.
- **The skip** is the pre-existing xfail `test_nbegm_topology_continuation::test_recurring_jump_period_matches_brute_with_side_faithful_read`.

### What the tests establish
- **Bitwise parity, blocked vs unblocked, at fp32 and fp64,** on three models:
  - `independent_types`;
  - a sector model with `pref_type` second after a moving `sector` axis and a typed terminal;
  - the same model with a type-free terminal.
- Blocks not starting at type 0 are covered.
- On 2 and 8 devices with sharded assets, at widths (1,1) and (3,9), values and shardings are bitwise equal.
- Against the oracle: rtol 1e-5. Simulated DataFrames are equal.
- **Changed params:** they match the unblocked run and differ from the base, so there is no stale reuse.
- **Compile count** equals the unblocked count on all three models.
- **Refusals at construction:** `absent`, `wealth`, width 2, a sharded `pref_type`, and a resetting `sector`.
- **Footprint.**
  - One device: a block's `transfer_workspace_bytes` is the typed terminal's bytes divided by 3.
  - Eight devices: block plus one shard of the block, (C/3)(1 + 1/8). At fp32 that is 324 B against 864 B unblocked.

### Certificate and checks
- `repin_corridors.py --changed-source`: 9 sources, 21 pins.
- `check_seals --fix`, check, anchors, verify, self-test and `ciw --check` all returned rc=0.
- Hand edits to `direct_flow.py`:
  - the `invariant_binding` field on the CoreProgram, MaterializedCoreProgram and ResolvedCoreProgram surfaces;
  - the grid dispatch mutation marker is now `compiled_cores[core_key]`;
  - the solve publication corridor is re-anchored on `output, blocked = _run_dispatch_unit(...)`, hash `c17bbbf8…`, computed with the verifier's helper.
- `direct_flow.py` is still mode 100755.
- The new test file is registered in `ci-workloads.json`.
- `prek` and `ty` returned rc=0, and the commit hooks passed.
- Added `# noqa: PLR0915` to `Model.__init__` and `_resolve_output_layouts_and_lowering_keys`.

### Docs
- `docs/user_guide/tuning.md` gets the section "Solve one invariant code at a time".
- A `CHANGES.md` entry.
- The `ExecutionConfig` docstring.

## Status
**Done:**
- opt-in interface, default unchanged;
- refusals before dispatch;
- bound-program family with no per-type compile;
- selected-view reads;
- per-solve generation;
- consumer-layout `required_sharding`;
- block-level plan record and footprint;
- bitwise parity at both precisions, including multi-device;
- docs.

**Deferred:**
- width B>1 and more than one blocked state, which are refused;
- block-major scheduling across regimes (blocks run inside each regime's unit);
- period-wide copy and scratch admission reservations, which are still summed over all blocks. That equals the unblocked charge and over-reserves; the per-core records do show C/K;
- a view shared by two carrier regimes, which stays cached until both commit;
- replay and capture of blocked periods, which are refused;
- **all Marvin measurements**, including ACA construction with blocking, which has never been tried;
- 3 types on 8 GPUs (sgpu has 4 per node);
- the Stage 0 production analysis (jobs 28020582 and 28021011 still pending).

## Evidence
`.task-evidence/invariant-state/stage3/`:
- `merge/`, `red/`, `dev/`, `cert/`, `cert.sh`, `battery.sh`;
- `junit/`, with SUMMARY, `verify_selftest`, the cap before/after snapshots and provenance;
- `drivers/stage3_arms.py`, which is the Stage 0 driver plus `--invariant-blocking`;
- `commit.log`, `diffstat.txt`, `diff.sha256`;
- `SHA256SUMS`, 98 files.
