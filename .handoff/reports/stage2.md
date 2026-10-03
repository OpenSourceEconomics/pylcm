# Stage 2 report: selected value views, transfers and ownership

The Stage 2 subagent wrote this as its hand-back message. The parent saved it here, condensed but complete in substance, because the harness does not let subagents write `.md` files.

## Source identity
- Commit `bb179c5d67175569ac5b27cb0aca79707794eade` on `feat/invariant-state-views`, tree `e9316e77`. Every test run below used that tree.
- Parent: `80c638b7`, the Stage 1 commit re-parented onto main f2b97fbb.
- Pushed by the parent.
- Diff against 80c638b7: 11 files, +2196/−76.

| File | Change |
|---|---|
| `abstract_program_inputs.py` | 12 |
| `core_program.py` | 38 |
| `donation.py` | 8 |
| `scheduler.py` | 21 |
| `value_transfer.py` | +544 |
| `value_views.py` | new, 151 |
| `tests/candidate_certificate/direct_flow.py` | 163 (mode 100755 kept) |
| `sources.json` | 30 |
| `ci-workloads.json` | +6 |
| `test_transfer_catalogue.py` | +202 |
| `test_value_views.py` | new, 1097 |

`execution_plan.py`, `backward_induction.py` and `contract.py` are untouched.

## New private interfaces (`_lcm.execution.value_transfer`)
- **`ValueViewLeaf`**: `SHARED` or `SELECTED`.
- **`CoordinateSelection(state_name, start, width, codes, keep_axis=False)`**: one contiguous interval along a named axis. It keeps the original codes.
- **`ValueViewDescriptor(artifact, leaf, stored_axis_names, stored_shape, dtype, weak_type, consumer_shape, required_sharding, selections=())`**:
  - It is validated, never inferred from lengths. A shared leaf carries no selection and keeps the stored shape.
  - `structure_key` excludes the codes. `identity_key` includes them.
- **`TransferStageKind` / `TransferStage`**: input and output shapes and layouts, the operator, and an `allocates` flag. Stages run in order.
- **`transfer_result_key(*, transfer, generation=None)`**:
  - Without a view it is unchanged, `(target, layout)`.
  - With a view it is generation + artifact + view identity + destination layout.
- **`_select_value_view`**: a single jitted selection executable. The starts are a runtime int32 operand, so a different code does not recompile.
- **Helpers**:
  - `_selection_operands`;
  - `_select_stored_block`;
  - `_fail_if_view_mismatches_transfer`;
  - `_fail_if_selections_invalid`;
  - `_selection_sharding`, which refuses selection along a partitioned axis (that is Stage 4).

## Changed private interfaces
- **`ResolvedValueTransfer`** gains `view=None`, `consumer_shape`, `selects`, `delivers_stored_buffer` and `stages`.
  - `expected_shape` stays the stored shape; the destination layout is checked against the consumer shape.
  - With a view, `kind` is classified from the selected block.
  - `specialization_key` appends `view.structure_key` only when a view is present. Keys without a view are byte-identical, and the version is not bumped.
  - `cost.temporary_bytes` is the per-device sum of fresh buffers, which gives the same numbers as before when there is no view.
- **`resolve_value_transfer(..., view=None)`**.
- **`apply_value_transfer`**: selection runs first, on the stored layout. The fresh block goes to `on_materialized` before the checks. Then the pass-through or `device_put` runs.
- **`core_program.ValueRead.view`**: it must address the read's target. `_validate_transfer_argument_metadata` refuses a mismatch between the read's view and the plan's view. A concrete argument is checked against the stored shape, an abstract one against the consumer shape.
- **`abstract_program_inputs`**: a view read takes the consumer shape and the view's weak typing.
- **`donation.resolve_donations`**: a selected block is always `TRANSFERRED_COPY`, never the donated owner.
- **`scheduler.PeriodTransferCache(..., generation=None)`**: keyed by `transfer_result_key`.

## New module `_lcm.execution.value_views` (not sealed)
- `ValueTransferFootprint` and `plan_value_transfer_footprint`: owner, block and copy bytes, per device.
- `fail_if_value_transfer_exceeds_budget`: refuses before dispatch.
- `lower_value_view_selection`: lowering for HLO and memory inspection.

## Status per plan item (§5)

| Item | Status |
|---|---|
| View descriptor: shared vs selected stated explicitly, no inference | implemented and tested |
| Stored and consumer shapes kept distinct | implemented and tested (`contract.py` needs no change) |
| Select, then communicate, as stages you can inspect | implemented and tested: stage test, HLO census, 4-device bitwise payload |
| Fused lowering | not built |
| Result keys include generation, artifact, codes, dtype and layout; no aliasing between types | implemented and tested |
| Executable keys depend on structure only; the code is a runtime operand | implemented and tested: cache size unchanged when the code or the params change |
| Ownership, liveness, donation and release after the last consumer | implemented and tested standalone. **Deferred to Stage 3:** five sites in `backward_induction.py` still key transfers by `(target, source_sharding)` and size them by `expected_shape`. They must switch to `transfer_result_key` and `consumer_shape`. Until then, no view may reach backward induction. |
| Tight-budget refusal before dispatch | standalone helper, tested. Wiring it into admission is Stage 3. |
| Selection along a partitioned axis | refused (Stage 4) |

## Exit-gate evidence (4 CPU devices, JAX 0.11.1, worktree `src`)
The files are `hlo/summary.json`, `hlo/selection.{stablehlo,compiled.hlo}.txt` and `hlo/control_full_type_gather.compiled.hlo.txt`, produced by `hlo_evidence.py` (rc=0).

The setup: a float32 array of shape (pref_type=3, assets=8, health=2), sharded on assets over devices 0–3. Type 2 is copied to the group of devices {2,3}.
- The delivered block is bitwise equal to `V[2]` and sits only on devices {2,3}, at 64 B per device against 192 B for the full array.
- The selection outputs 16 B per device and allocates no temporary.
- The compiled selection has zero collectives. The full-replication control has 3 all-gathers.
- The only communication is the planned 64 B cross-mesh copy.
- Repo tests in the four-device catalogue pin this down.

## Tests
**Red, before the change:**
- `test_value_views.py`: 56 tests, 55 failures. The one pass is the plain-read control.
- Four-device file: 27 tests, 6 failures.

**Green on the first attempt:** 56/0 and 27/0.

**Certificate**, following `certification.md` with `__pycache__` cleared between steps:
- Repinned `value_transfer`, `core_program`, `scheduler` and `abstract_program_inputs`.
- Added by hand: the field and method surfaces, enum contracts, callable digests and binding counts for the new code.
- The final repin showed no drift; `check_seals`, `verify` and `verify --self-test` all returned rc=0 (pass).
- Re-anchors:
  - The marker `"        return stored"` collided with a new `return stored_sharding`, so that parameter was renamed to `layout`.
  - The `PeriodTransferCache.get` mutation marker moved to `return self._arrays.get(key)`; the mutation itself is unchanged.
- `generate_ci_workloads --check`, `prek run --files`, `prek run ty` and the commit hooks all passed.

**Final runs** (serial battery subagent, tree e9316e77, `final/SUMMARY.txt`):

| Run | Tests | Failures | Errors | Skipped |
|---|---|---|---|---|
| `test_value_views.py`, fp64 | 56 | 0 | 0 | 0 |
| `test_value_views.py`, fp32 | 56 | 0 | 0 | 0 |
| Four-device catalogue, fp64 (fresh process) | 27 | 0 | 0 | 0 |
| Four-device catalogue, fp32 (fresh process) | 27 | 0 | 0 | 0 |
| `tests/execution`, `-n 2` | 1547 | 0 | 0 | 27 |
| `test_distributed` | 51 | 0 | 0 | 0 |
| `test_distributed_placement` | 72 | 0 | 0 | 0 |
| `test_distributed_simulation_value_reads` | 15 | 0 | 0 | 0 |
| `test_continuous_assets_sharding` | 13 | 0 | 0 | 11 |
| `test_distributed_simulation_eight_devices` | 1 | 0 | 0 | 1 |
| Grid-search candidate certificate | 88 | 0 | 0 | 0 |
| Related 17-file set, `-n 2` | 421 | **11** | **17** | 3 |

The red related run already fails on the base:
- Failing nodes (28):
  - `test_external_solver_conformance::test_declared_value_reads_feed_the_core_the_stored_next_values`;
  - `test_grid_search_streaming_value_dependent::test_value_dependent_model_declares_its_required_program_disposition[same-period-reference]` and `[edge-reference-and-gated-target]`;
  - 25 nodes in `test_negm_core_program.py`.
- Every failure is a CPU device 0 vs device 1 placement mismatch, apparently from a multi-device setting that leaks into a worker. Which co-scheduled file sets it was not traced.
- Control A: the three affected files run alone, fresh process, `-n 0`, give 102 tests and 0 failures.
- Control B: base 80c638b7 with the same command gives the same 28 node IDs and the same failure kinds (rc=1, 421/11/17/3).

## How Stage 3 uses this
1. **Build one `ValueViewDescriptor` per preserving edge** in Stage 1's `InvariantComponent.solve`:
   - `artifact`: the target V address;
   - `stored_axis_names`: the target V state order;
   - one `CoordinateSelection`, with `keep_axis=False` at width 1;
   - `required_sharding`: taken from the consumer's planned layout. Do **not** take it from `contract.place_on_regime_devices`, which matches templates by leading shape.
   
   Shared dependencies use `ValueViewLeaf.SHARED`.
2. **Use the same descriptor on both sides**: `ValueRead.view` and `resolve_value_transfer(..., view=...)`. Core resolution refuses a mismatch.
3. **In `backward_induction.py`**, key caches, consumer counts and shared-copy footprints by `transfer_result_key(transfer=..., generation=G)` and `transfer.consumer_shape`. Construct `PeriodTransferCache(generation=G)` with the same token.
4. **Admit each transfer by its stages**: the block on the stored devices plus the copy on the consumer devices.
5. **Blocks of equal shape share** one selection executable and one consumer-core lowering.

## Open fork, and the parent's decision
The generation token can be (a) a per-solve token, or (b) a parameter fingerprint, which is only needed once a cache outlives a solve (5B or cross-call reuse). **Parent decision: (a)** for Stage 3.

## Notes
- The per-transfer record does not yet include two fields from `03-CROSS-SOLVER-EXECUTION.md` §8: the expected transfer count and whether the transfer may overlap compute.
- The planning corpus has no `19-CURRENT-STATUS.md`. The subagent read `03` §§7, 8 and 10 instead.
- The subagent twice used bare `python3`: once for a text edit, once as a no-op. Neither changed results or the commit.
