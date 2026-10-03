# Stage 3 overhead: helper compiles and per-code host work

The subagent wrote this as its hand-back message; the parent saved it, condensed but complete in substance. Evidence is in this directory, with `SHA256SUMS` over 82 files.

## Identity
- Subagent commit `8fcf64ba` on base 9f03c34e: 6 files, +236/−68; `git diff 9f03c34e 8fcf64ba | sha256sum` = `f04b3c85…3646`.
- The parent replayed it onto the rebased branch (main 90e0c31a) as `ec882386ca8fe0717ed878b8eb8e7a74173c3df4`. The patch is identical (`diff <(git diff 9f03c34e 8fcf64ba) <(git diff HEAD~1 HEAD)` empty). At ec882386: check_seals, check_mutation_anchors, `verify.py --repo-root .` and `generate_ci_workloads --check` all rc=0 (logs `scratchpad/ec88-*.log`). Pushed 2f4bfbc9..ec882386.
- Every subagent process imported `/home/hmg/econ/dev-pylcm/pylcm-invariant/src/lcm/__init__.py`, JAX 0.11.1, host hmg-office under `cap`.

## Result
The exact gate (blocked cold compile requests = unblocked) is not reached. Two per-shape helpers remain: `jit(_select_view_blocks)` and `jit(_write_block)`. Removing them needs a design change; options below. Expected reduced-ACA effect, inferred from the test models and not yet measured on Marvin: 24–25 extra cold compiles down to about 15.

## Cold compile requests on CPU (caches cleared; `diag/count_compiles_{base_9f03c34e,after}.txt`)
| Workload | Unblocked | Blocked 9f03c34e | Blocked now |
|---|---|---|---|
| `independent_types` | 12 | 19 | 14 |
| `sector_typed_terminal` | 13 | 21 | 16 |
| `sector_type_free_terminal` | 13 | 20 | 15 |

Root causes at 9f03c34e:
- `jit(slice)` × K: `block_state_action_space` sliced the concrete grid eagerly. Fixed by building the one-element grid on the host (`np.asarray([binding.code], grid.dtype)`) and `device_put`, committed to the grid's sharding only if the grid is committed (otherwise 2/8-device lanes fail with incompatible devices). A weakly typed grid raises TypeError.
- `jit(convert_element_type)` from `jnp.int32(start)` / `jnp.asarray(starts)`. Fixed: numpy operands.
- `jit(_tile_block)` per output shape. Fixed: the first block writes into `jax.device_put(value_template, may_alias=False)`, which compiles nothing on CPU (`diag/probe_jax.py`; GPU unchecked). Peak unchanged (C + C/K). New guard raises `ExecutionPlanningError` if bound programs do not cover every code.
- `jit(_select_view_blocks)` per stored shape read through a view: remains.
- `jit(_write_block)` per blocked output shape: remains.

## Decision: removing select and write
1. Write-back inside the core: bound core takes a donated full accumulator plus start offset and returns the full value. Changes the bound-program signature; touches output-layout resolution, donation planning, lowering keys, compiler-reservation records, certified corridors.
2. Selection inside the core: core reads the stored value and slices by code. Free on one device; on several devices the core must reshard the block itself (collective inside the core) to keep the no-full-type-gather rule. Changes the Stage 2 "selection is a planned transfer stage" contract; `transfer_workspace_bytes` (pinned at C/K by `test_a_block_moves_one_type_of_each_continuation`) moves into the compiler reservation.
3. Accept the residual: +1 select per stored shape, +1 write per output shape, cold only; warm counts equal.

No cheaper middle ground: a fresh full buffer needs a compile per shape or a full host-to-device copy per warm solve; eager slices/updates compile; select and write have different signatures and timing.

The strict-xfail `test_a_blocked_cold_solve_compiles_exactly_the_unblocked_programs[4 workloads]` keeps the exact comparison in the tree and XPASS-fails once a fold lands.

## Warm overhead (`diag/profile_sector.txt`, `diag/warm_*.prof`)
`structural_resolution` re-runs every program on every solve, about 2 ms host time per program:
| Step | ms / program | Code-dependent |
|---|---|---|
| `materialize_core_program` | 0.44 | yes |
| `_prepare_abstract_program` | 0.72 | yes |
| `resolve_core_program_candidates` | 0.32 | yes |
| lowering keys | 0.24 | partly |
| `workspace_width_candidates` | 0.12 | no |
| `state_action_space` | 0.08 | no |

Hoisted only the code-independent steps: state space once per regime-period; width frontier once per (regime, period, family). Pure in this solve's parameters; nothing cached across solves; unblocked route unchanged. Most of the steady warm gap remains; no Marvin rerun yet.

## Tests
- `test_a_blocked_cold_solve_adds_only_view_selection_and_block_writes` (4 workloads incl. two carriers): nothing missing vs unblocked; extras ⊆ {select, write}.
- `test_a_blocked_solve_plans_each_family_once_per_regime_period` (4 workloads): spy on `SolutionPhase.state_action_space` and `workspace_width_candidates`; blocked = unblocked.
- Red at 9f03c34e (`red/red_all.xml`, rc=1): 8 failures (slice/convert/tile; planning counts (27,10) vs (21,4) and (38,13) vs (30,5)). Exact test with `--runxfail` (`red/compile_red.xml`): 4/4 fail.
- Green (`green/green_all.xml`, rc=0): 8 passed, 4 xfailed.

## Battery (`junit/SUMMARY.txt`; every run rc=0, 0 failures, 0 errors)
| Run | Tests | Skipped |
|---|---|---|
| `test_invariant_blocking.py` p64 / p32, `-n 2` | 39 / 39 | 4 / 4 (strict xfails) |
| `test_continuous_assets_sharding.py` 8-dev p64 / p32, `--full-suite -n 0` | 18 / 18 | 0 |
| `test_continuous_transfer_admission.py` 8-dev p64 / p32 | 3 / 3 | 0 |
| `tests/execution` p64, `-n 2` | 1560 | 27 (topology self-skips) |
| scheduler / liveness / donation / footprint set | 165 | 0 |
| `test_distributed` / `_placement` / `_lifetime`, `-n 0` | 51 / 72 / 23 | 0 |
| certificate files | 102 | 0 |
| `verify.py --self-test` | rc=0 | |

Bitwise parity blocked vs unblocked at fp32 and fp64 on 1, 2 and 8 devices. Caveats: the first topology run used `--ci-policy=full` (launcher refused, rc=4), rerun with `--full-suite`; the first sharding run failed 5/18 on the commitment bug, logs in `topology-fail/`; the scheduler, distributed and certificate runs and the self-test predate the final one-line `invariant_blocks.py` change (not a sealed source; seal check passes).

## Certificates
`repin_corridors.py` rewrote 4 pins (`_select_view_blocks`, `_selection_operands`, the `value_transfer.py` module surface, `_resolve_output_layouts_and_lowering_keys`). `check_seals --fix`, check, anchors, verify, self-test and `generate_ci_workloads --check` all rc=0 with `__pycache__` cleared between steps. No hand edits; `direct_flow.py` mode 100755. prek and ty rc=0.

## Open
- Fold decision (options 1–3).
- Marvin: compile logs for the template copy and host-built grid on GPU; does the blocked cold count drop to about +15.
