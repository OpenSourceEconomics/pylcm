# Stage 5A: type-aware simulation (stopped early on 2026-10-02 for the handoff)

The local subagent's hand-back, saved by the parent.

## Branch
Branch `feat/invariant-type-aware-simulation` is pushed. It is based on 6ede6e6b, which is #486 before Pro's round-0 fixes:
- 152d0d9d: Group simulated subjects by an invariant state when simulate keeps it fixed (source, tests, pins, CI manifest).
- c7abf345: Document grouped simulation under `invariant_block_widths`.

The diff is 25 files, +1813/−293, sha256 `69d81ff1…9062`, re-verified by the parent. The drivers, sbatch scripts and junit are in `../wip/stage5a/`.

## Design
- **Admission:** `admit_invariant_blocking` in `invariant_blocking.py` replaces `fail_if_invariant_blocking_is_unsafe_for_model`.
  - It still refuses unsafe solves.
  - It attaches a `SubjectGroupingRoute` (state, codes, stored value axis names) to `SimulationPrograms.grouping` only when simulate invariance is certified and no regime has taste shocks, gated edges, a policy replay or an external replay. Otherwise simulation takes today's route.
- **Programs:** `processing.py` builds a `type_local_decision` family through `_build_Q_and_F_per_period(co_map_state_names=...)`, which reads the typed V without the state axis. `forward_decision` dispatches it when grouped. The ordinary `decision` is kept for `lookup_policy`.
- **Subject mapping:** the new module `src/_lcm/simulation/subject_groups.py` has `plan_subject_groups`, which maps an original row to (code, chunk of width w, local row).
  - Each code's rows keep their original order. A short tail repeats its last row, and empty groups dispatch nothing.
  - Codes off the grid, or a missing column, join the first code's group.
  - `positions` restores the original order.
- **RNG:** `generate_simulation_keys(subject_slice=SubjectRows)` builds the identical full-population split and gathers it at the original rows. Nothing is regenerated per group.
- **Transfers:** no second residency manager. `PeriodSimulationReads` takes `grouping` and `code`, and `type_local_transfer` uses `selected_block_view`/`block_layout` from `invariant_blocks.py`, so only the selected block is copied. The budget is charged when `not delivers_stored_buffer`.
- **Result order:** `_restore_subject_order` gathers every leaf with `take_rows(positions)`, an unchanged byte copy.
- **Chunk planning:** grouped runs skip padding to a chunk multiple. Unbudgeted width is min(subject width, largest group rounded to devices); the budgeted frontier is bounded the same way.
- **Profiling and summary:** `profile_simulation_chunk(group_sizes=...)` profiles the grouped route. `SimulationPlanSummary.subject_grouping` reports which state is grouped.
- **Latent-type likelihood:** no likelihood path exists in `src`.

## Junit
| Run | Tests | Fail | Skip |
|---|---|---|---|
| red, base src, new grouped file at p64 | 23 | 23 | 0 |
| red_grouped8, base src | 4 | 4 | 0 |
| green p64 / p32 | 23 / 23 | 0 | 0 |
| grouped8 p64 (devices (0,1) and 8; budgets 2**30 and "device") | 4 | 0 | 0 |
| `tests/simulation` p64 at `-n 2`, before the fix | 4806 | 20 | 35 |
| the 5 failing files, after the fix | 103 | 0 | 0 |

All 20 failures came from beartype rejecting a `ShapeDtypeStruct` passed to `type_local_template(leaf: jax.Array)`. The annotation is now `jax.Array | jax.ShapeDtypeStruct`. The full battery has not been re-run since.

## Parity
Every comparison is bitwise, including panel bytes, signed zeros, NaNs, and raw leaves with dtype and shape:
- grouped vs ungrouped from the same archived solution, across budgeted and unbudgeted runs, typed and type-free dead regimes, and an unbalanced population with an empty group, one type, and a balanced population;
- type-free starts, a population with no `pref_type` column, and changed params;
- end to end, Stage 3 blocked solve plus grouped simulate vs unblocked solve plus ungrouped simulate, on 1, 2 and 8 devices;
- positive controls: changing the seed or the params moves the panel.

The accounting test shows the grouped read selects first: `consumer_shape == stored_shape[1:]`, a third of the bytes. A `Phased` simulate law that resets the state falls back, with `subject_grouping` None and an equal panel.

## Certificates
- **Repin and seal:** `repin_corridors` rewrote 37 pins over 14 sealed sources. The `"programs"` declaration in `_processing_caller_errors` (direct_flow.py) was hand-edited to add the `type_local_*` arguments.
- **Checks:** `check_seals --fix`, `verify` and `ciw --check` all exit 0. prek and ty passed at commit.
- **Self-test:** `verify --self-test` passed, but before the annotation fix, so it needs a re-run.
- **Coverage gap:** the type-local decision family is not traced as its own corridor.

## Still to do
1. Marvin batteries at c7abf345 using `../wip/stage5a/marvin/battery5a.sbatch`:
   - `SET=main` at `-n 96`, fp64 and fp32, with fp32 depending on fp64 (`after:<id>+2`);
   - `SET=topo` on `intelsr_short` with a 4 h limit.
2. `verify --self-test`.
3. The GPU driver, `../wip/stage5a/drivers/stage5a_simulate.sbatch`. It is compile-checked only. Run it with `sbatch --export=ALL,PYLCM_DIR=<clone>,COMMIT=c7abf3458173e9ef725017df8ddf92efc5a2d97f,ACA_SLURM_SRC=<dir>,OUT_ROOT=<dir>,PRECISION=64 stage5a_simulate.sbatch`.
4. Rebase or merge onto #486's 9b36adcb, which adds Pro's F1/F2 fixes. Expect pin conflicts in `tests/candidate_certificate/sources.json`; repin after the merge.
5. Deferred: grouping for taste shocks, gated edges and replay routes. The extra select and gather compiles have not been measured.
