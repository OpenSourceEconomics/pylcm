# Design: age-dependent continuous-state grids (shape-invariant)

Branch `feat/age-dependent-grids`, on top of `feat/age-specialized`.

## Motivation

A continuous state whose grid **bounds move with age** — the canonical case is an
asset state with an age-dependent borrowing floor `a̲(age)` (Guvenen & Smith 2014,
and any life-cycle model with an age-varying constraint). On a single fixed grid the
floor must span the loosest bound, so at ages where the bound is tighter the grid
cells below it are **unreachable and infeasible** (empty feasible set → `-inf`),
which then poisons the value function by interpolation and propagates backward. An
age-dependent grid whose floor tracks `a̲(age)` removes those cells entirely.

## Key scoping decision: shape-invariant (fixed `n_points`)

The grid may vary its **bounds/nodes** per age but keeps **the same `n_points`
(and grid class) at every age**. This is exactly what a moving floor needs, and it
is the decisive simplifier:

- Every period's V-array keeps the **same shape**, so the rolling-continuation
  topology, the V-array shapes, and the solver-kernel dedup are **unchanged**.
- Only two things become period-aware: the **node values** on the current period's
  state axis, and the **interpolation coordinates** for the continuation (which must
  use the *next* period's grid).

Fully arbitrary per-period `n_points` would force period-indexed V shapes and a
period-indexed rolling template — a much larger change — and is **out of scope for
v1**. (It can be added later; nothing here precludes it.)

## API

Reuse the `AgeSpecialized` marker, now also accepted as a **grid** in
`Regime.states` (its `build(age)` returns a `ContinuousGrid` instead of a function):

```python
from lcm import AgeSpecialized, LinSpacedGrid

states = {
    "assets": AgeSpecialized(
        build=lambda age: LinSpacedGrid(start=a_bar(age), stop=A_MAX, n_points=40),
        signature=lambda age: a_bar(age),   # ages with equal floor share a program
    ),
    ...,
}
```

Contract (validated at `Regime`/`Model` construction):
- every `build(age)` returns the **same grid class** and the **same `n_points`**;
- allowed for **continuous** states (Lin/Log/Irreg/Piecewise); process states and
  discrete states are not age-varying in v1;
- allowed in non-terminal and terminal regimes (terminal resolves at its single age);
- **actions** stay fixed-grid in v1 (only states vary) — a continuation-axis is a
  state, so this covers the borrowing-floor use case.

`signature(age)` is a correctness precondition exactly as for functions: equal
signature => identical grid.

## Threading (reusing the per-period build loops)

The per-period **function** builders already iterate periods with `age` in scope
(`processing.py:_build_Q_and_F_per_period`, `_build_next_state_vmapped`,
`_build_argmax_and_max_Q_over_a_per_period`). Age-dependent grids add per-period
plumbing at these points:

1. **Grid resolution.** A new resolver (mirroring `age_specialization.resolve_tree`)
   turns the `states` mapping into a concrete per-period `{name: Grid}` for a given
   age. Validate the shape-invariance across the horizon once.

2. **Current-period state axis (values).** `create_state_action_space` /
   `_build_base_state_action_spaces` resolve the age-varying state's grid at the
   period's age -> period-t node values feed period-t economic functions. Shape is
   identical across periods, so the V-array topology and rolling template are
   untouched; only the axis *values* differ. `backward_induction.solve` already
   calls the kernel per `(regime, period)`, so it passes period-t's axis.

3. **Continuation interpolation (the crux).** In `get_Q_and_F` /
   `_get_coordinate_finder` (V.py), the interpolation of `V_{t+1}` must use the
   **target regime's period-(t+1) grid**. Since `_build_Q_and_F_per_period` builds
   one program per period-group with `age` known, pass the target's
   period-(t+1) `VInterpolationInfo`. Extend the group signature to fold in the
   `(period, period+1)` grid identities so periods with different bounds do not
   false-share a compiled `Q_and_F`.

4. **Simulation.** The per-period argmax kernel and `_lookup_values_from_indices`
   use the period's resolved grid for the argmax axis and index->value lookup.

5. **Unchanged (shape-invariance):** V-array shapes/shardings, the zero-continuation
   template and its roll, and `GridSearch` kernel dedup topology.

No change to `coordinates.py` (the coordinate math already runs off
`(start, stop, n_points)` / `points`).

## Validation / errors

- `n_points` or grid class differing across ages -> `RegimeInitializationError`.
- age-varying **process** state or **discrete** state -> rejected (v1).
- `to_dataframe(additional_targets=...)` on an age-varying-grid-dependent target ->
  rejected, mirroring the AgeSpecialized rule (published functions use one
  representative age).

## Test plan

- unit: resolver returns the right grid per age; shape-invariance validator;
  signature dedup.
- solve: a 2-state life-cycle model with an age-shrinking floor solves with
  **finite V at every age** where a fixed grid produces `-inf`; compare against the
  free-assets reformulation (same economics) — value functions must match to
  interpolation tolerance.
- simulate: continuous state stays within `[a̲(age), A_MAX]`; policy sensible.
- regression: an age-invariant `signature` reproduces the plain fixed-grid solve
  bit-for-bit (dedup collapses to one program).

## Implementation status / remaining plan

**Done (committed):** merge of origin/main into feat/age-specialized (conflicts
resolved); `_AgeSpecialized` base + rename `AgeSpecialized`→`AgeSpecializedFunction`
+ new `AgeSpecializedGrid`; resolver/validator helpers in `age_specialization.py`
(`resolve_state_grids`, `state_grids_signature`, `has_age_specialized_grid`,
`validate_age_specialized_grids`). All 23 age-related tests green.

**Key simplification confirmed:** the current-period state axis is passed to the
Q_and_F kernel as *runtime* values (shape-invariant ⇒ same trace), so only the
*continuation* (target) grid is a trace-time constant. Therefore:
- state axis → resolve per period in the solve loop (runtime values, no recompile);
- continuation interp → bake period-(t+1) grid into period-t's Q_and_F (per-period
  kernel, grouping signature extended with target grid identity at t+1).

**Remaining wiring (solve, then simulate, then tests):**
1. `lcm/regime.py`: accept `AgeSpecializedGrid` in `states` (typing + treat as a
   continuous state in construction validation; concrete resolution deferred).
2. `model_processing.py`/`model.py`: call `validate_age_specialized_grids(states,
   ages)` at Model construction.
3. `processing.process_regimes`: build representative-resolved regimes (markers →
   `build(representative_age)`) for the invariant machinery (from_regime, get_grids,
   create_v_interpolation_info, create_state_action_space) so they are unchanged;
   build `period_to_regime_to_v_interp` (resolved per period) for the continuation.
4. `_build_Q_and_F_per_period`: extend the group signature with the target regimes'
   `state_grids_signature` at period+1; pass the period-(t+1) VInterpolationInfo.
5. `backward_induction`: per-period state axis (resolve age-varying states' nodes at
   the period's age; pass as runtime values). SolutionPhase gains an optional
   per-period grid table.
6. `simulation`: per-period argmax axis + index→value lookup use the period's grid.
7. Tests: solve finite-V where a fixed grid gives -inf; equivalence to the
   free-assets reformulation; age-invariant signature reproduces the fixed-grid
   solve bit-for-bit; shape-invariance + placement rejections.
