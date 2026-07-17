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

`AgeSpecializedGrid` is the grid-valued sibling of `AgeSpecializedFunction`, accepted
as a **grid** in `Regime.states` (its `build(age)` returns a `ContinuousGrid` instead
of a function):

```python
from lcm import AgeSpecializedGrid, LinSpacedGrid

states = {
    "assets": AgeSpecializedGrid(
        build=lambda age: LinSpacedGrid(start=a_bar(age), stop=A_MAX, n_points=40),
        signature=lambda age: a_bar(age),   # not the dedup key — see below
    ),
    ...,
}
```

Ages whose builders resolve to **equal nodes** share a compiled program; the
hand-written `signature` does *not* decide that for grids (unlike for
`AgeSpecializedFunction`, where a closure cannot be inspected).

Contract (validated at `Regime`/`Model` construction, over the owning regime's
**active** ages only — a builder may be undefined outside them):
- every `build(age)` returns the **same grid class**, the same **`batch_size`**, the
  same **points mode** (concrete vs supplied at runtime), and — for concrete grids —
  the same resolved **node-array shape and dtype**. Only bounds / node *values* may
  vary with age. `batch_size` is an execution-layout knob and a points-mode switch
  breaks build-time node resolution, so neither may move even though neither is
  geometry;
- `build(age)` must be **deterministic and side-effect-free**: the same age is
  resolved more than once (validation, representative regime, per-period map), so a
  stateful factory could be validated as one grid and installed as another;
- allowed for **continuous** states (Lin/Log/Irreg/Piecewise); process states and
  discrete states are not age-varying in v1;
- allowed in non-terminal and terminal regimes (terminal resolves at its single age);
  a regime active at **no** age may not carry one — there is no age to build at;
- **actions** stay fixed-grid in v1 (only states vary) — a continuation-axis is a
  state, so this covers the borrowing-floor use case.

**The invariant is the compiled input signature, not declared metadata.**
`_compile_all_functions` lowers ONE shared kernel against the *representative* state
axis and the solve then feeds it each period's axis, so a differing shape *or dtype* is
rejected by the compiled executable. Declared `n_points` is not sufficient to enforce
this — and is not even guaranteed to exist: only `to_jax()` is on the `Grid` base
contract, so `getattr(grid, "n_points", 0)` silently agreed at `0` for two custom grids
of different actual length. Concrete grids are therefore validated on their **resolved
`to_jax()` array** (the same source of truth `_grid_identity` keys on), and any declared
`n_points` must agree with it. Runtime grids cannot resolve nodes at build time, so only
their declared `n_points` is checked here.

Unlike `AgeSpecializedFunction`, `signature(age)` is **not** the dedup key for grids
and is not a correctness precondition. Grids dedup on their resolved nodes: a grid,
unlike a closure, can be asked what it actually is, so there is no reason to make a
hand-written signature load-bearing for correctness. (The two helpers that fingerprinted
grids via `signature(age)` were unused and have been deleted rather than left to imply a
rule the code does not follow.)

**Identity is bit-preserving, and there is no shortcut for the built-ins.** A key
derived from a grid's *description* rather than its nodes is one restatement away from
disagreeing with the array the kernel is actually handed, and two such disagreements
were real: keying `LinSpacedGrid`/`LogSpacedGrid` on `(start, stop, n_points)` as
Python floats collapsed `-0.0` and `+0.0` (which `jnp.linspace` preserves as different
endpoint bits), and `dtype.str` is not injective over JAX's extended floating types
(`float8_e4m3fnuz` and `float8_e5m2fnuz` both report `'<V1'`, so same-shape arrays with
identical raw bytes decode to different numbers). Both collisions suppressed the
period axis and changed the argmax. So: every concrete grid — uniform built-ins
included — is keyed on its resolved `to_jax()` array, fingerprinted as the exact
`np.dtype` **object**, shape, and raw bytes. Validation uses the same representation.

**Runtime mode is a property of the mode, not of the exact class.** `_grid_identity`
tests `isinstance(grid, IrregSpacedGrid) and grid.pass_points_at_runtime` **first**,
mirroring `V._get_coordinate_finder`'s dispatch — otherwise the two disagree about a
runtime-points *subclass*, which the interpolator supports but identity construction
would send to `to_jax()`, where it must raise. Concrete grids (including subclasses)
then fall through to the node fingerprint, so an overridden `to_jax()` stays
geometry-sensitive.

### Grid bounds are interpolation *support*, not feasibility limits

`V_{t+1}` is interpolated on period `t+1`'s grid, and pylcm's interpolation
**extrapolates linearly** beyond the grid rather than rejecting out-of-support points.
So the age-dependent bounds define the *interpolation support*, not a hard feasible set:
a period-`t` action whose next state lands below a tighter `t+1` floor (or above the
ceiling) is evaluated by extrapolation. **The model must keep every feasible transition
inside the next period's grid** — either the bounds coincide with the true feasibility
limits, or a constraint keeps next states in range. The borrowing-floor use satisfies
this by construction (`a_{t+1} ≥ a̲(t)` and period `t+1`'s floor is `a̲(t)`). If bounds are
*not* the feasibility limits, add a constraint or widen the grid, otherwise extrapolation
against `-inf` edge cells can produce `NaN`. (Audit F3.)

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

- grid class, `batch_size`, points mode, or the resolved node **shape/dtype** differing
  across the regime's active ages -> `RegimeInitializationError`.
- a concrete grid whose declared `n_points` disagrees with its own `to_jax()` length, or
  whose `to_jax()` is not 1-D -> `RegimeInitializationError`.
- a runtime-points grid that declares no `n_points` (its axis shape would be unknown at
  build time) -> `RegimeInitializationError`.
- an `AgeSpecializedGrid` on a regime active at no age -> `RegimeInitializationError`.
- age-varying **process** state or **discrete** state -> rejected (v1).
- `to_dataframe(additional_targets=...)` on an age-varying-grid-dependent target ->
  rejected, mirroring the `AgeSpecializedFunction` rule (published functions use one
  representative age).

## Test coverage

`tests/solution/test_age_specialized_grid_solve.py`:

- **solve**: a 2-state life-cycle model with an age-shrinking floor solves with
  **finite V at every age** where a fixed grid produces `-inf`; V is monotone in
  wealth.
- **simulate**: the continuous state stays within `[a̲(age), A_MAX]` and consumption
  is positive.
- **regression**: an age-**invariant** builder reproduces the plain fixed-grid solve
  bit-for-bit — the identity collapses to one program, so the constant-grid fast path
  (`_build_period_state_axes` → `None`) is preserved.
- **identity**: piecewise geometry, custom grids exposing `start`/`stop` while
  deriving nodes elsewhere, concrete subclasses overriding `to_jax`, runtime-points
  subclasses, signed-zero endpoints, and extended dtypes sharing a `dtype.str` are
  each kept distinct.
- **rejections**: non-shape-invariant grids, node-count and dtype changes, a declared
  `n_points` disagreeing with its own array, a points-mode switch, an
  `AgeSpecializedGrid` on a never-active regime, and a builder undefined outside its
  active ages.

## Implementation status

**Complete.** The wiring below is implemented and covered by the tests above:

1. `lcm/regime.py` accepts `AgeSpecializedGrid` in `states`.
2. `model.py` calls `validate_age_specialized_grids(states, ages)` at construction.
3. `processing.process_regimes` builds representative-resolved regimes for the
   invariant machinery, plus `period_to_regime_to_v_interp` for the continuation.
4. `_build_Q_and_F_per_period` folds the target regimes' grid identity at period+1
   into the group signature and passes the period-(t+1) `VInterpolationInfo`.
5. `backward_induction` resolves the per-period state axis and passes it as runtime
   values.
6. `simulation` uses the period's grid for the argmax axis and index→value lookup.

**Key simplification confirmed:** the current-period state axis reaches the Q_and_F
kernel as *runtime* values (shape-invariance ⇒ same trace), so only the *continuation*
(target) grid is a trace-time constant. Hence the state axis is resolved per period in
the solve loop with no recompile, while period-(t+1)'s grid is baked into period-t's
Q_and_F.

**Dedup is by resolved nodes, not by `signature`.** See the API and contract sections
above: `AgeSpecializedGrid.signature` is *not* the grid dedup key and not a correctness
precondition. (The `grid_signature`/`state_grids_signature` helpers that once
fingerprinted grids that way were unused and have been deleted.)
