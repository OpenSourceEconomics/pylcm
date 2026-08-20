---
title: Adding a Solver
---

# Adding a Solver

A regime's `solver` field selects the algorithm that computes its value function during
backward induction. pylcm ships four solvers (`GridSearch`, `DCEGM`, `NEGM`, `EGM`; see
`lcm.solvers`), and the engine is designed so that new ones can be added without
touching the backward-induction loop. This page specifies the contract a solver
implements, which of the three base classes it subclasses and why the regime decides
that, the lifecycle the engine drives it through, and the invariants that keep the
generic layer solver-agnostic.

The single normative source is `src/_lcm/solution/contract.py`. Everything the
backward-induction loop knows about a solver flows through the types defined there; this
page explains how they fit together.

## The design rule

**The backward-induction layer understands generic solver outputs; each solver
implementation understands how those outputs are produced.**

Concretely:

- `KernelResult` is the *only* boundary between solver-specific execution and the
  generic loop. A kernel returns one; the loop accumulates its fields. No kernel mutates
  loop state.
- The generic layer never asks "which solver is this?". It asks capability questions:
  `Solver.requires_continuation` at build time,
  `Regime.solution.solves_from_continuation` engine-side. Adding a solver must not add
  an `isinstance` check to `backward_induction.py` or `contract.py`.
- The generic continuation channel has one definition in `_lcm.continuation`. Backward
  induction threads `ContinuationPayload` opaquely; current endogenous-grid solvers
  additionally publish an `EGMContinuationSpec` bundling the concrete carry template
  with the static layout a reading parent needs. A future representation adds its own
  operations instead of redefining the alias in multiple engine modules.
- `SimulationPolicy` is likewise threaded as an optional artifact. Collection and host
  transfer are demand-driven: they occur only for an explicit inspection request or an
  actual simulation-policy consumer.

## The contract objects

**`Solver` (abstract base class).** The user-facing configuration object, attached as
`Regime(solver=...)`. A frozen dataclass carrying the solver's settings (grids, batch
sizes, thresholds). Three methods and two properties matter:

- `validate_model(context) -> None`: validate finalized user declarations, before the
  numerical build. Default: no-op.
- `validate_build(context) -> None`: validate processed period/layout information that
  does not exist at the earlier stage. Default: no-op.
- `build_period_kernels(context) -> SolutionKernels` (abstract): build the regime's
  per-period solve adapters after both validation stages pass.
- `requires_continuation -> bool`: whether this solver reads continuation payloads from
  reachable targets. Default: `False`.
- `egm_continuation_layout -> EGMContinuationLayout`: interpretation of a current EGM
  carry. Override only when the solver's row/candidate layout differs.

**`SolverBuildContext`.** Everything a solver may read at build time, bundled so the
method signature stays stable as solvers with different needs are added: the regime's
state-action space, grids, processed functions, constraints, transitions, the per-period
Q-and-F closures, interpolation info for every regime's value function, flat parameter
names (own and per-regime), JIT and taste-shock flags, the optional certainty
equivalent, and the distributed co-map axes. Each solver reads only the fields it uses.

**`SolutionKernels`.** What `build_period_kernels` hands back: an immutable mapping of
period to `PeriodKernel`, plus an optional `continuation_spec`. The current
`EGMContinuationSpec` contains both an all-finite payload template and its immutable
layout metadata. The template initializes the rolling continuation mapping and serves as
the lowering argument when a *parent's* kernel is AOT-compiled, so it must match every
real payload structurally; bundling the layout prevents producer and reader metadata
from drifting apart.

**`PeriodKernel` (protocol).** The loop's uniform call target — one non-jitted adapter
per regime-period. Plain closures or small frozen dataclasses satisfy it structurally.
It separates three concerns:

- `cores() -> Mapping[str, Callable]`: the shared jitted core(s), keyed by a stable
  per-kernel name. Most kernels have exactly one (`{"main": ...}`); NEGM has two
  (`{"keeper": ..., "adjuster": ...}`). AOT compilation lowers and deduplicates each
  core by identity, so periods that share a core share one compiled program.
- `build_lower_args(core_key=..., ...) -> Mapping[str, object]`: the named core's
  lowering kwargs for one period, built from the state-action space, the rolled
  `next_regime_to_V_arr` / `next_regime_to_continuation` mappings, flat params, period,
  and ages.
- `__call__(compiled_cores=..., ...) -> KernelResult`: invoke the compiled core(s) with
  the solver's own argument layout and assemble the result *outside* JIT.

`with_fixed_params(fixed_flat_params=...)` returns a copy with the regime's fixed params
bound into the core(s); the adapter owns its solver's binding rule so the engine never
switches on solver type to bind params.

**`KernelResult`.** One regime-period output, assembled outside JIT:

- `V_arr` (required): the value-function array on the regime's state grid.
- `continuation` (optional): the cross-period payload a continuation-based parent
  interpolates. `None` for regimes that publish none.
- `simulation_policy` (optional): the published off-grid policy artifact forward
  simulation can interpolate. `None` for regimes that publish none.

**`BackwardInductionResult`.** The loop's return value: `value_functions` (period →
regime → V array) and `simulation_policies` (period → regime → published policy, sparse
over regimes). Internal — `Model.solve` unpacks it into the public return shape.

## Choosing the base class

`Solver` is not the only base. The solver family has three members, all exported from
`lcm.solvers`, and **which one a new solver subclasses is forced by the regime it is
meant to attach to** — it is not a style choice.

| Regime class                     | Accepts                           | Base class to subclass |
| -------------------------------- | --------------------------------- | ---------------------- |
| `Regime`                         | any non-margin `Solver`           | `Solver`               |
| `ConsumptionSavingsRegime`       | `OneMarginSolver` or `GridSearch` | `OneMarginSolver`      |
| `NestedConsumptionSavingsRegime` | `TwoMarginSolver` or `GridSearch` | `TwoMarginSolver`      |

All three regime classes enforce the pairing at construction, not at solve time, and
they guard in both directions. Each specialised regime checks in `__post_init__` that
its solver is the base it wants, and a plain `Regime` checks that its solver is *not* a
margin-family one: hand it an `EGM`, `DCEGM` or `NEGM` and it raises
`RegimeInitializationError` telling you to use the specialised regime instead, because
the four role names such a solver needs have nowhere to be declared. So a solver
subclassing `Solver` directly is complete and correct for a plain `Regime` and a
`ConsumptionSavingsRegime` will refuse it, and the reverse is refused too. Either error
arrives when the regime is built, long before any kernel runs.

### The `Solver`-direct path

Subclass `Solver` when the algorithm reads its state and action grids straight off the
state-action space and needs no DAG node singled out. Grid search is exactly this case:
it maximises over the action grid at every state, so no node in the regime's DAG plays a
distinguished role. Implement `build_period_kernels`, override the validation hooks and
`requires_continuation` if the algorithm needs them, and nothing else. `GridSearch` is
the minimal reference.

### The margin families

An endogenous-grid method cannot work from the grids alone. It has to know *which* DAG
node is the liquid state, which is the action it inverts for, which node carries
resources, and which is the post-decision state — the Euler equation is written in those
four specific quantities. Those names belong to the model, so the **regime** declares
them and the **solver** carries numerical configuration only: grids, batch sizes,
envelope choice, tolerances.

`OneMarginSolver` is the marker for solvers consuming one such margin, and
`TwoMarginSolver` for solvers consuming a liquid margin plus an outer continuous
(durable or illiquid) margin. Each adds exactly one abstract operation to `Solver`: a
binding method that takes the regime's resolved names and returns an immutable copy of
the solver carrying them.

```python
class OneMarginSolver(Solver):
    def _with_liquid_margin(self, margin: _BoundLiquidMargin) -> OneMarginSolver: ...


class TwoMarginSolver(Solver):
    def _with_margins(
        self,
        *,
        liquid: _BoundLiquidMargin,
        outer: _BoundOuterContinuousMargin,
    ) -> TwoMarginSolver: ...
```

The binding method is private, and so is the margin type it takes — which is the outward
sign of a deliberate restriction: **the margin families are closed. Subclass one of the
shipped solvers, not a marker.** A solver deriving straight from `OneMarginSolver` or
`TwoMarginSolver` is refused when the model is built, by
`fail_if_solver_is_not_shipped`. A subclass of `EGM`, `DCEGM` or `NEGM` is accepted and
needs nothing special — it inherits the concrete type the engine tests for.

The reason is that three engine sites dispatch on the concrete shipped classes rather
than on the markers: the simulate-phase budget-constraint synthesis in
`regime_building/processing.py`, the terminal branch of `egm/budget.py`, and `_as_dcegm`
in `egm/regime_introspection.py`. A solver that satisfies the marker contract but is
none of those types passes every check, binds its margins, and is then solved with the
budget constraint and the carry read skipped — a wrong published policy with nothing
raised. Refusing it at build time is what turns that into an error message.

The restriction is scoped to what the engine can currently dispatch, and lifts when the
two-asset endogenous-grid solvers need genuinely custom implementations. Until then, the
`Solver`-direct path above is open and unrestricted: a solver that singles out no DAG
node subclasses `Solver` and works.

### Implementing the binding method

A solver added inside the package implements this; the closure above is what puts it out
of reach from outside. It is also why a subclass of a shipped solver reaches the engine
as itself — the implementation it inherits is the one below.

Do not construct the bound copy by hand. Route it through `bind_roles`, which returns an
object that is still the type the user constructed: a subclass keeps the fields it added
and the methods it overrides, so a custom solver reaches the engine as itself rather
than as the stock class it derived from.

```python
def _with_liquid_margin(self, margin: _BoundLiquidMargin) -> _BoundMySolver:
    """Bind regime-owned DAG names without exposing them on the public config."""
    return cast(
        "_BoundMySolver",
        bind_roles(
            solver=self,
            role_type=_BoundMySolver,
            continuous_state=margin.state,
            continuous_action=margin.action,
            resources=margin.resources,
            post_decision_function=margin.post_decision_state,
        ),
    )
```

`role_type` is a private frozen dataclass subclassing the public solver and declaring
one field per resolved name. The public class stays free of them, so a user never sees —
or can set — a role the regime owns. Inside the kernels, recover the names by casting
`self` to the bound type; the engine only ever hands back an instance that carries them.

### Worked examples in the tree

- **`Solver` directly** — `GridSearch`.
- **`OneMarginSolver`** — `EGM` and `DCEGM`. Both bind the same four liquid names, and
  `EGM._with_liquid_margin` is the shortest complete instance of the pattern above.
- **`TwoMarginSolver`** — `NEGM`, which binds a liquid margin for its inner solve and an
  outer margin for the durable grid search around it. A nest hands on its *bound* inner
  solver rather than the public one it was declared with; `bind_roles` supports that by
  letting a role entry replace a field the solver already carries.

## The lifecycle

1. **Model validation.** After model-level slots are merged and regimes finalized,
   central model processing calls `solver.validate_model(context)` uniformly. A known
   incompatibility raises `ModelInitializationError` during `Model(...)`.

1. **Build validation and construction.** `process_regimes` builds a
   `SolverBuildContext`, calls `solver.validate_build(context)`, then
   `solver.build_period_kernels(context)`. Engine-produced terminal/GridSearch EGM
   carries are added only to targets with an incoming retained edge from a solver whose
   `requires_continuation` is true; unreachable eligible regimes publish nothing.

1. **AOT compilation.** The loop collects every kernel's `cores()`, dedupes them by
   identity, and lowers each with the kwargs from `build_lower_args`. Continuation
   templates stand in for real payloads during lowering.

1. **The solve loop** (`src/_lcm/solution/backward_induction.py`). For each period, for
   each active regime, the loop calls the period's adapter and accumulates the result:

   - `V_arr` always enters `period_solution` (and the NaN/Inf diagnostics — automatic
     for every solver, no kernel involvement).
   - `continuation`, if present, enters `period_continuations`.
   - `simulation_policy`, if present, is retained and copied to host only when the solve
     request asks for the inspection artifact or fresh simulation has a qualifying
     policy-read consumer. Otherwise the output is discarded immediately.

   After the period, the loop rolls `next_regime_to_V_arr` and
   `next_regime_to_continuation` forward. Both mappings keep their full template key
   sets and update only the entries solved this period, so the pytree structure the
   compiled cores were lowered against never changes. Superseded payload buffers are
   deleted eagerly once rolled.

## Invariants a new solver must respect

- **Return, never mutate.** A kernel communicates exclusively through its
  `KernelResult`. The loop owns accumulation, rolling, host eviction, and diagnostics.
- **Assemble outside JIT.** The adapter is non-jitted; only the cores are compiled.
  Anything shape-dynamic belongs in the adapter, anything hot in a core.
- **Stable pytrees and layout.** Whatever a kernel publishes as `continuation` must
  match `continuation_spec.template` in structure, shapes, and dtypes every period, and
  its leading axes must obey `continuation_spec.layout`.
- **Capability checks, not identity checks.** If the engine needs to treat your solver
  differently somewhere, express the difference as a property on the contract (as
  `requires_continuation` does) — never as an `isinstance` in generic code. Note the
  existing engine-side predicate `solves_from_continuation` deliberately requires *both*
  a regime transition and a continuation template: a terminal regime that merely
  *produces* a closed-form continuation payload does not solve from one. The consumer
  that depends on this strictness is the `inverse_marginal_utility` exclusion in
  simulation targets — a bare template check would wrongly exclude a terminal
  carry-producer's targets. The diagnostics' U/F/E/Q breakdown skip is robust to either
  check (a terminal carry producer exposes an empty intermediates map, so the skip is a
  no-op there regardless).
- **Fail at the earliest informed stage.** `validate_model` rejects incompatibilities
  visible after finalization; `validate_build` handles processed layouts and period
  information. A solver that silently produces wrong numbers on an out-of-scope regime
  is a correctness bug, not a limitation.

## Where the code goes

- One module per solver under `src/_lcm/solution/` (`grid_search.py`, `dcegm.py`,
  `negm.py`, `egm.py` are the pattern). Shared lifecycle helpers live in
  `src/_lcm/solution/continuation_target.py`; heavy numerical machinery gets its own
  package (as EGM's does in `src/_lcm/egm/`).
- Re-export the class from the `lcm.solvers` façade and add it to that module's
  `__all__` and module docstring. Keep numerical imports function-local inside
  `build_period_kernels` so the façade stays import-light.
- The solver class itself is user-facing configuration: a frozen dataclass with inline
  field docstrings, `@beartype(conf=REGIME_CONF)` so invalid user input surfaces as a
  typed regime error.

## The minimal reference: `GridSearch`

`src/_lcm/solution/grid_search.py` is the whole contract in ~160 lines: a
configuration-free `Solver` whose `build_period_kernels` wraps each period's Q-and-F
closure in a jitted max-Q-over-a core (identity-deduped across periods), and a
`_GridSearchPeriodKernel` dataclass implementing the four protocol methods around a
single `"main"` core, returning `KernelResult(V_arr=...)` with no optional outputs. Read
it first; the EGM solvers are the same shape with more machinery inside the cores.

## Testing a new solver

Follow the repository's red-first discipline (see the testing section in `AGENTS.md`),
and cover at minimum:

- **A correctness oracle.** Solve a model with a known solution — analytic where
  possible, otherwise a brute-force `GridSearch` twin on a dense grid or an independent
  VFI implementation — and assert concrete values with explicit tolerances (the DS-2024
  housing tests under `tests/test_models/` are the pattern).
- **Scope rejection.** One test per model-stage and build-stage validation mode.
- **Both precisions.** The suite runs with `--precision 32` on GPU CI; precision-scale
  any tolerance (grep for `X64_ENABLED` in existing tests).
- **Cross-backend stability.** Comparisons inside a solver's numerics must not make
  keep/drop or tie decisions on quantities that are exact-arithmetic ties: backend
  reduction order flips the sign of such rounding noise, and CPU and GPU will silently
  produce structurally different solutions. Judge such decisions past a scale-aware
  noise floor (see `_savings_decrease_past_noise` in
  `src/_lcm/egm/upper_envelope/fues.py`).

## What deliberately does not exist

- No solver lifecycle hooks or callbacks in the loop, beyond the uniform kernel call.
- No artifact registry: a solver output that the engine should carry is a field on
  `KernelResult`, added deliberately, threaded opaquely.
- No public exposure of kernel internals: `Model.solve`'s return shape is independent of
  which solvers ran.
