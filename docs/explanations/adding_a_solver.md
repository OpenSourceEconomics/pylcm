---
title: Adding a Solver
---

# Adding a Solver

A regime's `solver` field selects the algorithm that computes its value function during
backward induction. pylcm ships five solvers (`GridSearch`, `DCEGM`, `NEGM`,
`OneAssetEGM`, `TwoDimEGM`; see `lcm.solvers`), and the engine is designed so that new
ones can be added without touching the backward-induction loop. This page specifies the
contract a solver implements, the lifecycle the engine drives it through, and the
invariants that keep the generic layer solver-agnostic.

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
- The only solver-specific types the generic layer names are the two aliases in
  `contract.py` — `ContinuationPayload` and `SimulationPolicy`. The engine threads both
  opaquely.

## The contract objects

**`Solver` (abstract base class).** The user-facing configuration object, attached as
`Regime(solver=...)`. A frozen dataclass carrying the solver's settings (grid names,
batch sizes, thresholds). Two methods and one property matter:

- `build_period_kernels(context) -> SolutionKernels` (abstract): build the regime's
  per-period solve adapters.
- `validate(context) -> None`: build-time scope check; raise a loud, typed error for any
  regime the solver cannot handle correctly. Default: no-op.
- `requires_continuation -> bool`: whether this solver reads a continuation payload from
  its target regimes. Default: `False`.

**`SolverBuildContext`.** Everything a solver may read at build time, bundled so the
method signature stays stable as solvers with different needs are added: the regime's
state-action space, grids, processed functions, constraints, transitions, the per-period
Q-and-F closures, interpolation info for every regime's value function, flat parameter
names (own and per-regime), JIT and taste-shock flags, the optional certainty
equivalent, and the distributed co-map axes. Each solver reads only the fields it uses.

**`SolutionKernels`.** What `build_period_kernels` hands back: an immutable mapping of
period to `PeriodKernel`, plus an optional `continuation_template` — an all-finite
payload with the regime's static shapes. The template initializes the rolling
continuation mapping and serves as the lowering argument when a *parent's* kernel is
AOT-compiled, so it must be shaped exactly like every real payload the kernels will
publish.

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

## The lifecycle

1. **Build.** `process_regimes` builds a `SolverBuildContext` per regime and calls
   `solver.validate(context)`, then `solver.build_period_kernels(context)`
   (`src/_lcm/regime_building/processing.py`). Terminal regimes produce their
   closed-form continuation payloads only when some regime's solver reports
   `requires_continuation` — the build reads the capability, not the type.

1. **AOT compilation.** The loop collects every kernel's `cores()`, dedupes them by
   identity, and lowers each with the kwargs from `build_lower_args`. Continuation
   templates stand in for real payloads during lowering.

1. **The solve loop** (`src/_lcm/solution/backward_induction.py`). For each period, for
   each active regime, the loop calls the period's adapter and accumulates the result:

   - `V_arr` always enters `period_solution` (and the NaN/Inf diagnostics — automatic
     for every solver, no kernel involvement).
   - `continuation`, if present, enters `period_continuations`.
   - `simulation_policy`, if present, enters `period_simulation_policies` and is evicted
     to host memory (policies are solve *outputs*; no backward step reads them, and
     leaving them on device would pin one continuation-sized buffer per period).

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
- **Stable pytrees.** Whatever a kernel publishes as `continuation` must match the
  regime's `continuation_template` in structure, shapes, and dtypes, every period.
- **Capability checks, not identity checks.** If the engine needs to treat your solver
  differently somewhere, express the difference as a property on the contract (as
  `requires_continuation` does) — never as an `isinstance` in generic code. Note the
  existing engine-side predicate `solves_from_continuation` deliberately requires *both*
  a regime transition and a continuation template: a terminal regime that merely
  *produces* a closed-form continuation payload does not solve from one, and the two
  consumers (the diagnostics' U/F/E/Q breakdown skip and the `inverse_marginal_utility`
  exclusion in simulation targets) rely on that distinction.
- **Fail loud at build time.** `validate` is the place to reject regimes outside the
  solver's scope (wrong number of continuous states, missing declared functions, a
  certainty equivalent a linear-expectation method cannot honor). A solver that silently
  produces wrong numbers on an out-of-scope regime is a correctness bug, not a
  limitation.

## Where the code goes

- One module per solver under `src/_lcm/solution/` (`grid_search.py`, `dcegm.py`,
  `negm.py`, `one_asset_egm.py`, `two_dim_egm.py` are the pattern). Shared lifecycle
  helpers live in `src/_lcm/solution/continuation_target.py`; heavy numerical machinery
  gets its own package (as EGM's does in `src/_lcm/egm/`).
- Re-export the class from the `lcm.solvers` façade and add it to `__all__` and the
  module docstring there. Keep numerical imports function-local inside
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
- **Scope rejection.** One test per `validate` failure mode.
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
