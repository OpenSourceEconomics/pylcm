---
title: Internal Architecture
---

# Internal Architecture

This page describes how pylcm's source tree is organised and *why* it is laid out that
way. The audience is contributors and advanced users who want to find code, add a
feature, or understand which module they should be editing. End-users only ever write
`from lcm import Model, Regime, ...` and never need anything here.

## The `lcm` / `_lcm` split

pylcm's source is two packages, and the split is a hard binary:

- **`src/lcm/`** — the public surface. Everything a user constructs or consumes lives
  here, physically: the user-facing classes, the `@categorical` decorator, the `as_leaf`
  helper, the public type aliases, and the exception classes. `lcm/__init__.py`
  re-exports the public symbols so users write `from lcm import Model`.
- **`src/_lcm/`** — the private implementation. The build pipeline, the canonical engine
  dataclasses, the JAX-traced solve / simulate machinery, validators, I/O plumbing, and
  the engine-side type aliases. The leading underscore on the *package* carries the
  entire "private" signal — modules inside `_lcm/` are plainly named.

There is no gradient and no per-module underscore convention: a module is either in
`lcm/` (public) or in `_lcm/` (private). Internal code reaches the user-facing classes
through `from lcm.regime import Regime as UserRegime` etc.; the public `lcm/__init__.py`
imports `_lcm` first so the jaxtyping patch and the beartype claw are installed before
anything else loads.

```
lcm/
├── __init__.py       ← re-export façade for the public symbols
├── ages.py           ← AgeGrid
├── grids.py          ← LinSpacedGrid, LogSpacedGrid, IrregSpacedGrid, DiscreteGrid,
│                       PiecewiseLinSpacedGrid, PiecewiseLogSpacedGrid,
│                       PiecewiseGridSegment, and the @categorical decorator
├── model.py          ← Model
├── params.py         ← as_leaf + the MappingLeaf / SequenceLeaf re-exports
├── persistence.py    ← SolveSnapshot, SimulateSnapshot, load_snapshot,
│                       save_solution, load_solution
├── processes.py      ← the seven *Process classes
├── regime.py         ← Regime (Phased and MarkovTransition re-exported)
├── result.py         ← SimulationResult
├── transition.py     ← transition helpers
├── typing.py         ← user-facing type aliases
└── exceptions.py     ← every project-specific exception class
```

```
_lcm/
├── __init__.py            ← applies the jaxtyping patch + registers the beartype claw
├── ages.py                ← AgeGrid validators and step parsing
├── beartype_conf.py       ← the beartype configurations
├── config.py              ← build-time configuration constants
├── dtypes.py              ← canonical-dtype resolution
├── engine.py              ← canonical / engine-side dataclasses
├── jaxtyping_patch.py     ← bootstrap patch run before any jaxtyping type
├── model_processing.py    ← Model.__init__ build pipeline
├── pandas_utils.py        ← pd.Series ↔ JAX array bridge
├── state_action_space.py  ← state / action space validators
├── transition_checks.py   ← pre-solve regime + state transition prob checks
├── typing.py              ← engine-side type aliases and protocols
├── user_regime_validation.py ← validators for the user-facing Regime
├── variables.py           ← factories that build `Variables` from `Regime`
├── version.py             ← generated version string (hatch-vcs)
├── grids/                 ← grid infrastructure
├── processes/             ← stochastic-process infrastructure
├── persistence/           ← snapshot I/O and writer internals
├── regime_building/       ← per-regime canonicalisation
├── solution/              ← backward induction (solve) + validate_V
├── simulation/            ← forward sampling (simulate) + result helpers
├── params/                ← params templating and processing
└── utils/                 ← small, dependency-free helpers
```

Names cross the boundary in exactly one direction (user → canonical form) and only twice
— once when `Model(regimes={...})` is called (which triggers
`model_processing.build_regimes_and_template` →
`regime_building.processing.process_regimes`), and once for `flat_params` at every
`solve` / `simulate` call.

## The public surface — `lcm/`

`lcm/` is the canonical home for every class users construct or consume. Keeping all of
them in one package makes the public surface easy to find, review, and keep stable —
anything in `_lcm/` is fair game for refactoring. **`lcm/` modules are shallow by
design**: each file holds class definitions and the small number of public top-level
functions that round out the surface. Validators, I/O plumbing, DataFrame assembly, and
similar implementation detail live in `_lcm/` and are imported back in.

The mapping of public names to files:

| File             | What lives there                                                                                                                                                                                          |
| ---------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `model.py`       | `Model`                                                                                                                                                                                                   |
| `regime.py`      | `Regime` and the private default Bellman aggregator `_default_H`. Validators live in `_lcm/user_regime_validation.py`; the phase normalizer in `_lcm/regime_building/phases.py`.                          |
| `ages.py`        | `AgeGrid`. Step parser and validators live in `_lcm/ages.py`.                                                                                                                                             |
| `grids.py`       | `LinSpacedGrid`, `LogSpacedGrid`, `IrregSpacedGrid`, `DiscreteGrid`, `PiecewiseLinSpacedGrid`, `PiecewiseLogSpacedGrid`, `PiecewiseGridSegment`, and the `@categorical` decorator                         |
| `processes.py`   | The seven `*Process` classes — `UniformIIDProcess`, `NormalIIDProcess`, `LogNormalIIDProcess`, `NormalMixtureIIDProcess`, `TauchenAR1Process`, `RouwenhorstAR1Process`, `TauchenNormalMixtureAR1Process`. |
| `persistence.py` | `SolveSnapshot`, `SimulateSnapshot`, `load_snapshot`, `save_solution`, `load_solution`. Snapshot writers and atomic-dump live in `_lcm/persistence/`.                                                     |
| `result.py`      | `SimulationResult`. DataFrame assembly, metadata, and additional-targets computation live in `_lcm/simulation/result_*.py` and `_lcm/simulation/additional_targets.py`.                                   |
| `params.py`      | `as_leaf` plus the `MappingLeaf` / `SequenceLeaf` re-exports. The leaf-class definitions and the engine params machinery live in `_lcm/params/`.                                                          |
| `typing.py`      | The model-authoring aliases (`FloatND`, `ScalarInt`, `Period`, `Age`, ...) and the `User*` boundary aliases.                                                                                              |
| `exceptions.py`  | Every project-specific exception class.                                                                                                                                                                   |

### Why a package boundary, not just naming?

A naming convention (a `_private_` prefix on every internal module) tells *readers* what
is internal. A package boundary makes it visible to *tools*: code-search, auto-import,
public-API audits, and the linter can all key off `_lcm/`. The boundary is enforced by
the absence of imports — `lcm/` modules import from `_lcm/` only at well-defined wiring
points, and `_lcm/` reaches the user-facing classes through aliased imports
(`from lcm.regime import Regime as UserRegime`).

## Grid and process infrastructure: `_lcm/grids/` and `_lcm/processes/`

```
_lcm/grids/
├── base.py            ← Grid, ContinuousGrid, UniformContinuousGrid (ABCs)
├── continuous.py      ← LinSpacedGrid, LogSpacedGrid, IrregSpacedGrid
├── discrete.py        ← DiscreteGrid
├── piecewise.py       ← PiecewiseLinSpacedGrid, PiecewiseLogSpacedGrid,
│                         PiecewiseGridSegment
├── categorical.py     ← @categorical decorator + validators
└── coordinates.py     ← coordinate lookup helpers used by interpolation

_lcm/processes/
├── base.py            ← _ContinuousStochasticProcess + Gauss-Hermite / mixture helpers
├── iid.py             ← UniformIIDProcess, NormalIIDProcess, LogNormalIIDProcess,
│                        NormalMixtureIIDProcess
└── ar1.py             ← TauchenAR1Process, RouwenhorstAR1Process,
                         TauchenNormalMixtureAR1Process
```

The leaf classes are surfaced through `lcm/grids.py` and `lcm/processes.py`; the ABCs
(`Grid`, `_ContinuousStochasticProcess`, etc.) are used by internal code but are not
part of the documented public API.

Two design points worth knowing:

- **Process classes bundle both a discretization grid AND a transition mechanism**,
  unlike ordinary grids which are pure outcome-space. Users place
  `UniformIIDProcess(...)` in `Regime(states=...)` directly — the transition is invoked
  automatically. Putting a process class in `state_transitions` is a bug.
- **All vocabulary in the engine speaks of `process`, not `shock`.** Use `is_process`
  (on `VariableInfo`), `process_names` (on `Variables`), and `ProcessName` (typing
  alias). `shock` is reserved for the colloquial meaning and never appears as an
  identifier.

## Private siblings of the public modules

Several `lcm/` modules have a private counterpart in `_lcm/` that holds their
implementation detail. The pattern is the same throughout: `lcm/` keeps the class
definitions and the public top-level functions; the `_lcm/` counterpart holds
validators, helpers, and I/O plumbing that internal code is free to refactor.

```
_lcm/ages.py            ← STEP_UNITS, _parse_step,
                          _validate_age_grid / _validate_range / _validate_values
_lcm/user_regime_validation.py  ← the validators called from Regime.__post_init__

_lcm/simulation/initial_conditions.py
                        ← MISSING_CAT_CODE, PSEUDO_STATE_NAMES, and the
                          build / validate helpers for initial conditions

_lcm/persistence/
├── io.py               ← _atomic_dump, _save_pkl, _save_h5, _load_h5,
│                          _get_platform, _next_counter, _enforce_retention,
│                          _write_metadata, _write_environment_files
└── snapshots.py        ← _save_solve_snapshot, _save_simulate_snapshot,
                          _strip_V_arr_from_result, _bind_forward_refs

_lcm/simulation/result_metadata.py
                        ← ResultMetadata + _compute_metadata, _get_output_dtypes
_lcm/simulation/result_dataframe.py
                        ← _create_flat_dataframe and the per-regime / per-period
                          assembly helpers, plus categorical conversion
_lcm/simulation/additional_targets.py
                        ← _resolve_targets, _compute_targets, and DAG helpers
                          for to_dataframe(additional_targets=...)
```

Why split these out? Two reasons:

- **The public surface is easier to audit.** `regime.py`, `persistence.py`, and
  `result.py` each contain only the dozen-or-so symbols users actually touch. A reader
  looking for "what is the public contract of a Regime?" sees that contract directly,
  without scrolling past validator bodies.
- **Internal helpers can move freely.** Anything in `_lcm/` is internal — its location,
  signature, and existence can change without bumping the user surface.

A note on shadowing: the canonical `Regime` lives in `_lcm/engine.py`. The validators in
`_lcm/user_regime_validation.py` operate on the user-facing `lcm.regime.Regime` and
reach it through TYPE_CHECKING-guarded imports to break the circular dependency at
import time; beartype resolves the forward references at first call.

## Engine-side: `_lcm/engine.py`

`engine.py` holds the **canonical** post-processing dataclasses — the form the DP
machinery operates on:

- `Regime` — the canonical regime (distinct from the user-facing `lcm.regime.Regime`; in
  source files that import both we alias the user-facing one as `UserRegime`).
- `StateActionSpace` — pre-built state and action grids for a regime, with a
  `state_action_space(params)` method that fills in runtime-supplied grid points.
- `SolveFunctions` / `SimulateFunctions` — the compiled function bundles consumed by
  `solve` and `simulate`.
- `Variables` / `VariableInfo` — name + kind + topology metadata for every state and
  action in a regime.
- `PeriodRegimeSimulationData` — raw simulation output for one (regime, period) pair,
  before `SimulationResult` materialises a DataFrame.

The file name `engine.py` reflects what's inside: the engine's view of a model.

## Build pipeline: `model_processing.py` and `regime_building/`

```
_lcm/model_processing.py  ← top-level pipeline:
                            user regimes + params → canonical Model

_lcm/regime_building/
├── processing.py         ← per-regime canonicalisation:
│                            UserRegime → engine.Regime
├── transitions.py        ← collect_state_transitions: walk user-supplied
│                            state_transitions into per-target callables
├── stochastic_state_transitions.py
│                         ← process-time AST + n_outcomes derivation for
│                            stochastic state transitions (raises
│                            InvalidStateTransitionProbabilitiesError on
│                            subscript-order mismatches)
├── Q_and_F.py            ← build (Q, F) closure for solve / simulate
├── argmax.py             ← argmax helpers over action grids
├── max_Q_over_a.py       ← argmax / max over action grids
├── V.py                  ← value-function interpolation info
├── h_dag.py              ← user-DAG resolution for H (Bellman aggregator)
├── next_state.py         ← compose per-state transitions into a single
│                            next_state function for simulation
├── ndimage.py            ← map-coordinates wrapper for continuous interp
└── diagnostics.py        ← cold-path machinery invoked by validate_V to
                            pinpoint *which* intermediate produced a NaN
```

The two-step name (`model_processing` at the model level, `regime_building` per regime)
reflects what each layer actually does — the top level merges regimes and resolves fixed
params; each regime is then canonicalised independently.

The numerical checks fired at solve / simulate time live outside `regime_building/`:

- `regime_building/stochastic_state_transitions.py` runs at `Model(...)` construction
  time and can fail the build before any params are involved. It catches malformed user
  functions (e.g., `probs_array[health, age]` where the signature is `(age, health)`)
  via AST analysis. Always on, never gated.
- `_lcm/transition_checks.py` runs from `Model.solve()` / `Model.simulate()` before
  backward induction starts. It evaluates the regime and state transition functions on
  the regime's grid Cartesian product and verifies output shape, [0, 1] range, and
  sum-to-1. State checks are gated by `log_level != "off"` because the Cartesian product
  can blow up on models with many continuous-grid-dependent stochastic states.
- `_lcm/solution/validate_V.py` runs *during* backward induction (after each period in
  `solve_brute.py`, and once on the V handed to `simulate.py`). On NaN it invokes the
  diagnostic-intermediates closure built in `regime_building/diagnostics.py` to pinpoint
  which intermediate (`U`, `F`, `E[V]`, `Q`) produced the NaN.

## Solve and simulate

```
_lcm/solution/
├── solve_brute.py      ← backward induction loop:
│                          V[T], V[T-1], ..., V[0] via max_Q_over_a
└── validate_V.py       ← per-period NaN / Inf validation

_lcm/simulation/
├── simulate.py         ← forward sampling loop with state-action draws
├── compile.py          ← compiled-function assembly for the simulate phase
├── random.py           ← PRNG-key handling for the sampling draws
├── transitions.py      ← per-state transition composition for simulation
└── initial_conditions.py
                        ← canonicalize / validate the user's
                          initial_conditions kwarg
```

These are the JAX-traced hot paths. The DP and sampling logic is the *only* thing here;
everything that constructs the inputs (parameters, grids, transitions, compiled
callables) lives in `regime_building/` and is read out of the canonical `Regime`
instances.

## Params: boundary form vs. canonical form

```
_lcm/params/
├── processing.py       ← cast_params_to_canonical_dtypes:
│                          User-supplied dicts (with int/float/np.array
│                          leaves) → flat MappingProxyType keyed by
│                          qualified names with JAX-array leaves.
├── regime_template.py  ← per-regime template construction:
│                          inspect Regime functions to derive what
│                          parameters they need.
├── mapping_leaf.py     ← UserMappingLeaf / MappingLeaf — wrapper that
│                          carries an immutable dict through a JAX pytree
│                          without becoming a Mapping itself.
└── sequence_leaf.py    ← UserSequenceLeaf / SequenceLeaf — same for
                           sequences.
```

The public `lcm/params.py` module exposes `as_leaf` and re-exports the four leaf
classes; their definitions and the engine params machinery live in `_lcm/params/`.

Two leaf types exist because params dicts can contain heterogeneous leaves (scalars,
arrays, named tuples, etc.). Wrapping them in `MappingLeaf` / `SequenceLeaf` lets the
pytree machinery treat them as opaque leaves rather than walking into them — important
when a "leaf" is itself a dict mapping named arguments to JAX arrays.

The `User*` types accept the wide boundary form (`int`, `float`, `np.ndarray`,
`pd.Series`, etc.). After `cast_params_to_canonical_dtypes` runs, only canonical
JAX-array leaves and canonical-narrow `MappingLeaf` / `SequenceLeaf` instances survive.
The downstream `solve` / `simulate` code only ever sees the canonical form.

## Pandas bridge: `_lcm/pandas_utils.py`

A single module for converting between user-friendly `pd.Series` / `pd.DataFrame`
representations and the JAX arrays the engine expects. `array_from_series` is the
workhorse: it inspects the function source via AST helpers (in
`utils/ast_inspection.py`) to determine the expected multi-index order, then
materialises a properly-shaped JAX array.

This file gets used both at params processing (for any `pd.Series` leaves in user
params) and at simulation output (for building the result DataFrame).

## Utilities: `_lcm/utils/`

Small, dependency-light helpers grouped by topic:

- `ast_inspection.py` — Parse a function body to find `probs_array[a, b]` subscript
  patterns. Used by the static AST check and by `pandas_utils`.
- `containers.py` — `ensure_containers_are_immutable`, `first_non_none`,
  `invert_regime_ids`.
- `dispatchers.py` — `productmap`, `vmap_1d`, `simulation_spacemap`. See
  [Dispatchers](dispatchers.ipynb).
- `error_messages.py` — `format_messages`, which collapses a list of validation errors
  into a single string.
- `functools.py` — `all_as_kwargs`, `get_union_of_args`.
- `logging.py` — `get_logger`, `format_duration`, log-formatting helpers.
- `namespace.py` — `flatten_regime_namespace` / `unflatten_regime_namespace` for the
  qualified-name pytree keys.

## Type aliases: `lcm/typing.py` vs `_lcm/typing.py`

```
lcm/typing.py    ← user-facing aliases: jaxtyping array shapes (FloatND,
                   ScalarInt, ...), Period, Age, and the User* boundary
                   aliases (UserParams, UserInitialConditions, ...)
_lcm/typing.py   ← engine-side aliases and protocols: string labels
                   (RegimeName, StateName, ...), compound mapping
                   aliases, canonical post-processing forms (Params,
                   InitialConditions, ...), and the structural Protocol
                   classes (EconFunction, TransitionFunction, ...)
```

The split mirrors the public / private package boundary. `lcm/typing.py` holds the
aliases a user needs to annotate model functions and the `User*` aliases that accept
wide boundary types; it imports nothing from `_lcm`. `_lcm/typing.py` holds the
engine-internal aliases — including the post-canonicalization forms (`Params`,
`InitialConditions`) — and builds on the public aliases it imports from `lcm.typing`.

## Exceptions: `lcm/exceptions.py`

Every project-specific exception class lives here, all inheriting from `PyLCMError`.
They split into two categories:

- **Initialization errors** — raised at `Model(...)` / `Regime(...)` /
  `LinSpacedGrid(...)` construction time. These map beartype violations on user-facing
  constructors to a project-typed error (so users see e.g. `ModelInitializationError`,
  not a `BeartypeCallHintViolation`).
- **Runtime errors** — `InvalidValueFunctionError`,
  `InvalidRegimeTransitionProbabilitiesError`,
  `InvalidStateTransitionProbabilitiesError`, `InvalidParamsError`,
  `InvalidInitialConditionsError`. These fire from `_lcm/transition_checks.py` and
  `_lcm/solution/validate_V.py` during solve / simulate.

The exception classes are public — both `from lcm.exceptions import InvalidParamsError`
and `except lcm.InvalidParamsError` work. `format_messages`, the helper that assembles a
list of validation errors into one string, is internal validation plumbing and lives in
`_lcm/utils/error_messages.py`.

## Bootstrap modules

A few `_lcm/` modules exist for ordering reasons rather than for any conceptual
grouping:

- `jaxtyping_patch.py` — Bootstrap patch that has to run before any
  `jaxtyping`-annotated type is created. `_lcm/__init__.py` applies it as its first
  statement.
- `beartype_conf.py` — Holds the beartype configurations used in the package (the
  internal-claw conf + the user-facing constructor-decorator confs).
- `config.py` — Build-time configuration constants (paths to test data, etc.).
- `dtypes.py` — Canonical-dtype resolution (`canonical_float_dtype()`), which depends on
  the JAX x64 setting.

## Reading order for new contributors

If you're reading the codebase for the first time, the path of least confusion is:

1. **`lcm/regime.py`** to see what users supply.
1. **`lcm/model.py`** to see what `Model.__init__` triggers.
1. **`_lcm/model_processing.py`** for the top-level pipeline.
1. **`_lcm/regime_building/processing.py`** for per-regime canonicalisation — the
   longest single file and the heart of the build.
1. **`_lcm/engine.py`** for the canonical dataclasses the DP machinery consumes.
1. **`_lcm/solution/solve_brute.py`** and **`_lcm/simulation/simulate.py`** for the
   actual DP and sampling.

By the time you reach (6), the canonical form should feel familiar and the JAX-traced
code becomes easy to read.
