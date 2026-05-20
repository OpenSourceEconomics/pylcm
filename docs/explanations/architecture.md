---
title: Internal Architecture
---

# Internal Architecture

This page describes how `lcm`'s source tree is organised and *why* it is laid out that
way. The audience is contributors and advanced users who want to find code, add a
feature, or understand which module they should be editing. End-users only ever write
`from lcm import Model, Regime, ...` and never need anything here.

## At a glance

```
lcm/
├── __init__.py             ← thin re-export façade
├── api/                    ← every user-facing class and helper
├── _ages.py                ← AgeGrid validators and step parsing
├── _grids/                 ← private grid infrastructure
├── _persistence/           ← snapshot I/O and writer privates
├── _processes/             ← private stochastic-process infrastructure
├── _regime/                ← Regime validators, default H, transition-probs helpers
├── _transition_checks.py   ← pre-solve regime + state transition prob checks
├── engine.py               ← canonical / engine-side dataclasses
├── model_processing.py     ← Model.__init__ build pipeline
├── regime_building/        ← per-regime canonicalisation
├── solution/               ← backward induction (solve) + validate_V
├── simulation/             ← forward sampling (simulate) + result helpers
├── params/                 ← params templating and processing
├── pandas_utils.py         ← pd.Series ↔ JAX array bridge
├── utils/                  ← small, dependency-free helpers
├── typing.py               ← engine-side type aliases
├── variables.py            ← factories that build `Variables` from `Regime`
└── exceptions.py           ← every project-specific exception class
```

The directory tree maps onto a single organising principle: **the closer a module is to
user-supplied input, the closer it sits to `api/`. The closer it is to the JAX-traced DP
machinery, the deeper it sits in the engine side.** Names cross this boundary in exactly
one direction (user → canonical form) and only twice — once when `Model(regimes={...})`
is called (which triggers `model_processing.build_regimes_and_template` →
`regime_building.processing.process_regimes`), and once for `flat_params` at every
`solve` / `simulate` call.

## The `api/` boundary

`lcm/api/` is the canonical home for every class users construct or consume. Putting all
of them in one directory makes the public surface easy to find, review, and keep stable
— anything outside `api/` is fair game for refactoring. **`api/` modules are shallow by
design**: each file holds class definitions and the small number of public top-level
functions that round out the surface. Validators, I/O plumbing, DataFrame assembly, and
similar implementation detail live in private siblings (leading-underscore files or
packages) and are imported back in.

The mapping of public names to files:

| File                 | What lives there                                                                                                                                                                                         |
| -------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `api/model.py`       | `Model`                                                                                                                                                                                                  |
| `api/regime.py`      | `Regime`, `MarkovTransition`, `SolveSimulateFunctionPair`. Validators and the default Bellman aggregator live in `lcm/_regime/`.                                                                         |
| `api/ages.py`        | `AgeGrid`. Step parser, validators, and `PSEUDO_STATE_NAMES` live in `lcm/_ages.py`.                                                                                                                     |
| `api/grids.py`       | `LinSpacedGrid`, `LogSpacedGrid`, `IrregSpacedGrid`, `DiscreteGrid`, `PiecewiseLinSpacedGrid`, `PiecewiseLogSpacedGrid`, `Piece`                                                                         |
| `api/processes.py`   | The seven `*Process` classes — `UniformIIDProcess`, `NormalIIDProcess`, `LogNormalIIDProcess`, `NormalMixtureIIDProcess`, `TauchenAR1Process`, `RouwenhorstAR1Process`, `TauchenNormalMixtureAR1Process` |
| `api/categorical.py` | The `@categorical` class decorator                                                                                                                                                                       |
| `api/persistence.py` | `SolveSnapshot`, `SimulateSnapshot`, `load_snapshot`, `save_solution`, `load_solution`. Snapshot writers and atomic-dump live in `lcm/_persistence/`.                                                    |
| `api/result.py`      | `SimulationResult`. DataFrame assembly, metadata, and additional-targets computation live in `lcm/simulation/_result_*.py` and `lcm/simulation/_additional_targets.py`.                                  |
| `api/typing.py`      | `UserAge`, `UserParams`, `UserInitialConditions`, `UserFunction`, `UserFacingParamsTemplate`                                                                                                             |

Internal code reaches these classes through `lcm.api.*`. Users keep writing
`from lcm import Model` — `lcm/__init__.py` is a thin re-export of every symbol in
`api/` plus a few utilities.

### Why physical separation, not just naming?

A naming convention (e.g., a `_private_` prefix) tells *readers* what is internal. A
physical directory makes it visible to *tools*: code-search, auto-import, public-API
audits, and the linter can all key off `lcm/api/`. The boundary is enforced by the
absence of imports — the rest of `lcm/` never imports *from* `api/` outside of
well-defined wiring points, and `api/` does not import from internals beyond the ABCs
and engine dataclasses it deliberately exposes.

## Private packages: `_grids/` and `_processes/`

```
_grids/
├── base.py            ← Grid, ContinuousGrid, UniformContinuousGrid (ABCs)
├── continuous.py      ← LinSpacedGrid, LogSpacedGrid, IrregSpacedGrid
├── discrete.py        ← DiscreteGrid
├── piecewise.py       ← PiecewiseLinSpacedGrid, PiecewiseLogSpacedGrid, Piece
├── categorical.py     ← @categorical decorator + validators
└── coordinates.py     ← coordinate lookup helpers used by interpolation

_processes/
├── _base.py           ← _ProcessGrid + Gauss-Hermite / mixture helpers
├── iid.py             ← UniformIIDProcess, NormalIIDProcess, LogNormalIIDProcess,
│                       NormalMixtureIIDProcess
└── ar1.py             ← TauchenAR1Process, RouwenhorstAR1Process,
                        TauchenNormalMixtureAR1Process
```

The leading underscore is a signal — both to readers and to the linter — that **user
code must not import from these packages directly**. The leaf classes are surfaced
through `api/grids.py` and `api/processes.py`; the ABCs (`Grid`, `_ProcessGrid`, etc.)
are used by internal code but are not part of the documented public API.

Two design points worth knowing:

- **Process classes bundle both a discretization grid AND a transition mechanism**,
  unlike ordinary grids which are pure outcome-space. Users place
  `UniformIIDProcess(...)` in `Regime(states=...)` directly — the transition is invoked
  automatically. Putting a process class in `state_transitions` is a bug.
- **All vocabulary in the engine speaks of `process`, not `shock`.** Use `is_process`
  (on `VariableInfo`), `process_names` (on `Variables`), and `ProcessName` (typing
  alias). The codebase previously used `shock`; that word is now reserved for the
  colloquial meaning and never appears as an identifier.

## Private siblings of `api/`

Several `api/` modules have a private sibling that holds their implementation detail.
The pattern is the same throughout: `api/` keeps the class definitions and the public
top-level functions; the sibling holds validators, helpers, and I/O plumbing that
internal code is free to refactor.

```
_ages.py                ← STEP_UNITS, PSEUDO_STATE_NAMES, _parse_step,
                          _validate_age_grid / _validate_range / _validate_values
_regime/
├── _helpers.py         ← _default_H (default Bellman aggregator)
└── _validation.py      ← the eight validators called from Regime.__post_init__

_persistence/
├── _io.py              ← _atomic_dump, _save_pkl, _save_h5, _load_h5,
│                          _get_platform, _next_counter, _enforce_retention,
│                          _write_metadata, _write_environment_files
└── _snapshots.py       ← _save_solve_snapshot, _save_simulate_snapshot,
                          _strip_V_arr_from_result, _bind_forward_refs

simulation/_result_metadata.py
                        ← ResultMetadata + _compute_metadata,
                          _get_output_dtypes (renamed from
                          get_simulation_output_dtypes)
simulation/_result_dataframe.py
                        ← _create_flat_dataframe and the per-regime / per-period
                          assembly helpers, plus categorical conversion
simulation/_additional_targets.py
                        ← _resolve_targets, _compute_targets, and DAG helpers
                          for to_dataframe(additional_targets=...)
```

Why split these out? Two reasons:

- **The public surface is easier to audit.** `api/regime.py`, `api/persistence.py`, and
  `api/result.py` each contain only the dozen-or-so symbols users actually touch. A
  reader looking for "what is the public contract of a Regime?" sees that contract
  directly, without scrolling past validator bodies.
- **Internal helpers can move freely.** Anything under a leading-underscore name is
  internal — its location, signature, and existence can change without bumping the user
  surface. The split makes it cheap to refactor things like `_save_solve_snapshot`
  without touching `api/persistence.py`.

A note on shadowing: the canonical `Regime` lives in `engine.py`. The validators in
`_regime/_validation.py` operate on the user-facing `lcm.api.regime.Regime` and reach it
through TYPE_CHECKING-guarded imports to break the circular dependency at import time;
beartype resolves the forward references at first call.

## Engine-side: `engine.py`

`engine.py` holds the **canonical** post-processing dataclasses — the form the DP
machinery operates on:

- `Regime` — the canonical regime (distinct from the user-facing
  `lcm.api.regime.Regime`; in source files that import both we alias the user-facing one
  as `UserRegime`).
- `StateActionSpace` — pre-built state and action grids for a regime, with a
  `state_action_space(params)` method that fills in runtime-supplied grid points.
- `SolveFunctions` / `SimulateFunctions` — the compiled function bundles consumed by
  `solve` and `simulate`.
- `Variables` / `VariableInfo` — name + kind + topology metadata for every state and
  action in a regime.
- `PeriodRegimeSimulationData` — raw simulation output for one (regime, period) pair,
  before `SimulationResult` materialises a DataFrame.

The file name `engine.py` reflects what's inside: the engine's view of a model. It
replaces the older name `interfaces.py`, which was too generic to be useful.

## Build pipeline: `model_processing.py` and `regime_building/`

```
model_processing.py       ← top-level pipeline:
                            user regimes + params → canonical Model

regime_building/
├── processing.py         ← per-regime canonicalisation:
│                            UserRegime → engine.Regime
├── transitions.py        ← collect_state_transitions: walk user-supplied
│                            state_transitions into per-target callables
├── Q_and_F.py            ← build (Q, F) closure for solve / simulate
├── max_Q_over_a.py       ← argmax / max over action grids
├── V.py                  ← value-function interpolation info
├── h_dag.py              ← user-DAG resolution for H (Bellman aggregator)
├── next_state.py         ← compose per-state transitions into a single
│                            next_state function for simulation
├── ndimage.py            ← Map-coordinates wrapper for continuous interp
├── static_checks.py      ← process-time AST + n_outcomes derivation for
│                            stochastic state transitions (raises
│                            InvalidStateTransitionProbabilitiesError
│                            on subscript-order mismatches)
└── diagnostics.py        ← cold-path machinery invoked by validate_V to
                            pinpoint *which* intermediate produced a NaN
```

The two-step name (`model_processing` at the model level, `regime_building` per regime)
reflects what each layer actually does — the top level merges regimes and resolves fixed
params; each regime is then canonicalised independently.

The numerical checks fired at solve / simulate time **do not live in
`regime_building/`**. That package is the build pipeline; runtime is a different
lifecycle. The split:

- `regime_building/static_checks.py` runs at `Model(...)` construction time and can fail
  the build before any params are involved. It catches malformed user functions (e.g.,
  `probs_array[health, age]` where the signature is `(age, health)`) via AST analysis.
  Always on, never gated.
- `_transition_checks.py` (top-level, private) runs from `Model.solve()` /
  `Model.simulate()` before backward induction starts. It evaluates the regime and state
  transition functions on the regime's grid Cartesian product and verifies output shape,
  [0, 1] range, and sum-to-1. State checks are gated by `log_level != "off"` because the
  Cartesian product can blow up on models with many continuous-grid-dependent stochastic
  states.
- `solution/validate_V.py` runs *during* backward induction (after each period in
  `solve_brute.py`, and once on the V handed to `simulate.py`). On NaN it invokes the
  diagnostic-intermediates closure built in `regime_building/diagnostics.py` to pinpoint
  which intermediate (`U`, `F`, `E[V]`, `Q`) produced the NaN.

## Solve and simulate

```
solution/
└── solve_brute.py      ← backward induction loop:
                           V[T], V[T-1], ..., V[0] via max_Q_over_a

simulation/
├── simulate.py         ← forward sampling loop with state-action draws
├── initial_conditions.py
                        ← canonicalize / validate the user's
                          initial_conditions kwarg
└── core_helpers.py     ← shared subroutines
```

These are the JAX-traced hot paths. The DP and sampling logic is the *only* thing here;
everything that constructs the inputs (parameters, grids, transitions, compiled
callables) lives in `regime_building/` and is read out of the canonical `Regime`
instances.

## Params: boundary form vs. canonical form

```
params/
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

Two leaf types exist because params dicts can contain heterogeneous leaves (scalars,
arrays, named tuples, etc.). Wrapping them in `MappingLeaf` / `SequenceLeaf` lets the
pytree machinery treat them as opaque leaves rather than walking into them — important
when a "leaf" is itself a dict mapping named arguments to JAX arrays.

The `User*` types accept the wide boundary form (`int`, `float`, `np.ndarray`,
`pd.Series`, etc.). After `cast_params_to_canonical_dtypes` runs, only canonical
JAX-array leaves and canonical-narrow `MappingLeaf` / `SequenceLeaf` instances survive.
The downstream `solve` / `simulate` code only ever sees the canonical form.

## Pandas bridge: `pandas_utils.py`

A single module for converting between user-friendly `pd.Series` / `pd.DataFrame`
representations and the JAX arrays the engine expects. `array_from_series` is the
workhorse: it inspects the function source via AST helpers (in
`utils/ast_inspection.py`) to determine the expected multi-index order, then
materialises a properly-shaped JAX array.

This file gets used both at params processing (for any `pd.Series` leaves in user
params) and at simulation output (for building the result DataFrame).

## Utilities: `utils/`

Small, dependency-light helpers grouped by topic:

- `ast_inspection.py` — Parse a function body to find `probs_array[a, b]` subscript
  patterns. Used by the static AST check and by `pandas_utils`.
- `containers.py` — `ensure_containers_are_immutable`, `first_non_none`,
  `invert_regime_ids`.
- `dispatchers.py` — `productmap`, `vmap_1d`, `simulation_spacemap`. See
  [Dispatchers](dispatchers.ipynb).
- `functools.py` — `all_as_kwargs`, `get_union_of_args`.
- `logging.py` — `get_logger`, `format_duration`, log-formatting helpers.
- `namespace.py` — `flatten_regime_namespace` / `unflatten_regime_namespace` for the
  qualified-name pytree keys.

## Type aliases: `typing.py` vs `api/typing.py`

```
typing.py        ← engine-side aliases (FloatND, RegimeName, EconFunction, ...)
api/typing.py    ← user-facing aliases (UserParams, UserInitialConditions, ...)
```

The split mirrors the boundary / canonical distinction: User\* aliases accept wide
boundary types; the corresponding engine aliases (`Params`, `InitialConditions`)
describe the post-canonicalization form. `typing.py` re-exports the User\* aliases at
the bottom for backwards compatibility so `from lcm.typing import UserParams` keeps
working.

## Exceptions: `exceptions.py`

Every project-specific exception class lives here, all inheriting from `PyLCMError`.
They split into two categories:

- **Initialization errors** — raised at `Model(...)` / `Regime(...)` /
  `LinSpacedGrid(...)` construction time. These map beartype violations on user-facing
  constructors to a project-typed error (so users see e.g. `ModelInitializationError`,
  not a `BeartypeCallHintViolation`).
- **Runtime errors** — `InvalidValueFunctionError`,
  `InvalidRegimeTransitionProbabilitiesError`,
  `InvalidStateTransitionProbabilitiesError`, `InvalidParamsError`,
  `InvalidInitialConditionsError`. These fire from `_transition_checks.py` and
  `solution/validate_V.py` during solve / simulate.

## What's *not* in this map

A few things deliberately don't fit the boundary metaphor:

- `_jaxtyping_patch.py` — Bootstrap patch that has to run before any
  `jaxtyping`-annotated type is created. Lives at top level because of ordering, not
  design.
- `_beartype_conf.py` — Holds the two beartype configurations used in the package
  (internal claw + user-facing constructor decorators).
- `_config.py` — Build-time configuration constants (paths to test data, etc.).
- `dtypes.py` — Canonical-dtype resolution (`canonical_float_dtype()`), which depends on
  the JAX x64 setting.

These are small, stable, and don't naturally belong to any of the larger groupings.

## Reading order for new contributors

If you're reading the codebase for the first time, the path of least confusion is:

1. **`api/regime.py`** to see what users supply.
1. **`api/model.py`** to see what `Model.__init__` triggers.
1. **`model_processing.py`** for the top-level pipeline.
1. **`regime_building/processing.py`** for per-regime canonicalisation — the longest
   single file and the heart of the build.
1. **`engine.py`** for the canonical dataclasses the DP machinery consumes.
1. **`solution/solve_brute.py`** and **`simulation/simulate.py`** for the actual DP and
   sampling.

By the time you reach (6), the canonical form should feel familiar and the JAX-traced
code becomes easy to read.
