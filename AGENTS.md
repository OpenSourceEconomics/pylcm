@.ai-instructions/profiles/tier-a.md @.ai-instructions/modules/jax.md
@.ai-instructions/modules/pandas.md @.ai-instructions/modules/plotting.md
@.ai-instructions/modules/dags.md

# PyLCM

## Overview

PyLCM is a Python package for specification, solution, and simulation of finite-horizon
discrete-continuous dynamic choice models. The package uses JAX for numerical
computations and supports GPU acceleration.

## Build & Test

This project uses [pixi](https://pixi.sh/) for dependency management and task
automation. Python 3.14+ is required.

- `pixi run tests` - Run all tests
- `pixi run tests-with-cov` - Run tests with coverage reporting
- `pytest tests/test_specific_module.py` - Run specific test file
- `pytest tests/test_specific_module.py::test_function_name` - Run specific test
- `prek run ty --all-files` - Type checking with ty (a pre-commit hook; resolves
  third-party imports from the pixi env named in `[tool.ty] environment.python`, so run
  `pixi install` once)
- `prek run --all-files` - Run all pre-commit hooks (includes the `ty` hook)
- `pixi run -e docs build-docs` - Build documentation
- `pixi run -e docs view-docs` - Live preview documentation
- `pixi install` - Install dependencies
- `pixi run explanation-notebooks` - Execute explanation notebooks
- `prek install` - Install pre-commit hooks (after `pixi global install prek`)

## Architecture

### Core Components

**Model Definition (`src/lcm/model.py`, `src/lcm/regime.py`)**

- `Model`: User-facing class for defining dynamic choice models
- `Regime` (from `lcm.regime`): User-facing regime definition with utility, constraints,
  functions, actions, states, and state transitions (the `state_transitions` field). The
  regime transition is set via the `transition` field.
- `Phased(solve=..., simulate=...)`: phase-specific variants of a regime-slot value
  (functions, states, state transitions, the regime transition). A bare value broadcasts
  to both phases.
- Models must have at least one terminal regime and one non-terminal regime
- Models support transitions between multiple regimes

**Canonical Processing (`src/_lcm/engine.py`)**

- The pipeline from user input to engine form: `Regime` → finalized `Regime` (same
  class, post-merge) → `PhasedRegimeSpec` (phase split + canonical laws) →
  `_lcm.engine.Regime` (compiled function sets).
- `finalize_regimes` (`src/_lcm/regime_building/finalize.py`): finalizes each user
  `Regime` at model build into the form the model actually runs — model-level
  `derived_categoricals` merged in, default `H` injected, completeness validated (a
  `utility` entry, state-transition coverage, state/action overlap, distributed-grid
  rules). The result is a plain `lcm.regime.Regime`, still user vocabulary, so the
  params template reads the user's coarseness off it. Internal signatures mark the
  post-merge form with `FinalizedUserRegime` (defined in `finalize.py`) — an alias the
  type checker treats as plain `Regime`; it enforces nothing and only documents that
  finalization has happened. A bare `Regime` validates only local, value-shape
  properties at construction — completeness may be satisfied only at the model level.
- `canonicalize_regimes` (`src/_lcm/regime_building/canonicalize.py`): the model-level
  canonicalization stage. Rewrites every phase slice's laws into the canonical
  target-granular form `Mapping[RegimeName, law]` over exactly the reachable targets
  carrying the state in that phase — bare laws broadcast, per-target dicts pass through,
  `fixed_transition` entries desugar into per-target identities. Reachability has a
  single source of truth — the regime transition (per-target dict ⇒ its key set; coarse
  ⇒ all regimes) — and is resolved here, once; the engine-side extraction is a pure
  transpose. Rule: the params template reads the user (finalized) spec, the engine reads
  the canonical spec.
- `Regime` (from `_lcm.engine`): Canonical representation produced by `process_regimes`
  from a user-facing `Regime`. Internal engine code threads this form. Inside boundary
  files that import both, alias the user form as
  `from lcm.regime import Regime as UserRegime`.
- The canonical `Regime` carries only phase-invariant data plus two frozen phase
  namespaces — every phase-dependent read names its phase in the access path:
  - `regime.solution` (`SolutionPhase`): solve variables and grids (a carried state
    contributes no axis; productmap order), compiled solve function sets,
    `state_action_space()`.
  - `regime.simulation` (`SimulationPhase`): per-subject variables (solve states plus
    carried-only states, appended — not a productmap order), grids including each
    carried state's domain, `carried_only_state_names` / `carried_grids`, compiled
    simulate function sets. Its published `functions` are imputation-free (carried
    states are leaves fed with carried values); only the decision functions (Q_and_F /
    argmax) keep the solve imputation.
- `normalize_regime_phases` (`src/_lcm/regime_building/phases.py`) is the single
  phase-resolution boundary: it expands every regime slot into per-phase
  `RegimePhaseSpec` slices (`PhasedRegimeSpec.solution` / `.simulation`) and applies the
  phase grammar. Phase is a broadcast dimension of the user spec — a bare slot value
  applies to both phases, `Phased(solve=..., simulate=...)` specifies each phase
  explicitly:
  - `functions` and `state_transitions` accept `Phased` (per-phase implementations /
    laws of motion); `transition` accepts `Phased` with matching forms (and, for
    per-target dicts, identical key sets).
  - `states` accept `Phased(solve=callable, simulate=Grid)` — the carried state: derived
    (no grid axis) during backward induction, a genuine seeded-and-evolved state in
    simulation, with its law of motion in the regular `state_transitions` slot. All
    other solve/simulate combinations are rejected.
  - `constraints`, `actions`, `active`, and `derived_categoricals` are phase-invariant;
    `Phased` is rejected there with an explanation. `Phased` is outermost-only (never
    inside a per-target dict) and never nested.
- `StateActionSpace`: Manages state-action combinations for solution/simulation
- `PeriodRegimeSimulationData`: Raw simulation results for one period in one regime

**Value Function Representation (`src/_lcm/regime_building/V.py`)**

- `VInterpolationInfo`: Metadata for working with function outputs on state spaces

**Solution (`src/_lcm/solution/`)**

- `backward_induction.py`: Brute force dynamic programming solver using backward
  induction
- Entry point: `model.solve()` method

**Simulation (`src/_lcm/simulation/`)**

- `simulate.py`: Forward simulation of solved models
- `SimulationResult` (`lcm/result.py`): result object with deferred DataFrame
  computation
- Entry point: Model methods (`solve()`, `simulate()`)

**Grid System (`src/_lcm/grids/`, `src/_lcm/processes/`)**

- `DiscreteGrid`: Categorical variables with string labels (pure outcome space).
- `LinSpacedGrid`: Linearly spaced grid (start, stop, n_points).
- `LogSpacedGrid`: Logarithmically spaced grid (start, stop, n_points).
- `IrregSpacedGrid`: Irregularly spaced grid (points tuple).
- `PiecewiseLinSpacedGrid`: Piecewise linearly spaced grid with breakpoints.
- `PiecewiseLogSpacedGrid`: Piecewise logarithmically spaced grid with breakpoints.
- `AgeGrid`: Lifecycle age grid (start, stop, step or exact_values)
- `@categorical(ordered=...)`: Decorator factory for creating categorical classes with
  auto-assigned `ScalarInt` (0-d `jnp.int32`) codes. Requires explicit `ordered=True` or
  `ordered=False`. Every field must be annotated as `ScalarInt` (from `lcm.typing`) —
  other annotations raise `CategoricalDefinitionError` at decoration time.
- **Stochastic processes** (in `src/_lcm/processes/`): `UniformIIDProcess`,
  `NormalIIDProcess`, `LogNormalIIDProcess`, `NormalMixtureIIDProcess`,
  `TauchenAR1Process`, `RouwenhorstAR1Process`, `TauchenNormalMixtureAR1Process`. These
  bundle a discretized grid and its transition mechanism — they go in `states` and must
  NOT appear in `state_transitions`. Import directly from `lcm`
  (`from lcm import NormalIIDProcess`).

Grid class hierarchy: `Grid` is the base class. `ContinuousGrid(Grid)` is the base for
continuous grids with `get_coordinate` method. `UniformContinuousGrid(ContinuousGrid)`
is for grids with start/stop/n_points (LinSpacedGrid, LogSpacedGrid inherit from it).
Other continuous grids (IrregSpacedGrid, PiecewiseLinSpacedGrid, PiecewiseLogSpacedGrid)
inherit directly from ContinuousGrid. `_ContinuousStochasticProcess(ContinuousGrid)` is
the base for the stochastic process classes. `DiscreteGrid` supports stochastic
transitions via `MarkovTransition`-wrapped callables in `state_transitions`.

Grids are pure outcome-space definitions — they define what values a variable can take.
**State transitions** live on the `Regime` via the `state_transitions` field, which maps
state names to transition functions (`fixed_transition(state_name)` for fixed states).
Wrap in `MarkovTransition` for stochastic transitions. Per-target dicts map target
regime names to transition functions for target-dependent transitions.

### Processing Pipeline

1. User defines `Regime`(s) with grids, functions, states/actions
1. User creates `Model` from a dict of regimes with `ages` and `regime_id_class`
1. `process_regimes()` converts user-facing `Regime` instances into canonical
   `_lcm.engine.Regime` objects and pre-compiles optimization functions
1. `model.solve()` performs backward induction using dynamic programming
1. `model.simulate()` performs forward simulation using solved policy functions
1. `SimulationResult.to_dataframe()` creates flat DataFrame output

### Key Numerical Components

- **Value Functions**: Computed via backward induction
- **Policy Functions**: Optimal actions given states
- **Q Functions**: Action-value functions for discrete choices
- **State Transitions**: Next period state computation
- **Constraints**: Feasibility filtering for state-action combinations

### Testing Structure

- `tests/test_models/`: Shared test models (deterministic, stochastic variants)
- `tests/solution/`: Tests for solution algorithms
- `tests/simulation/`: Tests for simulation functionality
- `tests/regime_building/`: Tests for regime compilation pipeline
- `tests/data/`: Analytical solutions and regression test data

## Model and Regime Interface

### Regime Definition

The `Regime` class defines a single regime in the model. The regime name is specified as
the key in the `regimes` dict passed to `Model`:

```python
# Non-terminal regime
Regime(
    transition=next_regime_func,                  # Required: regime transition function (None → terminal)
    active=lambda age: 25 <= age < 65,           # Optional: age-based predicate (default: always True)
    states={                                     # Pure outcome-space grids
        "wealth": LinSpacedGrid(...),
        "education": DiscreteGrid(EduStatus),
    },
    state_transitions={                          # How states evolve over time
        "wealth": next_wealth,                   # Deterministic transition
        "education": fixed_transition("education"),  # Fixed state (identity law)
    },
    actions={"action_name": Grid, ...},          # Action grids (can be empty)
    functions={                                  # Must include "utility"; other functions optional
        "utility": utility_function,
        "name": helper_func, ...
    },
    constraints={"name": constraint_func, ...},  # Optional: constraint functions
)

# Terminal regime (transition=None, no state_transitions)
Regime(
    transition=None,
    functions={"utility": terminal_utility},
    states={"wealth": LinSpacedGrid(...)},
)

# Target-dependent transitions (keyed by target regime name)
Regime(
    transition=next_regime_func,
    states={"health": DiscreteGrid(Health)},
    state_transitions={
        "health": {
            "working": MarkovTransition(health_probs_working),
            "retired": MarkovTransition(health_probs_retired),
        },
    },
    ...
)
```

**Regime Requirements:**

- `transition` is required: the regime transition, or `None` for terminal regimes.
  `terminal` is a derived property (`self.transition is None`). Three forms:
  - bare callable ⇒ deterministic, returns the target regime id; every regime is
    reachable
  - `MarkovTransition` ⇒ stochastic, returns a probability vector over all regimes;
    every regime is reachable
  - per-target dict `{target_regime: MarkovTransition(prob_func)}` ⇒ stochastic; each
    cell returns that target's probability and the key set declares the regime's
    reachable targets — omitted regimes are structurally unreachable. Cells must be
    `MarkovTransition`-wrapped; `transition={}` is rejected (terminality is `None`).
    Cell params nest under the target in the template
    (`template[regime][target]["next_regime"]`).
- `active` is optional; defaults to `lambda _age: True` (always active)
- `functions` must contain a `"utility"` entry (the utility function); checked when the
  model finalizes its regimes, not at `Regime` construction
- `state_transitions` maps state names to transition functions. Every non-process state
  in a non-terminal regime must have an entry (checked at model build).
  `fixed_transition(state_name)` marks a fixed state (identity law; its argument must
  match the dict key). `None` is rejected. Wrap in `MarkovTransition` for stochastic
  transitions.
- Per-target dicts in `state_transitions` map target regime names to transition
  functions — every reachable target carrying the state must be listed, and no
  unreachable or unknown target may be (checked at model build; narrow reachability with
  a per-target regime transition). Within a per-target dict, stochasticity must be
  consistent (all `MarkovTransition` or none).
- Stochastic processes have intrinsic transitions and must NOT appear in
  `state_transitions`.
- Terminal regimes must have empty `state_transitions`.
- Regime names (dict keys) cannot contain the reserved separator `__`

### Model Creation

```python
from lcm import AgeGrid, categorical


@categorical(ordered=False)
class RegimeId:
    working: ScalarInt
    retired: ScalarInt


Model(
    regimes={  # Required: dict mapping names to Regime instances
        "working": working_regime,
        "retired": retired_regime,
    },
    ages=AgeGrid(start=25, stop=75, step="Y"),  # Required: lifecycle age grid
    regime_id_class=RegimeId,  # Required: dataclass mapping names to indices
    description="Optional description",
    enable_jit=True,  # Control JAX compilation (default: True)
)
```

**Model-level regime slots (broadcast):**

`Model(functions=..., constraints=..., states=..., state_transitions=..., actions=...)`
accepts exactly what the regime-level slot accepts (incl. `Phased`, stochastic
processes, per-target dicts, `fixed_transition`). Each entry is merged into every regime
under the exactly-one-level rule — a name is defined at model level or regime level,
never both (ambiguity errors); a regime-level `None` masks the model entry (masking a
state also drops its broadcast law; an unbound mask errors). Broadcast laws are not
merged into terminal regimes.

Broadcast states and actions are pruned per regime by DAG reachability: a broadcast
variable survives only where a root computation (utility, `H`, constraints, derived
categoricals, the regime transition, or a law of motion toward a reachable target that
keeps the state) transitively reads it, per phase slice, pruned only when dead in both
phases. Regime-level declarations are never pruned. `model.pruned_variables` records the
result per regime.

`distributed=True` is legal only on model-level states; a sharded state pruned from a
non-terminal regime is an error (unshard or make the regime use it). `batch_size` stays
per-declaration.

**Model Requirements:**

- Must have at least one terminal regime and one non-terminal regime
- `regime_id_class` must be a dataclass with fields matching regime names (use
  `@categorical`)
- Field values are consecutive `ScalarInt` (0-d `jnp.int32`) scalars starting from 0,
  auto-assigned by `@categorical`

### Core Methods

- `model.solve(params=params, log_level="debug")` - Solve the model and return value
  function arrays per period and regime
- `model.simulate(params=params, initial_conditions=initial_conditions, period_to_regime_to_V_arr=period_to_regime_to_V_arr, log_level="debug")`
  \- Simulate forward given solution. `period_to_regime_to_V_arr` is optional; when
  `None`, the model is solved automatically before simulating.
- `log_level` is **required** on both `solve()` and `simulate()`
  (`off < warning < progress < debug`). It governs all runtime validation: `"off"` skips
  it, `"warning"` / `"progress"` warn and continue, `"debug"` raises. Start projects at
  `"debug"`.

### Derived Categoricals

When parameters are indexed by a DAG function output (not a model state/action), declare
`derived_categoricals={"name": DiscreteGrid(CategoryClass)}` on the `Regime` that uses
it. For convenience, model-level `derived_categoricals` on `Model(...)` are broadcast to
all regimes under the exactly-one-level rule — a name is declared at model level or
regime level, never both (ambiguity errors, also when the grids match). Functions used
as derived categoricals must return **integer** types, not booleans — JAX cannot use
booleans as array indices inside JIT. Use `jnp.int32(...)` to cast.

### SimulationResult

`simulate()` returns a `SimulationResult` object:

```python
result = model.simulate(
    params=params,
    initial_conditions=initial_conditions,
    period_to_regime_to_V_arr=None,
    log_level="debug",
)

# Convert to DataFrame (deferred computation)
df = result.to_dataframe()

# With additional computed targets (utility, functions, constraints)
df = result.to_dataframe(additional_targets=["utility", "consumption"])

# All available targets
df = result.to_dataframe(additional_targets="all")

# Integer codes instead of categorical labels
df = result.to_dataframe(use_labels=False)

# Keep the frozen post-entry terminal-regime rows (absorbing representation);
# the default (terminal_rows="first") emits each subject's terminal entry row only
df = result.to_dataframe(terminal_rows="all")

# Access metadata
result.regime_names  # list[str]
result.state_names  # list[str]
result.action_names  # list[str]
result.n_periods  # int
result.n_subjects  # int
result.available_targets  # list[str] - computable additional targets

# Access raw data for advanced users
result.raw_results  # dict[RegimeName, dict[int, PeriodRegimeSimulationData]]
result.flat_params  # FlatParams
result.period_to_regime_to_V_arr  # dict[int, dict[RegimeName, FloatND]]

# Persistence: writes `arrays/` (orbax), `metadata.pkl` (cloudpickle),
# and `simulated_data.arrow` (feather of `to_dataframe`).
result.save(directory="path/to/dir")
loaded = SimulationResult.load(directory="path/to/dir")
```

### Initial Conditions Format

Initial conditions use a flat dictionary with state names plus `"regime_id"`:

```python
initial_conditions = {
    "wealth": jnp.array([1.0, 2.0, 3.0]),
    "health": jnp.array([0.5, 0.8, 0.3]),
    "regime_id": jnp.array([RegimeId.working, RegimeId.working, RegimeId.retired]),
}
```

### Key Attributes

- `model.get_params_template()` - Mutable copy of the parameter template (dict by regime
  name)
- `model.user_regimes` - Immutable mapping of regime names to plain `Regime` objects:
  the regimes as the model runs them, finalized at model build (model-level slots
  merged, default `H` injected, completeness validated), still in user vocabulary
- `model._regimes` - Immutable mapping of regime names to canonical `Regime` objects
  (`_lcm.engine.Regime`) produced by `process_regimes`. Private — the canonical form is
  engine-internal; user code should read `user_regimes`.
- `model.pruned_variables` - Immutable mapping of regime names to the broadcast
  states/actions pruned from that regime by DAG reachability
- `model.ages` - The AgeGrid defining the lifecycle
- `model.n_periods` - Number of periods in the model (derived from `ages`)
- `model.regime_names_to_ids` - Immutable mapping from regime names to integer indices

## Testing

### Test-Driven Development — always

**Always write the test first, watch it fail, then implement.** No exceptions for new
behavior or bug fixes. Tests are not an afterthought, they are the spec.

The cycle:

1. **Red.** Write a failing test that asserts the desired behavior in user-facing terms.
   Run it. Confirm it fails for the *right* reason (the missing behavior — not a typo,
   not an import error).
1. **Green.** Write the smallest amount of code that makes the test pass.
1. **Refactor.** Clean up while keeping the test green.

Apply per case:

- **New feature** → red-green-refactor.
- **Bug fix** → reproduce as a failing test before writing the fix. The test then
  prevents regression.
- **Refactor (no behavior change)** → existing tests are the spec. Keep them green
  before, during, and after. No new test needed if behavior is unchanged; if you find a
  behavior gap, fill it with a new test *before* refactoring.

### Test docstrings — describe behavior, not history

Test docstrings state what *should* be true, in user-facing terms. Pretend the reader
has never seen the PR. They should not need to.

```python
# Good — behavior, in plain language
def test_simulate_with_chained_transitions_yields_expected_next_wealth():
    """`next_wealth_t = wealth_t - c_t + 0.1 * next_aime_t` holds in simulation."""


# Bad — rehearses the prior bug or implementation history
def test_solve_resolves_chain_via_dags():
    """Before the fix, `_resolve_fixed_params` raised
    `InvalidParamsError: Missing required parameter: ...` because
    `create_regime_params_template` classified ..."""
```

Rule of thumb: **would the docstring still make sense in 9 months without the PR
context?** If not, rewrite it.

### Concrete-value assertions

Assert *what* the result is, not just that it didn't crash.

```python
# Good — analytical value with explicit tolerance
np.testing.assert_allclose(curr["wealth"], expected_next_wealth, atol=1e-6)

# Bad — passes whether the math is right or not
assert not jnp.any(jnp.isnan(V_arr))
assert df["wealth"].notna().all()
```

`not isnan` and `no exception raised` belong in CI smoke tests, not in the unit tests
for the feature itself.

### Mechanics

- Use plain pytest functions, never test classes (`class TestFoo`)
- Use `@pytest.mark.parametrize` for test variations

## Docstring Style

Docstrings and inline comments describe the code's *current* state in user-facing terms.
The 9-month-without-PR-context reader is the audience: a docstring that survives that
test stays useful; one that rehearses the diff or the prior implementation rots
immediately.

This applies to **all** docstrings and comments — source and tests. For tests
specifically, see also "Test docstrings — describe behavior, not history" above.

### Describe state, not history

State what is true now. Don't reference prior designs, removed code, or what was
changed. Words like "earlier", "previously", "now", "formerly", "the old", "before the
fix" are red flags.

```python
# Good — forward-looking constraint
class _DiagnosticRow:
    """Metadata captured during the backward-induction loop.

    Holds only Python-scalar metadata — no device-array references —
    so every (regime, period) row stays at a few bytes regardless of
    grid size.
    """


# Bad — rehearses prior design
class _DiagnosticRow:
    """Metadata captured during the backward-induction loop.

    Holds only Python-scalar metadata. The earlier design captured
    state_action_space and a closure directly on each row, which
    pinned every period's V template in device memory until the
    post-loop flush.
    """
```

### No PR numbers, no model-specific magic numbers

PR references (`#334 removed the host stalls`, `the bug was fixed in #42`) rot as the
codebase evolves and provide no useful signal to a reader who isn't already in context.
Magic numbers tied to a specific model size or hardware
(`~2 MB at production grid sizes`, `fits on a 16 GB device`) imply a fixed scale that's
only true on whichever model/box the comment was written against. State the qualitative
dependency instead.

```python
# Good — qualitative dependency
# Frees per-period intermediate buffers (V_arr-shaped, so
# model-dependent) so they don't stack up across the loop.

# Bad — PR reference + magic number
# Frees per-period intermediate buffers (~2 MB each at production
# grid sizes) so we don't re-introduce the host stalls that #334
# removed.
```

### Bulleted lists for enumerated cases

When describing a fixed set of cases (log levels, regime kinds, parameter types,
dispatch strategies), use one bullet per case rather than running prose. Bullets scan;
prose hides cases.

```python
# Good — scannable
# Gate falls out of the public log level:
# - `"off"` ⇒ nothing (skips even the NaN fail-fast)
# - `"warning"` / `"progress"` ⇒ NaN/Inf only
# - `"debug"` ⇒ adds the min/max/mean trio


# Bad — buried in prose
# Gate falls out of the public log level: `"off"` ⇒ nothing,
# `"warning"` / `"progress"` ⇒ NaN/Inf only, `"debug"` ⇒ adds the
# min/max/mean trio. `"off"` skips even the NaN fail-fast.
```

## Development Notes

### JAX Integration

- All numerical computations use JAX arrays
- GPU support available via jax[cuda13] (Linux) or jax-metal (macOS)
- Functions are JIT-compiled during Model initialization for performance
- `MappingProxyType` is registered as a JAX pytree for use in JIT-compiled functions

### Immutability

- Internal data structures use `MappingProxyType` instead of `dict` for immutability
- Type annotations use `Mapping` for read-only dict-like interfaces
- User-provided dicts in `Regime` are automatically wrapped in `MappingProxyType`

### Type System

- Extensive use of typing with custom types: user-facing aliases in `src/lcm/typing.py`,
  engine-side aliases and protocols in `src/_lcm/typing.py`
- Type checking with ty (`prek run ty --all-files`)
- Use `# ty: ignore[error-code]` for type suppression, never `# type: ignore`
- JAX typing integration via jaxtyping

#### Domain string aliases

The following PEP 695 aliases (`type X = str`) live in `src/lcm/typing.py` (re-exported
from `src/_lcm/typing.py`, so `from _lcm.typing import RegimeName` keeps working) and
exist purely to make signatures self-documenting. They are runtime-equivalent to `str`;
ty erases them, so misuse never crashes — it just hides intent. Prefer the alias over
bare `str` whenever a string slot has a fixed semantic role.

| Alias                    | Use for                                                                                                                                                 |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `RegimeName`             | Names of regimes — keys of `regimes`, `internal_regimes`, `regime_names_to_ids`, `state_action_spaces`, `period_to_regime_to_V_arr`, `regime_to_V_arr`. |
| `StateName`              | Names of states — entries of `state_names`, keys of `regime.states`, `states_per_regime` values.                                                        |
| `ActionName`             | Names of actions — entries of `action_names`, keys of `regime.actions`.                                                                                 |
| `StateOrActionName`      | Mixed flat keys covering both states and actions — `flat_grids`, `all_grids[regime]` values, `state_and_discrete_action_names`.                         |
| `ProcessName`            | Subset of `StateName` for stochastic processes — keys of `_ContinuousStochasticProcess`-typed mappings, process-transition helpers.                     |
| `FunctionName`           | User-supplied function names — `"utility"`, `"H"`, helpers; keys of `Regime.functions`, `derived_categoricals`.                                         |
| `TransitionFunctionName` | Names of transition callables — `next_<state>`, `weight_next_<state>`; keys of `state_transitions` and per-target dicts.                                |

When a string slot covers more than one of the categories above, prefer a union (e.g.
`dict[RegimeName | TransitionFunctionName, ...]`) over bare `str`. Plain `str` is the
right type only when the keys really are heterogeneous and don't map onto any of the
aliases — DataFrame column labels, free-form param-template leaf strings, and similar.
Use `NewType` only when an opaque ID is required; the project has not needed that so
far.

### Code Standards

- Ruff for linting and formatting (configured in pyproject.toml)
- Google-style docstrings
- All functions require type annotations
- Pre-commit hooks ensure code quality
- Never use `from __future__ import annotations` — this project requires Python 3.14+

### Module Layout

Write "deep" modules: important public function(s) at the top, private helpers below.
Readers should see the API first without scrolling past implementation details.

Never add decorative section-separator comments like:

```python
# ---------------------------------------------------------------------------
# Section name
# ---------------------------------------------------------------------------
```

Code structure should be self-evident from function names and ordering.

### Naming and Docstring Conventions

- **Noun vs verb for phase vocabulary — "the name tells you what you get."** Use the
  noun (`solution`, `simulation`) when the name yields a *thing* — an artifact or bundle
  of phase artifacts: `regime.solution`, `SolutionPhase`, `SimulationResult`,
  `PhasedRegimeSpec.solution`. Use the verb (`solve`, `simulate`) when the name denotes
  the *act* or selects a variant to be used when acting: `model.solve()`,
  `Phased(solve=f, simulate=g)`, `phase: Literal["solve", "simulate"]`, and all
  attributive compounds in identifiers and prose (`solve_transitions`, "the solve-phase
  consumer", "the solve grid"). Litmus: you get a bag → noun; you get behavior or name
  the act → verb.
- **No unnecessary parameter aliases.** When a function has a single (or very few) call
  site(s), the parameter name should match the variable name being passed. Don't shorten
  parameter names just for brevity — e.g., use
  `regime_transition_probs=regime_transition_probs` not `probs=regime_transition_probs`.
- **Docstrings must match type annotations.** Use the type name from the annotation:
  - `Mapping[...]` → "Mapping of ..." in docstrings
  - `MappingProxyType[...]` → "Immutable mapping of ..." in docstrings
  - `tuple[...]` → "Tuple of ..." in docstrings
  - `list[...]` → "List of ..." in docstrings
  - Never write "Dict" when the annotation is `Mapping` or `MappingProxyType`
- **Consistent naming across a file.** When multiple functions in the same file use the
  same concept (e.g., `arg_names`), use the same parameter name everywhere — don't
  introduce synonyms like `parameters`.
- **Helper function names follow `{verb}_{qualifier}_noun` patterns.** E.g.,
  `get_irreg_coordinate`, `find_irreg_coordinate`, `get_linspace_coordinate` — not
  `get_coordinate_irreg`.
- **Pick the single narrowest jaxtyping alias — never scalar/array `@overload` pairs,
  never `ScalarX | XND` unions.** ty erases jaxtyping shape annotations: `ScalarFloat`,
  `Float1D`, and `FloatND` all reveal as `Array`, so scalar/array `@overload` pairs and
  `ScalarFloat | FloatND`-style unions add zero static precision — they are pure noise.
  At runtime, beartype treats a 0-d float array as satisfying both `ScalarFloat` and
  `FloatND`, so `ScalarFloat ⊆ FloatND` and the union is redundant. Annotate each slot
  with the one alias that matches its genuine rank: `ScalarFloat`/`ScalarInt` for
  fixed-0-d, `Float1D`/`Int1D` for fixed-1-d, `FloatND`/`IntND` for genuinely rank-
  polymorphic. Never use a bare `Array` annotation — always reach for the narrowest
  `lcm.typing` alias.
- **`func` for callable abbreviations** — use `func`, `func_name`, `func_params` (never
  `fn`). Full word `function(s)` in dataclass field names and public method names.
- **Singular `state_names` / `action_names`** — not `states_names` / `actions_names`.
- **`arg_names`** — not `argument_names`.
- **Imperative mood for docstring summary lines.** Write "Return the value" not "Returns
  the value". The summary line uses bare imperative: "Create", "Get", "Compute",
  "Convert", etc.
- **Inline field docstrings (PEP 257) for dataclass attributes.** Place a `"""..."""` on
  the line after each field instead of listing fields in an `Attributes:` section in the
  class docstring.
- **MyST syntax in docstrings, not reStructuredText.** Use `` `code` `` (single
  backticks) for inline code, `$...$` for inline math, ```` ```{math} ```` fences for
  display math, and `[text](url)` for links. Never use rST-style ``` `` code `` ```,
  `:math:`, `:func:`, or `` `link <url>`_ ``.

### Plotting

- Always use **plotly** for visualizations, never matplotlib. Use `plotly.graph_objects`
  and `plotly.subplots.make_subplots`.

### Notebooks

Explanation notebooks live in `docs/explanations/*.ipynb`. After editing one, verify:

- Each cell's `source` is a JSON array of lines (one array element per line), never a
  single multi-line string — a one-string `source` produces an unreadable diff.
- Outputs and execution counts are stripped (`pixi run nbstripout <file>`).
- Markdown and code use literal UTF-8 characters (`—`, `→`, `μ`), never `\u`-style
  escape sequences.

### Key Dependencies

- **jax**: Numerical computation
- **jaxtyping**: Array type annotations
- **pandas**: DataFrame output
- **dags**: Function composition and nested namespace flattening
- **plotly**: Plotting and visualization
