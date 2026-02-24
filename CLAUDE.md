# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in
this repository.

## Project Overview

PyLCM is a Python package for specification, solution, and simulation of finite-horizon
discrete-continuous dynamic choice models. The package uses JAX for numerical
computations and supports GPU acceleration.

## Development Environment

This project uses [pixi](https://pixi.sh/) for dependency management and task
automation. Python 3.14+ is required.

## Common Development Commands

### Testing

- `pixi run tests` - Run all tests
- `pixi run tests-with-cov` - Run tests with coverage reporting
- `pytest tests/test_specific_module.py` - Run specific test file
- `pytest tests/test_specific_module.py::test_function_name` - Run specific test

### Code Quality

- `pixi run ty` - Type checking with ty
- `prek run --all-files` - Run all pre-commit hooks

### Documentation

- `pixi run -e docs docs` - Build documentation
- `pixi run -e docs view-docs` - Live preview documentation

### Environment Setup

- `pixi install` - Install dependencies
- `pixi run explanation-notebooks` - Execute explanation notebooks
- `prek install` - Install pre-commit hooks (after `pixi global install prek`)

## Code Architecture

### Core Components

**Model Definition (`src/lcm/model.py`, `src/lcm/regime.py`)**

- `Model`: User-facing class for defining dynamic choice models
- `Regime`: Defines a single regime with utility, constraints, functions, actions, and
  states. State transitions live on grids; regime transitions live on the regime.
- Models must have at least one terminal regime and one non-terminal regime
- Models support transitions between multiple regimes

**Internal Processing (`src/lcm/interfaces.py`)**

- `InternalRegime`: Internal representation after processing user regime
- `StateActionSpace`: Manages state-action combinations for solution/simulation
- `StateSpaceInfo`: Metadata for working with function outputs on state spaces
- `PeriodRegimeSimulationData`: Raw simulation results for one period in one regime

**Solution (`src/lcm/solution/`)**

- `solve_brute.py`: Brute force dynamic programming solver using backward induction
- Entry point: `model.solve()` method

**Simulation (`src/lcm/simulation/`)**

- `simulate.py`: Forward simulation of solved models
- `result.py`: `SimulationResult` class with deferred DataFrame computation
- Entry point: Model methods (`solve()`, `simulate()`, `solve_and_simulate()`)

**Grid System (`src/lcm/grids.py`)**

- `DiscreteGrid`: Categorical variables with string labels. Optional `transition`
  parameter for state transitions.
- `LinSpacedGrid`: Linearly spaced grid (start, stop, n_points). Optional `transition`.
- `LogSpacedGrid`: Logarithmically spaced grid (start, stop, n_points). Optional
  `transition`.
- `IrregSpacedGrid`: Irregularly spaced grid (points tuple). Optional `transition`.
- `PiecewiseLinSpacedGrid`: Piecewise linearly spaced grid with breakpoints. Optional
  `transition`.
- `PiecewiseLogSpacedGrid`: Piecewise logarithmically spaced grid with breakpoints.
  Optional `transition`.
- `AgeGrid`: Lifecycle age grid (start, stop, step or precise_values)
- `@categorical`: Decorator for creating categorical classes with auto-assigned integer
  codes
- **ShockGrids** (in `src/lcm/shocks/`): `Rouwenhorst`, `Tauchen`, `Normal`, `Uniform`.
  These have intrinsic transitions — do NOT accept a `transition` parameter. Import as
  modules (`import lcm.shocks.iid`) and use qualified access
  (`lcm.shocks.iid.Uniform(...)`), never `from lcm.shocks.iid import Uniform`.

Grid class hierarchy: `Grid` is the base class. `ContinuousGrid(Grid)` is the base for
continuous grids with `get_coordinate` method. `UniformContinuousGrid(ContinuousGrid)`
is for grids with start/stop/n_points (LinSpacedGrid, LogSpacedGrid inherit from it).
Other continuous grids (IrregSpacedGrid, PiecewiseLinSpacedGrid, PiecewiseLogSpacedGrid)
inherit directly from ContinuousGrid. `_ShockGrid(ContinuousGrid)` is the base for
stochastic continuous grids. `DiscreteMarkovGrid(DiscreteGrid)` is the base for discrete
Markov-chain grids.

**State transitions** are attached directly to grid objects via the `transition`
parameter. A state with no `transition` is fixed (time-invariant) — an identity
transition is auto-generated during preprocessing.

### Processing Pipeline

1. User defines `Regime`(s) with grids, functions, states/actions
1. User creates `Model` from a dict of regimes with `ages` and `regime_id_class`
1. `process_regimes()` converts to `InternalRegime` and pre-compiles optimization
   functions
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
- `tests/input_processing/`: Tests for model processing pipeline
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
    states={                                     # State grids with optional transitions
        "wealth": LinSpacedGrid(..., transition=next_wealth),  # Time-varying state
        "education": DiscreteGrid(EduStatus, transition=None),   # Fixed state
    },
    actions={"action_name": Grid, ...},          # Action grids (can be empty)
    functions={                                  # Must include "utility"; other functions optional
        "utility": utility_function,
        "name": helper_func, ...
    },
    constraints={"name": constraint_func, ...},  # Optional: constraint functions
)

# Terminal regime (transition=None)
Regime(
    transition=None,
    functions={"utility": terminal_utility},
    states={"wealth": LinSpacedGrid(...)},
)
```

**Regime Requirements:**

- `transition` is required: the regime transition function, or `None` for terminal
  regimes. `terminal` is a derived property (`self.transition is None`).
- `active` is optional; defaults to `lambda _age: True` (always active)
- `functions` must contain a `"utility"` entry (the utility function)
- State transitions live on grids via the `transition` parameter. States without a
  `transition` are fixed (time-invariant) — an identity transition is auto-generated.
- ShockGrids have intrinsic transitions and do not need a `transition` parameter.
- Fixed states (no transition) must always pass `transition=None` explicitly — never
  rely on the default.
- Regime names (dict keys) cannot contain the reserved separator `__`

### Model Creation

```python
from lcm import AgeGrid, categorical


@categorical
class RegimeId:
    working: int
    retired: int


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

**Model Requirements:**

- Must have at least one terminal regime and one non-terminal regime
- `regime_id_class` must be a dataclass with fields matching regime names (use
  `@categorical`)
- Field values must be consecutive integers starting from 0 (auto-assigned by
  `@categorical`)

### Core Methods

- `model.solve(params)` - Solve the model and return value function arrays per period
  and regime
- `model.simulate(params, initial_states, initial_regimes, V_arr_dict)` - Simulate
  forward given solution
- `model.solve_and_simulate(params, initial_states, initial_regimes)` - Combined solve
  and simulate

### SimulationResult

Both `simulate()` and `solve_and_simulate()` return a `SimulationResult` object:

```python
result = model.solve_and_simulate(params, initial_states, initial_regimes)

# Convert to DataFrame (deferred computation)
df = result.to_dataframe()

# With additional computed targets (utility, functions, constraints)
df = result.to_dataframe(additional_targets=["utility", "consumption"])

# All available targets
df = result.to_dataframe(additional_targets="all")

# Integer codes instead of categorical labels
df = result.to_dataframe(use_labels=False)

# Access metadata
result.regime_names  # list[str]
result.state_names  # list[str]
result.action_names  # list[str]
result.n_periods  # int
result.n_subjects  # int
result.available_targets  # list[str] - computable additional targets

# Access raw data for advanced users
result.raw_results  # dict[RegimeName, dict[int, PeriodRegimeSimulationData]]
result.internal_params  # InternalParams
result.V_arr_dict  # dict[int, dict[RegimeName, FloatND]]

# Serialization (requires cloudpickle)
result.to_pickle("path/to/file.pkl")
loaded = SimulationResult.from_pickle("path/to/file.pkl")
```

### Initial States Format

Initial states use a flat dictionary format:

```python
initial_states = {
    "wealth": jnp.array([1.0, 2.0, 3.0]),
    "health": jnp.array([0.5, 0.8, 0.3]),
}
initial_regimes = ["working", "working", "retired"]
```

### Key Attributes

- `model.params_template` - Template for parameter dictionary structure (dict by regime
  name)
- `model.regimes` - Immutable mapping of regime names to user `Regime` objects
- `model.internal_regimes` - Immutable mapping of regime names to processed
  `InternalRegime` objects
- `model.ages` - The AgeGrid defining the lifecycle
- `model.n_periods` - Number of periods in the model (derived from `ages`)
- `model.regime_names_to_ids` - Immutable mapping from regime names to integer indices

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

- Extensive use of typing with custom types in `src/lcm/typing.py`
- Type checking with ty (pixi run ty)
- Use `# ty: ignore[error-code]` for type suppression, never `# type: ignore`
- JAX typing integration via jaxtyping

### Code Standards

- Ruff for linting and formatting (configured in pyproject.toml)
- Google-style docstrings
- All functions require type annotations
- Pre-commit hooks ensure code quality
- Never use `from __future__ import annotations` — this project requires Python 3.14+

### Naming and Docstring Conventions

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
- **Use `@overload` when a function accepts both scalar and array inputs.** When a
  function works with both `ScalarFloat` and `Array`, add overload declarations so the
  type checker can track `(ScalarFloat) -> ScalarFloat` and `(Array) -> Array`
  separately. Concrete subclass methods need their own overloads too (not just the
  abstract base).
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

### Testing Style

- Use plain pytest functions, never test classes (`class TestFoo`)
- Use `@pytest.mark.parametrize` for test variations

### Plotting

- Always use **plotly** for visualizations, never matplotlib. Use `plotly.graph_objects`
  and `plotly.subplots.make_subplots`.

### Key Dependencies

- **jax**: Numerical computation
- **jaxtyping**: Array type annotations
- **pandas**: DataFrame output
- **dags**: Function composition and nested namespace flattening
- **plotly**: Plotting and visualization
