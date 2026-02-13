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
- `ruff check .` - Linting (automatically runs with pre-commit)
- `ruff format .` - Code formatting (automatically runs with pre-commit)
- `pre-commit run --all-files` - Run all pre-commit hooks

### Environment Setup

- `pixi install` - Install dependencies
- `pixi run explanation-notebooks` - Execute explanation notebooks
- `pre-commit install` - Install pre-commit hooks (after
  `pixi global install pre-commit`)

## Code Architecture

### Core Components

**Model Definition (`src/lcm/model.py`, `src/lcm/regime.py`)**

- `Model`: User-facing class for defining dynamic choice models
- `Regime`: Defines a single regime with utility, constraints, transitions, functions,
  actions, and states
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

- `DiscreteGrid`: Categorical variables with string labels
- `LinSpacedGrid`: Linearly spaced grid (start, stop, n_points)
- `LogSpacedGrid`: Logarithmically spaced grid (start, stop, n_points)
- `IrregSpacedGrid`: Irregularly spaced grid (points tuple)
- `PiecewiseLinSpacedGrid`: Piecewise linearly spaced grid with breakpoints
- `PiecewiseLogSpacedGrid`: Piecewise logarithmically spaced grid with breakpoints
- `AgeGrid`: Lifecycle age grid (start, stop, step or precise_values)
- `@categorical`: Decorator for creating categorical classes with auto-assigned integer
  codes

Grid class hierarchy: `Grid` is the base class. `ContinuousGrid(Grid)` is the base for
continuous grids with `get_coordinate` method. `UniformContinuousGrid(ContinuousGrid)`
is for grids with start/stop/n_points (LinSpacedGrid, LogSpacedGrid inherit from it).
Other continuous grids (IrregSpacedGrid, PiecewiseLinSpacedGrid, PiecewiseLogSpacedGrid)
inherit directly from ContinuousGrid.

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
Regime(
    active=lambda age: 25 <= age < 65,           # Optional: age-based predicate (default: always True)
    constraints={"name": constraint_fn, ...},    # Optional: constraint functions
    transitions={                                # Required for non-terminal regimes
        "next_state1": transition_fn,
        "next_regime": lambda: {"regime_name": 1.0},
    },
    functions={                                  # Must include "utility"; other functions optional
        "utility": utility_function,
        "name": helper_fn, ...
    },
    actions={"action_name": Grid, ...},          # Action grids (can be empty)
    states={"state_name": Grid, ...},            # State grids (can be empty)
    absorbing=False,                             # Optional: absorbing regime flag
    terminal=False,                              # Optional: terminal regime (no transitions)
)
```

**Regime Requirements:**

- `active` is optional; defaults to `lambda _age: True` (always active)
- `functions` must contain a `"utility"` entry (the utility function)
- All transition function names must start with `next_`
- Non-terminal regimes must have transitions for ALL states across ALL regimes
- Non-terminal regimes must include a `next_regime` function returning
  `dict[str, float]`
- Terminal regimes (`terminal=True`) cannot have any transitions
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
- JAX typing integration via jaxtyping

### Code Standards

- Ruff for linting and formatting (configured in pyproject.toml)
- Google-style docstrings
- All functions require type annotations
- Pre-commit hooks ensure code quality

### Key Dependencies

- **jax**: Numerical computation
- **jaxtyping**: Array type annotations
- **pandas**: DataFrame output
- **dags**: Function composition and nested namespace flattening
