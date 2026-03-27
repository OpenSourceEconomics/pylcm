---
title: Package Structure
---

# Package Structure

Overview of how `src/lcm/` is organized and how the pieces fit together.

## Top level

```
src/lcm/
├── model.py           Model class — solve() and simulate() entry points
├── regime.py          Regime dataclass and MarkovTransition wrapper
├── ages.py            AgeGrid for lifecycle period management
├── exceptions.py      Custom exception hierarchy
├── interfaces.py      Internal data structures: InternalRegime, StateActionSpace,
│                      SolveFunctions, SimulateFunctions
├── typing.py          Type aliases and protocol definitions
├── model_processing.py  Model initialization: validation, template, fixed params
├── state_action_space.py  Creates StateActionSpace from grids and variable info
├── pandas_utils.py    params_from_pandas, initial_conditions_from_dataframe
├── persistence.py     Save/load solve and simulate snapshots
├── grids/             Grid definitions (see below)
├── regime_building/   Regime compilation pipeline (see below)
├── params/            Parameter tree and processing (see below)
├── solution/          Backward induction solver
├── simulation/        Forward simulation
├── shocks/            Shock grid specifications
└── utils/             Cross-cutting internal helpers
```

The top-level modules are either user-facing (`model`, `regime`, `ages`, `pandas_utils`,
`persistence`) or core type definitions that every sub-package imports (`interfaces`,
`typing`, `exceptions`).

See [Defining Models](../user_guide/defining_models.md) and
[Solving and Simulating](../user_guide/solving_and_simulating.md) for the user-facing
API built on these modules.

## How a model runs

```
Model.__init__()
│
├─ regime_building/processing.py    Compiles each Regime → InternalRegime
│  ├─ Q_and_F.py                    Assembles Q-function closures
│  ├─ next_state.py                 Assembles state transition functions
│  ├─ max_Q_over_a.py               Wraps Q with action optimization
│  └─ V.py                          Builds value function interpolators
│
├─ params/processing.py             Creates and validates parameter templates
│
model.solve()  →  solution/solve_brute.py    Backward induction over periods
model.simulate()  →  simulation/simulate.py  Forward simulation over periods
```

## `grids/`

Defines the outcome spaces for state and action variables. Each grid type specifies what
values a variable can take — discrete categories, linearly spaced points, log-spaced
points, or piecewise combinations. Grid objects are pure definitions; they do not
contain state transition logic.

```
grids/
├── base.py            Grid ABC (base class for all grids)
├── categorical.py     @categorical decorator and validate_category_class
├── continuous.py      ContinuousGrid, LinSpacedGrid, LogSpacedGrid,
│                      IrregSpacedGrid, UniformContinuousGrid
├── discrete.py        DiscreteGrid (categorical variables)
├── piecewise.py       Piece, PiecewiseLinSpacedGrid, PiecewiseLogSpacedGrid
└── coordinates.py     Coordinate computation: get_linspace_coordinate, etc.
```

See [Grids](../user_guide/grids.md) for usage and
[Interpolation](../explanations/interpolation.ipynb) for how continuous grid coordinates
work internally.

## `regime_building/`

Compiles each user `Regime` into an `InternalRegime` with pre-assembled JAX closures for
solving and simulating. Called once during `Model.__init__()`. This is where
user-defined functions (utility, transitions, constraints) get composed into the Bellman
equation components via `dags`.

```
regime_building/
├── processing.py      Orchestrator: process_regimes() iterates regimes,
│                      delegates to the modules below
├── Q_and_F.py         Composes utility + continuation value + feasibility
├── next_state.py      Assembles state transition functions via dags
├── max_Q_over_a.py    Wraps Q-and-F with action optimization (max / argmax)
├── V.py               Value function interpolation (VInterpolationInfo,
│                      get_V_interpolator)
├── validation.py      Regime input validation and state transition collection
├── variable_info.py   Classifies variables and extracts grids from a regime
├── argmax.py          Masked argmax_and_max for action optimization
└── ndimage.py         Linear interpolation via map_coordinates
```

See [Function Representation](../explanations/function_representation.ipynb) for how
`V.py` turns pre-computed arrays into callable functions, and
[Dispatchers](../explanations/dispatchers.ipynb) for the vectorization patterns used
throughout.

## `params/`

Handles the parameter lifecycle: discovering what parameters a model needs (from
function signatures), creating templates, broadcasting user-supplied values to the
internal structure, and converting `pd.Series` inputs to JAX arrays. `MappingLeaf` and
`SequenceLeaf` are JAX pytree leaves for structured parameter values.

```
params/
├── __init__.py          MappingLeaf, SequenceLeaf, as_leaf, process_params
├── mapping_leaf.py      Dict-valued parameter node
├── sequence_leaf.py     List-valued parameter node
├── regime_template.py   Discovers parameters from a regime's function signatures
└── processing.py        Template creation, 3-level broadcasting, runtime conversion
```

See [Parameters](../user_guide/parameters.md) and
[Working with DataFrames and Series](../user_guide/pandas_interop.md).

## `solution/`

Solves the model by backward induction. In each period, the solver evaluates the
Q-function (utility + discounted continuation value) over the full state-action grid and
takes the max over actions to obtain the value function.

```
solution/
└── solve_brute.py     Backward induction via grid search — loops over periods
                       calling pre-compiled max_Q_over_a closures
```

See [Solving and Simulating](../user_guide/solving_and_simulating.md).

## `simulation/`

Simulates agents forward through time using the solved policy functions. In each period,
the simulator looks up optimal actions, computes next-period states via transition
functions, and handles regime switching.

```
simulation/
├── simulate.py        Forward simulation — loops over periods applying optimal
│                      actions and state transitions for each subject
├── result.py          SimulationResult with deferred .to_dataframe() conversion
├── transitions.py     State and regime advancement between periods
├── initial_conditions.py  Build and validate initial conditions
└── random.py          JAX random key generation for stochastic transitions
```

See [Solving and Simulating](../user_guide/solving_and_simulating.md) and
[Stochastic Transitions](../explanations/stochastic_transitions.ipynb).

## `shocks/`

Shock grids discretize continuous stochastic processes onto finite grids with associated
transition probabilities. IID shocks use quadrature points and weights; AR(1) shocks use
Markov chain approximations (Tauchen, Rouwenhorst).

```
shocks/
├── _base.py           _ShockGrid ABC with Gauss-Hermite quadrature
├── iid.py             Uniform, Normal, LogNormal, NormalMixture
└── ar1.py             Tauchen, Rouwenhorst, TauchenNormalMixture
```

See [Shocks](../user_guide/shocks.md) and
[Approximating Continuous Shocks](../explanations/approximating_continuous_shocks.ipynb).

## `utils/`

Cross-cutting internal helpers used by multiple packages. Not part of the public API.

```
utils/
├── containers.py          Immutability wrappers (MappingProxyType), dataclass helpers
├── namespace.py           Regime namespace flattening/unflattening
├── functools.py           Function signature manipulation (all_as_kwargs, etc.)
├── dispatchers.py         JAX vectorization: productmap, vmap_1d, simulation_spacemap
├── error_handling.py      Validation of value functions and regime transitions
└── logging.py             Progress logging: period timing, value function stats
```
