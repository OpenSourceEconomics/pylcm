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

```
grids/
├── categorical.py     @categorical decorator and validate_category_class
├── continuous.py      Grid ABC, ContinuousGrid, LinSpacedGrid, LogSpacedGrid,
│                      IrregSpacedGrid, UniformContinuousGrid
├── discrete.py        DiscreteGrid (categorical variables)
├── piecewise.py       Piece, PiecewiseLinSpacedGrid, PiecewiseLogSpacedGrid
└── helpers.py         Coordinate computation: get_linspace_coordinate, etc.
```

## `regime_building/`

Compiles each user `Regime` into an `InternalRegime` with pre-assembled JAX closures for
solving and simulating. Called once during `Model.__init__()`.

```
regime_building/
├── processing.py      Orchestrator: process_regimes() iterates regimes,
│                      delegates to the modules below
├── Q_and_F.py         Composes utility + continuation value + feasibility
├── next_state.py      Assembles state transition functions via dags
├── max_Q_over_a.py    Wraps Q-and-F with action optimization (max / argmax)
├── V.py               Value function interpolation (StateSpaceInfo,
│                      get_V_interpolator)
├── variable_info.py   Classifies variables and extracts grids from a regime
├── argmax.py          Masked argmax_and_max for action optimization
└── ndimage.py         Linear interpolation via map_coordinates
```

## `params/`

Parameter tree primitives and the parameter processing pipeline.

```
params/
├── __init__.py          MappingLeaf, SequenceLeaf, as_leaf, process_params
├── mapping_leaf.py      Dict-valued parameter node
├── sequence_leaf.py     List-valued parameter node
├── regime_template.py   Discovers parameters from a regime's function signatures
└── processing.py        Template creation, 3-level broadcasting, runtime conversion
```

## `solution/`

```
solution/
└── solve_brute.py     Backward induction via grid search — loops over periods
                       calling pre-compiled max_Q_over_a closures
```

## `simulation/`

```
simulation/
├── simulate.py        Forward simulation — loops over periods applying optimal
│                      actions and state transitions for each subject
├── result.py          SimulationResult with deferred .to_dataframe() conversion
├── transitions.py     State and regime advancement between periods
├── validation.py      Validates initial conditions before simulation
└── random.py          JAX random key generation for stochastic transitions
```

## `shocks/`

```
shocks/
├── _base.py           _ShockGrid ABC with Gauss-Hermite quadrature
├── iid.py             Uniform, Normal, LogNormal, NormalMixture
└── ar1.py             Tauchen, Rouwenhorst, TauchenNormalMixture
```

## `utils/`

Cross-cutting internal helpers used by multiple packages. Not part of the public API.

```
utils/
├── containers.py          Immutability wrappers (MappingProxyType), dataclass helpers
├── namespace.py           Regime namespace flattening/unflattening
├── functools.py           Function signature manipulation (all_as_kwargs, etc.)
├── dispatchers.py         JAX vectorization: productmap, vmap_1d, simulation_spacemap
├── state_action_space.py  Creates StateActionSpace from grids and variable info
├── error_handling.py      Validation of value functions and regime transitions
└── logging.py             Progress logging: period timing, value function stats
```
