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
  `derived_categoricals` merged in, the model-level Koopmans aggregator and certainty
  equivalent injected, completeness validated (a `utility` entry, state-transition
  coverage, state/action overlap, distributed-grid rules). The result is a plain
  `lcm.regime.Regime`, still user vocabulary, so the params template reads the user's
  coarseness off it. Internal signatures mark the post-merge form with
  `FinalizedUserRegime` (defined in `finalize.py`) — an alias the type checker treats as
  plain `Regime`; it enforces nothing and only documents that finalization has happened.
  A bare `Regime` validates only local, value-shape properties at construction —
  completeness may be satisfied only at the model level.
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

**Koopmans Aggregation (`src/lcm/koopmans_aggregation.py`)**

- `LinearAggregator()`: the aggregator `U + β · CE`, and the model-level default. Called
  with `(utility, CE, discount_factor)`.
- `CESAggregator()`: the CES form, called with
  `(utility, CE, discount_factor, intertemporal_elasticity_of_substitution)`; pair with
  `certainty_equivalent=PowerMean()` for the full Epstein-Zin recursion. Neither name
  states a preference class — the form alone is not Epstein-Zin, and
  `LinearAggregator()` is time-additive only alongside `LinearExpectation()`. Both are
  weighted power means over the same kernel (`_lcm.power_mean.weighted_power_mean`) —
  the aggregator averages `(utility, CE)` at weights `(1-β, β)` and exponent `1 - 1/ψ`,
  the certainty equivalent the continuation lottery at exponent `1 - risk_aversion` — so
  a range one survives the other survives too. The naive `**` form of either loses the
  value entirely at small inputs and reverses action rankings near its unit-exponent
  limit.
- Both subclass `KoopmansAggregator`, which is a marker and a place to state the
  contract — it deliberately declares no parameter-name property, because an
  aggregator's signature is the sole declaration of what it consumes. The slot takes any
  callable with the same convention, so a form neither class covers stays a plain
  function.
- A regime's aggregator lives in the `koopmans_aggregator` slot, not in `functions`; its
  parameters beyond `utility` and `CE` surface in the params template under the
  pseudo-function key `koopmans_aggregator`. `Phased(solve=..., simulate=...)` gives the
  two phases different aggregators.

**Value Function Representation (`src/_lcm/regime_building/V.py`)**

- `VInterpolationInfo`: Metadata for working with function outputs on state spaces

**Solution (`src/_lcm/solution/`)**

- `backward_induction.py`: Brute force dynamic programming solver using backward
  induction
- Entry point: `model.solve()` method

**Simulation (`src/_lcm/simulation/`)**

- `simulate.py`: Forward simulation of solved models
- `SimulationResult` (`src/lcm/result.py`): result object with deferred DataFrame
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
