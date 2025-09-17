# Model Regimes Design Document

## Overview

This document outlines the design for implementing "Model Regimes" in PyLCM, which
enables models to be constructed from multiple modular components with potentially
different state-action spaces.


## Architecture Design

### Core Components

#### 1. Regime Class

```python
@dataclass(frozen=True)
class Regime:
    """A modular component defining a consistent state-action space and functions.

    Each Regime represents a distinct behavioral environment where the agent
    has a specific set of available states, actions, and functions.
    """
    name: str
    active: range
    actions: dict[str, Grid]
    states: dict[str, Grid]
    functions: dict[str, UserFunction]
    regime_transitions: dict[str, Callable] = field(default_factory=dict)

    # Computed during initialization (similar to current Model)
    internal_regime: InternalRegime = field(init=False)
```

#### 2. Enhanced Model Class

```python
@dataclass(frozen=True)
class Model:
    """A complete model composed of multiple Regimes or single-regime (legacy API)."""

    # New regime-based API (preferred)
    regimes: list[Regime] = field(default_factory=list)

    # Legacy single-regime API (with deprecation warning)
    n_periods: int | None = None
    actions: dict[str, Grid] = field(default_factory=dict)
    states: dict[str, Grid] = field(default_factory=dict)
    functions: dict[str, UserFunction] = field(default_factory=dict)

    # Computed during initialization
    computed_n_periods: int = field(init=False)
    regime_transition_dag: dict[str, dict[str, Callable]] = field(init=False)
    next_regime_state_function: Callable = field(init=False)
    state_action_spaces: dict[int, StateActionSpace] = field(init=False)
    is_regime_model: bool = field(init=False)
    # ... other computed attributes
```

#### 3. Internal Processing

```python
@dataclass(frozen=True)
class InternalRegime:
    """Internal representation of a Regime (similar to InternalModel)."""
    name: str
    grids: dict[str, Float1D | Int1D]
    gridspecs: dict[str, Grid]
    variable_info: pd.DataFrame
    functions: dict[str, InternalUserFunction]
    function_info: pd.DataFrame
    params: ParamsDict
    random_utility_shocks: ShockType
```

### Key Design Decisions

#### 1. Regime-Centered Design
- Each Regime contains its own `active` list
- Model validates consistency across all regimes
- Promotes encapsulation and self-contained regimes

#### 2. List-Based Regime Collection
- `regimes: list[Regime]` instead of `dict[str, Regime]`
- Eliminates redundancy since `Regime.name` provides identification
- Simpler API with no risk of key/name mismatches

#### 3. DAG-Based Transition Resolution
- `regime_transitions` dictionary defines possible transitions from each regime
- Model.__post_init__() validates transition DAG and creates unified transition function
- Enables complex transition logic while maintaining clarity

#### 4. Enhanced Solution Algorithm
- New `solve_regimes()` function in `src/lcm/solution/solve_brute_regimes.py`
- Handles state transitions between regimes during backward induction
- For non-concurrent regimes: maintains single value function per period
- For future concurrent regimes: maintains separate value function per regime

#### 5. No Backward Compatibility
- Deprecated: Model with `actions`, `states`, `functions` parameters
- Users must explicitly use Regime class or Model(regimes=[...])
- Raises deprecation error if old API is used

## Implementation Example

### Work-Retirement Model

```python
# Define work regime
work_regime = Regime(
    name="work",
    active=range(0, 7),  # Periods 0-6
    actions={
        "consumption": LinspaceGrid(start=1, stop=100, n_points=50),
        "working": DiscreteGrid(WorkingStatus),
    },
    states={
        "wealth": LinspaceGrid(start=1, stop=100, n_points=50),
    },
    functions={
        "utility": utility,
        "labor_income": labor_income,
        "next_wealth": next_wealth,
        "borrowing_constraint": borrowing_constraint,
    },
    regime_transitions={
        "retirement": work_to_retirement_transition,
    }
)

# Define retirement regime
retirement_regime = Regime(
    name="retirement",
    active=range(7, 10),  # Periods 7-9
    actions={
        "consumption": LinspaceGrid(start=1, stop=100, n_points=50),
    },
    states={
        "wealth": LinspaceGrid(start=1, stop=100, n_points=50),
    },
    functions={
        "utility": utility,
        "working": lambda: 0,  # Always not working
        "labor_income": labor_income,
        "next_wealth": next_wealth,
        "borrowing_constraint": borrowing_constraint,
    },
    regime_transitions={}  # No transitions from retirement (absorbing state)
)

# Define complete model (new regime-based API)
model = Model(
    regimes=[work_regime, retirement_regime]  # computed_n_periods automatically derived as 10
)

# Usage remains the same
solution = model.solve(params)
simulation = model.simulate(params, initial_states, solution)
```

### Single-Regime Model

```python
# Option 1: New regime-based API (preferred)
single_regime = Regime(
    name="default",
    active=range(10),  # Much cleaner than list(range(10))
    actions={"consumption": LinspaceGrid(start=1, stop=100, n_points=50)},
    states={"wealth": LinspaceGrid(start=1, stop=100, n_points=50)},
    functions={"utility": utility, "next_wealth": next_wealth}
)
model = Model(regimes=[single_regime])  # computed_n_periods automatically derived as 10

# Option 2: Legacy API (with deprecation warning)
model = Model(
    n_periods=10,
    actions={"consumption": LinspaceGrid(start=1, stop=100, n_points=50)},
    states={"wealth": LinspaceGrid(start=1, stop=100, n_points=50)},
    functions={"utility": utility, "next_wealth": next_wealth}
)  # Works but shows deprecation warning
```

## Validation Requirements

The Model.__post_init__() method must validate:

**For Regime-based Models:**
1. **Period Completeness**: The union of all `regime.active` ranges equals `range(computed_n_periods)`
   - Where `computed_n_periods = max(regime.active.stop for regime in regimes)`
   - This ensures no gaps, no overlaps, and complete coverage
2. **Transition Consistency**:
   - For each period transition, source regime has transition function to target regime
   - State mapping functions produce states that exist in target regime
3. **DAG Property**: Regime transitions form a valid DAG (no cycles)

**For Legacy Models:**
1. **Deprecation Warning**: Issue warning when using `n_periods`/`actions`/`states`/`functions` API
2. **Standard Validation**: Apply existing single-regime validation logic

## Codebase Changes Required

### 1. Core Class Updates
```
src/lcm/user_model.py:
- Add new Regime class
- Update Model class to accept regimes: list[Regime] parameter
- Auto-derive computed_n_periods from regime.active ranges
- Add deprecation warnings for legacy n_periods/actions/states/functions parameters
- Update validation methods to handle both APIs
```

### 2. Internal Processing Updates
```
src/lcm/interfaces.py:
- Add new InternalRegime interface
- Update type annotations for dual API support

src/lcm/model_initialization.py:
- Update to handle both regime-based and legacy models
- Add regime processing pipeline for new API
- Maintain full backward compatibility for existing models
```

### 3. New Processing Modules
```
src/lcm/input_processing/regime_processing.py:  # New
- Process Regime to InternalRegime
- Handle regime-specific validation
- Manage regime transitions

src/lcm/regime_transitions.py:  # New
- DAG validation and transition function creation
- Period mapping logic
- Regime consistency checks
```

### 4. Solution Algorithm Updates
```
src/lcm/solution/solve_brute_regimes.py:  # New
- solve_regimes() function
- Handle state transitions between regimes
- Maintain single value function

src/lcm/solution/solve_brute.py:
- Keep existing solve() for backward compatibility
- Update Model.solve() to route to appropriate solver
```

### 5. Test Updates
```
tests/test_regimes.py:  # New tests for regime functionality
- Test Regime class creation and validation
- Test range-based `active` periods functionality
- Test Model with regimes parameter
- Test computed_n_periods auto-derivation
- Test DAG validation for regime transitions

tests/test_legacy_compatibility.py:  # New backward compatibility tests
- Test deprecation warnings for legacy API
- Ensure all existing functionality still works
- Test migration from legacy to regime-based API
```

## Migration Strategy

### Phase 1: Add Regime Support (Non-Breaking)
1. **Add new Regime class**
2. **Add regimes parameter** to Model class (optional, defaults to empty list)
3. **Implement regime processing pipeline** for new API
4. **Add deprecation warnings** for legacy API usage
5. **Maintain 100% backward compatibility** - all existing code continues working

### Phase 2: Encourage Migration (Future)
1. **Update documentation** to recommend new Regime API
2. **Add migration guide** with examples
3. **Update tutorials** to use Regime-based examples

### Phase 3: Remove Legacy API (Next Major Version)
1. **Remove legacy parameters** from Model class
2. **Clean up legacy single-regime references**
3. **Clean up internal code** structure

## Future Extensions

**Deterministic Regime Scope (This Implementation):**
- Fixed period-based regime transitions (e.g., mandatory retirement at period 7)
- Single value function per period (for non-concurrent regimes)
- DAG validation for regime transition consistency

**Future Stochastic Extensions:**
1. **Choice-dependent transitions**: Optional retirement starting at period 5
2. **Concurrent regimes**: Multiple regimes active simultaneously
3. **Multiple value functions**: Separate value function per regime for concurrent cases
4. **Probabilistic transitions**: Stochastic regime switching based on agent choices

The current deterministic design provides a solid foundation for these future extensions while delivering immediate value for life-cycle transition models.

## Testing Strategy

- **Unit Tests**: Individual Regime validation and processing
- **Integration Tests**: Complete work-retirement model solving and simulation
- **Backward Compatibility Tests**: Ensure all existing functionality preserved
- **Performance Tests**: Compare solution times vs. equivalent single-regime models
- **Deprecation Tests**: Verify legacy API shows appropriate warnings
- **Migration Tests**: Test conversion from legacy to regime-based models

---

*This design document focuses on deterministic regime transitions as specified in the project scope. Phase 2 extensions for stochastic transitions will be addressed in future design documents.*
