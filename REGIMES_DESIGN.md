# Model Regimes Design Document

## Overview

This document outlines the design for implementing "Model Regimes" in PyLCM, which
enables models to be constructed from multiple modular components with potentially
different state-action spaces.


## Architecture Design

### Core Components

#### 1. Regime Class

```python
@dataclass(frozen=True, kw_only=True)
class Regime:
    """A modular component defining a consistent state-action space and functions.

    Each Regime represents a distinct behavioral environment where the agent
    has a specific set of available states, actions, and functions.
    """
    name: str
    description: str | None = None  # Optional description for documentation
    active: range | None = None  # None means active in all periods (determined at Model creation)
    actions: dict[str, Grid] = field(default_factory=dict)
    states: dict[str, Grid] = field(default_factory=dict)
    functions: dict[str, UserFunction] = field(default_factory=dict)
    regime_transitions: dict[str, Callable] = field(default_factory=dict)

    def to_model(
        self,
        n_periods: int | None = None,
        *,
        description: str | None = None,
        enable_jit: bool = True
    ) -> Model:
        """Create a single-regime Model from this Regime.

        Provides a fluent interface for ergonomic single-regime model creation.
        Uses this regime's description as fallback if no explicit description provided.
        """
        return Model(
            regimes=[self],
            n_periods=n_periods,
            description=description or self.description,
            enable_jit=enable_jit
        )

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

    # Flexible period specification that interacts with Regime.active
    n_periods: int | None = None  # None = derive from regime active ranges

    # Legacy single-regime API (with deprecation warning)
    actions: dict[str, Grid] = field(default_factory=dict)
    states: dict[str, Grid] = field(default_factory=dict)
    functions: dict[str, UserFunction] = field(default_factory=dict)

    # Additional model configuration
    description: str | None = None
    enable_jit: bool = True

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
- Each Regime contains its own `active` range (or None for all periods)
- Model validates consistency across all regimes with sophisticated interaction logic
- Promotes encapsulation and self-contained regimes with optional descriptions

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

#### 5. Backward Compatibility with Deprecation Strategy
- Legacy API: Model with `actions`, `states`, `functions` parameters still works
- Shows deprecation warnings to encourage migration to Regime-based API
- Test suite uses warning suppression to maintain clean legacy model usage during transition period
- Fluent interface Regime.to_model() provides ergonomic single-regime model creation

## Implementation Example

### Work-Retirement Model

```python
# Define work regime
work_regime = Regime(
    name="work",
    description="Working phase where agent earns labor income",
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
    description="Retirement phase with no labor income",
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
# Option 1: Auto-derive n_periods from regime active ranges
model = Model(
    regimes=[work_regime, retirement_regime]  # computed_n_periods automatically derived as 10
)

# Option 2: Explicit n_periods with validation
model = Model(
    regimes=[work_regime, retirement_regime],
    n_periods=10  # Validates alignment with regime active ranges
)

# Usage remains the same
solution = model.solve(params)
simulation = model.simulate(params, initial_states, solution)
```

### Single-Regime Model

```python
# Option 1: Fluent interface (most ergonomic for single regime)
model = Regime(
    name="simple_consumption_saving",
    description="Basic consumption-saving model",
    active=range(10),
    actions={"consumption": LinspaceGrid(start=1, stop=100, n_points=50)},
    states={"wealth": LinspaceGrid(start=1, stop=100, n_points=50)},
    functions={"utility": utility, "next_wealth": next_wealth}
).to_model()  # Uses regime description as model description

# Option 2: Fluent interface with explicit n_periods
model = Regime(
    name="flexible_model",
    # active=None - will be set to range(8) automatically
    actions={"consumption": LinspaceGrid(start=1, stop=100, n_points=50)},
    states={"wealth": LinspaceGrid(start=1, stop=100, n_points=50)},
    functions={"utility": utility, "next_wealth": next_wealth}
).to_model(n_periods=8)

# Option 3: Explicit Model creation
single_regime = Regime(
    name="default",
    active=range(10),
    actions={"consumption": LinspaceGrid(start=1, stop=100, n_points=50)},
    states={"wealth": LinspaceGrid(start=1, stop=100, n_points=50)},
    functions={"utility": utility, "next_wealth": next_wealth}
)
model = Model(regimes=[single_regime])  # computed_n_periods automatically derived as 10

# Option 4: Legacy API (with deprecation warning)
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
1. **Period Resolution**: Sophisticated interaction between `n_periods` and `Regime.active`:
   - If `n_periods=None`: Auto-derive from regime active ranges (requires at least one regime with explicit active range)
   - If `n_periods` specified: Validate alignment with regime active ranges
   - If `regime.active=None`: Set to all periods (range(n_periods)) after period resolution
2. **Period Completeness**: The union of all `regime.active` ranges equals `range(computed_n_periods)`
   - This ensures no gaps, no overlaps, and complete coverage
3. **Overlap Detection**: No two regimes can have overlapping active periods
4. **Transition Consistency**:
   - For each period transition, source regime has transition function to target regime
   - State mapping functions produce states that exist in target regime
5. **DAG Property**: Regime transitions form a valid DAG (no cycles)

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

## Implementation Status

### Phase 1: Core Infrastructure (✅ COMPLETED)
1. **✅ Regime class with description and fluent interface**: Added `description` attribute and `to_model()` method
2. **✅ Flexible n_periods interaction**: Sophisticated logic for period resolution and validation
3. **✅ Model.regimes parameter**: Support for list of Regime objects
4. **✅ Comprehensive validation**: Period completeness, overlap detection, alignment validation
5. **✅ Deprecation warnings**: Clean warnings for legacy API usage
6. **✅ Test suite conversion**: Eliminated deprecation warnings using suppression strategy

### Key Implemented Features

#### Flexible Period Resolution
- **Auto-derivation**: `n_periods=None` derives from regime active ranges
- **Validation**: Explicit `n_periods` validates alignment with regime active ranges
- **None handling**: `regime.active=None` automatically set to all periods
- **Error handling**: Comprehensive validation with descriptive error messages

#### Fluent Interface
- **Single-regime ergonomics**: `Regime(...).to_model()` for simple model creation
- **Description inheritance**: Uses regime description as fallback for model description
- **Flexible parameters**: Support for `n_periods`, `description`, and `enable_jit` overrides

#### Test Suite Strategy
- **Regime-based test models**: All test models now defined as Regime objects with descriptions
- **Warning suppression**: Clean handling of legacy API usage during transition period
- **Hybrid conversion**: `get_model()` function converts Regime to legacy Model for compatibility

## Migration Strategy

### Phase 1: Infrastructure Complete (✅ DONE)
1. **✅ Add new Regime class** with description and fluent interface
2. **✅ Add regimes parameter** to Model class with flexible n_periods interaction
3. **✅ Implement regime processing pipeline** with comprehensive validation
4. **✅ Add deprecation warnings** for legacy API usage
5. **✅ Convert test suite** to use Regime-based definitions
6. **✅ Maintain 100% backward compatibility** - all existing code continues working

### Phase 2: Encourage Migration (CURRENT)
1. **Update documentation** to showcase new Regime API and fluent interface
2. **Add migration guide** with examples for both multi-regime and single-regime models
3. **Update tutorials** to demonstrate regime-based model creation
4. **Performance comparisons** between equivalent single-regime and multi-regime models

### Phase 3: Complete Implementation (FUTURE)
1. **Implement regime model solving** (currently raises NotImplementedError)
2. **Add stochastic regime transitions** for choice-dependent regime switching
3. **Concurrent regime support** with multiple value functions
4. **Advanced regime transition logic** with probabilistic switching

### Phase 4: Remove Legacy API (Next Major Version)
1. **Remove legacy parameters** from Model class
2. **Clean up legacy single-regime references**
3. **Remove warning suppression** from test suite
4. **Clean up internal code** structure

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

### Implemented Testing Approach
- **✅ Unit Tests**: Individual Regime validation and processing in `tests/test_regimes.py`
- **✅ Comprehensive validation tests**: Period resolution, overlap detection, alignment validation
- **✅ Fluent interface tests**: `Regime.to_model()` functionality and parameter handling
- **✅ Description inheritance tests**: Verify regime descriptions propagate to models
- **✅ Backward Compatibility Tests**: Legacy API continues working with deprecation warnings
- **✅ Test suite conversion**: All test models converted to Regime-based definitions with descriptions
- **✅ Warning suppression strategy**: Clean test execution during transition period

### Test Model Architecture
```python
# tests/test_models/deterministic.py - Only Regime definitions
ISKHAKOV_ET_AL_2017 = Regime(
    name="iskhakov_et_al_2017",
    description="Corresponds to the example model in Iskhakov et al. (2017)...",
    # ... functions, actions, states
)

# tests/test_models/get_model.py - Converts Regime to Model
def get_model(regime_name: str, n_periods: int) -> Model:
    regime = deepcopy(TEST_REGIMES[regime_name])
    # Hybrid approach: extract regime parameters for legacy Model creation
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return Model(
            description=regime.description,
            n_periods=n_periods,
            functions=regime.functions,
            actions=regime.actions,
            states=regime.states
        )
```

### Future Testing Needs
- **Integration Tests**: Complete work-retirement model solving and simulation (blocked by NotImplementedError)
- **Performance Tests**: Compare solution times vs. equivalent single-regime models
- **Migration Tests**: Test conversion from legacy to regime-based models
- **Regime transition tests**: DAG validation and state mapping consistency

---

*This design document focuses on deterministic regime transitions as specified in the project scope. Phase 2 extensions for stochastic transitions will be addressed in future design documents.*
