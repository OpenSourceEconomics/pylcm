# Model Blocks Design Document

## Overview

This document outlines the design for implementing "Model Blocks" in PyLCM, which enables models to be constructed from multiple modular components with potentially different state-action spaces across periods.

## Motivation

Current PyLCM models have a single, consistent state-action space across all periods. Model Blocks will enable:

1. **Life-cycle transitions**: Models where agents transition between different phases (e.g., work → retirement)
2. **State-dependent structure**: Different available actions based on current state
3. **Modular model construction**: Reusable components for common economic scenarios

## Scope

**Phase 1 (This Design)**: Deterministic block transitions
- Fixed period-based transitions (e.g., mandatory retirement at period 7)
- Single value function that handles block transitions seamlessly

**Phase 2 (Future)**: Stochastic block transitions
- Choice-dependent transitions (e.g., optional retirement starting at period 5)
- Multiple value functions requiring iterative solution

## Architecture Design

### Core Components

#### 1. ModelBlock Class

```python
@dataclass(frozen=True)
class ModelBlock:
    """A modular component defining a consistent state-action space and functions.

    Each ModelBlock represents a distinct "phase" of the economic model where
    the agent has a specific set of available states, actions, and functions.
    """
    name: str
    actions: dict[str, Grid]
    states: dict[str, Grid]
    functions: dict[str, UserFunction]
    block_transitions: dict[str, Callable]  # target_block_name -> state_mapping_function

    # Computed during initialization (similar to current Model)
    internal_model_block: InternalModelBlock = field(init=False)
```

#### 2. Enhanced Model Class

```python
@dataclass(frozen=True)
class Model:
    """A complete model composed of multiple ModelBlocks."""
    blocks: dict[str, ModelBlock]
    n_periods: int
    block_schedule: dict[int, str]  # period -> active_block_name

    # Computed during initialization
    block_transition_dag: dict[str, dict[str, Callable]] = field(init=False)
    next_block_state_function: Callable = field(init=False)
    state_action_spaces: dict[int, StateActionSpace] = field(init=False)
    # ... other computed attributes
```

#### 3. Internal Processing

```python
@dataclass(frozen=True)
class InternalModelBlock:
    """Internal representation of a ModelBlock (similar to InternalModel)."""
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

#### 1. Static ModelBlocks
- Each ModelBlock has no concept of periods internally
- Period handling is managed at the Model level
- Simplifies block design and promotes reusability

#### 2. DAG-Based Transition Resolution
- `block_transitions` dictionary defines possible transitions from each block
- Model.__post_init__() validates transition DAG and creates unified transition function
- Enables complex transition logic while maintaining clarity

#### 3. Independent Internal Processing
- Each ModelBlock creates its own `InternalModelBlock`
- Model combines these into period-based `StateActionSpace` objects
- Maintains separation of concerns

#### 4. Enhanced Solution Algorithm
- New `solve_blocks()` function in `src/lcm/solution/solve_brute_blocks.py`
- Handles state transitions between blocks during backward induction
- Maintains single value function across all blocks

## Implementation Example

### Work-Retirement Model

```python
# Define work block
work_block = ModelBlock(
    name="work",
    actions={
        "consumption": LinspaceGrid(start=1, stop=100, n_points=50),
        "work_hours": LinspaceGrid(start=0, stop=1, n_points=11),
    },
    states={
        "wealth": LinspaceGrid(start=0, stop=500, n_points=100),
        "experience": LinspaceGrid(start=0, stop=40, n_points=41),
    },
    functions={
        "utility": work_utility,
        "next_wealth": next_wealth_work,
        "next_experience": next_experience,
        "labor_income": labor_income,
        "wage": wage_function,
    },
    block_transitions={
        "retirement": lambda wealth, experience: {
            "wealth": wealth,
            "pension": experience * 0.02  # 2% pension per year of experience
        }
    }
)

# Define retirement block
retirement_block = ModelBlock(
    name="retirement",
    actions={
        "consumption": LinspaceGrid(start=1, stop=100, n_points=50),
    },
    states={
        "wealth": LinspaceGrid(start=0, stop=500, n_points=100),
        "pension": LinspaceGrid(start=0, stop=50, n_points=51),
    },
    functions={
        "utility": retirement_utility,
        "next_wealth": next_wealth_retirement,
        "pension_income": pension_income,
    },
    block_transitions={}  # No transitions from retirement (absorbing state)
)

# Define complete model
model = Model(
    blocks={
        "work": work_block,
        "retirement": retirement_block
    },
    n_periods=10,
    block_schedule={
        0: "work", 1: "work", 2: "work", 3: "work",
        4: "work", 5: "work", 6: "work",  # Work periods 0-6
        7: "retirement", 8: "retirement", 9: "retirement"  # Retirement periods 7-9
    }
)

# Usage remains similar
solution = model.solve(params)
simulation = model.simulate(params, initial_states, solution)
```

## Validation Requirements

The Model.__post_init__() method must validate:

1. **Schedule Completeness**: `block_schedule` covers all periods [0, n_periods)
2. **Block Existence**: All blocks referenced in `block_schedule` exist in `blocks`
3. **Transition Consistency**:
   - For each period transition, source block has transition function to target block
   - State mapping functions produce states that exist in target block
4. **DAG Property**: Block transitions form a valid DAG (no cycles)

## Solution Algorithm Changes

### New solve_blocks() Function

```python
def solve_blocks(
    params: ParamsDict,
    block_schedule: dict[int, str],
    block_internal_models: dict[str, InternalModelBlock],
    state_action_spaces: dict[int, StateActionSpace],
    max_Q_over_a_functions: dict[int, MaxQOverAFunction],
    next_block_state_function: Callable,
    logger: logging.Logger,
) -> dict[int, FloatND]:
    """Solve a model with multiple blocks using dynamic programming.

    Key differences from solve():
    - Handles state transitions between blocks
    - Maps states through block transitions during backward induction
    - Single value function maintained across all blocks
    """
```

### Integration Points

- `Model.solve()` method calls `solve_blocks()` instead of `solve()`
- Existing `solve()` function remains for backward compatibility with single-block models
- State transition logic integrated into Q-function computation

## File Structure

```
src/lcm/
├── user_model.py                    # Enhanced Model class, new ModelBlock class
├── interfaces.py                    # New InternalModelBlock interface
├── model_initialization.py         # Enhanced to handle ModelBlocks
├── input_processing/
│   └── model_block_processing.py   # New: Process ModelBlock to InternalModelBlock
├── solution/
│   └── solve_brute_blocks.py       # New: solve_blocks() function
└── block_transitions.py            # New: DAG validation and transition function creation
```

## Future Extensions (Phase 2)

This design enables future stochastic block transitions by:

1. Adding choice-dependent block transitions to ModelBlock
2. Implementing multiple value functions with iterative solution
3. Extending solve_blocks() to handle probabilistic transitions

The current design provides a solid foundation for these extensions while delivering immediate value for deterministic transition models.

## Testing Strategy

- **Unit Tests**: Individual ModelBlock validation and processing
- **Integration Tests**: Complete work-retirement model solving and simulation
- **Regression Tests**: Ensure single-block models continue to work unchanged
- **Performance Tests**: Compare solution times vs. equivalent single-block models

---

*This design document focuses on deterministic block transitions as specified in the project scope. Phase 2 extensions for stochastic transitions will be addressed in future design documents.*
