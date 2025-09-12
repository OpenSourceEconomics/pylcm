# Model Refactor Implementation Plan

## Overview

This plan details moving the logic from `get_lcm_function()` into the `Model` class `__post_init__` method. The goal is to create a cleaner user interface where Model objects contain all pre-computed functions needed for solving and simulation.

## Current Architecture Analysis

### Key Components to Move

From `src/lcm/entry_point.py:84-191`, the `get_lcm_function()` contains:

1. **Model Processing** (lines 90-91):
   ```python
   internal_model = process_model(model)
   last_period = internal_model.n_periods - 1
   ```

2. **State-Action Space Creation** (lines 98-146):
   ```python
   state_action_spaces: dict[int, StateActionSpace] = {}
   state_space_infos: dict[int, StateSpaceInfo] = {}
   max_Q_over_a_functions: dict[int, MaxQOverAFunction] = {}
   argmax_and_max_Q_over_a_functions: dict[int, ArgmaxQOverAFunction] = {}
   ```

3. **Period Loop Logic** (lines 103-145):
   - Creates state action spaces per period
   - Builds Q and F functions
   - Creates max_Q_over_a and argmax functions
   - Applies JAX JIT compilation

4. **Function Partials** (lines 150-162):
   ```python
   solve_model = partial(solve, state_action_spaces=..., max_Q_over_a_functions=..., logger=...)
   simulate_model = partial(simulate, argmax_and_max_Q_over_a_functions=..., model=..., logger=...)
   ```

## Implementation Plan

### Phase 1: Extend Model Class

#### 1.1 Add New Attributes to Model Class (`src/lcm/user_model.py`)

```python
@dataclass(frozen=True)
class Model:
    # Existing attributes...
    description: str | None = None
    _: KW_ONLY
    n_periods: int
    functions: dict[str, UserFunction] = field(default_factory=dict)
    actions: dict[str, Grid] = field(default_factory=dict)
    states: dict[str, Grid] = field(default_factory=dict)
    
    # NEW: Add computed attributes (set in __post_init__)
    internal_model: InternalModel = field(init=False)
    params_template: ParamsDict = field(init=False)
    state_action_spaces: dict[int, StateActionSpace] = field(init=False)
    state_space_infos: dict[int, StateSpaceInfo] = field(init=False)
    max_Q_over_a_functions: dict[int, MaxQOverAFunction] = field(init=False)
    argmax_and_max_Q_over_a_functions: dict[int, ArgmaxQOverAFunction] = field(init=False)
```

#### 1.2 Add Required Imports to user_model.py

```python
from __future__ import annotations

import dataclasses
import jax
from dataclasses import KW_ONLY, dataclass, field
from typing import TYPE_CHECKING, Any

from lcm.exceptions import ModelInitilizationError, format_messages
from lcm.grids import Grid
from lcm.input_processing import process_model
from lcm.interfaces import StateActionSpace, StateSpaceInfo
from lcm.max_Q_over_a import get_argmax_and_max_Q_over_a, get_max_Q_over_a
from lcm.Q_and_F import get_Q_and_F
from lcm.state_action_space import create_state_action_space, create_state_space_info

if TYPE_CHECKING:
    from lcm.interfaces import InternalModel
    from lcm.typing import (
        ArgmaxQOverAFunction,
        MaxQOverAFunction,
        ParamsDict,
        UserFunction,
    )
```

#### 1.3 Implement New __post_init__ Method

```python
def __post_init__(self) -> None:
    _validate_attribute_types(self)
    _validate_logical_consistency(self)
    
    # Process model to internal representation
    internal_model = process_model(self)
    object.__setattr__(self, 'internal_model', internal_model)
    object.__setattr__(self, 'params_template', internal_model.params)
    
    # Initialize containers
    last_period = internal_model.n_periods - 1
    state_action_spaces: dict[int, StateActionSpace] = {}
    state_space_infos: dict[int, StateSpaceInfo] = {}
    max_Q_over_a_functions: dict[int, MaxQOverAFunction] = {}
    argmax_and_max_Q_over_a_functions: dict[int, ArgmaxQOverAFunction] = {}
    
    # Create functions for each period (reversed order like get_lcm_function)
    for period in reversed(range(internal_model.n_periods)):
        is_last_period = period == last_period
        
        # Create state action space
        state_action_space = create_state_action_space(
            model=internal_model,
            is_last_period=is_last_period,
        )
        
        # Create state space info  
        state_space_info = create_state_space_info(
            model=internal_model,
            is_last_period=is_last_period,
        )
        
        # Determine next state space info
        if is_last_period:
            next_state_space_info = _get_last_periods_next_state_space_info()
        else:
            next_state_space_info = state_space_infos[period + 1]
        
        # Create Q and F functions
        Q_and_F = get_Q_and_F(
            model=internal_model,
            next_state_space_info=next_state_space_info,
            period=period,
        )
        
        # Create optimization functions
        max_Q_over_a = get_max_Q_over_a(
            Q_and_F=Q_and_F,
            actions_names=tuple(state_action_space.continuous_actions)
            + tuple(state_action_space.discrete_actions),
            states_names=tuple(state_action_space.states),
        )
        
        argmax_and_max_Q_over_a = get_argmax_and_max_Q_over_a(
            Q_and_F=Q_and_F,
            actions_names=tuple(state_action_space.discrete_actions)
            + tuple(state_action_space.continuous_actions),
        )
        
        # Store results
        state_action_spaces[period] = state_action_space
        state_space_infos[period] = state_space_info
        max_Q_over_a_functions[period] = jax.jit(max_Q_over_a)  # Default JIT enabled
        argmax_and_max_Q_over_a_functions[period] = jax.jit(argmax_and_max_Q_over_a)
    
    # Set computed attributes using object.__setattr__ (frozen dataclass)
    object.__setattr__(self, 'state_action_spaces', state_action_spaces)
    object.__setattr__(self, 'state_space_infos', state_space_infos) 
    object.__setattr__(self, 'max_Q_over_a_functions', max_Q_over_a_functions)
    object.__setattr__(self, 'argmax_and_max_Q_over_a_functions', argmax_and_max_Q_over_a_functions)

def _get_last_periods_next_state_space_info() -> StateSpaceInfo:
    """Helper function moved from entry_point.py"""
    return StateSpaceInfo(
        states_names=(),
        discrete_states={},
        continuous_states={},
    )
```

### Phase 2: Add Convenience Methods to Model

#### 2.1 Add solve() Method

```python
def solve(
    self,
    params: ParamsDict,
    *,
    debug_mode: bool = True,
) -> dict[int, FloatND]:
    """Solve the model using the pre-computed functions.
    
    Args:
        params: Model parameters matching the template from self.params_template
        debug_mode: Whether to enable debug logging
    
    Returns:
        Dictionary mapping period to value function arrays
    """
    from lcm.logging import get_logger
    from lcm.solution.solve_brute import solve
    
    logger = get_logger(debug_mode=debug_mode)
    
    return solve(
        params=params,
        state_action_spaces=self.state_action_spaces,
        max_Q_over_a_functions=self.max_Q_over_a_functions,
        logger=logger,
    )
```

#### 2.2 Add simulate() Method

```python  
def simulate(
    self,
    params: ParamsDict,
    initial_states: dict[str, Array],
    V_arr_dict: dict[int, FloatND],
    *,
    additional_targets: list[str] | None = None,
    seed: int | None = None,
    debug_mode: bool = True,
) -> pd.DataFrame:
    """Simulate the model forward using pre-computed functions.
    
    Args:
        params: Model parameters 
        initial_states: Initial state values
        V_arr_dict: Value function arrays from solve()
        additional_targets: Additional targets to compute
        seed: Random seed
        debug_mode: Whether to enable debug logging
    
    Returns:
        Simulation results as DataFrame
    """
    from lcm.logging import get_logger
    from lcm.simulation.simulate import simulate
    
    logger = get_logger(debug_mode=debug_mode)
    
    return simulate(
        params=params,
        initial_states=initial_states,
        argmax_and_max_Q_over_a_functions=self.argmax_and_max_Q_over_a_functions,
        model=self.internal_model,
        logger=logger,
        V_arr_dict=V_arr_dict,
        additional_targets=additional_targets,
        seed=seed,
    )
```

#### 2.3 Add solve_and_simulate() Method

```python
def solve_and_simulate(
    self,
    params: ParamsDict,
    initial_states: dict[str, Array],
    *,
    additional_targets: list[str] | None = None,
    seed: int | None = None,
    debug_mode: bool = True,
) -> pd.DataFrame:
    """Solve and then simulate the model in one call.
    
    Args:
        params: Model parameters
        initial_states: Initial state values  
        additional_targets: Additional targets to compute
        seed: Random seed
        debug_mode: Whether to enable debug logging
    
    Returns:
        Simulation results as DataFrame
    """
    V_arr_dict = self.solve(params, debug_mode=debug_mode)
    return self.simulate(
        params=params,
        initial_states=initial_states,
        V_arr_dict=V_arr_dict,
        additional_targets=additional_targets,
        seed=seed,
        debug_mode=debug_mode,
    )
```

### Phase 3: Maintain Backward Compatibility

#### 3.1 Update get_lcm_function() for Backward Compatibility

Keep `get_lcm_function()` but simplify it to use the new Model methods:

```python
def get_lcm_function(
    model: Model,
    *,
    targets: Literal["solve", "simulate", "solve_and_simulate"],
    debug_mode: bool = True,
    jit: bool = True,  # Ignored - always JIT now
) -> tuple[Callable[..., dict[int, Array] | pd.DataFrame], ParamsDict]:
    """Entry point for users to get high level functions generated by lcm.
    
    NOTE: This function is deprecated. Use Model.solve(), Model.simulate(), 
    or Model.solve_and_simulate() methods directly instead.
    """
    import warnings
    warnings.warn(
        "get_lcm_function() is deprecated. Use Model.solve(), Model.simulate(), "
        "or Model.solve_and_simulate() methods instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    if targets == "solve":
        return partial(model.solve, debug_mode=debug_mode), model.params_template
    elif targets == "simulate":  
        return partial(model.simulate, debug_mode=debug_mode), model.params_template
    elif targets == "solve_and_simulate":
        return partial(model.solve_and_simulate, debug_mode=debug_mode), model.params_template
    else:
        raise NotImplementedError(f"Target {targets} not supported")
```

### Phase 4: Update Tests

#### 4.1 Test Files Requiring Updates

Based on grep analysis, these test files import `get_lcm_function`:
- `tests/test_stochastic.py`
- `tests/test_regression_test.py` 
- `tests/test_error_handling.py`
- `tests/test_entry_point.py`
- `tests/test_analytical_solution.py`
- `tests/simulation/test_simulate.py`
- `tests/test_solution_on_toy_model.py`

#### 4.2 New Test File: tests/test_model_methods.py

```python
"""Test new Model solve/simulate methods."""

from __future__ import annotations

import pytest
from pybaum import tree_map

from tests.test_models import get_model_config, get_params


def test_model_solve_method():
    """Test Model.solve() method works correctly."""
    model = get_model_config("iskhakov_et_al_2017_stripped_down", n_periods=3)
    params = tree_map(lambda _: 0.2, model.params_template)
    
    # Test solve method
    solution = model.solve(params)
    
    assert isinstance(solution, dict)
    assert len(solution) == 3
    assert all(period in solution for period in range(3))


def test_model_simulate_method():
    """Test Model.simulate() method works correctly.""" 
    model = get_model_config("iskhakov_et_al_2017_stripped_down", n_periods=3)
    params = tree_map(lambda _: 0.2, model.params_template)
    
    # Solve first
    solution = model.solve(params)
    
    # Create initial states
    initial_states = {
        "wealth": jnp.array([10.0, 20.0]),
        "lagged_retirement": jnp.array([0, 0]),
    }
    
    # Test simulate method
    results = model.simulate(
        params=params,
        initial_states=initial_states, 
        V_arr_dict=solution,
    )
    
    assert isinstance(results, pd.DataFrame)
    assert len(results) > 0


def test_model_solve_and_simulate_method():
    """Test Model.solve_and_simulate() method works correctly."""
    model = get_model_config("iskhakov_et_al_2017_stripped_down", n_periods=3)
    params = tree_map(lambda _: 0.2, model.params_template)
    
    initial_states = {
        "wealth": jnp.array([10.0, 20.0]),
        "lagged_retirement": jnp.array([0, 0]),
    }
    
    # Test combined method
    results = model.solve_and_simulate(
        params=params,
        initial_states=initial_states,
    )
    
    assert isinstance(results, pd.DataFrame)
    assert len(results) > 0


def test_model_initialization_timing():
    """Test that Model initialization completes without errors."""
    import time
    
    start_time = time.time()
    model = get_model_config("iskhakov_et_al_2017_stripped_down", n_periods=5)
    init_time = time.time() - start_time
    
    # Should complete initialization in reasonable time
    assert init_time < 30  # 30 seconds should be more than enough
    
    # Check all required attributes are present
    assert hasattr(model, 'internal_model')
    assert hasattr(model, 'params_template')
    assert hasattr(model, 'state_action_spaces')
    assert hasattr(model, 'max_Q_over_a_functions')
    assert hasattr(model, 'argmax_and_max_Q_over_a_functions')


def test_model_params_template_matches_internal():
    """Test that params_template matches internal_model.params."""
    model = get_model_config("iskhakov_et_al_2017_stripped_down", n_periods=3)
    
    assert model.params_template == model.internal_model.params


@pytest.mark.parametrize("model_name", [
    "iskhakov_et_al_2017",
    "iskhakov_et_al_2017_stripped_down", 
    "iskhakov_et_al_2017_discrete",
    "iskhakov_et_al_2017_stochastic",
])
def test_model_initialization_all_configs(model_name):
    """Test Model initialization works for all test configurations."""
    model = get_model_config(model_name, n_periods=2)
    
    # Should complete without error
    assert model.internal_model is not None
    assert len(model.state_action_spaces) == 2
    assert len(model.max_Q_over_a_functions) == 2
```

#### 4.3 Update Existing Tests

For each test file that uses `get_lcm_function`, add alternative tests using new methods:

```python
# In tests/test_entry_point.py - ADD these tests alongside existing ones

def test_model_solve_method_equivalent_to_get_lcm_function():
    """Test new Model.solve() gives same results as get_lcm_function."""
    model = get_model_config("iskhakov_et_al_2017_stripped_down", n_periods=3)
    params = tree_map(lambda _: 0.2, model.params_template)
    
    # Old approach
    solve_old, _ = get_lcm_function(model=model, targets="solve")
    solution_old = solve_old(params)
    
    # New approach  
    solution_new = model.solve(params)
    
    # Should give identical results
    assert tree_equal(solution_old, solution_new)
```

### Phase 5: Documentation Updates

#### 5.1 Update CLAUDE.md

Add section about new Model methods:

```markdown
## Model Methods

The `Model` class now includes built-in methods for solving and simulation:

### Solving
- `model.solve(params)` - Solve the model and return value function arrays
- Access pre-computed components via `model.state_action_spaces`, `model.max_Q_over_a_functions`

### Simulation  
- `model.simulate(params, initial_states, V_arr_dict)` - Simulate forward given solution
- `model.solve_and_simulate(params, initial_states)` - Combined solve and simulate

### Parameters
- `model.params_template` - Template for parameter dictionary structure
- `model.internal_model` - Processed internal model representation

### Backward Compatibility
- `get_lcm_function()` still works but is deprecated
- Prefer using Model methods directly for cleaner code
```

#### 5.2 Update Docstrings

Add comprehensive docstrings to all new Model methods with examples.

### Phase 6: Error Handling and Edge Cases

#### 6.1 Handle Initialization Errors

Add proper error handling in `__post_init__`:

```python
def __post_init__(self) -> None:
    try:
        _validate_attribute_types(self)
        _validate_logical_consistency(self)
        
        # ... existing initialization code ...
        
    except Exception as e:
        raise ModelInitilizationError(
            f"Failed to initialize Model. Error during function compilation: {e}"
        ) from e
```

#### 6.2 JIT Configuration

Add option to control JIT compilation:

```python
@dataclass(frozen=True) 
class Model:
    # ... existing attributes ...
    
    # New optional parameter
    enable_jit: bool = True
    
    def __post_init__(self) -> None:
        # ... in the period loop ...
        max_Q_over_a_functions[period] = (
            jax.jit(max_Q_over_a) if self.enable_jit else max_Q_over_a
        )
```

## Migration Strategy

### Step 1: Implement Core Changes
1. Add new attributes to Model class
2. Implement new `__post_init__` method
3. Add solve/simulate methods

### Step 2: Add Tests
1. Create `test_model_methods.py`
2. Add comparative tests in existing files
3. Ensure all test configs work

### Step 3: Maintain Compatibility
1. Update `get_lcm_function()` with deprecation warning
2. Ensure all existing tests still pass
3. Update documentation

### Step 4: Performance Validation
1. Compare initialization time vs runtime savings
2. Memory usage analysis
3. Benchmark against existing approach

### Step 5: Release Planning
1. Add feature to changelog
2. Update examples to use new methods
3. Consider deprecation timeline for `get_lcm_function()`

## Expected Benefits

1. **Cleaner User Interface**: `model.solve(params)` vs `get_lcm_function` complexity
2. **Early Error Detection**: Model validation happens at initialization
3. **Reusability**: Model objects can be reused with different parameters
4. **Performance**: Pre-compiled functions ready for multiple solve calls
5. **Introspection**: Users can inspect `model.state_action_spaces` etc.

## Risk Mitigation

1. **Memory Usage**: Monitor memory consumption during initialization
2. **Initialization Time**: Add progress indicators for large models
3. **Backward Compatibility**: Maintain `get_lcm_function` until next major version
4. **Testing**: Comprehensive test coverage including edge cases
5. **Documentation**: Clear migration guide for existing users