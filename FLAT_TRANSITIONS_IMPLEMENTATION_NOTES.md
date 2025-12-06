# Flat Transitions Implementation Notes

## Goal
Change user-facing transitions from nested format to flat format:
- **Old (nested):** `{"regime1": {"next_wealth": fn}, "regime2": {"next_wealth": fn}, "next_regime": fn}`
- **New (flat):** `{"next_wealth": fn, "next_regime": fn}`

Same transition function applies to all target regimes.

## Completed Work

### 1. Tests for conversion function (DONE)
File: `tests/input_processing/test_regime_processing.py`
- Added 5 tests for `convert_flat_to_nested_transitions()`:
  - `test_convert_flat_to_nested_single_regime`
  - `test_convert_flat_to_nested_multi_regime`
  - `test_convert_flat_to_nested_with_next_regime`
  - `test_convert_flat_to_nested_only_next_regime`
  - `test_convert_flat_to_nested_empty_transitions`

### 2. Conversion function (DONE)
File: `src/lcm/input_processing/regime_processing.py` (at bottom)
```python
def convert_flat_to_nested_transitions(
    flat_transitions: dict[str, UserFunction],
    regime_names: list[str],
) -> dict[str, dict[str, UserFunction] | UserFunction]:
    # Separates next_regime from state transitions
    # Replicates state transitions for each regime
    # Returns nested format with next_regime at top level
```

### 3. Updated Regime validation (DONE)
File: `src/lcm/regime.py`
- Changed `transitions` type annotation from nested to flat: `dict[str, UserFunction]`
- Updated `_validate_logical_consistency()` to work with flat format

### 4. Updated test models to flat format (DONE)
Files updated:
- `tests/test_models/deterministic.py` - ISKHAKOV_ET_AL_2017 and ISKHAKOV_ET_AL_2017_STRIPPED_DOWN
- `tests/test_models/discrete_deterministic.py` - ISKHAKOV_ET_AL_2017_DISCRETE
- `tests/test_models/stochastic.py` - ISKHAKOV_ET_AL_2017_STOCHASTIC
- `tests/regime_mock.py` - RegimeMock type annotation
- `tests/test_model.py` - All regime definitions
- `tests/test_multi_regime.py` - work_regime and retirement_regime
- `tests/test_stochastic.py` - regime replacements
- `tests/test_solution_on_toy_model.py` - DETERMINISTIC_REGIME
- `tests/test_error_handling_invalid_vf.py` - valid_regime fixture
- `tests/input_processing/test_regime_processing.py` - regime fixture
- `tests/input_processing/test_create_params_template.py` - all RegimeMock usages

### 5. Updated examples to flat format (DONE)
- `examples/consumption_saving/model.py`
- `examples/mahler_yum_2024/model.py`

### 6. Integration in process_regimes (PARTIAL)
File: `src/lcm/input_processing/regime_processing.py`
- Added `nested_transitions` dict creation after single-regime handling
- Started using `regime_with_nested_transitions = regime.replace(transitions=nested_transitions[regime.name])`

## Current Problem
When calling `regime.replace(transitions=nested_transitions)`, the `Regime.__post_init__` validation runs and fails because validation now expects flat format but we're passing nested format.

## Solution Options

### Option A: Skip validation for internal use (Recommended)
Create a helper that creates a regime-like object without validation for internal processing:
```python
@dataclass
class _InternalRegime:
    """Internal regime without validation for processing."""
    # Same fields as Regime but no __post_init__
```

### Option B: Make validation accept both formats
Update `_validate_logical_consistency` to detect and accept both flat and nested formats.

### Option C: Don't modify regime, pass nested_transitions separately
Modify `_get_internal_functions`, `create_params_template`, etc. to accept `nested_transitions` as a separate parameter instead of using `regime.transitions`.

## Files That Need Further Changes

### Core processing that uses `regime.transitions`:
1. `src/lcm/input_processing/regime_processing.py:_get_internal_functions()` - Uses `regime.transitions`
2. `src/lcm/input_processing/create_params_template.py` - Iterates over `regime.transitions`
3. `src/lcm/input_processing/util.py:get_variable_info()` - Uses `flatten_regime_namespace(regime.transitions)`

### Key insight:
The internal code expects nested format `{regime_name: {next_state: fn}}` but user provides flat `{next_state: fn}`.
The conversion happens early in `process_regimes`, but we need a way to pass nested transitions without triggering Regime validation.

## Params Template Changes
With flat transitions, params keys change from:
- `"regime_name__next_wealth": {...}` → `"next_wealth": {...}`

Tests and examples already updated for this.

## Test Command
```bash
pixi run tests
```

## Branch
`deterministic-regime-transitions` (or create new branch for this work)
