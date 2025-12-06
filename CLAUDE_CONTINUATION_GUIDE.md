# Claude Code Continuation Guide: Flat Transitions Interface

## Quick Start
```bash
cd /Users/timwork/Documents/pylcm
git checkout flat-transitions-interface
cat FLAT_TRANSITIONS_IMPLEMENTATION_NOTES.md
```

## What We're Building
Changing user-facing `transitions` dict from **nested** to **flat** format:

**Before (nested):**
```python
transitions={
    "work": {"next_wealth": fn, "next_health": fn},
    "retirement": {"next_wealth": fn, "next_health": fn},
    "next_regime": next_regime_fn,
}
```

**After (flat):**
```python
transitions={
    "next_wealth": fn,
    "next_health": fn,
    "next_regime": next_regime_fn,
}
```

The same transition function is used for all target regimes (simpler interface).

## Current Status
- ✅ Conversion function `convert_flat_to_nested_transitions()` written and tested
- ✅ `Regime` class validation updated for flat format
- ✅ All test models updated to flat format
- ✅ All examples updated to flat format
- ❌ **BLOCKED:** Integration into `process_regimes()` fails

## The Blocking Problem

In `src/lcm/input_processing/regime_processing.py`, we do:
```python
nested_transitions = {
    regime.name: convert_flat_to_nested_transitions(...)
    for regime in regimes
}

# This line FAILS:
regime_with_nested_transitions = regime.replace(
    transitions=nested_transitions[regime.name]
)
```

**Why it fails:** `Regime.replace()` calls `dataclasses.replace()` which creates a new `Regime`, triggering `__post_init__` validation. The validation now expects **flat** format, but we're passing **nested** format for internal processing.

## Recommended Solution

**Option C: Pass nested_transitions separately** (cleanest approach)

Instead of modifying the regime object, pass `nested_transitions` as a separate parameter to functions that need it:

1. Modify `_get_internal_functions()` signature:
```python
def _get_internal_functions(
    regime: Regime,
    nested_transitions: dict,  # ADD THIS
    grids: dict[RegimeName, dict[str, Array]],
    ...
)
```

2. Inside `_get_internal_functions()`, use `nested_transitions` instead of `regime.transitions` where the nested format is needed.

3. Similarly update `create_params_template()` in `src/lcm/input_processing/create_params_template.py`

## Key Files to Modify

1. **`src/lcm/input_processing/regime_processing.py`**
   - `_get_internal_functions()` - uses `regime.transitions` extensively
   - Lines ~230-290 use `flatten_regime_namespace(regime.transitions)`

2. **`src/lcm/input_processing/create_params_template.py`**
   - Line ~109: `for regime_name, regime_transitions in regime.transitions.items():`

3. **`src/lcm/input_processing/util.py`**
   - `get_variable_info()` at line ~112 uses `flatten_regime_namespace(regime.transitions)`

## Understanding the Code Flow

1. User creates `Regime` with flat transitions
2. `Model.__init__` calls `process_regimes()`
3. `process_regimes()` converts flat → nested via `convert_flat_to_nested_transitions()`
4. Internal processing expects nested format `{regime_name: {next_state: fn}}`
5. `flatten_regime_namespace()` converts `{a: {b: fn}}` → `{a__b: fn}`
6. `unflatten_regime_namespace()` does the reverse

## Helper Functions to Know

- `flatten_regime_namespace(d)` - Flattens nested dict with `__` separator
- `unflatten_regime_namespace(d)` - Unflattens `a__b` keys to nested `{a: {b: ...}}`
- These are in `src/lcm/utils.py` (imported from `dags` library)

## Running Tests
```bash
pixi run tests  # Run all tests
pixi run tests tests/input_processing/test_regime_processing.py  # Run specific file
```

## Files Already Updated (don't change these again)

### Test models (now use flat format):
- `tests/test_models/deterministic.py`
- `tests/test_models/discrete_deterministic.py`
- `tests/test_models/stochastic.py`

### Test files (now use flat format):
- `tests/test_model.py`
- `tests/test_multi_regime.py`
- `tests/test_stochastic.py`
- `tests/test_solution_on_toy_model.py`
- `tests/test_error_handling_invalid_vf.py`
- `tests/input_processing/test_regime_processing.py`
- `tests/input_processing/test_create_params_template.py`
- `tests/regime_mock.py`

### Examples (now use flat format):
- `examples/consumption_saving/model.py`
- `examples/mahler_yum_2024/model.py`

## Params Template Note

With flat transitions, param keys change:
- Old: `"regime_name__next_wealth": {"interest_rate": 0.05}`
- New: `"next_wealth": {"interest_rate": 0.05}`

The test/example updates already account for this.

## After Fixing the Integration

1. Run `pixi run tests` - all 305 tests should pass
2. Run `pixi run mypy` - fix any type errors
3. Run `pre-commit run --all-files` - fix any linting
4. Commit and create PR

## Questions to Ask User If Stuck

1. Should we support both flat AND nested formats for backwards compatibility?
2. For multi-regime models, is it acceptable that all regimes must use the same transition functions?
