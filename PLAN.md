# Implementation Plan: Flat Simulation Output with SimulationResult Object

## Overview

Change simulation output from `dict[RegimeName, pd.DataFrame]` to a `SimulationResult` object that:
1. Stores raw simulation data
2. Provides a `to_dataframe()` method that returns a single flat DataFrame
3. Supports deferred `additional_targets` computation
4. Caches the DataFrame when no targets are requested

## Design Principles

1. **Separation of concerns**: Simulation logic separate from result processing
2. **Simple first**: Don't optimize prematurely
3. **Flat state space**: Users see one unified state space
4. **Pre-computation**: Do as much as sensible during `SimulationResult` initialization

## Key Behaviors

### Flat DataFrame Structure
- Columns: `period`, `subject_id`, `regime`, `value`, all states, all actions, (optional targets)
- Sorted by `(subject_id, period)` for easy trajectory following
- NaN for states/actions not defined in subject's current regime
- No rows after subject enters terminal regime (handled naturally by simulation)

### Caching Strategy
- Cache DataFrame when `additional_targets=None`
- If targets requested, compute fresh (don't cache targeted results)

---

## Commits

### Commit 1: Add SimulationResult class with to_dataframe() method

**Files:**
- `src/lcm/simulation/result.py` (new)
- `src/lcm/simulation/__init__.py` (export)

**Changes:**
- Create `SimulationResult` dataclass with:
  - Raw results storage (`_raw_results`, `_internal_regimes`, `_params`, etc.)
  - Pre-computed metadata (`regime_names`, `state_names`, `action_names`, `n_periods`, `n_subjects`)
  - `to_dataframe(additional_targets=None)` method
  - Private `_cached_df` for caching
- Create `_create_flat_dataframe()` helper function that processes raw results into flat DataFrame
- Keep target computation logic (move/adapt from `processing.py`)

**Key implementation details:**
- Pre-compute `all_state_names` and `all_action_names` (union across regimes) during init
- Pre-compute `regime_states` and `regime_actions` mappings during init
- The `to_dataframe()` method should be vectorized (not row-by-row)

---

### Commit 2: Update simulate() to return SimulationResult

**Files:**
- `src/lcm/simulation/simulate.py`

**Changes:**
- Change return type from `dict[RegimeName, pd.DataFrame]` to `SimulationResult`
- Remove call to `process_simulated_data()` at the end
- Return `SimulationResult` with raw data instead
- Update docstring

---

### Commit 3: Update Model class methods

**Files:**
- `src/lcm/model.py`

**Changes:**
- Update `simulate()` return type annotation to `SimulationResult`
- Update `solve_and_simulate()` return type annotation to `SimulationResult`
- Update docstrings

---

### Commit 4: Update simulation tests

**Files:**
- `tests/simulation/test_simulate.py`
- `tests/simulation/test_initial_states.py` (if uses simulate)

**Changes:**
- Change `result["regime"]` to `result.to_dataframe()`
- Update assertions to work with flat DataFrame format
- Add `regime` column filtering where needed

---

### Commit 5: Update solution/model tests

**Files:**
- `tests/test_solution_and_simulation.py`
- `tests/test_solution_on_toy_model.py`
- `tests/test_model_methods.py` (if exists)

**Changes:**
- Update to use `.to_dataframe()`
- Adjust assertions for flat format

---

### Commit 6: Update stochastic and error handling tests

**Files:**
- `tests/test_stochastic.py`
- `tests/test_error_handling_invalid_vf.py`

**Changes:**
- Update to use `.to_dataframe()`

---

### Commit 7: Update regression tests and examples

**Files:**
- `tests/test_regression_test.py`
- `examples/consumption_saving/model.py` (if has simulation)
- `examples/mahler_yum_2024/model.py` (if has simulation)

**Changes:**
- Update to use `.to_dataframe()`

---

### Commit 8: Clean up old processing code

**Files:**
- `src/lcm/simulation/processing.py`

**Changes:**
- Remove `process_simulated_data()` function (no longer needed)
- Keep any helper functions that are still used by `result.py`
- Or delete file entirely if empty

---

## Detailed Design: SimulationResult Class

```python
@dataclasses.dataclass
class SimulationResult:
    """Result object from model simulation with deferred DataFrame computation.

    This class stores raw simulation results and provides a method to convert
    them to a flat pandas DataFrame. The DataFrame is cached for efficiency
    when no additional targets are requested.

    Attributes:
        regime_names: Names of all regimes in the model.
        state_names: Names of all state variables (union across regimes).
        action_names: Names of all action variables (union across regimes).
        n_periods: Number of periods in the simulation.
        n_subjects: Number of subjects simulated.

    """

    # Raw data (private)
    _raw_results: dict[str, dict[int, SimulationResults]]
    _internal_regimes: dict[RegimeName, InternalRegime]
    _params: ParamsDict

    # Pre-computed metadata (public, computed during __post_init__)
    regime_names: tuple[str, ...] = field(init=False)
    state_names: tuple[str, ...] = field(init=False)
    action_names: tuple[str, ...] = field(init=False)
    n_periods: int = field(init=False)
    n_subjects: int = field(init=False)

    # Pre-computed mappings (private, for efficient DataFrame creation)
    _regime_to_states: dict[str, tuple[str, ...]] = field(init=False)
    _regime_to_actions: dict[str, tuple[str, ...]] = field(init=False)

    # Cache
    _cached_df: pd.DataFrame | None = field(default=None, repr=False)

    def __post_init__(self):
        # Compute metadata from internal_regimes
        ...

    def to_dataframe(
        self,
        additional_targets: dict[str, list[str]] | None = None,
    ) -> pd.DataFrame:
        """Convert simulation results to a flat pandas DataFrame.

        Args:
            additional_targets: Optional dict mapping regime names to lists of
                target names to compute. Targets can be any function defined
                in the regime (utility, constraints, user functions).

        Returns:
            DataFrame with columns: period, subject_id, regime, value,
            all state variables, all action variables, and any requested targets.
            Sorted by (subject_id, period).

        """
        if additional_targets is None and self._cached_df is not None:
            return self._cached_df

        df = _create_flat_dataframe(
            raw_results=self._raw_results,
            internal_regimes=self._internal_regimes,
            params=self._params,
            state_names=self.state_names,
            action_names=self.action_names,
            regime_to_states=self._regime_to_states,
            regime_to_actions=self._regime_to_actions,
            additional_targets=additional_targets,
        )

        if additional_targets is None:
            object.__setattr__(self, "_cached_df", df)

        return df
```

## DataFrame Creation Strategy

For `_create_flat_dataframe()`, use vectorized operations:

1. For each regime, extract arrays for all periods where that regime was active
2. Stack into regime-level DataFrames with regime-specific columns
3. Concatenate all regime DataFrames
4. Fill missing columns (states/actions not in that regime) with NaN
5. Sort by (subject_id, period)

This avoids row-by-row iteration and leverages pandas/numpy vectorization.

---

## Questions to Resolve Before Starting

1. Should `SimulationResult` expose the raw results for advanced users? (I suggest no - keep it private)
2. Should we add a `__repr__` that shows summary info?
3. For `additional_targets`, should we validate that targets exist in the regime before computing?

---

Ready to start with Commit 1?
