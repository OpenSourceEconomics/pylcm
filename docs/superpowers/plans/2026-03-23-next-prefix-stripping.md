# `next_` Prefix Stripping Simplification

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace expensive tree-path round-trips with simple string operations when stripping the `next_` prefix from transition output keys during simulation.

**Architecture:** During simulation, transition functions output state names like `"regime__next_wealth"` which must be mapped back to `"regime__wealth"`. Currently this uses `tree_path_from_qname` / `qname_from_tree_path` from `dags.tree` (tuple decomposition + reconstruction). Since the separator is always `__` and the `next_` prefix is always on the leaf component, a single `str.replace("__next_", "__", 1)` call suffices. This runs every period for every regime, so removing the overhead matters.

**Tech Stack:** Python string operations, JAX arrays, `MappingProxyType`

---

### Background

**Qname convention:** Qualified names use `__` as separator. `"regime__next_wealth"` means regime=`"regime"`, leaf=`"next_wealth"`. The `next_` prefix is added by `_extract_transitions_from_regime` in `regime_processing.py` at model-build time and must be stripped at simulation time to map back to state names.

**Why keys are always namespaced:** `SolveFunctions.transitions` / `SimulateFunctions.transitions` has type `TransitionFunctionsMapping = MappingProxyType[RegimeName, MappingProxyType[str, InternalUserFunction]]` — always `{target_regime: {next_state: func}}`. After `flatten_regime_namespace`, keys are always `"target_regime__next_state"`. The `"next_regime"` key is excluded before storage (see `regime_processing.py:384`).

---

### Task 1: Add unit tests for `_update_states_for_subjects`

**Files:**
- Create: `tests/simulation/test_update_states.py`

There are currently no unit tests for `_update_states_for_subjects`. Add them before changing the implementation.

- [ ] **Step 1: Write tests for the current behavior**

```python
from types import MappingProxyType

import jax.numpy as jnp
from numpy.testing import assert_array_equal

from lcm.simulation.utils import _update_states_for_subjects


def test_update_states_strips_next_prefix():
    all_states = MappingProxyType({
        "working__wealth": jnp.array([10.0, 20.0, 30.0]),
    })
    computed_next_states = MappingProxyType({
        "working__next_wealth": jnp.array([15.0, 25.0, 35.0]),
    })
    subject_indices = jnp.array([True, False, True])

    result = _update_states_for_subjects(
        all_states=all_states,
        computed_next_states=computed_next_states,
        subject_indices=subject_indices,
    )

    assert_array_equal(result["working__wealth"], jnp.array([15.0, 20.0, 35.0]))


def test_update_states_multiple_regimes_and_states():
    all_states = MappingProxyType({
        "working__wealth": jnp.array([10.0, 20.0]),
        "working__health": jnp.array([1.0, 2.0]),
        "retired__wealth": jnp.array([100.0, 200.0]),
    })
    computed_next_states = MappingProxyType({
        "working__next_wealth": jnp.array([15.0, 25.0]),
        "working__next_health": jnp.array([1.5, 2.5]),
    })
    subject_indices = jnp.array([True, True])

    result = _update_states_for_subjects(
        all_states=all_states,
        computed_next_states=computed_next_states,
        subject_indices=subject_indices,
    )

    assert_array_equal(result["working__wealth"], jnp.array([15.0, 25.0]))
    assert_array_equal(result["working__health"], jnp.array([1.5, 2.5]))
    # Untouched state remains unchanged
    assert_array_equal(result["retired__wealth"], jnp.array([100.0, 200.0]))


def test_update_states_no_subjects_selected():
    all_states = MappingProxyType({
        "r__wealth": jnp.array([10.0, 20.0]),
    })
    computed_next_states = MappingProxyType({
        "r__next_wealth": jnp.array([99.0, 99.0]),
    })
    subject_indices = jnp.array([False, False])

    result = _update_states_for_subjects(
        all_states=all_states,
        computed_next_states=computed_next_states,
        subject_indices=subject_indices,
    )

    assert_array_equal(result["r__wealth"], jnp.array([10.0, 20.0]))
```

- [ ] **Step 2: Run tests to verify they pass (testing current implementation)**

Run: `pixi run -e tests-cpu tests -- tests/simulation/test_update_states.py -v`
Expected: 3 PASSED

- [ ] **Step 3: Commit**

```
git add tests/simulation/test_update_states.py
git commit -m "test: add unit tests for _update_states_for_subjects"
```

---

### Task 2: Replace tree-path round-trip with string replace

**Files:**
- Modify: `src/lcm/simulation/utils.py:282-293`

- [ ] **Step 1: Replace the tree-path stripping with string replace**

In `_update_states_for_subjects`, replace lines 282-292:

```python
    updated_states = dict(all_states)
    for next_state_name, next_state_values in computed_next_states.items():
        # State names may be prefixed with regime (e.g., "working__next_wealth")
        # We need to strip "next_" from the final component to get "working__wealth"
        path = tree_path_from_qname(next_state_name)
        state_name = qname_from_tree_path((*path[:-1], path[-1].removeprefix("next_")))
        updated_states[state_name] = jnp.where(
            subject_indices,
            next_state_values,
            all_states[state_name],
        )
```

With:

```python
    updated_states = dict(all_states)
    for next_state_name, next_state_values in computed_next_states.items():
        # Transition outputs are always namespaced: "regime__next_wealth" → "regime__wealth"
        state_name = next_state_name.replace("__next_", "__", 1)
        updated_states[state_name] = jnp.where(
            subject_indices,
            next_state_values,
            all_states[state_name],
        )
```

- [ ] **Step 2: Remove unused `qname_from_tree_path` import**

In `src/lcm/simulation/utils.py` line 5, change:

```python
from dags.tree import qname_from_tree_path, tree_path_from_qname
```

To:

```python
from dags.tree import tree_path_from_qname
```

(`tree_path_from_qname` is still used on line 112 for stochastic transition name matching.)

- [ ] **Step 3: Run the unit tests**

Run: `pixi run -e tests-cpu tests -- tests/simulation/test_update_states.py -v`
Expected: 3 PASSED

- [ ] **Step 4: Run the full regression test suite**

Run: `pixi run -e tests-cpu tests -- tests/test_regression_test.py -v`
Expected: All PASSED (GPU benchmark tests will be skipped on CPU)

- [ ] **Step 5: Run pre-commit and ty**

Run: `prek run --all-files && pixi run ty`
Expected: All checks passed

- [ ] **Step 6: Commit**

```
git add src/lcm/simulation/utils.py
git commit -m "refactor: replace tree-path round-trip with string replace in _update_states_for_subjects"
```
