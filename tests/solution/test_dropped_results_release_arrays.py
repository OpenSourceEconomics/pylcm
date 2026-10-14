"""A result nobody references any more owns nothing that outlives it.

pylcm runs under a beartype claw that decorates every function definition, including
one executed inside a call, and beartype memoizes each decorated function object for
the rest of the process. A per-call closure would therefore pin everything it closes
over — a store's value entries, an artifact's leaves — long after the result that
produced them is gone. The constructors and plan walkers keep their working state in
explicit arguments instead, so dropping a result releases its arrays.
"""

import gc
import weakref

import numpy as np
import pytest
from numpy.typing import NDArray

from lcm.solver_api import (
    ValueStore,
    _artifact_leaf_values_from_plan,
    _ArtifactLeafSlot,
    _ArtifactTuplePlan,
    _CanonicalValueEntry,
    _reconstruct_artifact_from_plan,
)
from tests.solution.test_solution_result import _small_grid_search_inputs

# Distinctive lengths that no other test array in the process is likely to share.
_VALUE_LENGTH = 7919
_LEAF_LENGTH = 7907


def _holds_array(
    *,
    candidate: object,  # noqa: PAN001 - gc exposes arbitrary interpreter objects
    length: int,
) -> bool:
    """Whether a gc-tracked container or entry holds an array of `length` first."""
    # The gc heap also holds weak proxies whose referent may be gone, and
    # `isinstance` dereferences a proxy, so proxies are excluded by exact type first.
    if type(candidate) in (weakref.ProxyType, weakref.CallableProxyType):
        return False
    if isinstance(candidate, _CanonicalValueEntry):
        held: object = candidate.value  # noqa: PAN001 - tuple/list heads may be anything
    elif isinstance(candidate, tuple | list) and candidate:
        held = candidate[0]
    else:
        return False
    return type(held) is np.ndarray and held.shape == (length,)


def _live_holders(length: int) -> int:
    # NumPy arrays are not gc-tracked themselves, so the probe counts the tracked
    # objects that hold one: a store's value entry, or a tuple/list. A tuple whose
    # members are all untracked is untracked too and escapes this count, which is
    # why the artifact-leaf probe below tracks the arrays by weak reference.
    gc.collect()
    return sum(
        1
        for candidate in gc.get_objects()
        if _holds_array(candidate=candidate, length=length)
    )


def _live_value_entries() -> int:
    gc.collect()
    return sum(
        1 for candidate in gc.get_objects() if type(candidate) is _CanonicalValueEntry
    )


def test_dropped_value_stores_release_their_values() -> None:
    """Constructing and discarding stores leaves none of their values alive."""
    for _ in range(8):
        ValueStore({0: {"alive": np.zeros(_VALUE_LENGTH)}})

    assert _live_holders(_VALUE_LENGTH) == 0


def _any_leaf_alive_after_walks(*, walker: str, retain: bool) -> bool:
    """Walk fresh leaves through an artifact plan; report whether any leaf survives.

    With `retain`, the leaf tuples are kept alive on purpose, so the probe must
    report a survivor.
    """
    plan = _ArtifactTuplePlan(children=(_ArtifactLeafSlot(index=0),))
    retained: list[tuple[NDArray[np.float64], ...]] = []
    references: list[weakref.ReferenceType[NDArray[np.float64]]] = []
    for _ in range(8):
        leaves = (np.zeros(_LEAF_LENGTH),)
        references.append(weakref.ref(leaves[0]))
        if walker == "reconstruct":
            _reconstruct_artifact_from_plan(plan=plan, leaves=leaves)
        else:
            _artifact_leaf_values_from_plan(payload=leaves, plan=plan, leaf_count=1)
        if retain:
            retained.append(leaves)
    del leaves
    gc.collect()
    return any(reference() is not None for reference in references)


@pytest.mark.parametrize("walker", ["reconstruct", "extract"])
def test_dropped_artifact_plan_walks_release_their_leaves(walker: str) -> None:
    """Building or reading an artifact through its plan retains no leaf."""
    assert not _any_leaf_alive_after_walks(walker=walker, retain=False)


@pytest.mark.parametrize("walker", ["reconstruct", "extract"])
def test_artifact_leaf_probe_reports_retained_leaf_tuples(walker: str) -> None:
    """A tuple holding only the leaf array keeps it alive, and the probe says so."""
    assert _any_leaf_alive_after_walks(walker=walker, retain=True)


def test_dropped_solution_and_simulation_release_value_entries() -> None:
    """Solving and simulating, then dropping both, leaves no new value entry alive."""
    model, params, initial_conditions = _small_grid_search_inputs()
    before = _live_value_entries()

    solved = model.solve(params=params, log_level="off")
    model.simulate(
        params=params,
        initial_conditions=initial_conditions,
        solution=solved,
        log_level="off",
    )
    del solved

    assert _live_value_entries() == before
