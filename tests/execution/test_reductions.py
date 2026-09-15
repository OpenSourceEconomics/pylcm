"""Reductions are declared at two levels: the contract, and the fold behind it."""

import pytest

from _lcm.execution.reductions import (
    EXACTNESS_VALUES,
    HARD_MAX_WITH_CARRY_REDUCTION,
    INTERVAL_ENVELOPE_REDUCTION,
    WEIGHTED_EXPECTATION_REDUCTION,
    ReductionDeclaration,
    ReductionSemantics,
)
from _lcm.solution.action_reduction import HARD_MAX_REDUCTION
from _lcm.solution.collective_action_reduction import COLLECTIVE_HARD_MAX_REDUCTION
from _lcm.solution.logsumexp_action_reduction import LOGSUMEXP_REDUCTION

_SHIPPED_FOLDS = {
    "hard_max": HARD_MAX_REDUCTION,
    "logsumexp": LOGSUMEXP_REDUCTION,
    "collective_hard_max": COLLECTIVE_HARD_MAX_REDUCTION,
    "weighted_expectation": WEIGHTED_EXPECTATION_REDUCTION,
    "hard_max_with_carry": HARD_MAX_WITH_CARRY_REDUCTION,
}

_DECLARATIONS_ONLY = {
    "interval_envelope": INTERVAL_ENVELOPE_REDUCTION,
}

_SPECS = _SHIPPED_FOLDS | _DECLARATIONS_ONLY

_EXPECTED_EXACTNESS = {
    "hard_max": "exact",
    "logsumexp": "tolerance_equivalent",
    "collective_hard_max": "exact",
    "weighted_expectation": "tolerance_equivalent",
    "hard_max_with_carry": "exact",
    "interval_envelope": "exact",
}


@pytest.mark.parametrize("key", sorted(_SPECS))
def test_reduction_satisfies_the_declaration_protocol(key: str) -> None:
    """Every specification names its contract through the declaration protocol."""
    assert isinstance(_SPECS[key], ReductionDeclaration)


@pytest.mark.parametrize("key", sorted(_SHIPPED_FOLDS))
def test_shipped_reduction_carries_a_planner_drivable_fold(key: str) -> None:
    """A specification whose kernel this module owns satisfies the fold protocol."""
    assert isinstance(_SHIPPED_FOLDS[key], ReductionSemantics)


@pytest.mark.parametrize("key", sorted(_DECLARATIONS_ONLY))
def test_declaration_only_reduction_carries_no_fold(key: str) -> None:
    """A specification whose kernel lives with its solver stays a declaration."""
    assert not isinstance(_DECLARATIONS_ONLY[key], ReductionSemantics)


@pytest.mark.parametrize("key", sorted(_SPECS))
def test_reduction_declares_its_exactness(key: str) -> None:
    """Each specification states whether block order can move the result."""
    assert _SPECS[key].exactness == _EXPECTED_EXACTNESS[key]


@pytest.mark.parametrize("key", sorted(_SPECS))
def test_reduction_exactness_is_a_known_value(key: str) -> None:
    """Exactness is one of the two published values."""
    assert _SPECS[key].exactness in EXACTNESS_VALUES


def test_semantic_keys_are_pairwise_distinct() -> None:
    """No two specifications share a semantic key."""
    keys = [spec.semantic_key for spec in _SPECS.values()]

    assert len(set(keys)) == len(keys)
