"""Every reduction specification satisfies the public protocol."""

import pytest

from _lcm.execution.reductions import (
    EXACTNESS_VALUES,
    HardMaxWithCarryReduction,
    IntervalEnvelopeReduction,
    ReductionSemantics,
    WeightedExpectationReduction,
)
from _lcm.solution.action_reduction import HARD_MAX_REDUCTION
from _lcm.solution.collective_action_reduction import COLLECTIVE_HARD_MAX_REDUCTION
from _lcm.solution.logsumexp_action_reduction import LOGSUMEXP_REDUCTION

_SPECS = {
    "hard_max": HARD_MAX_REDUCTION,
    "logsumexp": LOGSUMEXP_REDUCTION,
    "collective_hard_max": COLLECTIVE_HARD_MAX_REDUCTION,
    "weighted_expectation": WeightedExpectationReduction(),
    "hard_max_with_carry": HardMaxWithCarryReduction(),
    "interval_envelope": IntervalEnvelopeReduction(),
}

_EXPECTED_EXACTNESS = {
    "hard_max": "exact",
    "logsumexp": "tolerance_equivalent",
    "collective_hard_max": "exact",
    "weighted_expectation": "tolerance_equivalent",
    "hard_max_with_carry": "exact",
    "interval_envelope": "exact",
}


@pytest.mark.parametrize("key", sorted(_SPECS))
def test_reduction_satisfies_the_protocol(key: str) -> None:
    """A reduction specification is an instance of the runtime-checkable protocol."""
    assert isinstance(_SPECS[key], ReductionSemantics)


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
