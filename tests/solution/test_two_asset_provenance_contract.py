"""The declared accuracy budgets are internally satisfiable.

`median <= p90 <= max` holds for any sample, because each statistic dominates
the previous one by construction. A contract declaring budgets that violate that
ordering is therefore unsatisfiable: no implementation, correct or not, can meet
a `max` budget below its own `p90` budget without making the looser one dead
letter. Such a contract reports its own inconsistency rather than the code's
accuracy, so it is checked here rather than trusted.

Every workload must also declare all three statistics. A budget that names only
a median cannot see a minority of badly wrong nodes, which is the shape a
coverage failure takes.
"""

import pytest

from tests.solution.test_egm_continuation_grid_provenance import _CONTRACT

_WORKLOADS = sorted(_CONTRACT["workloads"])
_REQUIRED = ("median_value_regret", "p90_value_regret", "max_value_regret")


@pytest.mark.parametrize("workload", _WORKLOADS)
def test_every_workload_declares_all_three_statistics(workload):
    """A workload budgets its median, its p90 and its maximum."""
    budget = _CONTRACT["workloads"][workload]["budget"]
    assert set(budget) == set(_REQUIRED)


@pytest.mark.parametrize("workload", _WORKLOADS)
def test_declared_budgets_are_ordered_and_therefore_satisfiable(workload):
    """`median <= p90 <= max` in every declared budget."""
    budget = _CONTRACT["workloads"][workload]["budget"]
    assert (
        budget["median_value_regret"]
        <= budget["p90_value_regret"]
        <= budget["max_value_regret"]
    )


@pytest.mark.parametrize("workload", _WORKLOADS)
@pytest.mark.parametrize("profile", ["float64", "float32"])
def test_recorded_measurements_sit_inside_the_declared_budget(workload, profile):
    """The measurements the contract reports satisfy the budgets it declares.

    A budget below its own recorded measurement would fail the moment it were
    enforced, so the inconsistency belongs in the contract's own gate rather
    than in a solve.
    """
    entry = _CONTRACT["workloads"][workload]
    measured = entry["measured"][profile]
    budget = entry["budget"]
    for statistic, field in zip(("median", "p90", "max"), _REQUIRED, strict=True):
        assert measured[statistic] < budget[field]
