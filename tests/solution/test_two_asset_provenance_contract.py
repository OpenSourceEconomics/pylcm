"""Every declared accuracy budget can bind.

`median <= p90 <= max` holds for any sample, because each statistic dominates
the previous one by construction. A budget that declares them out of that order
is not impossible to meet -- a sample inside the tighter bound satisfies both --
but the tighter clause dominates, so the looser one can never bind. The contract
then states a promise it never enforces, and the unenforced clause reads as
though it were the operative one.

Ordering is therefore checked here rather than trusted: it is what makes each
declared statistic a real constraint instead of decoration.

Every workload must also declare all three statistics. A budget that names only
a median cannot see a minority of badly wrong nodes, which is the shape a
coverage failure takes.
"""

import pytest

from tests.solution.test_egm_continuation_grid_provenance import _CONTRACT

_WORKLOADS = sorted(_CONTRACT["workloads"])
_REQUIRED = ("median_value_regret", "p90_value_regret", "max_value_regret")
_MEASUREMENTS = [
    (workload, profile)
    for workload in _WORKLOADS
    for profile in sorted(_CONTRACT["workloads"][workload]["measured"])
]


@pytest.mark.parametrize("workload", _WORKLOADS)
def test_every_workload_declares_all_three_statistics(workload):
    """A workload budgets its median, its p90 and its maximum."""
    budget = _CONTRACT["workloads"][workload]["budget"]
    assert set(budget) == set(_REQUIRED)


@pytest.mark.parametrize("workload", _WORKLOADS)
def test_every_declared_budget_can_bind(workload):
    """`median <= p90 <= max`, so no declared statistic is dominated away."""
    budget = _CONTRACT["workloads"][workload]["budget"]
    assert (
        budget["median_value_regret"]
        <= budget["p90_value_regret"]
        <= budget["max_value_regret"]
    )


@pytest.mark.parametrize(("workload", "profile"), _MEASUREMENTS)
def test_recorded_measurements_sit_inside_the_declared_budget(workload, profile):
    """The measurements the contract reports satisfy the budgets it declares.

    A budget below its own recorded measurement would fail the moment it were
    enforced, so the inconsistency belongs in the contract's own gate rather
    than in a solve.

    Every profile the contract records is checked, so a measurement added on a
    new device or precision binds the budget instead of merely annotating it.
    """
    entry = _CONTRACT["workloads"][workload]
    measured = entry["measured"][profile]
    budget = entry["budget"]
    for statistic, field in zip(("median", "p90", "max"), _REQUIRED, strict=True):
        assert measured[statistic] < budget[field]
