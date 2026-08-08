"""The provenance contract declares a sentinel that can bind, on both statistics.

The contract is a **coverage sentinel**, not an accuracy budget. It exists to
catch a mask that stops covering its interior, or a backend that stops producing
a comparable solution at all. It does not certify any backend's accuracy, and
the levels it records must not be read as one — G2EGM's included.

That restricts which statistics may gate. A statistic can only be a sentinel if
it is stable across the profiles the contract records, because a bound on a
statistic that moves with the device is a bound on the device. Two qualify:

- the **median**, which moves when a solution degrades broadly;
- the **maximum**, which moves first when a mask stops covering a node, since an
  uncovered node is a large regret.

Both are declared for every workload, and both are checked here rather than
trusted:

- `median <= max` holds for any sample by construction, so a budget declaring
  them out of that order lets the tighter clause dominate and the looser one
  never bind. The contract would then state a promise it never enforces.
- Every workload declares both. A sentinel naming only a median cannot see a
  minority of badly wrong nodes, which is the shape a coverage failure takes.
"""

import pytest

from tests.solution.test_egm_continuation_grid_provenance import _CONTRACT

_WORKLOADS = sorted(_CONTRACT["workloads"])
_REQUIRED = ("median_value_regret", "max_value_regret")
_STATISTICS = ("median", "max")
_MEASUREMENTS = [
    (workload, profile)
    for workload in _WORKLOADS
    for profile in sorted(_CONTRACT["workloads"][workload]["measured"])
]


@pytest.mark.parametrize("workload", _WORKLOADS)
def test_every_workload_declares_both_sentinel_statistics(workload):
    """A workload declares its median and its maximum, and nothing else."""
    sentinel = _CONTRACT["workloads"][workload]["sentinel"]
    assert set(sentinel) == set(_REQUIRED)


@pytest.mark.parametrize("workload", _WORKLOADS)
def test_every_declared_sentinel_can_bind(workload):
    """`median <= max`, so neither declared statistic is dominated away."""
    sentinel = _CONTRACT["workloads"][workload]["sentinel"]
    assert sentinel["median_value_regret"] <= sentinel["max_value_regret"]


@pytest.mark.parametrize("workload", _WORKLOADS)
def test_no_workload_records_a_statistic_it_does_not_gate(workload):
    """Every statistic a profile records is one the sentinel bounds.

    A recorded statistic that nothing gates is decoration: it invites being
    refitted whenever a new device reports a different value, which is the
    signature of a number that measures the device rather than the code.
    """
    for measured in _CONTRACT["workloads"][workload]["measured"].values():
        assert set(measured) == set(_STATISTICS)


@pytest.mark.parametrize(("workload", "profile"), _MEASUREMENTS)
def test_recorded_measurements_sit_inside_the_declared_sentinel(workload, profile):
    """The measurements the contract reports satisfy the sentinel it declares.

    A bound below its own recorded measurement would fail the moment it were
    enforced, so the inconsistency belongs in the contract's own gate rather
    than in a solve.

    Every profile the contract records is checked, so a measurement added on a
    new device or precision binds the sentinel instead of merely annotating it.
    """
    entry = _CONTRACT["workloads"][workload]
    measured = entry["measured"][profile]
    sentinel = entry["sentinel"]
    for statistic, field in zip(_STATISTICS, _REQUIRED, strict=True):
        assert measured[statistic] < sentinel[field]
