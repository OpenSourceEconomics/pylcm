"""A ride-along NB-EGM kernel carries only the targets its own period reads.

The continuation carries are passed to a period's compiled core as a pytree. A
period reads the targets its own period reaches, which in a model whose reachable
set varies over the lifecycle is narrower than the union across all periods.
Filtering to the union would hand each core carries it never indexes.
"""

import pytest

from tests.test_models import nbegm_jump_ride_along_toy as toy

_REGIME = "alive"


@pytest.fixture(scope="module")
def model():
    """A ride-along NB-EGM model whose reachable target set varies by period."""
    return toy.build_model(
        variant="nbegm",
        n_periods=4,
        n_liquid=8,
        n_consumption=8,
        liquid_max=10.0,
        n_savings=8,
        savings_max=8.0,
    )


def test_the_toy_reaches_different_targets_in_different_periods(model):
    """The union of reachable targets is strictly wider than any single period's.

    Without this the carry-subset invariant below would hold trivially.
    """
    reachability = model.reachability.solution
    union = set(reachability.union_targets(source=_REGIME))
    per_period = [
        set(reachability.targets(period=period, source=_REGIME))
        for period in model._regimes[_REGIME].solution.period_kernels
    ]
    assert all(targets < union for targets in per_period)


def test_each_period_kernel_carries_exactly_its_own_periods_targets(model):
    """A period's carry set is that period's reachable targets, not the union."""
    reachability = model.reachability.solution
    period_kernels = model._regimes[_REGIME].solution.period_kernels
    carried = {
        period: set(kernel.stateful_targets)
        for period, kernel in period_kernels.items()
    }
    expected = {
        period: set(reachability.targets(period=period, source=_REGIME))
        for period in period_kernels
    }
    assert carried == expected
