"""A BQSEGM schedule regime publishes its value jumps inside its carry rows.

The parent's continuation read must never average across a child cell's
value jump. The child makes that structurally impossible: each carry row's
endogenous grid contains every jump preimage as a duplicated abscissa
carrying the exact left and right value limits, so the ordinary padded-row
read is one-sided by construction. `EGMCarry.breakpoints` carries the jump
locations with the ride axes leading (the stochastic-dim fold's topology
marker), and the lowering template shares both fixed shapes so compiled
parents never see a pytree change.
"""

import jax
import numpy as np
import pytest

import _lcm.egm.continuation as cont_mod
from tests.test_models import bqsegm_jump_ride_along_toy as toy

_CLIFF_BY_KIND = (15.0 - 1.0, 15.0 - 4.0)


@pytest.fixture
def captured_carries(monkeypatch):
    seen = []
    original = cont_mod._get_child_carry_reader

    def spy(*args, **kwargs):
        carry = kwargs["carry"] if "carry" in kwargs else args[0]
        if carry.breakpoints is not None:
            jax.debug.callback(
                lambda bp, xp, fp: seen.append(
                    (np.asarray(bp), np.asarray(xp), np.asarray(fp))
                ),
                carry.breakpoints,
                carry.endog_grid,
                carry.value,
            )
        return original(*args, **kwargs)

    monkeypatch.setattr(cont_mod, "_get_child_carry_reader", spy)
    return seen


def test_schedule_carry_breakpoints_are_the_per_kind_cliff_preimages(
    captured_carries,
):
    """The self-read carry exposes each kind's asset-space cliff location."""
    model = toy.build_model(
        variant="bqsegm",
        n_liquid=24,
        liquid_max=30.0,
        n_savings=40,
        savings_max=28.0,
        n_consumption=8,
    )
    model.solve(params=toy.build_params(), log_level="off")

    assert captured_carries, "no captured carry published breakpoints"
    breakpoints, _, _ = captured_carries[-1]
    assert breakpoints.shape == (2, 1)
    np.testing.assert_allclose(breakpoints[:, 0], _CLIFF_BY_KIND, atol=1e-6)


def test_schedule_carry_rows_hold_the_cliff_as_a_duplicated_abscissa(
    captured_carries,
):
    """Each carry row samples both one-sided limits at its kind's cliff.

    The row's endogenous grid contains the cliff preimage twice — once per
    side — so a read near the cliff brackets against the exact one-sided
    values instead of straddling the jump. The toy's value drops across the
    cliff, so the left limit strictly exceeds the right limit.
    """
    n_liquid = 24
    model = toy.build_model(
        variant="bqsegm",
        n_liquid=n_liquid,
        liquid_max=30.0,
        n_savings=40,
        savings_max=28.0,
        n_consumption=8,
    )
    model.solve(params=toy.build_params(), log_level="off")

    assert captured_carries, "no captured carry published breakpoints"
    _, endog, value = captured_carries[-1]
    assert endog.shape == (2, n_liquid + 2)
    for kind in range(2):
        at_cliff = np.isclose(endog[kind], _CLIFF_BY_KIND[kind], atol=1e-6)
        assert at_cliff.sum() == 2, "cliff preimage must appear exactly twice"
        left_value, right_value = value[kind, at_cliff]
        assert left_value > right_value + 1e-3
