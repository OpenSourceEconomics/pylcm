"""A BQSEGM schedule regime publishes its per-cell cliff preimages on the carry.

The parent's side-faithful continuation read needs to know where each child
cell's value jumps. The child computes those breakpoint preimages during its
own solve, so it publishes them on `EGMCarry.breakpoints` with the ride axes
leading — one row of jump locations per ride cell — and the lowering template
carries the same fixed shape so compiled parents never see a pytree change.
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
                lambda bp: seen.append(np.asarray(bp)), carry.breakpoints
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
    breakpoints = captured_carries[-1]
    assert breakpoints.shape == (2, 1)
    np.testing.assert_allclose(breakpoints[:, 0], _CLIFF_BY_KIND, atol=1e-6)
