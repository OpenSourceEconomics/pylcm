"""The DC-EGM solve publishes the refined consumption function for simulation.

The Euler inversion plus upper envelope recover the exact off-grid optimal
continuous action on the resources grid. The solve hands that policy to
`simulate` as a per-period `EGMSimPolicy`, so a simulated subject's continuous
action can be interpolated at its resources rather than snapped to the action
grid. This test pins the published artifact directly: interpolating it must
reproduce the closed-form consumption, including at resources strictly between
action-grid nodes (where a grid argmax cannot land).
"""

from collections.abc import Mapping
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.interp import interp_on_padded_grid
from _lcm.egm.published_policy import EGMSimPolicy, NNBEGMSimPolicy
from _lcm.solution import backward_induction
from _lcm.utils.logging import get_logger
from lcm import AgeGrid, LogSpacedGrid, Model
from lcm.regime import Regime as UserRegime
from lcm.solver_api import SIMULATION_POLICY
from lcm.typing import ContinuousState, FloatND, RegimeName, UserParams
from lcm_examples.iskhakov_et_al_2017 import WEALTH_GRID
from tests.conftest import EXACT_KERNEL_SKIP_REASON
from tests.test_models import n_nbegm_toy
from tests.test_models.deterministic import retirement_only
from tests.test_models.deterministic.dcegm_variants import (
    dcegm_retirement,
    get_retirement_only_params,
)

pytestmark = pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)


def _bequest_utility(*, wealth: ContinuousState, age: float) -> FloatND:
    return (age / 50.0) * jnp.log(wealth)


def _two_period_bequest_model() -> Model:
    """Two-period log-utility retirement model with a terminal bequest."""
    bequest_dead = UserRegime(
        transition=None,
        states={"wealth": LogSpacedGrid(start=0.25, stop=400.0, n_points=400)},
        functions={"utility": _bequest_utility},
    )
    return Model(
        regimes={
            "retirement": dcegm_retirement.replace(active=lambda age: age < 50),
            "dead": bequest_dead,
        },
        ages=AgeGrid(start=40, stop=50, step="10Y"),
        regime_id_class=retirement_only.RetirementOnlyRegimeId,
    )


def _kernel_published_policies(
    *, model: Model, params: UserParams, monkeypatch: pytest.MonkeyPatch
) -> Mapping[int, Mapping[RegimeName, object]]:
    """Return every simulation policy the model's kernels publish, by period.

    The solve retains a policy only where the regime's declared simulation
    route reads it; the kernel's own publication is read off its output.
    """
    published: dict[int, dict[RegimeName, object]] = {}
    original = backward_induction._run_period_kernel

    def recording(**kwargs: Any) -> Any:
        output = original(**kwargs)
        if SIMULATION_POLICY in output.replay:
            published.setdefault(kwargs["period"], {})[kwargs["regime_name"]] = (
                output.replay[SIMULATION_POLICY]
            )
        return output

    monkeypatch.setattr(backward_induction, "_run_period_kernel", recording)
    backward_induction.solve(
        flat_params=model._process_params(params),
        ages=model.ages,
        regimes=model._regimes,
        logger=get_logger(log_level="off"),
        enable_jit=model.enable_jit,
        retain_replay=True,
    )
    return published


def test_solve_publishes_policy_matching_closed_form_consumption(
    monkeypatch: pytest.MonkeyPatch,
):
    """Interpolating the published policy reproduces `c* = wealth / (1 + beta)`.

    With log utility, zero interest, a two-period horizon, and a terminal
    bequest `(age/50) log(wealth)` at age 50, the decision period's optimal
    consumption is `wealth / (1 + beta)` at every resources level. The published
    policy interpolated at off-grid resources must hit it.
    """
    discount_factor = 0.98
    params = get_retirement_only_params(n_periods=2, discount_factor=discount_factor)

    sim_policy = _kernel_published_policies(
        model=_two_period_bequest_model(), params=params, monkeypatch=monkeypatch
    )

    pol = sim_policy[0]["retirement"]
    assert isinstance(pol, EGMSimPolicy)

    # Resources = wealth here (zero interest, no labor income). Query strictly
    # between wealth-grid nodes to exercise the off-grid interpolation.
    wealth_nodes = np.asarray(WEALTH_GRID.to_jax())
    off_grid = 0.5 * (wealth_nodes[3:-1] + wealth_nodes[4:])
    consumption = interp_on_padded_grid(
        x_query=jnp.asarray(off_grid), xp=pol.endog_grid, fp=pol.policy
    )
    expected = off_grid / (1.0 + discount_factor)
    np.testing.assert_allclose(np.asarray(consumption), expected, rtol=2e-2)


def test_retained_policies_are_host_resident():
    """Solve evicts retained simulation policies to host, not device.

    The policies are a solve output no backward step reads; keeping them on the
    accelerator would pin one carry-sized buffer per period for the whole
    induction. So the returned policy arrays live on the host (CPU) device. A
    policy is retained only through a declared simulation route, so the witness
    is a nested solver regime whose route reads one.
    """
    model = n_nbegm_toy.build_model(variant="n_nbegm", n_periods=2)
    result = backward_induction.solve(
        flat_params=model._process_params({"discount_factor": 0.95}),
        ages=model.ages,
        regimes=model._regimes,
        logger=get_logger(log_level="off"),
        enable_jit=model.enable_jit,
        retain_replay=True,
    )

    pol = result.simulation_policies[0]["alive"]
    assert isinstance(pol, NNBEGMSimPolicy)
    assert all(
        device.platform == "cpu"
        for array in (pol.candidate_inner_action, pol.candidate_value)
        for device in array.devices()
    )
