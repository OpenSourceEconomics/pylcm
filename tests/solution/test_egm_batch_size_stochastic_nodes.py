"""DC-EGM splaying: `stochastic_node_batch_size` is a memory knob only.

The continuation expectation runs over the product of the child regime's
stochastic process nodes. That node axis is carried by the dominant `egm_step`
working buffer (the savings nodes times the child stochastic mesh times the
combo block). A positive block size accumulates the per-node reads in `lax.scan`
blocks rather than one fused vmap, folding the weighted sum into the scan carry
to shed peak working-set memory. The per-node results are summed with the joint
intrinsic weights either way, so the solved value function matches the unsplayed
(`stochastic_node_batch_size=0`) solve to tight numerical tolerance, whatever the
block size — including block sizes that do not divide the mesh. The block
reduction reorders the floating-point adds, so the match is to tolerance, not
bit-identical.
"""

import functools

import numpy as np
import pytest

from _lcm.typing import PeriodToRegimeToVArr
from lcm import AgeGrid, Model
from lcm.regime import Regime as UserRegime
from lcm.solvers import DCEGM
from tests.solution.test_egm_process_states import (
    CONSUMPTION_GRID,
    N_INCOME_NODES,
    N_PERIODS,
    SAVINGS_GRID,
    WEALTH_GRID,
    ProcessRegimeId,
    _get_params,
    _income_process,
    dead,
    inverse_marginal_utility,
    next_regime,
    next_wealth_from_savings_iid,
    resources,
    savings,
    utility_consumption_only,
)


@functools.cache
def _model(stochastic_node_batch_size: int) -> Model:
    """DC-EGM model with an IID income process and a splayable node expectation."""
    ages = AgeGrid(start=40, stop=40 + (N_PERIODS - 1) * 10, step="10Y")
    last_age = float(ages.exact_values[-1])
    working = UserRegime(
        transition=next_regime,
        active=lambda age, la=last_age: age < la,
        actions={"consumption": CONSUMPTION_GRID},
        states={"wealth": WEALTH_GRID, "income": _income_process("iid")},
        state_transitions={"wealth": next_wealth_from_savings_iid},
        functions={
            "utility": utility_consumption_only,
            "resources": resources,
            "savings": savings,
            "inverse_marginal_utility": inverse_marginal_utility,
        },
        solver=DCEGM(
            continuous_state="wealth",
            continuous_action="consumption",
            resources="resources",
            post_decision_function="savings",
            savings_grid=SAVINGS_GRID,
            n_constrained_points=64,
            stochastic_node_batch_size=stochastic_node_batch_size,
        ),
    )
    return Model(
        regimes={"alive": working, "dead": dead},
        ages=ages,
        regime_id_class=ProcessRegimeId,
    )


def _solve(stochastic_node_batch_size: int) -> PeriodToRegimeToVArr:
    return _model(stochastic_node_batch_size).solve(
        params=_get_params("iid"), log_level="debug"
    )


@pytest.mark.parametrize("stochastic_node_batch_size", [1, 2, 3, N_INCOME_NODES])
def test_stochastic_node_batch_size_leaves_value_function_unchanged(
    stochastic_node_batch_size,
):
    """Splaying the child node expectation into blocks does not change the solved V.

    `stochastic_node_batch_size` only changes how the per-node continuation
    reads are scheduled and reduced (`lax.scan` blocks accumulating the weighted
    sum, instead of one fused vmap), so the value function at every period
    matches the unsplayed `stochastic_node_batch_size=0` solve to tight
    numerical tolerance — including a block size (3) that does not divide the
    5-node income mesh, and the boundary size equal to the mesh length (which
    falls back to the vmap). The block reduction reorders the floating-point
    adds, so the match is to tolerance, not bit-identical.
    """
    reference = _solve(0)
    splayed = _solve(stochastic_node_batch_size)
    assert set(reference) == set(splayed)
    for period in sorted(reference):
        assert set(reference[period]) == set(splayed[period])
        for regime_name in reference[period]:
            ref_V = np.asarray(reference[period][regime_name])
            got_V = np.asarray(splayed[period][regime_name])
            assert ref_V.shape == got_V.shape
            np.testing.assert_allclose(
                got_V,
                ref_V,
                rtol=1e-9,
                atol=1e-9,
                err_msg=f"period={period}, regime={regime_name}",
            )
