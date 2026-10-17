"""Published finite nested candidate tuples match exact independent economics."""

from fractions import Fraction

import numpy as np
import pytest

from _lcm.egm.published_policy import NNBEGMSimPolicy
from lcm.solver_api import SIMULATION_POLICY
from tests.conftest import DECIMAL_PRECISION, EXACT_KERNEL_SKIP_REASON
from tests.solution import test_nested_dynamic_borrowing as finite
from tests.solution._finite_candidate_budget_oracle import (
    assert_tuple,
    conditional_bank,
    owner,
)

_LIQUID_NODES = (
    Fraction(0),
    Fraction(1, 4),
    Fraction(1, 2),
    Fraction(3, 4),
    Fraction(1),
)
_STOCK_NODES = (0, 1, 2)


@pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)
@pytest.mark.parametrize("dynamic", [False, True])
def test_finite_candidate_tuples_match_exact_budget_economics(
    *, dynamic: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every candidate's action, value and target reflect one feasible budget.

    A zero discount factor makes every legal optimum a binding saving-floor
    corner, so the owner assertions concern exact corner ties in keeper-first,
    declared-target order. The direct controls run the producer's own keeper
    and adjusters on the same prepared current-state floor.
    """
    model = finite._model(
        family="nnbegm", lower=finite.current_floor if dynamic else 0.0
    )
    published, controls = finite._branches(model=model, monkeypatch=monkeypatch)
    finite._assert_published_bank(
        published=published, outputs=controls, family="nnbegm"
    )
    policy = published.replay[SIMULATION_POLICY]
    assert isinstance(policy, NNBEGMSimPolicy)
    assert policy.state_names == ("liquid", "illiquid")
    assert policy.n_keeper_candidates == 1
    np.testing.assert_array_equal(policy.outer_grid_values, [0.0, 1.0, 2.0])
    tolerance = 1.5 * 10.0 ** (-DECIMAL_PRECISION)
    values = np.asarray(policy.candidate_value)
    actions = np.asarray(policy.candidate_inner_action)
    targets = np.asarray(policy.candidate_outer_target)
    for liquid_index, liquid in enumerate(_LIQUID_NODES):
        for stock_index, stock in enumerate(_STOCK_NODES):
            expected = conditional_bank(
                liquid=liquid, stock=stock, dynamic_floor=dynamic
            )
            for candidate, row in enumerate(expected):
                assert_tuple(
                    expected=row,
                    action=float(actions[candidate, liquid_index, stock_index]),
                    value=float(values[candidate, liquid_index, stock_index]),
                    target=float(targets[candidate, liquid_index, stock_index]),
                    abs_tol=tolerance,
                )
                np.testing.assert_array_almost_equal(
                    np.asarray(controls[candidate].value)[liquid_index, stock_index],
                    row.value,
                    decimal=DECIMAL_PRECISION,
                )
            assert int(np.argmax(values[:, liquid_index, stock_index])) == owner(
                expected
            )
