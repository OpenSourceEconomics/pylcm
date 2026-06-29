"""BQSEGM agreement with the brute-force oracle on the Medicaid one-asset toy.

The case-piece EGM solve must reproduce the dense-grid `GridSearch` value function
where both are exact: across the asset region away from the boundary, and through
the boundary jump itself. BQSEGM additionally rejects a model whose smooth piece
hides branching.
"""

from collections.abc import Mapping

import jax.numpy as jnp
import numpy as np
import pytest

import lcm
from lcm import AgeGrid, LinSpacedGrid, MarkovTransition, Model
from lcm.exceptions import BQSEGMCaseError
from lcm.regime import Regime
from lcm.solvers import BQSEGM, GridSearch
from tests.test_models import bqsegm_medicaid_toy as toy

_LIQUID = np.linspace(0.1, 30.0, 120)
_INTERIOR = (_LIQUID > 2.0) & (_LIQUID < 22.0)


def _solve(
    variant: str, params: dict, *, n_consumption: int = 120
) -> Mapping[int, Mapping[str, object]]:
    """Solve the Medicaid toy on the shared comparison grids."""
    model = toy.build_model(
        variant=variant,
        n_liquid=120,
        liquid_max=30.0,
        n_savings=150,
        savings_max=22.0,
        n_consumption=n_consumption,
    )
    return model.solve(params=params, log_level="off")


def _last_alive_period(solution: Mapping[int, Mapping[str, object]]) -> int:
    return max(period for period in solution if "alive" in solution[period])


def test_bqsegm_matches_brute_through_a_jump_with_smooth_continuation():
    """At the last working age the case-piece solve equals brute across assets.

    The continuation is the smooth terminal bequest, so the only non-smoothness is
    the within-period Medicaid jump — which BQSEGM resolves exactly. Agreement holds
    on the asset interior including a neighbourhood of the limit.
    """
    params = toy.build_params()
    bqsegm = _solve("bqsegm", params)
    brute = _solve("brute", params, n_consumption=1500)
    period = _last_alive_period(brute)
    np.testing.assert_allclose(
        np.asarray(bqsegm[period]["alive"])[_INTERIOR],
        np.asarray(brute[period]["alive"])[_INTERIOR],
        atol=2e-2,
        rtol=5e-3,
    )


def test_bqsegm_matches_brute_multiperiod_without_a_value_jump():
    """With equal subsidies (no jump) the case-piece solve equals brute every age.

    This isolates the multi-period EGM propagation through the masking-and-envelope
    merge from the value-jump continuation: the two cases coincide, so the merged
    solution must track brute across the whole horizon.
    """
    params = toy.build_params(subsidy_high=0.5, subsidy_low=0.5)
    bqsegm = _solve("bqsegm", params)
    brute = _solve("brute", params, n_consumption=1500)
    for period in brute:
        if "alive" not in brute[period] or "alive" not in bqsegm[period]:
            continue
        np.testing.assert_allclose(
            np.asarray(bqsegm[period]["alive"])[_INTERIOR],
            np.asarray(brute[period]["alive"])[_INTERIOR],
            atol=2e-2,
            rtol=5e-3,
            err_msg=f"period={period}",
        )


def test_bqsegm_reproduces_the_medicaid_value_drop_at_the_boundary():
    """Value drops as assets cross the Medicaid limit, matching the brute oracle.

    Just below the limit the agent receives the larger subsidy, so the value is
    higher than just above; the otherwise side owns the exact boundary.
    """
    params = toy.build_params()
    bqsegm = _solve("bqsegm", params)
    brute = _solve("brute", params, n_consumption=1500)
    period = _last_alive_period(brute)
    below = np.argmin(np.abs(_LIQUID - 7.5))
    above = np.argmin(np.abs(_LIQUID - 8.5))
    bqsegm_alive = np.asarray(bqsegm[period]["alive"])
    brute_alive = np.asarray(brute[period]["alive"])
    bqsegm_drop = float(bqsegm_alive[below] - bqsegm_alive[above])
    brute_drop = float(brute_alive[below] - brute_alive[above])
    assert bqsegm_drop > 0.0
    np.testing.assert_allclose(bqsegm_drop, brute_drop, atol=2e-2)


def test_bqsegm_rejects_a_piece_with_a_hidden_where():
    """A smooth piece hiding `jnp.where` fails the smoothness gate at model build."""

    @lcm.case_boundary(
        lcm.boundary("liquid", "limit", equality="otherwise", kind="jump")
    )
    def predicate(liquid, limit):
        return liquid < limit

    @lcm.piece("subsidy", when=predicate)
    def subsidy_when(liquid, limit):
        return jnp.where(liquid < limit, 1.0, 0.0)

    @lcm.piece("subsidy", otherwise=predicate)
    def subsidy_otherwise():
        return jnp.asarray(0.0)

    grid = LinSpacedGrid(start=0.1, stop=20.0, n_points=40)
    alive = Regime(
        actions={"consumption": LinSpacedGrid(start=0.1, stop=20.0, n_points=40)},
        states={"liquid": grid},
        state_transitions={
            "liquid": {"alive": toy.next_liquid, "dead": toy.next_liquid}
        },
        constraints={"feasible": toy.feasible},
        transition={
            "alive": MarkovTransition(toy.prob_stay_alive),
            "dead": MarkovTransition(toy.prob_die),
        },
        functions={
            "utility": toy.utility,
            "predicate": predicate,
            "subsidy_when": subsidy_when,
            "subsidy_otherwise": subsidy_otherwise,
            "coh": toy.coh,
        },
        active=lambda age: age < 1.0,
        solver=BQSEGM(savings_grid=LinSpacedGrid(start=0.0, stop=20.0, n_points=40)),
    )
    dead = Regime(
        transition=None,
        states={"liquid": grid},
        functions={"utility": toy.bequest},
        active=lambda age: age >= 1.0,
        solver=GridSearch(),
    )
    with pytest.raises(BQSEGMCaseError, match="smoothness gate"):
        Model(
            regimes={"alive": alive, "dead": dead},
            ages=AgeGrid(start=0, stop=1, step="Y"),
            regime_id_class=toy.RegimeId,
        )
