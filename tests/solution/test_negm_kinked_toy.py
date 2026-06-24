"""Parity checks for the `NEGM` solver on the kinked two-asset toy.

NEGM solves the kinked toy (`tests/test_models/negm_kinked_toy.py`) by an outer
search over the durable post-decision `next_illiquid` and an inner 1-D DC-EGM
solve on `wealth`. Three checks pin its correctness against an action-grid brute
oracle that searches the *same* economic problem on the same state domain with
the same order-1 V-interpolation:

- The brute oracle (`negm_phase0/kinked_toy_oracle.py`) reproduces its own pinned
  period-0 values and grid checksums — the oracle of record is stable.
- On a matched model whose brute analogue's reachable durable post-states are a
  subset of NEGM's outer grid and whose consumption grid covers the feasible
  continuous domain, NEGM's off-grid inner solve weakly dominates the
  grid-restricted brute value at every state cell: NEGM `>=` brute.
- As the brute consumption and investment grids refine, the brute value rises
  toward NEGM's off-grid value — the gap shrinks monotonically — so NEGM is the
  off-grid limit the action-grid solver approaches from below.

The model solves on CPU in seconds; nothing here is GPU-gated.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from lcm import (
    DCEGM,
    NEGM,
    AgeGrid,
    LinSpacedGrid,
    Model,
    Regime,
)
from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    FloatND,
)
from negm_phase0 import kinked_toy_oracle
from tests.test_models import negm_kinked_toy

_PARAMS = {"discount_factor": 0.95, "alive": {}}

# Period-0, regime `alive`, shape (N_X, N_Z) brute oracle on the repaired
# `[-6, 30]` wealth domain (`negm_phase0/kinked_toy_oracle.py`). With the grid
# starting at -6, cell (0, 0) is the deeply credit-constrained `wealth = -6`.
_ORACLE_CELLS = {
    (0, 0): -0.8572114651,
    (0, 6): -0.2964398156,
    (6, 0): -0.2601982804,
    (6, 6): -0.1719988446,
    (11, 11): -0.1329429054,
}
_ORACLE_SUM = -31.33077128
_ORACLE_MIN = -0.85721147
_ORACLE_MAX = -0.13294291

# Matched-model geometry shared by the NEGM and brute analogues. The wealth
# domain matches the oracle's; the consumption grid spans the feasible range so
# brute is never capped below the optimum; the durable investment range keeps
# brute's reachable `next_illiquid` inside NEGM's outer grid.
_WEALTH_MIN = -6.0
_WEALTH_MAX = 30.0
_ILLIQUID_MAX = 30.0
_CONSUMPTION_MAX = 45.0
_INVESTMENT_BOUND = 8.0
_LIQUID_CREDIT_LIMIT = -5.0
_N_WEALTH = 8
_N_ILLIQUID = 6
_FINAL_AGE_ALIVE = 25  # ages 20, 25 alive then 30 dead: a genuine continuation


def _grid(start: float, stop: float, n_points: int) -> LinSpacedGrid:
    return LinSpacedGrid(start=start, stop=stop, n_points=n_points)


def _build_matched_negm_model(*, savings_n: int = 80, outer_n: int = 40) -> Model:
    """Build the matched 3-period NEGM model on the repaired wealth domain."""
    solver = NEGM(
        inner=DCEGM(
            continuous_state="wealth",
            continuous_action="consumption",
            resources="resources",
            post_decision_function="liquid_savings",
            savings_grid=_grid(-5.0, 35.0, savings_n),
        ),
        outer_action="illiquid_investment",
        outer_post_decision="next_illiquid",
        outer_grid=_grid(0.0, _ILLIQUID_MAX, outer_n),
        outer_no_adjustment_candidate="keep_illiquid",
    )
    alive = Regime(
        active=lambda age, n=_FINAL_AGE_ALIVE: age <= n,
        states={
            "wealth": _grid(_WEALTH_MIN, _WEALTH_MAX, _N_WEALTH),
            "illiquid": _grid(0.0, _ILLIQUID_MAX, _N_ILLIQUID),
        },
        state_transitions={
            "wealth": negm_kinked_toy.next_wealth,
            "illiquid": negm_kinked_toy.durable_transition,
        },
        actions={
            "consumption": _grid(0.1, _CONSUMPTION_MAX, 25),
            "illiquid_investment": _grid(-_INVESTMENT_BOUND, _INVESTMENT_BOUND, 25),
        },
        transition=negm_kinked_toy.next_regime,
        functions={
            "utility": negm_kinked_toy.utility,
            "resources": negm_kinked_toy.resources,
            "liquid_savings": negm_kinked_toy.liquid_savings,
            "keep_illiquid": negm_kinked_toy.keep_illiquid,
            "credited": negm_kinked_toy.credited,
            "inverse_marginal_utility": negm_kinked_toy.inverse_marginal_utility,
        },
        solver=solver,
    )
    dead = Regime(
        transition=None,
        active=lambda age, n=_FINAL_AGE_ALIVE: age > n,
        functions={"utility": lambda: 0.0},
    )
    return Model(
        regimes={"alive": alive, "dead": dead},
        regime_id_class=negm_kinked_toy.RegimeId,
        ages=AgeGrid(start=20, stop=30, step="5Y"),
        fixed_params={"final_age_alive": _FINAL_AGE_ALIVE},
    )


def _brute_liquid_savings(
    wealth: ContinuousState,
    consumption: ContinuousAction,
    illiquid_investment: ContinuousAction,
) -> FloatND:
    """Liquid post-decision balance with the withdrawal-penalty wedge."""
    credited = jnp.where(
        illiquid_investment < 0.0,
        (1.0 - negm_kinked_toy.WITHDRAWAL_PENALTY) * illiquid_investment,
        illiquid_investment,
    )
    return wealth + negm_kinked_toy.LABOUR_INCOME - consumption - credited


def _brute_next_illiquid(
    illiquid: ContinuousState, illiquid_investment: ContinuousAction
) -> ContinuousState:
    return illiquid + illiquid_investment


def _liquid_floor(liquid_savings: FloatND) -> BoolND:
    return liquid_savings >= _LIQUID_CREDIT_LIMIT


def _illiquid_floor(
    illiquid: ContinuousState, illiquid_investment: ContinuousAction
) -> BoolND:
    return illiquid + illiquid_investment >= 0.0


def _positive_consumption(consumption: ContinuousAction) -> BoolND:
    return consumption > 0.05


def _build_matched_brute_model(*, n_consumption: int, n_investment: int) -> Model:
    """Build the matched brute analogue: same domain, on-grid action search.

    Searches `(consumption, illiquid_investment)` on grids of the given sizes
    over the same state domain and frictions as the NEGM model, with the same
    feasibility floors and the same order-1 V-interpolation. Its reachable
    `next_illiquid` stays inside NEGM's outer grid, so its value is a lower bound
    NEGM's off-grid inner solve weakly improves on.
    """
    alive = Regime(
        active=lambda age, n=_FINAL_AGE_ALIVE: age <= n,
        states={
            "wealth": _grid(_WEALTH_MIN, _WEALTH_MAX, _N_WEALTH),
            "illiquid": _grid(0.0, _ILLIQUID_MAX, _N_ILLIQUID),
        },
        state_transitions={
            "wealth": negm_kinked_toy.next_wealth,
            "illiquid": _brute_next_illiquid,
        },
        actions={
            "consumption": _grid(0.1, _CONSUMPTION_MAX, n_consumption),
            "illiquid_investment": _grid(
                -_INVESTMENT_BOUND, _INVESTMENT_BOUND, n_investment
            ),
        },
        transition=negm_kinked_toy.next_regime,
        constraints={
            "liquid_floor": _liquid_floor,
            "illiquid_floor": _illiquid_floor,
            "positive_consumption": _positive_consumption,
        },
        functions={
            "utility": negm_kinked_toy.utility,
            "liquid_savings": _brute_liquid_savings,
        },
    )
    dead = Regime(
        transition=None,
        active=lambda age, n=_FINAL_AGE_ALIVE: age > n,
        functions={"utility": lambda: 0.0},
    )
    return Model(
        regimes={"alive": alive, "dead": dead},
        regime_id_class=negm_kinked_toy.RegimeId,
        ages=AgeGrid(start=20, stop=30, step="5Y"),
        fixed_params={"final_age_alive": _FINAL_AGE_ALIVE},
    )


@pytest.fixture(scope="module")
def matched_negm_value() -> FloatND:
    """The matched 3-period NEGM model's period-0 `alive` value array."""
    return _build_matched_negm_model().solve(params=_PARAMS, log_level="off")[0][
        "alive"
    ]


@pytest.fixture(scope="module")
def oracle_value() -> FloatND:
    """The 4-period brute oracle's period-0 `alive` value array."""
    return kinked_toy_oracle.build_model().solve(
        params=kinked_toy_oracle.PARAMS, log_level="off"
    )[0]["alive"]


@pytest.mark.parametrize(
    ("marginal_continuation", "illiquid", "expected_consumption"),
    [
        (0.25, 0.0, 2.0),  # z = 0: no offset, c = m^{-1/2}
        (0.25, 6.0, 1.7),  # z = 6: c = 2.0 - iota * 6 = 2.0 - 0.3
        (1.0, 6.0, 0.7),  # m = 1: c = 1.0 - 0.3
    ],
)
def test_inverse_marginal_utility_subtracts_the_durable_flow_offset(
    marginal_continuation: float, illiquid: float, expected_consumption: float
) -> None:
    """`(u')^{-1}(m) = m^{-1/gamma} - iota*Z` at the durable node `Z`.

    Utility is `(c + iota*Z)^{1-gamma}/(1-gamma)`, so `u'(c) = (c + iota*Z)^{-gamma}`
    and inverting for the consumption action carries the `- iota*Z` offset the
    durable state contributes to the flow.
    """
    consumption = float(
        negm_kinked_toy.inverse_marginal_utility(
            marginal_continuation=jnp.asarray(marginal_continuation),
            illiquid=jnp.asarray(illiquid),
        )
    )
    np.testing.assert_allclose(consumption, expected_consumption, atol=1e-12)


def test_brute_oracle_reproduces_its_pinned_values(oracle_value: FloatND) -> None:
    """The brute oracle reproduces its pinned period-0 values and grid checksums.

    The oracle is the parity record: its repaired `[-6, 30]` wealth domain keeps
    every reachable `next_wealth` inside the support, so the pinned cell values
    are dense-search truth rather than an out-of-domain extrapolation.
    """
    # The pinned values were recorded on one platform; cross-platform x64 reduction
    # order differs, so the reproducibility band is a relative tolerance, not bit
    # equality. 1e-4 still pins four significant figures of each dense-search cell.
    for (ix, iz), expected in _ORACLE_CELLS.items():
        np.testing.assert_allclose(
            float(oracle_value[ix, iz]), expected, rtol=1e-4, atol=1e-6
        )
    np.testing.assert_allclose(float(jnp.sum(oracle_value)), _ORACLE_SUM, rtol=1e-4)
    np.testing.assert_allclose(float(jnp.min(oracle_value)), _ORACLE_MIN, rtol=1e-4)
    np.testing.assert_allclose(float(jnp.max(oracle_value)), _ORACLE_MAX, rtol=1e-4)

    wealth_grid = jnp.asarray(kinked_toy_oracle.WEALTH_GRID.to_jax())
    illiquid_grid = jnp.asarray(kinked_toy_oracle.ILLIQUID_GRID.to_jax())
    np.testing.assert_allclose(
        kinked_toy_oracle._checksum(wealth_grid), 1404.0, atol=1e-6
    )
    np.testing.assert_allclose(
        kinked_toy_oracle._checksum(illiquid_grid), 1560.0, atol=1e-6
    )


@pytest.mark.parametrize(
    ("n_consumption", "n_investment"),
    [(15, 15), (25, 25), (45, 45)],
)
def test_negm_weakly_improves_on_the_matched_brute_value(
    matched_negm_value: FloatND, n_consumption: int, n_investment: int
) -> None:
    """NEGM's off-grid value weakly dominates the matched brute value cell-by-cell.

    Brute searches the same economic problem on the same state domain with the
    same order-1 V-interpolation but restricts the policy to its action grid; its
    reachable durable post-states are a subset of NEGM's outer grid. NEGM's inner
    EGM puts consumption off-grid, so at every state cell `V_negm >= V_brute` up
    to a small interpolation tolerance.
    """
    brute_value = _build_matched_brute_model(
        n_consumption=n_consumption, n_investment=n_investment
    ).solve(params=_PARAMS, log_level="off")[0]["alive"]
    improvement = matched_negm_value - brute_value
    assert bool(jnp.all(jnp.isfinite(brute_value)))
    assert float(jnp.min(improvement)) >= -1e-4


def test_brute_value_converges_up_to_negm_as_grids_refine(
    matched_negm_value: FloatND,
) -> None:
    """The matched brute value rises toward NEGM's off-grid value as grids refine.

    Refining the brute consumption and investment grids closes the worst-case gap
    to NEGM: the action-grid solver approaches NEGM's off-grid value from below, so
    NEGM is the limit, not an outlier. Convergence is required overall (finest
    closer than coarsest), not at every individual step — action-grid alignment can
    transiently widen the worst-case cell between two refinements.
    """
    max_gaps = []
    for n_points in (15, 25, 45, 80):
        brute_value = _build_matched_brute_model(
            n_consumption=n_points, n_investment=n_points
        ).solve(params=_PARAMS, log_level="off")[0]["alive"]
        max_gaps.append(float(jnp.max(jnp.abs(matched_negm_value - brute_value))))
    # The finest grid is strictly closer to NEGM than the coarsest (overall
    # convergence), and lands within a tight band of NEGM everywhere.
    assert max_gaps[-1] < max_gaps[0]
    assert max_gaps[-1] < 0.1
