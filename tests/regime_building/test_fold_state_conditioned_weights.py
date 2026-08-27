"""A folded shock whose `sigma` is state-conditioned uses per-category weights.

`fold=True` integrates a shock's node axis into the stored value with the
process's own quadrature. When the process's `sigma` is `StateConditioned`, that
quadrature is not one row but one row per category of the conditioning state, so
the fold's weights are indexed by that state's integer code. Averaging every
category against a single row would price a low-variance subject with a
high-variance distribution — a difference that does not shrink as the node grid
refines.

Every expected value below is computed from the normal CDF directly, on nodes the
test asserts rather than assumes, and cross-checked against two unconditioned
models that fold the same two `sigma` values.
"""

import math

import jax.numpy as jnp
import numpy as np
import pytest

from lcm import (
    AgeGrid,
    DiscreteGrid,
    Model,
    NormalIIDProcess,
    Regime,
    categorical,
    fixed_transition,
)
from lcm.processes import StateConditioned
from lcm.typing import DiscreteAction, FloatND, ScalarInt
from tests.conftest import DECIMAL_PRECISION

N_POINTS = 5
N_STD = 3.0
MU = 0.0
SIGMA_BY_RISK = {"low": 0.05, "high": 1.0}
OUTSIDE_OPTION = 0.2
DISCOUNT_FACTOR = 0.9
AGES = AgeGrid(start=0, stop=2, step="Y")


@categorical(ordered=False)
class RegimeId:
    period0: ScalarInt
    terminal: ScalarInt


@categorical(ordered=True)
class RiskType:
    low: ScalarInt  # code 0
    high: ScalarInt  # code 1


@categorical(ordered=True)
class Work:
    leisure: ScalarInt  # code 0
    work: ScalarInt  # code 1


def _next_regime() -> ScalarInt:
    return RegimeId.terminal


def _utility(wage_shock: FloatND, work: DiscreteAction) -> FloatND:
    """Working pays the realized shock; leisure pays a fixed outside option."""
    return jnp.where(work == 1, wage_shock, OUTSIDE_OPTION)


def _standard_normal_cdf(z: float) -> float:
    """`Phi(z)`, from `math.erf` — independent of the production row builder."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _cdf_binned_row(*, nodes: np.ndarray, sigma: float) -> np.ndarray:
    """The CDF-binned probability row on `nodes` for `N(MU, sigma**2)`."""
    edges = 0.5 * (nodes[:-1] + nodes[1:])
    cdf_at_edges = np.array(
        [_standard_normal_cdf((edge - MU) / sigma) for edge in edges]
    )
    return np.concatenate(
        [
            cdf_at_edges[:1],
            np.diff(cdf_at_edges),
            1.0 - cdf_at_edges[-1:],
        ]
    )


def _expected_value(*, nodes: np.ndarray, sigma: float) -> float:
    """`E[max(shock, OUTSIDE_OPTION)]` under the CDF-binned row for `sigma`."""
    payoff = np.maximum(nodes, OUTSIDE_OPTION)
    return float(_cdf_binned_row(nodes=nodes, sigma=sigma) @ payoff)


def _shock(*, sigma: float | StateConditioned) -> NormalIIDProcess:
    return NormalIIDProcess(
        n_points=N_POINTS,
        gauss_hermite=False,
        mu=MU,
        n_std=N_STD,
        sigma=sigma,
        fold=True,
    )


def _conditioned_model() -> Model:
    period0 = Regime(
        transition=_next_regime,
        active=lambda age: age < 1,
        states={
            "risk_type": DiscreteGrid(RiskType),
            "wage_shock": _shock(
                sigma=StateConditioned(on="risk_type", by=SIGMA_BY_RISK)
            ),
        },
        state_transitions={"risk_type": fixed_transition("risk_type")},
        actions={"work": DiscreteGrid(Work)},
        functions={"utility": _utility},
    )
    terminal = Regime(
        transition=None,
        active=lambda age: age >= 1,
        functions={"utility": lambda: jnp.asarray(0.0)},
    )
    return Model(
        regimes={"period0": period0, "terminal": terminal},
        ages=AGES,
        regime_id_class=RegimeId,
    )


def _unconditioned_model(*, sigma: float) -> Model:
    """The same model with one scalar `sigma` and no conditioning state.

    Its nodes are placed by that same `sigma`, so it is a cross-route only for
    the widest category — see `test_high_risk_cell_matches_the_unconditioned_fold`.
    """
    period0 = Regime(
        transition=_next_regime,
        active=lambda age: age < 1,
        states={"wage_shock": _shock(sigma=sigma)},
        actions={"work": DiscreteGrid(Work)},
        functions={"utility": _utility},
    )
    terminal = Regime(
        transition=None,
        active=lambda age: age >= 1,
        functions={"utility": lambda: jnp.asarray(0.0)},
    )
    return Model(
        regimes={"period0": period0, "terminal": terminal},
        ages=AGES,
        regime_id_class=RegimeId,
    )


def _params() -> dict[str, dict[str, dict[str, float]]]:
    return {
        "period0": {"koopmans_aggregator": {"discount_factor": DISCOUNT_FACTOR}},
        "terminal": {},
    }


def _nodes() -> np.ndarray:
    """The fixed common nodes, read off the process rather than assumed."""
    return np.asarray(
        _shock(
            sigma=StateConditioned(on="risk_type", by=SIGMA_BY_RISK)
        ).get_gridpoints()
    )


def test_the_common_nodes_are_placed_by_the_widest_conditioned_sigma() -> None:
    """The oracle rows are binned on these nodes, so the test asserts them."""
    np.testing.assert_array_almost_equal(
        _nodes(), np.array([-3.0, -1.5, 0.0, 1.5, 3.0]), decimal=DECIMAL_PRECISION
    )


@pytest.mark.parametrize("risk", ["low", "high"])
def test_each_category_folds_against_its_own_quadrature_row(risk: str) -> None:
    """The stored value per category is `E[max(shock, 0.2)]` at that `sigma`.

    A single shared row would give both categories the same number; the two
    expected values here differ by more than the whole outside option.
    """
    nodes = _nodes()
    model = _conditioned_model()

    V0 = np.asarray(model.solve(params=_params(), log_level="debug")[0]["period0"])

    assert V0.shape == (len(SIGMA_BY_RISK),)
    np.testing.assert_almost_equal(
        V0[int(getattr(RiskType, risk))],
        _expected_value(nodes=nodes, sigma=SIGMA_BY_RISK[risk]),
        decimal=DECIMAL_PRECISION,
    )


def test_the_two_categories_do_not_share_one_folded_value() -> None:
    """Low-risk subjects take the outside option; high-risk ones do not.

    This is the structural consequence of per-category weights, asserted as a
    decision rather than through the tolerance on the two levels.
    """
    V0 = np.asarray(
        _conditioned_model().solve(params=_params(), log_level="debug")[0]["period0"]
    )

    assert V0[int(RiskType.low)] < V0[int(RiskType.high)]


def test_high_risk_cell_matches_the_unconditioned_fold() -> None:
    """The widest category reproduces the plain scalar-`sigma` fold exactly.

    Both models bin on the same nodes, so this is the same quadrature reached
    through the unconditioned code path — a second route to the same number.
    """
    conditioned = np.asarray(
        _conditioned_model().solve(params=_params(), log_level="debug")[0]["period0"]
    )
    unconditioned = np.asarray(
        _unconditioned_model(sigma=SIGMA_BY_RISK["high"]).solve(
            params=_params(), log_level="debug"
        )[0]["period0"]
    )

    np.testing.assert_almost_equal(
        conditioned[int(RiskType.high)],
        float(unconditioned),
        decimal=DECIMAL_PRECISION,
    )
