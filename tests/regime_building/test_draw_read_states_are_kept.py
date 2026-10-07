"""A shock state whose only reader is its next-period draw stays in the regime.

Costs realized *after* the period's choices are computed from the draws
`next_zeta` / `next_xi` inside the wealth law. Nothing this period reads the
lagged states `zeta` / `xi` directly, yet the draw of `next_zeta` is conditional
on `zeta` (an AR(1) row), and the draw of `next_xi` comes from `xi`'s process.
Reading a process state's draw is therefore a read of the state: the regime
that reads the draw keeps the state, whether it is declared at model or at
regime level. A regime that never reads the draw — the terminal `dead` regime,
valuing only wealth — does not keep it.

The IID state `xi` is kept too, although its lagged value is uninformative: the
engine draws `next_xi` from the state's process, so the axis stays, and the
value is constant along it.
"""

import itertools
from typing import Literal

import jax.numpy as jnp
import numpy as np
import pytest

from lcm import AgeGrid, LinSpacedGrid, Model, categorical
from lcm.processes import NormalIIDProcess, RouwenhorstAR1Process
from lcm.regime import Regime
from lcm.typing import ContinuousAction, ContinuousState, FloatND, ScalarInt
from tests.conftest import X64_ENABLED

DISCOUNT_FACTOR = 0.9
WEALTH_GRID = LinSpacedGrid(start=-10.0, stop=10.0, n_points=81)
CONSUMPTION_NODES = (0.5, 1.0, 1.5, 2.0)
_ATOL = 1e-10 if X64_ENABLED else 1e-5


@categorical(ordered=False)
class _RegimeId:
    active: ScalarInt
    dead: ScalarInt


def _cost(*, next_zeta: ContinuousState, next_xi: ContinuousState) -> FloatND:
    return jnp.exp(0.5 * next_zeta + 0.3 * next_xi)


def _next_wealth(
    *, wealth: ContinuousState, consumption: ContinuousAction, cost: FloatND
) -> ContinuousState:
    return wealth - consumption - cost


def _utility(consumption: ContinuousAction) -> FloatND:
    return jnp.log(consumption)


def _bequest(wealth: ContinuousState) -> FloatND:
    return jnp.log(wealth + 11.0)


def _zeta() -> RouwenhorstAR1Process:
    return RouwenhorstAR1Process(n_points=3, rho=0.9, sigma=(1 - 0.81) ** 0.5, mu=0.0)


def _xi() -> NormalIIDProcess:
    return NormalIIDProcess(n_points=3, gauss_hermite=True, mu=0.0, sigma=1.0)


def _model(*, declared_at: Literal["model", "regime"]) -> Model:
    shocks = {"zeta": _zeta(), "xi": _xi()}
    active = Regime(
        actions={
            "consumption": LinSpacedGrid(
                start=CONSUMPTION_NODES[0],
                stop=CONSUMPTION_NODES[-1],
                n_points=len(CONSUMPTION_NODES),
            )
        },
        states={"wealth": WEALTH_GRID} | (shocks if declared_at == "regime" else {}),
        state_transitions={"wealth": _next_wealth},
        functions={"utility": _utility, "cost": _cost},
    )
    dead = Regime(
        states={"wealth": WEALTH_GRID},
        functions={"utility": _bequest},
    )
    return Model(
        regimes={"active": active, "dead": dead},
        edges={"active": {"dead": 0}},
        ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
        regime_id_class=_RegimeId,
        initial_nodes={0: "active"},
        states=shocks if declared_at == "model" else {},
    )


def _params() -> dict:
    return {"discount_factor": DISCOUNT_FACTOR}


def _reference_V() -> np.ndarray:
    """Scalar-loop value of `active`, indexed `(wealth, zeta, xi)`.

    Consumption is chosen on its grid; the continuation integrates the next
    `zeta` over the Markov row of the lagged `zeta` and the next `xi` over its
    quadrature weights, reading the terminal value by linear interpolation in
    wealth.
    """
    zeta, xi = _zeta(), _xi()
    zeta_nodes = np.asarray(zeta.to_jax())
    zeta_probs = np.asarray(zeta.get_transition_probs())
    xi_nodes = np.asarray(xi.to_jax())
    xi_weights = np.asarray(xi.get_transition_probs())[0]
    wealth_nodes = np.asarray(WEALTH_GRID.to_jax())
    dead_V = np.log(wealth_nodes + 11.0)
    reference = np.empty((len(wealth_nodes), len(zeta_nodes), len(xi_nodes)))
    for i_w, i_z, i_x in itertools.product(
        range(len(wealth_nodes)), range(len(zeta_nodes)), range(len(xi_nodes))
    ):
        reference[i_w, i_z, i_x] = max(
            np.log(c)
            + DISCOUNT_FACTOR
            * sum(
                zeta_probs[i_z, j]
                * xi_weights[k]
                * np.interp(
                    wealth_nodes[i_w]
                    - c
                    - np.exp(0.5 * zeta_nodes[j] + 0.3 * xi_nodes[k]),
                    wealth_nodes,
                    dead_V,
                )
                for j in range(len(zeta_nodes))
                for k in range(len(xi_nodes))
            )
            for c in CONSUMPTION_NODES
        )
    return reference


def _active_V(model: Model) -> np.ndarray:
    """`active`'s value at period 0, axes reordered to `(wealth, zeta, xi)`."""
    solution = model.solve(params=_params(), log_level="off")
    V = np.asarray(solution.values[0]["active"])
    order = model.state_names(regime_name="active")
    return np.transpose(V, [order.index(name) for name in ("wealth", "zeta", "xi")])


@pytest.mark.parametrize("declared_at", ["model", "regime"])
def test_the_draw_reading_regime_keeps_the_shock_states(
    declared_at: Literal["model", "regime"],
) -> None:
    model = _model(declared_at=declared_at)

    assert set(model.state_names(regime_name="active")) == {"wealth", "zeta", "xi"}


def test_a_regime_that_never_reads_the_draw_does_not_keep_the_shock_states() -> None:
    model = _model(declared_at="model")

    assert model.state_names(regime_name="dead") == ("wealth",)
    assert model.pruned_variables["dead"] == frozenset({"zeta", "xi"})
    assert model.pruned_variables["active"] == frozenset()


@pytest.mark.slow
@pytest.mark.parametrize("declared_at", ["model", "regime"])
def test_value_matches_the_scalar_reference(
    declared_at: Literal["model", "regime"],
) -> None:
    V = _active_V(_model(declared_at=declared_at))
    reference = _reference_V()
    # The wealth law leaves the grid below `wealth = 0`, where the solver and
    # `np.interp` extrapolate differently; the reference covers the rest.
    on_grid = np.asarray(WEALTH_GRID.to_jax()) >= 0

    np.testing.assert_allclose(V[on_grid], reference[on_grid], rtol=0, atol=_ATOL)


@pytest.mark.slow
def test_value_is_constant_in_lagged_xi_and_varies_with_lagged_zeta() -> None:
    V = _active_V(_model(declared_at="model"))

    assert np.abs(V - V[:, :, :1]).max() == 0.0
    assert np.abs(V - V[:, :1, :]).max() > 0.1


@pytest.mark.slow
def test_simulation_draws_from_the_kept_states() -> None:
    model = _model(declared_at="model")
    n = 4
    result = model.simulate(
        params=_params(),
        initial_conditions={
            "wealth": jnp.array([2.0, 4.0, 6.0, 8.0]),
            "zeta": jnp.zeros(n),
            "xi": jnp.zeros(n),
            "age": jnp.zeros(n),
            "regime_id": jnp.array([_RegimeId.active] * n),
        },
        log_level="off",
        seed=0,
    )
    df = result.to_dataframe().query('regime_name == "active"')

    assert {"zeta", "xi"} <= set(df.columns)
    assert len(df) == n
