"""A folded IID shock that only a transition reads is drawn there and never stored.

Costs realized *after* the period's choices are computed from the draw `next_xi`
inside the wealth law. Nothing within the period reads the lagged state `xi`, so
the value function does not depend on it. Declaring the shock `fold=True` then
removes its axis from every stored value and from the period's state space: the
source's continuation still sums over the draw's nodes with the process's own
weights, against a target value that has no `xi` axis.

The oracle is a literal backward induction over every wealth node, every
consumption node and every draw node, with linear interpolation in wealth and
linear extrapolation past the grid ends, which is what the solver does.

A target that does read the folded shock within its period has a value that
depends on the realized node, so a draw taken by the source and a value averaged
over that draw would be correlated. That combination is refused.
"""

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from lcm import (
    AgeGrid,
    ByAge,
    IrregSpacedGrid,
    LinSpacedGrid,
    Model,
    NormalIIDProcess,
    Regime,
    Transition,
    categorical,
)
from lcm.exceptions import ModelInitializationError
from lcm.transition import StochasticTransition
from lcm.typing import ContinuousAction, ContinuousState, FloatND, ScalarInt
from tests.conftest import X64_ENABLED

DISCOUNT_FACTOR = 0.9
LAST_ALIVE_AGE = 2
AGES = AgeGrid(start=0, inclusive_stop=3, step="Y")
WEALTH_GRID = LinSpacedGrid(start=0.0, stop=12.0, n_points=25)
CONSUMPTION_NODES = (0.5, 1.0, 1.5, 2.0, 3.0)
N_XI = 5
_ATOL = 1e-10 if X64_ENABLED else 1e-5


@categorical(ordered=False)
class _RegimeId:
    alive: ScalarInt
    dead: ScalarInt


def _xi(*, fold: bool) -> NormalIIDProcess:
    return NormalIIDProcess(
        n_points=N_XI, gauss_hermite=True, mu=0.0, sigma=1.0, fold=fold
    )


def _cost(next_xi: ContinuousState) -> FloatND:
    return 0.4 * jnp.exp(0.5 * next_xi)


def _next_wealth(
    *, wealth: ContinuousState, consumption: ContinuousAction, cost: FloatND
) -> ContinuousState:
    return wealth - consumption - cost + 1.0


def _utility(consumption: ContinuousAction) -> FloatND:
    return jnp.log(consumption)


def _utility_reading_xi(
    *, consumption: ContinuousAction, xi: ContinuousState
) -> FloatND:
    return jnp.log(consumption) + 0.1 * xi


def _bequest(wealth: ContinuousState) -> FloatND:
    return jnp.log(wealth + 11.0)


def _probability_alive(age: float) -> FloatND:
    return jnp.where(age < LAST_ALIVE_AGE, 0.8, 0.0)


def _probability_dead(age: float) -> FloatND:
    return jnp.where(age < LAST_ALIVE_AGE, 0.2, 1.0)


def _model(*, fold: bool, alive_reads_xi: bool = False) -> Model:
    alive = Regime(
        actions={
            "consumption": IrregSpacedGrid(points=CONSUMPTION_NODES),
        },
        states={"wealth": WEALTH_GRID, "xi": _xi(fold=fold)},
        state_transitions={"wealth": _next_wealth},
        functions={
            "utility": _utility_reading_xi if alive_reads_xi else _utility,
            "cost": _cost,
        },
    )
    dead = Regime(
        states={"wealth": WEALTH_GRID},
        functions={"utility": _bequest},
    )
    return Model(
        regimes={"alive": alive, "dead": dead},
        edges={
            "alive": Transition(
                targets={"alive": (0, 1), "dead": (0, 1, 2)},
                law=ByAge.until(
                    stop_age_exclusive=LAST_ALIVE_AGE + 1,
                    law={
                        "alive": StochasticTransition(func=_probability_alive),
                        "dead": StochasticTransition(func=_probability_dead),
                    },
                    then="dead",
                ),
            )
        },
        ages=AGES,
        regime_id_class=_RegimeId,
        initial_nodes={0: "alive"},
    )


def _params() -> dict:
    return {"discount_factor": DISCOUNT_FACTOR}


def _interp_extrapolating(*, x: float, nodes: np.ndarray, values: np.ndarray) -> float:
    """Piecewise-linear interpolation, extended linearly beyond both ends."""
    i = int(np.clip(np.searchsorted(nodes, x, side="right") - 1, 0, len(nodes) - 2))
    slope = (values[i + 1] - values[i]) / (nodes[i + 1] - nodes[i])
    return float(values[i] + slope * (x - nodes[i]))


def _reference_alive_values() -> dict[int, np.ndarray]:
    """Literal backward induction of `alive`, keyed by period, indexed by wealth."""
    xi = _xi(fold=False)
    xi_nodes = np.asarray(xi.to_jax(), dtype=np.float64)
    xi_weights = np.asarray(xi.get_transition_probs(), dtype=np.float64)[0]
    wealth_nodes = np.asarray(WEALTH_GRID.to_jax(), dtype=np.float64)
    dead_V = np.log(wealth_nodes + 11.0)

    values: dict[int, np.ndarray] = {}
    next_alive_V: np.ndarray | None = None
    for period in (2, 1, 0):
        V = np.empty(len(wealth_nodes))
        for i_w in range(len(wealth_nodes)):
            best = -np.inf
            for c in CONSUMPTION_NODES:
                expected = 0.0
                for k in range(N_XI):
                    landing = (
                        wealth_nodes[i_w] - c - 0.4 * np.exp(0.5 * xi_nodes[k]) + 1.0
                    )
                    v_dead = _interp_extrapolating(
                        x=landing, nodes=wealth_nodes, values=dead_V
                    )
                    if period < LAST_ALIVE_AGE:
                        assert next_alive_V is not None
                        v_alive = _interp_extrapolating(
                            x=landing, nodes=wealth_nodes, values=next_alive_V
                        )
                        cont = 0.8 * v_alive + 0.2 * v_dead
                    else:
                        cont = v_dead
                    expected += xi_weights[k] * cont
                best = max(best, np.log(c) + DISCOUNT_FACTOR * expected)
            V[i_w] = best
        values[period] = V
        next_alive_V = V
    return values


def _alive_values(model: Model) -> dict[int, np.ndarray]:
    solution = model.solve(params=_params(), log_level="off")
    return {
        period: np.asarray(arrays["alive"])
        for period, arrays in solution.values.items()
        if "alive" in arrays
    }


def test_the_folded_transition_only_shock_is_not_a_state_of_the_regime() -> None:
    model = _model(fold=True)

    assert model.state_names(regime_name="alive") == ("wealth",)


def test_the_unfolded_shock_stays_a_state_of_the_regime() -> None:
    model = _model(fold=False)

    assert set(model.state_names(regime_name="alive")) == {"wealth", "xi"}


def test_the_folded_value_has_no_shock_axis_in_any_period() -> None:
    shapes = {period: V.shape for period, V in _alive_values(_model(fold=True)).items()}

    assert shapes == dict.fromkeys((0, 1, 2), (len(WEALTH_GRID.to_jax()),))


@pytest.mark.parametrize("period", [0, 1, 2])
def test_the_folded_value_matches_the_literal_backward_induction(period: int) -> None:
    V = _alive_values(_model(fold=True))[period]

    np.testing.assert_allclose(V, _reference_alive_values()[period], rtol=0, atol=_ATOL)


@pytest.mark.parametrize("period", [0, 1, 2])
def test_the_unfolded_value_matches_the_literal_backward_induction(
    period: int,
) -> None:
    """The oracle reproduces the model that stores the shock axis."""
    model = _model(fold=False)
    order = model.state_names(regime_name="alive")
    V = np.transpose(
        _alive_values(model)[period], [order.index("wealth"), order.index("xi")]
    )
    reference = _reference_alive_values()[period]

    np.testing.assert_allclose(
        V, np.broadcast_to(reference[:, None], V.shape), rtol=0, atol=_ATOL
    )


@pytest.mark.parametrize("period", [0, 1, 2])
def test_the_folded_value_equals_every_slice_of_the_unfolded_value(
    period: int,
) -> None:
    model = _model(fold=False)
    unfolded = _alive_values(model)[period]
    order = model.state_names(regime_name="alive")
    unfolded = np.transpose(unfolded, [order.index("wealth"), order.index("xi")])
    folded = _alive_values(_model(fold=True))[period]

    np.testing.assert_allclose(
        unfolded, np.broadcast_to(folded[:, None], unfolded.shape), rtol=0, atol=_ATOL
    )


def test_the_folded_policy_equals_the_unfolded_policy() -> None:
    """Simulated consumption agrees subject by subject in the first period.

    Both models start every subject at the same wealth; consumption at age 0 is
    chosen before any draw, so it depends only on the stored continuation values.
    """
    wealth = jnp.asarray(np.asarray(WEALTH_GRID.to_jax())[::3])
    n = wealth.shape[0]
    consumption = {}
    for fold in (False, True):
        initial_conditions = {
            "wealth": wealth,
            "age": jnp.zeros(n),
            "regime_id": jnp.full(n, _RegimeId.alive),
        }
        if not fold:
            initial_conditions["xi"] = jnp.zeros(n)
        df = (
            _model(fold=fold)
            .simulate(
                params=_params(),
                initial_conditions=initial_conditions,
                log_level="off",
                seed=0,
            )
            .to_dataframe()
            .query("age == 0")
        )
        consumption[fold] = df["consumption"].to_numpy()

    np.testing.assert_array_equal(consumption[True], consumption[False])


def test_an_initial_value_of_the_transition_only_shock_is_accepted_and_unread() -> None:
    """A panel that still carries the shock's column simulates as one without it."""
    n = 4
    initial_conditions = {
        "wealth": jnp.asarray([2.0, 4.0, 6.0, 8.0]),
        "age": jnp.zeros(n),
        "regime_id": jnp.full(n, _RegimeId.alive),
    }
    model = _model(fold=True)
    frames = [
        model.simulate(
            params=_params(),
            initial_conditions=initial_conditions | extra,
            log_level="off",
            seed=0,
        ).to_dataframe()
        for extra in ({}, {"xi": jnp.asarray([-1.0, 0.0, 1.0, 2.0])})
    ]

    pd.testing.assert_frame_equal(frames[0], frames[1])


def test_a_target_reading_its_folded_shock_refuses_a_source_draw() -> None:
    with pytest.raises(ModelInitializationError, match="reads it within its period"):
        _model(fold=True, alive_reads_xi=True)
