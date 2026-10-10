"""A law `next_wealth = base + shock` averages the target's value over the shock.

The shock is the negative of a medical bill paid after the period's choices; it
reads next period's draws: the health draw, which the target carries as a state,
and a transitory shock that only the transition reads. Declared through
`lcm.AdditiveShockTransition`, the solve averages the target's value function over
the transitory shock once per period, on the merged points `{w_j - shock_k}`, and
reads that average at the base the source carries into the period. With the value
function linear between its grid points the average is linear between the merged
points, so the route is exact: it agrees with interpolating the value at every node
of every draw.

The oracle is a literal backward induction over every wealth node, health state,
consumption node, health draw and shock node, with linear interpolation in
wealth and linear extrapolation past the grid ends.
"""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.grids.coordinates import get_irreg_coordinate
from _lcm.regime_building import Q_and_F
from _lcm.regime_building.ndimage import map_coordinates
from _lcm.regime_building.shock_average import average_over_shock
from lcm import (
    AdditiveShockTransition,
    AgeGrid,
    ByAge,
    DiscreteGrid,
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
from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    DiscreteState,
    FloatND,
    ScalarInt,
)
from tests.conftest import X64_ENABLED

DISCOUNT_FACTOR = 0.9
LAST_ALIVE_AGE = 2
AGES = AgeGrid(start=0, inclusive_stop=3, step="Y")
WEALTH_GRID = LinSpacedGrid(start=0.0, stop=12.0, n_points=25)
CONSUMPTION_NODES = (0.5, 1.0, 1.5, 2.0, 3.0)
N_XI = 5
POOR_BELOW = 4.0
HEALTH_PROBS = np.array([[0.7, 0.3], [0.2, 0.8]])
_ATOL = 1e-10 if X64_ENABLED else 1e-4


@categorical(ordered=False)
class _RegimeId:
    alive: ScalarInt
    dead: ScalarInt


@categorical(ordered=True)
class _Health:
    bad: ScalarInt
    good: ScalarInt


def _xi() -> NormalIIDProcess:
    return NormalIIDProcess(
        n_points=N_XI, gauss_hermite=True, mu=0.0, sigma=1.0, fold=True
    )


def _is_poor(wealth: ContinuousState) -> BoolND:
    return wealth < POOR_BELOW


def _cost(
    *, next_xi: ContinuousState, next_health: DiscreteState, is_poor: BoolND
) -> FloatND:
    return jnp.where(is_poor, 0.2, 0.4) * jnp.exp(0.5 * next_xi) * (1.0 + next_health)


def _cost_reading_wealth(
    *, next_xi: ContinuousState, wealth: ContinuousState
) -> FloatND:
    return 0.01 * wealth * jnp.exp(0.5 * next_xi)


def _negative_cost(cost: FloatND) -> FloatND:
    return -cost


def _before_bill(*, wealth: ContinuousState, consumption: ContinuousAction) -> FloatND:
    return wealth - consumption + 1.0


def _next_wealth(*, before_bill: FloatND, cost: FloatND) -> ContinuousState:
    return before_bill - cost


def _next_health(health: DiscreteState) -> FloatND:
    return jnp.asarray(HEALTH_PROBS)[health]


def _utility(*, consumption: ContinuousAction, health: DiscreteState) -> FloatND:
    return jnp.log(consumption) + 0.3 * health


def _bequest(wealth: ContinuousState) -> FloatND:
    return jnp.log(wealth + 11.0)


def _probability_alive(age: float) -> FloatND:
    return jnp.where(age < LAST_ALIVE_AGE, 0.8, 0.0)


def _probability_dead(age: float) -> FloatND:
    return jnp.where(age < LAST_ALIVE_AGE, 0.2, 1.0)


def _model(*, additive: bool, shock_reads_wealth: bool = False) -> Model:
    law = (
        AdditiveShockTransition(
            base="before_bill",
            shock="negative_cost",
            conditioners={"is_poor": (0, 1)},
        )
        if additive
        else _next_wealth
    )
    alive = Regime(
        actions={"consumption": IrregSpacedGrid(points=CONSUMPTION_NODES)},
        states={
            "wealth": WEALTH_GRID,
            "health": DiscreteGrid(_Health),
            "xi": _xi(),
        },
        state_transitions={
            "wealth": {"alive": law, "dead": law},
            "health": StochasticTransition(func=_next_health),
        },
        functions={
            "utility": _utility,
            "cost": _cost_reading_wealth if shock_reads_wealth else _cost,
            "is_poor": _is_poor,
            "before_bill": _before_bill,
        }
        | ({"negative_cost": _negative_cost} if additive else {}),
    )
    dead = Regime(states={"wealth": WEALTH_GRID}, functions={"utility": _bequest})
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
    """Piecewise-linear interpolation, extended linearly beyond both ends.

    A node holding `-inf` under a nonzero weight makes the read `-inf`.
    """
    i = int(np.clip(np.searchsorted(nodes, x, side="right") - 1, 0, len(nodes) - 2))
    upper = (x - nodes[i]) / (nodes[i + 1] - nodes[i])
    lower = 1.0 - upper
    if (np.isneginf(values[i]) and lower != 0) or (
        np.isneginf(values[i + 1]) and upper != 0
    ):
        return -np.inf
    terms = [w * v for w, v in ((lower, values[i]), (upper, values[i + 1])) if w != 0]
    return float(sum(terms))


def _reference_alive_values() -> dict[int, np.ndarray]:
    """Literal backward induction of `alive`, keyed by period, as `[wealth, health]`."""
    xi = _xi()
    xi_nodes = np.asarray(xi.to_jax(), dtype=np.float64)
    xi_weights = np.asarray(xi.get_transition_probs(), dtype=np.float64)[0]
    wealth_nodes = np.asarray(WEALTH_GRID.to_jax(), dtype=np.float64)
    dead_V = np.log(wealth_nodes + 11.0)

    values: dict[int, np.ndarray] = {}
    next_alive_V: np.ndarray | None = None
    for period in (2, 1, 0):
        V = np.empty((len(wealth_nodes), 2))
        for i_w, wealth in enumerate(wealth_nodes):
            scale = 0.2 if wealth < POOR_BELOW else 0.4
            for health in (0, 1):
                best = -np.inf
                for c in CONSUMPTION_NODES:
                    expected = 0.0
                    for next_health in (0, 1):
                        for k in range(N_XI):
                            cost = (
                                scale * np.exp(0.5 * xi_nodes[k]) * (1.0 + next_health)
                            )
                            landing = wealth - c + 1.0 - cost
                            v_dead = _interp_extrapolating(
                                x=landing, nodes=wealth_nodes, values=dead_V
                            )
                            if period < LAST_ALIVE_AGE:
                                assert next_alive_V is not None
                                v_alive = _interp_extrapolating(
                                    x=landing,
                                    nodes=wealth_nodes,
                                    values=next_alive_V[:, next_health],
                                )
                                cont = 0.8 * v_alive + 0.2 * v_dead
                            else:
                                cont = v_dead
                            expected += (
                                HEALTH_PROBS[health, next_health] * xi_weights[k] * cont
                            )
                    best = max(
                        best, np.log(c) + 0.3 * health + DISCOUNT_FACTOR * expected
                    )
                V[i_w, health] = best
        values[period] = V
        next_alive_V = V
    return values


def _alive_values(model: Model) -> dict[int, np.ndarray]:
    solution = model.solve(params=_params(), log_level="off")
    order = model.state_names(regime_name="alive")
    return {
        period: np.transpose(
            np.asarray(arrays["alive"]),
            [order.index("wealth"), order.index("health")],
        )
        for period, arrays in solution.values.items()
        if "alive" in arrays
    }


def _read_average(
    *, knots: np.ndarray, averaged: np.ndarray, at: np.ndarray
) -> np.ndarray:
    coordinate = get_irreg_coordinate(value=jnp.asarray(at), points=jnp.asarray(knots))
    return np.asarray(
        jax.vmap(
            lambda c: map_coordinates(input=jnp.asarray(averaged), coordinates=[c])
        )(coordinate)
    )


@pytest.mark.parametrize(
    ("shocks", "values"),
    [
        pytest.param(
            (-0.3, -1.1, -1.1, -2.5, -7.0),
            np.log(np.linspace(0.0, 12.0, 25) + 2.0),
            id="duplicate-shocks-and-landings-below-the-grid",
        ),
        pytest.param(
            (0.0, -0.5, -0.5, -0.5, -1.0),
            np.where(np.arange(25) < 2, -np.inf, np.linspace(0.0, 12.0, 25) ** 0.5),
            id="infeasible-low-nodes",
        ),
    ],
)
def test_the_average_on_merged_points_equals_interpolating_every_node(
    *, shocks: tuple[float, ...], values: np.ndarray
) -> None:
    """Reading the average at any point equals averaging the reads at each shock."""
    points = np.linspace(0.0, 12.0, 25)
    weights = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
    knots, averaged = average_over_shock(
        values=jnp.asarray(values),
        points=jnp.asarray(points),
        coordinate=lambda x: get_irreg_coordinate(value=x, points=jnp.asarray(points)),
        shocks=jnp.asarray(shocks),
        weights=jnp.asarray(weights),
    )
    at = np.concatenate(
        [
            np.random.default_rng(seed=0).uniform(-10.0, 25.0, size=400),
            points + 0.3,
            points + 1.1,
        ]
    )
    expected = np.array(
        [
            sum(
                w * _interp_extrapolating(x=z + s, nodes=points, values=values)
                for w, s in zip(weights, shocks, strict=True)
            )
            for z in at
        ]
    )

    np.testing.assert_allclose(
        _read_average(knots=np.asarray(knots), averaged=np.asarray(averaged), at=at),
        expected,
        rtol=1e-12 if X64_ENABLED else 1e-5,
        atol=_ATOL,
    )


@pytest.mark.parametrize(
    ("additive", "expected"),
    [(True, ["next_health"]), (False, ["next_health", "next_xi"])],
)
def test_only_the_plain_law_maps_the_continuation_over_the_shock(
    *,
    additive: bool,
    expected: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The additive shock transition averages the shock into the value it reads.

    The health draw keeps its node axis toward both targets: `alive` stores
    health, and toward `dead` the transition draws it with probabilities that
    depend on the source's health.
    """
    observed: list[list[str]] = []
    original = Q_and_F._build_target_continuation

    def observe(**kwargs: Any) -> Any:
        result = original(**kwargs)
        observed.append(sorted(result.lottery_axis_names))
        return result

    monkeypatch.setattr(Q_and_F, "_build_target_continuation", observe)
    _model(additive=additive).solve(params=_params(), log_level="off")

    assert observed
    assert all(axes == expected for axes in observed)


@pytest.mark.parametrize("period", [0, 1, 2])
def test_the_additive_shock_value_matches_the_literal_backward_induction(
    period: int,
) -> None:
    V = _alive_values(_model(additive=True))[period]

    np.testing.assert_allclose(V, _reference_alive_values()[period], rtol=0, atol=_ATOL)


@pytest.mark.parametrize("period", [0, 1, 2])
def test_the_plain_law_value_matches_the_literal_backward_induction(
    period: int,
) -> None:
    """The oracle reproduces the model that interpolates at every node."""
    V = _alive_values(_model(additive=False))[period]

    np.testing.assert_allclose(V, _reference_alive_values()[period], rtol=0, atol=_ATOL)


def test_the_additive_shock_policy_equals_the_plain_law_policy() -> None:
    """Simulated consumption agrees subject by subject in the first period."""
    wealth = jnp.asarray(np.asarray(WEALTH_GRID.to_jax())[::3])
    n = wealth.shape[0]
    consumption = {}
    for additive in (False, True):
        df = (
            _model(additive=additive)
            .simulate(
                params=_params(),
                initial_conditions={
                    "wealth": wealth,
                    "health": jnp.zeros(n, dtype=jnp.int32),
                    "age": jnp.zeros(n),
                    "regime_id": jnp.full(n, _RegimeId.alive),
                },
                log_level="off",
                seed=0,
            )
            .to_dataframe()
            .query("age == 0")
        )
        consumption[additive] = df["consumption"].to_numpy()

    np.testing.assert_array_equal(consumption[True], consumption[False])


def test_a_shock_reading_a_source_state_outside_its_conditioners_is_refused() -> None:
    with pytest.raises(ModelInitializationError, match="reads the source's"):
        _model(additive=True, shock_reads_wealth=True)
