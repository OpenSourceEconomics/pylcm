"""A shock read only as its draw `next_xi` is integrated out of the regime.

Costs realized *after* the period's choices are computed from the draw `next_xi`
inside the wealth law. Nothing reads the current value `xi`, so the shock is no
state of the regime: the value function stores no axis for it, and the source's
continuation sums over the draw's nodes with the shock's own weights. This holds
for a `Model(shocks=...)` group and for an IID process declared as a state alike.

The oracle is a literal backward induction over every wealth node, every
consumption node and every draw node, with linear interpolation in wealth and
linear extrapolation past the grid ends, which is what the solver does. A second
oracle is the same model with the shock carried as an ordinary IID state, whose
value is constant across the shock axis.

Utility and constraints are evaluated before the draw, so a read of `next_xi` from
either is refused when the model is built. A read of the current value `xi` gives
the shock an axis.
"""

from typing import Literal

import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal as aaae
from scipy.stats import kstest

from lcm import (
    AgeGrid,
    ByAge,
    ExecutionConfig,
    IrregSpacedGrid,
    LinSpacedGrid,
    Model,
    NormalIIDProcess,
    Regime,
    Transition,
    categorical,
)
from lcm.exceptions import InvalidNameError, ModelInitializationError
from lcm.shocks import GaussHermite, Normal, Product, ShockGroup
from lcm.transition import StochasticTransition
from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    FloatND,
    ReferenceName,
    ScalarInt,
    UserFunction,
)
from lcm_examples import iskhakov_et_al_2017 as iskhakov
from tests.conftest import DECIMAL_PRECISION, X64_ENABLED

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


def _cost_shock() -> ShockGroup:
    return ShockGroup(
        shocks={"xi": Normal(mu=0.0, sigma=1.0)},
        nodes=Product(xi=GaussHermite(N_XI)),
    )


def _xi_state(
    simulated_draws: Literal["continuous", "nodes"] = "continuous",
) -> NormalIIDProcess:
    return NormalIIDProcess(
        n_points=N_XI,
        gauss_hermite=True,
        mu=0.0,
        sigma=1.0,
        simulated_draws=simulated_draws,
    )


def _cost(next_xi: ContinuousState) -> FloatND:
    return 0.4 * jnp.exp(0.5 * next_xi)


def _next_wealth(
    *, wealth: ContinuousState, consumption: ContinuousAction, cost: FloatND
) -> ContinuousState:
    return wealth - consumption - cost + 1.0


def _next_wealth_from_savings_net_of_cost(
    *, savings: FloatND, labor_income: FloatND, interest_rate: float, cost: FloatND
) -> ContinuousState:
    return (1 + interest_rate) * savings + labor_income - cost


def _utility(consumption: ContinuousAction) -> FloatND:
    return jnp.log(consumption)


def _utility_reading_xi(*, consumption: ContinuousAction, xi: FloatND) -> FloatND:
    return jnp.log(consumption) + 0.1 * xi


def _utility_reading_cost(*, consumption: ContinuousAction, cost: FloatND) -> FloatND:
    return jnp.log(consumption) - cost


def _utility_reading_next_xi(
    *, consumption: ContinuousAction, next_xi: FloatND
) -> FloatND:
    return jnp.log(consumption) - next_xi


def _cost_is_affordable(*, wealth: ContinuousState, next_xi: FloatND) -> BoolND:
    return wealth >= 0.4 * jnp.exp(0.5 * next_xi)


def _bequest(wealth: ContinuousState) -> FloatND:
    return jnp.log(wealth + 11.0)


def _probability_alive(age: float) -> FloatND:
    return jnp.where(age < LAST_ALIVE_AGE, 0.8, 0.0)


def _probability_dead(age: float) -> FloatND:
    return jnp.where(age < LAST_ALIVE_AGE, 0.2, 1.0)


def _model(
    *,
    as_state: bool = False,
    utility: UserFunction = _utility,
    constraints: dict[str, UserFunction] | None = None,
    simulated_draws: Literal["continuous", "nodes"] = "continuous",
    execution_config: ExecutionConfig | None = None,
) -> Model:
    alive = Regime(
        actions={
            "consumption": IrregSpacedGrid(points=np.array(CONSUMPTION_NODES)),
        },
        states={"wealth": WEALTH_GRID}
        | ({"xi": _xi_state(simulated_draws)} if as_state else {}),
        state_transitions={"wealth": _next_wealth},
        functions={"utility": utility, "cost": _cost},
        constraints=constraints or {},
        shocks=() if as_state else ("cost_shock",),
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
        shocks={} if as_state else {"cost_shock": _cost_shock()},
        execution_config=ExecutionConfig()
        if execution_config is None
        else execution_config,
    )


def _params() -> dict[ReferenceName, float]:
    return {"discount_factor": DISCOUNT_FACTOR}


def _interp_extrapolating(*, x: float, nodes: np.ndarray, values: np.ndarray) -> float:
    """Piecewise-linear interpolation, extended linearly beyond both ends."""
    i = int(np.clip(np.searchsorted(nodes, x, side="right") - 1, 0, len(nodes) - 2))
    slope = (values[i + 1] - values[i]) / (nodes[i + 1] - nodes[i])
    return float(values[i] + slope * (x - nodes[i]))


def _reference_alive_values() -> dict[int, np.ndarray]:
    """Literal backward induction of `alive`, keyed by period, indexed by wealth."""
    raw_nodes, raw_weights = np.polynomial.hermite.hermgauss(N_XI)
    xi_nodes = np.sqrt(2.0) * raw_nodes
    xi_weights = raw_weights / np.sqrt(np.pi)
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


@pytest.mark.parametrize("as_state", [False, True], ids=["group", "iid-state"])
def test_a_shock_read_only_as_its_draw_is_not_a_state_of_the_regime(
    *, as_state: bool
) -> None:
    assert _model(as_state=as_state).state_names(regime_name="alive") == ("wealth",)


@pytest.mark.parametrize("as_state", [False, True], ids=["group", "iid-state"])
def test_a_shock_read_only_as_its_draw_leaves_no_axis_in_any_period(
    *, as_state: bool
) -> None:
    values = _alive_values(_model(as_state=as_state))
    shapes = {period: V.shape for period, V in values.items()}

    assert shapes == dict.fromkeys((0, 1, 2), (len(WEALTH_GRID.to_jax()),))


@pytest.mark.parametrize("as_state", [False, True], ids=["group", "iid-state"])
@pytest.mark.parametrize("period", [0, 1, 2])
def test_the_integrated_value_matches_the_literal_backward_induction(
    *, as_state: bool, period: int
) -> None:
    V = _alive_values(_model(as_state=as_state))[period]

    np.testing.assert_allclose(V, _reference_alive_values()[period], rtol=0, atol=_ATOL)


def test_the_model_repr_lists_the_integrated_shock() -> None:
    assert "alive: integrated=[cost_shock]" in repr(_model())
    assert "alive: integrated=[xi]" in repr(_model(as_state=True))


def test_the_integrated_policy_does_not_depend_on_the_declaration() -> None:
    """Simulated consumption agrees subject by subject in the first period.

    Both models start every subject at the same wealth; consumption at age 0 is
    chosen before any draw, so it depends only on the stored continuation values.
    """
    wealth = jnp.asarray(np.asarray(WEALTH_GRID.to_jax())[::3])
    n = wealth.shape[0]
    consumption = {}
    for as_state in (False, True):
        initial_conditions = {
            "wealth": wealth,
            "age": jnp.zeros(n),
            "regime_id": jnp.full(n, _RegimeId.alive),
        }
        df = (
            _model(as_state=as_state)
            .simulate(
                params=_params(),
                initial_conditions=initial_conditions,
                log_level="off",
                seed=0,
            )
            .to_dataframe()
            .query("age == 0")
        )
        consumption[as_state] = df["consumption"].to_numpy()

    np.testing.assert_array_equal(consumption[False], consumption[True])


N_SUBJECTS = 400
START_WEALTH = 8.0


def _first_transition(
    *,
    model: Model,
    seed: int = 0,
    shock_paths: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate from age 0 at one wealth; return next wealth and the cost it paid."""
    df = (
        model.simulate(
            params=_params(),
            initial_conditions={
                "wealth": jnp.full(N_SUBJECTS, START_WEALTH),
                "age": jnp.zeros(N_SUBJECTS),
                "regime_id": jnp.full(N_SUBJECTS, _RegimeId.alive),
            },
            shock_paths=shock_paths,
            log_level="off",
            seed=seed,
        )
        .to_dataframe()
        .sort_values("subject_id")
    )
    first = df.query("age == 0").set_index("subject_id")
    second = df.query("age == 1").set_index("subject_id")
    next_wealth = second["wealth"].to_numpy()
    return next_wealth, (first["wealth"] - first["consumption"] + 1.0).to_numpy() - (
        next_wealth
    )


def _node_costs() -> np.ndarray:
    raw_nodes, _ = np.polynomial.hermite.hermgauss(N_XI)
    return 0.4 * np.exp(0.5 * np.sqrt(2.0) * raw_nodes)


def _distance_to_nearest_node(cost: np.ndarray) -> np.ndarray:
    return np.abs(cost[:, None] - _node_costs()[None, :]).min(axis=1)


@pytest.mark.parametrize("as_state", [False, True], ids=["group", "iid-state"])
def test_simulated_cost_is_drawn_from_the_continuous_law(*, as_state: bool) -> None:
    """The draw feeding next wealth follows N(0, 1), not the quadrature lottery.

    The cost each subject paid is recovered from its wealth step, and the draw
    from the cost. Under the five-node lottery the Kolmogorov-Smirnov statistic
    is about 0.2; at 400 draws the 1% critical value is about 0.08.
    """
    _, cost = _first_transition(model=_model(as_state=as_state))
    xi = 2.0 * np.log(cost / 0.4)

    assert np.mean(_distance_to_nearest_node(cost) > _ATOL * 10) > 0.99
    assert kstest(xi, "norm").pvalue > 0.01


def test_simulated_cost_stays_on_the_nodes_when_the_process_asks_for_them() -> None:
    """`simulated_draws="nodes"` keeps the node lottery for an IID process state."""
    _, cost = _first_transition(model=_model(as_state=True, simulated_draws="nodes"))

    np.testing.assert_allclose(
        _distance_to_nearest_node(cost), 0.0, rtol=0, atol=_ATOL * 10
    )
    assert len(np.unique(np.round(cost, 4))) > 1


@pytest.mark.parametrize("as_state", [False, True], ids=["group", "iid-state"])
def test_continuous_draws_do_not_depend_on_subject_chunks(*, as_state: bool) -> None:
    """Each subject's draw is addressed by its row, not by how subjects are split."""
    whole, _ = _first_transition(model=_model(as_state=as_state))
    chunked, _ = _first_transition(
        model=_model(
            as_state=as_state,
            execution_config=ExecutionConfig(
                axis_widths={"subject": 7}, device_memory_bytes=100_000_000
            ),
        )
    )

    np.testing.assert_array_equal(chunked, whole)


@pytest.mark.parametrize("as_state", [False, True], ids=["group", "iid-state"])
def test_continuous_draws_replay_from_given_shock_paths(*, as_state: bool) -> None:
    """Supplying the recovered draws reproduces next wealth under another seed."""
    model = _model(as_state=as_state)
    next_wealth, cost = _first_transition(model=model, seed=0)
    path = np.zeros((N_SUBJECTS, AGES.n_periods))
    path[:, 1] = 2.0 * np.log(cost / 0.4)

    other_seed, _ = _first_transition(model=model, seed=1)
    replayed, _ = _first_transition(model=model, seed=1, shock_paths={"xi": path})

    assert np.abs(other_seed - next_wealth).max() > 1e-3
    aaae(replayed, next_wealth, decimal=DECIMAL_PRECISION)


def test_an_iid_state_read_as_its_value_and_its_draw_keeps_its_axis() -> None:
    model = _model(as_state=True, utility=_utility_reading_xi)
    order = model.state_names(regime_name="alive")
    V = _alive_values(model)[0]

    assert set(order) == {"wealth", "xi"}
    assert V.shape[order.index("xi")] == N_XI
    assert "alive: axis=[xi]" in repr(model)


def test_a_group_read_as_its_value_and_its_draw_is_refused() -> None:
    with pytest.raises(
        ModelInitializationError,
        match=r"(?s)'xi'.*'next_xi'.*cost_shock.*NormalIIDProcess",
    ):
        _model(utility=_utility_reading_xi)


@pytest.mark.parametrize("as_state", [False, True], ids=["group", "iid-state"])
@pytest.mark.parametrize(
    "utility",
    [_utility_reading_next_xi, _utility_reading_cost],
    ids=["direct", "through-a-helper"],
)
def test_a_utility_reading_the_draw_is_refused(
    *, as_state: bool, utility: UserFunction
) -> None:
    with pytest.raises(
        InvalidNameError, match=r"(?s)'utility' reads 'next_xi'.*'next_' prefix"
    ):
        _model(as_state=as_state, utility=utility)


@pytest.mark.parametrize("as_state", [False, True], ids=["group", "iid-state"])
def test_a_constraint_reading_the_draw_is_refused(*, as_state: bool) -> None:
    with pytest.raises(
        InvalidNameError, match=r"(?s)'affordable' reads 'next_xi'.*'next_' prefix"
    ):
        _model(as_state=as_state, constraints={"affordable": _cost_is_affordable})


def test_fold_is_refused_with_the_inference_rule() -> None:
    with pytest.raises(TypeError, match=r"(?s)fold.*removed.*next_"):
        NormalIIDProcess(
            n_points=N_XI,
            gauss_hermite=True,
            mu=0.0,
            sigma=1.0,
            fold=True,  # ty: ignore[unknown-argument]
        )


def test_a_post_decision_group_on_an_egm_regime_is_integrated_out() -> None:
    ages = AgeGrid(start=40, inclusive_stop=60, step="10Y")
    retirement = iskhakov.dcegm_retirement
    model = Model(
        regimes={
            "working_life": iskhakov.dcegm_working_life,
            "retirement": retirement.replace(
                functions={**retirement.functions, "cost": _cost},
                state_transitions={
                    **retirement.state_transitions,
                    "wealth": _next_wealth_from_savings_net_of_cost,
                },
                shocks=("cost_shock",),
            ),
            "dead": iskhakov.dead,
        },
        edges=iskhakov.get_edges(ages=ages),
        ages=ages,
        regime_id_class=iskhakov.RegimeId,
        initial_nodes={ages.exact_values[0]: "working_life"},
        shocks={"cost_shock": _cost_shock()},
    )

    assert model.state_names(regime_name="retirement") == ("wealth",)
    assert "retirement: integrated=[xi]" in repr(model)
