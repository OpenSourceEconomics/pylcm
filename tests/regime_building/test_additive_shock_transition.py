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

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import cloudpickle
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.grids.coordinates import get_irreg_coordinate
from _lcm.regime_building import Q_and_F
from _lcm.regime_building.ndimage import map_coordinates
from _lcm.regime_building.shock_average import average_over_shock
from _lcm.utils.logging import LogLevel
from lcm import (
    AdditiveShockTransition,
    AgeGrid,
    ByAge,
    DiscreteGrid,
    IrregSpacedGrid,
    LinearExpectation,
    LinSpacedGrid,
    Model,
    NormalIIDProcess,
    PowerMean,
    QuasiArithmeticMean,
    Regime,
    Transition,
    categorical,
)
from lcm.exceptions import ModelInitializationError, RegimeInitializationError
from lcm.transition import StochasticTransition
from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    DiscreteState,
    FloatND,
    IntND,
    OffGrid,
    ReferenceName,
    ScalarInt,
)
from tests.conftest import DECIMAL_PRECISION, X64_ENABLED

DISCOUNT_FACTOR = 0.9
LAST_ALIVE_AGE = 2
CONSUMPTION_NODES = (0.5, 1.0, 1.5, 2.0, 3.0)
N_XI = 5
POOR_BELOW = 4.0
HEALTH_PROBS = np.array([[0.7, 0.3], [0.2, 0.8]])
_ATOL = 1e-10 if X64_ENABLED else 1e-4
_FP32_ATOL = 1e-5


def _ages() -> AgeGrid:
    return AgeGrid(start=0, inclusive_stop=3, step="Y")


def _wealth_grid(*, off_grid: OffGrid = "extrapolate") -> LinSpacedGrid:
    """Build the wealth grid at call time.

    A grid stores its bounds as JAX scalars at the precision active when it is
    built, so the fp32 tests need a grid built under fp32.
    """
    return LinSpacedGrid(start=0.0, stop=12.0, n_points=25, off_grid=off_grid)


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
    # NumPy literals must enter the traced law at the active precision.
    return jnp.asarray(HEALTH_PROBS, dtype=jnp.result_type(0.0))[health]


def _utility(*, consumption: ContinuousAction, health: DiscreteState) -> FloatND:
    return jnp.log(consumption) + 0.3 * health


def _bequest(wealth: ContinuousState) -> FloatND:
    return jnp.log(wealth + 11.0)


def _probability_alive(age: float) -> FloatND:
    return jnp.where(age < LAST_ALIVE_AGE, 0.8, 0.0)


def _probability_dead(age: float) -> FloatND:
    return jnp.where(age < LAST_ALIVE_AGE, 0.2, 1.0)


def _model(
    *,
    additive: bool,
    shock_reads_wealth: bool = False,
    off_grid: OffGrid = "extrapolate",
) -> Model:
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
            "wealth": _wealth_grid(off_grid=off_grid),
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
    dead = Regime(
        states={"wealth": _wealth_grid(off_grid=off_grid)},
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
        ages=_ages(),
        regime_id_class=_RegimeId,
        initial_nodes={0: "alive"},
    )


def _params() -> dict[ReferenceName, float]:
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
    wealth_nodes = np.asarray(_wealth_grid().to_jax(), dtype=np.float64)
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

    def observer[**P](
        func: Callable[P, Q_and_F._TargetContinuation],
    ) -> Callable[P, Q_and_F._TargetContinuation]:
        def observe(*args: P.args, **kwargs: P.kwargs) -> Q_and_F._TargetContinuation:
            result = func(*args, **kwargs)
            observed.append(sorted(result.lottery_axis_names))
            return result

        return observe

    monkeypatch.setattr(Q_and_F, "_build_target_continuation", observer(original))
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


@pytest.mark.parametrize("off_grid", ["clip_lookup", "clip_state"])
def test_clipped_averaged_bill_matches_every_native_shock_read(
    off_grid: OffGrid,
) -> None:
    """The folded average preserves values, policies and actual state histories."""
    values = {}
    frames = {}
    for additive in (False, True):
        model = _model(additive=additive, off_grid=off_grid)
        values[additive] = _alive_values(model)
        frames[additive] = model.simulate(
            params=_params(),
            initial_conditions={
                "wealth": jnp.array([0.0, 6.0, 12.0]),
                "health": jnp.zeros(3, dtype=jnp.int32),
                "age": jnp.zeros(3),
                "regime_id": jnp.full(3, _RegimeId.alive),
            },
            log_level="off",
            seed=0,
        ).to_dataframe()
    for period in values[False]:
        np.testing.assert_allclose(
            values[True][period], values[False][period], rtol=0, atol=_ATOL
        )
    for column in ("consumption", "wealth"):
        np.testing.assert_allclose(
            frames[True][column],
            frames[False][column],
            rtol=0,
            atol=_ATOL,
            equal_nan=True,
        )


@pytest.mark.usefixtures("x64_disabled")
@pytest.mark.parametrize("additive", [True, False], ids=["additive", "plain"])
@pytest.mark.parametrize("period", [0, 1, 2])
def test_the_fp32_value_matches_the_fp64_reference_at_the_plain_law_accuracy(
    *, additive: bool, period: int
) -> None:
    """In fp32, either law's value lies within `_FP32_ATOL` of the fp64 oracle.

    Reads land below the wealth grid, where the value is extrapolated, so the
    tolerance also bounds the extrapolation of the averaged value.
    """
    V = _alive_values(_model(additive=additive))[period]

    assert V.dtype == np.dtype(np.float32)
    np.testing.assert_allclose(
        V, _reference_alive_values()[period], rtol=0, atol=_FP32_ATOL
    )


def test_the_additive_shock_policy_equals_the_plain_law_policy() -> None:
    """Simulated consumption agrees subject by subject in the first period."""
    wealth = jnp.asarray(np.asarray(_wealth_grid().to_jax())[::3])
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


@pytest.mark.parametrize("additive", [False, True])
def test_validate_transitions_accepts_plain_and_folded_shock_laws(
    *, additive: bool
) -> None:
    """The no-solve preflight accepts both declarations of the same shock law."""
    assert _model(additive=additive).validate_transitions(params=_params()) is None


@dataclass(frozen=True, kw_only=True)
class _HarmonicLinearSubclass(LinearExpectation):
    """Use a harmonic mean even though the public base class is linear."""

    def aggregate(
        self,
        *,
        values: FloatND,
        weights: FloatND,
        params: Mapping[str, FloatND],  # noqa: ARG002
    ) -> FloatND:
        return PowerMean().aggregate(
            values=values, weights=weights, params={"risk_aversion": jnp.asarray(2.0)}
        )

    def aggregate_scaled(
        self,
        *,
        values: FloatND,
        coefficients: FloatND,
        shifts: IntND,
        params: Mapping[str, FloatND],  # noqa: ARG002
    ) -> FloatND:
        return PowerMean().aggregate_scaled(
            values=values,
            coefficients=coefficients,
            shifts=shifts,
            params={"risk_aversion": jnp.asarray(2.0)},
        )


def _reciprocal(value: FloatND) -> FloatND:
    return 1.0 / value


def _constant_utility(wealth: ContinuousState) -> FloatND:
    return 1.0 + 0.0 * wealth


def _wealth_base(wealth: ContinuousState) -> FloatND:
    return wealth


def _additive_law(*, base: FloatND, shock: FloatND) -> ContinuousState:
    return base + shock


def _identity_shock(next_xi: ContinuousState) -> FloatND:
    return next_xi


def _terminal_value(wealth: ContinuousState) -> FloatND:
    return wealth + 10.0


def _terminal_value_with_xi(*, wealth: ContinuousState, xi: ContinuousState) -> FloatND:
    return wealth + 10.0 + xi


def _ce_model(
    *,
    additive: bool,
    ce_kind: str = "linear",
    ce_at_model: bool = True,
    live_terminal_shock: bool = False,
    shock_reads_wealth: bool = False,
    alive_reads_xi: bool = False,
) -> Model:
    grid = LinSpacedGrid(start=0.0, stop=4.0, n_points=5)
    ce = {
        "power": PowerMean(),
        "quasi": QuasiArithmeticMean(transform=_reciprocal, inverse=_reciprocal),
        "subclass": _HarmonicLinearSubclass(),
        "linear": LinearExpectation(),
        "default": None,
    }[ce_kind]
    alive = Regime(
        states={
            "wealth": grid,
            "xi": NormalIIDProcess(
                n_points=3, gauss_hermite=True, mu=0.0, sigma=1.0, fold=True
            ),
        },
        state_transitions={
            "wealth": _wealth_base
            if alive_reads_xi
            else (
                AdditiveShockTransition(base="base", shock="shock")
                if additive
                else _additive_law
            )
        },
        functions={
            "utility": _terminal_value_with_xi if alive_reads_xi else _constant_utility,
        }
        | (
            {}
            if alive_reads_xi
            else {
                "base": _wealth_base,
                "shock": _cost_reading_wealth
                if shock_reads_wealth
                else _identity_shock,
            }
        ),
        certainty_equivalent=None if ce_at_model else ce,
    )
    dead = Regime(
        states={"wealth": grid}
        | (
            {"xi": NormalIIDProcess(n_points=3, gauss_hermite=True, mu=0.0, sigma=1.0)}
            if live_terminal_shock
            else {}
        ),
        functions={
            "utility": _terminal_value_with_xi
            if live_terminal_shock
            else _terminal_value
        },
    )
    return Model(
        regimes={"alive": alive, "dead": dead},
        edges={"alive": Transition(targets={"dead": (0,)}, law="dead")},
        ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
        regime_id_class=_RegimeId,
        initial_nodes={0: ("alive", "dead")},
        **({"certainty_equivalent": ce} if ce_at_model and ce is not None else {}),
    )


@pytest.mark.parametrize(
    ("ce_kind", "ce_at_model"),
    [
        ("power", True),
        ("quasi", True),
        ("subclass", True),
        ("linear", True),
        ("linear", False),
        ("default", True),
    ],
)
@pytest.mark.parametrize("round_trip", [False, True], ids=["built", "restored"])
def test_additive_transition_preserves_each_certainty_equivalent(
    *, ce_kind: str, ce_at_model: bool, round_trip: bool
) -> None:
    """Arithmetic averaging is reserved for exact linear expectations.

    The independent oracle evaluates the arithmetic or harmonic mean directly
    at every wealth node, including landings outside the wealth grid.
    """
    # Three-node standard-normal Gauss-Hermite quadrature, independently stated.
    nodes = np.array([-np.sqrt(3.0), 0.0, np.sqrt(3.0)])
    weights = np.array([1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0])
    realized = 10.0 + np.arange(5)[:, None] + nodes[None, :]
    assert np.all(realized > 0.0)
    assert np.ptp(realized[2]) > 0.0
    ce = (
        (weights * realized).sum(axis=1)
        if ce_kind in {"linear", "default"}
        else (1.0 / (weights / realized).sum(axis=1))
    )
    expected = 1.0 + DISCOUNT_FACTOR * ce
    params = {"discount_factor": DISCOUNT_FACTOR} | (
        {"alive": {"certainty_equivalent": {"risk_aversion": 2.0}}}
        if ce_kind == "power"
        else {}
    )
    for additive in (False, True):
        model = _ce_model(additive=additive, ce_kind=ce_kind, ce_at_model=ce_at_model)
        if round_trip:
            model = cloudpickle.loads(cloudpickle.dumps(model))
        values = model.solve(params=params, log_level="off").values[0]["alive"]
        np.testing.assert_array_almost_equal(
            values, expected, decimal=DECIMAL_PRECISION
        )


@pytest.mark.parametrize("round_trip", [False, True], ids=["built", "restored"])
@pytest.mark.parametrize("log_level", ["off", "debug"])
def test_a_transition_only_name_preserves_another_regimes_live_initial_state(
    *, round_trip: bool, log_level: LogLevel
) -> None:
    """A terminal entrant retains xi 0.7 and receives terminal value 12.7."""
    model = _ce_model(additive=False, live_terminal_shock=True)
    if round_trip:
        model = cloudpickle.loads(cloudpickle.dumps(model))
    frame = model.simulate(
        params={"discount_factor": DISCOUNT_FACTOR},
        initial_conditions={
            "wealth": jnp.array([2.0]),
            "xi": jnp.array([0.7]),
            "age": jnp.array([0.0]),
            "regime_id": jnp.array([_RegimeId.dead]),
        },
        log_level=log_level,
    ).to_dataframe()
    np.testing.assert_array_almost_equal(frame["xi"], [0.7], decimal=DECIMAL_PRECISION)
    np.testing.assert_array_almost_equal(
        frame["value"], [12.7], decimal=DECIMAL_PRECISION
    )


@pytest.mark.parametrize("ce_kind", ["power", "quasi", "subclass"])
def test_nonlinear_fallback_still_validates_the_additive_decomposition(
    ce_kind: str,
) -> None:
    """An undeclared source-state dependency is refused even without averaging."""
    with pytest.raises(ModelInitializationError, match="reads the source's"):
        _ce_model(additive=True, ce_kind=ce_kind, shock_reads_wealth=True)


@pytest.mark.parametrize("ce_kind", ["power", "quasi", "subclass", "linear", "default"])
@pytest.mark.parametrize("ce_at_model", [False, True], ids=["regime-ce", "model-ce"])
def test_within_period_folding_requires_the_exact_linear_expectation(
    *, ce_kind: str, ce_at_model: bool
) -> None:
    """Within-period averaging admits only an absent or exact linear mean."""
    if ce_kind in {"power", "quasi", "subclass"}:
        with pytest.raises(RegimeInitializationError, match="certainty_equivalent"):
            _ce_model(
                additive=False,
                ce_kind=ce_kind,
                ce_at_model=ce_at_model,
                alive_reads_xi=True,
            )
    else:
        model = _ce_model(
            additive=False,
            ce_kind=ce_kind,
            ce_at_model=ce_at_model,
            alive_reads_xi=True,
        )
        assert "xi" in model.state_names(regime_name="alive")
        solution = model.solve(
            params={"discount_factor": DISCOUNT_FACTOR}, log_level="off"
        )
        np.testing.assert_array_almost_equal(
            solution.values[0]["alive"],
            (1.0 + DISCOUNT_FACTOR) * (10.0 + np.arange(5)),
            decimal=DECIMAL_PRECISION,
        )


@pytest.mark.parametrize("ce_kind", ["power", "quasi", "subclass"])
def test_an_explicit_regime_nonlinear_ce_keeps_its_fold_admission_restriction(
    ce_kind: str,
) -> None:
    """Regime construction refuses nonlinear means before draw-only pruning."""
    with pytest.raises(RegimeInitializationError, match="certainty_equivalent"):
        _ce_model(additive=True, ce_kind=ce_kind, ce_at_model=False)
