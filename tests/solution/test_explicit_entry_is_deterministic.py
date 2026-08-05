"""A declared entry law names one value, however the engine represents it.

A source that does not carry a target's continuous stochastic process may declare
an entry law for it. The law names a *physical value*. The target holds its value
function only on the process's nodes, so the engine expresses that value in the
node basis — the hat weights of linear interpolation — and the continuation is
`Σ_j w_j · V(node_j)`.

Those weights are basis coefficients, not probabilities: entering at `1.5` on
nodes `(1, 2)` is the single value `0.5·1 + 0.5·2`, not a coin flip between them.
The distinction is invisible under a linear expectation, which averages either
reading identically, and decisive under every other certainty equivalent, whose
transform is applied before any averaging. Every case below therefore uses a
nonlinear certainty equivalent — that is the only shape that separates the two
readings.

Oracles are arithmetic on a three-node grid whose value function is written out
in full, so they share no code with the engine under test.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from lcm import (
    AgeGrid,
    LinSpacedGrid,
    MarkovTransition,
    Model,
    NormalIIDProcess,
    PowerMean,
    QuasiArithmeticMean,
    Regime,
    UniformIIDProcess,
    categorical,
    fixed_transition,
)
from lcm.exceptions import InvalidValueFunctionError, ModelInitializationError
from lcm.typing import FloatND, ScalarFloat, ScalarInt
from tests.conftest import DECIMAL_PRECISION

# `mu=1, sigma=0.5, n_std=2` at three points puts the target's nodes on
# `(0, 1, 2)`, and its payoff is `shock**2`, so its value function is `(0, 1, 4)`.
_NODES = (0.0, 1.0, 2.0)
_V = (0.0, 1.0, 4.0)
_RISK_AVERSION = 2.0


@categorical(ordered=False)
class RegimeId:
    source: ScalarInt
    target: ScalarInt


def _zero_utility() -> FloatND:
    return jnp.asarray(0.0)


def _squared_shock_utility(shock: ScalarFloat) -> FloatND:
    return shock**2


def _one_probability() -> FloatND:
    return jnp.asarray(1.0)


def _source_is_early(age: float) -> bool:
    return age < 22


def _target_process() -> NormalIIDProcess:
    return NormalIIDProcess(
        n_points=3, gauss_hermite=False, mu=1.0, sigma=0.5, n_std=2.0
    )


def _power_mean(values: np.ndarray, weights: np.ndarray) -> float:
    """Return the weighted power mean at `_RISK_AVERSION`, computed independently."""
    normalized = weights / weights.sum()
    exponent = 1.0 - _RISK_AVERSION
    return float(np.sum(normalized * values**exponent) ** (1.0 / exponent))


def _build_model(*, entry_value: float, enable_jit: bool) -> Model:
    """Build a source entering the target's process at one physical value."""

    def _enter_at() -> ScalarFloat:
        return jnp.asarray(entry_value)

    return Model(
        regimes={
            "source": Regime(
                transition={"target": MarkovTransition(_one_probability)},
                active=_source_is_early,
                state_transitions={"shock": {"target": _enter_at}},
                functions={"utility": _zero_utility},
                certainty_equivalent=PowerMean(),
            ),
            "target": Regime(
                transition=None,
                states={"shock": _target_process()},
                functions={"utility": _squared_shock_utility},
            ),
        },
        ages=AgeGrid(start=20, stop=22, step="Y"),
        regime_id_class=RegimeId,
        enable_jit=enable_jit,
    )


_PARAMS = {
    "source": {
        "utility": {},
        "koopmans_aggregator": {"discount_factor": 1.0},
        "certainty_equivalent": {"risk_aversion": _RISK_AVERSION},
        "target": {"next_regime": {}, "next_shock": {}},
    },
    "target": {"utility": {}},
}


def _source_value(model: Model, params: dict) -> float:
    solution = model.solve(params=params, log_level="debug")
    last_living = max(period for period in solution if "source" in solution[period])
    return float(np.asarray(solution[last_living]["source"]).ravel()[0])


@pytest.mark.parametrize("enable_jit", [False, True], ids=["eager", "jit"])
def test_entry_between_nodes_interpolates_under_a_nonlinear_ce(
    enable_jit: bool,  # noqa: FBT001
) -> None:
    """Entering at `1.5` is worth the midpoint of `V(1)` and `V(2)`, i.e. `2.5`.

    Reading the same weights as a lottery and handing them to `PowerMean` at
    `risk_aversion = 2` gives the weighted harmonic mean `1.6` instead — a
    different number by a finite margin, not by rounding.
    """
    model = _build_model(entry_value=1.5, enable_jit=enable_jit)

    got = _source_value(model, _PARAMS)

    expected = 0.5 * _V[1] + 0.5 * _V[2]
    np.testing.assert_almost_equal(got, expected, decimal=DECIMAL_PRECISION)


@pytest.mark.parametrize("enable_jit", [False, True], ids=["eager", "jit"])
def test_entry_on_a_node_reads_that_node_under_a_nonlinear_ce(
    enable_jit: bool,  # noqa: FBT001
) -> None:
    """Entering at node `2.0` is worth `V(2) = 4`.

    The weights are one-hot here, so both readings agree. This is the negative
    control: it keeps a repair from being a blanket rescaling that happens to
    fix the off-node case.
    """
    model = _build_model(entry_value=2.0, enable_jit=enable_jit)

    got = _source_value(model, _PARAMS)

    np.testing.assert_almost_equal(got, _V[2], decimal=DECIMAL_PRECISION)


@pytest.mark.parametrize(
    "entry_value", [-1.0, 5.0], ids=["below_support", "above_support"]
)
def test_a_constant_entry_outside_the_support_is_rejected(entry_value: float) -> None:
    """A value the process cannot represent is an error, not an approximation.

    The target holds its value function on the process's nodes and has nothing
    beyond them, so clamping to the nearest node or extrapolating past it would
    publish a continuation the support does not justify. A law that reads no
    state has one value, known while the model builds, so it is rejected there
    with the state, the value, and the range named.
    """
    with pytest.raises(ModelInitializationError) as excinfo:
        _build_model(entry_value=entry_value, enable_jit=False)

    message = str(excinfo.value)
    assert "'shock'" in message
    assert str(entry_value) in message
    assert f"[{_NODES[0]}, {_NODES[-1]}]" in message


def test_a_state_dependent_entry_outside_the_support_fails_loudly() -> None:
    """A law that only leaves the support for some states does not go unnoticed.

    Its value is unknown until it runs, so the support check cannot happen while
    the model builds. The weights are poisoned instead, and the solve-time value
    check raises at `log_level="debug"` rather than letting a zero continuation
    pass for a real one.

    Reaching that report also exercises solve/diagnostics parity for a declared
    entry: the diagnostics closure recomputes the continuation from the same
    builder the Bellman uses, so a basis law missing from its signature would
    surface here as a missing argument rather than as the NaN report.
    """

    def _enter_at_wealth(wealth: ScalarFloat) -> ScalarFloat:
        return wealth

    model = Model(
        regimes={
            "source": Regime(
                transition={"target": MarkovTransition(_one_probability)},
                active=_source_is_early,
                # The top of this grid lies outside the target's `(0, 1, 2)`.
                states={"wealth": LinSpacedGrid(start=1.0, stop=9.0, n_points=3)},
                state_transitions={
                    "shock": {"target": _enter_at_wealth},
                    "wealth": fixed_transition("wealth"),
                },
                functions={"utility": _zero_utility},
            ),
            "target": Regime(
                transition=None,
                states={"shock": _target_process()},
                functions={"utility": _squared_shock_utility},
            ),
        },
        ages=AgeGrid(start=20, stop=22, step="Y"),
        regime_id_class=RegimeId,
        enable_jit=False,
    )
    params = {
        "source": {
            "utility": {},
            "koopmans_aggregator": {"discount_factor": 1.0},
            "target": {"next_regime": {}, "next_shock": {}},
        },
        "target": {"utility": {}},
    }

    with pytest.raises(InvalidValueFunctionError, match=r"(?i)nan") as excinfo:
        model.solve(params=params, log_level="debug")

    assert "compute_intermediates" in str(excinfo.value)


@pytest.mark.parametrize("enable_jit", [False, True], ids=["eager", "jit"])
def test_a_linear_payoff_entry_interpolates_to_its_own_value(
    enable_jit: bool,  # noqa: FBT001
) -> None:
    """On `V = 1 + shock`, entering at `0.5` is worth `1.5`, not `4/3`.

    A payoff linear in the shock makes the interpolated continuation equal the
    payoff at the declared value, so the expected number needs no arithmetic to
    believe. Reading the coefficients as probabilities gives the harmonic mean
    of `V(0) = 1` and `V(1) = 2`, which is `4/3` — below a competing action
    worth `1.4`, while the correct `1.5` is above it.
    """

    def _one_plus_shock(shock: ScalarFloat) -> FloatND:
        return 1.0 + shock

    def _enter_at_half() -> ScalarFloat:
        return jnp.asarray(0.5)

    model = Model(
        regimes={
            "source": Regime(
                transition={"target": MarkovTransition(_one_probability)},
                active=_source_is_early,
                state_transitions={"shock": {"target": _enter_at_half}},
                functions={"utility": _zero_utility},
                certainty_equivalent=PowerMean(),
            ),
            "target": Regime(
                transition=None,
                states={"shock": _target_process()},
                functions={"utility": _one_plus_shock},
            ),
        },
        ages=AgeGrid(start=20, stop=22, step="Y"),
        regime_id_class=RegimeId,
        enable_jit=enable_jit,
    )

    got = _source_value(model, _PARAMS)

    np.testing.assert_almost_equal(got, 1.5, decimal=DECIMAL_PRECISION)
    assert got > 1.4 > 4.0 / 3.0


def test_a_non_power_quasi_arithmetic_mean_also_sees_one_value() -> None:
    """The reading is a property of the entry law, not of `PowerMean`.

    An exponential-transform quasi-arithmetic mean is nonlinear for the same
    reason and shares no code with the power mean's anchored form, so it pins
    the interpolation independently. Its aggregate over a degenerate lottery is
    the value itself, which is `2.5` when the entry is interpolated and the
    strictly smaller `-log((exp(-1) + exp(-4)) / 2)` when the coefficients are
    read as probabilities.
    """

    def _negative_exponential(value: FloatND) -> FloatND:
        return -jnp.exp(-value)

    def _inverse_negative_exponential(value: FloatND) -> FloatND:
        return -jnp.log(-value)

    def _enter_at() -> ScalarFloat:
        return jnp.asarray(1.5)

    model = Model(
        regimes={
            "source": Regime(
                transition={"target": MarkovTransition(_one_probability)},
                active=_source_is_early,
                state_transitions={"shock": {"target": _enter_at}},
                functions={"utility": _zero_utility},
                certainty_equivalent=QuasiArithmeticMean(
                    transform=_negative_exponential,
                    inverse=_inverse_negative_exponential,
                ),
            ),
            "target": Regime(
                transition=None,
                states={"shock": _target_process()},
                functions={"utility": _squared_shock_utility},
            ),
        },
        ages=AgeGrid(start=20, stop=22, step="Y"),
        regime_id_class=RegimeId,
        enable_jit=False,
    )
    params = {
        "source": {
            "utility": {},
            "koopmans_aggregator": {"discount_factor": 1.0},
            "target": {"next_regime": {}, "next_shock": {}},
        },
        "target": {"utility": {}},
    }

    got = _source_value(model, params)

    lottery_reading = -np.log((np.exp(-_V[1]) + np.exp(-_V[2])) / 2.0)
    np.testing.assert_almost_equal(got, 2.5, decimal=DECIMAL_PRECISION)
    assert lottery_reading < 2.5


def test_two_declared_entries_into_one_target_interpolate_jointly() -> None:
    """Two declared coordinates are one point, not a product lottery.

    Each entry law names a value, so the pair names a single point of the
    target's two-dimensional node grid and its continuation is the bilinear
    interpolation there. Treating the two bases as independent lotteries would
    hand a nonlinear certainty equivalent a four-node product instead.
    """

    def _enter_first() -> ScalarFloat:
        return jnp.asarray(1.5)

    def _enter_second() -> ScalarFloat:
        return jnp.asarray(0.5)

    def _product_utility(shock: ScalarFloat, other: ScalarFloat) -> FloatND:
        return shock**2 + 10.0 * other

    model = Model(
        regimes={
            "source": Regime(
                transition={"target": MarkovTransition(_one_probability)},
                active=_source_is_early,
                state_transitions={
                    "shock": {"target": _enter_first},
                    "other": {"target": _enter_second},
                },
                functions={"utility": _zero_utility},
                certainty_equivalent=PowerMean(),
            ),
            "target": Regime(
                transition=None,
                states={
                    "shock": _target_process(),
                    "other": UniformIIDProcess(n_points=2, start=0.0, stop=1.0),
                },
                functions={"utility": _product_utility},
            ),
        },
        ages=AgeGrid(start=20, stop=22, step="Y"),
        regime_id_class=RegimeId,
        enable_jit=False,
    )
    params = {
        "source": {
            "utility": {},
            "koopmans_aggregator": {"discount_factor": 1.0},
            "certainty_equivalent": {"risk_aversion": _RISK_AVERSION},
            "target": {"next_regime": {}, "next_shock": {}, "next_other": {}},
        },
        "target": {"utility": {}},
    }

    got = _source_value(model, params)

    # `V(shock, other) = shock**2 + 10*other` on nodes `(0,1,2) x (0,1)`.
    # Entering at `(1.5, 0.5)` is the bilinear value `2.5 + 5.0`.
    grid = np.array(_V)[:, None] + 10.0 * np.array([0.0, 1.0])[None, :]
    weights = np.outer([0.0, 0.5, 0.5], [0.5, 0.5])
    np.testing.assert_almost_equal(
        got, float((grid * weights).sum()), decimal=DECIMAL_PRECISION
    )
    # The product-lottery reading is strictly smaller under this power mean, so
    # the assertion above is not satisfied by both.
    assert _power_mean(grid[weights > 0], weights[weights > 0]) < 7.5


def test_a_declared_entry_and_a_drawn_process_are_aggregated_differently() -> None:
    """One target, two processes: one entered deterministically, one drawn.

    The declared entry contributes a single interpolated value on its own axis;
    the process the source does not name is entered at its own unconditional law
    and stays a genuine lottery. The certainty equivalent must see the second and
    not the first, so the oracle applies `PowerMean` only across the drawn nodes
    of the interpolated surface.
    """

    def _enter_at() -> ScalarFloat:
        return jnp.asarray(1.5)

    def _sum_utility(shock: ScalarFloat, extra: ScalarFloat) -> FloatND:
        return shock**2 + extra

    model = Model(
        regimes={
            "source": Regime(
                transition={"target": MarkovTransition(_one_probability)},
                active=_source_is_early,
                state_transitions={"shock": {"target": _enter_at}},
                functions={"utility": _zero_utility},
                certainty_equivalent=PowerMean(),
            ),
            "target": Regime(
                transition=None,
                states={
                    "shock": _target_process(),
                    "extra": UniformIIDProcess(n_points=2, start=1.0, stop=3.0),
                },
                functions={"utility": _sum_utility},
            ),
        },
        ages=AgeGrid(start=20, stop=22, step="Y"),
        regime_id_class=RegimeId,
        enable_jit=False,
    )
    params = {
        "source": {
            "utility": {},
            "koopmans_aggregator": {"discount_factor": 1.0},
            "certainty_equivalent": {"risk_aversion": _RISK_AVERSION},
            "target": {"next_regime": {}, "next_shock": {}},
        },
        "target": {"utility": {}},
    }

    got = _source_value(model, params)

    # `V(shock, extra) = shock**2 + extra`. Interpolating the declared entry at
    # `1.5` collapses the shock axis to `0.5*V(1) + 0.5*V(2) = 2.5` per `extra`
    # node, leaving a two-node lottery the drawn process weights equally.
    interpolated = 2.5 + np.array([1.0, 3.0])
    expected = _power_mean(interpolated, np.array([0.5, 0.5]))
    np.testing.assert_almost_equal(got, expected, decimal=DECIMAL_PRECISION)


def test_the_entry_representation_decides_the_action() -> None:
    """A competing action worth more than the lottery reading but less than `2.5`.

    Entering is worth `2.5` when the declared value is interpolated and `1.6`
    when its weights are priced as a lottery, so a payoff of `2.0` reverses the
    Bellman argmax between the two readings. The assertion is the discrete
    choice, which no tolerance can absorb.
    """

    def _stay_utility(wealth: ScalarFloat) -> FloatND:
        return 2.0 + 0.0 * wealth

    def _enter_at() -> ScalarFloat:
        return jnp.asarray(1.5)

    @categorical(ordered=False)
    class _ThreeRegimeId:
        source: ScalarInt
        stay: ScalarInt
        enter: ScalarInt

    def _choose(go: ScalarInt) -> ScalarInt:
        return jnp.where(go == 1, _ThreeRegimeId.enter, _ThreeRegimeId.stay)

    model = Model(
        regimes={
            "source": Regime(
                transition=_choose,
                active=_source_is_early,
                actions={"go": LinSpacedGrid(start=0, stop=1, n_points=2)},
                state_transitions={
                    "wealth": {"stay": lambda: jnp.asarray(1.0)},
                    "shock": {"enter": _enter_at},
                },
                functions={"utility": _zero_utility},
                certainty_equivalent=PowerMean(),
            ),
            "stay": Regime(
                transition=None,
                states={"wealth": LinSpacedGrid(start=1.0, stop=2.0, n_points=2)},
                functions={"utility": _stay_utility},
            ),
            "enter": Regime(
                transition=None,
                states={"shock": _target_process()},
                functions={"utility": _squared_shock_utility},
            ),
        },
        ages=AgeGrid(start=20, stop=22, step="Y"),
        regime_id_class=_ThreeRegimeId,
        enable_jit=False,
    )
    params = {
        "source": {
            "utility": {},
            "koopmans_aggregator": {"discount_factor": 1.0},
            "certainty_equivalent": {"risk_aversion": _RISK_AVERSION},
            "next_wealth": {},
            "next_shock": {},
            "next_regime": {},
        },
        "stay": {"utility": {}},
        "enter": {"utility": {}},
    }

    got = _source_value(model, params)

    np.testing.assert_almost_equal(got, 2.5, decimal=DECIMAL_PRECISION)
