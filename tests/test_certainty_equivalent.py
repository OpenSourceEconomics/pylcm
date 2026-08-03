"""Tests for nonlinear certainty equivalents over the continuation value."""

from collections.abc import Callable
from decimal import Decimal, localcontext
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

from lcm import (
    AgeGrid,
    DiscreteGrid,
    H_epstein_zin,
    LinSpacedGrid,
    MarkovTransition,
    Model,
    Phased,
    PowerMean,
    QuasiArithmeticMean,
    Regime,
    categorical,
)
from lcm.exceptions import InvalidNameError, RegimeInitializationError
from lcm.solvers import DCEGM
from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    DiscreteState,
    FloatND,
    Period,
    ScalarInt,
)
from lcm_examples.epstein_zin import get_model, get_params


def test_power_certainty_equivalent_transform_and_inverse_are_inverses():
    """`inverse(transform(x)) == x` for positive values."""
    ce = PowerMean()
    x = jnp.array([0.5, 1.0, 2.0, 7.5])
    roundtrip = ce.inverse(
        value=ce.transform(value=x, risk_aversion=jnp.asarray(0.5)),
        risk_aversion=jnp.asarray(0.5),
    )
    np.testing.assert_allclose(roundtrip, x, rtol=1e-6)


def test_power_certainty_equivalent_param_names():
    """The power CE declares exactly the `risk_aversion` runtime param."""
    assert PowerMean().param_names == frozenset({"risk_aversion"})


def test_quasi_arithmetic_mean_param_names_union_over_both_callables():
    """`param_names` is the union of transform and inverse args minus `value`."""

    def g(value: FloatND, theta: FloatND) -> FloatND:
        return value * theta

    def g_inv(value: FloatND, theta: FloatND, offset: FloatND) -> FloatND:
        return value / theta + offset

    ce = QuasiArithmeticMean(transform=g, inverse=g_inv)
    assert ce.param_names == frozenset({"theta", "offset"})


def test_quasi_arithmetic_mean_rejects_callable_without_value_arg():
    """Both callables must take the value array via an argument named `value`."""

    def g(v: FloatND) -> FloatND:
        return v

    def g_inv(value: FloatND) -> FloatND:
        return value

    with pytest.raises(RegimeInitializationError, match="value"):
        QuasiArithmeticMean(transform=g, inverse=g_inv)


@categorical(ordered=False)
class _RegimeId:
    alive: ScalarInt
    dead: ScalarInt


def _utility_alive(consumption: ContinuousAction) -> FloatND:
    return jnp.log(consumption)


def _utility_dead(wealth: ContinuousState) -> FloatND:
    return jnp.sqrt(wealth)


def _next_wealth(
    wealth: ContinuousState, consumption: ContinuousAction
) -> ContinuousState:
    return wealth - consumption


def _budget(consumption: ContinuousAction, wealth: ContinuousState) -> BoolND:
    return consumption <= wealth


def _next_regime() -> ScalarInt:
    return _RegimeId.dead


_WEALTH = LinSpacedGrid(start=1.0, stop=10.0, n_points=5)
_CONSUMPTION = LinSpacedGrid(start=0.5, stop=5.0, n_points=5)


def _make_model(*, alive_kwargs: dict[str, Any], dead_kwargs: dict[str, Any]) -> Model:
    """Build a minimal two-regime model with extra kwargs spliced per regime."""
    base_alive: dict[str, Any] = {
        "transition": _next_regime,
        "states": {"wealth": _WEALTH},
        "state_transitions": {"wealth": _next_wealth},
        "actions": {"consumption": _CONSUMPTION},
        "constraints": {"budget": _budget},
        "functions": {"utility": _utility_alive},
        "active": lambda age: age < 41,
    }
    base_dead: dict[str, Any] = {
        "transition": None,
        "states": {"wealth": LinSpacedGrid(start=0.0, stop=10.0, n_points=5)},
        "functions": {"utility": _utility_dead},
    }
    alive = Regime(**(base_alive | alive_kwargs))
    dead = Regime(**(base_dead | dead_kwargs))
    return Model(
        regimes={"alive": alive, "dead": dead},
        ages=AgeGrid(start=40, stop=41, step="Y"),
        regime_id_class=_RegimeId,
    )


def test_regime_accepts_certainty_equivalent():
    """A non-terminal grid-search regime may declare a certainty equivalent."""
    model = _make_model(
        alive_kwargs={"certainty_equivalent": PowerMean()},
        dead_kwargs={},
    )
    assert model.user_regimes["alive"].certainty_equivalent is not None


def test_terminal_regime_rejects_certainty_equivalent():
    """Terminal regimes have no continuation to aggregate."""
    with pytest.raises(RegimeInitializationError, match=r"[Tt]erminal"):
        _make_model(
            alive_kwargs={},
            dead_kwargs={"certainty_equivalent": PowerMean()},
        )


def test_dcegm_rejects_certainty_equivalent():
    """DC-EGM's Euler inversion assumes expected utility; the guard names GridSearch."""
    dcegm = DCEGM(
        continuous_state="wealth",
        continuous_action="consumption",
        resources="resources",
        post_decision_function="savings",
        savings_grid=LinSpacedGrid(start=0.0, stop=10.0, n_points=5),
    )
    with pytest.raises(RegimeInitializationError, match="GridSearch"):
        _make_model(
            alive_kwargs={
                "certainty_equivalent": PowerMean(),
                "solver": dcegm,
            },
            dead_kwargs={},
        )


def test_certainty_equivalent_rejects_phased():
    """The certainty equivalent is phase-invariant; `Phased` is rejected."""
    with pytest.raises(RegimeInitializationError):
        _make_model(
            alive_kwargs={
                "certainty_equivalent": Phased(
                    solve=PowerMean(),
                    simulate=PowerMean(),
                ),
            },
            dead_kwargs={},
        )


def test_params_template_contains_certainty_equivalent_params():
    """CE params surface under the pseudo-function name `certainty_equivalent`."""
    model = _make_model(
        alive_kwargs={"certainty_equivalent": PowerMean()},
        dead_kwargs={},
    )
    template = model.get_params_template()
    assert template["alive"]["certainty_equivalent"] == {"risk_aversion": "float"}


def test_certainty_equivalent_name_collision_with_function_is_rejected():
    """A function named `certainty_equivalent` collides with the pseudo-function."""

    def certainty_equivalent(wealth: ContinuousState) -> FloatND:
        return wealth

    with pytest.raises(InvalidNameError, match="certainty_equivalent"):
        _make_model(
            alive_kwargs={
                "certainty_equivalent": PowerMean(),
                "functions": {
                    "utility": _utility_alive,
                    "certainty_equivalent": certainty_equivalent,
                },
            },
            dead_kwargs={},
        )


def test_nonlinear_certainty_equivalent_changes_solved_values():
    """With `risk_aversion = 2`, solved values differ from expected utility."""
    ez_model = get_model(certainty_equivalent=PowerMean())
    eu_model = get_model(certainty_equivalent=None)
    V_ez = ez_model.solve(params=get_params(risk_aversion=2.0), log_level="debug")
    V_eu = eu_model.solve(params=get_params(risk_aversion=None), log_level="debug")
    assert not np.allclose(
        np.asarray(V_ez[0]["alive"]), np.asarray(V_eu[0]["alive"]), rtol=1e-6
    )


def test_zero_risk_aversion_reduces_to_expected_utility():
    """`risk_aversion = 0` makes the power CE the linear expectation."""
    ez_model = get_model(certainty_equivalent=PowerMean())
    eu_model = get_model(certainty_equivalent=None)
    V_ez = ez_model.solve(params=get_params(risk_aversion=0.0), log_level="debug")
    V_eu = eu_model.solve(params=get_params(risk_aversion=None), log_level="debug")
    for period in V_eu:
        for regime_name in V_eu[period]:
            np.testing.assert_allclose(
                np.asarray(V_ez[period][regime_name]),
                np.asarray(V_eu[period][regime_name]),
                rtol=1e-5,
                err_msg=f"period={period}, regime={regime_name}",
            )


from lcm_examples.epstein_zin import (  # noqa: E402
    BAD_HEALTH_SURVIVAL_FACTOR,
    CONSUMPTION_GRID,
    DEAD_WEALTH_GRID,
    HEALTH_TRANSITION,
    INCOME,
    SURVIVAL_PROBS,
    WEALTH_GRID,
)


def _reference_transform_pair(
    risk_aversion: float,
) -> tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]:
    """Return the numpy transform pair `(g, g_inv)` of the power mean.

    `risk_aversion = 1` is the geometric-mean (log) limit.
    """
    if risk_aversion == 1.0:
        return np.log, np.exp
    exponent = 1.0 - risk_aversion

    def g(v: np.ndarray) -> np.ndarray:
        return v**exponent

    def g_inv(v: np.ndarray) -> np.ndarray:
        return v ** (1.0 / exponent)

    return g, g_inv


def _reference_backward_induction(
    *,
    risk_aversion: float,
    discount_factor: float,
    intertemporal_elasticity_of_substitution: float,
) -> tuple[dict[int, np.ndarray], np.ndarray]:
    """Independent numpy backward induction of the toy Epstein-Zin model.

    Mirrors the engine's computation order on the same grids: interpolate
    each target's V at next wealth, transform, average over health, weight
    by regime probabilities, invert, aggregate via the EZ `H`. Returns the
    per-period alive V arrays (shape `(n_wealth, n_health)`) and the
    period-0 argmax consumption (same shape).
    """
    wealth = np.linspace(WEALTH_GRID.start, WEALTH_GRID.stop, WEALTH_GRID.n_points)
    dead_wealth = np.linspace(
        DEAD_WEALTH_GRID.start, DEAD_WEALTH_GRID.stop, DEAD_WEALTH_GRID.n_points
    )
    consumption = np.linspace(
        CONSUMPTION_GRID.start, CONSUMPTION_GRID.stop, CONSUMPTION_GRID.n_points
    )
    health_transition = np.array(HEALTH_TRANSITION)
    g, g_inv = _reference_transform_pair(risk_aversion)
    rho = 1.0 - 1.0 / intertemporal_elasticity_of_substitution

    V_dead = np.sqrt(dead_wealth)
    n_decision_periods = len(SURVIVAL_PROBS)
    V_alive: dict[int, np.ndarray] = {}
    policy_c: dict[int, np.ndarray] = {}
    V_next: np.ndarray | None = None

    for period in reversed(range(n_decision_periods)):
        V_p = np.empty((len(wealth), 2))
        c_p = np.empty((len(wealth), 2))
        for iw, w in enumerate(wealth):
            for ih in range(2):
                survival = SURVIVAL_PROBS[period] * (
                    1.0 if ih == 1 else BAD_HEALTH_SURVIVAL_FACTOR
                )
                best_q, best_c = -np.inf, np.nan
                for c in consumption:
                    if c > w:
                        continue
                    w_next = w - c + INCOME
                    acc = (1.0 - survival) * g(np.interp(w_next, dead_wealth, V_dead))
                    if V_next is not None:
                        alive_vals = np.array(
                            [
                                np.interp(w_next, wealth, V_next[:, jh])
                                for jh in range(2)
                            ]
                        )
                        acc += survival * (health_transition[ih] @ g(alive_vals))
                    ce = g_inv(acc)
                    q = (
                        (1.0 - discount_factor) * c**rho + discount_factor * ce**rho
                    ) ** (1.0 / rho)
                    if q > best_q:
                        best_q, best_c = q, c
                V_p[iw, ih] = best_q
                c_p[iw, ih] = best_c
        V_alive[period] = V_p
        policy_c[period] = c_p
        V_next = V_p

    return V_alive, policy_c[0]


def test_power_mean_log_limit_is_geometric_mean():
    """At `risk_aversion = 1` the power-mean transform pair is `log`/`exp`."""
    ce = PowerMean()
    x = jnp.array([0.5, 1.0, 2.0, 7.5])
    one = jnp.asarray(1.0)
    np.testing.assert_allclose(
        ce.transform(value=x, risk_aversion=one), jnp.log(x), rtol=1e-6
    )
    np.testing.assert_allclose(
        ce.inverse(value=ce.transform(value=x, risk_aversion=one), risk_aversion=one),
        x,
        rtol=1e-6,
    )


@pytest.mark.parametrize("risk_aversion", [0.5, 1.0])
def test_epstein_zin_solved_values_match_numpy_reference(risk_aversion: float):
    """The solved alive-V equals an independent numpy backward induction.

    `risk_aversion = 1` exercises the geometric-mean (log) limit of the
    power mean, `CE = exp(E[log V'])`.
    """
    discount_factor, ies = 0.9, 2.0
    model = get_model(certainty_equivalent=PowerMean())
    solution = model.solve(
        params=get_params(
            risk_aversion=risk_aversion,
            discount_factor=discount_factor,
            intertemporal_elasticity_of_substitution=ies,
        ),
        log_level="debug",
    )
    expected, _ = _reference_backward_induction(
        risk_aversion=risk_aversion,
        discount_factor=discount_factor,
        intertemporal_elasticity_of_substitution=ies,
    )
    for period, expected_arr in expected.items():
        # Engine axis order: (health, wealth); reference: (wealth, health).
        np.testing.assert_allclose(
            np.asarray(solution[period]["alive"]),
            expected_arr.T,
            rtol=5e-5,
            err_msg=f"period={period}",
        )


def _stable_power_mean(
    values: tuple[float, ...],
    weights: tuple[float, ...],
    risk_aversion: float,
) -> float:
    """Evaluate the weighted power mean of a positive lottery in decimal arithmetic.

    An independent reference for `(Σ w · v^(1-ra))^(1/(1-ra))` on the
    mass-normalised lottery, carried at 120 significant digits so it is
    unaffected by the floating-point range that makes the naive
    `inverse(Σ w · transform(v))` route overflow. `risk_aversion = 1` is the
    weighted geometric mean.
    """
    with localcontext() as context:
        context.prec = 120
        vals = tuple(Decimal(str(v)) for v in values)
        raw_weights = tuple(Decimal(str(w)) for w in weights)
        mass = sum(raw_weights, start=Decimal(0))
        normalized = tuple(w / mass for w in raw_weights)
        exponent = Decimal(1) - Decimal(str(risk_aversion))
        logs = tuple(v.ln() for v in vals)
        if exponent == 0:
            log_mean = sum(
                (w * log_v for w, log_v in zip(normalized, logs, strict=True)),
                start=Decimal(0),
            )
            return float(log_mean.exp())
        # A negative exponent makes the smallest log the safe anchor, a positive
        # exponent the largest; either keeps every scaled exponent nonpositive.
        anchor = min(logs) if exponent < 0 else max(logs)
        scaled = sum(
            (
                w * (exponent * (log_v - anchor)).exp()
                for w, log_v in zip(normalized, logs, strict=True)
            ),
            start=Decimal(0),
        )
        return float((anchor + scaled.ln() / exponent).exp())


# Risk aversion and lottery scale combinations at which the naive
# `inverse(Σ w · transform(v))` route overflows the dtype.
_FLOAT64_STRESS_CASES = [(8.0, 1e-50), (12.0, 1e-30), (20.0, 1e-20), (50.0, 1e-8)]
_FLOAT32_STRESS_CASES = [(8.0, 1e-8), (12.0, 1e-5), (20.0, 1e-3)]


@pytest.mark.parametrize(("risk_aversion", "scale"), _FLOAT64_STRESS_CASES)
def test_power_mean_aggregate_matches_reference_at_float64_scales(
    x64_enabled: None,
    risk_aversion: float,
    scale: float,
):
    """The power mean of a tiny positive lottery equals its high-precision value."""
    values = (scale, 2.0 * scale)
    weights = (0.9, 0.1)
    got = PowerMean().aggregate(
        values=jnp.asarray(values, dtype=jnp.float64),
        weights=jnp.asarray(weights, dtype=jnp.float64),
        params={"risk_aversion": jnp.asarray(risk_aversion, dtype=jnp.float64)},
    )
    expected = _stable_power_mean(values, weights, risk_aversion)
    np.testing.assert_allclose(float(got), expected, rtol=1e-12, atol=0.0)


@pytest.mark.parametrize(("risk_aversion", "scale"), _FLOAT32_STRESS_CASES)
def test_power_mean_aggregate_matches_reference_at_float32_scales(
    x64_disabled: None,
    risk_aversion: float,
    scale: float,
):
    """The float32 power mean of a tiny positive lottery stays at its exact value."""
    values = (scale, 2.0 * scale)
    weights = (0.9, 0.1)
    got = PowerMean().aggregate(
        values=jnp.asarray(values, dtype=jnp.float32),
        weights=jnp.asarray(weights, dtype=jnp.float32),
        params={"risk_aversion": jnp.asarray(risk_aversion, dtype=jnp.float32)},
    )
    expected = _stable_power_mean(values, weights, risk_aversion)
    np.testing.assert_allclose(float(got), expected, rtol=5e-5, atol=0.0)


def test_power_mean_aggregate_is_geometric_mean_at_unit_risk_aversion():
    """At `risk_aversion = 1` the power mean is the weighted geometric mean."""
    got = PowerMean().aggregate(
        values=jnp.array([1.0, 9.0]),
        weights=jnp.array([0.5, 0.5]),
        params={"risk_aversion": jnp.asarray(1.0)},
    )
    np.testing.assert_allclose(float(got), 3.0, rtol=1e-6)


@pytest.mark.parametrize("risk_aversion", [1.0 - 1e-6, 1.0 + 1e-6])
def test_power_mean_aggregate_is_continuous_around_unit_risk_aversion(
    risk_aversion: float,
):
    """Just off `risk_aversion = 1` the power mean is still the geometric mean."""
    got = PowerMean().aggregate(
        values=jnp.array([1.0, 9.0]),
        weights=jnp.array([0.5, 0.5]),
        params={"risk_aversion": jnp.asarray(risk_aversion)},
    )
    np.testing.assert_allclose(float(got), 3.0, rtol=1e-5)


def test_power_mean_aggregate_is_linear_expectation_at_zero_risk_aversion():
    """At `risk_aversion = 0` the power mean is the probability-weighted mean."""
    got = PowerMean().aggregate(
        values=jnp.array([1.0, 9.0]),
        weights=jnp.array([0.25, 0.75]),
        params={"risk_aversion": jnp.asarray(0.0)},
    )
    np.testing.assert_allclose(float(got), 7.0, rtol=1e-6)


def test_power_mean_aggregate_drops_zero_weight_entries():
    """A zero-probability branch leaves the certainty equivalent unchanged."""
    kept = PowerMean().aggregate(
        values=jnp.array([1.0, 9.0]),
        weights=jnp.array([0.5, 0.5]),
        params={"risk_aversion": jnp.asarray(3.0)},
    )
    padded = PowerMean().aggregate(
        values=jnp.array([1.0, 9.0, 1e-30]),
        weights=jnp.array([0.5, 0.5, 0.0]),
        params={"risk_aversion": jnp.asarray(3.0)},
    )
    np.testing.assert_allclose(float(padded), float(kept), rtol=1e-12)


def test_power_mean_aggregate_is_homogeneous_of_degree_one():
    """Rescaling the lottery by `k > 0` rescales the certainty equivalent by `k`."""
    weights = jnp.array([0.3, 0.7])
    params = {"risk_aversion": jnp.asarray(6.0)}
    unit = PowerMean().aggregate(
        values=jnp.array([1.0, 4.0]), weights=weights, params=params
    )
    scaled = PowerMean().aggregate(
        values=jnp.array([1e-12, 4e-12]), weights=weights, params=params
    )
    np.testing.assert_allclose(float(scaled), 1e-12 * float(unit), rtol=1e-5)


def test_quasi_arithmetic_mean_aggregate_is_transform_sum_inverse():
    """A generic quasi-arithmetic mean aggregates as `g⁻¹(Σ w · g(v))`.

    Each callable receives exactly the runtime parameters its own signature
    declares: `transform` sees `theta`, `inverse` sees `theta` and `offset`.
    """

    def g(value: FloatND, theta: FloatND) -> FloatND:
        return value * theta

    def g_inv(value: FloatND, theta: FloatND, offset: FloatND) -> FloatND:
        return value / theta + offset

    got = QuasiArithmeticMean(transform=g, inverse=g_inv).aggregate(
        values=jnp.array([1.0, 3.0]),
        weights=jnp.array([0.25, 0.75]),
        params={"theta": jnp.asarray(2.0), "offset": jnp.asarray(0.5)},
    )
    # g: (2, 6); Σ w g(v) = 5; g_inv(5) = 5 / 2 + 0.5
    np.testing.assert_allclose(float(got), 3.0, rtol=1e-6)


@categorical(ordered=True)
class _Health:
    bad: ScalarInt
    good: ScalarInt


def _health_probs(health: DiscreteState) -> FloatND:
    return jnp.where(
        health == _Health.good, jnp.array([0.1, 0.9]), jnp.array([0.8, 0.2])
    )


def _survival_probs(health: DiscreteState, period: Period) -> FloatND:
    alive_next = jnp.where(health == _Health.good, 0.9, 0.7) * (period < 1)
    return jnp.array([alive_next, 1.0 - alive_next])


def _make_scale_equivariant_model(scale: float) -> Model:
    """Build a two-regime Epstein-Zin model whose value function scales with `scale`.

    Every primitive is homogeneous of degree one in `scale` - the grids, the
    income flow and both utilities - and `H_epstein_zin` and the power mean
    are themselves homogeneous of degree one. The solved value function of
    the model at `scale` is therefore exactly `scale` times the value
    function at `scale = 1`, and the optimal consumption grid point is
    identical.
    """

    def next_wealth(
        wealth: ContinuousState, consumption: ContinuousAction
    ) -> ContinuousState:
        return jnp.clip(wealth - consumption + 0.5 * scale, 0.5 * scale, 12.0 * scale)

    def utility_dead(wealth: ContinuousState) -> FloatND:
        return 0.5 * wealth + 1e-3 * scale

    def utility_alive(consumption: ContinuousAction) -> FloatND:
        return consumption

    alive = Regime(
        transition=MarkovTransition(_survival_probs),
        states={
            "wealth": LinSpacedGrid(start=0.5 * scale, stop=12.0 * scale, n_points=6),
            "health": DiscreteGrid(_Health),
        },
        state_transitions={
            "wealth": next_wealth,
            "health": {"alive": MarkovTransition(_health_probs)},
        },
        actions={
            "consumption": LinSpacedGrid(
                start=0.5 * scale, stop=5.0 * scale, n_points=7
            )
        },
        constraints={"budget": _budget},
        functions={"utility": utility_alive, "H": H_epstein_zin},
        certainty_equivalent=PowerMean(),
        active=lambda age: age < 27,
    )
    dead = Regime(
        transition=None,
        states={"wealth": LinSpacedGrid(start=0.0, stop=12.0 * scale, n_points=25)},
        functions={"utility": utility_dead},
    )
    return Model(
        regimes={"alive": alive, "dead": dead},
        ages=AgeGrid(start=25, stop=27, step="Y"),
        regime_id_class=_RegimeId,
    )


def _scaled_model_params(risk_aversion: float) -> dict:
    return {
        "alive": {
            "H": {
                "discount_factor": 0.9,
                "intertemporal_elasticity_of_substitution": 2.0,
            },
            "certainty_equivalent": {"risk_aversion": risk_aversion},
        },
        "dead": {},
    }


@pytest.mark.parametrize("risk_aversion", [2.0, 50.0])
def test_solved_values_are_equivariant_to_rescaling_the_model(
    x64_enabled: None,
    risk_aversion: float,
):
    """Scaling a homogeneous model by `k > 0` scales its solved values by `k`."""
    scale = 1e-7
    params = _scaled_model_params(risk_aversion)
    unit = _make_scale_equivariant_model(1.0).solve(params=params, log_level="debug")
    scaled = _make_scale_equivariant_model(scale).solve(
        params=params, log_level="debug"
    )
    for period in unit:
        for regime_name in unit[period]:
            np.testing.assert_allclose(
                np.asarray(scaled[period][regime_name]) / scale,
                np.asarray(unit[period][regime_name]),
                rtol=1e-6,
                err_msg=f"period={period}, regime={regime_name}",
            )
