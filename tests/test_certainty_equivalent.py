"""Tests for nonlinear certainty equivalents over the continuation value."""

from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

from lcm import (
    AgeGrid,
    LinSpacedGrid,
    Model,
    Phased,
    PowerMean,
    QuasiArithmeticMean,
    Regime,
    categorical,
)
from lcm.exceptions import InvalidNameError, RegimeInitializationError
from lcm.solvers import DCEGM, NBEGM
from lcm.taste_shocks import ExtremeValueTasteShocks
from lcm.typing import BoolND, ContinuousAction, ContinuousState, FloatND, ScalarInt
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


def test_power_mean_aggregate_stays_finite_for_small_values_high_risk_aversion():
    """`aggregate` returns the finite power mean where naive transform overflows.

    With `risk_aversion > 1` and continuation values near the borrowing
    constraint, the elementwise transform `v^(1 - risk_aversion)` overflows to
    infinity and the naive `inverse(mean(transform))` collapses the certainty
    equivalent to zero; the fused aggregation evaluates it in the log domain and
    returns the true finite value.
    """
    ce = PowerMean()
    values = jnp.array([1e-40, 2e-40])
    weights = jnp.array([0.5, 0.5])
    aggregated = ce.aggregate(
        values=values, weights=weights, risk_aversion=jnp.asarray(10.0)
    )
    # Log-domain reference for (0.5·v1^-9 + 0.5·v2^-9)^(-1/9).
    exponent = 1.0 - 10.0
    logs = exponent * np.log(np.asarray(values))
    shift = logs.max()
    log_ce = (shift + np.log(np.sum(0.5 * np.exp(logs - shift)))) / exponent
    np.testing.assert_allclose(np.asarray(aggregated), np.exp(log_ce), rtol=1e-10)
    assert np.isfinite(np.asarray(aggregated))
    assert np.asarray(aggregated) > 0.0


def test_power_mean_aggregate_matches_naive_form_on_well_scaled_values():
    """On well-scaled values the fused aggregation equals `g⁻¹(Σ w·g(v))`."""
    ce = PowerMean()
    values = jnp.array([0.5, 1.0, 2.0, 4.0])
    weights = jnp.array([0.1, 0.2, 0.3, 0.4])
    ra = jnp.asarray(3.0)
    aggregated = ce.aggregate(values=values, weights=weights, risk_aversion=ra)
    naive = ce.inverse(
        value=jnp.sum(weights * ce.transform(value=values, risk_aversion=ra)),
        risk_aversion=ra,
    )
    np.testing.assert_allclose(np.asarray(aggregated), np.asarray(naive), rtol=1e-10)


def test_power_mean_aggregate_log_limit_is_geometric_mean():
    """At `risk_aversion = 1` the aggregation is the weighted geometric mean."""
    ce = PowerMean()
    values = jnp.array([1.0, 2.0, 4.0])
    weights = jnp.array([0.25, 0.25, 0.5])
    aggregated = ce.aggregate(
        values=values, weights=weights, risk_aversion=jnp.asarray(1.0)
    )
    geometric = np.exp(np.sum(np.asarray(weights) * np.log(np.asarray(values))))
    np.testing.assert_allclose(np.asarray(aggregated), geometric, rtol=1e-10)


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


def test_nbegm_with_taste_shocks_rejects_certainty_equivalent():
    """Epstein-Zin and extreme-value taste shocks do not compose.

    The taste-shock logsum is not invariant under the certainty-equivalent
    transform, so a regime declaring both must error rather than silently mix an
    expected-utility smoothing with a recursive aggregator.
    """
    nbegm = NBEGM(
        post_decision_function="savings",
        budget_target="resources",
        savings_grid=LinSpacedGrid(start=0.0, stop=10.0, n_points=5),
    )
    with pytest.raises(RegimeInitializationError, match="taste_shocks"):
        _make_model(
            alive_kwargs={
                "certainty_equivalent": PowerMean(),
                "solver": nbegm,
                "taste_shocks": ExtremeValueTasteShocks(),
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
