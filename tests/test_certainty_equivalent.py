"""Tests for nonlinear certainty equivalents over the continuation value."""

import jax.numpy as jnp
import numpy as np
import pytest

from lcm import (
    AgeGrid,
    LinSpacedGrid,
    Model,
    Phased,
    PowerCertaintyEquivalent,
    Regime,
    TransformedExpectation,
    categorical,
)
from lcm.exceptions import InvalidNameError, RegimeInitializationError
from lcm.solvers import DCEGM
from lcm.typing import BoolND, ContinuousAction, ContinuousState, FloatND, ScalarInt
from tests.test_models.epstein_zin_health import get_model, get_params


def test_power_certainty_equivalent_transform_and_inverse_are_inverses():
    """`inverse(transform(x)) == x` for positive values."""
    ce = PowerCertaintyEquivalent()
    x = jnp.array([0.5, 1.0, 2.0, 7.5])
    roundtrip = ce.inverse(
        value=ce.transform(value=x, risk_aversion=jnp.asarray(0.5)),
        risk_aversion=jnp.asarray(0.5),
    )
    np.testing.assert_allclose(roundtrip, x, rtol=1e-6)


def test_power_certainty_equivalent_param_names():
    """The power CE declares exactly the `risk_aversion` runtime param."""
    assert PowerCertaintyEquivalent().param_names == frozenset({"risk_aversion"})


def test_transformed_expectation_param_names_union_over_both_callables():
    """`param_names` is the union of transform and inverse args minus `value`."""

    def g(value: FloatND, theta: FloatND) -> FloatND:
        return value * theta

    def g_inv(value: FloatND, theta: FloatND, offset: FloatND) -> FloatND:
        return value / theta + offset

    ce = TransformedExpectation(transform=g, inverse=g_inv)
    assert ce.param_names == frozenset({"theta", "offset"})


def test_transformed_expectation_rejects_callable_without_value_arg():
    """Both callables must take the value array via an argument named `value`."""

    def g(v: FloatND) -> FloatND:
        return v

    def g_inv(value: FloatND) -> FloatND:
        return value

    with pytest.raises(RegimeInitializationError, match="value"):
        TransformedExpectation(transform=g, inverse=g_inv)


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


def _make_model(*, alive_kwargs: dict, dead_kwargs: dict) -> Model:
    """Build a minimal two-regime model with extra kwargs spliced per regime."""
    base_alive = {
        "transition": _next_regime,
        "states": {"wealth": _WEALTH},
        "state_transitions": {"wealth": _next_wealth},
        "actions": {"consumption": _CONSUMPTION},
        "constraints": {"budget": _budget},
        "functions": {"utility": _utility_alive},
        "active": lambda age: age < 41,
    }
    base_dead = {
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
        alive_kwargs={"certainty_equivalent": PowerCertaintyEquivalent()},
        dead_kwargs={},
    )
    assert model.user_regimes["alive"].certainty_equivalent is not None


def test_terminal_regime_rejects_certainty_equivalent():
    """Terminal regimes have no continuation to aggregate."""
    with pytest.raises(RegimeInitializationError, match=r"[Tt]erminal"):
        _make_model(
            alive_kwargs={},
            dead_kwargs={"certainty_equivalent": PowerCertaintyEquivalent()},
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
                "certainty_equivalent": PowerCertaintyEquivalent(),
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
                    solve=PowerCertaintyEquivalent(),
                    simulate=PowerCertaintyEquivalent(),
                ),
            },
            dead_kwargs={},
        )


def test_params_template_contains_certainty_equivalent_params():
    """CE params surface under the pseudo-function name `certainty_equivalent`."""
    model = _make_model(
        alive_kwargs={"certainty_equivalent": PowerCertaintyEquivalent()},
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
                "certainty_equivalent": PowerCertaintyEquivalent(),
                "functions": {
                    "utility": _utility_alive,
                    "certainty_equivalent": certainty_equivalent,
                },
            },
            dead_kwargs={},
        )


def test_nonlinear_certainty_equivalent_changes_solved_values():
    """With `risk_aversion = 2`, solved values differ from expected utility."""
    ez_model = get_model(certainty_equivalent=PowerCertaintyEquivalent())
    eu_model = get_model(certainty_equivalent=None)
    V_ez = ez_model.solve(params=get_params(risk_aversion=2.0), log_level="debug")
    V_eu = eu_model.solve(params=get_params(risk_aversion=None), log_level="debug")
    assert not np.allclose(
        np.asarray(V_ez[0]["alive"]), np.asarray(V_eu[0]["alive"]), rtol=1e-6
    )


def test_zero_risk_aversion_reduces_to_expected_utility():
    """`risk_aversion = 0` makes the power CE the linear expectation."""
    ez_model = get_model(certainty_equivalent=PowerCertaintyEquivalent())
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
