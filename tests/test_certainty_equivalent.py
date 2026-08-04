"""Tests for nonlinear certainty equivalents over the continuation value."""

from collections.abc import Callable
from decimal import Decimal, localcontext
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.certainty_equivalent import power_inverse, power_transform
from lcm import (
    AgeGrid,
    CertaintyEquivalent,
    DiscreteGrid,
    LinearExpectation,
    LinSpacedGrid,
    MarkovTransition,
    Model,
    Phased,
    PowerMean,
    QuasiArithmeticMean,
    Regime,
    affine_breakpoint,
    categorical,
    fixed_transition,
    piecewise_affine,
)
from lcm.exceptions import InvalidNameError, RegimeInitializationError
from lcm.solvers import DCEGM, NBEGM, NNBEGM
from lcm.taste_shocks import ExtremeValueTasteShocks
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


def test_power_mean_aggregate_stays_finite_for_small_values_high_risk_aversion(
    x64_enabled: None,
):
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
        values=values,
        weights=weights,
        params={"risk_aversion": jnp.asarray(10.0)},
    )
    # Log-domain reference for (0.5·v1^-9 + 0.5·v2^-9)^(-1/9).
    exponent = 1.0 - 10.0
    logs = exponent * np.log(np.asarray(values))
    shift = logs.max()
    log_ce = (shift + np.log(np.sum(0.5 * np.exp(logs - shift)))) / exponent
    np.testing.assert_allclose(np.asarray(aggregated), np.exp(log_ce), rtol=1e-10)
    assert np.isfinite(np.asarray(aggregated))
    assert np.asarray(aggregated) > 0.0


def test_power_mean_aggregate_matches_naive_form_on_well_scaled_values(
    x64_enabled: None,
):
    """On well-scaled values the fused aggregation equals `g⁻¹(Σ w·g(v))`."""
    ce = PowerMean()
    values = jnp.array([0.5, 1.0, 2.0, 4.0])
    weights = jnp.array([0.1, 0.2, 0.3, 0.4])
    ra = jnp.asarray(3.0)
    aggregated = ce.aggregate(
        values=values, weights=weights, params={"risk_aversion": ra}
    )
    naive = ce.inverse(
        value=jnp.sum(weights * ce.transform(value=values, risk_aversion=ra)),
        risk_aversion=ra,
    )
    np.testing.assert_allclose(np.asarray(aggregated), np.asarray(naive), rtol=1e-10)


def test_power_mean_aggregate_log_limit_is_geometric_mean(x64_enabled: None):
    """At `risk_aversion = 1` the aggregation is the weighted geometric mean."""
    ce = PowerMean()
    values = jnp.array([1.0, 2.0, 4.0])
    weights = jnp.array([0.25, 0.25, 0.5])
    aggregated = ce.aggregate(
        values=values,
        weights=weights,
        params={"risk_aversion": jnp.asarray(1.0)},
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


def _resources(wealth: ContinuousState) -> FloatND:
    return wealth


def _savings(resources: FloatND, consumption: ContinuousAction) -> FloatND:
    return resources - consumption


_NBEGM_FUNCTIONS: dict[str, Any] = {
    "utility": _utility_alive,
    "resources": _resources,
    "savings": _savings,
}


def test_nbegm_rejects_a_non_power_mean_certainty_equivalent():
    """NBEGM implements the Epstein-Zin recursion for `PowerMean` only.

    The endogenous-grid kernels read the power mean's `risk_aversion` parameter
    and invert its generator in closed form; a general quasi-arithmetic mean has
    no such inverse-derivative interface, so declaring one with NBEGM must fail
    at model build rather than solve the wrong recursion.
    """

    def g(value: FloatND) -> FloatND:
        return jnp.log(value)

    def g_inv(value: FloatND) -> FloatND:
        return jnp.exp(value)

    nbegm = NBEGM(
        post_decision_function="savings",
        budget_target="resources",
        savings_grid=LinSpacedGrid(start=0.0, stop=10.0, n_points=5),
    )
    with pytest.raises(RegimeInitializationError, match="PowerMean"):
        _make_model(
            alive_kwargs={
                "certainty_equivalent": QuasiArithmeticMean(transform=g, inverse=g_inv),
                "solver": nbegm,
                "functions": dict(_NBEGM_FUNCTIONS),
            },
            dead_kwargs={},
        )


def test_nbegm_certainty_equivalent_requires_the_epstein_zin_aggregator():
    """NBEGM with a certainty equivalent needs `H_epstein_zin` as the regime's `H`.

    The Euler inversion and period value read the aggregator's intertemporal
    elasticity; with the default linear `H` the recursion the kernels implement
    is not the recursion the regime declares, so the combination must fail at
    model build.
    """
    nbegm = NBEGM(
        post_decision_function="savings",
        budget_target="resources",
        savings_grid=LinSpacedGrid(start=0.0, stop=10.0, n_points=5),
    )
    with pytest.raises(RegimeInitializationError, match="H_epstein_zin"):
        _make_model(
            alive_kwargs={
                "certainty_equivalent": PowerMean(),
                "solver": nbegm,
                "functions": dict(_NBEGM_FUNCTIONS),
            },
            dead_kwargs={},
        )


def test_nbegm_certainty_equivalent_requires_a_ride_along_route():
    """A zero-ride NBEGM regime cannot declare a certainty equivalent.

    The single-liquid-state smooth route solves the additive expected-utility
    step; only the ride-along route carries the Epstein-Zin kernels. Declaring
    a certainty equivalent on a regime without a ride-along state must fail at
    model build rather than silently solve the additive recursion.
    """
    nbegm = NBEGM(
        post_decision_function="savings",
        budget_target="resources",
        savings_grid=LinSpacedGrid(start=0.0, stop=10.0, n_points=5),
    )
    with pytest.raises(RegimeInitializationError, match="ride-along"):
        _make_model(
            alive_kwargs={
                "certainty_equivalent": PowerMean(),
                "solver": nbegm,
                "functions": dict(_NBEGM_FUNCTIONS) | {"H": H_epstein_zin},
            },
            dead_kwargs={},
        )


def test_nbegm_certainty_equivalent_rejects_a_jump_breakpoint():
    """EZ NBEGM covers smooth and pure-kink budgets; a jump is rejected at build.

    The unified jump-and-kink candidate step assumes the additive aggregator,
    so a regime combining a `certainty_equivalent` with a current-period jump
    breakpoint must fail when the model is built, not midway through a traced
    solve.
    """

    @categorical(ordered=False)
    class _Kind:
        lo: ScalarInt
        hi: ScalarInt

    def _u(consumption: ContinuousAction) -> FloatND:
        return consumption

    def _gross_income(wealth: ContinuousState, kind: DiscreteState) -> FloatND:
        return wealth + 0.5 * kind

    @piecewise_affine(
        output="subsidy",
        variable="gross_income",
        breakpoints=(affine_breakpoint(threshold="fpl_cliff", kind="jump"),),
    )
    def _subsidy(gross_income: FloatND, fpl_cliff: float) -> FloatND:
        return jnp.where(gross_income < fpl_cliff, 1.0, 0.0)

    def _jump_resources(gross_income: FloatND, subsidy: FloatND) -> FloatND:
        return gross_income + subsidy

    def _next_wealth_from_savings(savings: FloatND) -> ContinuousState:
        return savings

    nbegm = NBEGM(
        post_decision_function="savings",
        budget_target="resources",
        savings_grid=LinSpacedGrid(start=0.0, stop=10.0, n_points=5),
    )
    alive = Regime(
        transition=_next_regime,
        states={"wealth": _WEALTH, "kind": DiscreteGrid(_Kind)},
        state_transitions={
            "wealth": _next_wealth_from_savings,
            "kind": fixed_transition("kind"),
        },
        actions={"consumption": _CONSUMPTION},
        functions={
            "utility": _u,
            "gross_income": _gross_income,
            "subsidy": _subsidy,
            "resources": _jump_resources,
            "savings": _savings,
            "H": H_epstein_zin,
        },
        certainty_equivalent=PowerMean(),
        solver=nbegm,
        active=lambda age: age < 41,
    )

    def _dead_utility(wealth: ContinuousState, kind: DiscreteState) -> FloatND:
        return jnp.sqrt(wealth) + 0.0 * kind

    dead = Regime(
        transition=None,
        states={
            "wealth": LinSpacedGrid(start=0.0, stop=10.0, n_points=5),
            "kind": DiscreteGrid(_Kind),
        },
        functions={"utility": _dead_utility},
    )
    with pytest.raises(RegimeInitializationError, match="jump"):
        Model(
            regimes={"alive": alive, "dead": dead},
            ages=AgeGrid(start=40, stop=41, step="Y"),
            regime_id_class=_RegimeId,
        )


def test_nbegm_certainty_equivalent_rejects_a_varying_elasticity_flow():
    """A flow that is not a single power of consumption is rejected at build.

    The Epstein-Zin Euler inversion is closed-form only for `q = A c^phi` with
    `phi > 0`; the flow's consumption elasticity `c q'(c)/q(c)` is probed at
    several points, so a varying-elasticity flow (here `c + 0.1 c^2`) fails at
    model build instead of silently solving a locally fitted power's
    first-order condition.
    """

    @categorical(ordered=False)
    class _Kind:
        lo: ScalarInt
        hi: ScalarInt

    def _u(consumption: ContinuousAction, kind: DiscreteState) -> FloatND:
        return consumption + 0.1 * consumption**2 + 0.0 * kind

    def _ride_resources(wealth: ContinuousState, kind: DiscreteState) -> FloatND:
        return wealth + 0.5 * kind

    def _next_wealth_from_savings(savings: FloatND) -> ContinuousState:
        return savings

    def _dead_utility(wealth: ContinuousState, kind: DiscreteState) -> FloatND:
        return jnp.sqrt(wealth) + 0.0 * kind

    nbegm = NBEGM(
        post_decision_function="savings",
        budget_target="resources",
        savings_grid=LinSpacedGrid(start=0.0, stop=10.0, n_points=5),
    )
    alive = Regime(
        transition=_next_regime,
        states={"wealth": _WEALTH, "kind": DiscreteGrid(_Kind)},
        state_transitions={
            "wealth": _next_wealth_from_savings,
            "kind": fixed_transition("kind"),
        },
        actions={"consumption": _CONSUMPTION},
        functions={
            "utility": _u,
            "resources": _ride_resources,
            "savings": _savings,
            "H": H_epstein_zin,
        },
        certainty_equivalent=PowerMean(),
        solver=nbegm,
        active=lambda age: age < 41,
    )
    dead = Regime(
        transition=None,
        states={
            "wealth": LinSpacedGrid(start=0.0, stop=10.0, n_points=5),
            "kind": DiscreteGrid(_Kind),
        },
        functions={"utility": _dead_utility},
    )
    with pytest.raises(RegimeInitializationError, match="single power"):
        Model(
            regimes={"alive": alive, "dead": dead},
            ages=AgeGrid(start=40, stop=41, step="Y"),
            regime_id_class=_RegimeId,
        )


def test_nbegm_certainty_equivalent_accepts_a_single_power_flow_in_float32(
    x64_disabled,
):
    """A genuinely single-power flow builds under 32-bit precision.

    The elasticity probe differentiates the flow with `jax.grad`, whose
    roundoff scales with the active float dtype — under float32 the probed
    elasticities of `q = c` scatter by a few float32 ulps around one. The
    constancy window must scale with the dtype's precision so the probe keeps
    accepting the flows it is specified to accept.

    Grids capture the canonical float dtype at construction, so every grid is
    built inside the 32-bit scope — mirroring a process that runs in float32
    throughout.
    """

    @categorical(ordered=False)
    class _Kind:
        lo: ScalarInt
        hi: ScalarInt

    def _u(consumption: ContinuousAction, kind: DiscreteState) -> FloatND:
        return consumption * jnp.exp(0.1 * kind)

    def _ride_resources(wealth: ContinuousState, kind: DiscreteState) -> FloatND:
        return wealth + 0.5 * kind

    def _next_wealth_from_savings(savings: FloatND) -> ContinuousState:
        return savings

    def _dead_utility(wealth: ContinuousState, kind: DiscreteState) -> FloatND:
        return jnp.sqrt(wealth) + 0.0 * kind

    nbegm = NBEGM(
        post_decision_function="savings",
        budget_target="resources",
        savings_grid=LinSpacedGrid(start=0.0, stop=10.0, n_points=5),
    )
    alive = Regime(
        transition=_next_regime,
        states={
            "wealth": LinSpacedGrid(start=1.0, stop=10.0, n_points=5),
            "kind": DiscreteGrid(_Kind),
        },
        state_transitions={
            "wealth": _next_wealth_from_savings,
            "kind": fixed_transition("kind"),
        },
        actions={"consumption": LinSpacedGrid(start=0.5, stop=5.0, n_points=5)},
        functions={
            "utility": _u,
            "resources": _ride_resources,
            "savings": _savings,
            "H": H_epstein_zin,
        },
        certainty_equivalent=PowerMean(),
        solver=nbegm,
        active=lambda age: age < 41,
    )
    dead = Regime(
        transition=None,
        states={
            "wealth": LinSpacedGrid(start=0.0, stop=10.0, n_points=5),
            "kind": DiscreteGrid(_Kind),
        },
        functions={"utility": _dead_utility},
    )
    model = Model(
        regimes={"alive": alive, "dead": dead},
        ages=AgeGrid(start=40, stop=41, step="Y"),
        regime_id_class=_RegimeId,
    )
    template = model.get_params_template()
    assert template["alive"]["certainty_equivalent"] == {"risk_aversion": "float"}


def test_nbegm_certainty_equivalent_rejects_a_negative_flow():
    """A flow that is negative at the probe points is rejected at build.

    The Epstein-Zin recursion requires a strictly positive period flow
    `q = A c^phi` with `A > 0`: the power mean and the aggregator take
    fractional powers of it. A negative flow (here `-c`) has a *constant
    positive* consumption elasticity, so elasticity constancy alone cannot
    catch it — the probe must check the flow's sign directly.
    """

    @categorical(ordered=False)
    class _Kind:
        lo: ScalarInt
        hi: ScalarInt

    def _u(consumption: ContinuousAction, kind: DiscreteState) -> FloatND:
        return -consumption + 0.0 * kind

    def _ride_resources(wealth: ContinuousState, kind: DiscreteState) -> FloatND:
        return wealth + 0.5 * kind

    def _next_wealth_from_savings(savings: FloatND) -> ContinuousState:
        return savings

    def _dead_utility(wealth: ContinuousState, kind: DiscreteState) -> FloatND:
        return jnp.sqrt(wealth) + 0.0 * kind

    nbegm = NBEGM(
        post_decision_function="savings",
        budget_target="resources",
        savings_grid=LinSpacedGrid(start=0.0, stop=10.0, n_points=5),
    )
    alive = Regime(
        transition=_next_regime,
        states={"wealth": _WEALTH, "kind": DiscreteGrid(_Kind)},
        state_transitions={
            "wealth": _next_wealth_from_savings,
            "kind": fixed_transition("kind"),
        },
        actions={"consumption": _CONSUMPTION},
        functions={
            "utility": _u,
            "resources": _ride_resources,
            "savings": _savings,
            "H": H_epstein_zin,
        },
        certainty_equivalent=PowerMean(),
        solver=nbegm,
        active=lambda age: age < 41,
    )
    dead = Regime(
        transition=None,
        states={
            "wealth": LinSpacedGrid(start=0.0, stop=10.0, n_points=5),
            "kind": DiscreteGrid(_Kind),
        },
        functions={"utility": _dead_utility},
    )
    with pytest.raises(RegimeInitializationError, match="positive"):
        Model(
            regimes={"alive": alive, "dead": dead},
            ages=AgeGrid(start=40, stop=41, step="Y"),
            regime_id_class=_RegimeId,
        )


def test_nbegm_certainty_equivalent_rejects_a_liquid_reading_continuation():
    """EZ NBEGM rejects a continuation that depends on the current liquid state.

    When a next-period state law (or the regime transition) reads the current
    liquid state, the continuation differs by current-liquid interval and the
    per-interval candidate step applies. That step evaluates candidates with
    the additive expected-utility recursion, so combining it with a
    `certainty_equivalent` must fail at model build rather than silently
    compare candidates under the wrong objective.
    """

    @categorical(ordered=False)
    class _Kind:
        lo: ScalarInt
        hi: ScalarInt

    def _u(consumption: ContinuousAction, kind: DiscreteState) -> FloatND:
        return consumption + 0.0 * kind

    def _ride_resources(wealth: ContinuousState, kind: DiscreteState) -> FloatND:
        return wealth + 0.5 * kind

    def _next_wealth_with_transfer(
        savings: FloatND, wealth: ContinuousState
    ) -> ContinuousState:
        # An asset-tested transfer: piecewise-constant in the current liquid
        # state, so the interval-constancy probe passes and the continuation
        # routes through the per-interval step.
        return savings + jnp.where(wealth < 100.0, 0.4, 0.0)

    def _dead_utility(wealth: ContinuousState, kind: DiscreteState) -> FloatND:
        return jnp.sqrt(wealth) + 0.0 * kind

    nbegm = NBEGM(
        post_decision_function="savings",
        budget_target="resources",
        savings_grid=LinSpacedGrid(start=0.0, stop=10.0, n_points=5),
    )
    alive = Regime(
        transition=_next_regime,
        states={"wealth": _WEALTH, "kind": DiscreteGrid(_Kind)},
        state_transitions={
            "wealth": _next_wealth_with_transfer,
            "kind": fixed_transition("kind"),
        },
        actions={"consumption": _CONSUMPTION},
        functions={
            "utility": _u,
            "resources": _ride_resources,
            "savings": _savings,
            "H": H_epstein_zin,
        },
        certainty_equivalent=PowerMean(),
        solver=nbegm,
        active=lambda age: age < 41,
    )
    dead = Regime(
        transition=None,
        states={
            "wealth": LinSpacedGrid(start=0.0, stop=10.0, n_points=5),
            "kind": DiscreteGrid(_Kind),
        },
        functions={"utility": _dead_utility},
    )
    with pytest.raises(RegimeInitializationError, match="current liquid"):
        Model(
            regimes={"alive": alive, "dead": dead},
            ages=AgeGrid(start=40, stop=41, step="Y"),
            regime_id_class=_RegimeId,
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


def _minimal_nnbegm() -> Any:
    return NNBEGM(
        inner=NBEGM(
            continuous_state="wealth",
            post_decision_function="savings",
            budget_target="resources",
            savings_grid=LinSpacedGrid(start=0.0, stop=10.0, n_points=5),
        ),
        outer_action="investment",
        outer_post_decision="next_stock",
        outer_grid=LinSpacedGrid(start=0.0, stop=10.0, n_points=5),
        outer_no_adjustment_candidate="keep_stock",
    )


def test_nnbegm_rejects_a_non_power_mean_certainty_equivalent():
    """N-NB-EGM's inner solve runs the NBEGM kernels, so the same CE contract binds.

    The nested solver inherits the Epstein-Zin recursion from its inner NBEGM,
    which reads the power mean's `risk_aversion` in closed form; a general
    quasi-arithmetic mean must fail at model build for NNBEGM exactly as it
    does for standalone NBEGM.
    """

    def g(value: FloatND) -> FloatND:
        return jnp.log(value)

    def g_inv(value: FloatND) -> FloatND:
        return jnp.exp(value)

    with pytest.raises(RegimeInitializationError, match="PowerMean"):
        _make_model(
            alive_kwargs={
                "certainty_equivalent": QuasiArithmeticMean(transform=g, inverse=g_inv),
                "solver": _minimal_nnbegm(),
                "functions": dict(_NBEGM_FUNCTIONS),
            },
            dead_kwargs={},
        )


def test_nnbegm_certainty_equivalent_requires_the_epstein_zin_aggregator():
    """N-NB-EGM with a certainty equivalent needs `H_epstein_zin`, like NBEGM.

    The inner Euler inversion and period value read the aggregator's
    intertemporal elasticity; with the default linear `H` the nested solve
    would run a recursion the regime does not declare, so the combination
    must fail at model build.
    """
    with pytest.raises(RegimeInitializationError, match="H_epstein_zin"):
        _make_model(
            alive_kwargs={
                "certainty_equivalent": PowerMean(),
                "solver": _minimal_nnbegm(),
                "functions": dict(_NBEGM_FUNCTIONS),
            },
            dead_kwargs={},
        )


def test_power_mean_is_stable_one_ulp_from_unit_risk_aversion(
    x64_enabled: None,
) -> None:
    """One float64 step from `gamma = 1` the power mean sits on the geometric mean.

    `PowerMean.aggregate` divides a rounded log-sum by `1 - gamma`; at the
    representable neighbors of one that quotient must not lose the limit.
    """
    values = jnp.asarray([1.0, 4.0, 16.0])
    weights = jnp.asarray([0.25, 0.25, 0.5])
    geometric = float(jnp.exp(jnp.sum(weights * jnp.log(values))))

    for gamma in (
        np.nextafter(np.float64(1.0), np.float64(np.inf)),
        np.nextafter(np.float64(1.0), np.float64(-np.inf)),
    ):
        got = PowerMean().aggregate(
            values=values,
            weights=weights,
            params={"risk_aversion": jnp.asarray(gamma)},
        )
        np.testing.assert_allclose(float(got), geometric, rtol=1e-8)


def test_power_mean_is_stable_near_unit_gamma_for_quadrature_roundoff_mass(
    x64_enabled: None,
) -> None:
    """Quadrature weights whose float sum is one ULP below one hit the limit.

    A mathematically normalized lottery need not sum to one bit-exactly —
    normalized five-node Gauss-Hermite weights sum to `1 - 1 ULP` in float64.
    A roundoff-scale mass gap must not leak into the `log(W)/(1-gamma)` mass
    term, so at and one ULP around `gamma = 1` the power mean sits on the
    normalized weighted geometric mean.
    """
    _, raw_weights = np.polynomial.hermite.hermgauss(5)
    weights = jnp.asarray(raw_weights / np.sqrt(np.pi))
    values = jnp.asarray(np.exp(np.linspace(0.0, 2.0, 5)))

    normalized = weights / jnp.sum(weights)
    geometric = float(jnp.exp(jnp.sum(normalized * jnp.log(values))))

    for gamma in (
        np.nextafter(np.float64(1.0), np.float64(np.inf)),
        np.float64(1.0),
        np.nextafter(np.float64(1.0), np.float64(-np.inf)),
    ):
        got = PowerMean().aggregate(
            values=values,
            weights=weights,
            params={"risk_aversion": jnp.asarray(gamma)},
        )
        np.testing.assert_allclose(float(got), geometric, rtol=1e-8)


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


def _aggregate_power_mean(
    values: tuple[float, ...],
    weights: tuple[float, ...],
    risk_aversion: float,
    *,
    dtype: Any,
    execution: str = "eager",
) -> float:
    """Aggregate a lottery through `PowerMean`, eagerly or under `jax.jit`."""

    def call() -> FloatND:
        return PowerMean().aggregate(
            values=jnp.asarray(values, dtype=dtype),
            weights=jnp.asarray(weights, dtype=dtype),
            params={"risk_aversion": jnp.asarray(risk_aversion, dtype=dtype)},
        )

    return float(jax.jit(call)() if execution == "jit" else call())


def _tiny_anchor_lottery(
    *,
    n_nodes: int,
    anchor_weight: float,
    anchor_value: float,
    anchor_position: str,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Build a lottery whose lowest value carries almost none of the mass.

    The remaining mass sits on values of order one, so above unit risk
    aversion every non-anchor node's scaled exponential vanishes against the
    anchor and the complementary mass rounds to the full unit mass. The power
    mean is nevertheless finite and strictly positive, since the anchor keeps
    a positive share of it.
    """
    rest = (1.0 - anchor_weight) / (n_nodes - 1)
    values = (anchor_value, *(1.0 + 0.1 * i for i in range(n_nodes - 1)))
    weights = (anchor_weight, *((rest,) * (n_nodes - 1)))
    if anchor_position == "last":
        return values[::-1], weights[::-1]
    return values, weights


_TINY_ANCHOR_N_NODES = (2, 3, 5, 11)
_FLOAT64_ANCHOR_WEIGHTS = (1e-12, 1e-16, 1e-20, 1e-40)
_FLOAT32_ANCHOR_WEIGHTS = (1e-4, 1e-6, 1e-8, 1e-12)


@pytest.mark.parametrize("execution", ["eager", "jit"])
@pytest.mark.parametrize("anchor_position", ["first", "last"])
@pytest.mark.parametrize("anchor_weight", _FLOAT64_ANCHOR_WEIGHTS)
@pytest.mark.parametrize("n_nodes", _TINY_ANCHOR_N_NODES)
def test_power_mean_aggregate_keeps_a_tiny_anchor_mass_at_float64(
    x64_enabled: None,
    n_nodes: int,
    anchor_weight: float,
    anchor_position: str,
    execution: str,
):
    """A near-zero weight on the lowest value still gives the exact power mean."""
    values, weights = _tiny_anchor_lottery(
        n_nodes=n_nodes,
        anchor_weight=anchor_weight,
        anchor_value=1e-50,
        anchor_position=anchor_position,
    )
    got = _aggregate_power_mean(
        values, weights, 8.0, dtype=jnp.float64, execution=execution
    )
    expected = _stable_power_mean(values, weights, 8.0)
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("execution", ["eager", "jit"])
@pytest.mark.parametrize("anchor_position", ["first", "last"])
@pytest.mark.parametrize("anchor_weight", _FLOAT32_ANCHOR_WEIGHTS)
@pytest.mark.parametrize("n_nodes", _TINY_ANCHOR_N_NODES)
def test_power_mean_aggregate_keeps_a_tiny_anchor_mass_at_float32(
    x64_disabled: None,
    n_nodes: int,
    anchor_weight: float,
    anchor_position: str,
    execution: str,
):
    """The float32 aggregation keeps a near-zero anchor mass too."""
    values, weights = _tiny_anchor_lottery(
        n_nodes=n_nodes,
        anchor_weight=anchor_weight,
        anchor_value=1e-8,
        anchor_position=anchor_position,
    )
    got = _aggregate_power_mean(
        values, weights, 8.0, dtype=jnp.float32, execution=execution
    )
    expected = _stable_power_mean(values, weights, 8.0)
    np.testing.assert_allclose(got, expected, rtol=5e-5, atol=0.0)


def test_power_mean_aggregate_at_a_float64_witness_anchor_weight(x64_enabled: None):
    """`(1e-50, 1)` at weights `(1e-20, 1)` and risk aversion 8 is `7.1969e-48`."""
    got = _aggregate_power_mean(
        (1e-50, 1.0), (1e-20, 1.0 - 1e-20), 8.0, dtype=jnp.float64
    )
    np.testing.assert_allclose(got, 7.196856730011521e-48, rtol=1e-12, atol=0.0)


def test_power_mean_aggregate_at_a_float32_witness_anchor_weight(x64_disabled: None):
    """`(1e-8, 1)` at weights `(1e-8, 1)` and risk aversion 8 is `1.3895e-7`."""
    got = _aggregate_power_mean((1e-8, 1.0), (1e-8, 1.0 - 1e-8), 8.0, dtype=jnp.float32)
    np.testing.assert_allclose(got, 1.3894954943731376e-7, rtol=5e-5, atol=0.0)


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


def test_certainty_equivalent_subclass_must_supply_an_aggregation():
    """Aggregating the continuation lottery is part of the interface, not an extra.

    A subclass that declares only its parameter names cannot be instantiated,
    so the gap surfaces where the class is written rather than part-way
    through a solve.
    """

    class OnlyParamNames(CertaintyEquivalent):
        @property
        def param_names(self) -> frozenset[str]:
            return frozenset()

    with pytest.raises(TypeError, match="aggregate"):
        OnlyParamNames()


@pytest.mark.parametrize("overridden", ["transform", "inverse"])
def test_power_mean_rejects_a_custom_transform_or_inverse(overridden: str):
    """`PowerMean` aggregates the power transform; a custom pair needs the base class.

    Its stable aggregation is specific to `v^(1-risk_aversion)`, so accepting
    other callables would silently ignore them.
    """

    def g(value: FloatND, theta: FloatND) -> FloatND:
        return value * theta

    with pytest.raises(RegimeInitializationError, match="QuasiArithmeticMean"):
        PowerMean(**{overridden: g})


def test_power_mean_aggregate_is_zero_at_a_zero_valued_node(x64_enabled: None):
    """A zero-valued branch drives the power mean to zero above unit risk aversion.

    With `risk_aversion = 3` the transform `v^-2` sends the zero branch to
    infinity, so the mean of the transformed lottery is infinite and the
    certainty equivalent is `inf^(-1/2) = 0`.
    """
    got = _aggregate_power_mean((0.0, 1.0), (0.5, 0.5), 3.0, dtype=jnp.float64)
    np.testing.assert_allclose(got, 0.0, atol=0.0)


def test_power_mean_aggregate_is_zero_at_an_underflowed_node(x64_disabled: None):
    """A float32 branch that underflows to zero yields a zero certainty equivalent."""
    got = _aggregate_power_mean((1e-50, 1.0), (0.5, 0.5), 3.0, dtype=jnp.float32)
    np.testing.assert_allclose(got, 0.0, atol=0.0)


def test_power_mean_aggregate_is_nan_for_a_zero_mass_lottery():
    """A lottery carrying no probability mass has no certainty equivalent.

    The linear-expectation path reports the same undefined aggregate, so a
    state-action point whose regime transition assigns zero probability to
    every reachable target is caught by the NaN diagnostics either way.
    """
    got = PowerMean().aggregate(
        values=jnp.array([1.0, 9.0]),
        weights=jnp.array([0.0, 0.0]),
        params={"risk_aversion": jnp.asarray(3.0)},
    )
    assert jnp.isnan(got)


@pytest.mark.parametrize("bad_weight", [jnp.nan, jnp.inf])
def test_power_mean_aggregate_propagates_a_non_finite_weight(bad_weight: float):
    """A malformed node weight makes the whole aggregate NaN rather than dropping."""
    got = PowerMean().aggregate(
        values=jnp.array([1.0, 9.0]),
        weights=jnp.array([0.5, bad_weight]),
        params={"risk_aversion": jnp.asarray(3.0)},
    )
    assert jnp.isnan(got)


def test_power_mean_aggregate_has_finite_gradients_at_a_dead_zero_valued_node(
    x64_enabled: None,
):
    """A zero-weight branch at value zero leaves every gradient finite.

    A target regime reached with probability zero is ordinary — a survival
    transition in the last active period, say — so gradient-based estimation
    must not see NaN because of a branch that carries no mass.
    """
    values = jnp.array([1.0, 9.0, 0.0])
    weights = jnp.array([0.5, 0.5, 0.0])

    def aggregate_over_values(values: FloatND) -> FloatND:
        return PowerMean().aggregate(
            values=values,
            weights=weights,
            params={"risk_aversion": jnp.asarray(3.0)},
        )

    def aggregate_over_risk_aversion(risk_aversion: FloatND) -> FloatND:
        return PowerMean().aggregate(
            values=values, weights=weights, params={"risk_aversion": risk_aversion}
        )

    assert jnp.all(jnp.isfinite(jax.grad(aggregate_over_values)(values)))
    assert jnp.isfinite(
        jax.grad(aggregate_over_risk_aversion)(jnp.asarray(3.0)),
    )


def test_power_mean_aggregate_normalizes_a_non_unit_mass_lottery():
    """Scaling every weight by a constant leaves the certainty equivalent unchanged."""
    params = {"risk_aversion": jnp.asarray(3.0)}
    unit = PowerMean().aggregate(
        values=jnp.array([1.0, 9.0]), weights=jnp.array([0.25, 0.75]), params=params
    )
    scaled = PowerMean().aggregate(
        values=jnp.array([1.0, 9.0]), weights=jnp.array([2.5, 7.5]), params=params
    )
    np.testing.assert_allclose(float(scaled), float(unit), rtol=1e-12)


@pytest.mark.parametrize("risk_aversion", [1.0 - 1e-6, 1.0, 1.0 + 1e-6])
def test_power_mean_aggregate_is_continuous_across_unit_risk_aversion_off_unit_mass(
    risk_aversion: float,
):
    """A lottery whose weights miss unit mass stays continuous in risk aversion."""
    got = PowerMean().aggregate(
        values=jnp.array([1.0, 9.0]),
        weights=jnp.array([0.5, 0.5 - 1e-4]),
        params={"risk_aversion": jnp.asarray(risk_aversion)},
    )
    np.testing.assert_allclose(float(got), 3.0, rtol=1e-3)


def test_power_mean_aggregate_reports_a_bad_param_type_as_invalid_params():
    """A malformed runtime param points the user at their params, not their regime."""
    with pytest.raises(InvalidParamsError):
        PowerMean().aggregate(
            values=jnp.array([1.0, 9.0]),
            weights=jnp.array([0.5, 0.5]),
            params={"risk_aversion": "two"},  # ty: ignore[invalid-argument-type]
        )


def _power_transform(value: FloatND, risk_aversion: FloatND) -> FloatND:
    return value ** (1.0 - risk_aversion)


def _power_inverse(value: FloatND, risk_aversion: FloatND) -> FloatND:
    return value ** (1.0 / (1.0 - risk_aversion))


def test_solve_with_a_generic_quasi_arithmetic_mean_matches_the_power_mean():
    """A hand-written power transform on a `Regime` solves to the `PowerMean` values."""
    generic_model = get_model(
        certainty_equivalent=QuasiArithmeticMean(
            transform=_power_transform, inverse=_power_inverse
        )
    )
    power_model = get_model(certainty_equivalent=PowerMean())
    params = get_params(risk_aversion=2.0)
    V_generic = generic_model.solve(params=params, log_level="debug")
    V_power = power_model.solve(params=params, log_level="debug")
    for period in V_power:
        for regime_name in V_power[period]:
            np.testing.assert_allclose(
                np.asarray(V_generic[period][regime_name]),
                np.asarray(V_power[period][regime_name]),
                rtol=1e-6,
                err_msg=f"period={period}, regime={regime_name}",
            )


def test_quasi_arithmetic_mean_aggregate_normalizes_a_non_unit_mass_lottery():
    """Scaling every weight by a constant leaves the generic mean unchanged."""

    def g(value: FloatND, theta: FloatND) -> FloatND:
        return value * theta

    def g_inv(value: FloatND, theta: FloatND) -> FloatND:
        return value / theta

    ce = QuasiArithmeticMean(transform=g, inverse=g_inv)
    params = {"theta": jnp.asarray(2.0)}
    unit = ce.aggregate(
        values=jnp.array([1.0, 3.0]), weights=jnp.array([0.25, 0.75]), params=params
    )
    scaled = ce.aggregate(
        values=jnp.array([1.0, 3.0]), weights=jnp.array([2.5, 7.5]), params=params
    )
    np.testing.assert_allclose(float(scaled), float(unit), rtol=1e-12)


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
    income flow and both utilities - and `W_epstein_zin` and the power mean
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
        functions={"utility": utility_alive},
        koopmans_aggregator=W_epstein_zin,
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
            "koopmans_aggregator": {
                "discount_factor": 0.9,
                "intertemporal_elasticity_of_substitution": 2.0,
            },
            "certainty_equivalent": {"risk_aversion": risk_aversion},
        },
        "dead": {},
    }


def _make_mixed_target_model(scale: float) -> Model:
    """Build a scale-equivariant model whose terminal regime carries no state.

    `alive` reaches a stateful target — itself, with a stochastic health node
    — and a stateless one, so its continuation lottery mixes two interpolated
    nodes from an array-valued `V` with a single node read off a scalar `V`.
    """

    def next_wealth(
        wealth: ContinuousState, consumption: ContinuousAction
    ) -> ContinuousState:
        return jnp.clip(wealth - consumption + 0.5 * scale, 0.5 * scale, 12.0 * scale)

    def utility_dead() -> FloatND:
        return jnp.asarray(0.25 * scale)

    def utility_alive(consumption: ContinuousAction) -> FloatND:
        return consumption

    alive = Regime(
        transition=MarkovTransition(_survival_probs),
        states={
            "wealth": LinSpacedGrid(start=0.5 * scale, stop=12.0 * scale, n_points=6),
            "health": DiscreteGrid(_Health),
        },
        state_transitions={
            "wealth": {"alive": next_wealth},
            "health": {"alive": MarkovTransition(_health_probs)},
        },
        actions={
            "consumption": LinSpacedGrid(
                start=0.5 * scale, stop=5.0 * scale, n_points=7
            )
        },
        constraints={"budget": _budget},
        functions={"utility": utility_alive},
        koopmans_aggregator=W_epstein_zin,
        certainty_equivalent=PowerMean(),
        active=lambda age: age < 27,
    )
    dead = Regime(transition=None, states={}, functions={"utility": utility_dead})
    return Model(
        regimes={"alive": alive, "dead": dead},
        ages=AgeGrid(start=25, stop=27, step="Y"),
        regime_id_class=_RegimeId,
    )


def _assert_solved_values_are_equivariant(
    *,
    make_model: Callable[[float], Model],
    scale: float,
    risk_aversion: float,
    rtol: float,
) -> None:
    """Assert the model solved at `scale` is `scale` times the one solved at one."""
    params = _scaled_model_params(risk_aversion)
    unit = make_model(1.0).solve(params=params, log_level="debug")
    scaled = make_model(scale).solve(params=params, log_level="debug")
    for period in unit:
        for regime_name in unit[period]:
            np.testing.assert_allclose(
                np.asarray(scaled[period][regime_name]) / scale,
                np.asarray(unit[period][regime_name]),
                rtol=rtol,
                err_msg=f"period={period}, regime={regime_name}",
            )


@pytest.mark.parametrize("risk_aversion", [2.0, 50.0])
def test_solved_values_are_equivariant_to_rescaling_the_model(
    x64_enabled: None,
    risk_aversion: float,
):
    """Scaling a homogeneous model by `k > 0` scales its solved values by `k`."""
    _assert_solved_values_are_equivariant(
        make_model=_make_scale_equivariant_model,
        scale=1e-7,
        risk_aversion=risk_aversion,
        rtol=1e-6,
    )


@pytest.mark.parametrize("risk_aversion", [2.0, 20.0])
def test_solved_values_are_equivariant_to_rescaling_the_model_float32(
    x64_disabled: None,
    risk_aversion: float,
):
    """The float32 solve is equivariant too, at scales that overflow the naive route."""
    _assert_solved_values_are_equivariant(
        make_model=_make_scale_equivariant_model,
        scale=1e-3,
        risk_aversion=risk_aversion,
        rtol=1e-3,
    )


@pytest.mark.parametrize("risk_aversion", [2.0, 50.0])
def test_solved_values_are_equivariant_with_a_stateless_target_regime(
    x64_enabled: None,
    risk_aversion: float,
):
    """A lottery mixing a stateless and a stateful target aggregates equivariantly."""
    _assert_solved_values_are_equivariant(
        make_model=_make_mixed_target_model,
        scale=1e-7,
        risk_aversion=risk_aversion,
        rtol=1e-6,
    )


@pytest.mark.parametrize("risk_aversion", [2.0, 20.0])
def test_solved_values_are_equivariant_with_a_stateless_target_regime_float32(
    x64_disabled: None,
    risk_aversion: float,
):
    """The mixed stateless/stateful lottery is equivariant at float32 too."""
    _assert_solved_values_are_equivariant(
        make_model=_make_mixed_target_model,
        scale=1e-3,
        risk_aversion=risk_aversion,
        rtol=1e-3,
    )


def test_linear_expectation_aggregate_is_the_mass_normalized_mean():
    """`LinearExpectation` aggregates a lottery to its probability-weighted mean."""
    got = LinearExpectation().aggregate(
        values=jnp.array([1.0, 9.0]),
        weights=jnp.array([0.5, 1.5]),
        params={},
    )
    # Normalized weights are (0.25, 0.75), so the mean is 7.0.
    np.testing.assert_allclose(float(got), 7.0, rtol=1e-6)


def test_linear_expectation_takes_no_runtime_parameters():
    """The plain expectation has nothing to parameterize."""
    assert LinearExpectation().param_names == frozenset()


def test_linear_expectation_solves_to_the_same_values_as_the_generic_route():
    """The engine's per-target reduction agrees with the reference aggregation.

    A regime declaring `LinearExpectation` is reduced target by target, which
    is cheaper than flattening the joint lottery. Its `aggregate` states the
    same quantity the long way round, so solving either way must agree.
    """
    fast_model = get_model(certainty_equivalent=LinearExpectation())
    reference_model = get_model(
        certainty_equivalent=QuasiArithmeticMean(transform=_identity, inverse=_identity)
    )
    params = get_params(risk_aversion=None)
    V_fast = fast_model.solve(params=params, log_level="debug")
    V_reference = reference_model.solve(params=params, log_level="debug")
    for period in V_fast:
        for regime_name in V_fast[period]:
            np.testing.assert_allclose(
                np.asarray(V_fast[period][regime_name]),
                np.asarray(V_reference[period][regime_name]),
                rtol=1e-5,
                err_msg=f"period={period}, regime={regime_name}",
            )


def _identity(value: FloatND) -> FloatND:
    return value


@pytest.mark.parametrize("risk_aversion", [0.0, 0.5, 2.0, 5.0])
def test_power_mean_aggregate_matches_its_own_transform_and_inverse(
    x64_enabled: None,
    risk_aversion: float,
):
    """The anchored form agrees with the pair it is the stable evaluation of.

    `PowerMean.transform` and `PowerMean.inverse` define the mean; `aggregate`
    evaluates it in a form that survives ranges where applying them directly
    overflows. Where both are valid they must agree.
    """
    values = jnp.array([0.4, 1.0, 3.0, 7.5])
    weights = jnp.array([0.1, 0.2, 0.3, 0.4])
    params = {"risk_aversion": jnp.asarray(risk_aversion)}
    anchored = PowerMean().aggregate(values=values, weights=weights, params=params)
    naive = QuasiArithmeticMean(
        transform=power_transform, inverse=power_inverse
    ).aggregate(values=values, weights=weights, params=params)
    np.testing.assert_allclose(float(anchored), float(naive), rtol=1e-12)


@categorical(ordered=False)
class _StackedRegimeId:
    working: ScalarInt
    retired: ScalarInt
    dead: ScalarInt


def _to_retired() -> ScalarInt:
    return _StackedRegimeId.retired


def _to_dead() -> ScalarInt:
    return _StackedRegimeId.dead


def _make_stacked_model(
    *,
    model_kwargs: dict[str, Any],
    working_kwargs: dict[str, Any],
    retired_kwargs: dict[str, Any],
) -> Model:
    """Build a model with two non-terminal regimes and one terminal regime."""
    base: dict[str, Any] = {
        "states": {"wealth": _WEALTH},
        "state_transitions": {"wealth": _next_wealth},
        "actions": {"consumption": _CONSUMPTION},
        "constraints": {"budget": _budget},
        "functions": {"utility": _utility_alive},
    }
    working: dict[str, Any] = base | {"transition": _to_retired} | working_kwargs
    retired: dict[str, Any] = base | {"transition": _to_dead} | retired_kwargs
    return Model(
        regimes={
            "working": Regime(**working),
            "retired": Regime(**retired),
            "dead": Regime(
                transition=None,
                states={"wealth": LinSpacedGrid(start=0.0, stop=10.0, n_points=5)},
                functions={"utility": _utility_dead},
            ),
        },
        ages=AgeGrid(start=40, stop=42, step="Y"),
        regime_id_class=_StackedRegimeId,
        **model_kwargs,
    )


def test_default_certainty_equivalent_is_the_linear_expectation():
    """A regime that declares nothing aggregates the continuation linearly."""
    model = _make_stacked_model(model_kwargs={}, working_kwargs={}, retired_kwargs={})
    assert model.user_regimes["working"].certainty_equivalent == LinearExpectation()
    assert model.user_regimes["retired"].certainty_equivalent == LinearExpectation()


def test_model_level_certainty_equivalent_reaches_every_non_terminal_regime():
    """One model-level declaration serves all regimes with a continuation."""
    model = _make_stacked_model(
        model_kwargs={"certainty_equivalent": PowerMean()},
        working_kwargs={},
        retired_kwargs={},
    )
    assert model.user_regimes["working"].certainty_equivalent == PowerMean()
    assert model.user_regimes["retired"].certainty_equivalent == PowerMean()


def test_model_level_certainty_equivalent_is_withheld_from_terminal_regimes():
    """A terminal regime has no continuation, so it receives no aggregation."""
    model = _make_stacked_model(
        model_kwargs={"certainty_equivalent": PowerMean()},
        working_kwargs={},
        retired_kwargs={},
    )
    assert model.user_regimes["dead"].certainty_equivalent is None


def test_regime_level_certainty_equivalents_replace_the_model_level_one():
    """Declaring one in every regime with a continuation ignores the model level."""
    model = _make_stacked_model(
        model_kwargs={"certainty_equivalent": PowerMean()},
        working_kwargs={"certainty_equivalent": LinearExpectation()},
        retired_kwargs={"certainty_equivalent": LinearExpectation()},
    )
    assert model.user_regimes["working"].certainty_equivalent == LinearExpectation()
    assert model.user_regimes["retired"].certainty_equivalent == LinearExpectation()


def test_declaring_a_certainty_equivalent_in_only_some_regimes_is_rejected():
    """The certainty equivalent is declared once for the model, or once per regime.

    Declaring it in some regimes and leaving others on the model-level value
    mixes the two, which reads as if the silent regimes had opted out of
    something they never saw.
    """
    with pytest.raises(ModelInitializationError, match="retired"):
        _make_stacked_model(
            model_kwargs={},
            working_kwargs={"certainty_equivalent": PowerMean()},
            retired_kwargs={},
        )
