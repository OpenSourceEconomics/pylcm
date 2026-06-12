"""Spec for DC-EGM build-time validation (one case per contract rule).

A regime with `solver=DCEGM(...)` must satisfy the EGM contract; every violation
raises `ModelInitializationError` at `Model` construction with a message naming
the offending piece. The cases here mutate a valid DC-EGM regime one rule at a
time.
"""

import dataclasses

import jax.numpy as jnp
import pytest

from lcm import (
    AgeGrid,
    DiscreteGrid,
    IrregSpacedGrid,
    LinSpacedGrid,
    MarkovTransition,
    Model,
    Phased,
)
from lcm.exceptions import ModelInitializationError
from lcm.regime import Regime as UserRegime
from lcm.solvers import BruteForce
from lcm.transition import fixed_transition
from lcm.typing import (
    ContinuousAction,
    ContinuousState,
    FloatND,
    Period,
    ScalarInt,
)
from lcm_examples.mortality import (
    borrowing_constraint,
    dead,
    next_wealth,
    utility_retirement,
)
from tests.test_models.deterministic import (
    base,
    dcegm_variants,
    retirement_only,
)

N_PERIODS = 3


def _build_model(regime: UserRegime) -> Model:
    ages = AgeGrid(start=40, stop=40 + (N_PERIODS - 1) * 10, step="10Y")
    return Model(
        regimes={"retirement": regime, "dead": dead},
        ages=ages,
        regime_id_class=retirement_only.RetirementOnlyRegimeId,
    )


def _utility_with_direct_wealth_dependence(
    consumption: ContinuousAction, wealth: ContinuousState
) -> FloatND:
    return jnp.log(consumption) + 0.01 * wealth


def _custom_H(utility: FloatND, continuation: FloatND) -> FloatND:
    return utility + 0.9 * continuation


def _regime_transition_with_wealth_cliff(wealth: ContinuousState) -> ScalarInt:
    return jnp.where(
        wealth < 100.0,
        retirement_only.RetirementOnlyRegimeId.dead,
        retirement_only.RetirementOnlyRegimeId.retirement,
    )


def _stochastic_next_wealth(
    savings: FloatND,
    interest_rate: float,
    period: Period,  # noqa: ARG001
) -> FloatND:
    probs = jnp.where(interest_rate > 0, 0.5, 0.5) * jnp.ones_like(savings)
    return jnp.stack([probs, probs])


def _stochastic_next_aime(
    aime: ContinuousState,
    interest_rate: float,
) -> FloatND:
    probs = jnp.where(interest_rate > 0, 0.5, 0.5) * jnp.ones_like(aime)
    return jnp.stack([probs, probs])


def _next_aime_depending_on_consumption(
    aime: ContinuousState, consumption: ContinuousAction
) -> ContinuousState:
    return aime + 0.1 * consumption


def _next_aime_decaying(aime: ContinuousState) -> ContinuousState:
    return 0.95 * aime


def _impute_wealth_memo(wealth: ContinuousState) -> ContinuousState:
    return wealth


def _next_wealth_memo(wealth_memo: ContinuousState) -> ContinuousState:
    return wealth_memo


def _utility_reading_carried_state(
    consumption: ContinuousAction, wealth_memo: ContinuousState
) -> FloatND:
    return jnp.log(consumption) + 0.01 * wealth_memo


def _without_function(regime: UserRegime, name: str) -> UserRegime:
    functions = {k: v for k, v in regime.functions.items() if k != name}
    return regime.replace(functions=functions)


# A regime satisfying the full DC-EGM contract; every case below breaks one rule.
VALID = dcegm_variants.dcegm_retirement


CASES = {
    "missing_inverse_marginal_utility": (
        lambda: _without_function(VALID, "inverse_marginal_utility"),
        "inverse_marginal_utility",
    ),
    "missing_resources_function": (
        lambda: _without_function(VALID, "resources"),
        "resources",
    ),
    "missing_post_decision_function": (
        lambda: _without_function(VALID, "savings"),
        "savings",
    ),
    "transition_bypasses_post_decision_state": (
        lambda: VALID.replace(state_transitions={"wealth": next_wealth}),
        "post",
    ),
    "utility_depends_on_wealth_directly": (
        lambda: VALID.replace(
            functions={
                **dict(VALID.functions),
                "utility": _utility_with_direct_wealth_dependence,
            }
        ),
        "utility",
    ),
    "constraint_touches_continuous_variables": (
        lambda: VALID.replace(
            constraints={"borrowing_constraint": borrowing_constraint}
        ),
        "constraint",
    ),
    "utility_reads_continuous_state_through_carried_state": (
        lambda: VALID.replace(
            states={
                **dict(VALID.states),
                "wealth_memo": Phased(
                    solve=_impute_wealth_memo,
                    simulate=LinSpacedGrid(start=1.0, stop=400.0, n_points=4),
                ),
            },
            state_transitions={
                **dict(VALID.state_transitions),
                "wealth_memo": _next_wealth_memo,
            },
            functions={
                **dict(VALID.functions),
                "utility": _utility_reading_carried_state,
            },
        ),
        "utility",
    ),
    "custom_bellman_aggregator": (
        lambda: VALID.replace(functions={**dict(VALID.functions), "H": _custom_H}),
        "H",
    ),
    "second_continuous_action": (
        lambda: VALID.replace(
            actions={
                **dict(VALID.actions),
                "leisure": LinSpacedGrid(start=0.1, stop=1.0, n_points=5),
            }
        ),
        "continuous action",
    ),
    "stochastic_passive_state_transition": (
        lambda: VALID.replace(
            states={
                **dict(VALID.states),
                "aime": LinSpacedGrid(start=0.0, stop=5.0, n_points=4),
            },
            state_transitions={
                **dict(VALID.state_transitions),
                "aime": MarkovTransition(_stochastic_next_aime),
            },
        ),
        "'aime'.*is stochastic",
    ),
    "passive_state_transition_depends_on_consumption": (
        lambda: VALID.replace(
            states={
                **dict(VALID.states),
                "aime": LinSpacedGrid(start=0.0, stop=5.0, n_points=4),
            },
            state_transitions={
                **dict(VALID.state_transitions),
                "aime": _next_aime_depending_on_consumption,
            },
        ),
        "not passive",
    ),
    "regime_transition_cliff_in_wealth": (
        lambda: VALID.replace(transition=_regime_transition_with_wealth_cliff),
        "regime transition function.*discontinuous",
    ),
    "stochastic_euler_state_transition": (
        lambda: VALID.replace(
            state_transitions={"wealth": MarkovTransition(_stochastic_next_wealth)}
        ),
        "stochastic",
    ),
    "batched_euler_state_grid": (
        lambda: VALID.replace(
            states={
                "wealth": LinSpacedGrid(start=1, stop=400, n_points=100, batch_size=50)
            }
        ),
        "batch",
    ),
    "batched_discrete_state_grid": (
        lambda: VALID.replace(
            states={
                **dict(VALID.states),
                "skill": DiscreteGrid(base.LaborSupply, batch_size=1),
            },
            state_transitions={
                **dict(VALID.state_transitions),
                "skill": fixed_transition("skill"),
            },
        ),
        "batch",
    ),
    "runtime_savings_grid": (
        lambda: VALID.replace(
            solver=dataclasses.replace(
                dcegm_variants.DCEGM_SOLVER,
                savings_grid=IrregSpacedGrid(n_points=8),
            )
        ),
        "runtime",
    ),
    "runtime_euler_state_grid": (
        lambda: VALID.replace(states={"wealth": IrregSpacedGrid(n_points=100)}),
        "runtime",
    ),
    "runtime_continuous_action_grid": (
        lambda: VALID.replace(actions={"consumption": IrregSpacedGrid(n_points=50)}),
        "runtime",
    ),
}


@pytest.mark.parametrize(("build", "match"), CASES.values(), ids=CASES.keys())
def test_dcegm_contract_violation_raises(build, match):
    """Each contract violation fails fast at Model construction."""
    with pytest.raises(ModelInitializationError, match=match):
        _build_model(build())


def test_passive_continuous_state_constructs():
    """A passive continuous state (deterministic, decision-independent) is valid."""
    regime = VALID.replace(
        states={
            **dict(VALID.states),
            "aime": LinSpacedGrid(start=0.0, stop=5.0, n_points=4),
        },
        state_transitions={
            **dict(VALID.state_transitions),
            "aime": _next_aime_decaying,
        },
    )
    model = _build_model(regime)
    assert model.n_periods == N_PERIODS


def _impute_pension(age: int) -> ContinuousState:
    return jnp.asarray(0.1 * age)


def _next_pension(pension: ContinuousState) -> ContinuousState:
    return pension


def test_carried_state_with_decision_free_imputation_constructs():
    """A carried state imputed independently of the decision variables is valid.

    Carried states are derived functions in the solve phase — no grid axis —
    so they are invisible to the DC-EGM state classification, and a
    decision-free imputation keeps every consumer evaluable at the savings
    stage.
    """
    regime = VALID.replace(
        states={
            **dict(VALID.states),
            "pension": Phased(
                solve=_impute_pension,
                simulate=LinSpacedGrid(start=0.0, stop=10.0, n_points=4),
            ),
        },
        state_transitions={
            **dict(VALID.state_transitions),
            "pension": _next_pension,
        },
    )
    model = _build_model(regime)
    assert model.n_periods == N_PERIODS


def _cliff_supplement(wealth: ContinuousState) -> FloatND:
    return jnp.where(wealth <= 100.0, 5.0, 0.0)


def _kinked_supplement(wealth: ContinuousState) -> FloatND:
    return 5.0 * jnp.clip((150.0 - wealth) / 50.0, 0.0, 1.0)


def _next_wealth_with_cliff(
    savings: FloatND, cliff_supplement: FloatND
) -> ContinuousState:
    return savings + cliff_supplement


def _next_wealth_with_kink(
    savings: FloatND, kinked_supplement: FloatND
) -> ContinuousState:
    return savings + kinked_supplement


def test_euler_law_with_cliff_in_euler_state_raises():
    """A law whose residual jumps in the Euler state fails at model build.

    A jump makes the child's value function discontinuous, so the true
    policy bunches next-period wealth at the discontinuity — a corner where
    the Euler equation does not hold and which EGM's candidate set cannot
    represent. Kinked (continuous) residuals are solvable per asset node.
    """
    regime = VALID.replace(
        state_transitions={"wealth": _next_wealth_with_cliff},
        functions={**dict(VALID.functions), "cliff_supplement": _cliff_supplement},
    )
    with pytest.raises(ModelInitializationError, match=r"discontinuous.*bunches"):
        _build_model(regime)


def test_euler_law_with_kinked_phase_out_constructs():
    """A law reading the Euler state through a continuous kink is valid."""
    regime = VALID.replace(
        state_transitions={"wealth": _next_wealth_with_kink},
        functions={**dict(VALID.functions), "kinked_supplement": _kinked_supplement},
    )
    model = _build_model(regime)
    assert model.n_periods == N_PERIODS


def _retirement_stay_prob(age: int, final_age_alive: float) -> FloatND:
    return jnp.where(age >= final_age_alive, 0.0, 1.0)


def _retirement_death_prob(age: int, final_age_alive: float) -> FloatND:
    return jnp.where(age >= final_age_alive, 1.0, 0.0)


def _three_regime_model_with_brute_worker(retirement_transition) -> Model:
    """Model with a brute-force worker regime next to a DC-EGM retirement regime."""
    ages = AgeGrid(start=40, stop=40 + (N_PERIODS - 1) * 10, step="10Y")
    last_age = ages.exact_values[-1]
    return Model(
        regimes={
            "working_life": base.working_life.replace(
                active=lambda age, la=last_age: age < la
            ),
            "retirement": dcegm_variants.dcegm_retirement_full.replace(
                transition=retirement_transition,
                active=lambda age, la=last_age: age < la,
            ),
            "dead": dead,
        },
        ages=ages,
        regime_id_class=base.RegimeId,
    )


def test_granular_transition_excluding_brute_regime_passes():
    """Declared reachability narrows the target-compatibility check.

    A granular regime transition declares its key set as the reachable
    targets; regimes outside it are structurally unreachable. A DC-EGM
    regime whose declared targets are itself and a terminal regime may
    therefore coexist with a brute-force non-terminal regime it never
    transitions into (the brute regime targeting the DC-EGM regime is
    allowed in that direction).
    """
    model = _three_regime_model_with_brute_worker(
        {
            "retirement": MarkovTransition(_retirement_stay_prob),
            "dead": MarkovTransition(_retirement_death_prob),
        }
    )
    assert model.n_periods == N_PERIODS


def test_coarse_transition_reaching_brute_regime_raises():
    """A coarse regime transition declares every regime reachable.

    The same model fails the target-compatibility check once the DC-EGM
    regime's transition is a bare callable: the brute-force non-terminal
    regime becomes a declared target.
    """
    with pytest.raises(ModelInitializationError, match="BruteForce"):
        _three_regime_model_with_brute_worker(base.next_regime_from_retirement)


def test_non_dcegm_non_terminal_target_raises():
    """A DC-EGM regime may not target a brute-force non-terminal regime."""
    brute_target = dcegm_variants.dcegm_retirement.replace(
        solver=BruteForce(),
        state_transitions={"wealth": next_wealth},
        constraints={"borrowing_constraint": borrowing_constraint},
        functions={"utility": utility_retirement},
    )
    dcegm_source = dcegm_variants.dcegm_working_life.replace(
        active=lambda age: age < 60,
    )
    ages = AgeGrid(start=40, stop=60, step="10Y")
    with pytest.raises(
        ModelInitializationError,
        match="non-terminal target of a DCEGM regime must itself use the DCEGM",
    ):
        Model(
            regimes={
                "working_life": dcegm_source,
                "retirement": brute_target.replace(active=lambda age: age < 60),
                "dead": dead,
            },
            ages=ages,
            regime_id_class=base.RegimeId,
        )


def _ordinary_inverse_marginal_utility(marginal_continuation: FloatND) -> FloatND:
    return 1.0 / marginal_continuation


def test_brute_force_inverse_marginal_utility_keeps_its_params():
    """`marginal_continuation` stays a user param outside DC-EGM regimes.

    Only the DC-EGM kernel supplies `marginal_continuation` at solve time. In
    a brute-force regime, a function named `inverse_marginal_utility` is an
    ordinary regime function, so its argument must surface in the params
    template like any other.
    """
    regime = retirement_only.retirement.replace(
        functions={
            **dict(retirement_only.retirement.functions),
            "inverse_marginal_utility": _ordinary_inverse_marginal_utility,
        },
        active=lambda age: age < 60,
    )
    model = _build_model(regime)

    template = model.get_params_template()

    assert "marginal_continuation" in template["retirement"]["inverse_marginal_utility"]


def test_brute_force_solver_explicit_equals_default():
    """`solver=BruteForce()` is the default: identical solution either way."""
    params = retirement_only.get_params(N_PERIODS)

    default_model = retirement_only.get_model(N_PERIODS)
    explicit = retirement_only.retirement.replace(
        solver=BruteForce(),
        active=lambda age: age < 60,
    )
    explicit_model = _build_model(explicit)

    got_default = default_model.solve(params=params, log_level="debug")
    got_explicit = explicit_model.solve(params=params, log_level="debug")

    for period in got_default:
        for regime in got_default[period]:
            assert bool(
                jnp.array_equal(
                    got_default[period][regime], got_explicit[period][regime]
                )
            )
