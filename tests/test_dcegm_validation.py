"""Spec for DC-EGM build-time validation (one case per contract rule).

A regime with `solver=DCEGM(...)` must satisfy the EGM contract; every violation
raises `ModelInitializationError` at `Model` construction with a message naming
the offending piece. The cases here mutate a valid DC-EGM regime one rule at a
time.

Skips until `lcm.solvers` exists; red until the validation lands.
"""

import jax.numpy as jnp
import pytest

pytest.importorskip("lcm.solvers", reason="DC-EGM solver not yet implemented")

from lcm import AgeGrid, LinSpacedGrid, MarkovTransition, Model
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


def _regime_transition_depending_on_wealth(
    age: int, final_age_alive: float, wealth: ContinuousState
) -> ScalarInt:
    return jnp.where(
        (age >= final_age_alive) | (wealth < 0.0),
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
    "passive_continuous_state_not_yet_supported": (
        lambda: VALID.replace(
            states={
                **dict(VALID.states),
                "aime": LinSpacedGrid(start=0.0, stop=5.0, n_points=4),
            },
            state_transitions={
                **dict(VALID.state_transitions),
                "aime": fixed_transition("aime"),
            },
        ),
        "continuous state",
    ),
    "regime_transition_prob_depends_on_wealth": (
        lambda: VALID.replace(transition=_regime_transition_depending_on_wealth),
        "wealth",
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
}


@pytest.mark.parametrize(("build", "match"), CASES.values(), ids=CASES.keys())
def test_dcegm_contract_violation_raises(build, match):
    """Each contract violation fails fast at Model construction."""
    with pytest.raises(ModelInitializationError, match=match):
        _build_model(build())


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
    with pytest.raises(ModelInitializationError, match="DCEGM"):
        Model(
            regimes={
                "working_life": dcegm_source,
                "retirement": brute_target.replace(active=lambda age: age < 60),
                "dead": dead,
            },
            ages=ages,
            regime_id_class=base.RegimeId,
        )


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
