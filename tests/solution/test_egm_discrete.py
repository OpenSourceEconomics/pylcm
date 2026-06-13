"""DC-EGM with discrete states and discrete-only constraints.

Variants of the Iskhakov et al. (2017) retirement model exercise the two
discrete dimensions of the DC-EGM kernel beyond the discrete work/retire
action:

- a discrete *state* (a fixed skill type scaling the wage) that must remain
  an axis of the published value-function array and must select the matching
  carry rows when the regime targets itself, and
- a discrete-only *constraint* that masks a discrete-action combo everywhere,
  whose carry rows must be `-inf` with exactly-zero marginal utility so the
  parent's choice aggregation stays finite.
"""

import functools

import jax.numpy as jnp
import numpy as np
import pytest

from lcm import (
    AgeGrid,
    DiscreteGrid,
    MarkovTransition,
    Model,
    categorical,
    fixed_transition,
)
from lcm.exceptions import InvalidRegimeTransitionProbabilitiesError
from lcm.typing import BoolND, DiscreteAction, DiscreteState, FloatND, ScalarInt
from tests.test_models.deterministic import base
from tests.test_models.deterministic.dcegm_variants import (
    dcegm_retirement,
    dcegm_retirement_full,
    dcegm_working_life,
    get_full_model,
    get_full_params,
    get_retirement_only_params,
)
from tests.test_models.deterministic.retirement_only import RetirementOnlyRegimeId

N_PERIODS = 4


def _retirement_stay_prob(age: float, final_age_alive: float) -> FloatND:
    return jnp.where(age >= final_age_alive, 0.0, 1.0)


def _retirement_death_prob(age: float, final_age_alive: float) -> FloatND:
    return jnp.where(age >= final_age_alive, 1.0, 0.0)


# Retirement can only stay retired or die. Declaring this granularly (with
# indicator probabilities) narrows reachability so the bare wealth law never
# has to cover the skill-carrying working regime.
RETIREMENT_TRANSITION = {
    "retirement": MarkovTransition(_retirement_stay_prob),
    "dead": MarkovTransition(_retirement_death_prob),
}


@categorical(ordered=False)
class Skill:
    low: ScalarInt
    high: ScalarInt


def wage_factor(skill: DiscreteState) -> FloatND:
    return jnp.where(skill == Skill.high, 2.0, 1.0)


def labor_income_by_skill(
    is_working: BoolND, wage: float, wage_factor: FloatND
) -> FloatND:
    return jnp.where(is_working, wage * wage_factor, 0.0)


def must_retire(labor_supply: DiscreteAction) -> BoolND:
    return labor_supply == base.LaborSupply.retire


def _get_skill_model_params(*, wage: float = 20.0) -> dict:
    """Params for the models whose retirement regime names its target."""
    params = get_full_params(N_PERIODS, discount_factor=0.98, wage=wage)
    params["retirement"] = {"next_wealth": {"labor_income": 0.0}}
    return params


@functools.cache
def _get_skill_model() -> Model:
    """Full DC-EGM retirement model with a fixed skill state scaling the wage."""
    ages = AgeGrid(start=40, stop=40 + (N_PERIODS - 1) * 10, step="10Y")
    last_age = ages.exact_values[-1]
    working_life = dcegm_working_life.replace(
        states={
            "wealth": dcegm_working_life.states["wealth"],
            "skill": DiscreteGrid(Skill),
        },
        state_transitions={
            "wealth": dcegm_working_life.state_transitions["wealth"],
            "skill": fixed_transition("skill"),
        },
        functions={
            **dict(dcegm_working_life.functions),
            "wage_factor": wage_factor,
            "labor_income": labor_income_by_skill,
        },
        active=lambda age, la=last_age: age < la,
    )
    retirement = dcegm_retirement_full.replace(
        transition=RETIREMENT_TRANSITION,
        state_transitions={
            "wealth": dcegm_retirement_full.state_transitions["wealth"],
        },
        active=lambda age, la=last_age: age < la,
    )
    return Model(
        regimes={
            "working_life": working_life,
            "retirement": retirement,
            "dead": base.dead,
        },
        ages=ages,
        regime_id_class=base.RegimeId,
    )


@functools.cache
def _get_must_retire_model() -> Model:
    """Full DC-EGM retirement model where a constraint forbids working."""
    ages = AgeGrid(start=40, stop=40 + (N_PERIODS - 1) * 10, step="10Y")
    last_age = ages.exact_values[-1]
    return Model(
        regimes={
            "working_life": dcegm_working_life.replace(
                constraints={"must_retire": must_retire},
                active=lambda age, la=last_age: age < la,
            ),
            "retirement": dcegm_retirement_full.replace(
                active=lambda age, la=last_age: age < la
            ),
            "dead": base.dead,
        },
        ages=ages,
        regime_id_class=base.RegimeId,
    )


def test_discrete_state_slices_match_single_type_models():
    """Each skill slice of V equals the skill-free model solved at that wage.

    The skill state is fixed and scales the wage by 1 (low) or 2 (high), so
    the low slice must reproduce the base-wage model and the high slice the
    double-wage model — node for node. A kernel that selected the wrong
    carry rows for the child's skill value would mix the types' carries and
    break this equality.
    """
    skill_solution = _get_skill_model().solve(
        params=_get_skill_model_params(), log_level="debug"
    )
    base_model = get_full_model("dcegm", N_PERIODS)
    params = get_full_params(N_PERIODS, discount_factor=0.98, wage=20.0)
    single_type_solutions = {
        "low": base_model.solve(params=params, log_level="debug"),
        "high": base_model.solve(
            params=get_full_params(N_PERIODS, discount_factor=0.98, wage=40.0),
            log_level="debug",
        ),
    }

    for period in sorted(skill_solution)[:-1]:
        V_skill = np.asarray(skill_solution[period]["working_life"])
        assert V_skill.shape == (2, 100)
        for skill_index, label in [(0, "low"), (1, "high")]:
            np.testing.assert_allclose(
                V_skill[skill_index],
                np.asarray(single_type_solutions[label][period]["working_life"]),
                rtol=1e-9,
                atol=1e-9,
                err_msg=f"period={period}, skill={label}",
            )


def test_infeasible_discrete_action_recovers_retirement_value():
    """With work forbidden everywhere, the worker's V equals the retiree's V.

    The discrete-only constraint makes the work combo infeasible at every
    state: its value rows are `-inf` and its marginal-utility rows exactly
    zero, so the forced-retirement worker solves the retiree's Bellman
    equation. Any NaN leak from the infeasible rows would poison the parent's
    aggregation (the solve runs at `log_level="debug"`, which raises on NaN).
    """
    params = get_full_params(N_PERIODS, discount_factor=0.98, wage=20.0)

    solution = _get_must_retire_model().solve(params=params, log_level="debug")

    for period in sorted(solution)[:-1]:
        np.testing.assert_allclose(
            np.asarray(solution[period]["working_life"]),
            np.asarray(solution[period]["retirement"]),
            rtol=1e-9,
            atol=1e-9,
            err_msg=f"period={period}",
        )


@pytest.mark.parametrize("regime_name", ["working_life", "retirement"])
def test_discrete_state_layout_matches_brute_force(regime_name):
    """The skill model's V arrays have brute-force layout and values.

    Discrete-state axes lead and the continuous state is last, exactly as the
    brute-force solver lays out V; values agree on the wealth nodes where the
    brute solver is reliable.
    """
    params = _get_skill_model_params()
    ages = AgeGrid(start=40, stop=40 + (N_PERIODS - 1) * 10, step="10Y")
    last_age = ages.exact_values[-1]
    brute_model = Model(
        regimes={
            "working_life": base.working_life.replace(
                states={"wealth": base.WEALTH_GRID, "skill": DiscreteGrid(Skill)},
                state_transitions={
                    "wealth": base.next_wealth,
                    "skill": fixed_transition("skill"),
                },
                functions={
                    **dict(base.working_life.functions),
                    "wage_factor": wage_factor,
                    "labor_income": labor_income_by_skill,
                },
                active=lambda age, la=last_age: age < la,
            ),
            "retirement": base.retirement.replace(
                transition=RETIREMENT_TRANSITION,
                state_transitions={"wealth": base.next_wealth},
                active=lambda age, la=last_age: age < la,
            ),
            "dead": base.dead,
        },
        ages=ages,
        regime_id_class=base.RegimeId,
    )
    brute_solution = brute_model.solve(params=params, log_level="debug")
    dcegm_solution = _get_skill_model().solve(params=params, log_level="debug")

    n_brute_unstable_nodes = 10
    for period in sorted(brute_solution)[:-1]:
        brute_V = np.asarray(brute_solution[period][regime_name])
        dcegm_V = np.asarray(dcegm_solution[period][regime_name])
        assert brute_V.shape == dcegm_V.shape
        np.testing.assert_allclose(
            dcegm_V[..., n_brute_unstable_nodes:],
            brute_V[..., n_brute_unstable_nodes:],
            # The brute leg's grid-restricted max is the biased side here.
            atol=3e-2,
            rtol=1e-3,
            err_msg=f"period={period}",
        )


def _stay_prob_from_param(survival_rate: float) -> FloatND:
    return jnp.asarray(survival_rate)


def _death_prob_from_param(survival_rate: float) -> FloatND:
    return jnp.asarray(1.0 - survival_rate)


def test_nan_regime_transition_prob_surfaces_as_error():
    """A NaN regime-transition probability raises instead of vanishing.

    Masking unreachable targets must not swallow a NaN probability (`NaN > 0`
    is false): the runtime probability check rejects non-finite probabilities
    in the DC-EGM solve exactly as under brute force.
    """
    n_periods = 3
    ages = AgeGrid(start=40, stop=40 + (n_periods - 1) * 10, step="10Y")
    last_age = ages.exact_values[-1]
    model = Model(
        regimes={
            "retirement": dcegm_retirement.replace(
                transition={
                    "retirement": MarkovTransition(_stay_prob_from_param),
                    "dead": MarkovTransition(_death_prob_from_param),
                },
                active=lambda age, la=last_age: age < la,
            ),
            "dead": base.dead,
        },
        ages=ages,
        regime_id_class=RetirementOnlyRegimeId,
    )
    params = get_retirement_only_params(n_periods)
    # The granular transition replaces the age-based one, so its param goes
    # and the per-cell survival rate (set to NaN) arrives.
    del params["final_age_alive"]
    params["retirement"] = {
        **params.get("retirement", {}),
        "retirement": {"next_regime": {"survival_rate": float("nan")}},
        "dead": {"next_regime": {"survival_rate": float("nan")}},
    }

    with pytest.raises(InvalidRegimeTransitionProbabilitiesError):
        model.solve(params=params, log_level="debug")


@categorical(ordered=False)
class RegimeIdWithLost:
    working_life: ScalarInt
    retirement: ScalarInt
    dead: ScalarInt
    lost: ScalarInt


def _lost_utility() -> FloatND:
    return jnp.asarray(-1000.0)


def test_undeclared_stateless_regime_does_not_enter_the_continuation():
    """A stateless regime outside the declared targets contributes nothing.

    The DC-EGM retirement regime declares its reachable targets granularly
    (`{retirement, dead}`). A further stateless terminal regime in the model
    — reachable only from the brute-force worker — must be invisible to the
    retirement regime's continuation: its value function is unchanged by
    that regime's presence.
    """
    ages = AgeGrid(start=40, stop=40 + (N_PERIODS - 1) * 10, step="10Y")
    last_age = ages.exact_values[-1]
    lost = base.dead.replace(functions={"utility": _lost_utility})
    shared_regimes = {
        "working_life": base.working_life.replace(
            active=lambda age, la=last_age: age < la
        ),
        "retirement": dcegm_retirement_full.replace(
            transition=RETIREMENT_TRANSITION,
            active=lambda age, la=last_age: age < la,
        ),
        "dead": base.dead,
    }
    with_lost = Model(
        regimes={**shared_regimes, "lost": lost},
        ages=ages,
        regime_id_class=RegimeIdWithLost,
    )
    without_lost = Model(
        regimes=shared_regimes,
        ages=ages,
        regime_id_class=base.RegimeId,
    )
    params = _get_skill_model_params()

    solution_with = with_lost.solve(params=params, log_level="debug")
    solution_without = without_lost.solve(params=params, log_level="debug")

    for period in sorted(solution_without)[:-1]:
        np.testing.assert_allclose(
            np.asarray(solution_with[period]["retirement"]),
            np.asarray(solution_without[period]["retirement"]),
            atol=1e-12,
            err_msg=f"period={period}",
        )


def _nothing_is_feasible(labor_supply: DiscreteAction) -> BoolND:
    return jnp.zeros_like(labor_supply, dtype=bool)


def test_all_infeasible_regime_publishes_neg_inf_like_brute_force():
    """A regime whose every combo is infeasible publishes `-inf` V, never NaN.

    A discrete-only constraint that is false everywhere makes the regime's
    value `-inf` at every state; its parents (including its own earlier
    periods) must absorb the `-inf` continuation gracefully. Brute force
    publishes `-inf` in this case, and DC-EGM must match it.
    """
    base_model_regimes = {
        "working_life": dcegm_working_life.replace(
            constraints={"nothing_is_feasible": _nothing_is_feasible},
            active=lambda age: age < 70,
        ),
        "retirement": dcegm_retirement_full.replace(
            transition=RETIREMENT_TRANSITION,
            state_transitions={
                "wealth": dcegm_retirement_full.state_transitions["wealth"],
            },
            active=lambda age: age < 70,
        ),
        "dead": base.dead,
    }
    ages = AgeGrid(start=40, stop=40 + (N_PERIODS - 1) * 10, step="10Y")
    doomed_model = Model(
        regimes=base_model_regimes,
        ages=ages,
        regime_id_class=base.RegimeId,
    )
    params = get_full_params(N_PERIODS, discount_factor=0.98, wage=20.0)

    solution = doomed_model.solve(params=params, log_level="debug")

    for period in sorted(solution)[:-1]:
        working_V = np.asarray(solution[period]["working_life"])
        assert bool(np.isneginf(working_V).all()), f"period={period}"
        assert bool(np.isfinite(solution[period]["retirement"]).all())
