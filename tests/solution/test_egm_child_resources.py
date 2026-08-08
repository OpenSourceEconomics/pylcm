"""DC-EGM with a resources function reading more than the Euler state.

A carry target's resources function may read the child's discrete actions and
passive continuous states in addition to its Euler state — e.g. a work bonus
paid on top of wealth, or a pension scaling with an AIME-like skill level. The
child's resources space is then per-combo: the parent's carry read evaluates
$R'$ and the composed gradient $\\partial R'/\\partial A$ per child
discrete-action row and per passive neighbor node, so each carry row is
queried in the resources space it was built in.

The oracle is a dense-grid brute-force solve of a mathematically equivalent
spec (same state grids, dense consumption grid, explicit budget constraint on
the same resources).
"""

import functools

import jax.numpy as jnp
import numpy as np

from lcm import (
    AgeGrid,
    DiscreteGrid,
    IrregSpacedGrid,
    LinSpacedGrid,
    Model,
    categorical,
)
from lcm.regime import Regime as UserRegime
from lcm.solvers import DCEGM
from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    DiscreteAction,
    FloatND,
    ScalarInt,
)
from lcm_examples.iskhakov_et_al_2017 import dead

# Number of model periods; the last one is spent in the terminal `dead` regime.
N_PERIODS = 4

# Transfer added to resources when working — the discrete-action dependence of
# the child's resources space. A module constant (not a model param) so the
# child's resources DAG closes over it.
WORK_BONUS = 15.0

# Pension paid per unit of skill — the passive-state dependence of the child's
# resources space.
PENSION_RATE = 0.5

# Skill increment earned by one period of work; 0.4 is not a multiple of the
# skill grid's 0.25 node spacing, so working lands off-grid.
SKILL_GAIN = 0.4

# Upper bound of the skill grid; the skill transition clamps here.
SKILL_MAX = 1.5

WEALTH_GRID = LinSpacedGrid(start=1.0, stop=100.0, n_points=80)
SKILL_GRID = LinSpacedGrid(start=0.0, stop=SKILL_MAX, n_points=7)

# The consumption grid covers the maximum resources (wealth + bonus + pension),
# so the brute-force oracle is not artificially capped at high wealth.
CONSUMPTION_GRID = LinSpacedGrid(start=0.25, stop=120.0, n_points=400)

# Exogenous end-of-period savings grid, cubically clustered toward the
# borrowing limit where the value function curves hardest.
SAVINGS_GRID = IrregSpacedGrid(points=tuple(100.0 * (i / 149) ** 3 for i in range(150)))

# Lowest wealth nodes excluded from the brute-force comparison: there the
# coarse consumption grid makes brute force itself unreliable (the same
# exclusion the discrete and passive DC-EGM tests use).
N_BRUTE_UNSTABLE_NODES = 8


@categorical(ordered=False)
class BonusRegimeId:
    working_life: ScalarInt
    dead: ScalarInt


@categorical(ordered=True)
class LaborChoice:
    work: ScalarInt
    rest: ScalarInt


def is_working(labor_supply: DiscreteAction) -> BoolND:
    return labor_supply == LaborChoice.work


def labor_income(is_working: BoolND, wage: float) -> FloatND:
    return jnp.where(is_working, wage, 0.0)


def utility(
    consumption: ContinuousAction, is_working: BoolND, disutility_of_work: float
) -> FloatND:
    return jnp.log(consumption) - jnp.where(is_working, disutility_of_work, 0.0)


def resources_with_bonus(wealth: ContinuousState, is_working: BoolND) -> FloatND:
    return wealth + jnp.where(is_working, WORK_BONUS, 0.0)


def resources_with_bonus_and_pension(
    wealth: ContinuousState, is_working: BoolND, skill: ContinuousState
) -> FloatND:
    return wealth + jnp.where(is_working, WORK_BONUS, 0.0) + PENSION_RATE * skill


def savings(resources: FloatND, consumption: ContinuousAction) -> FloatND:
    return resources - consumption


def inverse_marginal_utility(marginal_continuation: FloatND) -> FloatND:
    return 1.0 / marginal_continuation


def next_wealth_from_savings(
    savings: FloatND, labor_income: FloatND, interest_rate: float
) -> ContinuousState:
    return (1 + interest_rate) * savings + labor_income


def next_skill(skill: ContinuousState, is_working: BoolND) -> ContinuousState:
    return jnp.minimum(skill + jnp.where(is_working, SKILL_GAIN, 0.0), SKILL_MAX)


def next_wealth_brute_bonus(
    wealth: ContinuousState,
    consumption: ContinuousAction,
    is_working: BoolND,
    labor_income: FloatND,
    interest_rate: float,
) -> ContinuousState:
    bonus = jnp.where(is_working, WORK_BONUS, 0.0)
    return (1 + interest_rate) * (wealth + bonus - consumption) + labor_income


def next_wealth_brute_bonus_and_pension(
    wealth: ContinuousState,
    consumption: ContinuousAction,
    is_working: BoolND,
    skill: ContinuousState,
    labor_income: FloatND,
    interest_rate: float,
) -> ContinuousState:
    bonus = jnp.where(is_working, WORK_BONUS, 0.0)
    pension = PENSION_RATE * skill
    return (1 + interest_rate) * (wealth + bonus + pension - consumption) + labor_income


def budget_constraint_bonus(
    consumption: ContinuousAction, wealth: ContinuousState, is_working: BoolND
) -> BoolND:
    return consumption <= wealth + jnp.where(is_working, WORK_BONUS, 0.0)


def budget_constraint_bonus_and_pension(
    consumption: ContinuousAction,
    wealth: ContinuousState,
    is_working: BoolND,
    skill: ContinuousState,
) -> BoolND:
    bonus = jnp.where(is_working, WORK_BONUS, 0.0)
    return consumption <= wealth + bonus + PENSION_RATE * skill


def next_regime(age: int, final_age_alive: float) -> ScalarInt:
    return jnp.where(
        age >= final_age_alive,
        BonusRegimeId.dead,
        BonusRegimeId.working_life,
    )


DCEGM_SOLVER = DCEGM(
    continuous_state="wealth",
    continuous_action="consumption",
    resources="resources",
    post_decision_function="savings",
    savings_grid=SAVINGS_GRID,
    n_constrained_points=64,
)


@functools.cache
def _get_model(variant: str) -> Model:
    """Build one model variant.

    - `"dcegm_bonus"`: DC-EGM; resources read the work choice.
    - `"brute_bonus"`: dense-grid brute force, mathematically equivalent spec.
    - `"dcegm_bonus_pension"`: DC-EGM; resources read the work choice and a
      passive skill state.
    - `"brute_bonus_pension"`: dense-grid brute force, equivalent spec.
    """
    ages = AgeGrid(start=40, stop=40 + (N_PERIODS - 1) * 10, step="10Y")
    last_age = float(ages.exact_values[-1])

    def active(age: int, la: float = last_age) -> bool:
        return age < la

    actions = {
        "labor_supply": DiscreteGrid(LaborChoice),
        "consumption": CONSUMPTION_GRID,
    }
    if variant == "dcegm_bonus":
        working = UserRegime(
            transition=next_regime,
            active=active,
            actions=actions,
            states={"wealth": WEALTH_GRID},
            state_transitions={"wealth": next_wealth_from_savings},
            functions={
                "utility": utility,
                "labor_income": labor_income,
                "is_working": is_working,
                "resources": resources_with_bonus,
                "savings": savings,
                "inverse_marginal_utility": inverse_marginal_utility,
            },
            solver=DCEGM_SOLVER,
        )
    elif variant == "brute_bonus":
        working = UserRegime(
            transition=next_regime,
            active=active,
            actions=actions,
            states={"wealth": WEALTH_GRID},
            state_transitions={"wealth": next_wealth_brute_bonus},
            constraints={"budget_constraint": budget_constraint_bonus},
            functions={
                "utility": utility,
                "labor_income": labor_income,
                "is_working": is_working,
            },
        )
    elif variant == "dcegm_bonus_pension":
        working = UserRegime(
            transition=next_regime,
            active=active,
            actions=actions,
            states={"wealth": WEALTH_GRID, "skill": SKILL_GRID},
            state_transitions={
                "wealth": next_wealth_from_savings,
                "skill": next_skill,
            },
            functions={
                "utility": utility,
                "labor_income": labor_income,
                "is_working": is_working,
                "resources": resources_with_bonus_and_pension,
                "savings": savings,
                "inverse_marginal_utility": inverse_marginal_utility,
            },
            solver=DCEGM_SOLVER,
        )
    else:
        working = UserRegime(
            transition=next_regime,
            active=active,
            actions=actions,
            states={"wealth": WEALTH_GRID, "skill": SKILL_GRID},
            state_transitions={
                "wealth": next_wealth_brute_bonus_and_pension,
                "skill": next_skill,
            },
            constraints={"budget_constraint": budget_constraint_bonus_and_pension},
            functions={
                "utility": utility,
                "labor_income": labor_income,
                "is_working": is_working,
            },
        )
    return Model(
        regimes={"working_life": working, "dead": dead},
        ages=ages,
        regime_id_class=BonusRegimeId,
    )


def _get_params() -> dict:
    final_age_alive = 40 + (N_PERIODS - 2) * 10
    return {
        "discount_factor": 0.95,
        "interest_rate": 0.0,
        "final_age_alive": final_age_alive,
        "working_life": {
            "labor_income": {"wage": 20.0},
            "utility": {"disutility_of_work": 0.5},
        },
    }


def test_action_dependent_resources_match_dense_brute_force():
    """A work bonus in the child's resources matches dense brute force.

    The work and rest carry rows of the self-targeting regime live in
    different resources spaces (offset by the bonus), so the parent's carry
    read must query each child action row at its own resources value and use
    each row's own composed gradient. Agreement is up to the brute solver's
    consumption-grid resolution, excluding the lowest wealth nodes where the
    coarse consumption grid makes brute force itself unreliable.
    """
    params = _get_params()
    dcegm_solution = _get_model("dcegm_bonus").solve(params=params, log_level="debug")
    brute_solution = _get_model("brute_bonus").solve(params=params, log_level="debug")

    for period in sorted(brute_solution)[:-1]:
        brute_V = np.asarray(brute_solution[period]["working_life"])
        dcegm_V = np.asarray(dcegm_solution[period]["working_life"])
        assert brute_V.shape == dcegm_V.shape == (80,)
        np.testing.assert_allclose(
            dcegm_V[N_BRUTE_UNSTABLE_NODES:],
            brute_V[N_BRUTE_UNSTABLE_NODES:],
            atol=1e-2,
            rtol=1e-3,
            err_msg=f"period={period}",
        )


def test_action_and_passive_dependent_resources_match_dense_brute_force():
    """A work bonus plus a skill pension in resources matches dense brute force.

    The child's resources read both a discrete action and a passive state, so
    every carry row (per action and per passive neighbor node) is queried at
    its own resources value with its own composed gradient; the passive blend
    happens across rows evaluated in their own node's resources space.
    """
    params = _get_params()
    dcegm_solution = _get_model("dcegm_bonus_pension").solve(
        params=params, log_level="debug"
    )
    brute_solution = _get_model("brute_bonus_pension").solve(
        params=params, log_level="debug"
    )

    for period in sorted(brute_solution)[:-1]:
        brute_V = np.asarray(brute_solution[period]["working_life"])
        dcegm_V = np.asarray(dcegm_solution[period]["working_life"])
        assert brute_V.shape == dcegm_V.shape == (80, 7)
        np.testing.assert_allclose(
            dcegm_V[N_BRUTE_UNSTABLE_NODES:, :],
            brute_V[N_BRUTE_UNSTABLE_NODES:, :],
            atol=1e-2,
            rtol=1e-3,
            err_msg=f"period={period}",
        )
