"""DC-EGM with a passive continuous state (AIME-like skill accumulation).

A passive continuous state is a deterministic, non-process continuous state
whose transition depends on neither the continuous action, the post-decision
function, nor the Euler state — here, a skill level driven by a discrete
labor choice. The DC-EGM kernel carries one grid axis per passive state and
reads the child carry with mixed interpolation: linear weights on the two
neighboring nodes of the child's passive grid, each neighbor's row
interpolated 1-D in resources, blended per discrete-action row before the
choice aggregation.

The oracle is a dense-grid brute-force solve of a mathematically equivalent
spec (same wealth and skill grids, dense consumption grid, explicit budget
constraint).
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
from lcm.transition import fixed_transition
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

# Skill increment earned by one period of work. With the 0.25 node spacing of
# SKILL_GRID, working transitions land off-grid (0.4 is not a multiple of
# 0.25) while resting (identity transition) and the clamp at SKILL_MAX land
# exactly on nodes — both interpolation paths are exercised across periods.
SKILL_GAIN = 0.4

# Upper bound of the skill grid; the skill transition clamps here.
SKILL_MAX = 1.5

WEALTH_GRID = LinSpacedGrid(start=1.0, stop=100.0, n_points=80)
SKILL_GRID = LinSpacedGrid(start=0.0, stop=SKILL_MAX, n_points=7)
CONSUMPTION_GRID = LinSpacedGrid(start=0.25, stop=100.0, n_points=400)

# Exogenous end-of-period savings grid, cubically clustered toward the
# borrowing limit where the value function curves hardest.
SAVINGS_GRID = IrregSpacedGrid(points=tuple(100.0 * (i / 149) ** 3 for i in range(150)))


@categorical(ordered=False)
class PassiveRegimeId:
    working_life: ScalarInt
    dead: ScalarInt


@categorical(ordered=True)
class LaborChoice:
    work: ScalarInt
    rest: ScalarInt


def is_working(labor_supply: DiscreteAction) -> BoolND:
    return labor_supply == LaborChoice.work


def labor_income(is_working: BoolND, skill: ContinuousState, wage: float) -> FloatND:
    return jnp.where(is_working, wage * (1.0 + skill), 0.0)


def utility(
    consumption: ContinuousAction, is_working: BoolND, disutility_of_work: float
) -> FloatND:
    return jnp.log(consumption) - jnp.where(is_working, disutility_of_work, 0.0)


def next_skill(skill: ContinuousState, is_working: BoolND) -> ContinuousState:
    return jnp.minimum(skill + jnp.where(is_working, SKILL_GAIN, 0.0), SKILL_MAX)


def resources(wealth: ContinuousState) -> FloatND:
    return wealth


def savings(resources: FloatND, consumption: ContinuousAction) -> FloatND:
    return resources - consumption


def inverse_marginal_utility(marginal_continuation: FloatND) -> FloatND:
    return 1.0 / marginal_continuation


def next_wealth_from_savings(
    savings: FloatND, labor_income: FloatND, interest_rate: float
) -> ContinuousState:
    return (1 + interest_rate) * savings + labor_income


def next_wealth_brute(
    wealth: ContinuousState,
    consumption: ContinuousAction,
    labor_income: FloatND,
    interest_rate: float,
) -> ContinuousState:
    return (1 + interest_rate) * (wealth - consumption) + labor_income


def borrowing_constraint(
    consumption: ContinuousAction, wealth: ContinuousState
) -> BoolND:
    return consumption <= wealth


def next_regime(age: int, final_age_alive: float) -> ScalarInt:
    return jnp.where(
        age >= final_age_alive,
        PassiveRegimeId.dead,
        PassiveRegimeId.working_life,
    )


DCEGM_SOLVER = DCEGM(
    continuous_state="wealth",
    continuous_action="consumption",
    resources="resources",
    post_decision_function="savings",
    savings_grid=SAVINGS_GRID,
    n_constrained_points=64,
)


def labor_income_fixed_skill(
    is_working: BoolND, wage: float, skill_level: float
) -> FloatND:
    """Labor income of the skill-free comparison model; skill is a parameter."""
    return jnp.where(is_working, wage * (1.0 + skill_level), 0.0)


@functools.cache
def _get_model(variant: str) -> Model:
    """Build one model variant.

    - `"dcegm"`: DC-EGM with the accumulating (off-grid) skill transition.
    - `"brute"`: dense-grid brute force, mathematically equivalent spec.
    - `"dcegm_fixed_skill"`: DC-EGM with an identity skill transition, so
      every child passive value lands exactly on a node.
    - `"dcegm_no_skill"`: DC-EGM without the skill state; skill enters labor
      income as the `skill_level` parameter.
    """
    ages = AgeGrid(start=40, stop=40 + (N_PERIODS - 1) * 10, step="10Y")
    last_age = ages.exact_values[-1]
    common = {
        "transition": next_regime,
        "active": lambda age, la=last_age: age < la,
    }
    if variant == "brute":
        working = UserRegime(
            **common,
            actions={
                "labor_supply": DiscreteGrid(LaborChoice),
                "consumption": CONSUMPTION_GRID,
            },
            states={"wealth": WEALTH_GRID, "skill": SKILL_GRID},
            state_transitions={"wealth": next_wealth_brute, "skill": next_skill},
            constraints={"borrowing_constraint": borrowing_constraint},
            functions={
                "utility": utility,
                "labor_income": labor_income,
                "is_working": is_working,
            },
        )
    elif variant == "dcegm_no_skill":
        working = UserRegime(
            **common,
            actions={
                "labor_supply": DiscreteGrid(LaborChoice),
                "consumption": CONSUMPTION_GRID,
            },
            states={"wealth": WEALTH_GRID},
            state_transitions={"wealth": next_wealth_from_savings},
            functions={
                "utility": utility,
                "labor_income": labor_income_fixed_skill,
                "is_working": is_working,
                "resources": resources,
                "savings": savings,
                "inverse_marginal_utility": inverse_marginal_utility,
            },
            solver=DCEGM_SOLVER,
        )
    else:
        skill_transition = (
            fixed_transition("skill") if variant == "dcegm_fixed_skill" else next_skill
        )
        working = UserRegime(
            **common,
            actions={
                "labor_supply": DiscreteGrid(LaborChoice),
                "consumption": CONSUMPTION_GRID,
            },
            states={"wealth": WEALTH_GRID, "skill": SKILL_GRID},
            state_transitions={
                "wealth": next_wealth_from_savings,
                "skill": skill_transition,
            },
            functions={
                "utility": utility,
                "labor_income": labor_income,
                "is_working": is_working,
                "resources": resources,
                "savings": savings,
                "inverse_marginal_utility": inverse_marginal_utility,
            },
            solver=DCEGM_SOLVER,
        )
    return Model(
        regimes={"working_life": working, "dead": dead},
        ages=ages,
        regime_id_class=PassiveRegimeId,
    )


def _get_params(*, skill_level: float | None = None) -> dict:
    final_age_alive = 40 + (N_PERIODS - 2) * 10
    labor_income_params = {"wage": 20.0}
    if skill_level is not None:
        labor_income_params["skill_level"] = skill_level
    return {
        "discount_factor": 0.95,
        "interest_rate": 0.0,
        "final_age_alive": final_age_alive,
        "working_life": {
            "labor_income": labor_income_params,
            "utility": {"disutility_of_work": 0.5},
        },
    }


def test_passive_state_matches_dense_brute_force():
    """DC-EGM with an off-grid passive transition matches dense brute force.

    Both solvers share the wealth and skill grids; working moves skill off the
    node spacing, so every period exercises the mixed passive read. Tolerance:
    both methods interpolate linearly in the passive dimension but in
    different objects (brute interpolates the aggregated V', DC-EGM blends
    choice-specific carry rows before aggregating), so they agree only up to
    the linear-interpolation error in skill plus the brute solver's
    consumption-grid resolution. The lowest wealth nodes are excluded: there
    the coarse consumption grid makes brute force itself unreliable (the same
    exclusion the discrete DC-EGM tests use).
    """
    params = _get_params()
    dcegm_solution = _get_model("dcegm").solve(params=params, log_level="debug")
    brute_solution = _get_model("brute").solve(params=params, log_level="debug")

    n_brute_unstable_nodes = 8
    for period in sorted(brute_solution)[:-1]:
        brute_V = np.asarray(brute_solution[period]["working_life"])
        dcegm_V = np.asarray(dcegm_solution[period]["working_life"])
        assert brute_V.shape == dcegm_V.shape == (80, 7)
        np.testing.assert_allclose(
            dcegm_V[n_brute_unstable_nodes:, :],
            brute_V[n_brute_unstable_nodes:, :],
            atol=1e-2,
            rtol=1e-3,
            err_msg=f"period={period}",
        )


def test_on_node_passive_value_reproduces_skill_free_solution():
    """A passive value landing exactly on a node reads that node's rows exactly.

    With an identity skill transition every child passive value is a grid
    node, so each skill slice solves an independent model in which skill is a
    constant. The slice must reproduce the skill-free DC-EGM model with that
    constant baked into labor income — node for node, with no interpolation
    error from the passive read.
    """
    sliced_solution = _get_model("dcegm_fixed_skill").solve(
        params=_get_params(), log_level="debug"
    )
    skill_nodes = np.asarray(SKILL_GRID.to_jax())

    for skill_index in [0, 3, 6]:
        single_solution = _get_model("dcegm_no_skill").solve(
            params=_get_params(skill_level=float(skill_nodes[skill_index])),
            log_level="debug",
        )
        for period in sorted(sliced_solution)[:-1]:
            np.testing.assert_allclose(
                np.asarray(sliced_solution[period]["working_life"])[:, skill_index],
                np.asarray(single_solution[period]["working_life"]),
                rtol=1e-9,
                atol=1e-9,
                err_msg=f"period={period}, skill_index={skill_index}",
            )
