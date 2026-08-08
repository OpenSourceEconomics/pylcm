"""DC-EGM savings-stage functions reading the current Euler state.

Regime-transition probabilities, stochastic transition weights, and passive
(non-Euler) state laws may read the current Euler state. The kernel then
solves per exogenous asset node — current assets are known there, so each
asset read is a per-combo constant — and the read's asset derivative enters
the published marginal value $dV/dR$ through the continuation's direct
Euler-state channel (for regime-transition probabilities the first-order
term $\\sum_{targets} \\partial P/\\partial a \\cdot EV$, which Danskin does
not cancel). The reads must be smooth at the resolution of the Euler grid:
cliffs — including smooth bands narrower than one grid cell — are rejected
at model build with a hint to add grid nodes across the band.

The oracle for the solved properties is a dense-grid brute-force solve of a
mathematically equivalent spec (same state grids, dense consumption grid,
explicit budget constraint).
"""

import functools

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.typing import PeriodToRegimeToVArr
from lcm import (
    AgeGrid,
    DiscreteGrid,
    IrregSpacedGrid,
    LinSpacedGrid,
    MarkovTransition,
    Model,
    categorical,
)
from lcm.exceptions import ModelInitializationError
from lcm.regime import Regime as UserRegime
from lcm.solvers import DCEGM, GridSearch
from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    DiscreteState,
    FloatND,
    ScalarInt,
)
from lcm_examples.iskhakov_et_al_2017 import dead

# Number of model periods; the last one is spent in the terminal regime.
N_PERIODS = 4

# Wealth band over which the savings-stage smoothsteps ramp. At ~32 cells of
# the wealth grid the band is well resolved at node resolution, so the
# build-time continuity check admits it.
BAND_START = 30.0
BAND_WIDTH = 20.0

# A smoothstep band narrower than one wealth grid cell (cell size ~0.62): a
# cliff at node resolution, rejected at model build.
NARROW_BAND_WIDTH = 0.3

# Survival probability below/above the wealth band. Death is absorbing with
# zero value while continued life is worth several utils, so the value gap
# across targets is material and the probability's wealth slope contributes
# a first-order term to the marginal value inside the band.
SURVIVAL_LOW = 0.55
SURVIVAL_HIGH = 0.95

# Probability of good health below/above the band for the Markov-weight
# fixture, and the utility penalty of bad health.
GOOD_HEALTH_LOW = 0.4
GOOD_HEALTH_HIGH = 0.9
BAD_HEALTH_PENALTY = 0.2

# Skill dynamics of the passive-law fixture: persistence plus a
# wealth-dependent gain, capped at the grid's top node; skill enters utility
# linearly so its blended read is exact for both solvers.
SKILL_DECAY = 0.9
SKILL_GAIN = 0.3
SKILL_MAX = 1.0
SKILL_WEIGHT = 0.4

# Deterministic labor income added to savings in the wealth law; keeps child
# wealth queries away from the grid's lower edge, where the two solvers
# clamp differently.
LABOR_INCOME = 5.0

# 160 wealth nodes: the brute-force oracle interpolates next-period V
# linearly in wealth (a downward bias where V curves), while the DC-EGM carry
# read uses exact slopes; the dense grid keeps that oracle-side bias below
# the comparison tolerance.
WEALTH_GRID = LinSpacedGrid(start=1.0, stop=100.0, n_points=160)
SKILL_GRID = LinSpacedGrid(start=0.0, stop=SKILL_MAX, n_points=7)

# The consumption grid covers the maximum resources so the brute-force oracle
# is not artificially capped at high wealth; 4000 points keep the oracle's own
# resolution error below the comparison tolerance.
CONSUMPTION_GRID = LinSpacedGrid(start=0.25, stop=110.0, n_points=4000)

# Exogenous end-of-period savings grid, cubically clustered toward the
# borrowing limit where the value function curves hardest.
SAVINGS_GRID = IrregSpacedGrid(points=tuple(100.0 * (i / 149) ** 3 for i in range(150)))

# Lowest wealth nodes excluded from the brute-force comparison: there the
# brute solver leans on consumption choices near its grid start and on coarse
# interpolation where log utility curves hardest (the same exclusion the
# sibling DC-EGM oracle tests use).
N_BRUTE_UNSTABLE_NODES = 16


@categorical(ordered=False)
class SavingsStageRegimeId:
    working_life: ScalarInt
    dead: ScalarInt


@categorical(ordered=False)
class Health:
    good: ScalarInt
    bad: ScalarInt


def smoothstep_in_band(wealth: ContinuousState) -> FloatND:
    """C² quintic smoothstep rising from 0 to 1 across the wealth band."""
    t = jnp.clip((wealth - BAND_START) / BAND_WIDTH, 0.0, 1.0)
    return t * t * t * (t * (6.0 * t - 15.0) + 10.0)


def survival_of_wealth(wealth: ContinuousState) -> FloatND:
    return SURVIVAL_LOW + (SURVIVAL_HIGH - SURVIVAL_LOW) * smoothstep_in_band(wealth)


def stay_prob(wealth: ContinuousState, age: int, final_age_alive: float) -> FloatND:
    return jnp.where(age >= final_age_alive, 0.0, survival_of_wealth(wealth))


def death_prob(wealth: ContinuousState, age: int, final_age_alive: float) -> FloatND:
    return 1.0 - stay_prob(wealth, age, final_age_alive)


def utility(consumption: ContinuousAction) -> FloatND:
    return jnp.log(consumption)


def utility_with_health(
    consumption: ContinuousAction, health: DiscreteState
) -> FloatND:
    return jnp.log(consumption) - jnp.where(
        health == Health.bad, BAD_HEALTH_PENALTY, 0.0
    )


def utility_with_skill(
    consumption: ContinuousAction, skill: ContinuousState
) -> FloatND:
    return jnp.log(consumption) + SKILL_WEIGHT * skill


def resources(wealth: ContinuousState) -> FloatND:
    return wealth


def savings(resources: FloatND, consumption: ContinuousAction) -> FloatND:
    return resources - consumption


def inverse_marginal_utility(marginal_continuation: FloatND) -> FloatND:
    return 1.0 / marginal_continuation


def budget_constraint(consumption: ContinuousAction, wealth: ContinuousState) -> BoolND:
    return consumption <= wealth


def next_regime(age: int, final_age_alive: float) -> ScalarInt:
    return jnp.where(
        age >= final_age_alive,
        SavingsStageRegimeId.dead,
        SavingsStageRegimeId.working_life,
    )


def next_wealth_dcegm(savings: FloatND) -> ContinuousState:
    return savings + LABOR_INCOME


def next_wealth_brute(
    wealth: ContinuousState, consumption: ContinuousAction
) -> ContinuousState:
    return wealth - consumption + LABOR_INCOME


def next_skill(skill: ContinuousState, wealth: ContinuousState) -> ContinuousState:
    return jnp.minimum(
        SKILL_DECAY * skill + SKILL_GAIN * smoothstep_in_band(wealth), SKILL_MAX
    )


def health_weights(wealth: ContinuousState) -> FloatND:
    p_good = GOOD_HEALTH_LOW + (
        GOOD_HEALTH_HIGH - GOOD_HEALTH_LOW
    ) * smoothstep_in_band(wealth)
    return jnp.stack([p_good, 1.0 - p_good])


def _ages() -> AgeGrid:
    return AgeGrid(start=40, stop=40 + (N_PERIODS - 1) * 10, step="10Y")


def _active(age: int) -> bool:
    last_age = 40 + (N_PERIODS - 1) * 10
    return age < last_age


DCEGM_SOLVER = DCEGM(
    continuous_state="wealth",
    continuous_action="consumption",
    resources="resources",
    post_decision_function="savings",
    savings_grid=SAVINGS_GRID,
    n_constrained_points=64,
)


def _dcegm_functions() -> dict:
    return {
        "utility": utility,
        "resources": resources,
        "savings": savings,
        "inverse_marginal_utility": inverse_marginal_utility,
    }


def _params() -> dict:
    return {
        "discount_factor": 0.95,
        "final_age_alive": 40 + (N_PERIODS - 2) * 10,
    }


def _assert_working_life_V_matches(
    *, dcegm_solution: PeriodToRegimeToVArr, brute_solution: PeriodToRegimeToVArr
) -> None:
    for period in sorted(brute_solution)[:-1]:
        brute_V = np.asarray(brute_solution[period]["working_life"])
        dcegm_V = np.asarray(dcegm_solution[period]["working_life"])
        assert brute_V.shape == dcegm_V.shape
        np.testing.assert_allclose(
            dcegm_V[N_BRUTE_UNSTABLE_NODES:],
            brute_V[N_BRUTE_UNSTABLE_NODES:],
            atol=1e-2,
            rtol=1e-3,
            err_msg=f"period={period}",
        )


@functools.cache
def _survival_prob_model(solver: str) -> Model:
    """Self-targeting regime whose stay probability reads wealth smoothly."""
    is_dcegm = solver == "dcegm"
    working = UserRegime(
        transition={
            "working_life": MarkovTransition(stay_prob),
            "dead": MarkovTransition(death_prob),
        },
        active=_active,
        actions={"consumption": CONSUMPTION_GRID},
        states={"wealth": WEALTH_GRID},
        state_transitions={
            "wealth": next_wealth_dcegm if is_dcegm else next_wealth_brute
        },
        constraints={} if is_dcegm else {"budget_constraint": budget_constraint},
        functions={**(_dcegm_functions() if is_dcegm else {}), "utility": utility},
        solver=DCEGM_SOLVER if is_dcegm else GridSearch(),
    )
    return Model(
        regimes={"working_life": working, "dead": dead},
        ages=_ages(),
        regime_id_class=SavingsStageRegimeId,
    )


def test_smoothstep_survival_probability_matches_brute_force():
    """A wealth-dependent survival probability matches the brute-force oracle.

    The stay probability rises across a wealth band spanning many grid cells
    while death is an absorbing zero-value regime, so inside the band the
    marginal value carries the first-order term
    $\\partial P_{stay}/\\partial wealth \\cdot EV_{stay}$ on top of the
    envelope term. A solver that drops that term misprices saving in the
    band and fails the multi-period value comparison.
    """
    params = _params()
    dcegm_solution = _survival_prob_model("dcegm").solve(
        params=params, log_level="debug"
    )
    brute_solution = _survival_prob_model("brute_force").solve(
        params=params, log_level="debug"
    )
    _assert_working_life_V_matches(
        dcegm_solution=dcegm_solution, brute_solution=brute_solution
    )


@functools.cache
def _markov_health_model(solver: str) -> Model:
    """Markov health weights reading wealth through a smoothstep."""
    is_dcegm = solver == "dcegm"
    working = UserRegime(
        transition=next_regime,
        active=_active,
        actions={"consumption": CONSUMPTION_GRID},
        states={"wealth": WEALTH_GRID, "health": DiscreteGrid(Health)},
        state_transitions={
            "wealth": next_wealth_dcegm if is_dcegm else next_wealth_brute,
            "health": MarkovTransition(health_weights),
        },
        constraints={} if is_dcegm else {"budget_constraint": budget_constraint},
        functions={
            **(_dcegm_functions() if is_dcegm else {}),
            "utility": utility_with_health,
        },
        solver=DCEGM_SOLVER if is_dcegm else GridSearch(),
    )
    return Model(
        regimes={"working_life": working, "dead": dead},
        ages=_ages(),
        regime_id_class=SavingsStageRegimeId,
    )


def test_markov_weights_reading_wealth_match_brute_force():
    """Asset-reading Markov weights of a discrete state match brute force.

    The health transition weights read current wealth through a smoothstep, so
    the regime is solved per exogenous asset node, where wealth is known. The
    weights' wealth derivative enters the published marginal value through the
    continuation's stochastic-weight channel (the $\\partial w/\\partial a
    \\cdot EV$ term), exactly as the asset-node brute-force oracle evaluates
    it. Values agree across the full wealth-by-health grid.
    """
    params = _params()
    dcegm_solution = _markov_health_model("dcegm").solve(
        params=params, log_level="debug"
    )
    brute_solution = _markov_health_model("brute_force").solve(
        params=params, log_level="debug"
    )
    for period in sorted(brute_solution)[:-1]:
        brute_V = np.asarray(brute_solution[period]["working_life"])
        dcegm_V = np.asarray(dcegm_solution[period]["working_life"])
        # V leads with the discrete health axis; wealth is the trailing axis.
        assert brute_V.shape == dcegm_V.shape == (2, 160)
        np.testing.assert_allclose(
            dcegm_V[:, N_BRUTE_UNSTABLE_NODES:],
            brute_V[:, N_BRUTE_UNSTABLE_NODES:],
            atol=1e-2,
            rtol=1e-3,
            err_msg=f"period={period}",
        )


@functools.cache
def _passive_skill_model(solver: str) -> Model:
    """Passive skill state whose law reads wealth through a smoothstep."""
    is_dcegm = solver == "dcegm"
    working = UserRegime(
        transition=next_regime,
        active=_active,
        actions={"consumption": CONSUMPTION_GRID},
        states={"wealth": WEALTH_GRID, "skill": SKILL_GRID},
        state_transitions={
            "wealth": next_wealth_dcegm if is_dcegm else next_wealth_brute,
            "skill": next_skill,
        },
        constraints={} if is_dcegm else {"budget_constraint": budget_constraint},
        functions={
            **(_dcegm_functions() if is_dcegm else {}),
            "utility": utility_with_skill,
        },
        solver=DCEGM_SOLVER if is_dcegm else GridSearch(),
    )
    return Model(
        regimes={"working_life": working, "dead": dead},
        ages=_ages(),
        regime_id_class=SavingsStageRegimeId,
    )


def test_passive_skill_law_reading_wealth_matches_brute_force():
    """`next_skill = decay * skill + gain * smoothstep(wealth)` matches brute.

    The passive state's law reads current wealth, so the child's skill value
    shifts per asset node and its wealth derivative feeds the marginal value
    through the continuation's skill channel. Values agree with the
    dense-grid brute-force oracle on the full wealth-by-skill grid.
    """
    params = _params()
    dcegm_solution = _passive_skill_model("dcegm").solve(
        params=params, log_level="debug"
    )
    brute_solution = _passive_skill_model("brute_force").solve(
        params=params, log_level="debug"
    )
    _assert_working_life_V_matches(
        dcegm_solution=dcegm_solution, brute_solution=brute_solution
    )


def cliff_stay_prob(wealth: ContinuousState) -> FloatND:
    return jnp.where(
        wealth <= BAND_START + 0.5 * BAND_WIDTH, SURVIVAL_LOW, SURVIVAL_HIGH
    )


def cliff_death_prob(wealth: ContinuousState) -> FloatND:
    return 1.0 - cliff_stay_prob(wealth)


def narrow_stay_prob(wealth: ContinuousState) -> FloatND:
    t = jnp.clip((wealth - BAND_START) / NARROW_BAND_WIDTH, 0.0, 1.0)
    smooth = t * t * t * (t * (6.0 * t - 15.0) + 10.0)
    return SURVIVAL_LOW + (SURVIVAL_HIGH - SURVIVAL_LOW) * smooth


def narrow_death_prob(wealth: ContinuousState) -> FloatND:
    return 1.0 - narrow_stay_prob(wealth)


def smooth_stay_prob(wealth: ContinuousState) -> FloatND:
    return survival_of_wealth(wealth)


def smooth_death_prob(wealth: ContinuousState) -> FloatND:
    return 1.0 - survival_of_wealth(wealth)


def _build_model_with_survival_cells(stay, die) -> Model:
    working = UserRegime(
        transition={
            "working_life": MarkovTransition(stay),
            "dead": MarkovTransition(die),
        },
        active=_active,
        actions={"consumption": CONSUMPTION_GRID},
        states={"wealth": WEALTH_GRID},
        state_transitions={"wealth": next_wealth_dcegm},
        functions=_dcegm_functions(),
        solver=DCEGM_SOLVER,
    )
    return Model(
        regimes={"working_life": working, "dead": dead},
        ages=_ages(),
        regime_id_class=SavingsStageRegimeId,
    )


@pytest.mark.parametrize(
    ("stay", "die"),
    [
        pytest.param(cliff_stay_prob, cliff_death_prob, id="boolean_step"),
        pytest.param(narrow_stay_prob, narrow_death_prob, id="sub_cell_smoothstep"),
    ],
)
def test_cliff_in_regime_transition_probability_raises(stay, die):
    """A survival probability that jumps at node resolution fails at build.

    A boolean step is a genuine cliff; a smoothstep narrower than one wealth
    grid cell is a cliff at node resolution — the grid has no nodes across
    the band, so the per-asset-node evaluation cannot resolve it. Both are
    rejected at model build with a hint to add grid nodes across the band.
    """
    with pytest.raises(
        ModelInitializationError, match="add grid nodes across the band"
    ):
        _build_model_with_survival_cells(stay, die)


def test_smooth_band_regime_transition_probability_constructs():
    """A survival probability ramping over many grid cells builds fine."""
    model = _build_model_with_survival_cells(smooth_stay_prob, smooth_death_prob)
    assert model.n_periods == N_PERIODS
