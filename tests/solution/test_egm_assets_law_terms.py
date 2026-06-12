"""DC-EGM with additive action-constant terms in the Euler state's law.

The admitted law shape is `next_wealth = savings + g(...)` per target regime,
where the residual `g` is constant in the continuous action. `g` may read own
discrete states, discrete actions, params, transition-DAG outputs of the same
target — and the current Euler state through decision-time functions, which is
evaluated exactly at the exogenous asset nodes (where the brute-force oracle
evaluates it too).

The oracle for every property is a dense-grid brute-force solve of a
mathematically equivalent spec (same state grids, dense consumption grid,
explicit budget constraint).
"""

import functools

import jax.numpy as jnp
import numpy as np

from _lcm.typing import PeriodToRegimeToVArr
from lcm import (
    AgeGrid,
    DiscreteGrid,
    IrregSpacedGrid,
    LinSpacedGrid,
    LogSpacedGrid,
    MarkovTransition,
    Model,
    Phased,
    categorical,
    fixed_transition,
)
from lcm.regime import Regime as UserRegime
from lcm.solvers import DCEGM, BruteForce
from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    DiscreteAction,
    DiscreteState,
    FloatND,
    ScalarInt,
)
from lcm_examples.iskhakov_et_al_2017 import dead

# Number of model periods; the last one is spent in the terminal regime.
N_PERIODS = 4

# Health transfer received in bad health, and out-of-pocket costs with and
# without private insurance — the discrete-state and discrete-action inputs of
# the additive law term. The net term stays non-negative so child queries do
# not pile onto the wealth grid's lower edge, where the two solvers clamp
# differently.
HEALTH_TRANSFER = 8.0
OOP_WITH_INSURANCE = 2.0
OOP_WITHOUT_INSURANCE = 6.0

# Utility cost of buying private insurance (premium); keeps the insurance
# choice interior so both rows of the law term are exercised.
INSURANCE_PREMIUM = 0.25

# Capital-income supplement: a means-tested transfer with a kinked PHASE-OUT
# (full below the means-test cap, linearly phased out toward zero) plus a
# proportional match — the Euler-state dependence of the law term. The
# phase-out keeps the term CONTINUOUS in wealth: a hard cliff
# (`jnp.where(capital_income <= cap, BASE, 0)`) is rejected at model build,
# because a jump in the residual makes the child's value function
# discontinuous and the true policy bunches at the discontinuity — a corner
# outside EGM's candidate set.
CAPITAL_RETURN = 0.04
MEANS_TEST_CAP = 2.0
PHASE_OUT_END = 3.0  # capital income at which the supplement is fully phased out
BASE_SUPPLEMENT = 5.0
CAPITAL_MATCH = 0.5

# Survival probability per period for the per-target-asymmetry model.
SURVIVAL_RATE = 0.8

# Solve-phase pension adjustment for the phase-variant law term.
SOLVE_ADJUSTMENT = 4.0

# Pension accrual per unit of next-period AIME for the chained-transition law
# term, and the AIME increment earned by one period of work.
PENSION_ACCRUAL = 0.5
AIME_GAIN = 0.4
AIME_MAX = 1.5

# 160 wealth nodes: the brute-force oracle interpolates next-period V
# linearly in wealth (a downward bias where V curves), while the DC-EGM carry
# read uses exact slopes; the dense grid keeps that oracle-side bias below
# the comparison tolerance.
WEALTH_GRID = LinSpacedGrid(start=1.0, stop=100.0, n_points=160)
AIME_GRID = LinSpacedGrid(start=0.0, stop=AIME_MAX, n_points=7)
BEQUEST_GRID = LogSpacedGrid(start=0.5, stop=150.0, n_points=200)

# The consumption grid covers the maximum resources so the brute-force oracle
# is not artificially capped at high wealth; 4000 points keep the oracle's own
# resolution error below the comparison tolerance even at low wealth, where
# the additive law terms shift child queries away from the brute solver's
# interpolation nodes.
CONSUMPTION_GRID = LinSpacedGrid(start=0.25, stop=110.0, n_points=4000)

# Exogenous end-of-period savings grid, cubically clustered toward the
# borrowing limit where the value function curves hardest.
SAVINGS_GRID = IrregSpacedGrid(points=tuple(100.0 * (i / 149) ** 3 for i in range(150)))

# Lowest wealth nodes (below wealth ~10) excluded from the brute-force
# comparison: there the brute solver leans on consumption choices near its
# grid start and on coarse interpolation where log utility curves hardest
# (the same exclusion the sibling DC-EGM oracle tests use, in this file's
# denser wealth-grid units).
N_BRUTE_UNSTABLE_NODES = 16


@categorical(ordered=False)
class LawTermRegimeId:
    working_life: ScalarInt
    dead: ScalarInt


@categorical(ordered=False)
class Health:
    good: ScalarInt
    bad: ScalarInt


@categorical(ordered=False)
class Insurance:
    skip: ScalarInt
    buy: ScalarInt


@categorical(ordered=True)
class LaborChoice:
    work: ScalarInt
    rest: ScalarInt


def utility(consumption: ContinuousAction) -> FloatND:
    return jnp.log(consumption)


def utility_with_premium(
    consumption: ContinuousAction, buy_private: DiscreteAction
) -> FloatND:
    return jnp.log(consumption) - jnp.where(
        buy_private == Insurance.buy, INSURANCE_PREMIUM, 0.0
    )


def utility_with_work(
    consumption: ContinuousAction, labor_supply: DiscreteAction
) -> FloatND:
    return jnp.log(consumption) - jnp.where(labor_supply == LaborChoice.work, 0.3, 0.0)


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
        LawTermRegimeId.dead,
        LawTermRegimeId.working_life,
    )


def health_net_transfer(health: DiscreteState, buy_private: DiscreteAction) -> FloatND:
    oop = jnp.where(
        buy_private == Insurance.buy, OOP_WITH_INSURANCE, OOP_WITHOUT_INSURANCE
    )
    return jnp.where(health == Health.bad, HEALTH_TRANSFER - oop, 0.0)


def capital_supplement(wealth: ContinuousState) -> FloatND:
    capital_income = CAPITAL_RETURN * wealth
    phase_out_share = jnp.clip(
        (PHASE_OUT_END - capital_income) / (PHASE_OUT_END - MEANS_TEST_CAP), 0.0, 1.0
    )
    return BASE_SUPPLEMENT * phase_out_share + CAPITAL_MATCH * capital_income


def _stay_prob(age: int, final_age_alive: float, survival_rate: float) -> FloatND:
    return jnp.where(age >= final_age_alive, 0.0, survival_rate)


def _death_prob(age: int, final_age_alive: float, survival_rate: float) -> FloatND:
    return jnp.where(age >= final_age_alive, 1.0, 1.0 - survival_rate)


def is_working(labor_supply: DiscreteAction) -> BoolND:
    return labor_supply == LaborChoice.work


def next_aime(aime: ContinuousState, is_working: BoolND) -> ContinuousState:
    return jnp.minimum(aime + jnp.where(is_working, AIME_GAIN, 0.0), AIME_MAX)


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
def _health_insurance_model(solver: str) -> Model:
    """Law term reading a discrete state and a discrete action."""

    def next_wealth_dcegm(
        savings: FloatND, health_net_transfer: FloatND
    ) -> ContinuousState:
        return savings + health_net_transfer

    def next_wealth_brute(
        wealth: ContinuousState,
        consumption: ContinuousAction,
        health_net_transfer: FloatND,
    ) -> ContinuousState:
        return wealth - consumption + health_net_transfer

    is_dcegm = solver == "dcegm"
    working = UserRegime(
        transition=next_regime,
        active=_active,
        actions={
            "buy_private": DiscreteGrid(Insurance),
            "consumption": CONSUMPTION_GRID,
        },
        states={"wealth": WEALTH_GRID, "health": DiscreteGrid(Health)},
        state_transitions={
            "wealth": next_wealth_dcegm if is_dcegm else next_wealth_brute,
            "health": fixed_transition("health"),
        },
        constraints={} if is_dcegm else {"budget_constraint": budget_constraint},
        functions={
            **(_dcegm_functions() if is_dcegm else {}),
            "utility": utility_with_premium,
            "health_net_transfer": health_net_transfer,
        },
        solver=DCEGM_SOLVER if is_dcegm else BruteForce(),
    )
    return Model(
        regimes={"working_life": working, "dead": dead},
        ages=_ages(),
        regime_id_class=LawTermRegimeId,
    )


def test_discrete_state_and_action_law_term_matches_brute_force():
    """`next_wealth = savings + transfer(health) - oop(buy_private)` solves.

    The additive term reads a discrete state and a discrete action — both
    per-combo constants at the savings stage — so the child query shifts per
    combo and the composed gradient is unchanged. Values agree with the
    dense-grid brute-force oracle up to its consumption-grid resolution.
    """
    params = _params()
    dcegm_solution = _health_insurance_model("dcegm").solve(
        params=params, log_level="debug"
    )
    brute_solution = _health_insurance_model("brute_force").solve(
        params=params, log_level="debug"
    )
    _assert_working_life_V_matches(
        dcegm_solution=dcegm_solution, brute_solution=brute_solution
    )


@functools.cache
def _means_test_model(solver: str) -> Model:
    """Law term reading the Euler state through a decision-time function."""

    def next_wealth_dcegm(
        savings: FloatND, capital_supplement: FloatND
    ) -> ContinuousState:
        return savings + capital_supplement

    def next_wealth_brute(
        wealth: ContinuousState,
        consumption: ContinuousAction,
        capital_supplement: FloatND,
    ) -> ContinuousState:
        return wealth - consumption + capital_supplement

    is_dcegm = solver == "dcegm"
    working = UserRegime(
        transition=next_regime,
        active=_active,
        actions={"consumption": CONSUMPTION_GRID},
        states={"wealth": WEALTH_GRID},
        state_transitions={
            "wealth": next_wealth_dcegm if is_dcegm else next_wealth_brute
        },
        constraints={} if is_dcegm else {"budget_constraint": budget_constraint},
        functions={
            **(_dcegm_functions() if is_dcegm else {}),
            "utility": utility,
            "capital_supplement": capital_supplement,
        },
        solver=DCEGM_SOLVER if is_dcegm else BruteForce(),
    )
    return Model(
        regimes={"working_life": working, "dead": dead},
        ages=_ages(),
        regime_id_class=LawTermRegimeId,
    )


def test_euler_state_law_term_with_means_test_matches_brute_force():
    """A means-tested capital supplement in the law matches brute force.

    The additive term reads the current Euler state through a decision-time
    function (capital income), with a kinked but CONTINUOUS phase-out — a
    hard cliff is rejected by validation, because the true policy bunches at
    the value discontinuity it induces, outside EGM's candidate set. The term
    is evaluated at the exogenous asset nodes — exactly where the brute-force
    oracle evaluates it — so the two solvers agree at every node up to the
    brute solver's consumption-grid resolution.
    """
    params = _params()
    dcegm_solution = _means_test_model("dcegm").solve(params=params, log_level="debug")
    brute_solution = _means_test_model("brute_force").solve(
        params=params, log_level="debug"
    )
    _assert_working_life_V_matches(
        dcegm_solution=dcegm_solution, brute_solution=brute_solution
    )


def _bequest_utility(wealth: ContinuousState) -> FloatND:
    return jnp.log(wealth)


@functools.cache
def _per_target_model(solver: str) -> Model:
    """Per-target asymmetry: the law term applies toward one target only."""
    bequest = UserRegime(
        transition=None,
        states={"wealth": BEQUEST_GRID},
        functions={"utility": _bequest_utility},
    )

    def next_wealth_self_dcegm(
        savings: FloatND, health_net_transfer: FloatND
    ) -> ContinuousState:
        return savings + health_net_transfer

    def next_wealth_bequest_dcegm(savings: FloatND) -> ContinuousState:
        return savings

    def next_wealth_self_brute(
        wealth: ContinuousState,
        consumption: ContinuousAction,
        health_net_transfer: FloatND,
    ) -> ContinuousState:
        return wealth - consumption + health_net_transfer

    def next_wealth_bequest_brute(
        wealth: ContinuousState, consumption: ContinuousAction
    ) -> ContinuousState:
        return wealth - consumption

    is_dcegm = solver == "dcegm"
    working = UserRegime(
        transition={
            "working_life": MarkovTransition(_stay_prob),
            "dead": MarkovTransition(_death_prob),
        },
        active=_active,
        actions={
            "buy_private": DiscreteGrid(Insurance),
            "consumption": CONSUMPTION_GRID,
        },
        states={"wealth": WEALTH_GRID, "health": DiscreteGrid(Health)},
        state_transitions={
            "wealth": {
                "working_life": (
                    next_wealth_self_dcegm if is_dcegm else next_wealth_self_brute
                ),
                "dead": (
                    next_wealth_bequest_dcegm if is_dcegm else next_wealth_bequest_brute
                ),
            },
            "health": fixed_transition("health"),
        },
        constraints={} if is_dcegm else {"budget_constraint": budget_constraint},
        functions={
            **(_dcegm_functions() if is_dcegm else {}),
            "utility": utility_with_premium,
            "health_net_transfer": health_net_transfer,
        },
        solver=DCEGM_SOLVER if is_dcegm else BruteForce(),
    )
    return Model(
        regimes={"working_life": working, "dead": bequest},
        ages=_ages(),
        regime_id_class=LawTermRegimeId,
    )


def test_per_target_law_term_asymmetry_matches_brute_force():
    """The law term applies toward the live target and not toward `dead`.

    Wealth carried into death is `savings` alone (the bequest receives no
    health transfer), while wealth carried into continued life includes the
    transfer — declared as a per-target dict. Values agree with the brute
    oracle solving the same per-target laws.
    """
    params = {**_params(), "survival_rate": SURVIVAL_RATE}
    dcegm_solution = _per_target_model("dcegm").solve(params=params, log_level="debug")
    brute_solution = _per_target_model("brute_force").solve(
        params=params, log_level="debug"
    )
    _assert_working_life_V_matches(
        dcegm_solution=dcegm_solution, brute_solution=brute_solution
    )


@functools.cache
def _phased_law_model(solver: str) -> Model:
    """Phase-variant law term: solve-phase adjustment, simulate-phase zero."""

    def next_wealth_solve_dcegm(savings: FloatND) -> ContinuousState:
        return savings + SOLVE_ADJUSTMENT

    def next_wealth_simulate_dcegm(savings: FloatND) -> ContinuousState:
        return savings

    def next_wealth_solve_brute(
        wealth: ContinuousState, consumption: ContinuousAction
    ) -> ContinuousState:
        return wealth - consumption + SOLVE_ADJUSTMENT

    def next_wealth_simulate_brute(
        wealth: ContinuousState, consumption: ContinuousAction
    ) -> ContinuousState:
        return wealth - consumption

    is_dcegm = solver == "dcegm"
    working = UserRegime(
        transition=next_regime,
        active=_active,
        actions={"consumption": CONSUMPTION_GRID},
        states={"wealth": WEALTH_GRID},
        state_transitions={
            "wealth": (
                Phased(
                    solve=next_wealth_solve_dcegm,
                    simulate=next_wealth_simulate_dcegm,
                )
                if is_dcegm
                else Phased(
                    solve=next_wealth_solve_brute,
                    simulate=next_wealth_simulate_brute,
                )
            ),
        },
        constraints={} if is_dcegm else {"budget_constraint": budget_constraint},
        functions={**(_dcegm_functions() if is_dcegm else {}), "utility": utility},
        solver=DCEGM_SOLVER if is_dcegm else BruteForce(),
    )
    return Model(
        regimes={"working_life": working, "dead": dead},
        ages=_ages(),
        regime_id_class=LawTermRegimeId,
    )


def test_phase_variant_law_term_solves_and_simulates_with_its_phase():
    """A `Phased` law term uses its solve variant in V and zero in simulate.

    The solve-phase law adds a constant adjustment to savings; the
    simulate-phase law is `savings` alone. Solved values agree with the brute
    oracle declaring the same `Phased` law. (That the simulated wealth path
    obeys the simulate variant is asserted where DC-EGM forward simulation
    exists, in `tests/simulation/test_simulate_dcegm.py`.)
    """
    params = _params()
    dcegm_solution = _phased_law_model("dcegm").solve(params=params, log_level="debug")
    brute_solution = _phased_law_model("brute_force").solve(
        params=params, log_level="debug"
    )
    _assert_working_life_V_matches(
        dcegm_solution=dcegm_solution, brute_solution=brute_solution
    )


@functools.cache
def _chained_law_model(solver: str) -> Model:
    """Law term consuming a transition-DAG output of the same target."""

    def next_wealth_dcegm(
        savings: FloatND, next_aime: ContinuousState
    ) -> ContinuousState:
        return savings + PENSION_ACCRUAL * next_aime

    def next_wealth_brute(
        wealth: ContinuousState,
        consumption: ContinuousAction,
        next_aime: ContinuousState,
    ) -> ContinuousState:
        return wealth - consumption + PENSION_ACCRUAL * next_aime

    is_dcegm = solver == "dcegm"
    working = UserRegime(
        transition=next_regime,
        active=_active,
        actions={
            "labor_supply": DiscreteGrid(LaborChoice),
            "consumption": CONSUMPTION_GRID,
        },
        states={"wealth": WEALTH_GRID, "aime": AIME_GRID},
        state_transitions={
            "wealth": next_wealth_dcegm if is_dcegm else next_wealth_brute,
            "aime": next_aime,
        },
        constraints={} if is_dcegm else {"budget_constraint": budget_constraint},
        functions={
            **(_dcegm_functions() if is_dcegm else {}),
            "utility": utility_with_work,
            "is_working": is_working,
        },
        solver=DCEGM_SOLVER if is_dcegm else BruteForce(),
    )
    return Model(
        regimes={"working_life": working, "dead": dead},
        ages=_ages(),
        regime_id_class=LawTermRegimeId,
    )


def test_chained_transition_law_term_matches_brute_force():
    """`next_wealth = savings + accrual * next_aime` matches brute force.

    The law term consumes another state's transition output within the same
    target DAG (the AIME accrual driven by the work choice), so the child
    query shifts by a per-combo value computed inside the transition DAG.
    """
    params = _params()
    dcegm_solution = _chained_law_model("dcegm").solve(params=params, log_level="debug")
    brute_solution = _chained_law_model("brute_force").solve(
        params=params, log_level="debug"
    )
    for period in sorted(brute_solution)[:-1]:
        brute_V = np.asarray(brute_solution[period]["working_life"])
        dcegm_V = np.asarray(dcegm_solution[period]["working_life"])
        assert brute_V.shape == dcegm_V.shape == (160, 7)
        np.testing.assert_allclose(
            dcegm_V[N_BRUTE_UNSTABLE_NODES:, :],
            brute_V[N_BRUTE_UNSTABLE_NODES:, :],
            atol=1e-2,
            rtol=1e-3,
            err_msg=f"period={period}",
        )
