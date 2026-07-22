"""DC-EGM carrying a stochastic Markov discrete state into the child.

A Markov discrete state is a node-valued discrete dimension whose next-period
node is drawn from an intrinsic transition law
$w(\\text{node}' \\mid \\text{node}, \\text{params})$ supplied by a
`MarkovTransition`. In a DC-EGM regime it rides on the own side exactly like a
plain discrete state (one carry row and one V slice per node) while the child
side takes an expectation: the child's node is distributed, so the carry read
indexes the child rows at every node, performs the full read there (resources
interpolation, passive blend, discrete-action aggregation), and weights the
resulting per-node values and marginals with the transition weights — the
expectation sits *outside* the action aggregation, matching the brute-force
solver's weighted average of the already action-aggregated next-period V.

The oracle is a dense-grid brute-force solve of a mathematically equivalent
spec (same state grids, dense consumption grid, explicit budget constraint).
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
    MarkovTransition,
    Model,
    RouwenhorstAR1Process,
    categorical,
)
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

# Number of model periods; the last one is spent in the terminal `dead` regime.
N_PERIODS = 4

# Utility penalty of being in bad health (consumption-separable).
BAD_HEALTH_PENALTY = 0.3

# Deterministic income added to savings in the wealth law; keeps child wealth
# queries away from the grid's lower edge, where the two solvers clamp
# differently.
LABOR_INCOME = 5.0

# 160 wealth nodes: the brute-force oracle interpolates next-period V linearly
# in wealth (a downward bias where V curves), while the DC-EGM carry read uses
# exact slopes; the dense grid keeps that oracle-side bias below the comparison
# tolerance.
WEALTH_GRID = LinSpacedGrid(start=1.0, stop=100.0, n_points=160)

# The consumption grid covers the maximum resources so the brute-force oracle
# is not artificially capped at high wealth; 4000 points keep the oracle's own
# resolution error below the comparison tolerance.
CONSUMPTION_GRID = LinSpacedGrid(start=0.25, stop=110.0, n_points=4000)

# Exogenous end-of-period savings grid, cubically clustered toward the
# borrowing limit where the value function curves hardest.
SAVINGS_GRID = IrregSpacedGrid(points=tuple(100.0 * (i / 149) ** 3 for i in range(150)))

# Lowest wealth nodes excluded from the brute-force comparison: there the brute
# solver leans on consumption choices near its grid start and on coarse
# interpolation where log utility curves hardest (the same exclusion the
# sibling DC-EGM oracle tests use).
N_BRUTE_UNSTABLE_NODES = 16


@categorical(ordered=False)
class MarkovRegimeId:
    working_life: ScalarInt
    dead: ScalarInt


@categorical(ordered=False)
class Health:
    good: ScalarInt
    bad: ScalarInt


def utility_with_health(
    consumption: ContinuousAction, health: DiscreteState
) -> FloatND:
    return jnp.log(consumption) - jnp.where(
        health == Health.bad, BAD_HEALTH_PENALTY, 0.0
    )


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
        MarkovRegimeId.dead,
        MarkovRegimeId.working_life,
    )


def next_wealth_dcegm(savings: FloatND) -> ContinuousState:
    return savings + LABOR_INCOME


def next_wealth_brute(
    wealth: ContinuousState, consumption: ContinuousAction
) -> ContinuousState:
    return wealth - consumption + LABOR_INCOME


def health_transition(health: DiscreteState, age: int) -> FloatND:
    """Markov health law varying by current health and period.

    Health is sticky and improves with age. The row over next-period
    `[good, bad]` returns a probability vector summing to one for each current
    health node.
    """
    # Older workers face a higher chance of staying healthy; the current
    # health node sets persistence.
    age_bonus = jnp.clip((age - 40) / 60.0, 0.0, 0.2)
    p_good = jnp.where(
        health == Health.good,
        0.65 + age_bonus,
        0.25 + age_bonus,
    )
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
        "utility": utility_with_health,
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
        # The Markov health node is the leading axis of V; wealth is the
        # trailing axis. Exclude the lowest wealth nodes from the comparison.
        np.testing.assert_allclose(
            dcegm_V[:, N_BRUTE_UNSTABLE_NODES:],
            brute_V[:, N_BRUTE_UNSTABLE_NODES:],
            atol=1e-2,
            rtol=1e-3,
            err_msg=f"period={period}",
        )


@functools.cache
def _same_grid_markov_model(solver: str) -> Model:
    """A Markov health state carried every period into the same-grid target."""
    is_dcegm = solver == "dcegm"
    working = UserRegime(
        transition=next_regime,
        active=_active,
        actions={"consumption": CONSUMPTION_GRID},
        states={"wealth": WEALTH_GRID, "health": DiscreteGrid(Health)},
        state_transitions={
            "wealth": next_wealth_dcegm if is_dcegm else next_wealth_brute,
            "health": MarkovTransition(health_transition),
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
        regime_id_class=MarkovRegimeId,
    )


def test_same_grid_markov_state_matches_brute_force():
    """A Markov health state carried into the same grid matches brute force.

    The child's health node is distributed per the `MarkovTransition` weights
    at the parent's node and period; the carry read indexes the child rows at
    every node and weight-sums the per-node values *outside* the consumption
    choice. Values agree with the dense-grid brute-force oracle on the full
    health-by-wealth grid, excluding the lowest wealth nodes.
    """
    params = _params()
    dcegm_solution = _same_grid_markov_model("dcegm").solve(
        params=params, log_level="debug"
    )
    brute_solution = _same_grid_markov_model("brute_force").solve(
        params=params, log_level="debug"
    )
    _assert_working_life_V_matches(
        dcegm_solution=dcegm_solution, brute_solution=brute_solution
    )


@categorical(ordered=False)
class Health3:
    good: ScalarInt
    fair: ScalarInt
    bad: ScalarInt


@categorical(ordered=False)
class CrossGridRegimeId:
    early: ScalarInt
    late: ScalarInt
    dead: ScalarInt


def utility_early(consumption: ContinuousAction, health: DiscreteState) -> FloatND:
    """Utility with a three-level health penalty."""
    penalty = jnp.where(
        health == Health3.good,
        0.0,
        jnp.where(health == Health3.fair, 0.5 * BAD_HEALTH_PENALTY, BAD_HEALTH_PENALTY),
    )
    return jnp.log(consumption) - penalty


def remap_health_to_two(health: DiscreteState) -> FloatND:
    """Markov remap from the three-level grid onto the two-level `[good, bad]`.

    `good` stays mostly good, `fair` splits, `bad` stays mostly bad — a
    probability vector of length two (the *target* grid), shorter than the
    source's three-level grid.
    """
    p_good = jnp.where(
        health == Health3.good,
        0.8,
        jnp.where(health == Health3.fair, 0.5, 0.1),
    )
    return jnp.stack([p_good, 1.0 - p_good])


def late_health_transition(health: DiscreteState) -> FloatND:
    """Sticky two-level health law within the late regime."""
    p_good = jnp.where(health == Health.good, 0.7, 0.3)
    return jnp.stack([p_good, 1.0 - p_good])


def to_live_prob(age: int, final_age_alive: float) -> FloatND:
    return jnp.where(age >= final_age_alive, 0.0, 1.0)


def to_dead_prob(age: int, final_age_alive: float) -> FloatND:
    return jnp.where(age >= final_age_alive, 1.0, 0.0)


@functools.cache
def _cross_grid_markov_model(solver: str) -> Model:
    """A three-level Markov health remapped onto a two-level child grid.

    The early regime carries `health` on a three-level grid and transitions
    every period into the late regime, whose `health` lives on a two-level
    grid. The remap weights returned for `next_health` have the *target* (late)
    grid's length, exercising a stochastic axis whose size differs from the
    source's discrete grid.
    """
    is_dcegm = solver == "dcegm"
    early = UserRegime(
        transition={
            "late": MarkovTransition(to_live_prob),
            "dead": MarkovTransition(to_dead_prob),
        },
        active=lambda age: age < 40 + (N_PERIODS - 1) * 10,
        actions={"consumption": CONSUMPTION_GRID},
        states={"wealth": WEALTH_GRID, "health": DiscreteGrid(Health3)},
        state_transitions={
            "wealth": next_wealth_dcegm if is_dcegm else next_wealth_brute,
            "health": {"late": MarkovTransition(remap_health_to_two)},
        },
        constraints={} if is_dcegm else {"budget_constraint": budget_constraint},
        functions={
            **(
                {
                    "resources": resources,
                    "savings": savings,
                    "inverse_marginal_utility": inverse_marginal_utility,
                }
                if is_dcegm
                else {}
            ),
            "utility": utility_early,
        },
        solver=DCEGM_SOLVER if is_dcegm else GridSearch(),
    )
    late = UserRegime(
        transition={
            "late": MarkovTransition(to_live_prob),
            "dead": MarkovTransition(to_dead_prob),
        },
        active=lambda age: age < 40 + (N_PERIODS - 1) * 10,
        actions={"consumption": CONSUMPTION_GRID},
        states={"wealth": WEALTH_GRID, "health": DiscreteGrid(Health)},
        state_transitions={
            "wealth": next_wealth_dcegm if is_dcegm else next_wealth_brute,
            "health": {"late": MarkovTransition(late_health_transition)},
        },
        constraints={} if is_dcegm else {"budget_constraint": budget_constraint},
        functions={
            **(_dcegm_functions() if is_dcegm else {}),
            "utility": utility_with_health,
        },
        solver=DCEGM_SOLVER if is_dcegm else GridSearch(),
    )
    return Model(
        regimes={"early": early, "late": late, "dead": dead},
        ages=_ages(),
        regime_id_class=CrossGridRegimeId,
    )


def test_cross_grid_markov_state_matches_brute_force():
    """A three-level Markov health remapped onto a two-level child grid solves.

    The early regime's `next_health` weight vector has the late (target)
    grid's length — shorter than the source's three-level health grid — so the
    integration ranges over the *child's* node axis, not the parent's. Values
    in the early regime agree with the dense-grid brute-force oracle.
    """
    params = _params()
    dcegm_solution = _cross_grid_markov_model("dcegm").solve(
        params=params, log_level="debug"
    )
    brute_solution = _cross_grid_markov_model("brute_force").solve(
        params=params, log_level="debug"
    )
    for period in sorted(brute_solution)[:-1]:
        brute_V = np.asarray(brute_solution[period]["early"])
        dcegm_V = np.asarray(dcegm_solution[period]["early"])
        assert brute_V.shape == dcegm_V.shape == (3, 160)
        np.testing.assert_allclose(
            dcegm_V[:, N_BRUTE_UNSTABLE_NODES:],
            brute_V[:, N_BRUTE_UNSTABLE_NODES:],
            atol=1e-2,
            rtol=1e-3,
            err_msg=f"period={period}",
        )


# Number of discretization nodes of the income process for the joint fixture.
N_INCOME_NODES = 5

# Unconditional income floor added to next wealth in the joint fixture; keeps
# continuation wealth inside the wealth grid even at zero savings.
BASE_INCOME = 5.0


def utility_with_health_only(
    consumption: ContinuousAction, health: DiscreteState
) -> FloatND:
    return jnp.log(consumption) - jnp.where(
        health == Health.bad, BAD_HEALTH_PENALTY, 0.0
    )


def next_wealth_joint_dcegm(
    savings: FloatND, income: ContinuousState
) -> ContinuousState:
    return savings + BASE_INCOME + jnp.exp(income)


def next_wealth_joint_brute(
    wealth: ContinuousState,
    consumption: ContinuousAction,
    income: ContinuousState,
) -> ContinuousState:
    return wealth - consumption + BASE_INCOME + jnp.exp(income)


def joint_health_transition(health: DiscreteState) -> FloatND:
    p_good = jnp.where(health == Health.good, 0.7, 0.35)
    return jnp.stack([p_good, 1.0 - p_good])


@functools.cache
def _joint_process_markov_model(solver: str) -> Model:
    """An AR(1) income process and a Markov health state on the same target.

    A self-targeting working regime carries both a Rouwenhorst income process
    (entering the wealth law) and a Markov health state (penalising utility).
    Both are stochastic node axes of the same carry target, so the child read
    integrates over the joint income-by-health node mesh.
    """
    is_dcegm = solver == "dcegm"
    working = UserRegime(
        transition=next_regime,
        active=_active,
        actions={"consumption": CONSUMPTION_GRID},
        states={
            "wealth": WEALTH_GRID,
            "income": RouwenhorstAR1Process(n_points=N_INCOME_NODES),
            "health": DiscreteGrid(Health),
        },
        state_transitions={
            "wealth": next_wealth_joint_dcegm if is_dcegm else next_wealth_joint_brute,
            "health": MarkovTransition(joint_health_transition),
        },
        constraints={} if is_dcegm else {"budget_constraint": budget_constraint},
        functions={
            **(
                {
                    "resources": resources,
                    "savings": savings,
                    "inverse_marginal_utility": inverse_marginal_utility,
                }
                if is_dcegm
                else {}
            ),
            "utility": utility_with_health_only,
        },
        solver=DCEGM_SOLVER if is_dcegm else GridSearch(),
    )
    return Model(
        regimes={"working_life": working, "dead": dead},
        ages=_ages(),
        regime_id_class=MarkovRegimeId,
    )


def _joint_params() -> dict:
    return {
        "discount_factor": 0.95,
        "final_age_alive": 40 + (N_PERIODS - 2) * 10,
        "working_life": {"income": {"mu": 0.0, "sigma": 0.25, "rho": 0.6}},
    }


def test_joint_process_and_markov_state_matches_brute_force():
    """An AR(1) process and a Markov state on one target match brute force.

    The child read integrates over the joint income-by-health node mesh: the
    income node feeds the wealth law and the health node selects the carry's
    leading health axis, weighted by the outer product of the AR(1) and Markov
    transition vectors. Values agree with the dense-grid brute-force oracle on
    the full income-by-health-by-wealth grid.
    """
    params = _joint_params()
    dcegm_solution = _joint_process_markov_model("dcegm").solve(
        params=params, log_level="debug"
    )
    brute_solution = _joint_process_markov_model("brute_force").solve(
        params=params, log_level="debug"
    )
    for period in sorted(brute_solution)[:-1]:
        brute_V = np.asarray(brute_solution[period]["working_life"])
        dcegm_V = np.asarray(dcegm_solution[period]["working_life"])
        assert brute_V.shape == dcegm_V.shape == (N_INCOME_NODES, 2, 160)
        np.testing.assert_allclose(
            dcegm_V[..., N_BRUTE_UNSTABLE_NODES:],
            brute_V[..., N_BRUTE_UNSTABLE_NODES:],
            atol=1e-2,
            rtol=1e-3,
            err_msg=f"period={period}",
        )


# Guaranteed-minimum-income floor: a transfer tops next-period cash-on-hand up
# to the floor, so low savings land on the constrained segment.
INCOME_FLOOR = 12.0


def income_transfer(savings: FloatND) -> FloatND:
    """Top-up transfer raising next-period resources to the income floor."""
    return jnp.maximum(0.0, INCOME_FLOOR - (savings + LABOR_INCOME))


def income_transfer_brute(
    wealth: ContinuousState, consumption: ContinuousAction
) -> FloatND:
    return jnp.maximum(0.0, INCOME_FLOOR - (wealth - consumption + LABOR_INCOME))


def next_wealth_floor_dcegm(
    savings: FloatND, income_transfer: FloatND
) -> ContinuousState:
    return savings + LABOR_INCOME + income_transfer


def next_wealth_floor_brute(
    wealth: ContinuousState, consumption: ContinuousAction, income_transfer: FloatND
) -> ContinuousState:
    return wealth - consumption + LABOR_INCOME + income_transfer


def point_mass_health_transition(health: DiscreteState) -> FloatND:
    """Degenerate Markov row: all mass on `good` regardless of current health."""
    p_good = jnp.where(health == Health.bad, 1.0, 1.0)
    return jnp.stack([p_good, 1.0 - p_good])


@functools.cache
def _point_mass_floor_model(solver: str) -> Model:
    """A point-mass Markov health in a consumption-floor model.

    The wealth law adds a guaranteed-minimum-income transfer, so low savings
    push next-period resources onto the floor (the constrained segment). The
    Markov health row places all mass on `good`, so its integration must
    reproduce the deterministic-index result.
    """
    is_dcegm = solver == "dcegm"
    working = UserRegime(
        transition=next_regime,
        active=_active,
        actions={"consumption": CONSUMPTION_GRID},
        states={"wealth": WEALTH_GRID, "health": DiscreteGrid(Health)},
        state_transitions={
            "wealth": next_wealth_floor_dcegm if is_dcegm else next_wealth_floor_brute,
            "health": MarkovTransition(point_mass_health_transition),
        },
        constraints={} if is_dcegm else {"budget_constraint": budget_constraint},
        functions={
            **(_dcegm_functions() if is_dcegm else {}),
            "utility": utility_with_health,
            "income_transfer": income_transfer if is_dcegm else income_transfer_brute,
        },
        solver=DCEGM_SOLVER if is_dcegm else GridSearch(),
    )
    return Model(
        regimes={"working_life": working, "dead": dead},
        ages=_ages(),
        regime_id_class=MarkovRegimeId,
    )


def test_point_mass_markov_with_income_floor_matches_brute_force():
    """A degenerate Markov row in a consumption-floor model matches brute force.

    The Markov health row places all mass on `good`, so the integration over
    the health node reproduces the deterministic-index result; the wealth law's
    income-floor transfer pushes low savings onto the constrained segment,
    where DC-EGM uses its closed-form credit-constrained candidates. Both the
    point-mass integration and the floor behaviour agree with the dense-grid
    brute-force oracle across the full health-by-wealth grid.
    """
    params = _params()
    dcegm_solution = _point_mass_floor_model("dcegm").solve(
        params=params, log_level="debug"
    )
    brute_solution = _point_mass_floor_model("brute_force").solve(
        params=params, log_level="debug"
    )
    _assert_working_life_V_matches(
        dcegm_solution=dcegm_solution, brute_solution=brute_solution
    )
