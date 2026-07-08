"""Ride-along toy with a binary discrete choice over a kinked budget.

A stochastic income node rides along a kink tax schedule (the continuation
integrates over the child income nodes), and each period the agent also chooses
whether to buy private insurance — a premium that lowers cash-on-hand. NBEGM
must solve the continuous consumption/savings subproblem per ride cell *and*
per discrete branch, then take the discrete choice by the upper envelope over
the branch values. The brute variant maximises over the discrete choice and
consumption on a dense grid and is the agreement oracle.

The schedule here is a pure kink (no jump), so the continuation carries no
in-period value jump: this isolates the discrete-envelope-over-ride-cells
composition from the published-jump topology.
"""

import jax.numpy as jnp

import lcm
from _lcm.grids.base import Grid
from lcm import DiscreteGrid, LinSpacedGrid, Model, NormalIIDProcess, categorical
from lcm.transition import MarkovTransition
from lcm.typing import (
    ContinuousAction,
    ContinuousState,
    DiscreteAction,
    FloatND,
    ScalarInt,
)
from tests.test_models.nbegm_common import (
    crra_utility,
    feasible,
    make_alive_dead_model,
    resolve_solver,
    savings,
    utility,
)

N_INCOME_NODES = 5
INCOME_SCALE = 0.5
LEISURE_UTILITY = 0.3


@categorical(ordered=False)
class BuyPrivate:
    no: ScalarInt
    yes: ScalarInt


@lcm.piecewise_affine(
    "tax",
    variable="liquid",
    breakpoints=(lcm.affine_breakpoint("tax_exemption", kind="continuous_kink"),),
)
def tax(liquid: ContinuousState, tax_rate: float, tax_exemption: float) -> FloatND:
    """Continuous tax: zero below the exemption, `tax_rate` on the excess above."""
    return tax_rate * jnp.maximum(liquid - tax_exemption, 0.0)


@lcm.piecewise_affine(
    "tax",
    variable="liquid",
    breakpoints=(lcm.affine_breakpoint("tax_exemption", kind="jump"),),
)
def tax_jump(liquid: ContinuousState, tax_lump: float, tax_exemption: float) -> FloatND:
    """Cliff tax: zero below the exemption, a flat lump above (an additive jump).

    A pure level drop in cash-on-hand at the exemption — the shape of an
    income-tested subsidy cliff. Paired with the discrete insurance choice and the
    income ride co-state, it is the jump-plus-discrete-plus-ride case the envelope
    resolves by taking the discrete max over each branch's published cliff limits.
    """
    return jnp.where(liquid >= tax_exemption, tax_lump, 0.0)


def derived_income(
    liquid: ContinuousState, buy_private: DiscreteAction, income_offset: float
) -> FloatND:
    """Monotone income the schedule sits on — the action shifts its intercept.

    Buying insurance moves the income the tax cliff is measured against, so the
    cliff's asset (liquid) preimage differs per `buy_private` branch — the case a
    single shared breakpoint partition cannot represent.
    """
    return liquid + income_offset * buy_private


@lcm.piecewise_affine(
    "tax",
    variable="derived_income",
    breakpoints=(lcm.affine_breakpoint("tax_exemption", kind="continuous_kink"),),
)
def tax_derived(
    derived_income: FloatND, tax_rate: float, tax_exemption: float
) -> FloatND:
    """Continuous tax on the derived income above the exemption."""
    return tax_rate * jnp.maximum(derived_income - tax_exemption, 0.0)


@lcm.piecewise_affine(
    "tax",
    variable="derived_income",
    breakpoints=(lcm.affine_breakpoint("tax_exemption", kind="jump"),),
)
def tax_derived_jump(
    derived_income: FloatND, tax_rate: float, tax_exemption: float
) -> FloatND:
    """Lump-sum tax jumping in at the exemption on the derived income.

    An action entering a *jumped* schedule variable moves the published cliff
    abscissa per branch — refused under the one-sided cliff read.
    """
    return jnp.where(derived_income >= tax_exemption, tax_rate * derived_income, 0.0)


def resources(
    liquid: ContinuousState,
    tax: FloatND,
    buy_private: DiscreteAction,
    base_income: float,
    premium: float,
) -> FloatND:
    """Cash-on-hand: liquid plus base income, net of tax and any premium."""
    return liquid + base_income - tax - premium * buy_private


def next_liquid(
    resources: FloatND,
    consumption: ContinuousAction,
    income: ContinuousState,
    return_liquid: float,
) -> ContinuousState:
    """Liquid law: saved cash earns the return, plus the realized income draw."""
    return (1.0 + return_liquid) * (resources - consumption) + INCOME_SCALE * jnp.exp(
        income
    )


def next_liquid_with_oop(
    resources: FloatND,
    consumption: ContinuousAction,
    income: ContinuousState,
    return_liquid: float,
    oop: FloatND,
) -> ContinuousState:
    """Liquid law with an off-budget out-of-pocket cost on next assets.

    The brute-force twin of `next_liquid_from_savings_with_oop` (`savings` is the
    post-decision `resources - consumption`), so both solvers model the same
    economics.
    """
    return (
        (1.0 + return_liquid) * (resources - consumption)
        + INCOME_SCALE * jnp.exp(income)
        - oop
    )


def next_liquid_from_savings(
    savings: FloatND,
    income: ContinuousState,
    return_liquid: float,
) -> ContinuousState:
    """Liquid law in post-decision form: saved cash earns the return, plus income."""
    return (1.0 + return_liquid) * savings + INCOME_SCALE * jnp.exp(income)


def next_streak(
    streak: ContinuousState, buy_private: DiscreteAction
) -> ContinuousState:
    """Coverage streak: grows while insured, resets otherwise (reads the action)."""
    return jnp.where(buy_private == BuyPrivate.yes, streak + 1.0, 0.0)


def next_tracker_smooth(
    tracker: ContinuousState, liquid: ContinuousState
) -> ContinuousState:
    """Co-state that varies smoothly (affine) in the current liquid state.

    Reads the liquid (Euler) state with a nonzero derivative between breakpoints, so
    the midpoint-bound continuation row is wrong within each interval — the
    misdeclaration the interval-constant continuation guard rejects.
    """
    return tracker + 0.1 * liquid


def next_tracker_step(
    tracker: ContinuousState, liquid: ContinuousState
) -> ContinuousState:
    """Co-state switched at a liquid threshold — piecewise-constant in liquid.

    Reads the liquid state only through a level switch, so its derivative between
    breakpoints is zero and the midpoint-bound continuation row is exact.
    """
    return jnp.where(liquid >= 10.0, tracker + 1.0, 0.0)


def next_tracker_unprobeable(
    tracker: ContinuousState, liquid: ContinuousState
) -> ContinuousState:
    """Co-state law the build-time constancy probe cannot differentiate.

    The Python-level branch on the liquid value raises under `jax.jacfwd`'s
    tracer, so the probe cannot establish piecewise-constancy — the build must
    refuse rather than assume the law is interval-constant.
    """
    if float(liquid) >= 10.0:
        return tracker + 1.0
    return tracker


def oop(buy_private: DiscreteAction, oop_uninsured: float) -> FloatND:
    """Out-of-pocket medical cost: zero when insured, a fixed hit when not.

    An action-dependent shift of *next* liquid that does not run through the
    current budget — the channel that makes a shared next-period continuation
    wrong, since each `buy_private` branch then lands on different next assets.
    """
    return jnp.where(buy_private == BuyPrivate.yes, 0.0, oop_uninsured)


def next_liquid_from_savings_with_oop(
    savings: FloatND,
    income: ContinuousState,
    return_liquid: float,
    oop: FloatND,
) -> ContinuousState:
    """Post-decision liquid law that also nets an action-dependent OOP cost."""
    return (1.0 + return_liquid) * savings + INCOME_SCALE * jnp.exp(income) - oop


def prob_stay_alive_action(
    age: int, final_age_alive: float, buy_private: DiscreteAction
) -> FloatND:
    """Survival probability that rises when insured (reads the action).

    The regime transition depends on the discrete action, so each `buy_private`
    branch weights the alive-vs-dead continuation differently — the case a single
    shared continuation cannot represent.
    """
    return jnp.where(age + 1 < final_age_alive, 0.85 + 0.1 * buy_private, 0.0)


def prob_die_action(
    age: int, final_age_alive: float, buy_private: DiscreteAction
) -> FloatND:
    """Death probability complementary to the action-dependent survival."""
    return 1.0 - prob_stay_alive_action(age, final_age_alive, buy_private)


def prob_stay_alive_liquid(
    age: int, final_age_alive: float, liquid: ContinuousState
) -> FloatND:
    """Survival probability switched at the declared tax-exemption cliff.

    The regime transition reads the current liquid state through a level switch at
    the declared breakpoint, so the alive-vs-dead continuation blend is constant
    within each declared interval but differs across intervals — the case that
    requires interval-specific continuation rows.
    """
    survives = 0.80 + 0.1 * (liquid >= 12.0)
    return jnp.where(age + 1 < final_age_alive, survives, 0.0)


def prob_die_liquid(
    age: int, final_age_alive: float, liquid: ContinuousState
) -> FloatND:
    """Death probability complementary to the liquid-dependent survival."""
    return 1.0 - prob_stay_alive_liquid(age, final_age_alive, liquid)


def prob_stay_alive_liquid_smooth(
    age: int, final_age_alive: float, liquid: ContinuousState
) -> FloatND:
    """Survival probability varying smoothly in the liquid state.

    A nonzero liquid-derivative between breakpoints makes the midpoint-bound
    continuation row wrong within an interval — the misdeclaration the
    interval-constant continuation guard rejects.
    """
    survives = jnp.clip(0.6 + 0.01 * liquid, 0.0, 1.0)
    return jnp.where(age + 1 < final_age_alive, survives, 0.0)


def prob_die_liquid_smooth(
    age: int, final_age_alive: float, liquid: ContinuousState
) -> FloatND:
    """Death probability complementary to the smooth liquid-dependent survival."""
    return 1.0 - prob_stay_alive_liquid_smooth(age, final_age_alive, liquid)


def discount_factor_action(buy_private: DiscreteAction) -> FloatND:
    """Discount factor reading the discrete action — an unsupported branch channel.

    The envelope evaluates the discount factor once per cell, not per branch, so an
    action-dependent discount factor is rejected at model build.
    """
    return 0.90 + 0.05 * buy_private


def utility_with_action(
    consumption: ContinuousAction, crra: float, buy_private: DiscreteAction
) -> FloatND:
    """CRRA consumption utility plus a per-branch leisure term reading the action.

    The action enters period utility directly (a leisure/effort-like shift), so each
    `buy_private` branch has a different utility level and marginal — the case a
    single shared per-cell utility cannot represent.
    """
    return crra_utility(consumption, crra) + LEISURE_UTILITY * buy_private


def build_model(  # noqa: C901, PLR0912
    *,
    variant: str = "brute",
    n_periods: int = 4,
    n_liquid: int = 120,
    n_consumption: int = 150,
    liquid_max: float = 30.0,
    n_savings: int = 150,
    savings_max: float = 28.0,
    action_in_costate: bool = False,
    action_in_liquid_law: bool = False,
    action_in_utility: bool = False,
    action_in_regime_transition: bool = False,
    action_in_schedule_variable: bool = False,
    jump_schedule: bool = False,
    costate_reads_liquid: bool = False,
    costate_smooth: bool = False,
    costate_unprobeable: bool = False,
    transition_reads_liquid: bool = False,
    transition_smooth: bool = False,
    action_in_discount: bool = False,
    branch_batch_size: int = 0,
    probe_failure: str = "reject",
) -> Model:
    """Create the (alive, dead) ride-along toy with a discrete insurance choice.

    With `action_in_costate`, a `streak` co-state carries a law of motion that
    reads `buy_private` — so the discrete action shifts the continuation, not just
    the current budget, which the shared-continuation envelope cannot represent.

    With `jump_schedule`, the budget carries a jump cliff (not a kink), so the
    discrete envelope must take the discrete choice over each branch's published
    one-sided cliff limits — the jump-plus-discrete-plus-ride composition.

    With `costate_reads_liquid`, a `tracker` co-state carries a law of motion that
    reads the current liquid state — piecewise-constant by default (a level switched
    at a threshold), or smoothly varying with `costate_smooth`. NBEGM binds the
    liquid state to each interval's node for such a law; the smooth variant is the
    misdeclaration the interval-constant continuation guard rejects.

    With `transition_reads_liquid`, the survival probability reads the current
    liquid state through a level switch at the declared cliff, so the alive-vs-dead
    continuation blend differs across declared intervals — interval-specific
    continuation rows are required even though no carried-state law reads liquid.
    """
    income_grid = NormalIIDProcess(n_points=N_INCOME_NODES, gauss_hermite=True)
    tax_func = tax_jump if jump_schedule else tax
    utility_func = utility_with_action if action_in_utility else utility
    alive_functions = {"utility": utility_func, "tax": tax_func, "resources": resources}
    if action_in_discount:
        alive_functions = {
            **alive_functions,
            "discount_factor": discount_factor_action,
        }
    if action_in_schedule_variable:
        # The tax cliff sits on a derived income that reads the action, so its asset
        # preimage differs per branch — each branch gets its own breakpoint partition
        # (a jump cliff on the derived income when `jump_schedule` is also set).
        alive_functions = {
            **alive_functions,
            "tax": tax_derived_jump if jump_schedule else tax_derived,
            "derived_income": derived_income,
        }
    extra_states: dict[str, Grid] = {"income": income_grid}
    extra_state_transitions: dict[str, object] = {}
    if action_in_costate:
        extra_states["streak"] = LinSpacedGrid(start=0.0, stop=4.0, n_points=5)
        extra_state_transitions["streak"] = {
            "alive": next_streak,
            "dead": next_streak,
        }
    if costate_reads_liquid:
        if costate_unprobeable:
            tracker_law = next_tracker_unprobeable
        else:
            tracker_law = next_tracker_smooth if costate_smooth else next_tracker_step
        extra_states["tracker"] = LinSpacedGrid(start=0.0, stop=4.0, n_points=5)
        extra_state_transitions["tracker"] = {
            "alive": tracker_law,
            "dead": tracker_law,
        }
    solver_kwargs: dict[str, object] = {}
    if probe_failure != "reject":
        solver_kwargs["probe_failure"] = probe_failure
    if branch_batch_size:
        solver_kwargs["branch_batch_size"] = branch_batch_size
    alive_solver = resolve_solver(
        variant,
        savings_grid=LinSpacedGrid(start=0.0, stop=savings_max, n_points=n_savings),
        continuous_state="liquid",
        post_decision_function="savings",
        **solver_kwargs,
    )
    if variant == "nbegm":
        alive_functions = {**alive_functions, "savings": savings}
        if action_in_liquid_law:
            alive_functions = {**alive_functions, "oop": oop}
            liquid_law = next_liquid_from_savings_with_oop
        else:
            liquid_law = next_liquid_from_savings
        constraints = {}
    elif action_in_liquid_law:
        alive_functions = {**alive_functions, "oop": oop}
        liquid_law = next_liquid_with_oop
        constraints = {"feasible": feasible}
    else:
        liquid_law = next_liquid
        constraints = {"feasible": feasible}

    if action_in_regime_transition:
        survival_transition = {
            "alive": MarkovTransition(prob_stay_alive_action),
            "dead": MarkovTransition(prob_die_action),
        }
    elif transition_reads_liquid:
        stay, die = (
            (prob_stay_alive_liquid_smooth, prob_die_liquid_smooth)
            if transition_smooth
            else (prob_stay_alive_liquid, prob_die_liquid)
        )
        survival_transition = {
            "alive": MarkovTransition(stay),
            "dead": MarkovTransition(die),
        }
    else:
        survival_transition = None
    return make_alive_dead_model(
        n_periods=n_periods,
        n_liquid=n_liquid,
        liquid_max=liquid_max,
        n_consumption=n_consumption,
        alive_functions=alive_functions,
        liquid_law=liquid_law,
        alive_solver=alive_solver,
        constraints=constraints,
        extra_states=extra_states,
        extra_state_transitions=extra_state_transitions,
        extra_actions={"buy_private": DiscreteGrid(BuyPrivate)},
        survival_transition=survival_transition,
    )


def build_params(
    *,
    discount_factor: float = 0.95,
    crra: float = 2.0,
    return_liquid: float = 0.03,
    base_income: float = 3.0,
    premium: float = 1.5,
    tax_rate: float = 0.2,
    tax_exemption: float = 12.0,
    final_age_alive: float = 3.0,
    jump_schedule: bool = False,
    tax_lump: float = 2.0,
    action_in_liquid_law: bool = False,
    oop_uninsured: float = 1.0,
    action_in_schedule_variable: bool = False,
    income_offset: float = 4.0,
) -> dict:
    """Get parameters for the ride-along discrete-choice toy."""
    alive_budget = {"return_liquid": return_liquid}
    tax_params = (
        {"tax_lump": tax_lump, "tax_exemption": tax_exemption}
        if jump_schedule
        else {"tax_rate": tax_rate, "tax_exemption": tax_exemption}
    )
    derived_income_params = (
        {"derived_income": {"income_offset": income_offset}}
        if action_in_schedule_variable
        else {}
    )
    oop_params = (
        {"oop": {"oop_uninsured": oop_uninsured}} if action_in_liquid_law else {}
    )
    return {
        "alive": {
            "utility": {"crra": crra},
            "H": {"discount_factor": discount_factor},
            "resources": {"base_income": base_income, "premium": premium},
            "income": {"mu": 0.0, "sigma": 1.0},
            "tax": tax_params,
            **oop_params,
            **derived_income_params,
            "alive": {
                "next_liquid": alive_budget,
                "next_regime": {"final_age_alive": final_age_alive},
            },
            "dead": {
                "next_liquid": alive_budget,
                "next_regime": {"final_age_alive": final_age_alive},
            },
        },
        "dead": {"utility": {"crra": crra}},
    }
