"""Ride-along toy with a binary discrete choice over a kinked budget.

A stochastic income node rides along a kink tax schedule (the continuation
integrates over the child income nodes), and each period the agent also chooses
whether to buy private insurance — a premium that lowers cash-on-hand. BQSEGM
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
from lcm.typing import (
    ContinuousAction,
    ContinuousState,
    DiscreteAction,
    FloatND,
    ScalarInt,
)
from tests.test_models.bqsegm_common import (
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


def coh(
    liquid: ContinuousState,
    tax: FloatND,
    buy_private: DiscreteAction,
    base_income: float,
    premium: float,
) -> FloatND:
    """Cash-on-hand: liquid plus base income, net of tax and any premium."""
    return liquid + base_income - tax - premium * buy_private


def next_liquid(
    coh: FloatND,
    consumption: ContinuousAction,
    income: ContinuousState,
    return_liquid: float,
) -> ContinuousState:
    """Liquid law: saved cash earns the return, plus the realized income draw."""
    return (1.0 + return_liquid) * (coh - consumption) + INCOME_SCALE * jnp.exp(income)


def next_liquid_with_oop(
    coh: FloatND,
    consumption: ContinuousAction,
    income: ContinuousState,
    return_liquid: float,
    oop: FloatND,
) -> ContinuousState:
    """Liquid law with an off-budget out-of-pocket cost on next assets.

    The brute-force twin of `next_liquid_from_savings_with_oop` (`savings` is the
    post-decision `coh - consumption`), so both solvers model the same economics.
    """
    return (
        (1.0 + return_liquid) * (coh - consumption)
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


def utility_with_action(
    consumption: ContinuousAction, crra: float, buy_private: DiscreteAction
) -> FloatND:
    """CRRA consumption utility plus a per-branch leisure term reading the action.

    The action enters period utility directly (a leisure/effort-like shift), so each
    `buy_private` branch has a different utility level and marginal — the case a
    single shared per-cell utility cannot represent.
    """
    return crra_utility(consumption, crra) + LEISURE_UTILITY * buy_private


def build_model(
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
    jump_schedule: bool = False,
    costate_reads_liquid: bool = False,
    costate_smooth: bool = False,
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
    at a threshold), or smoothly varying with `costate_smooth`. BQSEGM binds the
    liquid state to each interval's node for such a law; the smooth variant is the
    misdeclaration the interval-constant continuation guard rejects.
    """
    income_grid = NormalIIDProcess(n_points=N_INCOME_NODES, gauss_hermite=True)
    tax_func = tax_jump if jump_schedule else tax
    utility_func = utility_with_action if action_in_utility else utility
    alive_functions = {"utility": utility_func, "tax": tax_func, "coh": coh}
    extra_states: dict[str, Grid] = {"income": income_grid}
    extra_state_transitions: dict[str, object] = {}
    if action_in_costate:
        extra_states["streak"] = LinSpacedGrid(start=0.0, stop=4.0, n_points=5)
        extra_state_transitions["streak"] = {
            "alive": next_streak,
            "dead": next_streak,
        }
    if costate_reads_liquid:
        tracker_law = next_tracker_smooth if costate_smooth else next_tracker_step
        extra_states["tracker"] = LinSpacedGrid(start=0.0, stop=4.0, n_points=5)
        extra_state_transitions["tracker"] = {
            "alive": tracker_law,
            "dead": tracker_law,
        }
    alive_solver = resolve_solver(
        variant,
        savings_grid=LinSpacedGrid(start=0.0, stop=savings_max, n_points=n_savings),
        continuous_state="liquid",
        post_decision_function="savings",
    )
    if variant == "bqsegm":
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
) -> dict:
    """Get parameters for the ride-along discrete-choice toy."""
    alive_budget = {"return_liquid": return_liquid}
    tax_params = (
        {"tax_lump": tax_lump, "tax_exemption": tax_exemption}
        if jump_schedule
        else {"tax_rate": tax_rate, "tax_exemption": tax_exemption}
    )
    oop_params = (
        {"oop": {"oop_uninsured": oop_uninsured}} if action_in_liquid_law else {}
    )
    return {
        "alive": {
            "utility": {"crra": crra},
            "H": {"discount_factor": discount_factor},
            "coh": {"base_income": base_income, "premium": premium},
            "income": {"mu": 0.0, "sigma": 1.0},
            "tax": tax_params,
            **oop_params,
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
