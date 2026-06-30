"""One-asset BQSEGM toy with a subsidy cliff on derived income + a continuous co-state.

Mirrors the M1 ACA structure in miniature: the Euler/liquid state `liquid` carries
the consumption-saving margin, while a *continuous* co-state `wage` rides along. The
subsidy cliff is declared on the derived monotone quantity `gross_income = liquid +
wage`, so the cliff `gross_income == fpl_cliff` maps to the asset preimage `liquid =
fpl_cliff - wage`, a *different* liquid point in every `wage` cell. The BQSEGM
ride-along solver names `liquid` as its `continuous_state` (the Euler axis) and treats
`wage` as a ride-along axis the continuation reader integrates, recovering the per-cell
asset preimage of the cliff and solving the jumped budget against the savings-space
continuation. The brute variant (`GridSearch`) productmaps over
`(liquid, wage, consumption)` and is the dense agreement oracle.
"""

import jax.numpy as jnp

import lcm
from lcm import (
    AgeGrid,
    LinSpacedGrid,
    MarkovTransition,
    Model,
    categorical,
)
from lcm.regime import Regime
from lcm.solvers import GridSearch
from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    FloatND,
    ScalarInt,
)


@categorical(ordered=False)
class RegimeId:
    alive: ScalarInt
    dead: ScalarInt


def _crra(consumption: FloatND, crra: float) -> FloatND:
    return jnp.where(
        crra == 1.0,
        jnp.log(consumption),
        consumption ** (1.0 - crra) / (1.0 - crra),
    )


def utility(consumption: ContinuousAction, crra: float) -> FloatND:
    """CRRA consumption utility."""
    return _crra(consumption, crra)


def bequest(liquid: ContinuousState, crra: float) -> FloatND:
    """Terminal value: consume remaining liquid wealth."""
    return _crra(liquid, crra)


def gross_income(liquid: ContinuousState, wage: ContinuousState) -> FloatND:
    """Pre-tax income: liquid wealth plus the wage co-state (monotone in liquid)."""
    return liquid + wage


@lcm.piecewise_affine(
    "subsidy",
    variable="gross_income",
    breakpoints=(lcm.affine_breakpoint("fpl_cliff", kind="jump"),),
)
def subsidy(
    gross_income: FloatND, subsidy_low: float, subsidy_high: float, fpl_cliff: float
) -> FloatND:
    """Lump-sum subsidy: the higher amount below the income cliff, lower above."""
    return jnp.where(gross_income < fpl_cliff, subsidy_high, subsidy_low)


def coh(gross_income: FloatND, subsidy: FloatND) -> FloatND:
    """Cash-on-hand: pre-tax income plus the cliff-contingent subsidy."""
    return gross_income + subsidy


def next_liquid(
    coh: FloatND,
    consumption: ContinuousAction,
    return_liquid: float,
    income: float,
) -> ContinuousState:
    """Liquid law of motion: saved cash earns the liquid return, plus income."""
    return (1.0 + return_liquid) * (coh - consumption) + income


def savings(coh: FloatND, consumption: ContinuousAction) -> FloatND:
    """Post-decision savings: the cash-on-hand not consumed."""
    return coh - consumption


def next_liquid_from_savings(
    savings: FloatND,
    return_liquid: float,
    income: float,
) -> ContinuousState:
    """Liquid law in post-decision form: saved cash earns the return, plus income."""
    return (1.0 + return_liquid) * savings + income


def next_wage(wage: ContinuousState, wage_persistence: float) -> ContinuousState:
    """Wage co-state law: a deterministic decay that never reads the savings slot."""
    return wage_persistence * wage


def feasible(coh: FloatND, consumption: ContinuousAction) -> BoolND:
    """Borrowing constraint: consumption cannot exceed cash-on-hand."""
    return consumption <= coh


def prob_stay_alive(age: int, final_age_alive: float) -> FloatND:
    """Deterministic (0/1) probability of staying alive next period."""
    return jnp.where(age + 1 < final_age_alive, 1.0, 0.0)


def prob_die(age: int, final_age_alive: float) -> FloatND:
    """Deterministic (0/1) probability of dying next period."""
    return jnp.where(age + 1 >= final_age_alive, 1.0, 0.0)


def build_model(
    *,
    variant: str = "brute",
    n_periods: int = 4,
    n_liquid: int = 120,
    n_wage: int = 6,
    n_consumption: int = 160,
    liquid_max: float = 30.0,
    wage_max: float = 6.0,
    n_savings: int = 200,
    savings_max: float = 28.0,
) -> Model:
    """Create the (alive, dead) toy whose subsidy cliff lives on derived income.

    Args:
        variant: `"brute"` drives the alive regime by `GridSearch`; `"bqsegm"` by
            the `BQSEGM` schedule solver, which names `liquid` as its Euler axis and
            rides along the continuous `wage` co-state.
        n_periods: Number of lifecycle periods (the last is terminal).
        n_liquid: Liquid (Euler) state grid size.
        n_wage: Wage co-state grid size.
        n_consumption: Consumption-action grid size (brute only).
        liquid_max: Upper bound of the liquid grid.
        wage_max: Upper bound of the wage grid.
        n_savings: Post-decision savings grid size (BQSEGM only).
        savings_max: Upper bound of the savings grid (BQSEGM only).

    Returns:
        The assembled `Model`.

    """
    ages = AgeGrid(start=0, stop=n_periods - 1, step="Y")
    final_age = ages.exact_values[-1]
    liquid_grid = LinSpacedGrid(start=0.1, stop=liquid_max, n_points=n_liquid)
    wage_grid = LinSpacedGrid(start=0.5, stop=wage_max, n_points=n_wage)

    alive_functions = {
        "utility": utility,
        "gross_income": gross_income,
        "subsidy": subsidy,
        "coh": coh,
    }
    if variant == "brute":
        alive_solver = GridSearch()
        liquid_law = next_liquid
        constraints = {"feasible": feasible}
    elif variant == "bqsegm":
        from lcm.solvers import BQSEGM  # noqa: PLC0415

        alive_functions = {**alive_functions, "savings": savings}
        liquid_law = next_liquid_from_savings
        constraints = {}
        alive_solver = BQSEGM(
            savings_grid=LinSpacedGrid(start=0.0, stop=savings_max, n_points=n_savings),
            continuous_state="liquid",
            post_decision_function="savings",
        )
    else:
        msg = f"unknown variant {variant!r}; use 'brute' or 'bqsegm'."
        raise ValueError(msg)

    alive = Regime(
        actions={
            "consumption": LinSpacedGrid(
                start=0.1, stop=liquid_max, n_points=n_consumption
            )
        },
        states={"liquid": liquid_grid, "wage": wage_grid},
        state_transitions={
            "liquid": {"alive": liquid_law, "dead": liquid_law},
            "wage": {"alive": next_wage},
        },
        constraints=constraints,
        transition={
            "alive": MarkovTransition(prob_stay_alive),
            "dead": MarkovTransition(prob_die),
        },
        functions=alive_functions,
        active=lambda age, fa=final_age: age < fa,
        solver=alive_solver,
    )
    dead = Regime(
        transition=None,
        states={"liquid": liquid_grid},
        functions={"utility": bequest},
        active=lambda age, fa=final_age: age >= fa,
        solver=GridSearch(),
    )
    return Model(
        regimes={"alive": alive, "dead": dead},
        ages=ages,
        regime_id_class=RegimeId,
    )


def build_params(
    *,
    discount_factor: float = 0.95,
    crra: float = 2.0,
    return_liquid: float = 0.03,
    income: float = 1.0,
    wage_persistence: float = 0.9,
    subsidy_low: float = 0.0,
    subsidy_high: float = 3.0,
    fpl_cliff: float = 15.0,
    final_age_alive: float = 3.0,
) -> dict:
    """Get parameters for the derived-income subsidy-cliff toy.

    The cliff at `gross_income == fpl_cliff` maps to the asset preimage `liquid =
    fpl_cliff - wage`, distinct in every `wage` slice and grid-interior for the
    default wage grid. Cash-on-hand drops by `subsidy_high - subsidy_low` as liquid
    crosses each slice's preimage.
    """
    alive_budget = {"return_liquid": return_liquid, "income": income}
    return {
        "alive": {
            "utility": {"crra": crra},
            "H": {"discount_factor": discount_factor},
            "subsidy": {
                "subsidy_low": subsidy_low,
                "subsidy_high": subsidy_high,
                "fpl_cliff": fpl_cliff,
            },
            "alive": {
                "next_liquid": alive_budget,
                "next_wage": {"wage_persistence": wage_persistence},
                "next_regime": {"final_age_alive": final_age_alive},
            },
            "dead": {
                "next_liquid": alive_budget,
                "next_regime": {"final_age_alive": final_age_alive},
            },
        },
        "dead": {"utility": {"crra": crra}},
    }
