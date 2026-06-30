"""BQSEGM regime whose continuation reads a separate brute-solved living child.

The M1 ACA slice solves one regime by `BQSEGM` while every other living regime
stays on brute force, and the BQSEGM regime transitions *into* those brute
children. This toy isolates that topology: a ride-along `young` regime (solved by
`BQSEGM` or, in the reference, by `GridSearch`) transitions deterministically to a
distinct `old` regime that is always brute-solved, which in turn transitions to
the terminal `dead` regime. The BQSEGM `young` solve must read `old`'s brute value
array as its continuation — there is no EGM carry for `old` — and reproduce the
all-brute reference value in every `kind` slice.
"""

import jax.numpy as jnp

import lcm
from lcm import (
    AgeGrid,
    DiscreteGrid,
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
    DiscreteState,
    FloatND,
    ScalarInt,
)


@categorical(ordered=False)
class RegimeId:
    young: ScalarInt
    old: ScalarInt
    dead: ScalarInt


@categorical(ordered=False)
class ConsumerKind:
    lo: ScalarInt
    hi: ScalarInt


@categorical(ordered=False)
class WorkChoice:
    home: ScalarInt
    work: ScalarInt


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


@lcm.piecewise_affine(
    "tax",
    variable="liquid",
    breakpoints=(lcm.affine_breakpoint("tax_exemption", kind="continuous_kink"),),
)
def tax(liquid: ContinuousState, tax_rate: float, tax_exemption: float) -> FloatND:
    """Continuous tax: zero below the exemption, `tax_rate` on the excess above."""
    return tax_rate * jnp.maximum(liquid - tax_exemption, 0.0)


def coh(
    liquid: ContinuousState,
    kind: DiscreteState,
    tax: FloatND,
    base_income: FloatND,
) -> FloatND:
    """Cash-on-hand: liquid plus the kind's base income, net of the tax."""
    return liquid + base_income[kind] - tax


def coh_with_work(
    liquid: ContinuousState,
    kind: DiscreteState,
    tax: FloatND,
    base_income: FloatND,
    work: DiscreteState,
    wage: float,
) -> FloatND:
    """Cash-on-hand with a discrete work choice adding `work * wage` earnings."""
    return liquid + base_income[kind] + work * wage - tax


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


def feasible(coh: FloatND, consumption: ContinuousAction) -> BoolND:
    """Borrowing constraint: consumption cannot exceed cash-on-hand."""
    return consumption <= coh


def prob_to_old(age: int) -> FloatND:
    """Deterministic probability `young` survives into `old` next period."""
    return jnp.where(age + 1 < 2.0, 1.0, 0.0)


def prob_young_dead(age: int) -> FloatND:
    """Deterministic probability `young` dies next period."""
    return jnp.where(age + 1 >= 2.0, 1.0, 0.0)


def build_model(
    *,
    young_variant: str = "brute",
    old_discrete_action: bool = False,
    n_liquid: int = 120,
    n_consumption: int = 150,
    liquid_max: float = 30.0,
    n_savings: int = 150,
    savings_max: float = 28.0,
) -> Model:
    """Create the `young`→`old`→`dead` toy.

    Args:
        young_variant: `"brute"` drives `young` by `GridSearch` (the reference);
            `"bqsegm"` by the `BQSEGM` schedule solver, whose continuation must
            read `old`'s brute value array. `old` is always brute-solved.
        n_liquid: Liquid-state grid size.
        n_consumption: Consumption-action grid size (brute path).
        liquid_max: Upper bound of the liquid grid.
        n_savings: Post-decision savings grid size (BQSEGM path).
        savings_max: Upper bound of the savings grid (BQSEGM path).

    Returns:
        The assembled `Model`.

    """
    ages = AgeGrid(start=0, stop=2, step="Y")
    liquid_grid = LinSpacedGrid(start=0.1, stop=liquid_max, n_points=n_liquid)
    consumption_grid = LinSpacedGrid(start=0.1, stop=liquid_max, n_points=n_consumption)
    kind_grid = DiscreteGrid(ConsumerKind)

    young_functions = {"utility": utility, "tax": tax, "coh": coh}
    if young_variant == "brute":
        young_solver = GridSearch()
        young_liquid_law = next_liquid
        young_constraints = {"feasible": feasible}
    elif young_variant == "bqsegm":
        from lcm.solvers import BQSEGM  # noqa: PLC0415

        young_functions = {**young_functions, "savings": savings}
        young_liquid_law = next_liquid_from_savings
        young_constraints = {}
        young_solver = BQSEGM(
            savings_grid=LinSpacedGrid(start=0.0, stop=savings_max, n_points=n_savings),
            post_decision_function="savings",
        )
    else:
        msg = f"unknown young_variant {young_variant!r}; use 'brute' or 'bqsegm'."
        raise ValueError(msg)

    young = Regime(
        actions={"consumption": consumption_grid},
        states={"liquid": liquid_grid, "kind": kind_grid},
        state_transitions={
            "liquid": {"old": young_liquid_law, "dead": young_liquid_law},
            "kind": {"old": lcm.fixed_transition("kind")},
        },
        constraints=young_constraints,
        transition={
            "old": MarkovTransition(prob_to_old),
            "dead": MarkovTransition(prob_young_dead),
        },
        functions=young_functions,
        active=lambda age: age < 1,
        solver=young_solver,
    )
    old_actions = {"consumption": consumption_grid}
    old_functions = {"utility": utility, "tax": tax, "coh": coh}
    if old_discrete_action:
        old_actions = {"work": DiscreteGrid(WorkChoice), **old_actions}
        old_functions = {**old_functions, "coh": coh_with_work}
    old = Regime(
        actions=old_actions,
        states={"liquid": liquid_grid, "kind": kind_grid},
        state_transitions={
            "liquid": {"dead": next_liquid},
            "kind": {"dead": lcm.fixed_transition("kind")},
        },
        constraints={"feasible": feasible},
        transition={"dead": MarkovTransition(lambda: jnp.array(1.0))},
        functions=old_functions,
        active=lambda age: age == 1,
        solver=GridSearch(),
    )
    dead = Regime(
        transition=None,
        states={"liquid": liquid_grid},
        functions={"utility": bequest},
        active=lambda age: age >= 2,
        solver=GridSearch(),
    )
    return Model(
        regimes={"young": young, "old": old, "dead": dead},
        ages=ages,
        regime_id_class=RegimeId,
    )


def build_params(
    *,
    discount_factor: float = 0.95,
    crra: float = 2.0,
    return_liquid: float = 0.03,
    income: float = 1.0,
    base_income_lo: float = 1.0,
    base_income_hi: float = 4.0,
    tax_rate: float = 0.3,
    tax_exemption: float = 12.0,
    old_discrete_action: bool = False,
    wage: float = 3.0,
) -> dict:
    """Get parameters for the young→old→dead toy.

    `base_income` is a length-2 array indexed by the `kind` code (`lo`, `hi`), so
    the budget differs across the ride-along slices in both `young` and `old`.
    With `old_discrete_action`, `old` gains a discrete `work` choice whose
    earnings are `work * wage`, so its value array is maxed over `work` too.
    """
    base_income = jnp.array([base_income_lo, base_income_hi])
    budget = {"return_liquid": return_liquid, "income": income}
    tax_params = {"tax_rate": tax_rate, "tax_exemption": tax_exemption}
    old_coh = {"base_income": base_income}
    if old_discrete_action:
        old_coh = {"base_income": base_income, "wage": wage}
    return {
        "young": {
            "utility": {"crra": crra},
            "H": {"discount_factor": discount_factor},
            "tax": tax_params,
            "coh": {"base_income": base_income},
            "old": {"next_liquid": budget},
            "dead": {"next_liquid": budget},
        },
        "old": {
            "utility": {"crra": crra},
            "H": {"discount_factor": discount_factor},
            "tax": tax_params,
            "coh": old_coh,
            "dead": {"next_liquid": budget},
        },
        "dead": {"utility": {"crra": crra}},
    }
