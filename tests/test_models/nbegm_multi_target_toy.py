"""NBEGM ride-along toy with a stochastic multi-target lifecycle transition.

Two living regimes `alive_a` and `alive_b` share the same structure (a 1-D liquid
Euler state, a ride-along `kind` co-state, and a continuous tax-bracket budget)
but carry different `base_income`, so their value functions genuinely differ.
Each period a living agent transitions stochastically to `alive_a`, `alive_b`, or
`dead`. While both living regimes are active next period the continuation is a
probability-weighted blend of their two value functions — a single-target
continuation read would be wrong. The brute variant (`GridSearch`) solves the
same model and is the dense agreement oracle.
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
    ContinuousState,
    DiscreteState,
    FloatND,
    ScalarInt,
)
from tests.test_models.nbegm_common import (
    bequest,
    feasible,
    next_liquid,
    next_liquid_from_savings,
    resolve_solver,
    savings,
    utility,
)


@categorical(ordered=False)
class RegimeId:
    alive_a: ScalarInt
    alive_b: ScalarInt
    dead: ScalarInt


@categorical(ordered=False)
class ConsumerKind:
    lo: ScalarInt
    hi: ScalarInt


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


def prob_to_alive_a(age: int, final_age_alive: float) -> FloatND:
    """Transition probability into `alive_a` (zero in the last living period)."""
    return jnp.where(age + 1 < final_age_alive, 0.6, 0.0)


def prob_to_alive_b(age: int, final_age_alive: float) -> FloatND:
    """Transition probability into `alive_b` (zero in the last living period)."""
    return jnp.where(age + 1 < final_age_alive, 0.4, 0.0)


def prob_to_dead(age: int, final_age_alive: float) -> FloatND:
    """Transition probability into `dead` (one in the last living period)."""
    return jnp.where(age + 1 >= final_age_alive, 1.0, 0.0)


def _build_living_regime(
    *,
    variant: str,
    liquid_grid: LinSpacedGrid,
    n_consumption: int,
    liquid_max: float,
    n_savings: int,
    savings_max: float,
    final_age: float,
) -> Regime:
    """Assemble one living regime transitioning to both living regimes and dead."""
    functions = {"utility": utility, "tax": tax, "coh": coh}
    solver = resolve_solver(
        variant,
        savings_grid=LinSpacedGrid(start=0.0, stop=savings_max, n_points=n_savings),
        post_decision_function="savings",
    )
    if variant == "nbegm":
        functions = {**functions, "savings": savings}
        liquid_law = next_liquid_from_savings
        constraints = {}
    else:
        liquid_law = next_liquid
        constraints = {"feasible": feasible}

    return Regime(
        actions={
            "consumption": LinSpacedGrid(
                start=0.1, stop=liquid_max, n_points=n_consumption
            )
        },
        states={"liquid": liquid_grid, "kind": DiscreteGrid(ConsumerKind)},
        state_transitions={
            "liquid": {
                "alive_a": liquid_law,
                "alive_b": liquid_law,
                "dead": liquid_law,
            },
            "kind": {
                "alive_a": lcm.fixed_transition("kind"),
                "alive_b": lcm.fixed_transition("kind"),
            },
        },
        constraints=constraints,
        transition={
            "alive_a": MarkovTransition(prob_to_alive_a),
            "alive_b": MarkovTransition(prob_to_alive_b),
            "dead": MarkovTransition(prob_to_dead),
        },
        functions=functions,
        active=lambda age, fa=final_age: age < fa,
        solver=solver,
    )


def build_model(
    *,
    variant: str = "brute",
    n_periods: int = 4,
    n_liquid: int = 110,
    n_consumption: int = 140,
    liquid_max: float = 30.0,
    n_savings: int = 140,
    savings_max: float = 28.0,
) -> Model:
    """Create the (alive_a, alive_b, dead) toy with a stochastic multi-target law.

    Args:
        variant: `"brute"` drives both living regimes by `GridSearch`; `"nbegm"`
            by the `NBEGM` schedule solver, which must take the continuation as a
            probability-weighted blend over both living targets.
        n_periods: Number of lifecycle periods (the last is terminal).
        n_liquid: Liquid-state grid size.
        n_consumption: Consumption-action grid size (brute only).
        liquid_max: Upper bound of the liquid grid.
        n_savings: Post-decision savings grid size (NBEGM only).
        savings_max: Upper bound of the savings grid (NBEGM only).

    Returns:
        The assembled `Model`.

    """
    ages = AgeGrid(start=0, stop=n_periods - 1, step="Y")
    final_age = float(ages.exact_values[-1])
    liquid_grid = LinSpacedGrid(start=0.1, stop=liquid_max, n_points=n_liquid)

    def make() -> Regime:
        return _build_living_regime(
            variant=variant,
            liquid_grid=liquid_grid,
            n_consumption=n_consumption,
            liquid_max=liquid_max,
            n_savings=n_savings,
            savings_max=savings_max,
            final_age=final_age,
        )

    dead = Regime(
        transition=None,
        states={"liquid": liquid_grid},
        functions={"utility": bequest},
        active=lambda age, fa=final_age: age >= fa,
        solver=GridSearch(),
    )
    return Model(
        regimes={"alive_a": make(), "alive_b": make(), "dead": dead},
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
    base_income_shift_b: float = 2.0,
    tax_rate: float = 0.3,
    tax_exemption: float = 12.0,
    final_age_alive: float = 3.0,
) -> dict:
    """Get parameters for the multi-target tax toy.

    `alive_b` carries a higher `base_income` (shifted by `base_income_shift_b`)
    than `alive_a`, so the two living regimes have genuinely different value
    functions and the multi-target continuation is a non-trivial blend.
    """
    base_a = jnp.array([base_income_lo, base_income_hi])
    base_b = base_a + base_income_shift_b
    budget = {"return_liquid": return_liquid, "income": income}
    regime_age = {"next_regime": {"final_age_alive": final_age_alive}}

    def living(base_income: FloatND) -> dict:
        return {
            "utility": {"crra": crra},
            "H": {"discount_factor": discount_factor},
            "tax": {"tax_rate": tax_rate, "tax_exemption": tax_exemption},
            "coh": {"base_income": base_income},
            "alive_a": {"next_liquid": budget, **regime_age},
            "alive_b": {"next_liquid": budget, **regime_age},
            "dead": {"next_liquid": budget, **regime_age},
        }

    return {
        "alive_a": living(base_a),
        "alive_b": living(base_b),
        "dead": {"utility": {"crra": crra}},
    }
