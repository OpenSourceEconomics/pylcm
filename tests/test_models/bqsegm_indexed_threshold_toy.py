"""One-asset BQSEGM toy whose cliff threshold is a per-ride-along-cell table.

Isolates state-indexed *thresholds* from state-indexed *offsets*: the derived
income variable `gross_income = liquid + base_income` carries the SAME offset in
both `kind` slices, but the subsidy cliff sits at `gross_income == fpl_cliff[kind]`
— a threshold read from a length-2 table indexed by the ride-along `kind` state.
The cliff therefore maps to the asset preimage `liquid = fpl_cliff[kind] -
base_income`, a different liquid point per slice driven entirely by the threshold.
The BQSEGM ride-along solver must resolve each cell's threshold from the table
before mapping it to the per-cell preimage. The brute variant (`GridSearch`)
productmaps over `(liquid, kind, consumption)` and is the dense agreement oracle.
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
    alive: ScalarInt
    dead: ScalarInt


@categorical(ordered=False)
class ConsumerKind:
    lo: ScalarInt
    hi: ScalarInt


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


def gross_income(liquid: ContinuousState, base_income: float) -> FloatND:
    """Pre-tax income: liquid wealth plus a kind-invariant base income (monotone)."""
    return liquid + base_income


@lcm.piecewise_affine(
    "subsidy",
    variable="gross_income",
    breakpoints=(lcm.affine_breakpoint("fpl_cliff", kind="jump", indexed_by="kind"),),
)
def subsidy(
    gross_income: FloatND,
    kind: DiscreteState,
    subsidy_low: float,
    subsidy_high: float,
    fpl_cliff: FloatND,
) -> FloatND:
    """Lump-sum subsidy: higher below the kind's income cliff, lower above."""
    return jnp.where(gross_income < fpl_cliff[kind], subsidy_high, subsidy_low)


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
    n_liquid: int = 160,
    n_consumption: int = 200,
    liquid_max: float = 30.0,
    n_savings: int = 200,
    savings_max: float = 28.0,
) -> Model:
    """Create the (alive, dead) toy whose subsidy cliff threshold is kind-indexed.

    Args:
        variant: `"brute"` drives the alive regime by `GridSearch`; `"bqsegm"` by
            the `BQSEGM` schedule solver, which must read each cell's cliff
            threshold from the `fpl_cliff` table and map it to its asset preimage.
        n_periods: Number of lifecycle periods (the last is terminal).
        n_liquid: Liquid-state grid size.
        n_consumption: Consumption-action grid size (brute only).
        liquid_max: Upper bound of the liquid grid.
        n_savings: Post-decision savings grid size (BQSEGM only).
        savings_max: Upper bound of the savings grid (BQSEGM only).

    Returns:
        The assembled `Model`.

    """
    ages = AgeGrid(start=0, stop=n_periods - 1, step="Y")
    final_age = ages.exact_values[-1]
    liquid_grid = LinSpacedGrid(start=0.1, stop=liquid_max, n_points=n_liquid)

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
        states={"liquid": liquid_grid, "kind": DiscreteGrid(ConsumerKind)},
        state_transitions={
            "liquid": {"alive": liquid_law, "dead": liquid_law},
            "kind": {"alive": lcm.fixed_transition("kind")},
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
    base_income: float = 5.0,
    subsidy_low: float = 0.0,
    subsidy_high: float = 3.0,
    fpl_cliff_lo: float = 14.0,
    fpl_cliff_hi: float = 11.0,
    final_age_alive: float = 3.0,
) -> dict:
    """Get parameters for the kind-indexed-threshold subsidy-cliff toy.

    `fpl_cliff` is a length-2 array indexed by the `kind` code (`lo`, `hi`). With
    the kind-invariant `base_income`, the cliff `gross_income == fpl_cliff[kind]`
    maps to the asset preimage `liquid = fpl_cliff[kind] - base_income` — `9.0` for
    `lo`, `6.0` for `hi` at the defaults, both grid-interior — so the cash-on-hand
    discontinuity sits at different assets across slices, driven only by the
    per-cell threshold.
    """
    fpl_cliff = jnp.array([fpl_cliff_lo, fpl_cliff_hi])
    alive_budget = {"return_liquid": return_liquid, "income": income}
    return {
        "alive": {
            "utility": {"crra": crra},
            "H": {"discount_factor": discount_factor},
            "gross_income": {"base_income": base_income},
            "subsidy": {
                "subsidy_low": subsidy_low,
                "subsidy_high": subsidy_high,
                "fpl_cliff": fpl_cliff,
            },
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
