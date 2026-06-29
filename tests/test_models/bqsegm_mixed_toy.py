"""One-asset toy mixing a subsidy cliff (jump) and a tax bracket (kink).

A single `piecewise_affine` net-transfer schedule declares two thresholds of
different kinds: a jump where the subsidy steps down at an asset cliff, and a
continuous kink where a tax phases in above an exemption. Cash-on-hand therefore
jumps at the cliff and bends at the exemption. The BQSEGM schedule path routes this
mixed schedule to the unified step, which must reproduce the dense `GridSearch`
value through both, every age.
"""

import jax.numpy as jnp

import lcm
from lcm import AgeGrid, LinSpacedGrid, MarkovTransition, Model, categorical
from lcm.regime import Regime
from lcm.solvers import GridSearch
from lcm.typing import BoolND, ContinuousAction, ContinuousState, FloatND, ScalarInt


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


@lcm.piecewise_affine(
    "net_transfer",
    variable="liquid",
    breakpoints=(
        lcm.affine_breakpoint("cliff", kind="jump"),
        lcm.affine_breakpoint("exemption", kind="continuous_kink"),
    ),
)
def net_transfer(
    liquid: ContinuousState,
    subsidy_low: float,
    subsidy_high: float,
    cliff: float,
    tax_rate: float,
    exemption: float,
) -> FloatND:
    """Subsidy that steps down at the cliff, net of a tax above the exemption."""
    subsidy = jnp.where(liquid < cliff, subsidy_high, subsidy_low)
    tax = tax_rate * jnp.maximum(liquid - exemption, 0.0)
    return subsidy - tax


def coh(liquid: ContinuousState, net_transfer: FloatND) -> FloatND:
    """Cash-on-hand: liquid wealth plus the net transfer."""
    return liquid + net_transfer


def next_liquid(
    coh: FloatND,
    consumption: ContinuousAction,
    return_liquid: float,
    income: float,
) -> ContinuousState:
    """Liquid law of motion: saved cash earns the liquid return, plus income."""
    return (1.0 + return_liquid) * (coh - consumption) + income


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
    n_consumption: int = 150,
    liquid_max: float = 30.0,
    n_savings: int = 150,
    savings_max: float = 28.0,
) -> Model:
    """Create the two-regime (alive, dead) mixed jump-and-kink one-asset toy."""
    ages = AgeGrid(start=0, stop=n_periods - 1, step="Y")
    final_age = ages.exact_values[-1]
    liquid_grid = LinSpacedGrid(start=0.1, stop=liquid_max, n_points=n_liquid)

    alive_functions = {"utility": utility, "net_transfer": net_transfer, "coh": coh}
    if variant == "brute":
        alive_solver = GridSearch()
    elif variant == "bqsegm":
        from lcm.solvers import BQSEGM  # noqa: PLC0415

        alive_solver = BQSEGM(
            savings_grid=LinSpacedGrid(start=0.0, stop=savings_max, n_points=n_savings)
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
        states={"liquid": liquid_grid},
        state_transitions={"liquid": {"alive": next_liquid, "dead": next_liquid}},
        constraints={"feasible": feasible},
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
    subsidy_high: float = 3.0,
    subsidy_low: float = 0.5,
    cliff: float = 6.0,
    tax_rate: float = 0.3,
    exemption: float = 16.0,
    final_age_alive: float = 3.0,
) -> dict:
    """Get parameters for the mixed jump-and-kink one-asset toy.

    The cliff (jump) precedes the exemption (kink), so the schedule's declared
    breakpoint order is ascending in value.
    """
    alive_budget = {"return_liquid": return_liquid, "income": income}
    return {
        "alive": {
            "utility": {"crra": crra},
            "H": {"discount_factor": discount_factor},
            "net_transfer": {
                "subsidy_low": subsidy_low,
                "subsidy_high": subsidy_high,
                "cliff": cliff,
                "tax_rate": tax_rate,
                "exemption": exemption,
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
