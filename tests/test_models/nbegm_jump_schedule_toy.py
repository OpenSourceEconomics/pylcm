"""One-asset toy with a subsidy cliff declared as a jump `piecewise_affine`.

The same Medicaid-style asset cliff as `nbegm_medicaid_toy`, but expressed through
a jump-kind `lcm.piecewise_affine` schedule rather than `case_boundary`/`piece`.
Cash-on-hand jumps down as liquid crosses the cliff. The NBEGM schedule path
recognises the single jump and routes it to the binary case solver, so it must
reproduce the dense `GridSearch` value across the cliff at every age.
"""

import jax.numpy as jnp

import lcm
from lcm import LinSpacedGrid, Model
from lcm.typing import ContinuousState, FloatND
from tests.test_models.nbegm_common import (
    feasible,
    make_alive_dead_model,
    next_liquid_from_savings,
    resolve_solver,
    savings,
    utility,
)


@lcm.piecewise_affine(
    output="subsidy",
    variable="liquid",
    breakpoints=(lcm.affine_breakpoint(threshold="cliff", kind="jump"),),
)
def subsidy(
    *, liquid: ContinuousState, subsidy_low: float, subsidy_high: float, cliff: float
) -> FloatND:
    """Lump-sum subsidy: the higher amount below the cliff, the lower above it."""
    return jnp.where(liquid < cliff, subsidy_high, subsidy_low)


def resources(*, liquid: ContinuousState, subsidy: FloatND) -> FloatND:
    """Cash-on-hand: liquid wealth plus the cliff-contingent subsidy."""
    return liquid + subsidy


@lcm.piecewise_affine(
    output="resources",
    variable="liquid",
    breakpoints=(lcm.affine_breakpoint(threshold="cliff", kind="jump"),),
)
def resources_non_unit_above_cliff(
    *,
    liquid: ContinuousState,
    subsidy_low: float,
    subsidy_high: float,
    cliff: float,
) -> FloatND:
    """Cash-on-hand has slope one below the cliff and slope two above it."""
    below = liquid + subsidy_high
    above = 2.0 * liquid + subsidy_low
    return jnp.where(liquid < cliff, below, above)


def coh_non_additive(
    *, liquid: ContinuousState, subsidy: FloatND, coh_slope: float
) -> FloatND:
    """Cash-on-hand whose liquid slope is `coh_slope` (≠ 1) plus the jump subsidy.

    The schedule declares only the jump, so an all-jump route would solve this
    with the additive pure-jump step (unit-slope assumption) and silently
    mis-solve it — the case the build-time additivity guard must reject.
    """
    return coh_slope * liquid + subsidy


def coh_nonlinear(
    *, liquid: ContinuousState, subsidy: FloatND, curvature: float
) -> FloatND:
    """Cash-on-hand with genuine curvature in the liquid state, plus the subsidy.

    Smooth but not affine on any interval, so the per-interval affine segment
    recovered at the midpoint mis-tangents every other liquid point — the case the
    build-time affinity guard rejects.
    """
    return liquid + curvature * liquid**2 + subsidy


def coh_nonlinear_above_ten(
    *, liquid: ContinuousState, subsidy: FloatND, curvature: float
) -> FloatND:
    """Cash-on-hand is affine below ten and curved above ten."""
    excess = jnp.maximum(liquid - 10.0, 0.0)
    return liquid + curvature * excess**2 + subsidy


def build_model(
    *,
    variant: str = "brute",
    n_periods: int = 4,
    n_liquid: int = 120,
    n_consumption: int = 150,
    liquid_max: float = 30.0,
    n_savings: int = 150,
    savings_max: float = 28.0,
    non_additive: bool = False,
    non_additive_above_cliff: bool = False,
    nonlinear: bool = False,
    nonlinear_above_ten: bool = False,
) -> Model:
    """Create the two-regime (alive, dead) jump-schedule one-asset toy.

    With `non_additive`, cash-on-hand carries a non-unit (but affine) liquid slope
    while the schedule still declares only the jump — the misdeclaration the
    pure-jump unit-slope guard rejects. With `nonlinear`, cash-on-hand is smooth but
    not affine in the liquid state — the case the affinity guard rejects.
    """
    if non_additive_above_cliff:
        alive_functions = {
            "utility": utility,
            "resources": resources_non_unit_above_cliff,
            "savings": savings,
        }
    else:
        resources_func = resources
        if non_additive:
            resources_func = coh_non_additive
        elif nonlinear:
            resources_func = coh_nonlinear
        elif nonlinear_above_ten:
            resources_func = coh_nonlinear_above_ten
        alive_functions = {
            "utility": utility,
            "subsidy": subsidy,
            "resources": resources_func,
            "savings": savings,
        }
    alive_solver = resolve_solver(
        variant=variant,
        savings_grid=LinSpacedGrid(start=0.0, stop=savings_max, n_points=n_savings),
    )

    return make_alive_dead_model(
        n_periods=n_periods,
        n_liquid=n_liquid,
        liquid_max=liquid_max,
        n_consumption=n_consumption,
        alive_functions=alive_functions,
        liquid_law=next_liquid_from_savings,
        alive_solver=alive_solver,
        constraints={} if variant == "nbegm" else {"feasible": feasible},
    )


def build_params(
    *,
    discount_factor: float = 0.95,
    crra: float = 2.0,
    return_liquid: float = 0.03,
    income: float = 1.0,
    subsidy_high: float = 3.0,
    subsidy_low: float = 0.5,
    cliff: float = 8.0,
    final_age_alive: float = 3.0,
    non_additive: bool = False,
    non_additive_above_cliff: bool = False,
    coh_slope: float = 0.8,
    nonlinear: bool = False,
    curvature: float = 0.05,
    nonlinear_above_ten: bool = False,
) -> dict:
    """Get parameters for the jump-schedule one-asset toy."""
    alive_budget = {"return_liquid": return_liquid, "income": income}
    resources_params = {}
    if non_additive_above_cliff:
        resources_params = {
            "resources": {
                "subsidy_low": subsidy_low,
                "subsidy_high": subsidy_high,
                "cliff": cliff,
            }
        }
    elif non_additive:
        resources_params = {"resources": {"coh_slope": coh_slope}}
    elif nonlinear or nonlinear_above_ten:
        resources_params = {"resources": {"curvature": curvature}}
    return {
        "alive": {
            "utility": {"crra": crra},
            "koopmans_aggregator": {"discount_factor": discount_factor},
            **resources_params,
            **(
                {}
                if non_additive_above_cliff
                else {
                    "subsidy": {
                        "subsidy_low": subsidy_low,
                        "subsidy_high": subsidy_high,
                        "cliff": cliff,
                    }
                }
            ),
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
