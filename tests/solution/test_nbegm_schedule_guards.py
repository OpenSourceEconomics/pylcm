"""NB-EGM refuses schedule declarations its single-liquid cores cannot solve.

The non-ride-along schedule path builds its interval partition from one
schedule's thresholds and dispatches on that schedule's breakpoint kinds. Two
declarations fall outside what those cores represent, and each produced a
plausible finite solve rather than an error:

- a second liquid-direct schedule, whose thresholds never enter the partition,
  so an interval straddling the second schedule's discontinuity is tangented as
  if it were smooth;
- a hard-constraint floor declared alongside a value jump, which routes to the
  step that takes no flat-interval mask, so the floor's plateau reaches an
  inversion that assumes a strictly increasing budget.

Both are rejected at model build with the alternative named.
"""

from collections.abc import Callable, Mapping

import jax.numpy as jnp
import pytest

import lcm
from lcm import LinSpacedGrid, Model
from lcm.exceptions import RegimeInitializationError
from lcm.typing import ContinuousState, FloatND
from tests.test_models.nbegm_common import (
    feasible,
    make_alive_dead_model,
    next_liquid_from_savings,
    resolve_solver,
    savings,
    utility,
)

SAVINGS_GRID = LinSpacedGrid(start=0.0, stop=28.0, n_points=40)


@lcm.piecewise_affine(
    output="tax",
    variable="liquid",
    breakpoints=(lcm.affine_breakpoint(threshold="tax_kink", kind="continuous_kink"),),
)
def tax(liquid: ContinuousState, tax_rate: float, tax_kink: float) -> FloatND:
    """Continuous tax on liquid wealth above the kink."""
    return tax_rate * jnp.maximum(liquid - tax_kink, 0.0)


@lcm.piecewise_affine(
    output="subsidy",
    variable="liquid",
    breakpoints=(lcm.affine_breakpoint(threshold="subsidy_limit", kind="jump"),),
)
def subsidy(
    liquid: ContinuousState, subsidy_level: float, subsidy_limit: float
) -> FloatND:
    """Means-tested transfer that cuts off at the asset limit."""
    return jnp.where(liquid < subsidy_limit, subsidy_level, 0.0)


@lcm.piecewise_affine(
    output="net_transfer",
    variable="liquid",
    breakpoints=(
        lcm.affine_breakpoint(threshold="floor_limit", kind="hard_constraint"),
        lcm.affine_breakpoint(threshold="cliff_limit", kind="jump"),
    ),
)
def net_transfer(
    liquid: ContinuousState,
    floor_level: float,
    floor_limit: float,
    cliff_level: float,
    cliff_limit: float,
) -> FloatND:
    """A consumption floor below `floor_limit` plus a cliffed transfer above it."""
    floor_part = jnp.maximum(floor_level - jnp.minimum(liquid, floor_limit), 0.0)
    cliff_part = jnp.where(liquid < cliff_limit, cliff_level, 0.0)
    return floor_part + cliff_part


def _resources_two_schedules(
    liquid: ContinuousState, tax: FloatND, subsidy: FloatND
) -> FloatND:
    return liquid - tax + subsidy


def _resources_floor_and_cliff(
    liquid: ContinuousState, net_transfer: FloatND
) -> FloatND:
    return liquid + net_transfer


def _build(*, alive_functions: Mapping[str, Callable[..., object]]) -> Model:
    return make_alive_dead_model(
        n_periods=3,
        n_liquid=20,
        liquid_max=30.0,
        n_consumption=20,
        alive_functions=alive_functions,
        liquid_law=next_liquid_from_savings,
        alive_solver=resolve_solver(
            "nbegm", savings_grid=SAVINGS_GRID, post_decision_function="savings"
        ),
        constraints={"feasible": feasible},
    )


def test_two_liquid_direct_schedules_are_rejected():
    """Only one liquid-direct schedule enters the single-liquid partition."""
    with pytest.raises(RegimeInitializationError, match="one liquid-direct schedule"):
        _build(
            alive_functions={
                "utility": utility,
                "savings": savings,
                "tax": tax,
                "subsidy": subsidy,
                "resources": _resources_two_schedules,
            }
        )


def test_a_hard_constraint_declared_with_a_jump_is_rejected():
    """The mixed jump-and-kink step carries no flat-interval mask."""
    with pytest.raises(RegimeInitializationError, match="hard-constraint"):
        _build(
            alive_functions={
                "utility": utility,
                "savings": savings,
                "net_transfer": net_transfer,
                "resources": _resources_floor_and_cliff,
            }
        )
