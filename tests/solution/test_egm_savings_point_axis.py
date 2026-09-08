"""DC-EGM tiles the per-savings-node continuation under `savings_point`.

The dominant DC-EGM working buffer is the per-savings-node continuation — the
savings nodes times the child stochastic mesh times the combo block. That node
loop is the `savings_point` axis the value and replay programs declare, so the
block it runs in is an execution fact the plan owns rather than a field on a
grid. The upper envelope still runs on the gathered full endogenous grid, so
every width publishes the same value to the working format's rounding —
including widths that do not divide the grid.
"""

import functools
from collections.abc import Mapping

import numpy as np
import pytest

from _lcm.execution.core_program import core_program_graph
from _lcm.execution.workspace_planning import workspace_width_candidates
from lcm import ExecutionConfig, LinSpacedGrid, MarkovTransition, Model
from lcm.consumption_savings_regime import ConsumptionSavingsRegime, LiquidMargin
from lcm.regime import Regime as UserRegime
from lcm.solvers import DCEGM, SAVINGS_POINT_AXIS
from lcm.typing import FloatND
from tests.conftest import EXACT_KERNEL_SKIP_REASON, assert_agrees_to_ulp
from tests.solution.test_egm_euler_point_axis import (
    CONSUMPTION_GRID,
    N_WEALTH,
    RegimeId,
    _ages,
    _params,
    bequest,
    death_prob,
    inverse_marginal_utility,
    next_wealth,
    savings,
    stay_prob,
    utility,
)

pytestmark = pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)

# Prime, so every tile width but 1 leaves a ragged final tile. Sized for the
# cheapest solve that still shows that; invariance under the partition does not
# depend on resolution.
# Tiling the savings-node loop keeps every operation and operand order, so the two
# solves differ only by the vectorized kernel XLA emits for each tile width. The gap
# over widths 1 and 7 tops out at 1 ULP at float64 and 2 ULP at float32, so this
# bound leaves two binades of headroom.
_INVARIANCE_ULP = 8

N_SAVINGS = 17


@functools.cache
def _model(width: int | None = None) -> Model:
    """Asset-row DC-EGM model whose per-savings-node loop the plan tiles.

    A `width` fixes the savings-node axis in the model's execution plan; `None`
    leaves the width to the plan.
    """
    config = (
        ExecutionConfig()
        if width is None
        else ExecutionConfig(axis_widths={SAVINGS_POINT_AXIS: width})
    )
    ages = _ages()
    last_age = ages.exact_values[-1]
    working = ConsumptionSavingsRegime(
        transition={
            "working": MarkovTransition(stay_prob),
            "dead": MarkovTransition(death_prob),
        },
        active=lambda age, la=last_age: age < la,
        actions={"consumption": CONSUMPTION_GRID},
        states={"wealth": LinSpacedGrid(start=1.0, stop=100.0, n_points=N_WEALTH)},
        state_transitions={"wealth": next_wealth},
        functions={
            "utility": utility,
            "savings": savings,
            "inverse_marginal_utility": inverse_marginal_utility,
        },
        solver=DCEGM(
            savings_grid=LinSpacedGrid(start=0.0, stop=110.0, n_points=N_SAVINGS),
            n_constrained_points=32,
        ),
        liquid=LiquidMargin(
            state="wealth",
            action="consumption",
            resources="wealth",
            post_decision_state="savings",
        ),
    )
    dead = UserRegime(
        transition=None,
        states={"wealth": LinSpacedGrid(start=1.0, stop=120.0, n_points=40)},
        functions={"utility": bequest},
    )
    return Model(
        regimes={"working": working, "dead": dead},
        ages=ages,
        regime_id_class=RegimeId,
        execution_config=config,
    )


def _solve(width: int) -> Mapping[int, Mapping[str, FloatND]]:
    """Solve with the savings-node axis tiled at `width`."""
    return _model(width).solve(params=_params(), log_level="debug").values


def _savings_point_axis():
    """Return the `savings_point` axis the regime's value program declares."""
    kernels = _model()._regimes["working"].solution.period_kernels
    program = core_program_graph(kernel=next(iter(kernels.values())))["main"]
    (axis,) = [
        candidate
        for candidate in program.requirements.tiled_axes
        if candidate.name == SAVINGS_POINT_AXIS
    ]
    return axis


def test_value_program_declares_the_savings_point_axis() -> None:
    """The per-savings-node loop is declared as an axis of the value program."""
    kernels = _model()._regimes["working"].solution.period_kernels
    program = core_program_graph(kernel=next(iter(kernels.values())))["main"]

    assert SAVINGS_POINT_AXIS in program.requirements.axis_names


def test_savings_point_axis_spans_the_exogenous_savings_grid() -> None:
    """The axis runs over one cell per node of the solver's savings grid."""
    assert _savings_point_axis().extent == N_SAVINGS


def test_savings_point_axis_names_the_cores_width_keyword() -> None:
    """The axis names the keyword the DC-EGM core takes its tile width on."""
    assert _savings_point_axis().width_keyword == "_lcm_savings_point_width"


@pytest.mark.parametrize("width", [1, 7, N_SAVINGS])
def test_a_fixed_savings_point_width_is_the_width_the_plan_selects(
    *, width: int
) -> None:
    """A width `axis_widths` fixes is the tile width the plan hands the core."""
    axis = _savings_point_axis()
    (candidate,) = workspace_width_candidates(
        axes=(axis,), fixed_widths={SAVINGS_POINT_AXIS: width}
    )

    assert candidate[SAVINGS_POINT_AXIS] == width


@pytest.mark.parametrize("width", [1, 7])
def test_value_agrees_across_savings_point_widths(*, width: int) -> None:
    """Tiling the savings-node loop at any width reproduces the untiled value.

    Includes a width (7) that does not divide the seventeen-node grid, so the
    last tile is short. Only the schedule changes; the envelope still runs on
    the gathered full endogenous grid.
    """
    reference = _solve(N_SAVINGS)
    tiled = _solve(width)
    for period in sorted(reference):
        for regime_name in reference[period]:
            assert_agrees_to_ulp(
                got=np.asarray(tiled[period][regime_name]),
                expected=np.asarray(reference[period][regime_name]),
                n_ulp=_INVARIANCE_ULP,
                err_msg=f"period={period}, regime={regime_name}",
            )
