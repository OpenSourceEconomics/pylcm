"""DC-EGM tiles the asset-row solve under the `euler_point` planner axis.

A DC-EGM regime in asset-row mode solves the single-post-state pipeline once
per exogenous Euler-state (asset) node. That node loop is the `euler_point`
axis the value and replay programs declare, so the block it runs in is an
execution fact the plan owns rather than a field on a grid. Tiles are
concatenated, never folded, so which cells are feasible is unchanged at every
width and the published value agrees to the working format's rounding —
including widths that do not divide the grid.
"""

import functools
from collections.abc import Mapping

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.execution.core_program import core_program_graph
from _lcm.execution.workspace_planning import workspace_width_candidates
from lcm import (
    AgeGrid,
    ExecutionConfig,
    IrregSpacedGrid,
    LinSpacedGrid,
    MarkovTransition,
    Model,
    categorical,
)
from lcm.consumption_savings_regime import ConsumptionSavingsRegime, LiquidMargin
from lcm.regime import Regime as UserRegime
from lcm.solvers import DCEGM, EULER_POINT_AXIS
from lcm.typing import (
    ContinuousAction,
    ContinuousState,
    FloatND,
    ScalarInt,
)
from tests.conftest import EXACT_KERNEL_SKIP_REASON, invariance_tolerances

pytestmark = pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)

N_PERIODS = 4
N_WEALTH = 11  # prime, so every block size but 1 leaves a ragged final block
BAND_START = 5.0
BAND_WIDTH = 40.0

# Block assembly is a scheduling property, so it is detected at any model size:
# these grids are sized for the cheapest solve that still exercises a ragged
# final block, not for resolution.
CONSUMPTION_GRID = LinSpacedGrid(start=0.25, stop=100.0, n_points=60)
SAVINGS_GRID = IrregSpacedGrid(points=tuple(110.0 * (i / 29) ** 3 for i in range(30)))


@categorical(ordered=False)
class RegimeId:
    working: ScalarInt
    dead: ScalarInt


def smoothstep(value: FloatND) -> FloatND:
    t = jnp.clip((value - BAND_START) / BAND_WIDTH, 0.0, 1.0)
    return t * t * t * (t * (6.0 * t - 15.0) + 10.0)


def survival_of_wealth(wealth: ContinuousState) -> FloatND:
    # Reading the Euler state in the regime-transition probability switches the
    # kernel into the per-exogenous-asset-node (asset-row) solve.
    return 0.5 + 0.45 * smoothstep(wealth)


def stay_prob(*, wealth: ContinuousState, age: int, final_age_alive: float) -> FloatND:
    return jnp.where(age >= final_age_alive, 0.0, survival_of_wealth(wealth))


def death_prob(*, wealth: ContinuousState, age: int, final_age_alive: float) -> FloatND:
    return 1.0 - stay_prob(wealth=wealth, age=age, final_age_alive=final_age_alive)


def utility(consumption: ContinuousAction) -> FloatND:
    return jnp.log(consumption)


def savings(*, wealth: FloatND, consumption: ContinuousAction) -> FloatND:
    return wealth - consumption


def inverse_marginal_utility(marginal_continuation: FloatND) -> FloatND:
    return 1.0 / marginal_continuation


def next_wealth(savings: FloatND) -> ContinuousState:
    return savings + 3.0


def bequest(wealth: ContinuousState) -> FloatND:
    return jnp.log(wealth + 1.0)


def _ages() -> AgeGrid:
    return AgeGrid(start=40, stop=40 + (N_PERIODS - 1) * 10, step="10Y")


@functools.cache
def _model() -> Model:
    """Asset-row DC-EGM model whose per-node solve the plan tiles."""
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
            savings_grid=SAVINGS_GRID,
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
    )


def _params() -> dict:
    return {"discount_factor": 0.95, "final_age_alive": 40 + (N_PERIODS - 2) * 10}


def _solve(width: int | None) -> Mapping[int, Mapping[str, FloatND]]:
    """Solve with the node axis tiled at `width`, or at the plan's own choice."""
    config = (
        ExecutionConfig()
        if width is None
        else ExecutionConfig(axis_widths={EULER_POINT_AXIS: width})
    )
    return (
        _model()
        .solve(params=_params(), log_level="debug", execution_config=config)
        .values
    )


def _euler_point_axis():
    """Return the `euler_point` axis the regime's value program declares."""
    kernels = _model()._regimes["working"].solution.period_kernels
    program = core_program_graph(kernel=next(iter(kernels.values())))["main"]
    (axis,) = [
        candidate
        for candidate in program.requirements.tiled_axes
        if candidate.name == EULER_POINT_AXIS
    ]
    return axis


def test_value_program_declares_the_euler_point_axis() -> None:
    """The asset-row node loop is declared as an axis of the value program."""
    kernels = _model()._regimes["working"].solution.period_kernels
    program = core_program_graph(kernel=next(iter(kernels.values())))["main"]

    assert EULER_POINT_AXIS in program.requirements.axis_names


def test_euler_point_axis_spans_the_exogenous_euler_grid() -> None:
    """The axis runs over one cell per node of the regime's Euler-state grid."""
    assert _euler_point_axis().extent == N_WEALTH


def test_euler_point_axis_names_the_cores_width_keyword() -> None:
    """The axis names the keyword the asset-row core takes its tile width on."""
    assert _euler_point_axis().width_keyword == "_lcm_euler_point_width"


@pytest.mark.parametrize("width", [1, 4, N_WEALTH])
def test_a_fixed_euler_point_width_is_the_width_the_plan_selects(*, width: int) -> None:
    """A width `axis_widths` fixes is the tile width the plan hands the core."""
    axis = _euler_point_axis()
    (candidate,) = workspace_width_candidates(
        axes=(axis,), fixed_widths={EULER_POINT_AXIS: width}
    )

    assert candidate[EULER_POINT_AXIS] == width


@pytest.mark.parametrize("width", [1, 4])
def test_value_agrees_across_euler_point_widths(*, width: int) -> None:
    """Tiling the node loop at any width reproduces the untiled solve's value.

    Includes a width (4) that does not divide the eleven-node grid, so the last
    tile is short. Feasibility is structural and matches exactly; the published
    value moves only by the vectorized kernel XLA emits per tile width.
    """
    reference = _solve(N_WEALTH)
    tiled = _solve(width)
    for period in sorted(reference):
        for regime_name in reference[period]:
            ref_V = np.asarray(reference[period][regime_name])
            got_V = np.asarray(tiled[period][regime_name])
            np.testing.assert_array_equal(
                np.isfinite(got_V),
                np.isfinite(ref_V),
                err_msg=f"feasibility differs: period={period}, regime={regime_name}",
            )
            rtol, atol = invariance_tolerances(ref_V)
            np.testing.assert_allclose(
                got_V,
                ref_V,
                rtol=rtol,
                atol=atol,
                err_msg=f"period={period}, regime={regime_name}",
            )
