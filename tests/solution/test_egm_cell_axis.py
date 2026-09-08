"""DC-EGM tiles its output state cells under the `cell` planner axis.

The kernel maps the per-combo solve over the Cartesian product of the regime's
discrete states, passive states, and discrete actions. The state part of that
product is the `cell` axis the value and replay programs declare — one cell per
output state combination — so the block it runs in is an execution fact the plan
owns rather than a field on a grid. Discrete-action axes are never tiled: the
action aggregation needs every action's value at once. Tiles are concatenated
into the same canonical combo order at every width, so the published value moves
only by the vectorized kernel XLA emits per tile width.
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
    DiscreteGrid,
    ExecutionConfig,
    IrregSpacedGrid,
    LinSpacedGrid,
    MarkovTransition,
    Model,
    categorical,
    fixed_transition,
)
from lcm.consumption_savings_regime import ConsumptionSavingsRegime, LiquidMargin
from lcm.exceptions import ModelInitializationError
from lcm.regime import Regime as UserRegime
from lcm.solvers import CELL_AXIS, DCEGM
from lcm.typing import (
    ContinuousAction,
    ContinuousState,
    DiscreteState,
    FloatND,
    ScalarInt,
)
from tests.conftest import EXACT_KERNEL_SKIP_REASON, assert_agrees_to_ulp

pytestmark = pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)

# Tiling the cell axis only reschedules the `lax.map`, leaving every operation and
# its operand order untouched; the two solves differ only by the vectorized kernel
# XLA emits for each tile width — a gap of a few ULP, not of an economic magnitude.
# Over widths 1 and 2, with and without a discrete action, that gap tops out at 2 ULP
# at float64 and 2 ULP at float32.
_INVARIANCE_ULP = 16

N_PERIODS = 4
N_WEALTH = 12
BAND_START = 5.0
BAND_WIDTH = 40.0

# Block assembly is a scheduling property, so it is detected at any model size:
# these grids are sized for the cheapest solve that still exercises the
# flatten-and-transpose path, not for resolution.
CONSUMPTION_GRID = LinSpacedGrid(start=0.25, stop=100.0, n_points=60)
SAVINGS_GRID = IrregSpacedGrid(points=tuple(110.0 * (i / 29) ** 3 for i in range(30)))


# Health has three levels and marital status two, so a cell width of 2 leaves a
# ragged final tile on the one-state model and divides the two-state product.
N_HEALTH = 3
N_MARITAL = 2


@categorical(ordered=False)
class Health:
    bad: ScalarInt
    fair: ScalarInt
    good: ScalarInt


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


def health_transition(health: DiscreteState) -> FloatND:
    # A genuine Markov health combo axis carried into the child.
    stay = jnp.where(health == Health.good, 0.7, 0.5)
    others = (1.0 - stay) / 2.0
    return jnp.stack([others, others, stay])


def utility(*, consumption: ContinuousAction, health: DiscreteState) -> FloatND:
    penalty = jnp.where(
        health == Health.bad, 0.2, jnp.where(health == Health.fair, 0.1, 0.0)
    )
    return jnp.log(consumption) - penalty


def savings(*, wealth: FloatND, consumption: ContinuousAction) -> FloatND:
    return wealth - consumption


def inverse_marginal_utility(marginal_continuation: FloatND) -> FloatND:
    return 1.0 / marginal_continuation


def next_wealth(savings: FloatND) -> ContinuousState:
    return savings + 3.0


def bequest(wealth: ContinuousState) -> FloatND:
    return jnp.log(wealth + 1.0)


def _cell_width_config(width: int | None) -> ExecutionConfig:
    """Return the plan that fixes the cell axis at `width`, or leaves it open."""
    if width is None:
        return ExecutionConfig()
    return ExecutionConfig(axis_widths={CELL_AXIS: width})


def _ages() -> AgeGrid:
    return AgeGrid(start=40, stop=40 + (N_PERIODS - 1) * 10, step="10Y")


@functools.cache
def _model(width: int | None = None) -> Model:
    """Asset-row DC-EGM with a Markov health state the cell axis tiles over.

    A `width` fixes the cell axis in the model's execution plan; `None` leaves
    the width to the plan.
    """
    config = _cell_width_config(width)
    ages = _ages()
    last_age = ages.exact_values[-1]
    working = ConsumptionSavingsRegime(
        transition={
            "working": MarkovTransition(stay_prob),
            "dead": MarkovTransition(death_prob),
        },
        active=lambda age, la=last_age: age < la,
        actions={"consumption": CONSUMPTION_GRID},
        states={
            "wealth": LinSpacedGrid(start=1.0, stop=100.0, n_points=N_WEALTH),
            "health": DiscreteGrid(category_class=Health),
        },
        state_transitions={
            "wealth": next_wealth,
            "health": MarkovTransition(health_transition),
        },
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
        execution_config=config,
    )


def _params() -> dict:
    return {"discount_factor": 0.95, "final_age_alive": 40 + (N_PERIODS - 2) * 10}


def _solve(width: int) -> Mapping[int, Mapping[str, FloatND]]:
    """Solve the one-state model with the cell axis tiled at `width`."""
    return _model(width).solve(params=_params(), log_level="debug").values


def _model_with_batched_health() -> Model:
    """Build the same model with a `batch_size` on its discrete health grid."""
    ages = _ages()
    last_age = ages.exact_values[-1]
    working = ConsumptionSavingsRegime(
        transition={
            "working": MarkovTransition(stay_prob),
            "dead": MarkovTransition(death_prob),
        },
        active=lambda age, la=last_age: age < la,
        actions={"consumption": CONSUMPTION_GRID},
        states={
            "wealth": LinSpacedGrid(start=1.0, stop=100.0, n_points=N_WEALTH),
            "health": DiscreteGrid(category_class=Health, batch_size=2),
        },
        state_transitions={
            "wealth": next_wealth,
            "health": MarkovTransition(health_transition),
        },
        functions={
            "utility": utility,
            "savings": savings,
            "inverse_marginal_utility": inverse_marginal_utility,
        },
        solver=DCEGM(savings_grid=SAVINGS_GRID, n_constrained_points=32),
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


@categorical(ordered=False)
class Work:
    stays_home: ScalarInt
    works: ScalarInt


def utility_with_action(
    *, consumption: ContinuousAction, health: DiscreteState, works: DiscreteState
) -> FloatND:
    penalty = jnp.where(
        health == Health.bad, 0.2, jnp.where(health == Health.fair, 0.1, 0.0)
    )
    effort = jnp.where(works == Work.works, 0.15, 0.0)
    return jnp.log(consumption) - penalty - effort


@functools.cache
def _action_model(width: int | None = None) -> Model:
    """Asset-row DC-EGM with a health state cell axis and a discrete action."""
    config = _cell_width_config(width)
    ages = _ages()
    last_age = ages.exact_values[-1]
    working = ConsumptionSavingsRegime(
        transition={
            "working": MarkovTransition(stay_prob),
            "dead": MarkovTransition(death_prob),
        },
        active=lambda age, la=last_age: age < la,
        actions={
            "consumption": CONSUMPTION_GRID,
            "works": DiscreteGrid(category_class=Work),
        },
        states={
            "wealth": LinSpacedGrid(start=1.0, stop=100.0, n_points=N_WEALTH),
            "health": DiscreteGrid(category_class=Health),
        },
        state_transitions={
            "wealth": next_wealth,
            "health": MarkovTransition(health_transition),
        },
        functions={
            "utility": utility_with_action,
            "savings": savings,
            "inverse_marginal_utility": inverse_marginal_utility,
        },
        solver=DCEGM(savings_grid=SAVINGS_GRID, n_constrained_points=32),
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


def _solve_action_model(width: int) -> Mapping[int, Mapping[str, FloatND]]:
    """Solve the discrete-action model with the cell axis tiled at `width`."""
    return _action_model(width).solve(params=_params(), log_level="debug").values


def _cell_axis(*, model: Model):
    """Return the `cell` axis `model`'s working value program declares."""
    kernels = model._regimes["working"].solution.period_kernels
    program = core_program_graph(kernel=next(iter(kernels.values())))["main"]
    (axis,) = [
        candidate
        for candidate in program.requirements.tiled_axes
        if candidate.name == CELL_AXIS
    ]
    return axis


def test_a_batched_grid_is_refused_in_a_dcegm_regime() -> None:
    """A `batch_size` a DC-EGM regime cannot read is refused, not ignored."""
    with pytest.raises(ModelInitializationError, match="batch_size=2"):
        _model_with_batched_health()


def test_value_program_declares_the_cell_axis() -> None:
    """The output state-cell loop is declared as an axis of the value program."""
    kernels = _model()._regimes["working"].solution.period_kernels
    program = core_program_graph(kernel=next(iter(kernels.values())))["main"]

    assert CELL_AXIS in program.requirements.axis_names


def test_cell_axis_spans_the_regimes_output_state_cells() -> None:
    """The axis runs over one cell per discrete- and passive-state combination."""
    assert _cell_axis(model=_model()).extent == N_HEALTH


def test_cell_axis_excludes_the_discrete_action_axes() -> None:
    """A discrete action is not an output state, so it is outside the tiled axis.

    The action aggregation needs every action's value at once, so the action
    axis stays vmapped inside each tile rather than being tiled with the cells.
    """
    assert "works" not in _cell_axis(model=_action_model()).state_names


def test_cell_axis_spans_only_the_states_when_an_action_is_present() -> None:
    """With a discrete action the axis still runs over the state cells alone."""
    assert _cell_axis(model=_action_model()).extent == N_HEALTH


@pytest.mark.parametrize("width", [1, 2])
def test_value_agrees_across_cell_widths_with_a_discrete_action(*, width: int) -> None:
    """A narrow cell width reproduces the untiled value with an action present.

    This is the tiled leg with a non-empty vmapped remainder: the state cells
    are flattened into one `lax.map` while the discrete-action axis is vmapped
    within each tile, and the tiles are transposed back into the canonical combo
    order — which the published value would show if it were wrong.
    """
    reference = _solve_action_model(N_HEALTH)
    tiled = _solve_action_model(width)
    for period in sorted(reference):
        for regime_name in reference[period]:
            assert_agrees_to_ulp(
                got=np.asarray(tiled[period][regime_name]),
                expected=np.asarray(reference[period][regime_name]),
                n_ulp=_INVARIANCE_ULP,
                err_msg=f"period={period}, regime={regime_name}",
            )


def test_cell_axis_names_the_cores_width_keyword() -> None:
    """The axis names the keyword the DC-EGM core takes its tile width on."""
    assert _cell_axis(model=_model()).width_keyword == "_lcm_cell_width"


def test_cell_axis_spans_the_product_of_two_state_axes() -> None:
    """With two discrete states the axis runs over their Cartesian product."""
    extent = N_HEALTH * N_MARITAL

    assert _cell_axis(model=_two_combo_model()).extent == extent


@pytest.mark.parametrize("width", [1, 2, 3])
def test_a_fixed_cell_width_is_the_width_the_plan_selects(*, width: int) -> None:
    """A width `axis_widths` fixes is the tile width the plan hands the core."""
    axis = _cell_axis(model=_model())
    (candidate,) = workspace_width_candidates(
        axes=(axis,), fixed_widths={CELL_AXIS: width}
    )

    assert candidate[CELL_AXIS] == width


@pytest.mark.parametrize("width", [1, 2])
def test_value_agrees_across_cell_widths(*, width: int) -> None:
    """Tiling the state-cell loop at any width reproduces the untiled value.

    Includes a width (2) that does not divide the three-level health axis, so
    the last tile is short.
    """
    reference = _solve(N_HEALTH)
    tiled = _solve(width)
    for period in sorted(reference):
        for regime_name in reference[period]:
            assert_agrees_to_ulp(
                got=np.asarray(tiled[period][regime_name]),
                expected=np.asarray(reference[period][regime_name]),
                n_ulp=_INVARIANCE_ULP,
                err_msg=f"period={period}, regime={regime_name}",
            )


@categorical(ordered=False)
class Marital:
    single: ScalarInt
    married: ScalarInt


def utility_two_combos(
    *, consumption: ContinuousAction, health: DiscreteState, married: DiscreteState
) -> FloatND:
    penalty = jnp.where(
        health == Health.bad, 0.2, jnp.where(health == Health.fair, 0.1, 0.0)
    )
    bonus = jnp.where(married == Marital.married, 0.05, 0.0)
    return jnp.log(consumption) - penalty + bonus


@functools.cache
def _two_combo_model(width: int | None = None) -> Model:
    """Asset-row DC-EGM with TWO discrete state axes (health + married)."""
    config = _cell_width_config(width)
    ages = _ages()
    last_age = ages.exact_values[-1]
    working = ConsumptionSavingsRegime(
        transition={
            "working": MarkovTransition(stay_prob),
            "dead": MarkovTransition(death_prob),
        },
        active=lambda age, la=last_age: age < la,
        actions={"consumption": CONSUMPTION_GRID},
        states={
            "wealth": LinSpacedGrid(start=1.0, stop=100.0, n_points=N_WEALTH),
            "health": DiscreteGrid(category_class=Health),
            "married": DiscreteGrid(category_class=Marital),
        },
        state_transitions={
            "wealth": next_wealth,
            "health": MarkovTransition(health_transition),
            "married": fixed_transition("married"),
        },
        functions={
            "utility": utility_two_combos,
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
        execution_config=config,
    )


@pytest.mark.parametrize("width", [1, 2, 4])
def test_value_agrees_across_cell_widths_over_two_state_axes(*, width: int) -> None:
    """Tiling a two-state cell product reproduces the untiled value.

    The kernel runs one `lax.map` over the flattened (health-by-married) product
    rather than nesting one per axis, and concatenates the tiles back into the
    canonical combo order — which this guards along with the value.
    """
    extent = N_HEALTH * N_MARITAL

    def solve(at_width: int) -> Mapping[int, Mapping[str, FloatND]]:
        return (
            _two_combo_model(at_width).solve(params=_params(), log_level="debug").values
        )

    reference = solve(extent)
    tiled = solve(width)
    for period in sorted(reference):
        for regime_name in reference[period]:
            assert_agrees_to_ulp(
                got=np.asarray(tiled[period][regime_name]),
                expected=np.asarray(reference[period][regime_name]),
                n_ulp=_INVARIANCE_ULP,
                err_msg=f"period={period}, regime={regime_name}",
            )
