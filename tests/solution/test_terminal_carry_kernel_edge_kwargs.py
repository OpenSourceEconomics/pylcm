"""A gated-edge source solves in a model that also contains an endogenous-grid regime.

Two independent branches share one lifecycle. On one branch a household chooses
consumption on a liquid asset and is solved by `EGM`. On the other a household
holds `wealth`, is solved by `GridSearch`, and reaches its terminal regime
through a `GatedEdge`: it moves only where moving beats staying, and takes the
stay value everywhere else.

Nothing connects the two branches — no transition, no same-period reference —
so the endogenous-grid household's presence must leave the gated-edge
household's value function exactly where the gated-edge branch alone puts it.

Hand computation on the `wealth` grid $\\{0, 2, 4\\}$ with $\\beta = 0.95$:

- `moved_terminal` pays $2w$, so its value is $(0, 4, 8)$.
- `stay_terminal` pays $w + 3$, so its value is $(3, 5, 7)$.
- The gate opens where moving beats staying, $2w > w + 3$: closed, closed, open.
- The gated continuation is therefore $\\bar W = (3, 5, 8)$ — the stay value at
  the two closed nodes, the move value at the open one.
- `mover` pays $w$ and carries `wealth` unchanged, so its period-0 value is
  $w + 0.95 \\bar W(w) = (2.85, 6.75, 11.6)$.

Reading the raw target value instead of the gated one would give
$(0, 5.8, 11.6)$, so the two closed nodes separate the gated fold from its
absence.
"""

import jax.numpy as jnp
import numpy as np

from lcm import AgeGrid, LinSpacedGrid, MarkovTransition, Model, categorical
from lcm.regime import EdgeLeg, GatedEdge, Regime, SamePeriodRef
from lcm.solvers import EGM, GridSearch
from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    FloatND,
    ScalarInt,
)
from tests.conftest import DECIMAL_PRECISION

DISCOUNT_FACTOR = 0.95

# Three ages, so the source regimes are active at age 0 and the terminals from
# age 1 on.
AGES = AgeGrid(start=0, stop=2, step="Y")

# The gated-edge branch's only state, shared by all three of its regimes.
WEALTH_GRID = LinSpacedGrid(start=0.0, stop=4.0, n_points=3)

# The endogenous-grid branch's grids, sized only to keep its solve cheap.
ASSET_GRID = LinSpacedGrid(start=1.0, stop=10.0, n_points=12)
SAVINGS_GRID = LinSpacedGrid(start=0.0, stop=10.0, n_points=24)
CONSUMPTION_GRID = LinSpacedGrid(start=0.05, stop=10.0, n_points=24)

# `mover`'s period-0 value function, indexed by the `wealth` grid.
MOVER_V_PERIOD_0 = (2.85, 6.75, 11.6)


@categorical(ordered=False)
class GatedRegimeId:
    """Regime ids of the gated-edge branch on its own."""

    mover: ScalarInt
    moved_terminal: ScalarInt
    stay_terminal: ScalarInt


@categorical(ordered=False)
class MixedRegimeId:
    """Regime ids of the gated-edge branch beside the endogenous-grid branch."""

    mover: ScalarInt
    moved_terminal: ScalarInt
    stay_terminal: ScalarInt
    saver: ScalarInt
    saver_terminal: ScalarInt


def test_gated_edge_source_solves_beside_an_endogenous_grid_regime():
    """A gated-edge source keeps its value when the model also runs `EGM`.

    The endogenous-grid branch neither transitions into nor references the
    gated-edge branch, so `mover`'s value function is the one the gated-edge
    branch produces on its own.
    """
    model = Model(
        regimes=_make_mixed_regimes(),
        ages=AGES,
        regime_id_class=MixedRegimeId,
    )
    solution = model.solve(
        params={"discount_factor": DISCOUNT_FACTOR}, log_level="debug"
    )
    np.testing.assert_allclose(
        np.asarray(solution[0]["mover"]),
        np.array(MOVER_V_PERIOD_0),
        rtol=10.0**-DECIMAL_PRECISION,
    )


def test_gated_edge_source_solves_on_its_own():
    """The gated-edge branch alone folds the move value against the stay value.

    Fixes the value the mixed model has to reproduce, and shows the gate is
    genuinely closed at the two low-wealth nodes.
    """
    model = Model(
        regimes=_make_gated_regimes(),
        ages=AGES,
        regime_id_class=GatedRegimeId,
    )
    solution = model.solve(
        params={"discount_factor": DISCOUNT_FACTOR}, log_level="debug"
    )
    np.testing.assert_allclose(
        np.asarray(solution[0]["mover"]),
        np.array(MOVER_V_PERIOD_0),
        rtol=10.0**-DECIMAL_PRECISION,
    )


def _make_gated_regimes() -> dict[str, Regime]:
    """Build the gated-edge branch: `mover`, `moved_terminal`, `stay_terminal`."""
    mover = Regime(
        transition={"moved_terminal": MarkovTransition(_prob_one)},
        active=lambda age: age < 1,
        states={"wealth": WEALTH_GRID},
        state_transitions={"wealth": _next_wealth},
        functions={"utility": _mover_utility},
        gated_edges={
            "moved_terminal": GatedEdge(
                gate=_move_gate,
                legs={
                    "own": EdgeLeg(
                        fallback=SamePeriodRef(
                            regime="stay_terminal",
                            projection={"wealth": _identity_wealth},
                        ),
                    )
                },
                gate_refs={
                    "V_stay_ref": SamePeriodRef(
                        regime="stay_terminal",
                        projection={"wealth": _identity_wealth},
                    )
                },
            )
        },
        solver=GridSearch(),
    )
    moved_terminal = Regime(
        transition=None,
        active=lambda age: age >= 1,
        states={"wealth": WEALTH_GRID},
        functions={"utility": _moved_terminal_utility},
        solver=GridSearch(),
    )
    stay_terminal = Regime(
        transition=None,
        active=lambda age: age >= 1,
        states={"wealth": WEALTH_GRID},
        functions={"utility": _stay_terminal_utility},
        solver=GridSearch(),
    )
    return {
        "mover": mover,
        "moved_terminal": moved_terminal,
        "stay_terminal": stay_terminal,
    }


def _make_mixed_regimes() -> dict[str, Regime]:
    """Add an unconnected endogenous-grid branch to the gated-edge branch."""
    saver = Regime(
        transition={"saver_terminal": MarkovTransition(_prob_one)},
        active=lambda age: age < 1,
        states={"assets": ASSET_GRID},
        state_transitions={"assets": _next_assets},
        actions={"consumption": CONSUMPTION_GRID},
        constraints={"affordable": _affordable},
        functions={"utility": _saver_utility, "savings": _savings},
        solver=EGM(savings_grid=SAVINGS_GRID, post_decision_function="savings"),
    )
    saver_terminal = Regime(
        transition=None,
        active=lambda age: age >= 1,
        states={"assets": ASSET_GRID},
        functions={"utility": _saver_terminal_utility},
        solver=GridSearch(),
    )
    return _make_gated_regimes() | {
        "saver": saver,
        "saver_terminal": saver_terminal,
    }


def _prob_one(age: FloatND) -> FloatND:
    """Regime transition taken with probability one."""
    return jnp.ones_like(age, dtype=float)


def _mover_utility(wealth: ContinuousState) -> FloatND:
    """The mover consumes its wealth in the period it decides."""
    return wealth


def _next_wealth(wealth: ContinuousState) -> ContinuousState:
    """Wealth is carried into the terminal regime unchanged."""
    return wealth


def _moved_terminal_utility(wealth: ContinuousState) -> FloatND:
    """Moving doubles wealth, which is worth it only at the top node."""
    return 2.0 * wealth


def _stay_terminal_utility(wealth: ContinuousState) -> FloatND:
    """Staying keeps wealth and adds a flat premium."""
    return wealth + 3.0


def _identity_wealth(wealth: ContinuousState) -> ContinuousState:
    """The stay regime is read at the same wealth the target grid names."""
    return wealth


def _move_gate(V_target: FloatND, V_stay_ref: FloatND) -> BoolND:
    """The household moves only where the move value beats the stay value."""
    return V_target > V_stay_ref


def _saver_utility(consumption: ContinuousAction) -> FloatND:
    """Log utility of consumption."""
    return jnp.log(consumption)


def _savings(assets: ContinuousState, consumption: ContinuousAction) -> FloatND:
    """The post-decision balance the asset law of motion is written through."""
    return assets - consumption


def _next_assets(savings: FloatND) -> ContinuousState:
    """Assets carry forward at a gross return of one."""
    return savings


def _affordable(assets: ContinuousState, consumption: ContinuousAction) -> BoolND:
    """The household cannot consume more than it holds."""
    return consumption <= assets


def _saver_terminal_utility(assets: ContinuousState) -> FloatND:
    """The last period consumes everything on hand."""
    return jnp.log(assets)
