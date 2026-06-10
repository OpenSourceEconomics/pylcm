"""Two-period toy model for taste-shock tests, sized for in-test numpy references.

One decision period (log consumption, binary work choice, linear budget) followed
by a terminal bequest regime. Grids are small enough that the exact smoothed value
function can be recomputed in a test with numpy on the same grids:

- `Qc(w, d) = max_c [log(c) - kappa*d + beta * interp(V_done)(w - c + wage*d)]`
  over feasible `c <= w`,
- `V_alive(w) = scale * logsumexp_d(Qc(w, d) / scale)`.

The terminal wealth grid covers every reachable next-period wealth value so that
linear interpolation (which clamps in numpy and extrapolates in pylcm) agrees
between the reference and the solver.

Importable only once `lcm.taste_shocks` exists.
"""

import jax.numpy as jnp

from lcm import AgeGrid, DiscreteGrid, LinSpacedGrid, Model, categorical
from lcm.regime import Regime as UserRegime
from lcm.taste_shocks import ExtremeValueTasteShocks  # ty: ignore[unresolved-import]
from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    DiscreteAction,
    FloatND,
    ScalarInt,
)

WEALTH_GRID = LinSpacedGrid(start=1.0, stop=10.0, n_points=6)
TERMINAL_WEALTH_GRID = LinSpacedGrid(start=0.0, stop=12.0, n_points=25)
CONSUMPTION_GRID = LinSpacedGrid(start=0.5, stop=5.0, n_points=8)

KAPPA = 0.3
WAGE = 2.0


@categorical(ordered=False)
class ToyRegimeId:
    alive: ScalarInt
    done: ScalarInt


@categorical(ordered=True)
class WorkChoice:
    off: ScalarInt
    on: ScalarInt


def utility_alive(
    consumption: ContinuousAction, work: DiscreteAction, kappa: float
) -> FloatND:
    return jnp.log(consumption) - kappa * work


def utility_done(wealth: ContinuousState) -> FloatND:
    return jnp.log(wealth + 1.0)


def next_wealth(
    wealth: ContinuousState,
    consumption: ContinuousAction,
    work: DiscreteAction,
    wage: float,
) -> ContinuousState:
    return wealth - consumption + wage * work


def budget_constraint(consumption: ContinuousAction, wealth: ContinuousState) -> BoolND:
    return consumption <= wealth


def next_regime(age: int) -> ScalarInt:  # noqa: ARG001
    return ToyRegimeId.done


alive = UserRegime(
    transition=next_regime,
    states={"wealth": WEALTH_GRID},
    state_transitions={"wealth": next_wealth},
    actions={
        "work": DiscreteGrid(WorkChoice),
        "consumption": CONSUMPTION_GRID,
    },
    constraints={"budget_constraint": budget_constraint},
    functions={"utility": utility_alive},
    taste_shocks=ExtremeValueTasteShocks(),  # ty: ignore[unknown-argument]
    active=lambda age: age < 41,
)

done = UserRegime(
    transition=None,
    states={"wealth": TERMINAL_WEALTH_GRID},
    functions={"utility": utility_done},
)


def get_model() -> Model:
    return Model(
        regimes={"alive": alive, "done": done},
        ages=AgeGrid(start=40, stop=41, step="Y"),
        regime_id_class=ToyRegimeId,
    )


def get_params(*, scale: float, discount_factor: float = 0.95) -> dict:
    return {
        "discount_factor": discount_factor,
        "alive": {
            "utility": {"kappa": KAPPA},
            "next_wealth": {"wage": WAGE},
            "taste_shocks": {"scale": scale},
        },
    }
