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

`get_household_model` writes the same model as a household in which exactly one
member carries a positive Pareto weight and so decides.

Importable only once `lcm.taste_shocks` exists.
"""

from collections.abc import Mapping

import jax.numpy as jnp

from lcm import (
    AgeGrid,
    CollectiveUtility,
    DiscreteGrid,
    ExecutionConfig,
    LinSpacedGrid,
    Model,
    ParetoObjective,
    categorical,
)
from lcm.regime import Regime as UserRegime
from lcm.taste_shocks import ExtremeValueTasteShocks
from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    DiscreteAction,
    FloatND,
    ScalarInt,
    UserParamsNode,
)

WEALTH_GRID = LinSpacedGrid(start=1.0, stop=10.0, n_points=6, off_grid="extrapolate")
TERMINAL_WEALTH_GRID = LinSpacedGrid(
    start=0.0, stop=12.0, n_points=25, off_grid="extrapolate"
)
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
    *, consumption: ContinuousAction, work: DiscreteAction, kappa: float
) -> FloatND:
    return jnp.log(consumption) - kappa * work


def utility_done(wealth: ContinuousState) -> FloatND:
    return jnp.log(wealth + 1.0)


def next_wealth(
    *,
    wealth: ContinuousState,
    consumption: ContinuousAction,
    work: DiscreteAction,
    wage: float,
) -> ContinuousState:
    return wealth - consumption + wage * work


def budget_constraint(
    *, consumption: ContinuousAction, wealth: ContinuousState
) -> BoolND:
    return consumption <= wealth


alive = UserRegime(
    states={"wealth": WEALTH_GRID},
    state_transitions={"wealth": next_wealth},
    actions={
        "work": DiscreteGrid(category_class=WorkChoice),
        "consumption": CONSUMPTION_GRID,
    },
    constraints={"budget_constraint": budget_constraint},
    functions={"utility": utility_alive},
    taste_shocks=ExtremeValueTasteShocks(),
)

done = UserRegime(
    states={"wealth": TERMINAL_WEALTH_GRID},
    functions={"utility": utility_done},
)


def get_model(*, execution_config: ExecutionConfig | None = None) -> Model:
    return Model(
        edges={"alive": {"done": 40}},
        regimes={"alive": alive, "done": done},
        ages=AgeGrid(start=40, inclusive_stop=41, step="Y"),
        regime_id_class=ToyRegimeId,
        execution_config=ExecutionConfig()
        if execution_config is None
        else execution_config,
        initial_nodes={40: "alive"},
    )


def get_params(
    *, scale: float, discount_factor: float = 0.95
) -> dict[str, UserParamsNode]:
    return {
        "discount_factor": discount_factor,
        "alive": {
            "utility": {"kappa": KAPPA},
            "next_wealth": {"wage": WAGE},
            "taste_shocks": {"scale": scale},
        },
    }


PARTNER_WORK_TASTE = 0.5


def utility_partner(*, consumption: ContinuousAction, work: DiscreteAction) -> FloatND:
    return jnp.sqrt(consumption) + PARTNER_WORK_TASTE * work


def utility_partner_done(wealth: ContinuousState) -> FloatND:
    return jnp.sqrt(wealth)


# Member `x` holds the single's utilities, member `y` the partner's; each pair is
# (decision-period utility, terminal utility).
HOUSEHOLD_UTILITIES = {
    "x": (utility_alive, utility_done),
    "y": (utility_partner, utility_partner_done),
}


def get_household_model(*, weights: Mapping[str, float]) -> Model:
    """Return the toy as a household with the members and weights in `weights`.

    The keys of `weights` are the members, in order, each `"x"` or `"y"`. With
    `weights={"x": 1.0}` it is the single above written as a household of one.
    """
    alive_household = alive.replace(
        functions={
            "utility": CollectiveUtility(
                utilities={name: HOUSEHOLD_UTILITIES[name][0] for name in weights},
                objective=ParetoObjective(weights=weights),
            )
        },
    )
    done_household = done.replace(
        functions={
            "utility": CollectiveUtility(
                utilities={name: HOUSEHOLD_UTILITIES[name][1] for name in weights}
            )
        },
    )
    return Model(
        edges={"alive": {"done": 40}},
        regimes={"alive": alive_household, "done": done_household},
        ages=AgeGrid(start=40, inclusive_stop=41, step="Y"),
        regime_id_class=ToyRegimeId,
        initial_nodes={40: "alive"},
    )


def get_household_params(
    *, scale: float, discount_factor: float = 0.95
) -> dict[str, UserParamsNode]:
    return {
        "discount_factor": discount_factor,
        "alive": {
            "utility_x": {"kappa": KAPPA},
            "next_wealth": {"wage": WAGE},
            "taste_shocks": {"scale": scale},
        },
    }
