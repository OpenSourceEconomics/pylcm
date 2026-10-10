"""A small-state, large-action consumption model for action-partition tests.

One working regime chooses a discrete labor supply and a gridded consumption
level, so its action product (`N_ACTIONS` cells) dwarfs its state grid; a
terminal regime pays a bequest. Wealth is declared at model level so the
narrow continuous sharding route can spread it over devices, and an optional
preference type held fixed by `fixed_transition` makes every type an
independent continuation problem.
"""

import jax.numpy as jnp

from lcm import (
    AgeGrid,
    DiscreteGrid,
    ExecutionConfig,
    LinSpacedGrid,
    Model,
    categorical,
    fixed_transition,
)
from lcm.regime import Regime
from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    DiscreteAction,
    DiscreteState,
    FloatND,
    ScalarInt,
    UserParamsLeaf,
    UserParamsNode,
)

N_WEALTH_POINTS = 8
N_CONSUMPTION_POINTS = 21
N_ACTIONS = 2 * N_CONSUMPTION_POINTS


@categorical(ordered=False)
class RegimeId:
    working: ScalarInt
    dead: ScalarInt


@categorical(ordered=True)
class Labor:
    no: ScalarInt
    yes: ScalarInt


@categorical(ordered=False)
class PrefType:
    patient: ScalarInt
    middle: ScalarInt
    impatient: ScalarInt


def _utility(
    *,
    consumption: ContinuousAction,
    labor: DiscreteAction,
    disutility: FloatND,
) -> FloatND:
    return jnp.log(consumption) - disutility * labor


def _typed_utility(
    *,
    consumption: ContinuousAction,
    labor: DiscreteAction,
    pref_type: DiscreteState,
    disutility: FloatND,
    curvature: FloatND,
) -> FloatND:
    return consumption ** curvature[pref_type] - disutility * labor


def _next_wealth(
    *, wealth: ContinuousState, consumption: ContinuousAction, labor: DiscreteAction
) -> ContinuousState:
    return 1.05 * (wealth - consumption) + 1.5 * labor


def _borrowing(*, consumption: ContinuousAction, wealth: ContinuousState) -> BoolND:
    return consumption <= wealth


def _bequest(*, wealth: ContinuousState) -> FloatND:
    return jnp.sqrt(wealth)


def _typed_bequest(
    *, wealth: ContinuousState, pref_type: DiscreteState, bequest_weight: FloatND
) -> FloatND:
    return bequest_weight[pref_type] * jnp.sqrt(wealth)


def get_model(
    *,
    execution_config: ExecutionConfig | None = None,
    typed: bool = False,
    type_at_model_level: bool = False,
) -> Model:
    """Return the model, with a fixed preference type when `typed`.

    `type_at_model_level` declares the type once on the model rather than in
    each regime, which is what sharding it requires.
    """
    regime_type = typed and not type_at_model_level
    ages = AgeGrid(start=0, inclusive_stop=3, step="Y")
    pref_type = DiscreteGrid(category_class=PrefType)
    working = Regime(
        states={"pref_type": pref_type} if regime_type else {},
        state_transitions={
            "wealth": _next_wealth,
            **({"pref_type": fixed_transition("pref_type")} if typed else {}),
        },
        actions={
            "labor": DiscreteGrid(category_class=Labor),
            "consumption": LinSpacedGrid(
                start=0.5, stop=10.0, n_points=N_CONSUMPTION_POINTS
            ),
        },
        functions={"utility": _typed_utility if typed else _utility},
        constraints={"borrowing": _borrowing},
    )
    dead = Regime(
        states={"pref_type": pref_type} if regime_type else {},
        functions={"utility": _typed_bequest if typed else _bequest},
    )
    return Model(
        regimes={"working": working, "dead": dead},
        states={
            "wealth": LinSpacedGrid(start=1.0, stop=10.0, n_points=N_WEALTH_POINTS),
            **({"pref_type": pref_type} if typed and type_at_model_level else {}),
        },
        ages=ages,
        regime_id_class=RegimeId,
        initial_nodes={ages.exact_values[0]: "working"},
        edges={"working": {"working": (0, 1), "dead": 2}},
        execution_config=execution_config or ExecutionConfig(),
    )


def get_params(*, typed: bool = False, scale: float = 1.0) -> dict[str, UserParamsNode]:
    """Return parameters; `scale` moves the disutility of work."""
    working: dict[str, dict[str, UserParamsLeaf]] = {
        "utility": {"disutility": 0.3 * scale}
    }
    dead: dict[str, dict[str, UserParamsLeaf]] = {}
    if typed:
        working["utility"]["curvature"] = jnp.asarray([0.3, 0.5, 0.7])
        dead = {"utility": {"bequest_weight": jnp.asarray([1.0, 0.8, 0.6])}}
    return {"discount_factor": 0.95, "working": working, "dead": dead}
