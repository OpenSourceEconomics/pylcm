import functools
from typing import Any, Literal

from jax import numpy as jnp

from _lcm.grids import DiscreteGrid, LinSpacedGrid, categorical
from lcm import (
    DeterministicTransition,
    LogNormalIIDProcess,
    NormalIIDProcess,
    RouwenhorstAR1Process,
    TauchenAR1Process,
    Transition,
    UniformIIDProcess,
)
from lcm.ages import AgeGrid
from lcm.execution import ExecutionConfig
from lcm.model import Model
from lcm.regime import Regime as UserRegime
from lcm.regime import StochasticTransition
from lcm.typing import (
    ContinuousAction,
    ContinuousState,
    DiscreteState,
    FloatND,
    ScalarInt,
    UserParams,
)

_SHOCK_GRID_CLASSES = {
    "uniform": UniformIIDProcess,
    "normal": NormalIIDProcess,
    "lognormal": LogNormalIIDProcess,
    "tauchen": TauchenAR1Process,
    "rouwenhorst": RouwenhorstAR1Process,
}

# Heterogeneous per-class constructor kwargs, splatted with `**`, so the value type
# is genuinely `Any`: a checker must assume any key of the target class could receive
# one. (It typed as `bool` only by luck — every param used to accept a bool, since
# `bool` is an `int`.)
_SHOCK_GRID_KWARGS: dict[str, dict[str, Any]] = {
    "uniform": {},
    "normal": {"gauss_hermite": True},
    "lognormal": {"gauss_hermite": True},
    "tauchen": {"gauss_hermite": True},
    "rouwenhorst": {},
}


def next_health(*, health: DiscreteState, probs_array: FloatND) -> FloatND:
    return probs_array[health]


def next_wealth(*, consumption: ContinuousAction, wealth: ContinuousState) -> FloatND:
    return wealth - consumption


def next_regime(*, age: float, final_age_alive: float) -> ScalarInt:
    return jnp.where(
        age >= final_age_alive,
        RegimeId.dead,
        RegimeId.alive,
    )


def wealth_constraint(
    *, wealth: ContinuousState, income: ContinuousState, consumption: ContinuousAction
):
    return wealth - consumption + jnp.exp(income) >= 0


def utility(
    *,
    wealth: ContinuousState,  # noqa: ARG001
    income: ContinuousState,  # noqa: ARG001
    health: DiscreteState,
    consumption: ContinuousAction,
) -> FloatND:
    return jnp.log(consumption) * (1.0 - (1.0 - health) * 0.3)


@categorical(ordered=True)
class Health:
    bad: ScalarInt
    good: ScalarInt


@categorical(ordered=False)
class RegimeId:
    alive: ScalarInt
    dead: ScalarInt


@functools.cache
def get_model(
    *,
    n_periods: int,
    distribution_type: Literal[
        "uniform", "normal", "lognormal", "tauchen", "rouwenhorst"
    ],
):
    final_age_alive = n_periods - 2
    # The alive regime can both stay and die only when it lives past age 0.
    has_law = final_age_alive > 0

    alive = UserRegime(
        states={
            "wealth": LinSpacedGrid(start=1, stop=5, n_points=5),
            "income": _SHOCK_GRID_CLASSES[distribution_type](
                n_points=5, **_SHOCK_GRID_KWARGS[distribution_type]
            ),
            "health": DiscreteGrid(category_class=Health),
        },
        state_transitions={
            "wealth": next_wealth,
            "health": StochasticTransition(func=next_health),
        },
        actions={
            "consumption": LinSpacedGrid(start=0.1, stop=2, n_points=4),
        },
        constraints={"wealth_constraint": wealth_constraint},
        functions={"utility": utility},
    )
    dead = UserRegime(
        functions={"utility": lambda: 0.0},
    )
    return Model(
        regimes={"alive": alive, "dead": dead},
        regime_id_class=RegimeId,
        ages=AgeGrid(start=0, inclusive_stop=n_periods - 1, step="Y"),
        fixed_params={"final_age_alive": final_age_alive} if has_law else {},
        initial_nodes={0: "alive"},
        edges={
            "alive": Transition(
                targets={
                    "alive": tuple(range(final_age_alive)),
                    "dead": tuple(range(final_age_alive + 1)),
                },
                law=DeterministicTransition(func=next_regime),
            )
            if has_law
            else {"dead": tuple(range(final_age_alive + 1))}
        },
    )


@categorical(ordered=False)
class MultiRegimeId:
    work: ScalarInt
    retire: ScalarInt
    dead: ScalarInt


def get_multi_regime_model(
    *,
    n_periods: int,
    distribution_type: Literal[
        "uniform", "normal", "lognormal", "tauchen", "rouwenhorst"
    ],
    execution_config: ExecutionConfig | None = None,
) -> Model:
    """Create a model with two non-terminal regimes that each have shock grids.

    Triggers cross-regime shock transitions (work → retire), which is the
    scenario that fails when shock stubs leak across regime boundaries. Under
    the default execution config the model is built once per argument pair.
    """
    if execution_config is None:
        return _get_default_multi_regime_model(
            n_periods=n_periods, distribution_type=distribution_type
        )
    work_final_age = n_periods // 2 - 1
    retire_final_age = n_periods - 2

    shock_grid_cls = _SHOCK_GRID_CLASSES[distribution_type]
    shock_kwargs = _SHOCK_GRID_KWARGS[distribution_type]

    # `work` and `retire` share one declaration but are built separately, so the
    # two regimes hold distinct regime and grid objects across the shock handoff.
    living_regimes = {
        name: UserRegime(
            states={
                "wealth": LinSpacedGrid(start=1, stop=5, n_points=5),
                "income": shock_grid_cls(n_points=5, **shock_kwargs),
                "health": DiscreteGrid(category_class=Health),
            },
            state_transitions={
                "wealth": next_wealth,
                "health": StochasticTransition(func=next_health),
            },
            actions={
                "consumption": LinSpacedGrid(start=0.1, stop=2, n_points=4),
            },
            constraints={"wealth_constraint": wealth_constraint},
            functions={"utility": utility},
        )
        for name in ("work", "retire")
    }
    dead_regime = UserRegime(
        functions={"utility": lambda: 0.0},
    )
    return Model(
        regimes={**living_regimes, "dead": dead_regime},
        regime_id_class=MultiRegimeId,
        ages=AgeGrid(start=0, inclusive_stop=n_periods - 1, step="Y"),
        initial_nodes={0: "work"},
        edges={
            "work": {
                **(
                    {"work": tuple(range(work_final_age))} if work_final_age > 0 else {}
                ),
                "retire": work_final_age,
            },
            "retire": {
                **(
                    {"retire": tuple(range(work_final_age + 1, retire_final_age))}
                    if retire_final_age > work_final_age + 1
                    else {}
                ),
                "dead": retire_final_age,
            },
        },
        execution_config=execution_config,
    )


@functools.cache
def _get_default_multi_regime_model(
    *,
    n_periods: int,
    distribution_type: Literal[
        "uniform", "normal", "lognormal", "tauchen", "rouwenhorst"
    ],
) -> Model:
    """Return the multi-regime model under the default execution config, cached."""
    return get_multi_regime_model(
        n_periods=n_periods,
        distribution_type=distribution_type,
        execution_config=ExecutionConfig(),
    )


def get_multi_regime_params(
    distribution_type: Literal[
        "uniform", "normal", "lognormal", "tauchen", "rouwenhorst"
    ] = "tauchen",
) -> UserParams:
    """Return parameter dict for the multi-regime shock model."""
    shock_params = _SHOCK_PARAMS[distribution_type]
    health_probs = jnp.full((2, 2), fill_value=0.5)
    regime_params = {
        "discount_factor": 1.0,
        "next_health": {"probs_array": health_probs},
        "income": shock_params,
    }
    return {
        "work": regime_params,
        "retire": regime_params,
        "dead": {},
    }


_SHOCK_PARAMS: dict[str, dict[str, float]] = {
    "uniform": {"start": 0.0, "stop": 1.0},
    "normal": {"mu": 0.0, "sigma": 1.0},
    "lognormal": {"mu": 0.0, "sigma": 1.0},
    "tauchen": {"rho": 0.975, "sigma": 1.0, "mu": 0.0},
    "rouwenhorst": {"rho": 0.975, "sigma": 1.0, "mu": 0.0},
}


def get_params(
    distribution_type: Literal[
        "uniform", "normal", "lognormal", "tauchen", "rouwenhorst"
    ] = "tauchen",
):
    return {
        "alive": {
            "discount_factor": 1.0,
            "next_health": {"probs_array": jnp.full((2, 2), fill_value=0.5)},
            "income": _SHOCK_PARAMS[distribution_type],
        },
        "dead": {},
    }
