import jax
import jax.numpy as jnp

from lcm import (
    AgeGrid,
    DiscreteGrid,
    LinSpacedGrid,
    Model,
    categorical,
    fixed_transition,
)
from lcm.execution import ExecutionConfig
from lcm.regime import Regime as UserRegime
from lcm.solvers import GridSearch, Solver
from lcm.typing import ScalarInt


@categorical(ordered=False)
class _ThreeTypeRegimeId:
    """Regime vocabulary of the three-valued-type model."""

    working: ScalarInt
    retired: ScalarInt


@categorical(ordered=True)
class _Type:
    """A three-valued preference type; its extent is the mesh size."""

    low: ScalarInt
    mid: ScalarInt
    high: ScalarInt


def _constant_retired_value(*, wealth: jax.Array, type1: jax.Array) -> jax.Array:
    """Declare both state axes without reading either in the numerical body."""
    del wealth, type1
    return jnp.asarray(2.0)


def _make_three_type_model(
    *,
    distributed: bool,
    sharded: tuple[str, ...] = (),
    devices: tuple[int, ...] | None = None,
    solver: Solver | None = None,
    budget_bytes: int | None = None,
    enable_jit: bool = True,
    constant_retired: bool = False,
) -> Model:
    """A working regime over a three-valued type beside a single-device terminal one.

    Both regimes are active before the final age and read nothing of each other
    within a period, so on four devices the working regime runs on three and the
    terminal one on the fourth. `sharded` names the same axis through
    `ExecutionConfig`; either spelling places the regime the same way.
    `devices` restricts the model to a subset of the four.
    """
    working = UserRegime(
        active=lambda age: age < 4,
        solver=GridSearch() if solver is None else solver,
        functions={
            "utility": lambda wealth, consumption, type1: (
                (jnp.log(consumption) + wealth * 0.001) * (type1 + 1)
            ),
        },
        states={"wealth": LinSpacedGrid(start=1, stop=100, n_points=12)},
        state_transitions={"wealth": lambda wealth, consumption: wealth - consumption},
        actions={"consumption": LinSpacedGrid(start=1, stop=50, n_points=10)},
        transition=lambda age: jnp.where(
            age >= 3, _ThreeTypeRegimeId.retired, _ThreeTypeRegimeId.working
        ),
    )
    retired = UserRegime(
        transition=None,
        functions={
            "utility": (
                _constant_retired_value
                if constant_retired
                else (lambda wealth: wealth * 0.5)
            )
        },
        states={"wealth": LinSpacedGrid(start=1, stop=100, n_points=12)},
    )
    return Model(
        regimes={"working": working, "retired": retired},
        ages=AgeGrid(start=0, stop=4, step="Y"),
        regime_id_class=_ThreeTypeRegimeId,
        enable_jit=enable_jit,
        states={"type1": DiscreteGrid(category_class=_Type)},
        state_transitions={"type1": fixed_transition("type1")},
        execution_config=ExecutionConfig(
            device_memory_bytes=budget_bytes,
            sharded_states=tuple(
                dict.fromkeys((*sharded, *(("type1",) if distributed else ())))
            ),
            devices=devices,
        ),
    )


_PARAMS = {"discount_factor": 0.95}
