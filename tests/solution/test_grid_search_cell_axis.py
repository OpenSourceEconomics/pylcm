"""Public GridSearch solves tile state cells while preserving scalar reducers."""

from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.execution.core_program import CoreExecutionDisposition, core_program_graph
from _lcm.utils import dispatchers
from lcm import (
    AgeGrid,
    CollectiveUtility,
    DiscreteGrid,
    ExecutionConfig,
    LinSpacedGrid,
    Model,
    Regime,
    categorical,
    fixed_transition,
)
from lcm.solver_api import DISSOLUTION_FLAG
from lcm.taste_shocks import ExtremeValueTasteShocks
from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    DiscreteAction,
    FloatND,
    ScalarInt,
)
from tests.conftest import assert_agrees_to_ulp


@categorical(ordered=False)
class _RegimeId:
    acting: ScalarInt
    done: ScalarInt


@categorical(ordered=True)
class _Work:
    off: ScalarInt
    on: ScalarInt


def _utility(
    *, first: ContinuousState, second: ContinuousState, work: DiscreteAction
) -> FloatND:
    return 10.0 * first + second + 20.0 * work


def _other_utility(
    *, first: ContinuousState, second: ContinuousState, work: DiscreteAction
) -> FloatND:
    return first + 2.0 * second + 10.0 * work


def _feasible(first: ContinuousState) -> BoolND:
    return first > 1.0


def _next_regime() -> ScalarInt:
    return _RegimeId.done


def _model(*, kind: str, width: int) -> Model:
    """Use two state axes and an unchanged action reducer of each supported kind."""
    utility = (
        CollectiveUtility(utilities={"f": _utility, "m": _other_utility})
        if kind == "collective"
        else _utility
    )
    states = {
        "first": LinSpacedGrid(start=1.0, stop=3.0, n_points=2),
        "second": LinSpacedGrid(start=2.0, stop=6.0, n_points=3),
    }
    common: dict[str, Any] = {
        "states": states,
        "actions": {"work": DiscreteGrid(category_class=_Work)},
        "functions": {"utility": utility},
        "constraints": {"feasible": _feasible} if kind == "collective" else {},
        "taste_shocks": ExtremeValueTasteShocks() if kind == "ev1" else None,
    }
    return Model(
        regimes={
            "acting": Regime(
                transition=_next_regime,
                active=lambda age: age < 1,
                state_transitions={name: fixed_transition(name) for name in states},
                **common,
            ),
            "done": Regime(transition=None, active=lambda age: age >= 1, **common),
        },
        ages=AgeGrid(start=0, stop=1, step="Y"),
        regime_id_class=_RegimeId,
        execution_config=ExecutionConfig(axis_widths={"cell": width}),
    )


def _params(kind: str) -> dict[str, Any]:
    params: dict[str, Any] = {"discount_factor": 0.5}
    if kind == "ev1":
        params.update(
            {name: {"taste_shocks": {"scale": 0.2}} for name in ("acting", "done")}
        )
    return params


@pytest.mark.parametrize("kind", ["singleton", "ev1", "collective"])
def test_public_solve_dispatches_the_planned_cell_width(
    *, kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The public solve reaches the flat mapper with the requested static width."""
    observed: set[int] = set()
    original = dispatchers._TiledProductMap.__call__

    def observe(self: dispatchers._TiledProductMap, **kwargs: Any) -> Any:
        observed.add(kwargs.get(self.width_keyword, 1))
        return original(self, **kwargs)

    monkeypatch.setattr(dispatchers._TiledProductMap, "__call__", observe)
    _model(kind=kind, width=4).solve(params=_params(kind), log_level="off")
    assert observed == {4}


@pytest.mark.parametrize("kind", ["singleton", "ev1", "collective"])
def test_state_tiling_leaves_action_reducer_declarations_distinct(*, kind: str) -> None:
    """Dense canonical action reductions can share the planned state-cell loop."""
    model = _model(kind=kind, width=4)
    program = core_program_graph(
        kernel=model._regimes["acting"].solution.period_kernels[0]
    )["main"]
    assert (
        program.disposition,
        program.disposition_reason,
        tuple(axis.name for axis in program.requirements.reduced_axes),
        tuple((axis.name, axis.extent) for axis in program.requirements.tiled_axes),
    ) == (
        CoreExecutionDisposition.PLANNED,
        None,
        ("action_product",) if kind == "singleton" else (),
        (("cell", 6),),
    )


@pytest.mark.parametrize("kind", ["singleton", "ev1", "collective"])
@pytest.mark.parametrize("width", [1, 4, 6])
def test_cell_width_preserves_public_values(*, kind: str, width: int) -> None:
    """Scalar, remainder, and full widths publish the same two-axis value tree."""
    reference = _model(kind=kind, width=1).solve(params=_params(kind), log_level="off")
    candidate = _model(kind=kind, width=width).solve(
        params=_params(kind), log_level="off"
    )
    assert_agrees_to_ulp(
        got=np.asarray(candidate.values[0]["acting"]),
        expected=np.asarray(reference.values[0]["acting"]),
        n_ulp=4,
    )


@pytest.mark.parametrize("width", [1, 4, 6])
def test_cell_width_preserves_exact_dissolution_flags(*, width: int) -> None:
    """The Boolean output keeps its state axes without a stakeholder dimension."""
    result = _model(kind="collective", width=width).solve(
        params=_params("collective"), log_level="off"
    )
    flags = result.replay_artifacts.project(DISSOLUTION_FLAG)
    np.testing.assert_array_equal(
        flags[0]["acting"], np.asarray([[True, True, True], [False, False, False]])
    )


def _collision_utility(
    *,
    wealth: ContinuousState,
    _lcm_cell_width: ContinuousAction,
    _lcm_cell_width_1: ContinuousAction,
) -> FloatND:
    return wealth + 10.0 * _lcm_cell_width + 100.0 * _lcm_cell_width_1


def _collision_model() -> Model:
    states = {"wealth": LinSpacedGrid(start=1.0, stop=3.0, n_points=2)}
    common: dict[str, Any] = {
        "states": states,
        "actions": {
            "_lcm_cell_width": LinSpacedGrid(start=1.0, stop=2.0, n_points=2),
            "_lcm_cell_width_1": LinSpacedGrid(start=1.0, stop=2.0, n_points=2),
        },
        "functions": {"utility": _collision_utility},
    }
    return Model(
        regimes={
            "acting": Regime(
                transition=_next_regime,
                active=lambda age: age < 1,
                state_transitions={name: fixed_transition(name) for name in states},
                **common,
            ),
            "done": Regime(transition=None, active=lambda age: age >= 1, **common),
        },
        ages=AgeGrid(start=0, stop=1, step="Y"),
        regime_id_class=_RegimeId,
        execution_config=ExecutionConfig(axis_widths={"cell": 1}),
    )


def test_cell_width_keyword_avoids_action_names() -> None:
    """Both occupied spellings stay numerical inputs to the declared program."""
    model = _collision_model()
    program = core_program_graph(
        kernel=model._regimes["acting"].solution.period_kernels[0]
    )["main"]
    assert program.requirements.tiled_axes[0].width_keyword == "_lcm_cell_width_2"


def test_colliding_cell_names_keep_their_economic_values() -> None:
    result = _collision_model().solve(params={"discount_factor": 0.5}, log_level="off")
    assert_agrees_to_ulp(
        got=np.asarray(result.values[0]["acting"]),
        expected=np.asarray([331.5, 334.5]),
        n_ulp=4,
    )


def _constant_utility() -> FloatND:
    return jnp.asarray(1.0)


def _single_state_utility(first: ContinuousState) -> FloatND:
    return first


@pytest.mark.parametrize("with_state", [False, True])
def test_trivial_state_product_does_not_declare_a_cell_axis(
    *, with_state: bool
) -> None:
    states = (
        {"first": LinSpacedGrid(start=1.0, stop=2.0, n_points=1)} if with_state else {}
    )
    common: dict[str, Any] = {
        "states": states,
        "functions": {
            "utility": _single_state_utility if with_state else _constant_utility
        },
    }
    model = Model(
        regimes={
            "acting": Regime(
                transition=_next_regime,
                active=lambda age: age < 1,
                state_transitions={name: fixed_transition(name) for name in states},
                **common,
            ),
            "done": Regime(transition=None, active=lambda age: age >= 1, **common),
        },
        ages=AgeGrid(start=0, stop=1, step="Y"),
        regime_id_class=_RegimeId,
    )
    program = core_program_graph(
        kernel=model._regimes["acting"].solution.period_kernels[0]
    )["main"]
    model.solve(params={"discount_factor": 0.5}, log_level="off")
    assert program.requirements.tiled_axes == ()
