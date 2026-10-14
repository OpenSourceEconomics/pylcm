"""Public GridSearch solves tile state cells while preserving scalar reducers."""

from collections.abc import Mapping
from typing import TypedDict

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.execution.core_program import CoreExecutionDisposition, core_program_graph
from _lcm.grids import Grid
from _lcm.typing import ArrayTree, QAndFArg
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
from lcm.regime import FunctionEntry
from lcm.solver_api import DISSOLUTION_FLAG
from lcm.taste_shocks import ExtremeValueTasteShocks
from lcm.typing import (
    ActionName,
    BoolND,
    ContinuousAction,
    ContinuousState,
    DiscreteAction,
    FloatND,
    FunctionName,
    ScalarInt,
    StateName,
    UserFunction,
    UserParams,
    UserParamsNode,
)
from tests.conftest import assert_agrees_to_ulp


class _RegimeCommon(TypedDict, total=False):
    states: Mapping[StateName, Grid]
    actions: Mapping[ActionName, Grid]
    functions: Mapping[FunctionName, FunctionEntry]
    constraints: Mapping[FunctionName, UserFunction]
    taste_shocks: ExtremeValueTasteShocks | None


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
    common: _RegimeCommon = {
        "states": states,
        "actions": {"work": DiscreteGrid(category_class=_Work)},
        "functions": {"utility": utility},
        "constraints": {"feasible": _feasible} if kind == "collective" else {},
        "taste_shocks": ExtremeValueTasteShocks() if kind == "ev1" else None,
    }
    return Model(
        regimes={
            "acting": Regime(
                state_transitions={name: fixed_transition(name) for name in states},
                **common,
            ),
            "done": Regime(**common),
        },
        ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
        regime_id_class=_RegimeId,
        execution_config=ExecutionConfig(axis_widths={"cell": width}),
        initial_nodes={0: "acting"},
        edges={"acting": {"done": 0}},
    )


def _params(kind: str) -> UserParams:
    params: dict[str, UserParamsNode] = {"discount_factor": 0.5}
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

    def observe(self: dispatchers._TiledProductMap, **kwargs: QAndFArg) -> ArrayTree:
        width = kwargs.get(self.width_keyword, 1)
        assert isinstance(width, int)
        observed.add(width)
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
    lcm_cell_width: ContinuousAction,
    lcm_cell_width_1: ContinuousAction,
) -> FloatND:
    return wealth + 10.0 * lcm_cell_width + 100.0 * lcm_cell_width_1


def _collision_model() -> Model:
    states = {"wealth": LinSpacedGrid(start=1.0, stop=3.0, n_points=2)}
    common: _RegimeCommon = {
        "states": states,
        "actions": {
            "lcm_cell_width": LinSpacedGrid(start=1.0, stop=2.0, n_points=2),
            "lcm_cell_width_1": LinSpacedGrid(start=1.0, stop=2.0, n_points=2),
        },
        "functions": {"utility": _collision_utility},
    }
    return Model(
        regimes={
            "acting": Regime(
                state_transitions={name: fixed_transition(name) for name in states},
                **common,
            ),
            "done": Regime(**common),
        },
        ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
        regime_id_class=_RegimeId,
        execution_config=ExecutionConfig(axis_widths={"cell": 1}),
        initial_nodes={0: "acting"},
        edges={"acting": {"done": 0}},
    )


def test_cell_width_keyword_avoids_action_names() -> None:
    """Actions spelled like the cell width keyword leave it its reserved name."""
    model = _collision_model()
    program = core_program_graph(
        kernel=model._regimes["acting"].solution.period_kernels[0]
    )["main"]
    assert program.requirements.tiled_axes[0].width_keyword == "_lcm_cell_width"


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
    common: _RegimeCommon = {
        "states": states,
        "functions": {
            "utility": _single_state_utility if with_state else _constant_utility
        },
    }
    model = Model(
        regimes={
            "acting": Regime(
                state_transitions={name: fixed_transition(name) for name in states},
                **common,
            ),
            "done": Regime(**common),
        },
        ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
        regime_id_class=_RegimeId,
        initial_nodes={0: "acting"},
        edges={"acting": {"done": 0}},
    )
    program = core_program_graph(
        kernel=model._regimes["acting"].solution.period_kernels[0]
    )["main"]
    model.solve(params={"discount_factor": 0.5}, log_level="off")
    assert program.requirements.tiled_axes == ()
