"""Namespace-safe planner widths for streamed GridSearch programs."""

import dataclasses
import inspect
from collections.abc import Callable
from types import MappingProxyType

import jax.numpy as jnp
import pytest
from numpy.testing import assert_array_equal

from _lcm.execution.core_program import CoreProgramAware
from _lcm.regime_building import processing
from _lcm.regime_building.collective import ParetoWeights
from _lcm.solution.contract import SolverBuildContext
from _lcm.solution.grid_search import _select_action_width_keyword
from lcm import AgeGrid, LinSpacedGrid, Model, categorical
from lcm.regime import Regime
from lcm.typing import ContinuousAction, FloatND, ScalarInt


@categorical(ordered=False)
class _RegimeId:
    acting: ScalarInt
    done: ScalarInt


def _next_regime() -> ScalarInt:
    """Move from the decision regime to its terminal target."""
    return _RegimeId.done


def _one_collision_utility(*, _lcm_action_block_width: ContinuousAction) -> FloatND:
    """Make one legal colliding action name observable in the value function."""
    return _lcm_action_block_width


def _two_collision_utility(
    *,
    _lcm_action_block_width: ContinuousAction,
    _lcm_action_block_width_1: ContinuousAction,
) -> FloatND:
    """Make two legal colliding action names observable in the value function."""
    return _lcm_action_block_width + _lcm_action_block_width_1


def _selector_model_utility(*, action: ContinuousAction) -> FloatND:
    """Return a noncolliding action for the selector's real build context."""
    return action


def _selector_weights(*, _lcm_action_block_width_2: float) -> dict[str, FloatND]:
    """Expose a Pareto parameter name without evaluating the weights."""
    return {"f": jnp.asarray(_lcm_action_block_width_2)}


def _terminal_utility() -> FloatND:
    """Return an action-neutral terminal value."""
    return jnp.asarray(0.0)


def test_width_keyword_selector_covers_every_runtime_namespace(
    *,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Q, flat-parameter, and Pareto names jointly determine the suffix."""
    seen: list[SolverBuildContext] = []
    original = processing.SolverBuildContext

    def spy(**kwargs):
        context = original(**kwargs)
        seen.append(context)
        return context

    monkeypatch.setattr(processing, "SolverBuildContext", spy)
    Model(
        regimes={
            "acting": Regime(
                transition=_next_regime,
                active=lambda age: age < 1,
                actions={
                    "action": LinSpacedGrid(start=1.0, stop=3.0, n_points=3),
                },
                functions={"utility": _selector_model_utility},
            ),
            "done": Regime(
                transition=None,
                active=lambda age: age >= 1,
                functions={"utility": _terminal_utility},
            ),
        },
        ages=AgeGrid(start=0, stop=1, step="Y"),
        regime_id_class=_RegimeId,
        enable_jit=True,
    )

    acting_context = next(
        context for context in seen if context.regime_name == "acting"
    )
    context = dataclasses.replace(
        acting_context,
        flat_param_names=frozenset({"_lcm_action_block_width_1"}),
        Q_and_F_functions=MappingProxyType({0: _one_collision_utility}),
        pareto_weights=ParetoWeights(
            compute=_selector_weights,
            declared=_selector_weights,
            arg_names=("_lcm_action_block_width_2",),
            param_names=("_lcm_action_block_width_2",),
            normalization="none",
        ),
    )

    assert _select_action_width_keyword(context=context) == "_lcm_action_block_width_3"


@pytest.mark.parametrize(
    ("utility", "actions", "expected_width_keyword", "expected_value"),
    [
        pytest.param(
            _one_collision_utility,
            {
                "_lcm_action_block_width": LinSpacedGrid(
                    start=1.0,
                    stop=3.0,
                    n_points=3,
                )
            },
            "_lcm_action_block_width_1",
            3.0,
            id="base-name-collision",
        ),
        pytest.param(
            _two_collision_utility,
            {
                "_lcm_action_block_width": LinSpacedGrid(
                    start=1.0,
                    stop=3.0,
                    n_points=3,
                ),
                "_lcm_action_block_width_1": LinSpacedGrid(
                    start=10.0,
                    stop=20.0,
                    n_points=2,
                ),
            },
            "_lcm_action_block_width_2",
            23.0,
            id="two-consecutive-collisions",
        ),
    ],
)
def test_width_keyword_collision_keeps_grid_search_streamed(
    *,
    utility: Callable[..., FloatND],
    actions: dict[str, LinSpacedGrid],
    expected_width_keyword: str,
    expected_value: float,
) -> None:
    """Planner width selection leaves colliding action inputs model-owned."""
    model = Model(
        regimes={
            "acting": Regime(
                transition=_next_regime,
                active=lambda age: age < 1,
                actions=actions,
                functions={"utility": utility},
            ),
            "done": Regime(
                transition=None,
                active=lambda age: age >= 1,
                functions={"utility": _terminal_utility},
            ),
        },
        ages=AgeGrid(start=0, stop=1, step="Y"),
        regime_id_class=_RegimeId,
        enable_jit=True,
    )

    kernel = model._regimes["acting"].solution.period_kernels[0]
    assert isinstance(kernel, CoreProgramAware)
    program = kernel.build_core_program(core_key="main", arguments={})

    assert program is not None
    assert len(program.requirements.streamable_axes) == 1
    axis = program.requirements.streamable_axes[0]
    assert axis.coordinate_names == tuple(actions)
    assert axis.width_keyword == expected_width_keyword
    assert axis.width_keyword not in actions
    assert axis.width_keyword in inspect.signature(program.function).parameters

    solution = model.solve(
        params={"acting": {"discount_factor": 0.5}},
        log_level="debug",
    )

    assert_array_equal(solution[0]["acting"], jnp.asarray(expected_value))
