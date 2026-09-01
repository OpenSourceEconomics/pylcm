"""Dense fallback for a model argument that collides with the planner width."""

import jax.numpy as jnp
from numpy.testing import assert_array_equal

from _lcm.execution.core_program import CoreProgramAware
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


def _utility(*, _lcm_action_block_width: ContinuousAction) -> FloatND:
    """Make the legal action name observable in the value function."""
    return _lcm_action_block_width


def _terminal_utility() -> FloatND:
    """Return an action-neutral terminal value."""
    return jnp.asarray(0.0)


def test_width_keyword_collision_solves_via_dense_grid_search() -> None:
    """A model argument may use the planner's private block-width spelling."""
    model = Model(
        regimes={
            "acting": Regime(
                transition=_next_regime,
                active=lambda age: age < 1,
                actions={
                    "_lcm_action_block_width": LinSpacedGrid(
                        start=1.0,
                        stop=3.0,
                        n_points=3,
                    )
                },
                functions={"utility": _utility},
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
    assert kernel.build_core_program(core_key="main", arguments={}) is None

    solution = model.solve(
        params={"acting": {"discount_factor": 0.5}},
        log_level="debug",
    )

    assert_array_equal(solution[0]["acting"], jnp.asarray(3.0))
