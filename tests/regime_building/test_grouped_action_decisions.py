"""Grouped state windows preserve the canonical streamed action decision."""

import functools
from collections.abc import Callable
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_array_equal

from _lcm.solution.action_reduction import HardMaxResult
from _lcm.solution.action_streaming import build_streaming_max_Q_over_a
from _lcm.utils.dispatchers import tiled_productmap
from lcm.typing import BoolND, FloatND, IntND


def _decision_Q_and_F(
    *, region: IntND, sector: IntND, wealth: FloatND, choice: IntND
) -> tuple[FloatND, BoolND]:
    """Expose cross-block ties, feasible negative infinity, and empty feasible sets."""
    target = (region + sector) % 3 + 0.5
    value = jnp.where(wealth == 0, -jnp.inf, -jnp.abs(choice - target))
    return value, (wealth >= 0) & (choice != 3)


def _decision_cell(
    *, region: IntND, sector: IntND, wealth: FloatND, choice: IntND
) -> HardMaxResult:
    """Publish value, global action identity, and feasibility from one scalar state."""
    streamed = build_streaming_max_Q_over_a(
        Q_and_F=_decision_Q_and_F, action_names=("choice",), block_width=3
    )
    return streamed(region=region, sector=sector, wealth=wealth, choice=choice)


@pytest.mark.parametrize("width", [1, 4, 11, 21, 40])
@pytest.mark.parametrize("untiled", [(), ("sector",)])
def test_grouped_cells_preserve_value_global_action_and_feasibility(
    *, width: int, untiled: tuple[str, ...]
) -> None:
    """Each state selects the earliest feasible winner in the supplied action order."""
    actions = [4, 1, 3, 0, 2]
    wealth_grid = [-1.0, 0.0, 0.5, 1.0, 2.0]
    values = np.full((2, 3, 5), -np.inf, dtype=np.float32)
    identities = np.full((2, 3, 5), -1, dtype=np.int32)
    feasible = np.zeros((2, 3, 5), dtype=bool)
    for region in range(2):
        for sector in range(3):
            for index, wealth in enumerate(wealth_grid):
                for action_id, action in enumerate(actions):
                    if wealth < 0 or action == 3:
                        continue
                    value = (
                        -np.inf
                        if wealth == 0
                        else -abs(action - ((region + sector) % 3 + 0.5))
                    )
                    location = (region, sector, index)
                    if not feasible[location] or value > values[location]:
                        values[location] = value
                        identities[location] = action_id
                    feasible[location] = True
    mapped = functools.partial(
        tiled_productmap(
            func=cast("Callable[..., Any]", _decision_cell),
            variables=("region", "sector", "wealth"),
            width_keyword="cell_width",
            untiled_variables=untiled,
        ),
        cell_width=width,
    )
    result = jax.jit(mapped)(
        region=jnp.arange(2, dtype=jnp.int32),
        sector=jnp.arange(3, dtype=jnp.int32),
        wealth=jnp.asarray(wealth_grid, dtype=jnp.float32),
        choice=jnp.asarray(actions, dtype=jnp.int32),
    )
    assert_array_equal(result.best_value, values)
    assert_array_equal(result.best_global_action_id, identities)
    assert_array_equal(result.any_feasible, feasible)
