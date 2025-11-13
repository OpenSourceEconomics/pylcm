from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_array_almost_equal as aaae

from lcm.interfaces import PeriodVariantContainer, StateActionSpace
from lcm.logging import get_logger
from lcm.max_Q_over_a import get_max_Q_over_a
from lcm.ndimage import map_coordinates
from lcm.solution.solve_brute import solve

if TYPE_CHECKING:
    from lcm.typing import MaxQOverAFunction


@dataclasses.dataclass(frozen=True)
class InternalRegimeMock:
    """Mock InternalRegime with only the attributes required by solve().

    The solve() function only accesses:
    - state_action_spaces: PeriodVariantContainer of StateActionSpace objects
    - max_Q_over_a_functions: PeriodVariantContainer of max_Q_over_a functions
    """

    state_action_spaces: PeriodVariantContainer[StateActionSpace]
    max_Q_over_a_functions: PeriodVariantContainer[MaxQOverAFunction]


def test_solve_brute():
    """Test solve brute with hand written inputs.

    Normally, these inputs would be created from a model specification. For now this can
    be seen as reference of what the functions that process a model specification need
    to produce.

    """
    # ==================================================================================
    # create the params
    # ==================================================================================
    params = {"beta": 0.9}

    # ==================================================================================
    # create the list of state_action_spaces
    # ==================================================================================
    state_action_space = StateActionSpace(
        discrete_actions={
            # pick [0, 1] such that no label translation is needed
            # lazy is like a type, it influences utility but is not affected by actions
            "lazy": jnp.array([0, 1]),
            "working": jnp.array([0, 1]),
        },
        continuous_actions={
            "consumption": jnp.array([0, 1, 2, 3]),
        },
        states={
            # pick [0, 1, 2] such that no coordinate mapping is needed
            "wealth": jnp.array([0.0, 1.0, 2.0]),
        },
        states_and_discrete_actions_names=("lazy", "working", "wealth"),
    )
    state_action_spaces = PeriodVariantContainer(
        terminal=state_action_space, non_terminal=state_action_space
    )
    # ==================================================================================
    # create the Q_and_F functions
    # ==================================================================================

    def _Q_and_F(consumption, lazy, wealth, working, next_V_arr, params, period):  # noqa: ARG001
        next_wealth = wealth + working - consumption
        next_lazy = lazy

        # next_V_arr is now a dict of regime names to arrays
        regime_name = "default"
        if regime_name not in next_V_arr or next_V_arr[regime_name].size == 0:
            # this is the last period, when next_V_arr = {regime_name: jnp.empty(0)}
            expected_V = 0
        else:
            expected_V = map_coordinates(
                input=next_V_arr[regime_name][next_lazy],
                coordinates=jnp.array([next_wealth]),
            )

        U_arr = consumption - 0.2 * lazy * working
        F_arr = next_wealth >= 0

        Q_arr = U_arr + params["beta"] * expected_V

        return Q_arr, F_arr

    max_Q_over_a = get_max_Q_over_a(
        Q_and_F=_Q_and_F,
        actions_names=("consumption", "working"),
        states_names=("lazy", "wealth"),
    )

    max_Q_over_a_functions = PeriodVariantContainer(
        terminal=max_Q_over_a, non_terminal=max_Q_over_a
    )

    # ==================================================================================
    # call solve function
    # ==================================================================================

    internal_regime = InternalRegimeMock(
        state_action_spaces=state_action_spaces,
        max_Q_over_a_functions=max_Q_over_a_functions,
    )

    solution = solve(
        params=params,
        n_periods=2,
        internal_regimes={"default": internal_regime},  # type: ignore[dict-item]
        logger=get_logger(debug_mode=False),
    )

    # Solution is now dict[int, dict[RegimeName, FloatND]]
    assert isinstance(solution, dict)
    assert 0 in solution
    assert 1 in solution
    assert "default" in solution[0]
    assert "default" in solution[1]


def test_solve_brute_single_period_Qc_arr():
    state_action_space = StateActionSpace(
        discrete_actions={
            "a": jnp.array([0, 1.0]),
            "b": jnp.array([2, 3.0]),
            "c": jnp.array([4, 5, 6]),
        },
        continuous_actions={
            "d": jnp.arange(12.0),
        },
        states={},
        states_and_discrete_actions_names=("a", "b", "c"),
    )
    state_action_spaces = PeriodVariantContainer(
        terminal=state_action_space, non_terminal=state_action_space
    )

    def _Q_and_F(a, c, b, d, next_V_arr, params, period):  # noqa: ARG001
        # next_V_arr is now a dict but not used in this test
        util = d
        feasib = d <= a + b + c
        return util, feasib

    max_Q_over_a = get_max_Q_over_a(
        Q_and_F=_Q_and_F,
        actions_names=("d",),
        states_names=("a", "b", "c"),
    )

    expected = np.array([[[6.0, 7, 8], [7, 8, 9]], [[7, 8, 9], [8, 9, 10]]])

    # by setting max_Qc_over_d to identity, we can test that the max_Q_over_c function
    # is correctly applied to the state_action_space

    internal_regime = InternalRegimeMock(
        state_action_spaces=state_action_spaces,
        max_Q_over_a_functions=PeriodVariantContainer(
            terminal=max_Q_over_a, non_terminal=max_Q_over_a
        ),
    )

    got = solve(
        params={},
        n_periods=2,
        internal_regimes={"default": internal_regime},  # type: ignore[dict-item]
        logger=get_logger(debug_mode=False),
    )

    # Solution is now dict[int, dict[RegimeName, FloatND]], need to extract the V_arr
    aaae(got[0]["default"], expected)
