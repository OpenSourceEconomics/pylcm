import dataclasses
from types import MappingProxyType

import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_array_almost_equal as aaae

from lcm.ages import AgeGrid
from lcm.interfaces import StateActionSpace
from lcm.logging import get_logger
from lcm.max_Q_over_a import get_max_Q_over_a
from lcm.ndimage import map_coordinates
from lcm.solution.solve_brute import solve
from lcm.typing import MaxQOverAFunction


@dataclasses.dataclass(frozen=True)
class InternalRegimeMock:
    """Mock InternalRegime with only the attributes required by solve().

    The solve() function only accesses:
    - state_action_spaces: StateActionSpace object
    - max_Q_over_a_functions: dict mapping period to max_Q_over_a function
    - active: list of periods the regime is active
    """

    state_action_spaces: StateActionSpace
    max_Q_over_a_functions: dict[int, MaxQOverAFunction]
    active_periods: list[int]


def test_solve_brute():
    """Test solve brute with hand written inputs.

    Normally, these inputs would be created from a model specification. For now this can
    be seen as reference of what the functions that process a model specification need
    to produce.

    """
    # ==================================================================================
    # create the params
    # ==================================================================================
    params = {"discount_factor": 0.9}

    # ==================================================================================
    # create the list of state_action_spaces
    # ==================================================================================
    state_action_space = StateActionSpace(
        discrete_actions=MappingProxyType(
            {
                # pick [0, 1] such that no label translation is needed
                # lazy is like a type, it influences utility but is not affected
                # by actions
                "lazy": jnp.array([0, 1]),
                "working": jnp.array([0, 1]),
            }
        ),
        continuous_actions=MappingProxyType(
            {
                "consumption": jnp.array([0, 1, 2, 3]),
            }
        ),
        states=MappingProxyType(
            {
                # pick [0, 1, 2] such that no coordinate mapping is needed
                "wealth": jnp.array([0.0, 1.0, 2.0]),
            }
        ),
        states_and_discrete_actions_names=("lazy", "working", "wealth"),
    )
    # ==================================================================================
    # create the Q_and_F functions
    # ==================================================================================

    def _Q_and_F(consumption, lazy, wealth, working, next_V_arr, params):
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

        Q_arr = U_arr + params["discount_factor"] * expected_V

        return Q_arr, F_arr

    max_Q_over_a = get_max_Q_over_a(
        Q_and_F=_Q_and_F,
        actions_names=("consumption", "working"),
        states_names=("lazy", "wealth"),
    )

    # ==================================================================================
    # call solve function
    # ==================================================================================

    internal_regime = InternalRegimeMock(
        state_action_spaces=state_action_space,
        max_Q_over_a_functions={0: max_Q_over_a, 1: max_Q_over_a},
        active_periods=[0, 1],
    )

    solution = solve(
        params=params,
        ages=AgeGrid(start=0, stop=2, step="Y"),
        internal_regimes={"default": internal_regime},  # ty: ignore[invalid-argument-type]
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
        discrete_actions=MappingProxyType(
            {
                "a": jnp.array([0, 1.0]),
                "b": jnp.array([2, 3.0]),
                "c": jnp.array([4, 5, 6]),
            }
        ),
        continuous_actions=MappingProxyType(
            {
                "d": jnp.arange(12.0),
            }
        ),
        states=MappingProxyType({}),
        states_and_discrete_actions_names=("a", "b", "c"),
    )

    def _Q_and_F(a, c, b, d, next_V_arr, params):  # noqa: ARG001
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
        state_action_spaces=state_action_space,
        max_Q_over_a_functions={0: max_Q_over_a, 1: max_Q_over_a},
        active_periods=[0, 1],
    )

    got = solve(
        params={},
        ages=AgeGrid(start=0, stop=2, step="Y"),
        internal_regimes={"default": internal_regime},  # ty: ignore[invalid-argument-type]
        logger=get_logger(debug_mode=False),
    )

    # Solution is now dict[int, dict[RegimeName, FloatND]], need to extract the V_arr
    aaae(got[0]["default"], expected)
