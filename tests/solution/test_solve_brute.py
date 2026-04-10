import dataclasses
from types import MappingProxyType

import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_array_almost_equal as aaae

from lcm.ages import AgeGrid
from lcm.interfaces import StateActionSpace
from lcm.regime_building.max_Q_over_a import get_max_Q_over_a
from lcm.regime_building.ndimage import map_coordinates
from lcm.solution.solve_brute import solve
from lcm.typing import MaxQOverAFunction
from lcm.utils.logging import get_logger


@dataclasses.dataclass(frozen=True)
class SolveFunctionsMock:
    """Mock SolveFunctions with only max_Q_over_a."""

    max_Q_over_a: dict[int, MaxQOverAFunction]
    compute_intermediates: dict = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class InternalRegimeMock:
    """Mock InternalRegime with only the attributes required by solve().

    The solve() function only accesses:
    - _base_state_action_space: StateActionSpace object
    - state_action_space(): method returning the state-action space
    - solve_functions.max_Q_over_a: dict mapping period to max_Q_over_a function
    - active_periods: list of periods the regime is active
    """

    _base_state_action_space: StateActionSpace
    solve_functions: SolveFunctionsMock
    active_periods: list[int]

    def state_action_space(self, regime_params):  # noqa: ARG002
        return self._base_state_action_space


def test_solve_brute():
    """Test solve brute with hand written inputs.

    Normally, these inputs would be created from a model specification. For now this can
    be seen as reference of what the functions that process a model specification need
    to produce.

    """
    # ==================================================================================
    # create the params
    # ==================================================================================
    internal_params = MappingProxyType({"discount_factor": 0.9})

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
                "labor_supply": jnp.array([0, 1]),
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
        state_and_discrete_action_names=("lazy", "labor_supply", "wealth"),
    )
    # ==================================================================================
    # create the Q_and_F functions
    # ==================================================================================

    def _Q_and_F(
        consumption,
        lazy,
        wealth,
        labor_supply,
        next_regime_to_V_arr,
        discount_factor=0.9,
    ):
        next_wealth = wealth + labor_supply - consumption
        next_lazy = lazy

        # next_regime_to_V_arr is now a dict of regime names to arrays
        regime_name = "default"
        if (
            regime_name not in next_regime_to_V_arr
            or next_regime_to_V_arr[regime_name].size == 0
        ):
            # last period: next_regime_to_V_arr = {regime_name: jnp.empty(0)}
            expected_V = 0
        else:
            expected_V = map_coordinates(
                input=next_regime_to_V_arr[regime_name][next_lazy],
                coordinates=jnp.array([next_wealth]),
            )

        U_arr = consumption - 0.2 * lazy * labor_supply
        F_arr = next_wealth >= 0

        Q_arr = U_arr + discount_factor * expected_V

        return Q_arr, F_arr

    max_Q_over_a = get_max_Q_over_a(
        Q_and_F=_Q_and_F,
        action_names=("consumption", "labor_supply"),
        state_names=("lazy", "wealth"),
        batch_sizes={"lazy": 0, "wealth": 0},
    )

    # ==================================================================================
    # call solve function
    # ==================================================================================

    internal_regime = InternalRegimeMock(
        _base_state_action_space=state_action_space,
        solve_functions=SolveFunctionsMock(
            max_Q_over_a={0: max_Q_over_a, 1: max_Q_over_a},
        ),
        active_periods=[0, 1],
    )

    solution = solve(
        internal_params=MappingProxyType({"default": internal_params}),
        ages=AgeGrid(start=0, stop=2, step="Y"),
        internal_regimes={"default": internal_regime},  # ty: ignore[invalid-argument-type]
        logger=get_logger(log_level="off"),
    )

    # Solution is now MappingProxyType[int, MappingProxyType[RegimeName, FloatND]]
    assert isinstance(solution, MappingProxyType)
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
        state_and_discrete_action_names=("a", "b", "c"),
    )

    def _Q_and_F(a, c, b, d, next_regime_to_V_arr):  # noqa: ARG001
        # next_regime_to_V_arr is now a dict but not used in this test
        util = d
        feasib = d <= a + b + c
        return util, feasib

    max_Q_over_a = get_max_Q_over_a(
        Q_and_F=_Q_and_F,
        action_names=("d",),
        state_names=("a", "b", "c"),
        batch_sizes={"a": 0, "b": 0, "c": 0},
    )

    expected = np.array([[[6.0, 7, 8], [7, 8, 9]], [[7, 8, 9], [8, 9, 10]]])

    # by setting max_Q_over_a to identity, we can test that the function
    # is correctly applied to the state_action_space

    internal_regime = InternalRegimeMock(
        _base_state_action_space=state_action_space,
        solve_functions=SolveFunctionsMock(
            max_Q_over_a={0: max_Q_over_a, 1: max_Q_over_a},
        ),
        active_periods=[0, 1],
    )

    got = solve(
        internal_params=MappingProxyType({"default": MappingProxyType({})}),
        ages=AgeGrid(start=0, stop=2, step="Y"),
        internal_regimes={"default": internal_regime},  # ty: ignore[invalid-argument-type]
        logger=get_logger(log_level="off"),
    )

    # Solution is now dict[int, dict[RegimeName, FloatND]], need to extract the V_arr
    aaae(got[0]["default"], expected)
