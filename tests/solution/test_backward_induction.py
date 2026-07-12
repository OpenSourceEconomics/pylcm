import dataclasses
from types import MappingProxyType

import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_array_almost_equal as aaae

from _lcm.engine import Regime, StateActionSpace
from _lcm.grids import Grid
from _lcm.regime_building.max_Q_over_a import get_max_Q_over_a
from _lcm.regime_building.ndimage import map_coordinates
from _lcm.solution.backward_induction import solve
from _lcm.solution.contract import PeriodKernel
from _lcm.solution.solvers import _GridSearchPeriodKernel
from _lcm.typing import MaxQOverAFunction, StateOrActionName
from _lcm.utils.logging import get_logger
from lcm.ages import AgeGrid


@dataclasses.dataclass(frozen=True)
class MockSolutionPhase:
    """Mock SolutionPhase with only the attributes solve() reads."""

    period_kernels: dict[int, PeriodKernel]
    _base_state_action_space: StateActionSpace
    grids: MappingProxyType[StateOrActionName, Grid]
    compute_intermediates: dict = dataclasses.field(default_factory=dict)
    continuation_template: None = None

    def state_action_space(self, regime_params):  # noqa: ARG002
        return self._base_state_action_space


def _grid_search_period_kernels(
    *, max_Q_over_a: dict[int, MaxQOverAFunction], regime_name: str
) -> dict[int, PeriodKernel]:
    """Wrap per-period grid-search cores in their uniform period adapters."""
    return {
        period: _GridSearchPeriodKernel(core=core, regime_name=regime_name)
        for period, core in max_Q_over_a.items()
    }


class MockRegime(Regime):
    """Mock Regime with only the attributes required by solve().

    Inherits from `Regime` so `isinstance(x, Regime)` holds at
    the beartype-checked perimeter of `solve()`, but bypasses the dataclass
    `__init__` so tests can supply only the attributes `solve()` reads:
    - `solution`: a `MockSolutionPhase` with max_Q_over_a, grids, and the
      state-action space
    - `active_periods`: list of periods the regime is active
    """

    def __init__(
        self,
        *,
        solution: MockSolutionPhase,
        active_periods: list[int],
    ) -> None:
        object.__setattr__(self, "solution", solution)
        object.__setattr__(self, "active_periods", active_periods)


def test_backward_induction():
    """Test solve brute with hand written inputs.

    Normally, these inputs would be created from a model specification. For now this can
    be seen as reference of what the functions that process a model specification need
    to produce.

    """
    # ==================================================================================
    # create the params
    # ==================================================================================
    flat_params = MappingProxyType({"discount_factor": jnp.asarray(0.9)})

    # ==================================================================================
    # create the list of state_action_spaces
    # ==================================================================================
    state_action_space = StateActionSpace(
        discrete_actions=MappingProxyType(
            {
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
                # lazy is like a type, it influences utility but is not affected
                # by actions
                "lazy": jnp.array([0, 1]),
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
        period,  # noqa: ARG001
        age,  # noqa: ARG001
        discount_factor=0.9,
    ):
        next_wealth = wealth + labor_supply - consumption
        next_lazy = lazy
        # next_regime_to_V_arr always contains all regimes with proper shapes.
        # Interpolate the next-period V array at the next state.
        expected_V = map_coordinates(
            input=next_regime_to_V_arr["default"],
            coordinates=jnp.array([next_wealth, next_lazy]),
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

    regime = MockRegime(
        solution=MockSolutionPhase(
            period_kernels=_grid_search_period_kernels(
                max_Q_over_a={0: max_Q_over_a, 1: max_Q_over_a},
                regime_name="default",
            ),
            _base_state_action_space=state_action_space,
            grids=MappingProxyType({}),
        ),
        active_periods=[0, 1],
    )

    solution, _sim_policies, _dissolution_flags = solve(
        flat_params=MappingProxyType({"default": flat_params}),
        ages=AgeGrid(start=0, stop=2, step="Y"),
        regimes=MappingProxyType({"default": regime}),
        logger=get_logger(log_level="debug"),
        enable_jit=False,
    )

    # Solution is now MappingProxyType[int, MappingProxyType[RegimeName, FloatND]]
    assert isinstance(solution, MappingProxyType)
    assert 0 in solution
    assert 1 in solution
    assert "default" in solution[0]
    assert "default" in solution[1]


def test_backward_induction_single_period_Qc_arr():
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

    def _Q_and_F(a, c, b, d, next_regime_to_V_arr, period, age):  # noqa: ARG001
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

    regime = MockRegime(
        solution=MockSolutionPhase(
            period_kernels=_grid_search_period_kernels(
                max_Q_over_a={0: max_Q_over_a, 1: max_Q_over_a},
                regime_name="default",
            ),
            _base_state_action_space=state_action_space,
            grids=MappingProxyType({}),
        ),
        active_periods=[0, 1],
    )

    got, _sim_policies, _dissolution_flags = solve(
        flat_params=MappingProxyType({"default": MappingProxyType({})}),
        ages=AgeGrid(start=0, stop=2, step="Y"),
        regimes=MappingProxyType({"default": regime}),
        logger=get_logger(log_level="debug"),
        enable_jit=False,
    )

    # Solution is now dict[int, dict[RegimeName, FloatND]], need to extract the V_arr
    aaae(got[0]["default"], expected)
