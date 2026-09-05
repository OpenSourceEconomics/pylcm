import dataclasses
from types import MappingProxyType, SimpleNamespace

import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_array_almost_equal as aaae

from _lcm.engine import Regime, StateActionSpace
from _lcm.execution.core_program import (
    CoreExecutionDisposition,
    CoreExecutionRequirements,
    CoreProgram,
)
from _lcm.execution.output_layout import VALUE
from _lcm.grids import Grid
from _lcm.reachability import EdgeStatus, PhaseReachability
from _lcm.regime_building.max_Q_over_a import get_max_Q_over_a
from _lcm.regime_building.ndimage import map_coordinates
from _lcm.solution.backward_induction import _drain_V_arr_shards, solve
from _lcm.solution.contract import PeriodKernel
from _lcm.solution.grid_search import (
    _GridSearchArgumentBuilder,
    _GridSearchPeriodKernel,
)
from _lcm.typing import MaxQOverAFunction, StateOrActionName
from _lcm.utils.logging import get_logger
from lcm.ages import AgeGrid


@dataclasses.dataclass(frozen=True)
class MockSolutionPhase:
    """Mock SolutionPhase with only the attributes solve() reads."""

    period_kernels: dict[int, PeriodKernel]
    _base_state_action_space: StateActionSpace
    grids: MappingProxyType[StateOrActionName, Grid]
    state_names: tuple[StateOrActionName, ...]
    """Solve-phase state names, mirroring the real `SolutionPhase` property.

    Sizes the stored value function, so it must name exactly the states of the
    `StateActionSpace` this mock hands out — a mock that claims axes the space
    does not carry makes the V topology and the rank rule disagree.
    """
    compute_intermediates: dict = dataclasses.field(default_factory=dict)
    continuation_template: None = None
    continuation_spec: None = None
    period_state_axes: (
        MappingProxyType[int, MappingProxyType[StateOrActionName, object]] | None
    ) = None
    reachability: PhaseReachability = dataclasses.field(
        default_factory=lambda: _single_regime_reachability(n_periods=2)
    )
    """The solve-phase regime graph; one regime, reachable from itself."""

    def state_action_space(self, regime_params):  # noqa: ARG002
        return self._base_state_action_space


def _single_regime_reachability(*, n_periods: int) -> PhaseReachability:
    """The regime graph of a model whose only regime is `default`."""
    return PhaseReachability(
        n_periods=n_periods,
        active_regimes_by_period=tuple(
            frozenset({"default"}) for _period in range(n_periods)
        ),
        candidate_targets_by_source=MappingProxyType({"default": ("default",)}),
        targets_by_period=tuple(
            MappingProxyType({"default": ("default",)})
            for _period in range(n_periods - 1)
        ),
        edge_status_by_period=tuple(
            MappingProxyType({("default", "default"): EdgeStatus.CONDITIONAL})
            for _period in range(n_periods - 1)
        ),
    )


def _grid_search_period_kernels(
    *, max_Q_over_a: dict[int, MaxQOverAFunction], regime_name: str
) -> dict[int, PeriodKernel]:
    """Wrap hand-written dense cores in native GridSearch program graphs."""
    argument_builder = _GridSearchArgumentBuilder(regime_name=regime_name)
    return {
        period: _GridSearchPeriodKernel(
            _core_programs=MappingProxyType(
                {
                    "main": CoreProgram(
                        name="main",
                        function=core,
                        argument_builder=argument_builder,
                        requirements=CoreExecutionRequirements(),
                        output_roles=VALUE,
                        disposition=CoreExecutionDisposition.DENSE,
                        disposition_reason="test_dense_fixture",
                    )
                }
            )
        )
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
    - `simulation`: a namespace declaring no replay route, so the solve
      dispatches the values-only programs
    """

    def __init__(
        self,
        *,
        solution: MockSolutionPhase,
        active_periods: list[int],
    ) -> None:
        object.__setattr__(self, "solution", solution)
        object.__setattr__(self, "active_periods", active_periods)
        object.__setattr__(self, "simulation", SimpleNamespace(egm_policy_read=None))


def test_drain_V_arr_shards_flattens_immutable_return_mappings(monkeypatch):
    """The solve barrier presents every immutable mapping leaf to JAX."""
    V_0 = jnp.asarray([1.0, 2.0])
    V_1 = jnp.asarray([3.0])
    dissolution = jnp.asarray([True, False])
    drained = []

    def _record_drained_arrays(arrays):
        drained.extend(arrays)

    monkeypatch.setattr(
        "_lcm.solution.backward_induction.jax.block_until_ready",
        _record_drained_arrays,
    )

    _drain_V_arr_shards(
        solution={
            0: MappingProxyType({"working": V_0}),
            1: MappingProxyType({"retired": V_1}),
        },
        dissolution_flags={
            0: MappingProxyType({"working": dissolution}),
            1: MappingProxyType({}),
        },
    )

    assert [id(array) for array in drained] == [
        id(V_0),
        id(V_1),
        id(dissolution),
    ]


def test_backward_induction():
    """`solve` runs backward induction over hand-written engine inputs.

    The inputs a model specification would normally produce -- flat params, a
    state-action space, and a per-period `Q_and_F` -- are written out literally
    here, so this doubles as a reference for what the regime-building pipeline
    has to hand `solve`.
    """
    flat_params = MappingProxyType({"discount_factor": jnp.asarray(0.9)})

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

    def _Q_and_F(
        *,
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

    regime = MockRegime(
        solution=MockSolutionPhase(
            period_kernels=_grid_search_period_kernels(
                max_Q_over_a={0: max_Q_over_a, 1: max_Q_over_a},
                regime_name="default",
            ),
            _base_state_action_space=state_action_space,
            grids=MappingProxyType({}),
            state_names=("lazy", "wealth"),
        ),
        active_periods=[0, 1],
    )

    solution = solve(
        flat_params=MappingProxyType({"default": flat_params}),
        ages=AgeGrid(start=0, stop=2, step="Y"),
        regimes=MappingProxyType({"default": regime}),
        logger=get_logger(log_level="debug"),
        enable_jit=False,
    )

    # The value functions are an immutable period -> regime -> array mapping.
    assert isinstance(solution.value_functions, MappingProxyType)
    assert 0 in solution.value_functions
    assert 1 in solution.value_functions
    assert "default" in solution.value_functions[0]
    assert "default" in solution.value_functions[1]


def test_backward_induction_single_period_Qc_arr():
    state_action_space = StateActionSpace(
        discrete_actions=MappingProxyType({}),
        continuous_actions=MappingProxyType(
            {
                "d": jnp.arange(12.0),
            }
        ),
        states=MappingProxyType(
            {
                "a": jnp.array([0, 1.0]),
                "b": jnp.array([2, 3.0]),
                "c": jnp.array([4, 5, 6]),
            }
        ),
        state_and_discrete_action_names=("a", "b", "c"),
    )

    def _Q_and_F(*, a, c, b, d, next_regime_to_V_arr, period, age):  # noqa: ARG001
        # `next_regime_to_V_arr` is part of the kernel signature; this Q ignores it.
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
            state_names=("a", "b", "c"),
        ),
        active_periods=[0, 1],
    )

    got = solve(
        flat_params=MappingProxyType({"default": MappingProxyType({})}),
        ages=AgeGrid(start=0, stop=2, step="Y"),
        regimes=MappingProxyType({"default": regime}),
        logger=get_logger(log_level="debug"),
        enable_jit=False,
    )

    # `value_functions` is keyed by period, then by regime name.
    aaae(got.value_functions[0]["default"], expected)
