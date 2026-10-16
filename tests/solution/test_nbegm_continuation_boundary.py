"""NB-EGM plain continuation boundaries preserve their supported return contract."""

from types import MappingProxyType
from typing import Literal

import jax.numpy as jnp
import numpy as np
import pytest
from jax.typing import DTypeLike

from _lcm.egm import continuation as continuation_module
from _lcm.egm.carry import EGMCarry
from _lcm.egm.continuation import ContinuationPlan, _BoundContinuation
from _lcm.solution.feasibility import BoundedResult, QueryInputStatus
from _lcm.solution.nbegm import (
    _interval_rows,
    _NBEGMCellContinuation,
    _NBEGMContinuationRows,
    _NBEGMRideAlongStatics,
    _NBEGMScheduleSpec,
)
from _lcm.typing import EconFunctionArg, EconFunctionKwargs
from lcm.typing import FloatND, RegimeName, ScalarFloat, StateName


def _probabilities(
    *_args: EconFunctionArg,
    **_kwargs: EconFunctionArg,
) -> MappingProxyType[RegimeName, FloatND]:
    return MappingProxyType({"child": jnp.asarray(1.0)})


def _resources(*, liquid: FloatND) -> FloatND:
    return liquid


def _utility(*, consumption: FloatND) -> FloatND:
    return consumption


def _continuation(*, bounded: bool) -> _BoundContinuation:
    """Bind the actual shared aggregator to one explicit linear child read."""

    def child(
        savings: ScalarFloat,
    ) -> (
        tuple[ScalarFloat, ScalarFloat] | BoundedResult[tuple[ScalarFloat, ScalarFloat]]
    ):
        pair = (2.0 * savings + 5.0, jnp.asarray(2.0))
        if not bounded:
            return pair
        return BoundedResult(
            output=pair,
            bound_status=(),
            query_status=(
                QueryInputStatus(
                    jnp.asarray(a=True), jnp.asarray(a=False), jnp.asarray(a=False)
                ),
            ),
        )

    plan = ContinuationPlan(
        stateful_targets=("child",),
        scalar_targets=(),
        child_reads=MappingProxyType({}),
        compute_regime_transition_probs=_probabilities,
        post_decision_name="savings",
        query_targets=("child",) if bounded else (),
    )
    return _BoundContinuation(
        plan=plan,
        regime_transition_probs={"child": jnp.asarray(1.0)},
        risk_aversion=None,
        child_readers={"child": child},
        next_regime_to_continuation=MappingProxyType({}),
        dtype=jnp.asarray(1.0).dtype,
    )


def _cell(*, continuation: _BoundContinuation) -> _NBEGMCellContinuation:
    """Build a single plain ride cell using the actual immutable NB records."""
    statics = _NBEGMRideAlongStatics(
        sources=(),
        jump_flags_arr=jnp.asarray([], dtype=bool),
        n_jumps=0,
        publish_jump_topology=False,
        has_jump=False,
        static_jump_positions=(),
        dynamic_jumps=False,
        liquid_name="liquid",
        ride_names=(),
        state_names=("liquid",),
        continuation_reads_liquid=False,
        interval_width=0,
        branch_width=0,
        stochastic_node_width=None,
        consumption_action_name="consumption",
        utility_param_names=(),
        utility_state_names=(),
        coh_state_names=(),
        discount_param_names=(),
        discount_state_names=(),
        discount_action_names=(),
        n_intervals=1,
        n_savings=3,
        envelope_arithmetic="ordinary",
        cell_width=0,
        n_action_branches=0,
        discrete_action_names=(),
        discrete_action_codes=(),
    )
    schedule = _NBEGMScheduleSpec(
        coh_of_liquid_dag=_resources,
        coh_param_names=(),
        utility_dag=_utility,
        consumption_action_name="consumption",
        liquid_state_name="liquid",
        ride_along_state_names=(),
        liquid_axis_pos=0,
        threshold_param_names=(),
        breakpoint_kinds=(),
    )
    return _NBEGMCellContinuation(
        kwargs={"liquid": jnp.asarray([1.0, 2.0, 3.0])},
        carry=MappingProxyType({}),
        comap_bindings=MappingProxyType({}),
        savings_grid=jnp.asarray([1.0, 2.0, 3.0]),
        continuation_plan=continuation.plan,
        statics=statics,
        regime_name="parent",
        cliff_candidates=False,
        schedule_spec=schedule,
    )


@pytest.mark.parametrize("boundary", ["cell", "interval"])
@pytest.mark.parametrize("bounded", [False, True])
def test_nbegm_plain_boundary_refuses_uninstalled_query_status(
    *,
    boundary: Literal["cell", "interval"],
    bounded: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep concrete plain rows and reject query status without its NB adapter."""
    continuation = _continuation(bounded=bounded)
    specimen = continuation(jnp.asarray(1.0))
    if bounded:
        assert isinstance(specimen, BoundedResult)
        assert specimen.query_status is not None
        np.testing.assert_array_equal(specimen.query_status, [[True, False, False]])
        np.testing.assert_array_equal(specimen.output, [7.0, 2.0])
    else:
        assert not isinstance(specimen, BoundedResult)
        np.testing.assert_array_equal(specimen, [7.0, 2.0])

    def bind_fixed_continuation(
        *,
        plan: ContinuationPlan,
        combo_pool: EconFunctionKwargs,
        next_regime_to_continuation: MappingProxyType[RegimeName, EGMCarry],
        dtype: DTypeLike,
        stochastic_node_width: int | None,
        co_map_state_names: tuple[StateName, ...],
    ) -> _BoundContinuation:
        del (
            plan,
            combo_pool,
            next_regime_to_continuation,
            dtype,
            stochastic_node_width,
            co_map_state_names,
        )
        return continuation

    # Isolate only the factory seam: real NB consumers vmap the real aggregator,
    # which publishes the genuine registered BoundedResult and mandatory status.
    monkeypatch.setattr(
        continuation_module, "bind_continuation", bind_fixed_continuation
    )
    cell = _cell(continuation=continuation)

    def read_rows() -> _NBEGMContinuationRows:
        if boundary == "cell":
            return cell._cell_rows_for_pool(combo_pool={})
        return _interval_rows(
            (jnp.asarray(0.0),),
            combo_pool={},
            liquid_name="liquid",
            continuation_plan=continuation.plan,
            carry=MappingProxyType({}),
            dtype=jnp.asarray(1.0).dtype,
            stochastic_node_width=None,
            co_map_names=(),
            savings_grid=cell.savings_grid,
        )

    if bounded:
        with pytest.raises(TypeError, match=r"NB-EGM.*query status"):
            read_rows()
    else:
        rows = read_rows()
        assert not isinstance(rows, BoundedResult)
        np.testing.assert_array_equal(rows[0], [7.0, 9.0, 11.0])
        np.testing.assert_array_equal(rows[1], [2.0, 2.0, 2.0])
