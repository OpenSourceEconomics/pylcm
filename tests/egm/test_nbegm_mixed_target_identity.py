"""Equal-shaped living and terminal targets retain their canonical producer roles."""

from types import MappingProxyType

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.carry import EGMCarry
from _lcm.egm.continuation import _BoundContinuation, bind_continuation
from _lcm.regime_building.processing import _TerminalCarryPeriodKernel
from _lcm.solution.continuation_target import union_free_params
from _lcm.solution.nbegm import _RideAlongNBEGMPeriodKernel
from _lcm.typing import EconFunctionKwargs
from lcm import (
    AgeGrid,
    ByAge,
    ConsumptionSavingsRegime,
    IrregSpacedGrid,
    LiquidMargin,
    Model,
    Regime,
    StochasticTransition,
    Transition,
    categorical,
    fixed_transition,
    post_decision_lower_bound,
)
from lcm.solvers import NBEGM, GridSearch
from lcm.typing import (
    ContinuousAction,
    ContinuousState,
    FloatND,
    ScalarInt,
    UserParams,
)
from tests.conftest import EXACT_KERNEL_SKIP_REASON
from tests.solution._nbegm_direct_oracle import ride_along_kernel

pytestmark = pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)


@categorical(ordered=False)
class _RegimeId:
    parent: ScalarInt
    living: ScalarInt
    done: ScalarInt


def test_equal_shaped_living_and_terminal_targets_keep_distinct_marginals() -> None:
    """Blend genuine same-period targets using each target's own producer contract."""
    model, params = _model_and_params()
    _assert_canonical_target_identity(model=model)
    kernel, context = ride_along_kernel(
        model=model, params=params, regime_name="parent", period=0
    )
    assert isinstance(kernel, _RideAlongNBEGMPeriodKernel)
    assert kernel.regime_name == "parent"
    assert context["period"] == 0
    plan = kernel.continuation_plan
    assert set(plan.stateful_targets) == {"living", "done"}
    assert plan.scalar_targets == plan.query_targets == ()
    assert plan.risk_aversion_param_name is None
    carries = {}
    expected_values = jnp.asarray([[1.0, 8.0], [1.0, 8.0]])
    expected_grid = jnp.asarray([[1.0, 2.0], [1.0, 2.0]])
    for target in ("living", "done"):
        read = plan.child_reads[target]
        assert read.query is None
        assert read.query_target is None
        assert read.passive_state_names == ("z",)
        assert read.row_block_shape == (2,)
        assert read.query_action_names == read.discrete_state_names == ()
        carry = context["next_regime_to_continuation"][target]
        assert isinstance(carry, EGMCarry)
        assert carry.value.dtype == expected_values.dtype
        assert carry.marginal_utility.dtype == expected_values.dtype
        assert carry.marginal_utility.shape == carry.value.shape
        assert carry.query_data_for_transport() is None
        np.testing.assert_array_equal(carry.endog_grid, expected_grid)
        np.testing.assert_array_equal(carry.value, expected_values)
        np.testing.assert_array_equal(
            context["next_regime_to_V_arr"][target], expected_values.T
        )
        carries[target] = carry
    np.testing.assert_array_equal(
        carries["living"].marginal_utility, [[7.0, 7.0], [7.0, 7.0]]
    )
    np.testing.assert_array_equal(
        carries["done"].marginal_utility, [[3.0, 12.0], [3.0, 12.0]]
    )
    pool: EconFunctionKwargs = {
        **union_free_params(
            flat_params=context["flat_params"],
            regime_name="parent",
            transition_target_names=kernel.transition_target_names,
        ),
        "period": jnp.int32(0),
        "age": context["ages"].values[0],
        "liquid": jnp.asarray(2.0),
        "z": jnp.asarray(0.5),
    }
    continuation = bind_continuation(
        plan=plan,
        combo_pool=pool,
        next_regime_to_continuation=MappingProxyType(carries),
        dtype=carries["living"].value.dtype,
    )
    assert isinstance(continuation, _BoundContinuation)
    assert continuation.plan is plan
    for target in ("living", "done"):
        assert continuation.next_regime_to_continuation[target] is carries[target]
    _assert_actual_mixed_read(continuation=continuation, pool=pool)


def _assert_canonical_target_identity(*, model: Model) -> None:
    """Distinguish graph laws even when both kernels share one decorator class."""
    living, done = model._regimes["living"], model._regimes["done"]
    assert not living.terminal
    assert done.terminal
    assert not model._graph.laws["living"].terminal
    assert model._graph.laws["done"].terminal
    assert living.active_periods == (1,)
    assert done.active_periods == (1, 2)
    assert set(model.reachability.solution.targets(period=0, source="parent")) == {
        "living",
        "done",
    }
    assert model.reachability.solution.targets(period=1, source="living") == ("done",)
    assert living.solution.state_names == ("liquid", "z")
    assert done.solution.state_names == ("liquid", "z")
    assert living.solution.discrete_state_names == ()
    assert done.solution.discrete_state_names == ()
    assert isinstance(model.user_regimes["living"].solver, GridSearch)
    assert not living.solution.action_names
    assert not done.solution.action_names
    for regime in (living, done):
        kernel = regime.solution.period_kernels[1]
        assert isinstance(kernel, _TerminalCarryPeriodKernel)
        assert kernel.regime_name == regime.name


def _assert_actual_mixed_read(
    *, continuation: _BoundContinuation, pool: EconFunctionKwargs
) -> None:
    """Check both actual readers and their original half-weight continuation fold."""
    for target, expected in (
        ("living", (9.0 / 2.0, 7.0)),
        ("done", (27.0 / 8.0, 15.0 / 2.0)),
    ):
        np.testing.assert_array_equal(continuation.regime_transition_probs[target], 0.5)
        read = continuation.plan.child_reads[target]
        np.testing.assert_array_equal(
            read.euler_state_func(**pool, savings=jnp.asarray(1.5))["next_liquid"],
            1.5,
        )
        np.testing.assert_array_equal(read.resources_func(liquid=jnp.asarray(1.5)), 1.5)
        reader = continuation.child_readers[target]
        for call in (reader, jax.jit(reader)):
            np.testing.assert_array_equal(call(jnp.asarray(1.5)), expected)
    for call in (continuation, jax.jit(continuation)):
        np.testing.assert_array_equal(call(jnp.asarray(1.5)), [63.0 / 16.0, 29.0 / 4.0])


def _parent_utility(*, consumption: ContinuousAction, z: ContinuousState) -> FloatND:
    return jnp.log(consumption) + 0.0 * z


def _resources(liquid: ContinuousState) -> FloatND:
    return liquid


def _savings(*, resources: FloatND, consumption: ContinuousAction) -> FloatND:
    return resources - consumption


def _next_liquid(savings: FloatND) -> ContinuousState:
    return savings


def _identity_liquid(liquid: ContinuousState) -> ContinuousState:
    return liquid


def _zero_utility(*, liquid: ContinuousState, z: ContinuousState) -> FloatND:
    return 0.0 * liquid + 0.0 * z


def _terminal_utility(*, liquid: ContinuousState, z: ContinuousState) -> FloatND:
    return liquid**3 + 0.0 * z


def _half() -> FloatND:
    return jnp.asarray(0.5)


def _model_and_params() -> tuple[Model, UserParams]:
    """Construct a live zero-flow bridge and a terminal with identical value nodes."""
    passive = IrregSpacedGrid(points=(0.0, 1.0))
    margin = LiquidMargin(
        state="liquid",
        action="consumption",
        resources="resources",
        post_decision_state="savings",
    )
    parent = ConsumptionSavingsRegime(
        liquid=margin,
        states={"liquid": IrregSpacedGrid(points=(2.0, 3.0)), "z": passive},
        actions={"consumption": IrregSpacedGrid(points=(0.125, 0.25, 0.5, 1.0, 2.0))},
        functions={
            "utility": _parent_utility,
            "resources": _resources,
            "savings": _savings,
        },
        constraints={"borrowing": post_decision_lower_bound(margin=margin, lower=1.0)},
        state_transitions={
            "liquid": {"living": _next_liquid, "done": _next_liquid},
            "z": {"living": fixed_transition("z"), "done": fixed_transition("z")},
        },
        solver=NBEGM(savings_grid=IrregSpacedGrid(points=(1.0, 1.5, 2.0))),
    )
    child_states = {"liquid": IrregSpacedGrid(points=(1.0, 2.0)), "z": passive}
    living = Regime(
        states=child_states,
        functions={"utility": _zero_utility},
        state_transitions={"liquid": _identity_liquid, "z": fixed_transition("z")},
        solver=GridSearch(),
    )
    done = Regime(
        states=child_states,
        functions={"utility": _terminal_utility},
        solver=GridSearch(),
    )
    model = Model(
        regimes={"parent": parent, "living": living, "done": done},
        ages=AgeGrid(start=0, inclusive_stop=2, step="Y"),
        edges={
            "parent": Transition(
                targets={"living": 0, "done": 0},
                law=ByAge(
                    cases={
                        0: {
                            "living": StochasticTransition(func=_half),
                            "done": StochasticTransition(func=_half),
                        }
                    }
                ),
            ),
            "living": {"done": 1},
        },
        regime_id_class=_RegimeId,
        initial_nodes={0: "parent"},
    )
    return model, {
        "parent": {"koopmans_aggregator": {"discount_factor": 1.0}},
        "living": {"koopmans_aggregator": {"discount_factor": 1.0}},
        "done": {},
    }
