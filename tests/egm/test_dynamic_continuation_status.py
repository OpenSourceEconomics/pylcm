"""Installed child-query summaries survive continuation aggregation."""

from functools import partial
from types import MappingProxyType

import jax
import jax.numpy as jnp
import numpy as np

from _lcm.egm.continuation import (
    ContinuationPlan,
    _accumulate_ez_partials_block,
    _BoundContinuation,
    _EZPartials,
    _joint_node_weights,
    _on_node_scale,
    _WeightedNodeBlockFold,
)
from _lcm.egm.ez_kernel import ez_invert_partials, ez_transform_partials
from _lcm.execution.reductions import WEIGHTED_EXPECTATION_REDUCTION
from _lcm.probability import is_live
from _lcm.solution.feasibility import BoundedResult, QueryInputStatus
from _lcm.typing import EconFunctionArg
from lcm.typing import FloatND, IntND, RegimeName, ScalarFloat


def _unused_probabilities(
    *_args: EconFunctionArg,
    **_kwargs: EconFunctionArg,
) -> MappingProxyType[RegimeName, FloatND]:
    return MappingProxyType({})


def test_direct_continuation_preserves_ordered_active_child_status() -> None:
    """A positive subnormal edge retains its own slot; exact-zero edges are neutral."""
    dtype = jnp.asarray(1.0).dtype
    subnormal = jnp.asarray(np.finfo(np.dtype(dtype)).smallest_subnormal, dtype=dtype)
    reads: list[str] = []

    def zeta(savings: ScalarFloat) -> BoundedResult[tuple[ScalarFloat, ScalarFloat]]:
        reads.append("zeta")
        return BoundedResult(
            output=(2.0 * savings, jnp.asarray(2.0)),
            bound_status=(),
            query_status=(
                QueryInputStatus(
                    jnp.asarray(a=True), jnp.asarray(a=False), jnp.asarray(a=False)
                ),
            ),
        )

    def alpha(savings: ScalarFloat) -> BoundedResult[tuple[ScalarFloat, ScalarFloat]]:
        reads.append("alpha")
        return BoundedResult(
            output=(3.0 * savings, jnp.asarray(3.0)),
            bound_status=(),
            query_status=(
                QueryInputStatus(
                    jnp.asarray(a=False), jnp.asarray(a=True), jnp.asarray(a=False)
                ),
            ),
        )

    def dead(savings: ScalarFloat) -> BoundedResult[tuple[ScalarFloat, ScalarFloat]]:
        reads.append("dead")
        return BoundedResult(
            output=(4.0 * savings, jnp.asarray(4.0)),
            bound_status=(),
            query_status=(
                QueryInputStatus(
                    jnp.asarray(a=True), jnp.asarray(a=True), jnp.asarray(a=True)
                ),
            ),
        )

    plan = ContinuationPlan(
        stateful_targets=("zeta", "alpha", "dead"),
        scalar_targets=(),
        child_reads=MappingProxyType({}),
        compute_regime_transition_probs=_unused_probabilities,
        post_decision_name="savings",
        query_targets=("zeta", "alpha", "dead"),
    )
    continuation = _BoundContinuation(
        plan=plan,
        regime_transition_probs={
            "zeta": jnp.asarray(1.0),
            "alpha": subnormal,
            "dead": jnp.asarray(0.0),
        },
        risk_aversion=None,
        child_readers={"zeta": zeta, "alpha": alpha, "dead": dead},
        next_regime_to_continuation=MappingProxyType({}),
        dtype=dtype,
    )
    result = continuation(jnp.asarray(3.0))
    assert isinstance(result, BoundedResult)
    np.testing.assert_allclose(result.output, [6.0, 2.0], atol=0.0, rtol=0.0)
    assert result.bound_status == ()
    np.testing.assert_array_equal(
        result.query_status,
        [[True, False, False], [False, True, False], [False, False, False]],
    )
    assert reads == ["zeta", "alpha", "dead"]


def test_ez_continuation_keeps_status_as_auxiliary_of_value_gradient() -> None:
    """A harmonic continuation and its derivative retain distinct child flags."""
    risk = jnp.asarray(2.0)

    def first(savings: ScalarFloat) -> BoundedResult[_EZPartials]:
        return BoundedResult(
            output=ez_transform_partials(
                child_values=(2.0 * savings)[None],
                child_marginals=jnp.asarray([2.0]),
                weights=jnp.asarray([1.0]),
                risk_aversion=risk,
            ),
            bound_status=(),
            query_status=(
                QueryInputStatus(
                    jnp.asarray(a=True), jnp.asarray(a=False), jnp.asarray(a=False)
                ),
            ),
        )

    def second(savings: ScalarFloat) -> BoundedResult[_EZPartials]:
        return BoundedResult(
            output=ez_transform_partials(
                child_values=(4.0 * savings)[None],
                child_marginals=jnp.asarray([4.0]),
                weights=jnp.asarray([1.0]),
                risk_aversion=risk,
            ),
            bound_status=(),
            query_status=(
                QueryInputStatus(
                    jnp.asarray(a=False), jnp.asarray(a=False), jnp.asarray(a=True)
                ),
            ),
        )

    continuation = _BoundContinuation(
        plan=ContinuationPlan(
            stateful_targets=("second", "first"),
            scalar_targets=(),
            child_reads=MappingProxyType({}),
            compute_regime_transition_probs=_unused_probabilities,
            post_decision_name="savings",
            query_targets=("second", "first"),
        ),
        regime_transition_probs={"first": jnp.asarray(0.5), "second": jnp.asarray(0.5)},
        risk_aversion=risk,
        child_readers={"first": first, "second": second},
        next_regime_to_continuation=MappingProxyType({}),
        dtype=risk.dtype,
    )

    def value_with_aux(
        savings: ScalarFloat,
    ) -> tuple[ScalarFloat, tuple[ScalarFloat, tuple[QueryInputStatus, ...]]]:
        result = continuation(savings)
        assert isinstance(result, BoundedResult)
        assert result.query_status is not None
        return result.output[0], (result.output[1], result.query_status)

    (value, (marginal, status)), gradient = jax.jit(
        jax.value_and_grad(value_with_aux, has_aux=True)
    )(jnp.asarray(3.0))
    np.testing.assert_allclose(
        [value, marginal, gradient], [8.0, 8.0 / 3.0, 8.0 / 3.0], rtol=2e-6
    )
    np.testing.assert_array_equal(status, [[False, False, True], [True, False, False]])


def test_weighted_node_scan_reduces_status_on_scaled_live_weights() -> None:
    """Live tiny products retain flags; a zero-weight padded node is neutral."""
    dtype = jnp.asarray(1.0).dtype
    small = jnp.sqrt(jnp.asarray(jnp.finfo(dtype).tiny)) / 2.0
    joint_weights, shift = _joint_node_weights(
        weight_vecs=(
            jnp.stack([jnp.asarray(0.0), small, 2.0 * small]),
            small[None],
        ),
        node_indices=(
            jnp.asarray([0, 1, 2], dtype=jnp.int32),
            jnp.asarray([0, 0, 0], dtype=jnp.int32),
        ),
    )
    np.testing.assert_array_equal(is_live(joint_weights), [False, True, True])
    np.testing.assert_array_equal(
        _on_node_scale(values=joint_weights, shift=shift), [0.0, 0.0, 0.0]
    )
    # The final tile reuses the impossible node but gives it exactly zero mass.
    indices = (jnp.asarray([[0, 1], [2, 0]], dtype=jnp.int32),)
    weights = jnp.concatenate([joint_weights, jnp.zeros(1, dtype=dtype)]).reshape(2, 2)

    def read_plain(node_indices: tuple[IntND, ...]) -> tuple[FloatND, FloatND]:
        index = node_indices[0]
        return (
            jnp.asarray([99.0, 3.0, 6.0])[index],
            jnp.asarray([99.0, 1.0, 2.0])[index],
        )

    def read_bounded(
        node_indices: tuple[IntND, ...],
    ) -> BoundedResult[tuple[FloatND, FloatND]]:
        index = node_indices[0]
        return BoundedResult(
            output=read_plain(node_indices),
            bound_status=(),
            query_status=(QueryInputStatus(index == 0, index == 1, index == 2),),
        )

    reduction = WEIGHTED_EXPECTATION_REDUCTION.bind(subnormal_is_accounted_for=True)
    empty = reduction.initialize(value_template=jnp.asarray(0.0))
    plain, _ = jax.lax.scan(
        _WeightedNodeBlockFold(read_at_nodes=read_plain, reduction=reduction),
        (empty, empty),
        (indices, weights),
    )
    assert not isinstance(plain, BoundedResult)
    np.testing.assert_allclose(
        [reduction.finalize(accumulator=part).expectation for part in plain],
        [5.0, 5.0 / 3.0],
        rtol=2e-6,
    )
    zero_status = QueryInputStatus(
        jnp.asarray(a=False), jnp.asarray(a=False), jnp.asarray(a=False)
    )
    result, _ = jax.lax.scan(
        _WeightedNodeBlockFold(read_at_nodes=read_bounded, reduction=reduction),
        BoundedResult(
            output=(empty, empty), bound_status=(), query_status=(zero_status,)
        ),
        (indices, weights),
    )
    assert isinstance(result, BoundedResult)
    np.testing.assert_allclose(
        [reduction.finalize(accumulator=part).expectation for part in result.output],
        [5.0, 5.0 / 3.0],
        rtol=2e-6,
    )
    assert result.bound_status == ()
    np.testing.assert_array_equal(result.query_status, [[False, True, True]])


def test_ez_node_scan_keeps_scaled_live_status_outside_partial_transforms() -> None:
    """Harmonic partials retain tiny-live flags and ignore zero-weight padding."""
    dtype = jnp.asarray(1.0).dtype
    small = jnp.sqrt(jnp.asarray(jnp.finfo(dtype).tiny)) / 2.0
    joint_weights, shift = _joint_node_weights(
        weight_vecs=(
            jnp.stack([jnp.asarray(0.0), small, 2.0 * small]),
            small[None],
        ),
        node_indices=(
            jnp.asarray([0, 1, 2], dtype=jnp.int32),
            jnp.asarray([0, 0, 0], dtype=jnp.int32),
        ),
    )
    np.testing.assert_array_equal(is_live(joint_weights), [False, True, True])
    np.testing.assert_array_equal(
        _on_node_scale(values=joint_weights, shift=shift), [0.0, 0.0, 0.0]
    )
    indices = (jnp.asarray([[0, 1], [2, 0]], dtype=jnp.int32),)
    weights = jnp.concatenate([joint_weights, jnp.zeros(1, dtype=dtype)]).reshape(2, 2)
    risk = jnp.asarray(2.0)
    zero = jnp.asarray(0.0)
    empty = (jnp.asarray(1.0), zero, zero, zero)

    def read_plain(node_indices: tuple[IntND, ...]) -> tuple[FloatND, FloatND]:
        index = node_indices[0]
        return (
            jnp.asarray([99.0, 3.0, 6.0])[index],
            jnp.asarray([99.0, 1.0, 2.0])[index],
        )

    def read_bounded(
        node_indices: tuple[IntND, ...],
    ) -> BoundedResult[tuple[FloatND, FloatND]]:
        index = node_indices[0]
        return BoundedResult(
            output=read_plain(node_indices),
            bound_status=(),
            query_status=(QueryInputStatus(index == 0, index == 1, index == 2),),
        )

    def invert(partials: _EZPartials) -> tuple[FloatND, FloatND]:
        return ez_invert_partials(
            certainty_equivalent=partials[0],
            weight_sum=partials[1],
            marginal_log_scale=partials[2],
            marginal_mantissa=partials[3],
            risk_aversion=risk,
        )

    plain, _ = jax.lax.scan(
        partial(
            _accumulate_ez_partials_block,
            read_at_nodes=read_plain,
            risk_aversion=risk,
            unit_probs=jnp.ones(2, dtype=dtype),
        ),
        empty,
        (indices, weights),
    )
    assert not isinstance(plain, BoundedResult)
    np.testing.assert_allclose(invert(plain), [4.5, 1.5], rtol=2e-6)
    zero_status = QueryInputStatus(
        jnp.asarray(a=False), jnp.asarray(a=False), jnp.asarray(a=False)
    )
    result, _ = jax.lax.scan(
        partial(
            _accumulate_ez_partials_block,
            read_at_nodes=read_bounded,
            risk_aversion=risk,
            unit_probs=jnp.ones(2, dtype=dtype),
        ),
        BoundedResult(output=empty, bound_status=(), query_status=(zero_status,)),
        (indices, weights),
    )
    assert isinstance(result, BoundedResult)
    np.testing.assert_allclose(invert(result.output), [4.5, 1.5], rtol=2e-6)
    assert result.bound_status == ()
    np.testing.assert_array_equal(result.query_status, [[False, True, True]])
