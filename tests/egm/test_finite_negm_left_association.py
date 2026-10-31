"""Each publishable finite nested side respects the actual saving constraint."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.continuation import _blend_passive_axes, _collapse_stacked_candidates
from _lcm.egm.finite_negm_query import (
    _FiniteRowRead,
    _read_finite_native_row,
    _validate_finite_outer_rows,
)
from _lcm.egm.query_data import EGMQueryData, QueryNodeKind
from _lcm.egm.upper_envelope.segment_envelope import (
    DEFAULT_MAX_RUNS,
    OwnedEnvelope,
    refine_envelope_exact,
)
from lcm.typing import FloatND
from tests.conftest import DECIMAL_PRECISION, EXACT_KERNEL_SKIP_REASON
from tests.egm._finite_selection_component import _FiniteOuterRead, component_view

pytestmark = pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)


def _data(*, emitted: OwnedEnvelope) -> EGMQueryData:
    return EGMQueryData(
        savings=emitted.savings,
        node_kind=jnp.where(
            jnp.arange(8) < emitted.n_kept, QueryNodeKind.FINITE, QueryNodeKind.PAD
        ).astype(jnp.int32),
        link_valid=emitted.link_valid,
        owner_id=emitted.owner_id,
        side_id=emitted.side_id,
        floor_used=jnp.asarray(0.0),
        corner_continuation=jnp.asarray([-10.0]),
        corner_kind=jnp.asarray([QueryNodeKind.FINITE], dtype=jnp.int32),
        corner_interval_id=jnp.asarray([0], dtype=jnp.int32),
        corner_side_id=jnp.asarray([0], dtype=jnp.int32),
    )


def _read(*, emitted: OwnedEnvelope, marginal: FloatND) -> _FiniteRowRead:
    return _read_finite_native_row(
        coordinate=jnp.asarray(1.0),
        grid=emitted.grid,
        value=emitted.value,
        marginal=marginal,
        data=_data(emitted=emitted),
        owner_capacity=DEFAULT_MAX_RUNS,
        side_is_row_side=True,
    )


def _duplicate_with_distinct_savings() -> _FiniteRowRead:
    emitted = refine_envelope_exact(
        endog_grid=jnp.asarray([0.0, 1.0, 0.0, 2.0]),
        policy=jnp.asarray([0.0, 0.5, 0.0, 1.0]),
        value=jnp.asarray([0.0, 2.0, 0.0, 1.0]),
        savings=jnp.asarray([0.125, 0.25, 0.375, 0.5]),
        n_refined=8,
    )
    assert isinstance(emitted, OwnedEnvelope)
    assert int(emitted.n_kept) == 4
    np.testing.assert_array_equal(emitted.grid[:4], [0.0, 1.0, 1.0, 2.0])
    np.testing.assert_array_equal(emitted.value[:4], [0.0, 2.0, 0.5, 1.0])
    np.testing.assert_array_equal(emitted.owner_id[:4], [0, 0, 1, 1])
    np.testing.assert_array_equal(emitted.side_id[1:3], [-1, 1])
    np.testing.assert_array_equal(emitted.savings[1:3], [0.25, 7.0 / 16.0])
    marginal = jnp.where(
        emitted.owner_id == 0, 2.0, jnp.where(emitted.owner_id == 1, 0.5, jnp.nan)
    )
    raw = _read(emitted=emitted, marginal=marginal)
    assert bool(raw.ordinary.supported)
    assert bool(raw.left.supported)
    assert not bool(raw.ordinary.invalid_numeric | raw.left.invalid_numeric)
    np.testing.assert_array_equal(raw.ordinary.value, 0.5)
    np.testing.assert_array_equal(raw.ordinary.marginal, 0.5)
    np.testing.assert_array_equal(raw.ordinary.savings, 7.0 / 16.0)
    np.testing.assert_array_equal(raw.left.value, 2.0)
    np.testing.assert_array_equal(raw.left.marginal, 2.0)
    np.testing.assert_array_equal(raw.left.savings, 0.25)
    return raw


def _terminal_with_legal_saving() -> _FiniteRowRead:
    emitted = refine_envelope_exact(
        endog_grid=jnp.asarray([0.5, 1.0]),
        policy=jnp.asarray([0.0, 0.5]),
        value=jnp.asarray([0.0, 0.5]),
        savings=jnp.asarray([0.5, 0.5]),
        n_refined=8,
    )
    assert isinstance(emitted, OwnedEnvelope)
    assert int(emitted.n_kept) == 2
    np.testing.assert_array_equal(emitted.grid[:2], [0.5, 1.0])
    np.testing.assert_array_equal(emitted.savings[:2], [0.5, 0.5])
    marginal = jnp.where(jnp.arange(8) < emitted.n_kept, 1.0, jnp.nan)
    raw = _read(emitted=emitted, marginal=marginal)
    assert bool(raw.ordinary.supported & raw.left.supported)
    np.testing.assert_array_equal(raw.ordinary.value, 0.5)
    np.testing.assert_array_equal(raw.ordinary.marginal, 1.0)
    np.testing.assert_array_equal(raw.left.marginal, 1.0)
    np.testing.assert_array_equal(raw.left.savings, 0.5)
    return raw


def _validate(*, raw: _FiniteRowRead) -> _FiniteOuterRead:
    # Independent log corner at R1,b3/8 with retained W_B=-10,beta1.
    consumption = 5.0 / 8.0
    return component_view(
        profile=_validate_finite_outer_rows(
            raw=raw,
            queries=jnp.asarray(1.0),
            gradients=jnp.asarray(1.0),
            floor=jnp.asarray(3.0 / 8.0),
            upper_saving=jnp.asarray(2.0),
            resources=jnp.asarray(1.0),
            eligible=jnp.asarray(a=True),
            corner_value=jnp.asarray(math.log(consumption) - 10.0),
            corner_marginal=jnp.asarray(1.0 / consumption),
            supported_corner=jnp.asarray(a=True),
            finite_corner=jnp.asarray(a=True),
            invalid_corner=jnp.asarray(a=False),
            paired_marginal_read=False,
        )
    )


def test_left_record_recovers_independently_of_legal_right_saving() -> None:
    """An illegal left saving cannot lend its M to a legal right tuple."""
    raw = _duplicate_with_distinct_savings()
    assert bool(raw.ordinary.savings >= 3.0 / 8.0)
    assert bool(raw.left.savings < 3.0 / 8.0)
    candidate = _validate(raw=raw)
    assert not bool(candidate.coverage_missing)
    np.testing.assert_array_equal(candidate.value, 0.5)
    np.testing.assert_array_equal(candidate.right_marginal, 0.5)
    np.testing.assert_allclose(
        candidate.left_marginal,
        8.0 / 5.0,
        rtol=10 ** (-DECIMAL_PRECISION),
        atol=10 ** (-DECIMAL_PRECISION),
    )


def test_signed_passive_side_choice_uses_recovered_left_tuple() -> None:
    """A signed stencil cannot resurrect an inadmissible left saving record."""
    candidates = jax.tree.map(
        lambda *leaves: jnp.stack(leaves)[:, None],
        _validate(raw=_duplicate_with_distinct_savings()),
        _validate(raw=_terminal_with_legal_saving()),
    )
    np.testing.assert_array_equal(candidates.value[:, 0], [0.5, 0.5])
    np.testing.assert_array_equal(candidates.right_dead[:, 0], [False, True])
    assert not bool(jnp.any(candidates.coverage_missing))
    value, right, left, right_alive = _collapse_stacked_candidates(
        value_at_child=candidates.value,
        ordinary_marginal_at_child=candidates.right_marginal,
        left_marginal_at_child=candidates.left_marginal,
        right_dead_at_child=candidates.right_dead,
        left_dead_at_child=candidates.left_dead,
    )
    np.testing.assert_array_equal(right_alive, [1.0, 0.0])
    np.testing.assert_array_equal(right, [0.5, 0.0])
    actual_value, actual_marginal = _blend_passive_axes(
        value_at_child=value,
        marginal_arrays=(right, left, right_alive),
        child_passive_values=(jnp.asarray(1.5),),
        child_passive_grids=(jnp.asarray([0.0, 1.0]),),
        n_outer_candidates=1,
    )
    np.testing.assert_array_equal(actual_value, 0.5)
    expected = -0.5 * (8.0 / 5.0) + 1.5 * 1.0
    np.testing.assert_allclose(
        actual_marginal,
        expected,
        rtol=10 ** (-DECIMAL_PRECISION),
        atol=10 ** (-DECIMAL_PRECISION),
    )
