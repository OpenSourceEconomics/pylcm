"""Native binding ownership preserves the literal envelope geometry."""

from collections.abc import Sequence
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.typing import ArrayLike

from _lcm.egm.interp import interp_on_padded_grid
from _lcm.egm.step_core import _refined_rows
from _lcm.egm.upper_envelope import _exact_backend, segment_envelope
from lcm.solvers import ExactEnvelope
from lcm.typing import Float1D, ScalarInt
from tests.conftest import EXACT_KERNEL_SKIP_REASON
from tests.solution.test_segment_envelope import (
    F1_GRID,
    F1_POLICY,
    F1_VALUE,
    _tolerance,
)

pytestmark = pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)

type _NumericalInput = ArrayLike | Sequence[ArrayLike]


def _bits_equal(*, actual: _NumericalInput, expected: _NumericalInput) -> None:
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert actual.tobytes() == expected.tobytes()


def _refine(
    *,
    grid: _NumericalInput,
    policy: _NumericalInput,
    value: _NumericalInput,
    floor_sources: int,
    queries: _NumericalInput,
    capacity: int = 64,
    max_runs: int = 24,
) -> segment_envelope._NegmBindingEnvelope:
    """Compare the same exact native emission with classification off and on."""
    grid, policy, value = jnp.asarray(grid), jnp.asarray(policy), jnp.asarray(value)
    ordinary = jax.jit(
        lambda g, p, v: segment_envelope.refine_envelope_exact(
            endog_grid=g,
            policy=p,
            value=v,
            n_refined=capacity,
            max_runs=max_runs,
        )
    )(grid, policy, value)
    assert type(ordinary) is tuple
    assert len(ordinary) == 4
    ordinary = cast("tuple[Float1D, Float1D, Float1D, ScalarInt]", ordinary)
    # Private floor ownership uses the same emission without owned savings rows.
    classified = jax.jit(
        lambda g, p, v, q: segment_envelope.refine_envelope_exact(
            endog_grid=g,
            policy=p,
            value=v,
            n_refined=capacity,
            max_runs=max_runs,
            _binding_source_count=floor_sources,
            _binding_queries=q,
        )
    )(grid, policy, value, jnp.asarray(queries))
    assert not isinstance(classified, segment_envelope.OwnedEnvelope)
    assert isinstance(classified, segment_envelope._NegmBindingEnvelope)
    for actual, expected in zip(classified[:4], ordinary, strict=True):
        _bits_equal(actual=actual, expected=expected)
    assert len(classified) == 5
    assert not hasattr(classified, "binding_node")
    assert not hasattr(classified, "binding_link")
    assert classified.binding_query.shape == jnp.asarray(queries).shape
    assert classified.binding_query.dtype == jnp.bool_
    return classified


def test_floor_source_links_and_their_queries_are_certified() -> None:
    """Actual source identities classify the whole floor-owned chain."""
    row = _refine(
        grid=[1.0, 2.0, 3.0],
        policy=[0.5, 1.0, 1.5],
        value=[0.0, 1.0, 1.5],
        floor_sources=3,
        queries=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, np.nan],
    )
    np.testing.assert_array_equal(
        row.binding_query, [False, True, True, True, True, True, False, False]
    )


def test_a_floor_euler_endpoint_does_not_make_its_interior_link_binding() -> None:
    """One floor source followed by positive saving leaves a mixed link."""
    row = _refine(
        grid=[1.0, 2.0, 3.0],
        policy=[0.5, 1.0, 1.5],
        value=[0.0, 1.0, 1.5],
        floor_sources=2,
        queries=[1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
    )
    np.testing.assert_array_equal(
        row.binding_query, [True, True, True, False, False, False]
    )


def test_a_nonconcave_winning_interior_source_stays_interior() -> None:
    """The middle branch's true ownership survives a floor branch on its left."""
    row = _refine(
        grid=F1_GRID,
        policy=F1_POLICY,
        value=F1_VALUE,
        floor_sources=2,
        queries=[0.2, 0.8, 1.1],
    )
    np.testing.assert_array_equal(row.binding_query, [True, False, False])
    # The independent existing rational witness gives B's policy3 at q=.8.
    np.testing.assert_allclose(
        interp_on_padded_grid(x_query=jnp.asarray(0.8), xp=row.grid, fp=row.policy),
        3.0,
        rtol=0.0,
        atol=_tolerance(),
    )


def test_duplicate_crossings_use_the_original_readers_right_side() -> None:
    """The incoming interior owner wins the exact duplicated crossing."""
    point = jnp.asarray(1.0)
    left, right = jnp.nextafter(point, -jnp.inf), jnp.nextafter(point, jnp.inf)
    row = _refine(
        grid=[0.0, 2.0, 0.0, 2.0],
        policy=[10.0, 12.0, 20.0, 22.0],
        value=[2.0, 0.0, 0.0, 2.0],
        floor_sources=2,
        queries=jnp.asarray([0.0, 0.5, left, 1.0, right, 1.5, 2.0]),
        capacity=16,
    )
    np.testing.assert_array_equal(row.grid[:4], [0.0, 1.0, 1.0, 2.0])
    np.testing.assert_array_equal(
        row.binding_query, [True, True, True, False, False, False, False]
    )


def test_coincident_value_and_policy_do_not_erase_the_source_handover() -> None:
    """An outgoing binding link can end at an incoming nonbinding node."""
    point = jnp.asarray(1.0)
    left = jnp.nextafter(point, -jnp.inf)
    row = _refine(
        grid=[0.0, 2.0, 0.0, 2.0],
        policy=[1.0, 1.0, 1.0, 1.0],
        value=[2.0, 0.0, 0.0, 2.0],
        floor_sources=2,
        queries=jnp.asarray([0.0, 0.5, left, 1.0, 1.5, 2.0]),
        capacity=16,
    )
    assert int(row.n_kept) == 3
    np.testing.assert_array_equal(row.grid[:3], [0.0, 1.0, 2.0])
    np.testing.assert_array_equal(
        row.binding_query, [True, True, True, False, False, False]
    )


def test_a_compacted_gap_keeps_the_owned_query_and_rejects_the_gap_query() -> None:
    """Lost legacy gap endpoints cannot make an owned query indistinguishable."""
    row = _refine(
        grid=[0.0, 1.0, np.nan, 3.0, 4.0],
        policy=[10.0, 11.0, np.nan, 13.0, 14.0],
        value=[0.0, 1.0, np.nan, 3.0, 4.0],
        floor_sources=5,
        queries=[-0.5, 0.5, 1.0, 2.0, 3.0, 3.5, 4.0, 4.5],
        capacity=16,
    )
    # Legacy emission intentionally remains sparse: OwnedEnvelope would add1.
    assert int(row.n_kept) == 3
    np.testing.assert_array_equal(row.grid[:3], [0.0, 3.0, 4.0])
    np.testing.assert_array_equal(
        row.binding_query, [False, True, True, False, True, True, True, False]
    )


def test_overflow_keeps_the_original_poison_and_withholds_all_certificates() -> None:
    """Exceeding the existing run capacity cannot acquire binding authority."""
    # Five rising runs, of which four cover the cells between 13 and 14, so a
    # capacity of three runs per cell overflows there.
    grid = [10.0, 14.0, 11.0, 17.0, 12.0, 20.0, 13.0, 23.0, 15.0, 26.0]
    row = _refine(
        grid=grid,
        policy=[1.0] * len(grid),
        value=[float(i) for i in range(len(grid))],
        floor_sources=2,
        queries=[10.0, 14.0, 20.0],
        max_runs=3,
    )
    assert int(row.n_kept) > 64
    assert np.isnan(np.asarray(row.grid)).all()
    assert not np.asarray(row.binding_query).any()


def test_binding_metadata_cannot_be_misread_as_legacy_read_support() -> None:
    """Typed metadata stays separate from the optional scalar support verdict."""
    row = _refine(
        grid=[1.0, 2.0, 3.0],
        policy=[0.5, 1.0, 1.5],
        value=[0.0, 1.0, 1.5],
        floor_sources=3,
        queries=[1.5],
    )
    separated = _refined_rows(refined=row, query_spec=None)
    assert len(separated) == 5
    assert separated[-1].shape == ()
    assert bool(separated[-1])
    for actual, expected in zip(separated[:4], row[:4], strict=True):
        _bits_equal(actual=actual, expected=expected)


def test_exact_backend_uses_saving_sources_without_owned_emission() -> None:
    """Policy/resource coincidences cannot substitute for source construction."""
    row = _refine(
        grid=[0.0, 2.0, 0.0, 2.0],
        policy=[10.0, 12.0, 20.0, 22.0],
        value=[2.0, 0.0, 0.0, 2.0],
        floor_sources=2,
        queries=[0.5, 1.0, 1.5],
        capacity=16,
    )
    result = _exact_backend(
        endog_grid=jnp.asarray([0.0, 2.0, 0.0, 2.0]),
        policy=jnp.asarray([10.0, 12.0, 20.0, 22.0]),
        value=jnp.asarray([2.0, 0.0, 0.0, 2.0]),
        marginal_utility=jnp.ones(4),
        savings=jnp.asarray([0.25, 0.25, 1.0, 2.0]),
        max_runs=ExactEnvelope().max_runs,
        n_refined=16,
        publish_provenance=False,
        _binding_source_count=2,
        _binding_queries=jnp.asarray([0.5, 1.0, 1.5]),
    )
    assert not isinstance(result, segment_envelope.OwnedEnvelope)
    assert isinstance(result, segment_envelope._NegmBindingEnvelope)
    for actual, expected in zip(result[:4], row[:4], strict=True):
        _bits_equal(actual=actual, expected=expected)
    np.testing.assert_array_equal(result.binding_query, [True, False, False])


def test_emitted_capacity_overflow_withholds_queries_without_changing_prefix_bits() -> (
    None
):
    """Resolved two-cell ownership still overflows three records into two slots."""
    row = _refine(
        grid=[1.0, 2.0, 3.0],
        policy=[0.5, 1.0, 1.5],
        value=[0.0, 1.0, 1.5],
        floor_sources=3,
        queries=[1.0, 1.5, 2.0],
        capacity=2,
    )
    assert int(row.n_kept) > 2
    assert not np.asarray(row.binding_query).any()


def test_gap_extrapolation_channel_refusal_withholds_a_known_floor_owner() -> None:
    """A refused emitted policy read outranks resolved floor source ownership."""
    maximum = jnp.finfo(jnp.asarray(0.0).dtype).max
    row = _refine(
        grid=[0.0, 1.0, np.nan, 3.0, 4.0],
        policy=[maximum / 2.0, maximum, np.nan, 13.0, 14.0],
        value=[0.0, 1.0, np.nan, 3.0, 4.0],
        floor_sources=2,
        queries=[0.5, 2.0],
        capacity=16,
    )
    kept = int(row.n_kept)
    assert 0 < kept <= 16
    assert np.isnan(np.asarray(row.policy)[:kept]).any()
    assert np.isnan(np.asarray(row.value)[:kept]).any()
    np.testing.assert_array_equal(row.binding_query, [False, False])
