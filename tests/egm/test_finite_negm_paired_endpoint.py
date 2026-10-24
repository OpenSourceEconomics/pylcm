"""Paired endpoint economics remain independent of finite value-germ ownership."""

from dataclasses import replace

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
from _lcm.egm.upper_envelope.segment_envelope import DEFAULT_MAX_RUNS, OwnedEnvelope
from lcm.typing import FloatND
from tests.conftest import EXACT_KERNEL_SKIP_REASON
from tests.egm._finite_selection_component import _FiniteOuterRead, component_view
from tests.egm.test_finite_negm_owned_germs import _emit

pytestmark = pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)


def _paired_component(
    *, row: OwnedEnvelope, marginal: FloatND, data: EGMQueryData, coordinate: float
) -> tuple[_FiniteRowRead, _FiniteOuterRead, FloatND, FloatND]:
    """Validate one native component and retain its existing passive-side arithmetic."""
    query = jnp.asarray(coordinate)
    raw = _read_finite_native_row(
        coordinate=query,
        grid=row.grid,
        value=row.value,
        marginal=marginal,
        data=data,
        owner_capacity=DEFAULT_MAX_RUNS,
        side_is_row_side=True,
    )
    candidate = component_view(
        profile=_validate_finite_outer_rows(
            raw=raw,
            queries=query,
            gradients=jnp.asarray(1.0),
            floor=jnp.asarray(0.0),
            upper_saving=jnp.asarray(4.0),
            resources=jnp.asarray(5.0),
            eligible=jnp.asarray(a=True),
            corner_value=jnp.asarray(-10.0),
            corner_marginal=jnp.asarray(0.0),
            # No corner is represented in this ordinary-row component.
            supported_corner=jnp.asarray(a=False),
            finite_corner=jnp.asarray(a=False),
            invalid_corner=jnp.asarray(a=False),
            paired_marginal_read=True,
        )
    )
    value, right, left, right_alive = _collapse_stacked_candidates(
        value_at_child=candidate.value[None],
        ordinary_marginal_at_child=candidate.right_marginal[None],
        left_marginal_at_child=candidate.left_marginal[None],
        right_dead_at_child=candidate.right_dead[None],
        left_dead_at_child=candidate.left_dead[None],
        right_germ_at_child=(
            candidate.right_germ[0][None],
            candidate.right_germ[1][None],
            candidate.right_germ[2][None],
            candidate.right_germ[3][None],
        ),
        left_germ_at_child=(
            candidate.left_germ[0][None],
            candidate.left_germ[1][None],
            candidate.left_germ[2][None],
            candidate.left_germ[3][None],
        ),
    )
    published_value, published_marginal = _blend_passive_axes(
        value_at_child=value,
        marginal_arrays=(right, left, right_alive),
        child_passive_values=(),
        child_passive_grids=(),
        n_outer_candidates=1,
    )
    return raw, candidate, published_value, published_marginal


def _singleton_owned_record() -> tuple[
    OwnedEnvelope, OwnedEnvelope, FloatND, EGMQueryData
]:
    """Truncate a genuine producer row while preserving its ownership provenance."""
    emitted, singleton_marginal, singleton_data = _emit(
        coordinates=(1.0, 2.0), values=(0.0, 1.0), marginals=(2.0, 2.0)
    )
    # Truncation tests transport of an authenticated first record; it does not
    # assert that the native producer emits singleton runs as live intervals.
    first = jnp.arange(emitted.grid.size) == 0
    singleton_row = emitted._replace(
        grid=jnp.where(first, emitted.grid, jnp.nan),
        value=jnp.where(first, emitted.value, jnp.nan),
    )
    singleton_data = replace(
        singleton_data,
        node_kind=jnp.where(first, QueryNodeKind.FINITE, QueryNodeKind.PAD).astype(
            jnp.int32
        ),
        link_valid=jnp.zeros_like(singleton_data.link_valid),
    )
    return emitted, singleton_row, singleton_marginal, singleton_data


def test_paired_endpoint_marginal_remains_distinct_from_right_clamp_germ() -> None:
    """Paired M keeps the endpoint slope while clamp germs and exterior M are zero."""
    row, marginal, data = _emit(
        coordinates=(1.0, 2.0, 3.0),
        values=(0.0, 1.0, 4.0),
        marginals=(0.0, 2.0, 4.0),
    )
    terminal, terminal_candidate, terminal_value, terminal_m = _paired_component(
        row=row, marginal=marginal, data=data, coordinate=3.0
    )
    assert bool(terminal.ordinary.supported & terminal.left.supported)
    assert not bool(terminal.ordinary.invalid_numeric | terminal.left.invalid_numeric)
    assert not bool(terminal_candidate.coverage_missing)
    np.testing.assert_array_equal(terminal.ordinary.value, 4.0)
    np.testing.assert_array_equal(terminal.ordinary.marginal, 4.0)
    np.testing.assert_array_equal(terminal_candidate.right_marginal, 4.0)
    np.testing.assert_array_equal(terminal_candidate.left_marginal, 4.0)
    assert bool(terminal_candidate.right_dead)
    assert not bool(terminal_candidate.left_dead)
    np.testing.assert_array_equal(terminal_value, 4.0)
    np.testing.assert_array_equal(terminal_m, 4.0)

    above, above_candidate, above_value, above_m = _paired_component(
        row=row, marginal=marginal, data=data, coordinate=3.5
    )
    assert bool(above.ordinary.supported)
    assert not bool(above.ordinary.invalid_numeric | above.left.invalid_numeric)
    assert not bool(above_candidate.coverage_missing)
    np.testing.assert_array_equal(above.ordinary.value, 4.0)
    np.testing.assert_array_equal(above.ordinary.marginal, 4.0)
    np.testing.assert_array_equal(above_candidate.right_marginal, 0.0)
    np.testing.assert_array_equal(above_candidate.left_marginal, 0.0)
    np.testing.assert_array_equal(above_value, 4.0)
    np.testing.assert_array_equal(above_m, 0.0)

    emitted, singleton_row, singleton_marginal, singleton_data = (
        _singleton_owned_record()
    )
    np.testing.assert_array_equal(singleton_row.owner_id, emitted.owner_id)
    np.testing.assert_array_equal(singleton_row.side_id, emitted.side_id)
    np.testing.assert_array_equal(singleton_data.savings[0], emitted.savings[0])
    singleton, singleton_candidate, singleton_value, singleton_m = _paired_component(
        row=singleton_row,
        marginal=singleton_marginal,
        data=singleton_data,
        coordinate=1.0,
    )
    assert bool(singleton.ordinary.supported)
    assert not bool(singleton.ordinary.invalid_numeric | singleton.left.invalid_numeric)
    assert not bool(singleton_candidate.coverage_missing)
    np.testing.assert_array_equal(singleton.ordinary.value, 0.0)
    np.testing.assert_array_equal(singleton.ordinary.marginal, 2.0)
    np.testing.assert_array_equal(singleton_candidate.right_marginal, 0.0)
    np.testing.assert_array_equal(singleton_candidate.left_marginal, 0.0)
    assert bool(singleton_candidate.right_dead & singleton_candidate.left_dead)
    np.testing.assert_array_equal(singleton_value, 0.0)
    np.testing.assert_array_equal(singleton_m, 0.0)

    # Economic endpoint controls qualify before the side-ownership assertions.
    _assert_paired_germ_ownership(
        terminal=terminal,
        terminal_candidate=terminal_candidate,
        above=above,
        singleton=singleton,
    )


def _assert_paired_germ_ownership(
    *,
    terminal: _FiniteRowRead,
    terminal_candidate: _FiniteOuterRead,
    above: _FiniteRowRead,
    singleton: _FiniteRowRead,
) -> None:
    """Require exact side-germ ownership after the economic controls qualify."""
    assert bool(terminal.right_germ[0])
    np.testing.assert_array_equal(jnp.stack(terminal.right_germ[1:]), [0.0, 0.0, 0.0])
    assert bool(terminal_candidate.right_germ[0])
    np.testing.assert_array_equal(
        jnp.stack(terminal_candidate.right_germ[1:]), [0.0, 0.0, 0.0]
    )
    assert bool(terminal.left_germ[0])
    np.testing.assert_array_equal(jnp.stack(terminal.left_germ[1:]), [4.0, 2.0, 0.0])
    assert bool(above.right_germ[0] & above.left_germ[0])
    np.testing.assert_array_equal(jnp.stack(above.right_germ[1:]), [0.0, 0.0, 0.0])
    np.testing.assert_array_equal(jnp.stack(above.left_germ[1:]), [0.0, 0.0, 0.0])
    assert bool(singleton.right_germ[0])
    assert not bool(singleton.left_germ[0])
    np.testing.assert_array_equal(jnp.stack(singleton.right_germ[1:]), [0.0, 0.0, 0.0])
