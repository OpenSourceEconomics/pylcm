"""Owned finite rows keep their side finiteness and independent economic records."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.finite_negm_query import (
    _FiniteOuterProfile,
    _FiniteRowRead,
    _read_finite_native_row,
    _select_finite_outer_profiles,
    _validate_finite_outer_rows,
)
from _lcm.egm.query_data import EGMQueryData, QueryNodeKind
from _lcm.egm.upper_envelope.segment_envelope import (
    DEFAULT_MAX_RUNS,
    OwnedEnvelope,
    refine_envelope_exact,
)
from lcm.typing import FloatND
from tests.conftest import EXACT_KERNEL_SKIP_REASON

pytestmark = pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)


def _emit(
    *,
    coordinates: tuple[float, ...],
    values: tuple[float, ...],
    marginals: tuple[float, ...],
) -> tuple[OwnedEnvelope, FloatND, EGMQueryData]:
    """Emit source provenance for a monotone component geometry."""
    assert len(coordinates) == len(values) == len(marginals)
    savings = jnp.linspace(0.125, 0.25, len(coordinates))
    emitted = refine_envelope_exact(
        endog_grid=jnp.asarray(coordinates),
        policy=jnp.asarray(coordinates) - savings,
        value=jnp.asarray(values),
        savings=savings,
        n_refined=8,
    )
    assert isinstance(emitted, OwnedEnvelope)
    assert int(emitted.n_kept) == len(coordinates)
    np.testing.assert_array_equal(emitted.grid[: emitted.n_kept], coordinates)
    np.testing.assert_array_equal(emitted.value[: emitted.n_kept], values)
    np.testing.assert_array_equal(emitted.savings[: emitted.n_kept], savings)
    assert bool(jnp.all(emitted.link_valid[: max(len(coordinates) - 1, 0)]))
    n = len(marginals)
    native_marginal = (
        jnp.full_like(emitted.grid, jnp.nan).at[:n].set(jnp.asarray(marginals))
    )
    data = EGMQueryData(
        savings=emitted.savings,
        node_kind=jnp.where(
            jnp.arange(8) < emitted.n_kept, QueryNodeKind.FINITE, QueryNodeKind.PAD
        ).astype(jnp.int32),
        link_valid=emitted.link_valid,
        owner_id=emitted.owner_id,
        side_id=emitted.side_id,
        floor_used=jnp.asarray(0.0),
        corner_continuation=jnp.asarray([0.0]),
        corner_kind=jnp.asarray([QueryNodeKind.FINITE], dtype=jnp.int32),
        corner_interval_id=jnp.asarray([0], dtype=jnp.int32),
        corner_side_id=jnp.asarray([0], dtype=jnp.int32),
    )
    return emitted, native_marginal, data


def _read(
    *, row: OwnedEnvelope, marginal: FloatND, data: EGMQueryData, coordinate: float
) -> _FiniteRowRead:
    return _read_finite_native_row(
        coordinate=jnp.asarray(coordinate),
        grid=row.grid,
        value=row.value,
        marginal=marginal,
        data=data,
        owner_capacity=DEFAULT_MAX_RUNS,
        side_is_row_side=True,
    )


@pytest.mark.parametrize(
    ("coordinate", "expected_value", "expected_marginal", "left_finite"),
    [
        (1.0, 0.0, 0.0, False),
        (2.0, 1.0, 2.0, True),
        (3.0, 4.0, 4.0, True),
        (3.5, 4.0, 4.0, True),
    ],
)
def test_owned_quadratic_sides_at_nodes_and_clamp(
    *,
    coordinate: float,
    expected_value: float,
    expected_marginal: float,
    left_finite: bool,
) -> None:
    """The exact quadratic's left finiteness is separate from its economic record."""
    row, marginal, data = _emit(
        coordinates=(1.0, 2.0, 3.0), values=(0.0, 1.0, 4.0), marginals=(0.0, 2.0, 4.0)
    )
    raw = _read(row=row, marginal=marginal, data=data, coordinate=coordinate)
    assert bool(raw.ordinary.supported)
    assert not bool(raw.ordinary.invalid_numeric)
    np.testing.assert_array_equal(raw.ordinary.value, expected_value)
    np.testing.assert_array_equal(raw.ordinary.marginal, expected_marginal)
    assert bool(raw.left_finite) is left_finite


def test_duplicate_keeps_each_actual_side_record() -> None:
    """A native discontinuity keeps its own left and right line and marginal."""
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
    marginal = jnp.where(
        emitted.owner_id == 0, 2.0, jnp.where(emitted.owner_id == 1, 0.5, jnp.nan)
    )
    data = EGMQueryData(
        savings=emitted.savings,
        node_kind=jnp.where(
            jnp.arange(8) < emitted.n_kept, QueryNodeKind.FINITE, QueryNodeKind.PAD
        ).astype(jnp.int32),
        link_valid=emitted.link_valid,
        owner_id=emitted.owner_id,
        side_id=emitted.side_id,
        floor_used=jnp.asarray(0.0),
        corner_continuation=jnp.asarray([0.0]),
        corner_kind=jnp.asarray([QueryNodeKind.FINITE], dtype=jnp.int32),
        corner_interval_id=jnp.asarray([0], dtype=jnp.int32),
        corner_side_id=jnp.asarray([0], dtype=jnp.int32),
    )
    raw = _read(row=emitted, marginal=marginal, data=data, coordinate=1.0)
    np.testing.assert_array_equal(raw.ordinary.value, 0.5)
    np.testing.assert_array_equal(raw.ordinary.marginal, 0.5)
    np.testing.assert_array_equal(raw.left.value, 2.0)
    np.testing.assert_array_equal(raw.left.marginal, 2.0)
    assert bool(raw.left_finite)


def test_invalid_link_is_missing_without_fabricated_side() -> None:
    """A missing owned link cannot contribute a finite neighborhood."""
    row, marginal, data = _emit(
        coordinates=(1.0, 2.0), values=(0.0, 1.0), marginals=(1.0, 1.0)
    )
    raw = _read(
        row=row,
        marginal=marginal,
        data=replace(data, link_valid=jnp.zeros_like(data.link_valid)),
        coordinate=1.5,
    )
    assert not bool(raw.ordinary.supported)
    assert not bool(raw.ordinary.invalid_numeric)
    assert not bool(raw.left_finite)


def test_single_record_has_no_left_side_and_independent_economic_marginal() -> None:
    """An isolated authenticated record has no left support and a constant right ray."""
    emitted, marginal, data = _emit(
        coordinates=(1.0, 2.0), values=(0.0, 1.0), marginals=(2.0, 2.0)
    )
    # This component truncates a genuine native record; it is not a claim that
    # refine_envelope_exact publishes singleton runs as supported intervals.
    live = jnp.arange(8) == 0
    row = emitted._replace(
        grid=jnp.where(live, emitted.grid, jnp.nan),
        value=jnp.where(live, emitted.value, jnp.nan),
    )
    data = replace(
        data,
        node_kind=jnp.where(live, QueryNodeKind.FINITE, QueryNodeKind.PAD).astype(
            jnp.int32
        ),
        link_valid=jnp.zeros_like(data.link_valid),
    )
    raw = _read(row=row, marginal=marginal, data=data, coordinate=1.0)
    assert bool(raw.ordinary.supported)
    np.testing.assert_array_equal(raw.ordinary.value, 0.0)
    np.testing.assert_array_equal(raw.ordinary.marginal, 2.0)
    assert not bool(raw.left_finite)


def _profile(raw: _FiniteRowRead) -> _FiniteOuterProfile:
    return _validate_finite_outer_rows(
        raw=raw,
        queries=jnp.asarray(2.0),
        gradients=jnp.asarray(1.0),
        floor=jnp.asarray(0.0),
        upper_saving=jnp.asarray(2.0),
        resources=jnp.asarray(4.0),
        eligible=jnp.asarray(a=True),
        corner_value=jnp.asarray(-10.0),
        corner_marginal=jnp.asarray(0.0),
        supported_corner=jnp.asarray(a=False),
        finite_corner=jnp.asarray(a=False),
        invalid_corner=jnp.asarray(a=False),
        paired_marginal_read=False,
    )


@pytest.mark.parametrize("increasing", [False, True])
def test_exact_value_tie_goes_to_the_lowest_outer_index(*, increasing: bool) -> None:
    """An exact value tie selects the lowest O, whatever the right-side shapes."""
    clamp, clamp_m, clamp_data = _emit(
        coordinates=(1.0, 2.0), values=(0.0, 1.0), marginals=(1.0, 1.0)
    )
    other, other_m, other_data = _emit(
        coordinates=(1.0, 3.0),
        values=(0.0, 2.0) if increasing else (2.0, 0.0),
        marginals=(0.5, 0.5),
    )
    raw_clamp = _read(row=clamp, marginal=clamp_m, data=clamp_data, coordinate=2.0)
    raw_other = _read(row=other, marginal=other_m, data=other_data, coordinate=2.0)
    np.testing.assert_array_equal(raw_clamp.ordinary.value, 1.0)
    np.testing.assert_array_equal(raw_other.ordinary.value, 1.0)
    candidates = jax.tree.map(
        lambda *leaves: jnp.stack(leaves, axis=-1),
        _profile(raw_clamp),
        _profile(raw_other),
    )
    np.testing.assert_array_equal(candidates.right.value, [1.0, 1.0])
    assert bool(candidates.right.dead[0])
    selection = _select_finite_outer_profiles(profiles=candidates)
    np.testing.assert_array_equal(selection.outer_index, [0])
    np.testing.assert_array_equal(selection.right.ordinary.marginal, [1.0])


def test_finite_node_with_economic_right_bracket_keeps_its_left_side() -> None:
    """A finite node before an economic -inf right interval keeps its left side."""
    emitted, marginal, data = _emit(
        coordinates=(1.0, 2.0, 3.0), values=(0.0, 1.0, 2.0), marginals=(1.0, 1.0, 1.0)
    )
    economic = jnp.arange(8) == 2
    row = emitted._replace(value=jnp.where(economic, -jnp.inf, emitted.value))
    data = replace(
        data,
        node_kind=jnp.where(economic, QueryNodeKind.ECONOMIC_MINUS_INF, data.node_kind),
    )
    raw = _read(
        row=row, marginal=jnp.where(economic, 0.0, marginal), data=data, coordinate=2.0
    )
    assert bool(raw.ordinary.supported)
    np.testing.assert_array_equal(raw.ordinary.value, 1.0)
    np.testing.assert_array_equal(raw.ordinary.marginal, 1.0)
    assert bool(raw.left_finite)
