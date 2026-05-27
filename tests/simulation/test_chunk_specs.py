"""Tests for V_arr chunk-spec derivation."""

from collections.abc import Mapping

from _lcm.simulation.chunk_specs import (
    _chunk_spec_from_state_order,
    _ChunkSpec,
)


def test_chunk_spec_picks_first_axis_with_positive_batch_size():
    """A batched state at axis 1 yields a spec with `chunk_axis=1`."""
    state_names = ("wealth", "aime", "health")
    batch_sizes: Mapping[str, int] = {"wealth": 0, "aime": 1, "health": 0}

    spec = _chunk_spec_from_state_order(
        state_names=state_names,
        batch_sizes=batch_sizes,
    )

    assert spec == _ChunkSpec(chunk_axis=1, chunk_size=1)


def test_chunk_spec_returns_no_chunking_when_every_state_batch_size_zero():
    """All-zero `batch_size` means no axis was scanned over — no chunking required."""
    state_names = ("wealth", "health")
    batch_sizes: Mapping[str, int] = {"wealth": 0, "health": 0}

    spec = _chunk_spec_from_state_order(
        state_names=state_names,
        batch_sizes=batch_sizes,
    )

    assert spec == _ChunkSpec(chunk_axis=None, chunk_size=0)


def test_chunk_spec_picks_outermost_when_multiple_states_are_batched():
    """When several states have `batch_size > 0`, the spec picks the outermost axis."""
    state_names = ("aime", "assets", "health")
    batch_sizes: Mapping[str, int] = {"aime": 4, "assets": 2, "health": 0}

    spec = _chunk_spec_from_state_order(
        state_names=state_names,
        batch_sizes=batch_sizes,
    )

    assert spec == _ChunkSpec(chunk_axis=0, chunk_size=4)
