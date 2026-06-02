"""Tests for subject-batch autotuning from compiled-program memory estimates."""

import dataclasses

from _lcm.simulation.autotune import estimate_peak_bytes, pick_batch_size


@dataclasses.dataclass(frozen=True)
class _FakeStats:
    """Stand-in for `jax.stages.Compiled.memory_analysis()` output."""

    temp_size_in_bytes: int
    argument_size_in_bytes: int
    output_size_in_bytes: int
    alias_size_in_bytes: int


def test_estimate_peak_bytes_subtracts_aliased_buffers():
    """Peak is temp + argument + output minus the in-place-aliased bytes."""
    stats = _FakeStats(
        temp_size_in_bytes=100,
        argument_size_in_bytes=50,
        output_size_in_bytes=30,
        alias_size_in_bytes=20,
    )
    assert estimate_peak_bytes(stats=stats) == 160


def test_pick_batch_size_fits_line_through_two_probes():
    """The chosen batch is the largest whose affine-extrapolated peak fits budget."""
    # peak(b) = 400 + 1.6 b  →  budget 1000 ⇒ b = (1000 - 400) / 1.6 = 375.
    batch = pick_batch_size(
        probes=((1000, 2000), (500, 1200)),
        budget_bytes=1000,
        max_batch=1000,
    )
    assert batch == 375


def test_pick_batch_size_clamps_to_max_batch():
    """A budget that admits more than the population caps at the population size."""
    batch = pick_batch_size(
        probes=((1000, 500), (500, 300)),
        budget_bytes=10_000,
        max_batch=1000,
    )
    assert batch == 1000


def test_pick_batch_size_floors_at_one():
    """A budget below the fixed overhead still yields a runnable batch of 1."""
    batch = pick_batch_size(
        probes=((1000, 5000), (500, 4500)),
        budget_bytes=100,
        max_batch=1000,
    )
    assert batch == 1


def test_pick_batch_size_handles_degenerate_identical_batch_probes():
    """Two probes at the same batch size don't divide by zero (population of 1).

    A one-subject population makes the full- and half-population probes identical,
    so the picker must fall back to a proportional model instead of fitting a line
    through a zero-width interval — and clamp to the single available batch.
    """
    batch = pick_batch_size(
        probes=((1, 1000), (1, 800)),
        budget_bytes=500,
        max_batch=1,
    )
    assert batch == 1
