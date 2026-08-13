"""`--release-compiled-programs` hands a worker's freed heap back to the OS.

Dropping JAX's in-memory cache is only half of bounding a worker's resident
memory. XLA:CPU builds every program through LLVM, whose intermediate
representation is ordinary heap, and a C allocator does not return a freed
block to the operating system on its own -- it keeps it in an arena for the
next request. Compilation is multi-threaded, so the next request routinely
lands in a different arena and allocates fresh, and a worker's resident memory
then tracks the *sum* of the programs it has ever built rather than the largest
one live at once.

So the release step also trims the allocator. What that has to deliver is
measurable and is what this file asserts: heap a test has finished with stops
counting against the worker.
"""

import gc

import pytest

from tests.conftest import (
    resident_mebibytes,
    return_free_heap_to_os,
)

_ALLOCATION_MIB = 512
# Small enough that the allocator serves a block from an arena rather than from
# `mmap`, which it would hand back unprompted and so would not exercise the trim.
_BLOCK_BYTES = 2048


def test_resident_mebibytes_reports_a_plausible_size_for_this_process() -> None:
    """The resident-size probe reports this process's real footprint."""
    resident = resident_mebibytes()
    if resident is None:
        pytest.skip("no resident-size probe on this platform")
    assert 1.0 < resident < 1_000_000.0


def test_freed_heap_stops_counting_against_the_worker_once_it_is_returned() -> None:
    """Heap a test has finished with is given back, not held in an arena.

    Resident memory after the trim is back within a small margin of where it
    started, so a worker that builds one large program after another stays at
    the size of one rather than growing by every program it has built.
    """
    gc.collect()
    if not return_free_heap_to_os():
        pytest.skip("no allocator trim on this platform")
    baseline = resident_mebibytes()
    if baseline is None:
        pytest.skip("no resident-size probe on this platform")

    blocks = [
        bytes(_BLOCK_BYTES) for _ in range(_ALLOCATION_MIB * 1024**2 // _BLOCK_BYTES)
    ]
    while_held = resident_mebibytes()
    assert while_held is not None
    assert while_held > baseline + _ALLOCATION_MIB / 2

    del blocks
    gc.collect()
    return_free_heap_to_os()

    once_returned = resident_mebibytes()
    assert once_returned is not None
    assert once_returned < baseline + _ALLOCATION_MIB / 8
