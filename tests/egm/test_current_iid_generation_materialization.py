"""Held real entry-owned bank/template materialization and compiler admission."""

import sys
from collections.abc import Callable

import jax
import jax.numpy as jnp
import pytest

from _lcm.simulation.entry_allocations import SimulationEntryAllocations
from _lcm.simulation.entry_inputs import SimulationEntryInputs
from _lcm.solution.fingerprint import _semantic_fingerprint, _SemanticHasher
from _lcm.typing import PytreeValue
from lcm.typing import FloatND

pytestmark = pytest.mark.coverage(backends=("cpu",), precisions="both")


def _owner(
    *,
    roots: tuple[PytreeValue, ...] = (),
    retained: Callable[[], tuple[PytreeValue, ...]] | None = None,
) -> SimulationEntryAllocations:
    return SimulationEntryAllocations(
        original_inputs=SimulationEntryInputs(arrays=()),
        solution=None,
        model_roots=roots,
        retained_generation_roots=retained,
        devices=(jax.devices("cpu")[0],),
        budget_bytes=2**20,
    )


_REFERENCED_DEVICE_BIAS: jax.Array | None = None
_UNRELATED_DEVICE_BANK: jax.Array | None = None


def _function_with_referenced_device_bias(*, value: FloatND) -> FloatND:
    return value + _REFERENCED_DEVICE_BIAS  # ty: ignore[unsupported-operator] - the test binds the array before the visit


def test_semantic_array_observer_keeps_digest_and_exact_referenced_roots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The genuine hasher reports consumed buffers without a broad globals scan."""
    referenced = jnp.asarray(0.125)
    unrelated = jnp.ones((32,))
    monkeypatch.setattr(sys.modules[__name__], "_REFERENCED_DEVICE_BIAS", referenced)
    monkeypatch.setattr(sys.modules[__name__], "_UNRELATED_DEVICE_BANK", unrelated)
    roots: list[jax.Array] = []
    observed = _SemanticHasher(array_recorder=roots.append)
    observed.visit(value=_function_with_referenced_device_bias)
    assert observed.hexdigest() == _semantic_fingerprint(
        _function_with_referenced_device_bias
    )
    assert any(root is referenced for root in roots)
    assert all(root is not unrelated for root in roots)
