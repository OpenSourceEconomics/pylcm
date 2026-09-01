"""The five NBEGM partition sites use the shared EGM batching contract."""

import ast
from collections import Counter
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from _lcm.solution import nbegm


def _record_batches(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Collect the actual batch size passed to the shared dispatcher."""
    calls: list[int] = []

    def spy(*, func, xs, batch_size):
        calls.append(batch_size)
        return jax.vmap(func)(xs)

    monkeypatch.setattr(nbegm, "map_over_leading_axis", spy)
    return calls


def _identity(row):
    return row


@pytest.mark.parametrize(
    ("wrapper", "requested"),
    [(nbegm._map_ride_partitioned, 3), (nbegm._map_branch_partitioned, 1)],
)
def test_partition_request_reaches_the_shared_dispatcher(
    *, monkeypatch: pytest.MonkeyPatch, wrapper, requested: int
) -> None:
    """Ride and branch requests become actual shared-dispatcher batch widths."""
    calls = _record_batches(monkeypatch)
    result = wrapper(
        func=_identity,
        xs=jnp.arange(5),
        requested_block_size=requested,
    )

    assert result.shape == (5,)
    assert calls == [requested]


def test_all_five_production_sites_use_the_axis_specific_wrappers() -> None:
    """The source contains exactly three ride and two branch routing calls."""
    tree = ast.parse(Path(nbegm.__file__).read_text(encoding="utf-8"))
    names = Counter(
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    )

    assert names["_map_ride_partitioned"] == 3
    assert names["_map_branch_partitioned"] == 2
