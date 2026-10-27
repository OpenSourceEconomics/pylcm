"""Closed retained storage preserves actual future carries and their read paths."""

import pickle
from dataclasses import replace

import jax
import numpy as np
import pytest

from _lcm.egm.carry import EGMCarry, build_template_egm_carry
from _lcm.egm.current_iid_case_u_replay import CaseUReplayChildren, CaseUReplayInput
from lcm._solver_api.authority import _snapshot_artifact_template_once

pytestmark = pytest.mark.coverage(backends=("cpu",), precisions="both")


def test_case_u_future_tuple_keeps_original_carries_and_closed_snapshot() -> None:
    first = build_template_egm_carry(n_rows=3)
    second = build_template_egm_carry(n_rows=4)
    payload = CaseUReplayChildren(
        source="original",
        period=1,
        child_names=("zeta", "alpha"),
        child_values=(first, second),
    )
    assert payload.children["zeta"] is first
    assert payload.children["alpha"] is second
    assert tuple(payload.children) == ("zeta", "alpha")
    original = jax.tree.leaves(payload)
    assert all(
        actual is wanted
        for actual, wanted in zip(
            original, (*jax.tree.leaves(first), *jax.tree.leaves(second)), strict=True
        )
    )
    snapshot, _ = _snapshot_artifact_template_once(
        template=payload, payload_runtime_type=CaseUReplayChildren
    )
    detached = snapshot.payload
    assert isinstance(detached, CaseUReplayChildren)
    restored = pickle.loads(pickle.dumps(detached))
    assert isinstance(restored, CaseUReplayChildren)
    for item in (detached, restored):
        assert item.source == payload.source
        assert item.period == payload.period
        assert tuple(item.children) == tuple(payload.children)
        assert all(isinstance(child, EGMCarry) for child in item.children.values())
        assert jax.tree.structure(item) == jax.tree.structure(payload)
        for actual, wanted in zip(jax.tree.leaves(item), original, strict=True):
            assert actual is not wanted
            assert actual.shape == wanted.shape
            assert actual.dtype == wanted.dtype
            np.testing.assert_array_equal(actual, wanted)
    reads = CaseUReplayInput(payload=payload).reads(core="actual_future")
    paths = jax.tree_util.tree_flatten_with_path(payload)[0]
    assert len(reads) == len(paths)
    for read, (path, leaf), wanted in zip(reads, paths, original, strict=True):
        assert read.source.path[0] == "child_values"
        assert read.target.leaf_path == tuple(
            f"{type(step).__name__}:{step}" for step in path
        )
        assert leaf is wanted


@pytest.mark.parametrize("names", [("zeta", "zeta"), ("zeta",)])
def test_case_u_future_tuple_refuses_ambiguous_or_missing_names(
    *, names: tuple[str, ...]
) -> None:
    payload = CaseUReplayChildren(
        source="original",
        period=1,
        child_names=("zeta", "alpha"),
        child_values=(
            build_template_egm_carry(n_rows=3),
            build_template_egm_carry(n_rows=4),
        ),
    )
    with pytest.raises(TypeError, match="exact original child names"):
        replace(payload, child_names=names)
