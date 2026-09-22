"""Budget inventory observes retained solution storage without public reads."""

import importlib
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from types import MappingProxyType
from typing import cast
from unittest.mock import Mock

import jax
import jax.numpy as jnp
import pytest

from _lcm.egm.outer_inversion import DeclaredOuterInverse
from _lcm.egm.outer_replay_capability import OuterReplayCapability
from _lcm.egm.published_policy import EGMSimPolicy
from _lcm.persistence import solution as persistence
from _lcm.simulation.replay_inputs import PreparedReplayReader
from _lcm.solution.artifacts import OwnedSolutionView
from _lcm.solution.model_authority import SolutionAuthority
from lcm._solver_api import authority as authority_module
from lcm._solver_api.entries import (
    _canonical_artifact_entry_from_authority,
    _CanonicalValueEntry,
    _LazyEntry,
)
from lcm.exceptions import ExecutionPlanningError
from lcm.persistence import load_solution, save_solution
from lcm.solver_api import (
    ArtifactStore,
    ExecutableReplayRoute,
    LoadState,
    ReplayRouteSnapshot,
    SimulationBuildContext,
    SolutionResult,
    ValueStore,
)
from tests.solution.test_solution_persistence import (
    _make_solution,
    _make_stateful_pytree_solution,
    _make_values_only_solution,
    _StatefulPersistenceTree,
)


def _buffers(solution: SolutionResult) -> tuple[jax.Array, ...]:
    module = importlib.import_module("_lcm.solution.retained_buffers")
    return module.retained_solution_buffers(solution=solution)


def _backing_values(solution: SolutionResult) -> tuple[jax.Array, ...]:
    store = cast("ValueStore", solution.values)
    return tuple(
        cast("jax.Array", cast("_CanonicalValueEntry", entry).value)
        for entry in store._entries.values()
    )


def _authority_arrays(solution: SolutionResult) -> tuple[jax.Array, ...]:
    arrays = []
    for authority in solution._artifact_authority.values():
        binding = authority_module._ARTIFACT_AUTHORITY_TEMPLATE_BINDINGS[id(authority)]
        arrays.extend(binding.public_template_leaves)
        if binding.snapshot is not None:
            arrays.extend(binding.snapshot.leaves)
    return tuple(arrays)


def test_eager_entries_and_both_authority_templates_are_observed_without_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The exact existing buffer identities survive forbidden public read methods."""
    solution = _make_solution()
    expected = (
        *_backing_values(solution),
        *solution.replay_artifacts._entries.values(),
        *_authority_arrays(solution),
    )

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("Public store materialization was called")

    monkeypatch.setattr(ValueStore, "__getitem__", forbidden)
    monkeypatch.setattr(ArtifactStore, "__getitem__", forbidden)
    monkeypatch.setattr(ValueStore, "materialize", forbidden)
    monkeypatch.setattr(ArtifactStore, "materialize", forbidden)
    assert {id(array) for array in _buffers(solution)} == {
        id(array) for array in expected
    }


def test_every_retained_artifact_channel_contributes_its_buffers() -> None:
    """Continuation, replay, auxiliary and diagnostics storage all remain live."""
    base = _make_solution()
    ref = next(iter(base.replay_artifacts))
    arrays = tuple(jnp.arange(size, dtype=jnp.int32) for size in range(1, 5))
    solution = replace(
        base,
        retained_continuations=ArtifactStore({ref: arrays[0]}),
        replay_artifacts=ArtifactStore({ref: arrays[1]}),
        auxiliary_artifacts=ArtifactStore({ref: arrays[2]}),
        diagnostics=ArtifactStore({ref: arrays[3]}),
    )
    expected = (*_backing_values(solution), *arrays)
    assert {id(array) for array in _buffers(solution)} == {
        id(array) for array in expected
    }


def test_canonical_artifact_keeps_payload_and_reconstruction_template() -> None:
    """A canonical entry owns concrete leaves as well as its reconstruction plan."""
    base = _make_solution()
    ref = next(iter(base.replay_artifacts))
    entry = _canonical_artifact_entry_from_authority(
        payload=base.replay_artifacts[ref], authority=base._artifact_authority[ref]
    )
    solution = replace(base, replay_artifacts=ArtifactStore({ref: entry}))
    expected = (*_backing_values(solution), *entry.leaves, *entry.plan_snapshot.leaves)
    assert {id(array) for array in _buffers(solution)} == {
        id(array) for array in expected
    }


@pytest.mark.parametrize("loaded", [False, True])
def test_archive_observation_never_opens_or_decodes_the_archive(
    *, tmp_path: Path, loaded: bool
) -> None:
    """Unloaded storage is empty; loaded private caches count without fresh copies."""
    original = _make_values_only_solution(value=[1.0, 2.0], dtype="float32")
    path = save_solution(solution=original, path=tmp_path / "solution.h5")
    solution = load_solution(path=path)
    if loaded:
        solution.value(period=0, regime="working")
    entry = cast(
        "persistence._LazyHdf5Entry",
        cast("ValueStore", solution.values)._raw(period=0, regime="working"),
    )
    expected = (
        cast("persistence._LoadedEntryPayload", entry._cache.value).leaves
        if loaded
        else ()
    )
    path.unlink()
    observed = _buffers(solution)
    assert {id(array) for array in observed} == {id(array) for array in expected}
    assert entry.load_state is (LoadState.LOADED if loaded else LoadState.UNLOADED)


def test_owned_and_every_previously_consumed_view_remain_in_inventory() -> None:
    """Earlier consumers' private values and policies stay charged with the source."""
    solution = _make_values_only_solution(value=[1.0], dtype="float32")
    arrays = tuple(jnp.arange(size, dtype=jnp.float32) for size in range(1, 8))
    flags = jnp.array([True, False])
    empty = MappingProxyType({})
    view = OwnedSolutionView(
        model_instance_id="owner",
        params_fingerprint="0" * 64,
        values=MappingProxyType({0: MappingProxyType({"working": arrays[0]})}),
        simulation_policies=MappingProxyType(
            {
                0: MappingProxyType(
                    {
                        "working": EGMSimPolicy(
                            endog_grid=arrays[1],
                            policy=arrays[1],
                            value=arrays[2],
                            marginal_utility=arrays[2],
                        ),
                    }
                )
            }
        ),
        dissolution_flags=MappingProxyType({0: MappingProxyType({"working": flags})}),
        replay_artifacts=empty,
        authority=SolutionAuthority(values=empty, replay=empty),
    )
    object.__setattr__(solution, "_engine_view", view)
    solution._consumed_views["first"] = (
        {0: {"working": arrays[3]}},
        {0: {"working": arrays[4]}},
        {},
        {},
    )
    solution._consumed_views["second"] = (
        {0: {"working": arrays[5]}},
        {},
        {0: {"working": arrays[6]}},
        {},
    )
    expected = (*_backing_values(solution), *arrays, flags)
    assert {id(array) for array in _buffers(solution)} == {
        id(array) for array in expected
    }


def test_consumed_reader_retains_snapshot_authorities_and_grid_context() -> None:
    """Prepared readers retain payload and grid arrays before their period runs."""
    base = _make_solution()
    ref = next(iter(base.replay_artifacts))
    state = jnp.arange(3, dtype=jnp.int32)
    action = jnp.arange(4, dtype=jnp.int32)
    payload = base.replay_artifacts._raw(ref)
    reader = PreparedReplayReader(
        route=cast("ExecutableReplayRoute", Mock(spec=ExecutableReplayRoute)),
        snapshot=ReplayRouteSnapshot(
            artifacts={ref.key: payload},
            authorities={ref.key: base._artifact_authority[ref]},
            metadata=base.metadata,
        ),
        context=SimulationBuildContext(
            period=0,
            regime_name="working",
            state_names=("state",),
            action_names=("action",),
            state_nodes={"state": state},
            action_nodes={"action": action},
        ),
    )
    solution = replace(base, replay_artifacts=ArtifactStore())
    solution._consumed_views["consumer"] = ({}, {}, {}, {0: {"working": reader}})
    authority = base._artifact_authority[ref]
    binding = authority_module._ARTIFACT_AUTHORITY_TEMPLATE_BINDINGS[id(authority)]
    assert binding.snapshot is not None
    expected = (
        *_backing_values(solution),
        payload,
        state,
        action,
        *binding.public_template_leaves,
        *binding.snapshot.leaves,
    )
    assert {id(array) for array in _buffers(solution)} == {
        id(array) for array in expected
    }


class _OpaqueLazy(_LazyEntry):
    """Unknown lazy owners cannot be assumed empty by observing their state."""

    @property
    def load_state(self) -> LoadState:
        raise AssertionError("Unknown lazy state was called")

    def materialize(self, *, template: object | None = None) -> object:
        del template
        raise AssertionError("Unknown lazy decoder was called")


def test_unknown_lazy_storage_is_refused_without_calling_its_decoder() -> None:
    """Opaque lazy entries cannot create a zero-byte budget admission."""
    base = _make_values_only_solution(value=[1.0], dtype="float32")
    solution = replace(base, values=ValueStore({(0, "working"): _OpaqueLazy()}))
    with pytest.raises(ExecutionPlanningError, match="unsupported retained"):
        _buffers(solution)


def test_unknown_raw_artifact_is_refused_without_pytree_callbacks() -> None:
    """An opaque payload may retain arrays outside any declared tree leaves."""
    base = _make_solution()
    ref = next(iter(base.replay_artifacts))
    solution = replace(base, replay_artifacts=ArtifactStore({ref: object()}))
    with pytest.raises(ExecutionPlanningError, match="unsupported retained"):
        _buffers(solution)


def test_invalid_archive_cache_is_refused_without_materialization(
    tmp_path: Path,
) -> None:
    """A known handle with unknown cache state cannot silently count as unloaded."""
    base = _make_values_only_solution(value=[1.0], dtype="float32")
    path = save_solution(solution=base, path=tmp_path / "solution.h5")
    solution = load_solution(path=path)
    entry = cast(
        "persistence._LazyHdf5Entry",
        cast("ValueStore", solution.values)._raw(period=0, regime="working"),
    )
    entry._cache.value = object()
    with pytest.raises(ExecutionPlanningError, match="unsupported retained"):
        _buffers(solution)


def test_loaded_pytree_cache_includes_its_template_without_callbacks(
    tmp_path: Path,
) -> None:
    """A loaded custom artifact keeps both cache payload and reconstruction leaves."""
    original, ref = _make_stateful_pytree_solution()
    path = save_solution(solution=original, path=tmp_path / "pytree.h5")
    solution = load_solution(path=path)
    solution.replay_artifacts.materialize(
        ref,
        template=original._artifact_authority[ref].template,
    )
    entry = cast("persistence._LazyHdf5Entry", solution.replay_artifacts._raw(ref))
    cached = cast("persistence._LoadedEntryPayload", entry._cache.value)
    assert cached.template_snapshot is not None
    expected = (*cached.leaves, *cached.template_snapshot.leaves)
    path.unlink()
    _StatefulPersistenceTree.reset()
    try:
        actual = _buffers(solution)
        assert {id(array) for array in actual} == {id(array) for array in expected}
        assert _StatefulPersistenceTree.flatten_sources == []
        assert _StatefulPersistenceTree.unflatten_count == 0
    finally:
        _StatefulPersistenceTree.reset()


def test_builtin_nested_policy_metadata_has_no_hidden_payload() -> None:
    """The certified rational inverse is part of the supported closed policy record."""
    base = _make_solution()
    ref = next(iter(base.replay_artifacts))
    metadata = OuterReplayCapability(
        inverse=DeclaredOuterInverse(coefficient=Fraction(1), low=0.0, high=1.0),
        undeclared_functions=(),
        unbindable_functions=(),
        unavailable_keeper_states=(),
        unaddressable_passive_states=(),
        unaddressable_discrete_actions=(),
    )
    solution = replace(base, replay_artifacts=ArtifactStore({ref: metadata}))
    assert {id(array) for array in _buffers(solution)} == {
        id(array) for array in _backing_values(solution)
    }
