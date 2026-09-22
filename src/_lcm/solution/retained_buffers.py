"""Observe known retained solution payloads without decoding or copying them.

This budget-only ownership boundary reads exact built-in stores, canonical entries,
HDF5 cache state and authority bindings. It never uses public store getters, template
snapshot accessors, plugin PyTree callbacks or lazy decoders. Unknown lazy owners and
opaque raw payload types are refused, even if they report themselves unloaded.

The returned tuple temporarily borrows existing JAX arrays. Measure it immediately,
discard it, and retain only footprint metadata while the original owners stay alive.
Call after solution validation has populated its caches and consumed views; subsequent
cache growth needs another observation. Locks protect each cache/binding observation,
not concurrent mutation of an entire SolutionResult by another consumer.

Supported payloads are numerical host leaves, plain containers, canonical entry leaf
banks and the explicitly listed built-in policy/diagnostics records. Prepared replay
readers expose their snapshot payloads, authority templates and grid context. Route
code, arbitrary callable closure captures, allocator/executable storage and other
models' global state are outside this known payload inventory. Canonical descriptive
metadata contains no payload arrays. Host NumPy storage is not JAX device storage.
"""

from dataclasses import dataclass, field, fields
from fractions import Fraction
from types import MappingProxyType
from typing import cast

import jax
import numpy as np

from _lcm.egm.carry import EGMCarry
from _lcm.egm.nested_published_policy import NestedEGMSimPolicy, OuterPolicyBank
from _lcm.egm.outer_inversion import DeclaredOuterInverse
from _lcm.egm.outer_replay_capability import OuterReplayCapability
from _lcm.egm.published_policy import EGMSimPolicy, NBEGMGridPolicy, NNBEGMSimPolicy
from _lcm.solution.artifacts import OwnedSolutionView
from _lcm.solution.model_authority import SolutionAuthority
from _lcm.solution.solver_diagnostics import SolverDiagnostics
from lcm._solver_api import authority as authority_module
from lcm._solver_api.entries import (
    _CanonicalArtifactEntry,
    _CanonicalValueEntry,
    _LazyEntry,
)
from lcm._solver_api.result import SolutionResult
from lcm._solver_api.stores import ArtifactStore, ValueStore
from lcm.exceptions import ExecutionPlanningError

_PAYLOAD_RECORDS = (
    EGMCarry,
    EGMSimPolicy,
    NBEGMGridPolicy,
    NNBEGMSimPolicy,
    NestedEGMSimPolicy,
    OuterPolicyBank,
    OuterReplayCapability,
    DeclaredOuterInverse,
    SolverDiagnostics,
)

type _PayloadRecord = (
    EGMCarry
    | EGMSimPolicy
    | NBEGMGridPolicy
    | NNBEGMSimPolicy
    | NestedEGMSimPolicy
    | OuterPolicyBank
    | OuterReplayCapability
    | SolverDiagnostics
    | DeclaredOuterInverse
)


def retained_solution_buffers(*, solution: object) -> tuple[jax.Array, ...]:
    """Borrow all known arrays held by an already-resolved exact solution result."""
    if type(solution) is not SolutionResult:
        raise _unsupported(solution)
    collector = _RetainedBuffers()
    for store in (
        solution.values,
        solution.retained_continuations,
        solution.replay_artifacts,
        solution.auxiliary_artifacts,
        solution.diagnostics,
    ):
        collector.collect(store)
    collector.collect(solution._artifact_authority)  # noqa: SLF001
    collector.collect(solution._engine_view)  # noqa: SLF001
    collector.collect(solution._consumed_views)  # noqa: SLF001
    return tuple(collector.arrays.values())


@dataclass(kw_only=True)
class _RetainedBuffers:
    """One transient traversal; no array-bearing state escapes except its result."""

    arrays: dict[int, jax.Array] = field(default_factory=dict)
    seen: set[int] = field(default_factory=set)

    def collect(self, value: object) -> None:  # noqa: C901, PLR0912
        """Read only explicit payload owners, never an arbitrary object's fields."""
        if isinstance(value, jax.Array):
            self.arrays[id(value)] = value
            return
        if type(value) in (type(None), bool, int, float, complex, str, bytes, Fraction):
            return
        if isinstance(value, np.ndarray | np.generic):
            if value.dtype.hasobject:
                raise _unsupported(value)
            return
        if id(value) in self.seen:
            return
        self.seen.add(id(value))
        if type(value) in (ValueStore, ArtifactStore):
            store = cast("ValueStore | ArtifactStore", value)
            self.collect(store._entries)  # noqa: SLF001
        elif type(value) in (dict, MappingProxyType):
            mapping = cast(
                "dict[object, object] | MappingProxyType[object, object]", value
            )
            for child in mapping.values():
                self.collect(child)
        elif type(value) in (tuple, list):
            for child in cast("tuple[object, ...] | list[object]", value):
                self.collect(child)
        elif type(value) is _CanonicalValueEntry:
            self.collect(value.value)
        elif type(value) is _CanonicalArtifactEntry:
            self.collect(value.leaves)
            self.collect(value.plan_snapshot.leaves)
        elif isinstance(value, _LazyEntry):
            self.collect_lazy(value)
        elif type(value) is authority_module.ArtifactAuthority:
            self.collect_authority(value)
        elif type(value) is SolutionAuthority:
            self.collect(value.artifacts)
        elif type(value) is OwnedSolutionView:
            for name in (
                "values",
                "simulation_policies",
                "dissolution_flags",
                "replay_artifacts",
                "authority",
            ):
                self.collect(getattr(value, name))
        elif type(value) in _PAYLOAD_RECORDS:
            for descriptor in fields(cast("_PayloadRecord", value)):
                self.collect(getattr(value, descriptor.name))
        else:
            self.collect_reader(value)

    def collect_lazy(self, entry: _LazyEntry) -> None:
        """Pin a known archive cache without calling its materialization API."""
        # The archive imports solution ownership; resolve this type only at call time.
        from _lcm.persistence import solution as persistence  # noqa: PLC0415

        if type(entry) is not persistence._LazyHdf5Entry:  # noqa: SLF001
            raise _unsupported(entry)
        with entry._cache.lock:  # noqa: SLF001
            cached = entry._cache.value  # noqa: SLF001
        if entry.standard_template_snapshot is not None:
            self.collect(entry.standard_template_snapshot.leaves)
        if cached is persistence._UNLOADED:  # noqa: SLF001
            return
        if type(cached) is not persistence._LoadedEntryPayload:  # noqa: SLF001
            raise _unsupported(cached)
        self.collect(cached.leaves)
        if cached.template_snapshot is not None:
            self.collect(cached.template_snapshot.leaves)

    def collect_authority(self, authority: authority_module.ArtifactAuthority) -> None:
        """Observe both identity-bound leaf banks without the copying accessor."""
        with authority_module._ARTIFACT_AUTHORITY_TEMPLATE_LOCK:  # noqa: SLF001
            binding = authority_module._ARTIFACT_AUTHORITY_TEMPLATE_BINDINGS.get(  # noqa: SLF001
                id(authority)
            )
            if binding is None or binding.authority_ref() is not authority:
                raise _unsupported(authority)
            if (
                authority.template is not binding.template
                or authority.payload_runtime_type is not binding.payload_runtime_type
                or authority.container_runtime_types
                is not binding.container_runtime_types
                or authority.leaves is not binding.leaves
            ):
                raise _unsupported(authority)
            public = binding.public_template_leaves
            snapshot = binding.snapshot
        self.collect(public)
        if snapshot is not None:
            self.collect(snapshot.leaves)

    def collect_reader(self, value: object) -> None:
        """Read prepared snapshot/context roots without building the route's reader."""
        # Model imports the ownership boundary before simulation readers are complete.
        from _lcm.simulation.replay_inputs import PreparedReplayReader  # noqa: PLC0415

        if type(value) is not PreparedReplayReader:
            raise _unsupported(value)
        self.collect(value.snapshot.artifacts)
        self.collect(value.snapshot.authorities)
        self.collect(value.context.state_nodes)
        self.collect(value.context.action_nodes)


def _unsupported(value: object) -> ExecutionPlanningError:
    """Refuse incomplete budget provenance rather than reporting fictitious zero."""
    return ExecutionPlanningError(
        "Budgeted simulation encountered unsupported retained solution storage: "
        f"{type(value).__module__}.{type(value).__qualname__}."
    )
