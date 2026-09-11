"""Lazy entries that materialize a canonical value or artifact payload once."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import jax
import numpy as np

from lcm._solver_api.authority import (
    ArtifactAuthority,
    _ArrayCopier,
    _artifact_authority_template_snapshot,
    _CanonicalArtifactTemplate,
    _canonicalize_artifact_payload_snapshot,
    _copy_artifact_array_leaf,
    _reconstruct_artifact_from_template_snapshot,
)
from lcm._solver_api.identity import (
    LoadState,
)


@runtime_checkable
class _ValueMaterializer(Protocol):  # noqa: PYI046 — private store boundary protocol
    """Call-local trusted value loader; never retained by a public store."""

    def __call__(self, *, entry: object) -> object:
        """Materialize one explicitly admitted entry with its private ownership."""
        ...


class _LazyEntry(ABC):
    """Private implementation hook shared by value and artifact stores."""

    @property
    @abstractmethod
    def load_state(self) -> LoadState:
        """Return the materialization state without loading the entry."""

    @abstractmethod
    def materialize(self, *, template: object | None = None) -> object:
        """Load and verify the entry, optionally rebuilding a declared PyTree."""

    def materialize_from_template_snapshot(
        self,
        *,
        template_snapshot: object,
    ) -> object:
        """Fallback for lazy implementations that only consume a template object."""
        if type(template_snapshot) is not _CanonicalArtifactTemplate:
            raise TypeError("Lazy materialization requires an exact template snapshot.")
        template = _reconstruct_artifact_from_template_snapshot(
            template_snapshot=template_snapshot,
            leaves=tuple(template_snapshot.leaves),
        )
        return self.materialize(template=template)


def _materialize_entry(
    *,
    entry: object,
    template: object | None = None,
    template_snapshot: object | None = None,
) -> object:
    """Materialize an internal lazy entry while leaving eager objects untouched."""
    if isinstance(entry, _LazyEntry):
        if template is not None and template_snapshot is not None:
            raise TypeError("Supply a template or a template snapshot, not both.")
        if template_snapshot is not None:
            return entry.materialize_from_template_snapshot(
                template_snapshot=template_snapshot
            )
        return entry.materialize(template=template)
    return entry


@dataclass(frozen=True, slots=True, kw_only=True)
class _CanonicalArtifactEntry(_LazyEntry):
    """Private artifact state that returns a fresh detached graph on every read."""

    plan_snapshot: _CanonicalArtifactTemplate
    leaves: tuple[jax.Array, ...]

    @property
    def load_state(self) -> LoadState:
        """An owned eager entry is already loaded."""
        return LoadState.LOADED

    def _fresh(
        self,
        *,
        template_snapshot: _CanonicalArtifactTemplate,
    ) -> object:
        """Copy private buffers and reconstruct without a plugin callback."""
        if type(template_snapshot) is not _CanonicalArtifactTemplate:
            raise TypeError("Owned artifact reconstruction requires an exact snapshot.")
        if template_snapshot.leaf_paths != self.plan_snapshot.leaf_paths:
            raise TypeError("Owned artifact TreePaths differ from model authority.")
        if len(self.leaves) != len(template_snapshot.leaves):
            raise TypeError("Owned artifact leaf count differs from model authority.")
        copied: list[jax.Array] = []
        for index, (leaf, expected) in enumerate(
            zip(self.leaves, template_snapshot.leaves, strict=True)
        ):
            if not isinstance(leaf, jax.Array) or leaf.is_deleted():
                raise TypeError("Owned artifact private leaf was deleted.")
            if tuple(leaf.shape) != tuple(expected.shape) or np.dtype(
                leaf.dtype
            ) != np.dtype(expected.dtype):
                raise TypeError("Owned artifact leaf differs from model authority.")
            copied.append(
                _copy_artifact_array_leaf(
                    leaf=leaf,
                    label=f"Owned artifact leaf {index}",
                )
            )
        return _reconstruct_artifact_from_template_snapshot(
            template_snapshot=template_snapshot,
            leaves=tuple(copied),
        )

    def materialize(self, *, template: object | None = None) -> object:
        """Return a fresh graph; eager-entry compatibility ignores raw templates."""
        del template
        return self._fresh(template_snapshot=self.plan_snapshot)

    def materialize_from_template_snapshot(
        self,
        *,
        template_snapshot: object,
    ) -> object:
        """Return a fresh graph through the current model-authoritative plan."""
        if type(template_snapshot) is not _CanonicalArtifactTemplate:
            raise TypeError("Owned artifact reconstruction requires an exact snapshot.")
        return self._fresh(template_snapshot=template_snapshot)


def _canonical_artifact_entry_from_authority(
    *,
    payload: object,
    authority: ArtifactAuthority,
    borrow: bool = False,
) -> _CanonicalArtifactEntry:
    """Detach one eager artifact before any other result callback may run.

    With `borrow` set the entry keeps the validated leaves themselves rather than
    private copies: the engine uses it for buffers its own solve allocated.
    """
    canonical = _canonicalize_artifact_payload_snapshot(
        payload=payload,
        authority=authority,
        borrow=borrow,
    )
    plan_snapshot = _artifact_authority_template_snapshot(authority)
    if plan_snapshot is None:
        raise TypeError("Model authority supplies no artifact reconstruction plan.")
    private_leaves = (
        canonical.leaves
        if borrow
        else tuple(
            _copy_artifact_array_leaf(
                leaf=leaf,
                label=f"Owned artifact private leaf {index}",
            )
            for index, leaf in enumerate(canonical.leaves)
        )
    )
    return _CanonicalArtifactEntry(
        plan_snapshot=plan_snapshot,
        leaves=private_leaves,
    )


def _copy_solution_value(
    *, value: object, label: str, array_copier: _ArrayCopier | None = None
) -> object:
    """Copy one numerical value without changing its concrete representation."""
    if isinstance(value, jax.Array):
        if array_copier is None:
            return _copy_artifact_array_leaf(leaf=value, label=label)
        return _copy_artifact_array_leaf(
            leaf=value, label=label, array_copier=array_copier
        )
    if isinstance(value, np.ndarray):
        copied = np.array(value, copy=True, order="K", subok=False)
        if not (
            np.issubdtype(copied.dtype, np.number)
            or np.issubdtype(copied.dtype, np.bool_)
        ):
            raise TypeError(f"{label} is not numerical or Boolean.")
        copied.flags.writeable = False
        return copied
    if isinstance(value, np.generic):
        if not (
            np.issubdtype(value.dtype, np.number)
            or np.issubdtype(value.dtype, np.bool_)
        ):
            raise TypeError(f"{label} is not numerical or Boolean.")
        return np.array(value, copy=True)[()]
    if any(type(value) is allowed for allowed in (bool, int, float, complex)):
        return value
    raise TypeError(f"{label} is not a supported numerical value.")


@dataclass(frozen=True, slots=True, kw_only=True)
class _CanonicalValueEntry(_LazyEntry):
    """Private numerical state that returns a fresh value on every read."""

    value: object

    @property
    def load_state(self) -> LoadState:
        """An owned eager value is already loaded."""
        return LoadState.LOADED

    def materialize(self, *, template: object | None = None) -> object:
        """Return an independent numerical value."""
        del template
        return self._fresh()

    def _fresh(self, *, array_copier: _ArrayCopier | None = None) -> object:
        """Copy with an optional call-local allocator, without storing it."""
        if array_copier is None:
            return _copy_solution_value(value=self.value, label="Owned solution value")
        return _copy_solution_value(
            value=self.value, label="Owned solution value", array_copier=array_copier
        )


def _canonical_value_entry(
    *, value: object, array_copier: _ArrayCopier | None = None
) -> _CanonicalValueEntry:
    """Detach one eager value before any lazy result callback may run."""
    if type(value) is _CanonicalValueEntry:
        source = (
            value.materialize()
            if array_copier is None
            else value._fresh(array_copier=array_copier)  # noqa: SLF001
        )
    else:
        source = value
    private = (
        _copy_solution_value(value=source, label="Solution value")
        if array_copier is None
        else _copy_solution_value(
            value=source, label="Solution value", array_copier=array_copier
        )
    )
    return _CanonicalValueEntry(value=private)
