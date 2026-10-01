"""Call-local admission for built-in, independently verified archive payloads.

Public store code receives only the structural loader protocol. This engine boundary
recognizes the exact native handle and never dispatches an arbitrary lazy decoder.
Value arrays and model-authoritative artifact PyTrees share one writer and copier:
each archive leaf is verified on the host before its admitted upload, and every
detached copy is admitted against the same device budget.
"""

from dataclasses import dataclass
from typing import cast

from _lcm.dtypes import CanonicalArrayWriter
from lcm._solver_api.authority import _ArrayCopier, _CanonicalArtifactTemplate
from lcm._solver_api.entries import _CanonicalArtifactEntry
from lcm.exceptions import ExecutionPlanningError


@dataclass(frozen=True, kw_only=True)
class NativeValueMaterializer:
    """Transient upload/copy dependencies, absent from entries and result memos."""

    array_writer: CanonicalArrayWriter
    """Own and admit a verified host leaf on the first selected device."""
    array_copier: _ArrayCopier
    """Own and admit each detached copy in the cached array's exact layout."""

    @staticmethod
    def require_entry(*, entry: object) -> None:
        """Reject unsupported decoders before any archive leaf is read."""
        # Persistence imports result snapshots; resolve its exact type at call time.
        from _lcm.persistence.solution import _LazyHdf5Entry  # noqa: PLC0415

        if (
            type(entry) is not _LazyHdf5Entry
            or entry.payload_kind != "array"
            or len(entry.leaves) != 1
            or entry.standard_template_snapshot is not None
        ):
            raise ExecutionPlanningError(
                "Budgeted native materialization requires an exact built-in "
                "single-array value entry; arbitrary decoders and artifacts "
                "are not profiled."
            )

    def __call__(
        self,
        *,
        entry: object,
        template_snapshot: _CanonicalArtifactTemplate | None = None,
    ) -> object:
        """Verify/upload one cache bank, then return an admitted detached copy.

        Without `template_snapshot` the entry must be a single-array value; with it,
        an archive artifact PyTree is rebuilt in that model-authoritative layout.
        """
        from _lcm.persistence.solution import _LazyHdf5Entry  # noqa: PLC0415

        if template_snapshot is None:
            self.require_entry(entry=entry)
        native = cast("_LazyHdf5Entry", entry)
        return native._materialize(  # noqa: SLF001 — trusted native engine boundary
            template=None,
            template_snapshot=template_snapshot,
            array_writer=self.array_writer,
            array_copier=self.array_copier,
        )

    def materialize_artifact(
        self, *, entry: object, template_snapshot: object
    ) -> object:
        """Return an admitted detached graph of one archive or engine-owned artifact.

        An unloaded archive entry verifies each host leaf, then uploads it through the
        admitted writer into its private cache; every returned leaf is an admitted
        copy. Raw payloads and arbitrary lazy decoders are refused before any read.
        """
        from _lcm.persistence.solution import _LazyHdf5Entry  # noqa: PLC0415

        if type(template_snapshot) is not _CanonicalArtifactTemplate:
            raise TypeError("Admitted artifact materialization requires a snapshot.")
        if type(entry) is _LazyHdf5Entry:
            return self(entry=entry, template_snapshot=template_snapshot)
        if type(entry) is _CanonicalArtifactEntry:
            return entry.materialize_from_template_snapshot(
                template_snapshot=template_snapshot, array_copier=self.array_copier
            )
        raise ExecutionPlanningError(
            "Budgeted artifact materialization requires a built-in archive entry or "
            "an engine-owned payload; raw payloads and arbitrary decoders are not "
            "profiled."
        )
