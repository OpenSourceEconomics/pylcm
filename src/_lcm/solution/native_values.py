"""Call-local admission for built-in, independently verified archive value arrays.

Public store code receives only the structural loader protocol. This engine boundary
recognizes the exact native handle and never dispatches an arbitrary lazy decoder.
"""

from dataclasses import dataclass
from typing import cast

from _lcm.dtypes import CanonicalArrayWriter
from lcm._solver_api.authority import _ArrayCopier
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

    def __call__(self, *, entry: object) -> object:
        """Verify/upload one cache bank, then return an admitted detached copy."""
        from _lcm.persistence.solution import _LazyHdf5Entry  # noqa: PLC0415

        self.require_entry(entry=entry)
        native = cast("_LazyHdf5Entry", entry)
        return native._materialize(  # noqa: SLF001 — trusted native engine boundary
            template=None,
            template_snapshot=None,
            array_writer=self.array_writer,
            array_copier=self.array_copier,
        )
