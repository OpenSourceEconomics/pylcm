"""Lazy, exactly-addressed value and artifact stores."""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Any,
    TypeAlias,
    cast,
)

from lcm._solver_api.contract import (
    ArtifactRef,
    _same_exact_artifact_contract,
)
from lcm._solver_api.entries import (
    _canonical_value_entry,
    _CanonicalValueEntry,
    _copy_solution_value,
    _LazyEntry,
    _materialize_entry,
)
from lcm._solver_api.identity import (
    ArtifactKey,
    LoadState,
)
from lcm.typing import FloatND, RegimeName

if TYPE_CHECKING:
    _FloatValueBoundary: TypeAlias = FloatND  # noqa: UP040
    _RegimeValuesBoundary: TypeAlias = Mapping[RegimeName, FloatND]  # noqa: UP040
    _MaterializedValuesBoundary: TypeAlias = MappingProxyType[  # noqa: UP040
        int, MappingProxyType[str, FloatND]
    ]
    _ValueStoreBoundary: TypeAlias = "ValueStore"  # noqa: UP040
    _ArtifactStoreBoundary: TypeAlias = "ArtifactStore"  # noqa: UP040
    _ValuePeriodBoundary: TypeAlias = int  # noqa: UP040
    _RegimeNameBoundary: TypeAlias = RegimeName  # noqa: UP040
    _ArtifactRefBoundary: TypeAlias = ArtifactRef  # noqa: UP040
    _ArtifactKeyBoundary: TypeAlias = ArtifactKey  # noqa: UP040
else:
    # Lazy stores own their validation and materialization boundaries. Runtime
    # annotation traversal would load them before those explicit checks run.
    _FloatValueBoundary = object
    _RegimeValuesBoundary = object
    _MaterializedValuesBoundary = object
    _ValueStoreBoundary = object
    _ArtifactStoreBoundary = object
    _ValuePeriodBoundary = object
    _RegimeNameBoundary = object
    _ArtifactRefBoundary = object
    _ArtifactKeyBoundary = object


def _traverse_public_mapping_items(
    *, mapping: object, label: str
) -> list[tuple[object, object]]:
    """Consume one item traversal of a public mapping into an owned list of pairs.

    A public mapping may be any `Mapping` implementation, so its key view, item
    view, and length can disagree or execute backing code. Store constructors read
    this one item traversal and nothing else, and check every raw key exactly before
    it is hashed into a store, so an equality alias (`True` for `1`) or a repeated
    address cannot contract away unseen.
    """
    if not isinstance(mapping, Mapping):
        raise TypeError(f"{label} must be a mapping.")
    try:
        items = list(mapping.items())
    except Exception as error:
        raise TypeError(f"{label} cannot be traversed as mapping items.") from error
    for item in items:
        if type(item) is not tuple or len(item) != 2:  # noqa: PLR2004
            raise TypeError(f"{label} items must be exact key-value pairs.")
    return items


def _require_exact_value_period(period: object) -> int:
    if type(period) is not int:
        raise TypeError("Value periods must be exact ints.")
    if period < 0:
        raise ValueError("Value periods must be nonnegative.")
    return period


def _require_exact_regime_name(regime: object) -> RegimeName:
    if type(regime) is not str:
        raise TypeError("Value regime names must be exact strs.")
    if not regime:
        raise ValueError("Value regime names must not be empty.")
    return regime


def _require_exact_artifact_key(key: object) -> ArtifactKey:
    if type(key) is not ArtifactKey:
        raise TypeError("Artifact keys must be exact ArtifactKey objects.")
    if type(key.type_id) is not str or not key.type_id:
        raise TypeError("Artifact key type_id must be a nonempty exact str.")
    if type(key.schema_version) is not int or key.schema_version < 1:
        raise TypeError("Artifact key schema_version must be a positive exact int.")
    return key


def _require_exact_artifact_ref(ref: object) -> ArtifactRef:
    if type(ref) is not ArtifactRef:
        raise TypeError("Artifact addresses must be exact ArtifactRef objects.")
    _require_exact_value_period(ref.period)
    _require_exact_regime_name(ref.regime)
    _require_exact_artifact_key(ref.key)
    return ref


@dataclass(frozen=True, eq=False)
class _ValuePeriodView(Mapping[RegimeName, FloatND]):
    """Read-through view of one period in a :class:`ValueStore`."""

    store: _ValueStoreBoundary
    period: int

    def __getitem__(self, regime: _RegimeNameBoundary) -> _FloatValueBoundary:
        return self.store._load(period=self.period, regime=regime)  # noqa: SLF001

    def __iter__(self) -> Iterator[RegimeName]:
        return iter(self.store._regimes_by_period[self.period])  # noqa: SLF001

    def __len__(self) -> int:
        return len(self.store._regimes_by_period[self.period])  # noqa: SLF001

    def __contains__(self, regime: object) -> bool:
        """Check one regime coordinate without materializing its value."""
        if type(regime) is not str or not regime:
            return False
        return regime in self.store._regimes_by_period[self.period]  # noqa: SLF001


def _admit_value_entry(
    *,
    period: object,
    regime: object,
    value: object,
    entries: dict[tuple[int, RegimeName], object],
    regimes_by_period: dict[int, list[RegimeName]],
) -> None:
    """Check one raw coordinate exactly and for uniqueness, then own its value."""
    coordinate = (
        _require_exact_value_period(period),
        _require_exact_regime_name(regime),
    )
    if coordinate in entries:
        raise ValueError(f"ValueStore coordinate {coordinate!r} appears twice.")
    entries[coordinate] = (
        value
        if isinstance(value, _LazyEntry) and type(value) is not _CanonicalValueEntry
        else _canonical_value_entry(value=value)
    )
    regimes_by_period.setdefault(coordinate[0], []).append(coordinate[1])


@dataclass(frozen=True, eq=False)
class ValueStore(Mapping[int, Mapping[RegimeName, FloatND]]):
    """Immutable, independently materializable value-function store.

    Eager solves and restored archives expose the same mapping interface.  The
    latter keep a lazy entry per ``(period, regime)``; inspecting coordinates or
    load state never reads a numerical payload.
    """

    _entries: Mapping[object, object] = field(default_factory=dict, repr=False)
    _regimes_by_period: Mapping[int, tuple[RegimeName, ...]] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        # The mapping is read through exactly one item traversal. Its form — flat
        # ``(period, regime)`` tuples or ``period -> regime -> value`` — is decided
        # from those same items, and every raw coordinate is checked exactly and for
        # uniqueness before it is inserted, so ``True`` cannot overwrite ``1`` and a
        # repeated address cannot be contracted into one.
        items = _traverse_public_mapping_items(
            mapping=self._entries, label="ValueStore entries"
        )
        is_flat = [type(key) is tuple for key, _ in items]
        if any(is_flat) and not all(is_flat):
            raise TypeError(
                "ValueStore entries must be keyed either by (period, regime) tuples "
                "or by periods, not by both."
            )

        entries: dict[tuple[int, RegimeName], object] = {}
        regimes_by_period: dict[int, list[RegimeName]] = {}

        if all(is_flat):
            for coordinate, value in items:
                typed_coordinate = cast("tuple[object, ...]", coordinate)
                if len(typed_coordinate) != 2:  # noqa: PLR2004
                    raise ValueError("A ValueStore coordinate must have two entries.")
                _admit_value_entry(
                    entries=entries,
                    regimes_by_period=regimes_by_period,
                    period=typed_coordinate[0],
                    regime=typed_coordinate[1],
                    value=value,
                )
        else:
            for period, regime_to_value in items:
                exact_period = _require_exact_value_period(period)
                if not isinstance(regime_to_value, Mapping):
                    raise TypeError(
                        f"ValueStore period {exact_period} must map to a mapping of "
                        "regime names to values."
                    )
                for regime, value in _traverse_public_mapping_items(
                    mapping=regime_to_value,
                    label=f"ValueStore period {exact_period} entries",
                ):
                    _admit_value_entry(
                        entries=entries,
                        regimes_by_period=regimes_by_period,
                        period=exact_period,
                        regime=regime,
                        value=value,
                    )
        object.__setattr__(self, "_entries", MappingProxyType(entries))
        object.__setattr__(
            self,
            "_regimes_by_period",
            MappingProxyType(
                {
                    period: tuple(regimes)
                    for period, regimes in regimes_by_period.items()
                }
            ),
        )

    def __getitem__(self, period: _ValuePeriodBoundary) -> _RegimeValuesBoundary:
        period = _require_exact_value_period(period)
        if period not in self._regimes_by_period:
            raise KeyError(period)
        return _ValuePeriodView(store=self, period=period)

    def __iter__(self) -> Iterator[int]:
        return iter(self._regimes_by_period)

    def __len__(self) -> int:
        return len(self._regimes_by_period)

    def __contains__(self, period: object) -> bool:
        """Check one period coordinate without materializing a value."""
        if type(period) is not int or period < 0:
            return False
        return period in self._regimes_by_period

    def _load(
        self,
        *,
        period: _ValuePeriodBoundary,
        regime: _RegimeNameBoundary,
    ) -> _FloatValueBoundary:
        period = _require_exact_value_period(period)
        regime = _require_exact_regime_name(regime)
        entry = self._entries[(period, regime)]
        value = _materialize_entry(entry=entry)
        if type(entry) is not _CanonicalValueEntry:
            value = _copy_solution_value(
                value=value,
                label=f"Solution value at period={period}, regime={regime!r}",
            )
        return cast("FloatND", value)

    def _raw(
        self, *, period: _ValuePeriodBoundary, regime: _RegimeNameBoundary
    ) -> object:
        """Return one eager value or lazy handle without materializing it."""
        period = _require_exact_value_period(period)
        regime = _require_exact_regime_name(regime)
        return self._entries[(period, regime)]

    def load_state(
        self, *, period: _ValuePeriodBoundary, regime: _RegimeNameBoundary
    ) -> LoadState:
        """Return one value entry's state without materializing it."""
        period = _require_exact_value_period(period)
        regime = _require_exact_regime_name(regime)
        entry = self._entries[(period, regime)]
        return entry.load_state if isinstance(entry, _LazyEntry) else LoadState.LOADED

    def materialize(self) -> _MaterializedValuesBoundary:
        """Return an exact immutable built-in snapshot of every value entry."""
        return MappingProxyType(
            {
                period: MappingProxyType(
                    {
                        regime: self._load(period=period, regime=regime)
                        for regime in view
                    }
                )
                for period, view in self.items()
            }
        )


@dataclass(frozen=True, eq=False)
class ArtifactStore(Mapping[ArtifactRef, object]):
    """Immutable store of explicitly addressed solution artifacts.

    The mapping interface keeps artifacts solver-extensible. ``project`` gives engine
    consumers the nested ``period -> regime -> payload`` view for one known artifact
    key.
    """

    # Typed with ``Any`` keys so the runtime annotation check does not traverse the
    # key view; ``__post_init__`` is the one admission boundary.
    _entries: Mapping[Any, object] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        # One item traversal; each raw address is an exact ``ArtifactRef`` and unique
        # before it is inserted, so no equality alias or repeat can contract.
        entries: dict[ArtifactRef, object] = {}
        for raw_ref, payload in _traverse_public_mapping_items(
            mapping=self._entries, label="ArtifactStore entries"
        ):
            ref = _require_exact_artifact_ref(raw_ref)
            if ref in entries:
                raise ValueError(f"ArtifactStore address {ref!r} appears twice.")
            entries[ref] = payload
        object.__setattr__(self, "_entries", MappingProxyType(entries))

    def __getitem__(self, ref: _ArtifactRefBoundary) -> object:
        ref = _require_exact_artifact_ref(ref)
        return _materialize_entry(entry=self._entries[ref])

    def __iter__(self) -> Iterator[ArtifactRef]:
        return iter(cast("Mapping[ArtifactRef, object]", self._entries))

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, ref: object) -> bool:
        """Check one artifact address without materializing its payload."""
        try:
            ref = _require_exact_artifact_ref(ref)
        except TypeError, ValueError:
            return False
        return ref in self._entries

    def project(
        self, key: _ArtifactKeyBoundary
    ) -> Mapping[int, Mapping[RegimeName, object]]:
        """Project one artifact schema to an immutable nested period mapping."""
        key = _require_exact_artifact_key(key)
        projected: dict[int, dict[RegimeName, object]] = {}
        for ref in self._entries:
            _require_exact_artifact_ref(ref)
            if _same_exact_artifact_contract(actual=ref.key, expected=key):
                projected.setdefault(ref.period, {})[ref.regime] = self[ref]
        return MappingProxyType(
            {
                period: MappingProxyType(regime_to_payload)
                for period, regime_to_payload in sorted(projected.items())
            }
        )

    def _raw(self, ref: _ArtifactRefBoundary) -> object:
        """Return one eager payload or lazy handle without materializing it."""
        ref = _require_exact_artifact_ref(ref)
        return self._entries[ref]

    def load_state(self, ref: _ArtifactRefBoundary) -> LoadState:
        """Return one artifact entry's state without materializing it."""
        ref = _require_exact_artifact_ref(ref)
        entry = self._entries[ref]
        return entry.load_state if isinstance(entry, _LazyEntry) else LoadState.LOADED

    # keyword-only-exempt: primary-argument=ref
    def materialize(
        self, ref: _ArtifactRefBoundary, *, template: object | None = None
    ) -> object:
        """Load one entry and optionally rebuild its declared PyTree shape."""
        ref = _require_exact_artifact_ref(ref)
        return _materialize_entry(
            entry=self._entries[ref],
            template=template,
        )

    # keyword-only-exempt: primary-argument=ref
    def _materialize_from_template_snapshot(
        self,
        ref: _ArtifactRefBoundary,
        *,
        template_snapshot: object,
    ) -> object:
        """Load one entry through an engine-owned cached PyTree declaration."""
        ref = _require_exact_artifact_ref(ref)
        return _materialize_entry(
            entry=self._entries[ref],
            template_snapshot=template_snapshot,
        )
