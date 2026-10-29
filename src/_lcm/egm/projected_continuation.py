"""Pre-shock value publication, separate from native resources-row replay."""

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, fields
from types import MappingProxyType
from typing import TYPE_CHECKING, NamedTuple, Protocol, cast, runtime_checkable

import jax
import jax.numpy as jnp

from _lcm.egm.query_data import QueryNodeKind
from lcm._solver_api.replay import ContinuationLeaf
from lcm.exceptions import UnsupportedOperationError
from lcm.solver_api import ArtifactKey, ContinuationCapabilities
from lcm.typing import (
    BoolND,
    Float1D,
    FloatND,
    Int1D,
    IntND,
    RegimeName,
    ScalarBool,
    ScalarFloat,
    ScalarInt,
    StateName,
)

if TYPE_CHECKING:
    from _lcm.typing import PytreeChild
else:
    # The engine type vocabulary imports its continuation payloads.
    type PytreeChild = object

PROJECTED_EGM_CONTINUATION = ArtifactKey(
    type_id="pylcm.egm.pre_shock_continuation", schema_version=1
)


@dataclass(frozen=True, kw_only=True)
class ProjectedChildAuthority:
    """Actual producer schema owning a pre-shock-domain diagnostic slot.

    This is not borrowing-floor authority: the reader consumes the projected
    child liquid coordinate directly and makes no conditional floor read.
    Date/key/schema authority does not grant smooth links or complete coverage.
    """

    producer: RegimeName
    """Regime publishing the original projected artifact."""
    period: int
    """Exact producing period; consumers cannot substitute another date."""
    liquid_state: StateName
    """Physical pre-shock coordinate of the explicit economic marginal."""
    capacity: int
    """Padded producer projection capacity, including one-sided records."""
    artifact_key: ArtifactKey = PROJECTED_EGM_CONTINUATION
    """Actual producer artifact identity, never an EGM resources-row key."""
    domain_paths: tuple[tuple[str, ...], tuple[str, ...]] = (
        ("domain_lower",),
        ("domain_upper",),
    )
    """Original stored leaves defining each row's closed supported domain."""

    def __post_init__(self) -> None:
        if (
            type(self.period) is not int
            or self.period < 0
            or type(self.capacity) is not int
            or self.capacity < _MIN_PROJECTION_CAPACITY
            or not self.producer
            or not self.liquid_state
            or self.artifact_key != PROJECTED_EGM_CONTINUATION
            or self.domain_paths != (("domain_lower",), ("domain_upper",))
        ):
            raise ValueError("Invalid projected-child producer authority.")


class ProjectedRowRead(NamedTuple):
    """A piecewise-affine level and independently published economic marginal."""

    value: FloatND
    economic_marginal: FloatND
    owner_id: IntND
    side_id: IntND
    valid: BoolND

    def __reduce__(
        self,
    ) -> tuple[
        Callable[[None, Iterable[PytreeChild]], ProjectedRowRead],
        tuple[None, tuple[PytreeChild, ...]],
    ]:
        """Retain the fixed keyed schema through original-buffer reconstruction."""
        return _unflatten_projected_row_read, (None, tuple(self))


@dataclass(frozen=True, kw_only=True)
class ProjectedEGMCarry:
    """Aggregated pre-shock rows with explicit domain, topology and integrity.

    All numerical channels have the producing passive-row axes followed by a
    padded projection axis. The current quadrature and discrete-action axes
    have already been reduced. This payload contains no resources-row policy.
    """

    grid: FloatND
    """Declared pre-shock coordinate and inserted owned refinement records."""
    value: FloatND
    """Current-shock and action-aggregated economic value."""
    economic_marginal: FloatND
    """Explicit weighted economic derivative in the pre-shock coordinate."""
    owner_id: IntND
    """Producer-owned common projection interval identity."""
    side_id: IntND
    """Proved source side, or zero for an exact represented-coordinate snapshot."""
    node_kind: IntND
    """Finite, economically impossible or unrepresented record classification."""
    link_valid: BoolND
    """Supported adjacent smooth links, excluding represented jumps and gaps."""
    valid_length: IntND
    """Actual emitted prefix length for each passive row."""
    domain_lower: FloatND
    """Lower endpoint of the declared pre-shock liquid-state interval."""
    domain_upper: FloatND
    """Upper endpoint of that interval, never an extrapolation permission."""
    valid: BoolND
    """Mandatory law/source/capacity integrity; links grant local coverage only.

    Exact-demand samples can be intact while open intervals remain unsupported.
    This flag alone never establishes complete topology or full-domain coverage.
    """

    def __post_init__(self) -> None:
        if self.grid.ndim < 1 or self.grid.shape[-1] < _MIN_PROJECTION_CAPACITY:
            raise ValueError("A projected EGM row needs a padded interval capacity.")
        _validate_projected_dtypes(carry=self)
        leading = self.grid.shape[:-1]
        for name in ("value", "economic_marginal", "owner_id", "side_id", "node_kind"):
            if getattr(self, name).shape != self.grid.shape:
                raise ValueError(f"Projected EGM {name} has a different row schema.")
        if self.link_valid.shape != (*leading, self.grid.shape[-1] - 1):
            raise ValueError("Projected EGM links do not match adjacent row records.")
        for name in ("valid_length", "domain_lower", "domain_upper", "valid"):
            if getattr(self, name).shape != leading:
                raise ValueError(f"Projected EGM {name} has a different cell schema.")

    @property
    def artifact_key(self) -> ArtifactKey:
        """Identify this projection schema independently of EGM resources rows."""
        return PROJECTED_EGM_CONTINUATION

    @property
    def capabilities(self) -> ContinuationCapabilities:
        """Require the status-aware model-bound projected child reader."""
        return ContinuationCapabilities()

    def leaves(self) -> Mapping[tuple[str, ...], ContinuationLeaf]:
        """Expose original stored objects for native custody and lifetime."""
        return MappingProxyType(
            {(field.name,): getattr(self, field.name) for field in fields(self)}
        )

    @property
    def guard_required(self) -> bool:
        """Identify carries requiring original-source ownership verification."""
        return False

    @property
    def value_slope_channel(self) -> FloatND | None:
        """Stored derivative of the value records, or None for a linear level read."""
        return None

    def value_at(self, *, query: FloatND) -> FloatND:
        """Require the parent to consume the projection's mandatory status."""
        del query
        raise UnsupportedOperationError(
            "Projected EGM rows require their status-aware model-bound reader."
        )

    def marginal_at(self, *, query: FloatND, state: StateName) -> FloatND:
        """Require economic-marginal composition through the native parent read."""
        del query, state
        raise UnsupportedOperationError(
            "Projected EGM rows require their status-aware model-bound reader."
        )


@runtime_checkable
class ProjectedGuardProgram(Protocol):
    """Model-owned static program consuming the acquired numerical guard leaves."""

    def validate(
        self, *, carry: ProjectedEGMCarry, query: ScalarFloat
    ) -> ScalarBool: ...


def _validate_projected_dtypes(*, carry: ProjectedEGMCarry) -> None:
    """Check numeric construction apart from trusted abstract reconstruction."""
    for name in ("value", "economic_marginal", "domain_lower", "domain_upper"):
        if getattr(carry, name).dtype != carry.grid.dtype:
            raise TypeError("Projected economic channels must share the grid dtype.")
    if not jnp.issubdtype(carry.grid.dtype, jnp.floating):
        raise TypeError("Projected EGM coordinates must be floating arrays.")
    for name in ("owner_id", "side_id", "node_kind", "valid_length"):
        if getattr(carry, name).dtype != jnp.int32:
            raise TypeError("Projected EGM index channels must be int32 arrays.")
    for name in ("link_valid", "valid"):
        if getattr(carry, name).dtype != jnp.bool_:
            raise TypeError("Projected EGM integrity channels must be Boolean.")


def read_projected_row(
    *,
    query: FloatND,
    grid: Float1D,
    value: Float1D,
    economic_marginal: Float1D,
    owner_id: Int1D,
    side_id: Int1D,
    node_kind: Int1D,
    link_valid: BoolND,
    valid_length: ScalarInt,
    domain_lower: FloatND,
    domain_upper: FloatND,
    valid: ScalarBool,
    value_slope: Float1D | None = None,
) -> ProjectedRowRead:
    """Read one original row, preserving exact nodes before interpolation.

    The caller selects a passive row before this operation and must consume
    ``valid`` even if its value is subsequently discounted or masked out.
    A query outside the declared domain is unsupported rather than clamped.
    With ``value_slope``, a finite smooth link reads its level by cubic Hermite
    in the stored value derivative; the economic marginal stays linear.
    """
    capacity = grid.size
    active = jnp.arange(capacity) < valid_length
    search = jnp.where(active, grid, jnp.inf)
    insertion = jnp.searchsorted(search, query, side="right")
    last = jnp.clip(valid_length - 1, 0, capacity - 1)
    exact_index = jnp.clip(insertion - 1, 0, last)
    exact = (valid_length > 0) & (grid[exact_index] == query)
    upper = jnp.clip(insertion, 1, jnp.maximum(last, 1))
    lower = upper - 1
    span = grid[upper] - grid[lower]
    represented_link = (
        (valid_length > 1)
        & (upper < valid_length)
        & link_valid[lower]
        & (span > 0.0)
        & (owner_id[lower] == owner_id[upper])
    )
    finite_link = (node_kind[lower] == QueryNodeKind.FINITE) & (
        node_kind[upper] == QueryNodeKind.FINITE
    )
    economic_link = (node_kind[lower] == QueryNodeKind.ECONOMIC_MINUS_INF) & (
        node_kind[upper] == QueryNodeKind.ECONOMIC_MINUS_INF
    )
    smooth = represented_link & (finite_link | economic_link)
    weight = (query - grid[lower]) / jnp.where(span > 0.0, span, 1.0)
    arithmetic_value = jnp.where(jnp.isfinite(value), value, 0.0)
    rise = arithmetic_value[upper] - arithmetic_value[lower]
    level = arithmetic_value[lower] + weight * rise
    if value_slope is not None:
        # A non-finite stored slope on a finite link invalidates the level.
        level = level + weight * (1.0 - weight) * (
            (1.0 - weight) * (span * value_slope[lower] - rise)
            - weight * (span * value_slope[upper] - rise)
        )
    marginal = economic_marginal[lower] + weight * (
        economic_marginal[upper] - economic_marginal[lower]
    )
    level = jnp.where(economic_link, -jnp.inf, level)
    marginal = jnp.where(economic_link, 0.0, marginal)
    chosen = jnp.where(exact, exact_index, lower)
    level = jnp.where(exact, value[exact_index], level)
    marginal = jnp.where(exact, economic_marginal[exact_index], marginal)
    numeric = (
        (
            (
                (node_kind == QueryNodeKind.FINITE)
                & jnp.isfinite(value)
                & jnp.isfinite(economic_marginal)
            )
            | (
                (node_kind == QueryNodeKind.ECONOMIC_MINUS_INF)
                & jnp.isneginf(value)
                & (economic_marginal == 0.0)
            )
        )
        & (owner_id >= 0)
        & (side_id >= -1)
        & (side_id <= 1)
    )
    supported = (
        valid
        & (valid_length > 0)
        & (valid_length <= capacity)
        & jnp.isfinite(query)
        & jnp.isfinite(domain_lower)
        & jnp.isfinite(domain_upper)
        & (domain_lower <= query)
        & (query <= domain_upper)
        & (exact | smooth)
        & jnp.where(exact, numeric[exact_index], numeric[lower] & numeric[upper])
        & (
            jnp.isfinite(level)
            | (
                (
                    (
                        exact
                        & (node_kind[exact_index] == QueryNodeKind.ECONOMIC_MINUS_INF)
                    )
                    | (~exact & represented_link & economic_link)
                )
                & jnp.isneginf(level)
            )
        )
        & jnp.isfinite(marginal)
    )
    return ProjectedRowRead(
        value=level,
        economic_marginal=marginal,
        owner_id=owner_id[chosen],
        side_id=side_id[chosen],
        valid=supported,
    )


_MIN_PROJECTION_CAPACITY: int = 2
_PROJECTED_FIELDS = tuple(field.name for field in fields(ProjectedEGMCarry))


def _flatten_projected(
    carry: ProjectedEGMCarry,
) -> tuple[tuple[tuple[jax.tree_util.GetAttrKey, PytreeChild], ...], None]:
    return (
        tuple(
            (jax.tree_util.GetAttrKey(name), getattr(carry, name))
            for name in _PROJECTED_FIELDS
        ),
        None,
    )


# keyword-only-exempt: library-callback=jax.tree_util.register_pytree_with_keys
def _unflatten_projected(
    metadata: None, leaves: Iterable[PytreeChild]
) -> ProjectedEGMCarry:
    del metadata
    # Native role/abstract restoration follows this engine-owned fixed schema;
    # the producer and mandatory read status validate the numerical records.
    carry = object.__new__(ProjectedEGMCarry)
    for name, leaf in zip(_PROJECTED_FIELDS, leaves, strict=True):
        object.__setattr__(carry, name, leaf)
    return carry


jax.tree_util.register_pytree_with_keys(
    ProjectedEGMCarry, _flatten_projected, _unflatten_projected
)


def _flatten_projected_row_read(
    record: ProjectedRowRead,
) -> tuple[tuple[PytreeChild, ...], None]:
    return tuple(record), None


def _flatten_projected_row_read_with_keys(
    record: ProjectedRowRead,
) -> tuple[tuple[tuple[jax.tree_util.GetAttrKey, PytreeChild], ...], None]:
    return tuple(
        (jax.tree_util.GetAttrKey(name), leaf)
        for name, leaf in zip(ProjectedRowRead._fields, record, strict=True)
    ), None


# keyword-only-exempt: library-callback=jax.tree_util.register_pytree_with_keys
def _unflatten_projected_row_read(
    metadata: None,
    children: Iterable[PytreeChild],
) -> ProjectedRowRead:
    """Restore native numeric, abstract or role leaves under the fixed schema."""
    del metadata
    leaves = tuple(children)
    if len(leaves) != len(ProjectedRowRead._fields):
        raise TypeError(
            "Native ProjectedRowRead reconstruction requires its exact fields."
        )
    return cast("ProjectedRowRead", tuple.__new__(ProjectedRowRead, leaves))


jax.tree_util.register_pytree_with_keys(
    ProjectedRowRead,
    _flatten_projected_row_read_with_keys,
    _unflatten_projected_row_read,
    _flatten_projected_row_read,
)
