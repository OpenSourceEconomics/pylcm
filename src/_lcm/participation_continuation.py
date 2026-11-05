"""Numeric operand custody for a dated participation policy reconstruction.

The model owns the scalar query and the operand-tree schema. This payload owns
only its dynamic leaves; it contains no policy answers, selected action IDs or
callables. A consuming source reconstructs the query at its actual landing.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

import jax

from _lcm.continuation import ContinuationPayload, ContinuationSpec
from _lcm.execution.value_transfer import ValueArtifactAddress, ValueArtifactKind
from _lcm.typing import PytreeChild
from lcm.exceptions import UnsupportedOperationError
from lcm.solver_api import ArtifactKey, ContinuationCapabilities
from lcm.typing import BoolND, FloatND, IntND, RegimeName, StateName

PARTICIPATION_CONTINUATION = ArtifactKey(
    type_id="pylcm.participation.query_operands", schema_version=1
)

type ParticipationOperand = FloatND | IntND | BoolND
type _ParticipationMetadata = tuple[RegimeName, int, str, tuple[str, ...]]

# Leading part of a payload leaf name holding a stored participation surface.
PARTICIPATION_SURFACE_PREFIX = "_lcm_participation_surface:"


def participation_surface_name(*, target: RegimeName) -> str:
    """Name the payload leaf holding a source's stored surface into `target`."""
    return f"{PARTICIPATION_SURFACE_PREFIX}{target}"


def participation_surface_leaves(
    payload: ContinuationPayload,
) -> MappingProxyType[str, ParticipationOperand]:
    """The stored participation surfaces a published payload carries, by leaf name.

    Every other leaf of a participation payload borrows a stored value, so only the
    surfaces are new arrays a later replay needs retained.
    """
    if not isinstance(payload, ParticipationContinuation):
        return MappingProxyType({})
    return MappingProxyType(
        {
            name: value
            for name, value in payload.operands.items()
            if name.startswith(PARTICIPATION_SURFACE_PREFIX)
        }
    )


@dataclass(frozen=True, kw_only=True)
class ParticipationContinuationSpec(ContinuationSpec):
    """Dated numerical schemas and explicit custody of their stored inputs."""

    origins_by_period: Mapping[int, Mapping[str, ValueArtifactAddress]] = field(
        default_factory=lambda: MappingProxyType({})
    )
    """Operand name to its actual later stored value, excluding parameter leaves."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if (
            self.artifact_key != PARTICIPATION_CONTINUATION
            or not self.templates_by_period
        ):
            raise ValueError(
                "Participation custody requires its key and actual dated schemas."
            )
        for period, template in self.templates_by_period.items():
            if (
                not isinstance(template, ParticipationContinuation)
                or template.period != period
            ):
                raise ValueError(
                    "A participation schema must name its actual producer date."
                )
        origins = {}
        for period, declarations in self.origins_by_period.items():
            template = self.template_at(period=period)
            if not isinstance(template, ParticipationContinuation):
                raise TypeError(
                    "Participation schemas require a ParticipationContinuation payload."
                )
            if any(name not in template.operands for name in declarations):
                raise ValueError("A transported origin names an undeclared operand.")
            if any(origin.period < period for origin in declarations.values()):
                raise ValueError(
                    "Participation custody cannot transport a past value as future."
                )
            origins[period] = MappingProxyType(dict(declarations))
        object.__setattr__(self, "origins_by_period", MappingProxyType(origins))

    def stored_value_aliases(
        self,
    ) -> MappingProxyType[ValueArtifactAddress, ValueArtifactAddress]:
        """Extend the originating buffer's lifetime through its participation reads."""
        aliases = {}
        for period, declarations in self.origins_by_period.items():
            template = self.template_at(period=period)
            if not isinstance(template, ParticipationContinuation):
                raise TypeError(
                    "Participation schemas require a ParticipationContinuation payload."
                )
            for name, origin in declarations.items():
                aliases[
                    ValueArtifactAddress(
                        kind=ValueArtifactKind.CONTINUATION_LEAF,
                        period=period,
                        regime=template.regime,
                        artifact_key=self.artifact_key,
                        leaf_path=("operands", name),
                    )
                ] = origin
        return MappingProxyType(aliases)


@dataclass(frozen=True, kw_only=True, init=False)
class ParticipationContinuation:
    """The exact producer-date dynamic operands, retaining array identities."""

    regime: RegimeName
    """Actual regime whose current action problem will be reconstructed."""
    period: int
    """Actual participation/producer date, never the consuming predecessor's date."""
    schema: str
    """Model-owned identity of the declared read objects and operand paths."""
    names: tuple[str, ...]
    """Static logical names, in the declared numerical operand order."""
    values: tuple[ParticipationOperand, ...]
    """Closed numeric storage, including any transported later continuation."""

    def __init__(
        self,
        *,
        regime: RegimeName,
        period: int,
        schema: str,
        operands: Mapping[str, ParticipationOperand],
    ) -> None:
        """Snapshot logical names while retaining only exact-tuple dynamic storage."""
        if type(period) is not int or period < 0:
            raise ValueError(
                "Participation continuation needs a nonnegative producer period."
            )
        if not regime or not schema:
            raise ValueError(
                "Participation continuation needs its regime and schema identity."
            )
        names = tuple(operands)
        if any(type(name) is not str or not name for name in names):
            raise ValueError("Participation operands need nonempty string names.")
        object.__setattr__(self, "regime", regime)
        object.__setattr__(self, "period", period)
        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "values", tuple(operands[name] for name in names))

    @property
    def operands(self) -> Mapping[str, ParticipationOperand]:
        """Expose logical read names without storing a runtime mapping container."""
        return MappingProxyType(dict(zip(self.names, self.values, strict=True)))

    @property
    def artifact_key(self) -> ArtifactKey:
        """Identify the numeric participation-operand schema."""
        return PARTICIPATION_CONTINUATION

    @property
    def capabilities(self) -> ContinuationCapabilities:
        """Require its model-bound policy query rather than impersonating a value."""
        return ContinuationCapabilities()

    def value_at(self, *, query: FloatND) -> FloatND:
        """Refuse an untyped value lookup on policy reconstruction operands."""
        del query
        raise UnsupportedOperationError(
            "Participation operands require their declared model-bound policy query."
        )

    def marginal_at(self, *, query: FloatND, state: StateName) -> FloatND:
        """Refuse a marginal whose represented policy has not been declared."""
        del query, state
        raise UnsupportedOperationError(
            "Participation operands do not publish an interpolated value or marginal."
        )

    def leaves(self) -> MappingProxyType[tuple[str, ...], ParticipationOperand]:
        """Address each original array without allocating copies or views."""
        return MappingProxyType(
            {("operands", name): value for name, value in self.operands.items()}
        )


def _flatten(
    payload: ParticipationContinuation,
) -> tuple[tuple[tuple[PytreeChild, ...], ...], _ParticipationMetadata]:
    """Expose only actual numerical operands to JAX transforms."""
    return (payload.values,), (
        payload.regime,
        payload.period,
        payload.schema,
        payload.names,
    )


def _flatten_with_keys(
    payload: ParticipationContinuation,
) -> tuple[
    tuple[tuple[jax.tree_util.GetAttrKey, tuple[PytreeChild, ...]], ...],
    _ParticipationMetadata,
]:
    """Preserve the same operand paths used by native continuation readers."""
    values, metadata = _flatten(payload)
    return ((jax.tree_util.GetAttrKey("values"), values[0]),), metadata


# keyword-only-exempt: library-callback=jax.tree_util.register_pytree_with_keys
def _unflatten(
    metadata: _ParticipationMetadata, values: Iterable[tuple[PytreeChild, ...]]
) -> ParticipationContinuation:
    """Rebuild transformed placeholders without running the user boundary."""
    payload = object.__new__(ParticipationContinuation)
    regime, period, schema, names = metadata
    object.__setattr__(payload, "regime", regime)
    object.__setattr__(payload, "period", period)
    object.__setattr__(payload, "schema", schema)
    object.__setattr__(payload, "names", names)
    (operands,) = values
    object.__setattr__(payload, "values", operands)
    return payload


jax.tree_util.register_pytree_with_keys(
    ParticipationContinuation, _flatten_with_keys, _unflatten, _flatten
)
