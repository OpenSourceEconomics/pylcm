"""Numeric operand custody for a dated arrival policy reconstruction.

The model owns the scalar query and the operand-tree schema. This payload owns
only its dynamic leaves; it contains no policy answers, selected action IDs or
callables. A consuming source reconstructs the query at its actual landing.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

import jax

from _lcm.continuation import ContinuationSpec
from _lcm.execution.value_transfer import ValueArtifactAddress, ValueArtifactKind
from _lcm.typing import PytreeChild
from lcm.exceptions import UnsupportedOperationError
from lcm.solver_api import ArtifactKey, ContinuationCapabilities
from lcm.typing import BoolND, FloatND, IntND, RegimeName, StateName

ARRIVAL_CONTINUATION = ArtifactKey(
    type_id="pylcm.arrival.query_operands", schema_version=1
)

type ArrivalOperand = FloatND | IntND | BoolND
type _ArrivalMetadata = tuple[RegimeName, int, str, tuple[str, ...]]


@dataclass(frozen=True, kw_only=True)
class ArrivalContinuationSpec(ContinuationSpec):
    """Dated numerical schemas and explicit custody of their stored inputs."""

    origins_by_period: Mapping[int, Mapping[str, ValueArtifactAddress]] = field(
        default_factory=lambda: MappingProxyType({})
    )
    """Operand name to its actual later stored value, excluding parameter leaves."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.artifact_key != ARRIVAL_CONTINUATION or not self.templates_by_period:
            raise ValueError(
                "Arrival custody requires its key and actual dated schemas."
            )
        for period, template in self.templates_by_period.items():
            if (
                not isinstance(template, ArrivalContinuation)
                or template.period != period
            ):
                raise ValueError(
                    "An arrival schema must name its actual producer date."
                )
        origins = {}
        for period, declarations in self.origins_by_period.items():
            template = self.template_at(period=period)
            if not isinstance(template, ArrivalContinuation):
                raise TypeError(
                    "Arrival schemas require an ArrivalContinuation payload."
                )
            if any(name not in template.operands for name in declarations):
                raise ValueError("A transported origin names an undeclared operand.")
            if any(origin.period < period for origin in declarations.values()):
                raise ValueError(
                    "Arrival custody cannot transport a past value as future."
                )
            origins[period] = MappingProxyType(dict(declarations))
        object.__setattr__(self, "origins_by_period", MappingProxyType(origins))

    def stored_value_aliases(
        self,
    ) -> MappingProxyType[ValueArtifactAddress, ValueArtifactAddress]:
        """Extend the originating buffer's lifetime through its arrival consumers."""
        aliases = {}
        for period, declarations in self.origins_by_period.items():
            template = self.template_at(period=period)
            if not isinstance(template, ArrivalContinuation):
                raise TypeError(
                    "Arrival schemas require an ArrivalContinuation payload."
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
class ArrivalContinuation:
    """The exact producer-date dynamic operands, retaining array identities."""

    regime: RegimeName
    """Actual regime whose current action problem will be reconstructed."""
    period: int
    """Actual arrival/producer date, never the consuming predecessor's date."""
    schema: str
    """Model-owned identity of the declared read objects and operand paths."""
    names: tuple[str, ...]
    """Static logical names, in the declared numerical operand order."""
    values: tuple[ArrivalOperand, ...]
    """Closed numeric storage, including any transported later continuation."""

    def __init__(
        self,
        *,
        regime: RegimeName,
        period: int,
        schema: str,
        operands: Mapping[str, ArrivalOperand],
    ) -> None:
        """Snapshot logical names while retaining only exact-tuple dynamic storage."""
        if type(period) is not int or period < 0:
            raise ValueError(
                "Arrival continuation needs a nonnegative producer period."
            )
        if not regime or not schema:
            raise ValueError(
                "Arrival continuation needs its regime and schema identity."
            )
        names = tuple(operands)
        if any(type(name) is not str or not name for name in names):
            raise ValueError("Arrival operands need nonempty string names.")
        object.__setattr__(self, "regime", regime)
        object.__setattr__(self, "period", period)
        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "values", tuple(operands[name] for name in names))

    @property
    def operands(self) -> Mapping[str, ArrivalOperand]:
        """Expose logical read names without storing a runtime mapping container."""
        return MappingProxyType(dict(zip(self.names, self.values, strict=True)))

    @property
    def artifact_key(self) -> ArtifactKey:
        """Identify the numeric arrival-operand schema."""
        return ARRIVAL_CONTINUATION

    @property
    def capabilities(self) -> ContinuationCapabilities:
        """Require its model-bound policy query rather than impersonating a value."""
        return ContinuationCapabilities()

    def value_at(self, *, query: FloatND) -> FloatND:
        """Refuse an untyped value lookup on policy reconstruction operands."""
        del query
        raise UnsupportedOperationError(
            "Arrival operands require their declared model-bound policy query."
        )

    def marginal_at(self, *, query: FloatND, state: StateName) -> FloatND:
        """Refuse a marginal whose represented policy has not been declared."""
        del query, state
        raise UnsupportedOperationError(
            "Arrival operands do not publish an interpolated value or marginal."
        )

    def leaves(self) -> MappingProxyType[tuple[str, ...], ArrivalOperand]:
        """Address each original array without allocating copies or views."""
        return MappingProxyType(
            {("operands", name): value for name, value in self.operands.items()}
        )


def _flatten(
    payload: ArrivalContinuation,
) -> tuple[tuple[tuple[PytreeChild, ...], ...], _ArrivalMetadata]:
    """Expose only actual numerical operands to JAX transforms."""
    return (payload.values,), (
        payload.regime,
        payload.period,
        payload.schema,
        payload.names,
    )


def _flatten_with_keys(
    payload: ArrivalContinuation,
) -> tuple[
    tuple[tuple[jax.tree_util.GetAttrKey, tuple[PytreeChild, ...]], ...],
    _ArrivalMetadata,
]:
    """Preserve the same operand paths used by native continuation readers."""
    values, metadata = _flatten(payload)
    return ((jax.tree_util.GetAttrKey("values"), values[0]),), metadata


# keyword-only-exempt: library-callback=jax.tree_util.register_pytree_with_keys
def _unflatten(
    metadata: _ArrivalMetadata, values: Iterable[tuple[PytreeChild, ...]]
) -> ArrivalContinuation:
    """Rebuild transformed placeholders without running the user boundary."""
    payload = object.__new__(ArrivalContinuation)
    regime, period, schema, names = metadata
    object.__setattr__(payload, "regime", regime)
    object.__setattr__(payload, "period", period)
    object.__setattr__(payload, "schema", schema)
    object.__setattr__(payload, "names", names)
    (operands,) = values
    object.__setattr__(payload, "values", operands)
    return payload


jax.tree_util.register_pytree_with_keys(
    ArrivalContinuation, _flatten_with_keys, _unflatten, _flatten
)
