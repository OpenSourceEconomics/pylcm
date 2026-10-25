"""Invert supplied current-IID observations without manufacturing latent evidence.

This scalar-cell producer owns no random stream. Categorical intervals and unresolved
deterministic coordinates reuse the caller's named addressed draw. Every numerical
support/recovery failure reaches the mandatory validity channel.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, fields
from types import MappingProxyType
from typing import Literal, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import ndtr, ndtri
from jaxtyping import Bool

from _lcm.processes.base import sealed
from _lcm.typing import PytreeChild
from lcm.exceptions import ShockDeclarationError
from lcm.shocks import (
    BaseInnovations,
    Discrete,
    HostArray,
    Law,
    LawValue,
    LogNormal,
    Normal,
    NormalMixture,
    PhysicalValues,
    ShockGroup,
    ShockName,
    ShockOverride,
    Uniform,
    _check_scalar_law_parameters,
    _continuous_value,
    _discrete_cdf,
    _Independence,
    _law_validity,
    _resolve_parameter,
    _sample_discrete,
)
from lcm.typing import BoolND, FloatND, IntND, ParameterName, ReferenceName


@dataclass(frozen=True, kw_only=True)
class CurrentIIDOverride:
    """Private native operand with a static tag and explicit dated activation.

    Abstract profiles use descriptors here; public records accept numerical arrays
    only. The actual scalar producer validates the public record at consumption.
    """

    kind: Literal["base", "physical"]
    """Scientific interpretation of the supplied numeric coordinate."""
    values: FloatND | IntND | HostArray | jax.ShapeDtypeStruct
    """One subject-axis leaf, or its exact native prototype."""
    # Jaxtyping consumes the string as an array shape, not a forward reference.
    active: BoolND | Bool[np.ndarray, "..."] | jax.ShapeDtypeStruct  # noqa: UP037
    """Whether this row consumes the input at this computational period."""


def _flatten_override(
    item: CurrentIIDOverride,
) -> tuple[tuple[PytreeChild, PytreeChild], tuple[Literal["base", "physical"]]]:
    return (item.values, item.active), (item.kind,)


def _flatten_override_with_keys(
    item: CurrentIIDOverride,
) -> tuple[
    tuple[tuple[jax.tree_util.GetAttrKey, PytreeChild], ...],
    tuple[Literal["base", "physical"]],
]:
    return (
        (
            (jax.tree_util.GetAttrKey("values"), item.values),
            (jax.tree_util.GetAttrKey("active"), item.active),
        ),
        (item.kind,),
    )


# keyword-only-exempt: library-callback=jax.tree_util.register_pytree_with_keys
def _unflatten_override(
    metadata: tuple[Literal["base", "physical"]], children: Iterable[PytreeChild]
) -> CurrentIIDOverride:
    """Rebuild JAX's abstract/transform tree without widening checked inputs."""
    values, active = children
    result = object.__new__(CurrentIIDOverride)
    object.__setattr__(result, "kind", metadata[0])
    object.__setattr__(result, "values", values)
    object.__setattr__(result, "active", active)
    return result


jax.tree_util.register_pytree_with_keys(
    CurrentIIDOverride,
    _flatten_override_with_keys,
    _unflatten_override,
    _flatten_override,
)


class OverrideBases(NamedTuple):
    """Held independent coordinates, exact supplied physical bits and status."""

    bases: Mapping[ShockName, FloatND]
    physical: Mapping[ShockName, FloatND | IntND]
    valid: BoolND
    requires_witness: BoolND


def apply_overrides(  # noqa: C901 - scalar law/support cases and dated activation
    *,
    group: ShockGroup,
    addressed: Mapping[ShockName, FloatND],
    overrides: Mapping[ShockName, ShockOverride],
    params: Mapping[ParameterName, LawValue],
    inputs: Mapping[ReferenceName, LawValue],
    admitted_correlation_support: BoolND | None,
    active: Mapping[ShockName, BoolND] = MappingProxyType({}),
    fallback_physical: Mapping[ShockName, FloatND | IntND] = MappingProxyType({}),
) -> OverrideBases:
    """Recover one complete scalar frame under its original producing law.

    Partial correlated physical inputs and mixed correlated forms are unsupported.
    This helper neither marks supplied values observed nor changes the group's law.
    """
    physical_names = {
        name for name, item in overrides.items() if isinstance(item, PhysicalValues)
    }
    physical: dict[ShockName, FloatND | IntND] = {}
    correlated = group.correlation is not _Independence.DEFAULT
    identity = group._has_identity_correlation(params=params)  # noqa: SLF001 - shared exact current-binding proof
    partial_physical = bool(physical_names) and physical_names != group.shocks.keys()
    if (
        not active
        and partial_physical
        and correlated
        and isinstance(identity, bool)
        and not identity
    ):
        raise ShockDeclarationError(
            "Correlated overrides require a complete physical frame alone; "
            f"physical={sorted(physical_names)}, group={sorted(group.shocks)}."
        )
    bases = dict(addressed)
    valid = jnp.asarray(a=True)
    physical_active = jnp.stack(
        tuple(
            jnp.asarray(active.get(name, True))
            if name in physical_names
            else jnp.asarray(a=False)
            for name in group.shocks
        )
    )
    any_physical, complete_physical = (
        jnp.any(physical_active),
        jnp.all(physical_active),
    )
    if active:
        valid &= identity | ~any_physical | complete_physical
    elif partial_physical:
        valid &= identity
    requires_witness = jnp.asarray(a=False)
    coordinates, informative = {}, {}
    for name, item in overrides.items():
        if name not in group.shocks:
            raise ShockDeclarationError(f"Unknown shock override {name!r}.")
        if not isinstance(item, BaseInnovations | PhysicalValues):
            raise ShockDeclarationError(
                "Shock inputs require BaseInnovations or PhysicalValues."
            )
        value = jnp.asarray(item.values)
        if value.ndim or value.dtype.kind not in "fiu":
            raise ShockDeclarationError(
                "A current-IID override must be a real scalar cell."
            )
        law = group.shocks[name]
        enabled = jnp.asarray(active.get(name, True))
        if active:
            # Sanitize before inversion, so unread NaNs never enter arithmetic.
            fallback = (
                addressed[name]
                if isinstance(item, BaseInnovations)
                else fallback_physical[name]
            )
            if isinstance(item, PhysicalValues) and value.dtype != fallback.dtype:
                raise ShockDeclarationError(
                    "Physical values must use their producing law's dtype."
                )
            value = jnp.where(enabled, value, fallback)
        valid &= jnp.isfinite(value)
        if isinstance(item, BaseInnovations):
            # The canonical forward producer validates each law's base support:
            # Uniform includes one; categorical inverse-CDF sampling excludes it.
            bases[name] = value.astype(addressed[name].dtype)
        else:
            physical[name] = value
            parameters = _law_parameters(
                name=name, law=law, params=params, inputs=inputs
            )
            valid &= _law_validity(law=law, parameters=parameters)
            coordinate, observed, inverse_valid = _inverse(
                law=law,
                physical=value,
                parameters=parameters,
                residual=addressed[name],
                enabled=enabled,
            )
            valid &= ~enabled | inverse_valid
            coordinates[name], informative[name] = coordinate, observed
            bases[name] = coordinate
        bases[name] = jnp.where(enabled, bases[name], addressed[name])
    if physical and correlated and not partial_physical:
        correlated_bases, correlated_valid, requires_witness = _recover_correlated(
            group=group,
            params=params,
            coordinates=coordinates,
            informative=informative,
            addressed=addressed,
            admitted_correlation_support=admitted_correlation_support,
            identity=jnp.asarray(identity),
            enabled=any_physical,
        )
        bases.update(correlated_bases)
        valid &= ~any_physical | correlated_valid
        requires_witness &= any_physical
        bases = {
            name: jnp.where(any_physical, value, addressed[name])
            for name, value in bases.items()
        }
    return OverrideBases(
        MappingProxyType(bases), MappingProxyType(physical), valid, requires_witness
    )


def _recover_correlated(
    *,
    group: ShockGroup,
    params: Mapping[ParameterName, LawValue],
    coordinates: Mapping[ShockName, FloatND],
    informative: Mapping[ShockName, BoolND],
    addressed: Mapping[ShockName, FloatND],
    admitted_correlation_support: BoolND | None,
    identity: BoolND,
    enabled: BoolND,
) -> tuple[Mapping[ShockName, FloatND], BoolND, BoolND]:
    """Invert the admitted triangular factor, retaining addressed null coordinates."""
    return _recover_factor(
        group=group,
        params=params,
        coordinates=coordinates,
        informative=informative,
        completion=addressed,
        admitted_correlation_support=admitted_correlation_support,
        identity=identity,
        enabled=enabled,
    )


def _recover_factor(
    *,
    group: ShockGroup,
    params: Mapping[ParameterName, LawValue],
    coordinates: Mapping[ShockName, FloatND],
    informative: Mapping[ShockName, BoolND],
    completion: Mapping[ShockName, FloatND] | None,
    admitted_correlation_support: BoolND | None,
    identity: BoolND,
    enabled: BoolND,
) -> tuple[Mapping[ShockName, FloatND], BoolND, BoolND]:
    """Share exact-factor inversion, keeping support witnesses distinct from draws."""
    valid = jnp.asarray(a=True)
    factor, factor_valid = group._correlation_factor(  # noqa: SLF001 - same admitted factor as the forward producer
        params=params, admitted_correlation_support=admitted_correlation_support
    )
    valid &= factor_valid
    observed = jnp.stack(tuple(informative.values()))
    # A physically complete frame can still be latent-incomplete. The all
    # deterministic case conditions nothing and keeps every addressed base.
    all_observed, none_observed = jnp.all(observed), ~jnp.any(observed)
    valid &= identity | all_observed | none_observed
    # Only exact-zero factor columns (or constant marginal outputs) may be
    # completed canonically for a support witness. These are never draw evidence.
    independent = (
        jnp.stack(tuple(completion.values()))
        if completion is not None
        else jnp.zeros_like(jnp.stack(tuple(coordinates.values())))
    )
    for index, name in enumerate(group.shocks):
        contribution = jnp.sum(factor[index, :index] * independent[:index])
        residual = jnp.where(enabled, coordinates[name] - contribution, 0.0)
        diagonal = factor[index, index]
        recovered = residual / jnp.where(enabled & (diagonal > 0), diagonal, 1.0)
        # External support is proved by the actual producing forward evaluation,
        # after recovering the candidate. A proximity envelope cannot authenticate
        # a complete external frame, even with this exact admitted rank.
        independent = independent.at[index].set(
            jnp.where(
                identity,
                coordinates[name],
                jnp.where(all_observed & (diagonal > 0), recovered, independent[index]),
            )
        )
    return (
        MappingProxyType(dict(zip(group.shocks, independent, strict=True))),
        valid,
        jnp.any(jnp.diag(factor) == 0),
    )


def _inverse(
    *,
    law: Law,
    physical: FloatND | IntND,
    parameters: Mapping[ParameterName, FloatND],
    residual: FloatND,
    enabled: BoolND,
) -> tuple[FloatND, BoolND, BoolND]:
    """Complete simulation's inverse with its genuine addressed residual."""
    coordinate, informative, valid = _inverse_support(
        law=law, physical=physical, parameters=parameters, enabled=enabled
    )
    if isinstance(law, Discrete):
        probs = parameters["probs"]
        safe_code = jnp.clip(physical, 0, probs.shape[-1] - 1)
        upper = _discrete_cdf(probs=probs)
        lower = jnp.concatenate((jnp.zeros(1, dtype=probs.dtype), upper[:-1]))
        lo, hi = lower[safe_code], upper[safe_code]
        coordinate = jnp.minimum(lo + (hi - lo) * residual, jnp.nextafter(hi, -jnp.inf))
        valid &= (coordinate >= lo) & (coordinate < hi)
        valid &= (coordinate >= 0) & (coordinate < 1)
    else:
        coordinate = jnp.where(informative, coordinate, residual)
    return coordinate, informative, valid


def _inverse_support(
    *,
    law: Law,
    physical: FloatND | IntND,
    parameters: Mapping[ParameterName, FloatND],
    enabled: BoolND,
) -> tuple[FloatND, BoolND, BoolND]:
    """Construct a support witness without claiming a latent random coordinate."""
    if isinstance(law, Discrete):
        if physical.dtype.kind not in "iu":
            raise ShockDeclarationError(
                "Physical categorical values require integer category codes."
            )
        probs = parameters["probs"]
        code = physical
        safe_code = jnp.clip(code, 0, probs.shape[-1] - 1)
        upper = _discrete_cdf(probs=probs)
        lower = jnp.concatenate((jnp.zeros(1, dtype=probs.dtype), upper[:-1]))
        lo, hi = lower[safe_code], upper[safe_code]
        # An interval endpoint proves category support; it is not a drawn uniform.
        coordinate = lo
        valid = (
            (code >= 0) & (code < probs.shape[-1]) & (probs[safe_code] > 0) & (hi > lo)
        )
        valid &= (coordinate >= 0) & (coordinate < 1)
        return coordinate, jnp.asarray(a=True), valid
    if isinstance(law, Normal | LogNormal):
        scale, mu = parameters["sigma"], parameters["mu"]
        deterministic = jnp.exp(mu) if isinstance(law, LogNormal) else mu
        value = (
            jnp.log(jnp.where(physical > 0, physical, 1.0))
            if isinstance(law, LogNormal)
            else physical
        )
        coordinate = jnp.where(enabled, value - mu, 0.0) / jnp.where(
            enabled & (scale > 0), scale, 1.0
        )
        support = (physical > 0) if isinstance(law, LogNormal) else jnp.asarray(a=True)
        support &= (scale > 0) | (physical == deterministic)
        return (
            coordinate,
            scale > 0,
            support & jnp.isfinite(coordinate),
        )
    if isinstance(law, Uniform):
        start, stop = parameters["start"], parameters["stop"]
        width = stop - start
        coordinate = jnp.where(enabled, physical - start, 0.0) / jnp.where(
            enabled & (width > 0), width, 1.0
        )
        support = (physical >= start) & (physical <= stop)
        support &= (width == 0) | ((coordinate >= 0) & (coordinate <= 1))
        return (
            coordinate,
            width > 0,
            support & jnp.isfinite(coordinate),
        )
    if isinstance(law, NormalMixture):
        p = parameters["p1"]
        first = jnp.where(enabled, physical - parameters["mu1"], 0.0) / jnp.where(
            enabled & (parameters["sigma1"] > 0), parameters["sigma1"], 1.0
        )
        second = jnp.where(enabled, physical - parameters["mu2"], 0.0) / jnp.where(
            enabled & (parameters["sigma2"] > 0), parameters["sigma2"], 1.0
        )
        cdf = p * ndtr(first) + (1 - p) * ndtr(second)
        survival = p * ndtr(-first) + (1 - p) * ndtr(-second)
        cdf = jnp.where(enabled, cdf, 0.5)
        survival = jnp.where(enabled, survival, 0.5)
        coordinate = jnp.where(cdf <= 0.5, ndtri(cdf), -ndtri(survival))  # noqa: PLR2004 - choose the numerically resolved CDF/survival half
        return coordinate, jnp.asarray(a=True), jnp.isfinite(coordinate)
    raise ShockDeclarationError("Unsupported physical inverse.")


def physical_recovery_status(
    *, produced: FloatND | IntND, supplied: FloatND | IntND, requires_witness: BoolND
) -> tuple[BoolND, FloatND]:
    """Keep physical bits while authenticating singular producing-image support."""
    if produced.dtype != supplied.dtype:
        raise ShockDeclarationError(
            "Physical values must use their producing law's dtype."
        )
    if produced.dtype.kind == "f":
        bits = jnp.uint64 if produced.dtype.itemsize == 8 else jnp.uint32  # noqa: PLR2004 - active IEEE word width
        witnessed = jax.lax.bitcast_convert_type(
            produced, bits
        ) == jax.lax.bitcast_convert_type(supplied, bits)
    else:
        witnessed = produced == supplied
    error = produced.astype(jnp.result_type(0.0)) - supplied.astype(
        jnp.result_type(0.0)
    )
    return (~requires_witness | witnessed) & jnp.isfinite(error), error


def _law_parameters(
    *,
    name: ShockName,
    law: Law,
    params: Mapping[ParameterName, LawValue],
    inputs: Mapping[ReferenceName, LawValue],
) -> dict[ParameterName, FloatND]:
    """Resolve the same producing-law scalar parameters for either entry route."""
    parameters = {
        item.name: _resolve_parameter(
            value=getattr(law, item.name),
            name=f"{name}__{item.name}",
            params=params,
            inputs=inputs,
        )
        for item in fields(law)
        if item.name != "categories"
    }
    _check_scalar_law_parameters(
        parameters=parameters, group_name="supplied", shock_name=name
    )
    return parameters


def _forward_supplied_law(
    *,
    law: Law,
    coordinate: FloatND,
    parameters: Mapping[ParameterName, FloatND],
) -> tuple[FloatND | IntND, BoolND]:
    """Use the canonical marginal forward arithmetic, without filling other columns."""
    if isinstance(law, Discrete):
        return _sample_discrete(coordinate=coordinate, probs=parameters["probs"])
    value, valid = _continuous_value(
        law=law, coordinate=coordinate, parameters=parameters
    )
    if isinstance(law, Uniform):
        valid &= (coordinate >= 0) & (coordinate <= 1)
    return sealed(value), valid & jnp.isfinite(value)


def supplied_group_values(
    *,
    group: ShockGroup,
    supplied: Mapping[ShockName, ShockOverride],
    params: Mapping[ParameterName, LawValue],
    inputs: Mapping[ReferenceName, LawValue],
    admitted_correlation_support: BoolND | None,
) -> tuple[Mapping[ShockName, FloatND | IntND], BoolND]:
    """Validate supplied values or coordinates; return no latent draw evidence.

    A complete mixed-informative correlated physical frame retains the existing
    unresolved support status, rather than pretending to be a partial-input deferral.
    """
    physical_names, independent, identity = _supplied_form(
        group=group, supplied=supplied, params=params
    )
    physical, coordinates, informative = {}, {}, {}
    valid = jnp.asarray(a=True)
    for name, item in supplied.items():
        law = group.shocks[name]
        value = jnp.asarray(item.values)
        if value.ndim or value.dtype.kind not in "fiu":
            raise ShockDeclarationError(
                "A current-IID override must be a real scalar cell."
            )
        parameters = _law_parameters(name=name, law=law, params=params, inputs=inputs)
        valid &= jnp.isfinite(value) & _law_validity(law=law, parameters=parameters)
        if isinstance(item, PhysicalValues):
            expected = (
                jnp.dtype(jnp.int32)
                if isinstance(law, Discrete)
                else jnp.dtype(jnp.result_type(0.0))
            )
            if value.dtype != expected:
                raise ShockDeclarationError(
                    "Physical values must use their producing law's dtype."
                )
            coordinate, observed, supported = _inverse_support(
                law=law,
                physical=value,
                parameters=parameters,
                enabled=jnp.asarray(a=True),
            )
            coordinates[name], informative[name] = coordinate, observed
            physical[name] = value
            valid &= supported
            if independent:
                produced, resolved = _forward_supplied_law(
                    law=law, coordinate=coordinate, parameters=parameters
                )
                recovery, _ = physical_recovery_status(
                    produced=produced,
                    supplied=value,
                    requires_witness=jnp.asarray(a=False),
                )
                valid &= resolved & recovery
        else:
            coordinates[name] = value
            if independent:
                physical[name], resolved = _forward_supplied_law(
                    law=law, coordinate=value, parameters=parameters
                )
                valid &= resolved
    if not independent:
        physical, correlated_valid = _finish_supplied_group(
            group=group,
            coordinates=coordinates,
            informative=informative,
            physical=physical,
            physical_names=physical_names,
            params=params,
            inputs=inputs,
            identity=jnp.asarray(identity),
            admitted_correlation_support=admitted_correlation_support,
        )
        valid &= correlated_valid
    return MappingProxyType(physical), valid


def _finish_supplied_group(
    *,
    group: ShockGroup,
    coordinates: Mapping[ShockName, FloatND],
    informative: Mapping[ShockName, BoolND],
    physical: dict[ShockName, FloatND | IntND],
    physical_names: set[ShockName],
    params: Mapping[ParameterName, LawValue],
    inputs: Mapping[ReferenceName, LawValue],
    identity: BoolND,
    admitted_correlation_support: BoolND | None,
) -> tuple[dict[ShockName, FloatND | IntND], BoolND]:
    """Judge the complete joint frame with its canonical producing witness."""
    valid = jnp.asarray(a=True)
    coordinates = {name: coordinates[name] for name in group.shocks}
    informative = {
        name: informative[name] for name in group.shocks if name in informative
    }
    requires_witness = jnp.asarray(a=False)
    if physical_names:
        coordinates, factor_valid, requires_witness = _recover_factor(
            group=group,
            params=params,
            coordinates=coordinates,
            informative=informative,
            completion=None,
            admitted_correlation_support=admitted_correlation_support,
            identity=jnp.asarray(identity),
            enabled=jnp.asarray(a=True),
        )
        valid &= factor_valid
    realised, _ = group._realise(  # noqa: SLF001 - same producing forward image
        base_nodes=jnp.stack(tuple(coordinates.values()))[None],
        nodes=None,
        params=params,
        inputs=inputs,
        scalar_cell_group="entry",
        admitted_correlation_support=admitted_correlation_support,
    )
    valid &= realised.valid
    for name, produced in realised.values.items():
        value = produced[0]
        if name in physical_names:
            recovery, _ = physical_recovery_status(
                produced=value,
                supplied=physical[name],
                requires_witness=requires_witness,
            )
            valid &= recovery
        else:
            physical[name] = value
    return physical, valid


def _supplied_form(
    *,
    group: ShockGroup,
    supplied: Mapping[ShockName, ShockOverride],
    params: Mapping[ParameterName, LawValue],
) -> tuple[set[ShockName], bool, bool | BoolND]:
    """Apply the existing complete-correlated-frame grammar before inversion."""
    unknown = supplied.keys() - group.shocks.keys()
    if unknown:
        raise ShockDeclarationError(f"Unknown shock overrides {sorted(unknown)}.")
    correlated = group.correlation is not _Independence.DEFAULT
    identity = group._has_identity_correlation(params=params)  # noqa: SLF001 - canonical identity proof
    independent = not correlated or (isinstance(identity, bool) and identity)
    physical_names = {
        name for name, item in supplied.items() if isinstance(item, PhysicalValues)
    }
    if not independent and physical_names and physical_names != group.shocks.keys():
        raise ShockDeclarationError(
            "Correlated overrides require a complete physical frame alone."
        )
    if (
        not independent
        and not physical_names
        and supplied.keys() != group.shocks.keys()
    ):
        raise ShockDeclarationError(
            "Supplied correlated base frames must determine every coordinate."
        )
    return physical_names, independent, identity
