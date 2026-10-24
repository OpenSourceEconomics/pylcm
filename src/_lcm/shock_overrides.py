"""Invert supplied current-IID observations without manufacturing latent evidence.

This scalar-cell producer owns no random stream. Categorical intervals and unresolved
deterministic coordinates reuse the caller's named addressed draw. Every numerical
support/recovery failure reaches the mandatory validity channel.
"""

from collections.abc import Mapping
from dataclasses import fields
from types import MappingProxyType
from typing import NamedTuple

import jax.numpy as jnp
from jax.scipy.special import ndtr, ndtri

from lcm.exceptions import ShockDeclarationError
from lcm.shocks import (
    BaseInnovations,
    Discrete,
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
    _Independence,
    _law_validity,
    _resolve_parameter,
)
from lcm.typing import BoolND, FloatND, IntND, ParameterName, ReferenceName


class OverrideBases(NamedTuple):
    """Held independent coordinates, exact supplied physical bits and status."""

    bases: Mapping[ShockName, FloatND]
    physical: Mapping[ShockName, FloatND | IntND]
    valid: BoolND
    requires_witness: BoolND


def apply_overrides(
    *,
    group: ShockGroup,
    addressed: Mapping[ShockName, FloatND],
    overrides: Mapping[ShockName, ShockOverride],
    params: Mapping[ParameterName, LawValue],
    inputs: Mapping[ReferenceName, LawValue],
    admitted_correlation_support: BoolND | None,
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
    if partial_physical and correlated and isinstance(identity, bool) and not identity:
        raise ShockDeclarationError(
            "Correlated overrides require a complete physical frame alone; "
            f"physical={sorted(physical_names)}, group={sorted(group.shocks)}."
        )
    bases = dict(addressed)
    valid = jnp.asarray(a=True)
    if partial_physical:
        valid &= identity
    requires_witness = jnp.asarray(a=False)
    coordinates = {}
    informative = {}
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
        valid &= jnp.isfinite(value)
        if isinstance(item, BaseInnovations):
            # The canonical forward producer validates each law's base support:
            # Uniform includes one; categorical inverse-CDF sampling excludes it.
            bases[name] = value.astype(addressed[name].dtype)
        else:
            physical[name] = value
            parameters = {
                field.name: _resolve_parameter(
                    value=getattr(law, field.name),
                    name=f"{name}__{field.name}",
                    params=params,
                    inputs=inputs,
                )
                for field in fields(law)
                if field.name != "categories"
            }
            valid &= _law_validity(law=law, parameters=parameters)
            coordinate, observed, inverse_valid = _inverse(
                law=law, physical=value, parameters=parameters, residual=addressed[name]
            )
            valid &= inverse_valid
            coordinates[name], informative[name] = coordinate, observed
            bases[name] = coordinate
    if physical and correlated and not partial_physical:
        correlated_bases, correlated_valid, requires_witness = _recover_correlated(
            group=group,
            params=params,
            coordinates=coordinates,
            informative=informative,
            addressed=addressed,
            admitted_correlation_support=admitted_correlation_support,
            identity=jnp.asarray(identity),
        )
        bases.update(correlated_bases)
        valid &= correlated_valid
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
) -> tuple[Mapping[ShockName, FloatND], BoolND, BoolND]:
    """Invert the admitted triangular factor, retaining addressed null coordinates."""
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
    independent = jnp.stack(tuple(addressed.values()))
    for index, name in enumerate(group.shocks):
        contribution = jnp.sum(factor[index, :index] * independent[:index])
        residual = coordinates[name] - contribution
        diagonal = factor[index, index]
        recovered = residual / jnp.where(diagonal > 0, diagonal, 1.0)
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
) -> tuple[FloatND, BoolND, BoolND]:
    """Return a recovered coordinate, whether it reveals a base, and support."""
    if isinstance(law, Discrete):
        if physical.dtype.kind not in "iu":
            raise ShockDeclarationError(
                "Physical categorical values require integer category codes."
            )
        probs = parameters["probs"]
        code = physical
        safe_code = jnp.clip(code, 0, probs.shape[-1] - 1)
        last_positive = jnp.max(jnp.where(probs > 0, jnp.arange(probs.shape[-1]), -1))
        upper = jnp.where(
            jnp.arange(probs.shape[-1]) >= last_positive, 1.0, jnp.cumsum(probs)
        )
        lower = jnp.concatenate((jnp.zeros(1, dtype=probs.dtype), upper[:-1]))
        lo, hi = lower[safe_code], upper[safe_code]
        coordinate = lo + (hi - lo) * residual
        coordinate = jnp.minimum(coordinate, jnp.nextafter(hi, -jnp.inf))
        valid = (
            (code >= 0) & (code < probs.shape[-1]) & (probs[safe_code] > 0) & (hi > lo)
        )
        valid &= (
            (coordinate >= lo)
            & (coordinate < hi)
            & (coordinate >= 0)
            & (coordinate < 1)
        )
        return coordinate, jnp.asarray(a=True), valid
    if isinstance(law, Normal | LogNormal):
        scale, mu = parameters["sigma"], parameters["mu"]
        deterministic = jnp.exp(mu) if isinstance(law, LogNormal) else mu
        value = (
            jnp.log(jnp.where(physical > 0, physical, 1.0))
            if isinstance(law, LogNormal)
            else physical
        )
        coordinate = (value - mu) / jnp.where(scale > 0, scale, 1.0)
        support = (physical > 0) if isinstance(law, LogNormal) else jnp.asarray(a=True)
        support &= (scale > 0) | (physical == deterministic)
        return (
            jnp.where(scale > 0, coordinate, residual),
            scale > 0,
            support & jnp.isfinite(coordinate),
        )
    if isinstance(law, Uniform):
        start, stop = parameters["start"], parameters["stop"]
        width = stop - start
        coordinate = (physical - start) / jnp.where(width > 0, width, 1.0)
        support = (physical >= start) & (physical <= stop)
        support &= (width == 0) | ((coordinate >= 0) & (coordinate <= 1))
        return (
            jnp.where(width > 0, coordinate, residual),
            width > 0,
            support & jnp.isfinite(coordinate),
        )
    if isinstance(law, NormalMixture):
        p = parameters["p1"]
        first = (physical - parameters["mu1"]) / jnp.where(
            parameters["sigma1"] > 0, parameters["sigma1"], 1.0
        )
        second = (physical - parameters["mu2"]) / jnp.where(
            parameters["sigma2"] > 0, parameters["sigma2"], 1.0
        )
        cdf = p * ndtr(first) + (1 - p) * ndtr(second)
        survival = p * ndtr(-first) + (1 - p) * ndtr(-second)
        coordinate = jnp.where(cdf <= 0.5, ndtri(cdf), -ndtri(survival))  # noqa: PLR2004 - choose the numerically resolved CDF/survival half
        return coordinate, jnp.asarray(a=True), jnp.isfinite(coordinate)
    raise ShockDeclarationError("Unsupported physical inverse.")
