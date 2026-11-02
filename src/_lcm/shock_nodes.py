"""Build joint shock rules outside tracing."""

import hashlib
import itertools
import json
import math
from fractions import Fraction
from types import MappingProxyType

import jax.numpy as jnp
import numpy as np
import scipy
from jax.dtypes import canonicalize_dtype
from scipy.special import ndtr, ndtri
from scipy.stats import qmc

from _lcm.dtypes import CanonicalArrayWriter, canonical_float_dtype
from lcm.exceptions import ShockDeclarationError
from lcm.grids import DiscreteGrid
from lcm.shocks import (
    Discrete,
    EqualSpaced,
    EqualSpacedUniform,
    GaussHermite,
    HostArray,
    MonteCarlo,
    NodeSet,
    Product,
    QuasiMonteCarlo,
    ScalarRule,
    ShockGroup,
    ShockGroupName,
    ShockName,
    Uniform,
    UserNodes,
    _fingerprint,
    _validate_weights,
)


def resolve_nodes(
    *,
    group: ShockGroup,
    group_name: ShockGroupName,
    age: Fraction,
    coordinate_kind: str,
    array_writer: CanonicalArrayWriter | None = None,
) -> NodeSet:
    continuous = tuple(
        name for name, law in group.shocks.items() if not isinstance(law, Discrete)
    )
    uniform = tuple(isinstance(group.shocks[name], Uniform) for name in continuous)
    dimension = len(continuous)
    rule = group.nodes
    if isinstance(rule, UserNodes):
        if any(isinstance(law, Discrete) for law in group.shocks.values()):
            raise ShockDeclarationError(
                "UserNodes cannot replace exact category enumeration."
            )
        points, weights = np.asarray(rule.nodes), np.asarray(rule.weights)
    elif isinstance(rule, Product):
        axes = [_scalar_rule(rule=rule.rules[name]) for name in continuous]
        points, weights = _tensor(axes)
    elif isinstance(rule, MonteCarlo | QuasiMonteCarlo):
        encoded = json.dumps(
            (
                "pylcm-solve-rule",
                1,
                rule.seed,
                coordinate_kind,
                age.numerator,
                age.denominator,
                group_name,
            ),
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode()
        seed = int.from_bytes(hashlib.sha256(encoded).digest(), "big")
        generator = np.random.Generator(np.random.PCG64(seed))
        if dimension == 0:
            points, weights = np.zeros((1, 0)), np.ones(1)
        else:
            uniforms = (
                generator.random((rule.n, dimension))
                if isinstance(rule, MonteCarlo)
                else qmc.Sobol(
                    d=dimension, scramble=True, bits=52, rng=generator
                ).random_base2(m=rule.n.bit_length() - 1)
            )
            # Open endpoints before inverse-normal mapping; retain every Sobol point.
            uniforms = np.clip(uniforms, np.nextafter(0.0, 1.0), np.nextafter(1.0, 0.0))
            points = np.column_stack(
                [
                    uniforms[:, column] if base_uniform else ndtri(uniforms[:, column])
                    for column, base_uniform in enumerate(uniform)
                ]
            )
            weights = np.full(rule.n, 1.0 / rule.n)
    elif rule is None and not continuous:
        points, weights = np.zeros((1, 0)), np.ones(1)
    else:
        raise ShockDeclarationError("Unsupported or missing node rule.")
    _validate_weights(weights=weights, rule=type(rule).__name__)
    rounded_weights = (
        jnp.asarray(weights)
        if array_writer is None
        else array_writer(
            value=weights,
            dtype=np.dtype(canonicalize_dtype(weights.dtype)),
            name=f"{group_name}:validation_weights",
        )
    )
    _validate_weights(weights=np.asarray(rounded_weights), rule=type(rule).__name__)
    del rounded_weights
    points, weights, codes = _enumerate_categories(
        group=group, points=points, weights=weights, continuous=continuous
    )
    active_nodes = (
        jnp.asarray(points, dtype=jnp.result_type(0.0))
        if array_writer is None
        else array_writer(
            value=points,
            dtype=np.dtype(canonical_float_dtype()),
            name=f"{group_name}:base_nodes",
        )
    )
    active_weights = (
        jnp.asarray(weights, dtype=jnp.result_type(0.0))
        if array_writer is None
        else array_writer(
            value=weights,
            dtype=np.dtype(canonical_float_dtype()),
            name=f"{group_name}:weights",
        )
    )
    identity = _fingerprint(
        (
            "shock-node-rule",
            1,
            group.nodes,
            tuple(group.shocks),
            group_name,
            coordinate_kind,
            (age.numerator, age.denominator),
            (
                "PCG64",
                "sobol-LMS-shift",
                "sobol-bits",
                52,
                "normal-inverse",
                "scipy.special.ndtri",
                np.__version__,
                scipy.__version__,
            ),
            np.asarray(active_nodes),
            np.asarray(active_weights),
            codes,
        )
    )
    return NodeSet(
        base_nodes=active_nodes,
        weights=active_weights,
        discrete_codes=MappingProxyType(
            {
                name: (
                    jnp.asarray(values, dtype=jnp.int32)
                    if array_writer is None
                    else array_writer(
                        value=values,
                        dtype=np.dtype(np.int32),
                        name=f"{group_name}:codes:{name}",
                    )
                )
                for name, values in codes.items()
            }
        ),
        group_name=group_name,
        fingerprint=identity,
    )


def _scalar_rule(*, rule: ScalarRule) -> tuple[HostArray, HostArray]:
    if isinstance(rule, GaussHermite):
        nodes, weights = np.polynomial.hermite.hermgauss(rule.n)
        return nodes * math.sqrt(2.0), weights / math.sqrt(math.pi)
    if isinstance(rule, EqualSpaced):
        nodes = np.linspace(-rule.n_std, rule.n_std, rule.n)
        edges = (nodes[:-1] + nodes[1:]) * 0.5
        weights = np.diff(np.concatenate(([0.0], ndtr(edges), [1.0])))
        return nodes, weights
    if isinstance(rule, EqualSpacedUniform):
        return (
            np.asarray([0.5]) if rule.n == 1 else np.linspace(0.0, 1.0, rule.n)
        ), np.full(rule.n, 1.0 / rule.n)
    raise ShockDeclarationError("Unsupported scalar rule.")


def _tensor(axes: list[tuple[HostArray, HostArray]]) -> tuple[HostArray, HostArray]:
    if not axes:
        return np.zeros((1, 0)), np.ones(1)
    nodes = np.stack(
        np.meshgrid(*(axis[0] for axis in axes), indexing="ij"), axis=-1
    ).reshape(-1, len(axes))
    weights = np.ones(1)
    for _, axis_weights in axes:
        weights = (weights[:, None] * axis_weights[None, :]).reshape(-1)
    return nodes, weights


def _enumerate_categories(
    *,
    group: ShockGroup,
    points: HostArray,
    weights: HostArray,
    continuous: tuple[ShockName, ...],
) -> tuple[HostArray, HostArray, dict[ShockName, HostArray]]:
    discrete = {
        name: DiscreteGrid(law.categories).codes
        for name, law in group.shocks.items()
        if isinstance(law, Discrete)
    }
    category_rows = tuple(itertools.product(*discrete.values()))
    if not category_rows:
        category_rows = ((),)
    repeats = len(category_rows)
    base = np.zeros((len(points) * repeats, len(group.shocks)))
    codes = {
        name: np.tile(np.asarray([row[column] for row in category_rows]), len(points))
        for column, name in enumerate(discrete)
    }
    for column, name in enumerate(group.shocks):
        if name in continuous:
            base[:, column] = np.repeat(points[:, continuous.index(name)], repeats)
        else:
            # Exact category enumeration is not an innovation: these placeholders are
            # never consumed by a categorical law, which reads discrete_codes instead.
            base[:, column] = (codes[name] + 0.5) / len(discrete[name])
    return base, np.repeat(weights, repeats), codes
