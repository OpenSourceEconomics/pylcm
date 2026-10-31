"""Build joint shock rules outside tracing, merging nested nodes by construction ids.

Genz-Keister table provenance:
https://chaospy.readthedocs.io/en/master/_modules/chaospy/quadrature/genz_keister.html
Only the nested 1/3/9/19 prefix is supported. Table coordinates integrate exp(-x*x);
normal-base coordinates multiply by sqrt(2), weights divide by sqrt(pi).
"""

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
    NodeRule,
    NodeSet,
    Product,
    QuasiMonteCarlo,
    ScalarRule,
    ShockGroup,
    ShockGroupName,
    ShockName,
    Smolyak,
    Uniform,
    UserNodes,
    _fingerprint,
    _validate_weights,
)

type _NodeId = int | Fraction

_GK_COORDINATES = (
    0.0,
    0.52403354748695763,
    0.87004089535290285,
    1.2247448713915889,
    1.8357079751751868,
    2.0232301911005157,
    2.2665132620567876,
    2.9592107790638380,
    3.6677742159463378,
    4.4995993983103881,
)
_GK_LEVELS = (
    ((0,), (1.7724538509055159,)),
    ((0, 3), (1.1816359006036772, 0.29540897515091930)),
    (
        (0, 1, 3, 5, 7),
        (
            0.45014700975378197,
            0.47869428549114124,
            0.16811892894767771,
            0.014173117873979098,
            1.6708826306882348e-4,
        ),
    ),
    (
        (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
        (
            0.53788160700510168,
            0.36924643368920851,
            0.10838861955003017,
            0.11360729895748269,
            0.032055243099445879,
            -0.011232438489069229,
            5.1133174390883855e-3,
            1.0656589772852267e-4,
            1.0802767206624762e-6,
            1.5295717705322357e-9,
        ),
    ),
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
    elif isinstance(rule, Smolyak):
        points, weights = _smolyak(level=rule.level, uniform=uniform)
    elif rule is None and not continuous:
        points, weights = np.zeros((1, 0)), np.ones(1)
    else:
        raise ShockDeclarationError("Unsupported or missing node rule.")
    _validate_weights(weights=weights, rule=_rule_label(rule))
    rounded_weights = (
        jnp.asarray(weights)
        if array_writer is None
        else array_writer(
            value=weights,
            dtype=np.dtype(canonicalize_dtype(weights.dtype)),
            name=f"{group_name}:validation_weights",
        )
    )
    _validate_weights(weights=np.asarray(rounded_weights), rule=_rule_label(rule))
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


def _rule_label(rule: NodeRule | None) -> str:
    return (
        f"Smolyak(level={rule.level})"
        if isinstance(rule, Smolyak)
        else (type(rule).__name__)
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


def _nested_rule(*, level: int, uniform: bool) -> dict[_NodeId, float]:
    if not uniform:
        if level >= len(_GK_LEVELS):
            raise ShockDeclarationError(
                "Genz-Keister-19 supports normal levels 0 through 3 only."
            )
        ids, weights = _GK_LEVELS[level]
        rule = {
            index: weight / math.sqrt(math.pi)
            for index, weight in zip(ids, weights, strict=True)
        }
        return {**{-index: weight for index, weight in rule.items() if index}, **rule}
    if level == 0:
        return {Fraction(1, 2): 1.0}
    # Integrate the cosine interpolant in the Chebyshev basis; canonical dyadic angles
    # retain nested identity. https://github.com/chebfun/chebfun/blob/master/@chebtech2/quadwts.m
    count = 2**level
    theta = np.arange(count + 1) * math.pi / count
    interior = np.ones(count - 1)
    for frequency in range(1, count // 2):
        interior -= (
            2 * np.cos(2 * frequency * theta[1:-1]) / (4 * frequency * frequency - 1)
        )
    interior -= np.cos(count * theta[1:-1]) / (count * count - 1)
    weights = np.concatenate(
        ([0.5 / (count * count - 1)], interior / count, [0.5 / (count * count - 1)])
    )
    return {
        Fraction(index, count): float(weight) for index, weight in enumerate(weights)
    }


def _smolyak(*, level: int, uniform: tuple[bool, ...]) -> tuple[HostArray, HostArray]:
    dimension = len(uniform)
    if not dimension:
        return np.zeros((1, 0)), np.ones(1)
    if not all(uniform) and level >= len(_GK_LEVELS):
        raise ShockDeclarationError(
            "Genz-Keister-19 supports normal levels 0 through 3 only."
        )
    terms: dict[tuple[_NodeId, ...], list[float]] = {}
    for multi in itertools.product(range(level + 1), repeat=dimension):
        difference = level - sum(multi)
        if difference < 0 or difference >= dimension:
            continue
        coefficient = (-1) ** difference * math.comb(dimension - 1, difference)
        axes = [
            _nested_rule(level=order, uniform=kind)
            for order, kind in zip(multi, uniform, strict=True)
        ]
        for indices in itertools.product(*(tuple(axis) for axis in axes)):
            weight = coefficient * math.prod(
                axis[index] for axis, index in zip(axes, indices, strict=True)
            )
            terms.setdefault(indices, []).append(weight)
    combined = {
        key: math.fsum(value)
        for key, value in sorted(terms.items())
        if math.fsum(value) != 0.0
    }
    nodes = np.asarray(
        [
            [
                _coordinate(index=index, uniform=kind)
                for index, kind in zip(indices, uniform, strict=True)
            ]
            for indices in combined
        ]
    )
    return nodes, np.asarray(tuple(combined.values()))


def _coordinate(*, index: _NodeId, uniform: bool) -> float:
    if uniform:
        if index == Fraction(1, 2):
            return 0.5
        return (1 - math.cos(math.pi * float(index))) * 0.5
    return math.copysign(_GK_COORDINATES[abs(int(index))] * math.sqrt(2), int(index))


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
