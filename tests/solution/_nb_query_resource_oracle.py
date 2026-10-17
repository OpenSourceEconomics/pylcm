"""Scalar reference that classifies an actual budget before any economic test.

The reference is not a Bellman or value interpolator. Each resource fixture is
affine in liquid and finite at both represented passive source nodes, and the
nonfinite fixtures are invalid only at the reached interior stock 1/2.
"""

import math
from enum import Enum


class ResourceKind(Enum):
    """Child resource fixtures, named by their value at stock 1/2."""

    NAN = "nan"
    POSITIVE_INFINITY = "positive_infinity"
    NEGATIVE_INFINITY = "negative_infinity"
    POSITIVE = "positive"
    ZERO_AT_FLOOR = "zero_at_floor"
    NEGATIVE_FINITE = "negative_finite"


class ResourceClass(Enum):
    """Integrity first, then the economic budget comparison."""

    INVALID_ARITHMETIC = "invalid_arithmetic"
    INVALID_BOUND = "invalid_bound"
    FEASIBLE = "feasible"
    ECONOMIC_EMPTY = "economic_empty"


NONFINITE_KINDS = frozenset(
    {ResourceKind.NAN, ResourceKind.POSITIVE_INFINITY, ResourceKind.NEGATIVE_INFINITY}
)
SOURCE_LIQUID = (0.0, 0.25, 0.5, 0.75, 1.0, 2.0, 4.0, 8.0)
SOURCE_STOCK = (0.0, 1.0)


def resource_value(*, kind: ResourceKind, liquid: float, stock: float) -> float:
    """Evaluate a fixture literally in scalar Python arithmetic."""
    match kind:
        case ResourceKind.NAN:
            radicand = (stock - 0.5) ** 2 - 0.01
            return math.nan if radicand < 0.0 else liquid + 1.25 + math.sqrt(radicand)
        case ResourceKind.POSITIVE_INFINITY:
            denominator = (stock - 0.5) ** 2
            return math.inf if denominator == 0.0 else liquid + 1.25 + 1.0 / denominator
        case ResourceKind.NEGATIVE_INFINITY:
            denominator = (stock - 0.5) ** 2
            return -math.inf if denominator == 0.0 else liquid + 8.0 - 1.0 / denominator
        case ResourceKind.POSITIVE:
            return liquid + 1.25 + 4.0 * stock * (1.0 - stock)
        case ResourceKind.ZERO_AT_FLOOR:
            return liquid + 1.25 - 5.0 * stock * (1.0 - stock)
        case ResourceKind.NEGATIVE_FINITE:
            return liquid + 1.25 - 64.0 * stock * (1.0 - stock)


def classify_resource(*, resources: float, floor: float) -> ResourceClass:
    """Classify a budget; integrity has priority over `R > b`."""
    if not math.isfinite(resources):
        return ResourceClass.INVALID_ARITHMETIC
    if not math.isfinite(floor):
        return ResourceClass.INVALID_BOUND
    return ResourceClass.FEASIBLE if resources > floor else ResourceClass.ECONOMIC_EMPTY


def expected_interior_class(*, kind: ResourceKind) -> ResourceClass:
    """State the class each fixture is built to have at liquid 0, stock 1/2."""
    if kind in NONFINITE_KINDS:
        return ResourceClass.INVALID_ARITHMETIC
    if kind is ResourceKind.POSITIVE:
        return ResourceClass.FEASIBLE
    return ResourceClass.ECONOMIC_EMPTY


def fail_if_fixture_domain_wrong(*, kind: ResourceKind) -> None:
    """Fail unless source nodes are finite and the interior has its stated class."""
    bad = [
        (liquid, stock)
        for stock in SOURCE_STOCK
        for liquid in SOURCE_LIQUID
        if not math.isfinite(resource_value(kind=kind, liquid=liquid, stock=stock))
    ]
    if bad:
        msg = f"Fixture {kind} is nonfinite at source nodes {bad}."
        raise ValueError(msg)
    actual = classify_resource(
        resources=resource_value(kind=kind, liquid=0.0, stock=0.5), floor=0.0
    )
    if actual is not expected_interior_class(kind=kind):
        msg = f"Fixture {kind} has interior class {actual}."
        raise ValueError(msg)
