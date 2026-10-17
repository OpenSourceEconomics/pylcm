"""Nonfinite NB budgets are flagged exactly inside their root, not at or beyond it.

The child budget `x + 5/4 + sqrt((z - 1/2)^2 - 1/64)` has dyadic roots 3/8 and
5/8, exact in both binary precisions. The parent's middle outer node sits at a
root or one representable step inside or outside it; only the inside step has
a NaN budget. The expected outcome comes from exact rational arithmetic, not
from any solver code.
"""

import itertools
import math
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction

import jax.numpy as jnp
import pytest

from lcm.exceptions import InvalidValueFunctionError
from lcm.typing import ContinuousState, FloatND
from tests.conftest import EXACT_KERNEL_SKIP_REASON
from tests.solution._nb_query_resource_oracle import ResourceClass, classify_resource
from tests.solution.test_nbegm_query_actual_resources import (
    PARAMS,
    fail_if_not_actual_nb_query_path,
    make_model,
)

pytestmark = pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)


def dyadic_resources(*, liquid: ContinuousState, stock: ContinuousState) -> FloatND:
    return liquid + 1.25 + jnp.sqrt((stock - 0.5) ** 2 - 1.0 / 64.0)


def _outer_node(*, root: float, side: str) -> float:
    point = jnp.asarray(root)
    if side != "exact":
        toward = 0.5 if side == "inside" else (0.0 if root < 0.5 else 1.0)
        point = jnp.nextafter(point, jnp.asarray(toward, dtype=point.dtype))
    return float(point)


@pytest.mark.parametrize("runtime_checks", [False, True])
@pytest.mark.parametrize("root", [0.375, 0.625])
@pytest.mark.parametrize("side", ["exact", "inside", "outside"])
def test_adjacent_budget_boundary_sets_status_only_inside_root(
    *, root: float, side: str, runtime_checks: bool
) -> None:
    """Only the representable step inside a root raises `query_coverage_missing`."""
    target = _outer_node(root=root, side=side)
    delta = Fraction(target) - Fraction(1, 2)
    invalid = delta * delta < Fraction(1, 64)
    assert invalid == (side == "inside")
    model = make_model(resources=dyadic_resources, outer_nodes=(0.0, target, 1.0))
    fail_if_not_actual_nb_query_path(model=model)
    if invalid:
        with pytest.raises(InvalidValueFunctionError, match="query_coverage_missing"):
            model.solve(params=PARAMS, log_level="off", runtime_checks=runtime_checks)
    else:
        model.solve(params=PARAMS, log_level="off", runtime_checks=runtime_checks)


@dataclass(frozen=True)
class _Lane:
    """Scalar facts about one query lane that a status rule may consult."""

    bad: bool
    """The actual budget is nonfinite."""
    domain: bool
    """The budget exceeds the floor."""
    eligible: bool
    """Every declared predicate holds."""
    winner: bool
    """The lane's candidate wins the outer comparison."""
    checked: bool
    """Runtime checks are on."""


_GATED_RULES: dict[str, Callable[[_Lane], bool]] = {
    "drop": lambda lane: False,  # noqa: ARG005
    "gate_domain": lambda lane: lane.bad and lane.domain,
    "gate_predicate": lambda lane: lane.bad and lane.eligible,
    "gate_winner": lambda lane: lane.bad and lane.winner,
    "gate_checks": lambda lane: lane.bad and lane.checked,
    "flag_empty": lambda lane: lane.bad or not lane.domain,
}


@pytest.mark.parametrize("name", sorted(_GATED_RULES))
def test_reference_distinguishes_gated_status_rules(*, name: str) -> None:
    """The reference rejects status rules gated on domain, winner or checks."""
    rule = _GATED_RULES[name]
    disagreements = 0
    for resource in (math.nan, math.inf, -math.inf, -1.0, 0.0, 1.0):
        expected = (
            classify_resource(resources=resource, floor=0.0)
            is ResourceClass.INVALID_ARITHMETIC
        )
        for eligible, winner, checked in itertools.product((False, True), repeat=3):
            observed = rule(
                _Lane(
                    bad=not math.isfinite(resource),
                    domain=resource > 0.0,
                    eligible=eligible,
                    winner=winner,
                    checked=checked,
                )
            )
            disagreements += observed != expected
    assert disagreements > 0
