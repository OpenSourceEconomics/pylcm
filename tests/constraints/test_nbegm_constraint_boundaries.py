"""NBEGM compiles only conjunctive liquid-state feasibility boundaries."""

from types import MappingProxyType
from typing import Literal

import pytest

from _lcm.constraints.dispositions import (
    CompileBoundary,
    ConstraintContext,
    Reject,
)
from _lcm.constraints.ir import Const, Ref
from _lcm.constraints.processed import normalize_constraints
from _lcm.constraints.routes import BoundConstraint, ConstraintSite
from _lcm.egm.nbegm_constraint_boundaries import (
    NBEGMFeasibilityBoundaryProgram,
    build_nbegm_feasibility_boundary_compiler,
)
from lcm import Condition, implies, ref
from lcm.typing import BoolND, FloatND


def _compile(
    declaration: Condition | object,
    *,
    param_names: frozenset[str] = frozenset({"limit"}),
) -> CompileBoundary | Reject:
    constraints = normalize_constraints(
        constraints={"eligible": declaration}  # ty: ignore[invalid-argument-type]
    )
    site = ConstraintSite(
        stage="savings_stage",
        function_pool=MappingProxyType({}),
        available_names=frozenset(),
    )
    compiler = build_nbegm_feasibility_boundary_compiler(liquid_state="liquid")
    disposition = compiler(
        bound=BoundConstraint(
            constraint=constraints["eligible"],
            site=site,
            transitive_inputs=constraints["eligible"].dependencies,
        ),
        context=ConstraintContext(
            regime_name="working",
            phase="solve",
            grids=MappingProxyType({}),
            function_names=frozenset({"computed_limit"}),
            param_names=param_names,
        ),
    )
    assert disposition is not None
    return disposition


@pytest.mark.parametrize(
    ("declaration", "feasible_side", "boundary_membership", "threshold_kind"),
    [
        (ref("liquid") < 5.0, "below", "excluded", Const),
        (ref("liquid") <= 5.0, "below", "included", Const),
        (ref("liquid") > 5.0, "above", "excluded", Const),
        (ref("liquid") >= 5.0, "above", "included", Const),
        (ref("limit") > ref("liquid"), "below", "excluded", Ref),
        (ref("limit") >= ref("liquid"), "below", "included", Ref),
        (ref("limit") < ref("liquid"), "above", "excluded", Ref),
        (ref("limit") <= ref("liquid"), "above", "included", Ref),
    ],
)
def test_nbegm_compiles_each_ordered_liquid_boundary(
    declaration: Condition,
    feasible_side: str,
    boundary_membership: Literal["included", "excluded"],
    threshold_kind: type[Ref | Const],
) -> None:
    """Operand order and comparison strictness determine side and ownership."""
    disposition = _compile(declaration)

    assert isinstance(disposition, CompileBoundary)
    payload = disposition.program.payload
    assert isinstance(payload, NBEGMFeasibilityBoundaryProgram)
    assert payload.constraint_name == "eligible"
    assert payload.liquid_state == "liquid"
    assert len(payload.surfaces) == 1
    surface = payload.surfaces[0]
    assert (
        surface.feasible_side,
        surface.includes_boundary,
        isinstance(surface.threshold, threshold_kind),
    ) == (feasible_side, boundary_membership == "included", True)


def test_nbegm_compiles_a_conjunction_in_declaration_order() -> None:
    """Each comparison in an intersection becomes one feasibility surface."""
    declaration = (ref("liquid") >= ref("limit")) & (ref("liquid") < 10.0)

    disposition = _compile(declaration)

    assert isinstance(disposition, CompileBoundary)
    payload = disposition.program.payload
    assert isinstance(payload, NBEGMFeasibilityBoundaryProgram)
    assert [
        (surface.feasible_side, surface.includes_boundary)
        for surface in payload.surfaces
    ] == [("above", True), ("below", False)]
    assert disposition.program.surfaces == disposition.constraint.boundary_surfaces


def test_nbegm_qualifies_a_constraint_parameter_threshold() -> None:
    """A declared parameter is stored under its processed callable's flat name."""
    disposition = _compile(
        ref("liquid") >= ref("limit"),
        param_names=frozenset({"eligible__limit"}),
    )

    assert isinstance(disposition, CompileBoundary)
    payload = disposition.program.payload
    assert isinstance(payload, NBEGMFeasibilityBoundaryProgram)
    assert payload.surfaces[0].threshold == Ref("eligible__limit")


@pytest.mark.parametrize(
    "declaration",
    [
        (ref("liquid") >= 0.0) | (ref("liquid") < 10.0),
        ~(ref("liquid") >= 0.0),
        implies(premise=ref("liquid") >= 0.0, consequent=ref("liquid") < 10.0),
    ],
)
def test_nbegm_rejects_non_conjunctive_structure(declaration: Condition) -> None:
    """A union, complement, or implication is not a set of intersected surfaces."""
    disposition = _compile(declaration)

    assert isinstance(disposition, Reject)
    assert "eligible" in disposition.reason
    assert "working" in disposition.reason
    assert "conjunction" in disposition.reason


def _opaque(liquid: FloatND) -> BoolND:
    return liquid >= 0


def test_nbegm_rejects_an_opaque_constraint() -> None:
    """Arbitrary executable logic supplies no boundary surfaces to compile."""
    disposition = _compile(_opaque)

    assert isinstance(disposition, Reject)
    assert "eligible" in disposition.reason
    assert "opaque" in disposition.reason


@pytest.mark.parametrize("declaration", [ref("liquid") == 5.0, ref("liquid") != 5.0])
def test_nbegm_rejects_non_ordering_comparisons(declaration: Condition) -> None:
    """Equality and inequality do not name one feasible side of a threshold."""
    disposition = _compile(declaration)

    assert isinstance(disposition, Reject)
    assert "<, <=, >, or >=" in disposition.reason


@pytest.mark.parametrize(
    ("declaration", "message"),
    [
        (ref("wealth") >= 0.0, "liquid"),
        (ref("liquid") >= ref("wealth"), "flat parameter"),
        (ref("liquid") >= ref("computed_limit"), "flat parameter"),
        (ref("liquid") >= ref("limit"), "flat parameter"),
    ],
)
def test_nbegm_rejects_unsupported_boundary_operands(
    declaration: Condition, message: str
) -> None:
    """A surface is liquid versus a literal or a parameter available unchanged."""
    param_names = (
        frozenset() if "limit" in declaration.dependencies else frozenset({"limit"})
    )

    disposition = _compile(declaration, param_names=param_names)

    assert isinstance(disposition, Reject)
    assert "eligible" in disposition.reason
    assert message in disposition.reason
