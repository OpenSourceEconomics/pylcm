"""Public capability metadata describes configured solvers and renders their tables."""

import dataclasses
import importlib
import importlib.util
from pathlib import Path

import pytest

import lcm.solvers as solvers_api
from lcm import LinSpacedGrid, solver_api
from lcm.solvers import (
    DCEGM,
    EGM,
    NBEGM,
    NEGM,
    NNBEGM,
    ExactEnvelope,
    FiniteOuterGrid,
    FUESEnvelope,
    GridSearch,
    Solver,
)

_GRID = LinSpacedGrid(start=0.0, stop=2.0, n_points=3)
_SOLVERS = {
    "GridSearch": GridSearch(),
    "EGM": EGM(savings_grid=_GRID),
    "DCEGM": DCEGM(savings_grid=_GRID),
    "NBEGM": NBEGM(savings_grid=_GRID),
    "NEGM": NEGM(inner=DCEGM(savings_grid=_GRID), outer_grid=_GRID),
    "NNBEGM": NNBEGM(
        inner=NBEGM(savings_grid=_GRID), outer_search=FiniteOuterGrid(grid=_GRID)
    ),
}
_EXPECTED = {
    "GridSearch": (("action_product",), ("cell",), (), True, True),
    "EGM": ((), (), (), False, False),
    "DCEGM": (
        ("stochastic_node",),
        ("cell", "savings_point", "euler_point", "envelope_cell"),
        (),
        True,
        False,
    ),
    "NBEGM": (("stochastic_node", "interval", "branch"), ("cell",), (), False, True),
    "NEGM": (
        ("stochastic_node", "outer_candidate"),
        ("cell", "savings_point", "euler_point", "envelope_cell"),
        (),
        False,
        False,
    ),
    "NNBEGM": (
        ("stochastic_node", "interval", "branch"),
        ("cell",),
        ("outer_candidate",),
        False,
        True,
    ),
}


@pytest.mark.parametrize("name", tuple(_SOLVERS))
def test_configured_solver_exposes_its_execution_roles(name: str) -> None:
    """Metadata distinguishes reduced, tiled and host axes and preference support."""
    cap = _SOLVERS[name].capabilities
    assert (
        cap.reduced_axes,
        cap.tiled_axes,
        cap.host_axes,
        cap.supports_ev1_taste_shocks,
        cap.supports_nonlinear_certainty_equivalent,
    ) == _EXPECTED[name]


def test_public_capabilities_share_one_dependency_safe_type() -> None:
    """Both public entry points export the exact same capability type."""
    assert (
        solver_api.SolverExecutionCapabilities
        is solvers_api.SolverExecutionCapabilities
    )


def test_required_capability_declaration_has_a_new_api_version() -> None:
    """A plugin must explicitly target the capability-bearing solver contract."""
    assert solver_api.SOLVER_API_VERSION == 3


def test_solver_requires_explicit_capability_metadata() -> None:
    """An extension cannot inherit an accidental promise about execution support."""
    assert "capabilities" in Solver.__abstractmethods__


@pytest.mark.parametrize(
    ("envelope", "expected"), [(ExactEnvelope(), True), (FUESEnvelope(), False)]
)
def test_dcegm_envelope_axis_follows_the_selected_backend(
    *, envelope: ExactEnvelope | FUESEnvelope, expected: bool
) -> None:
    """Only the exact backend advertises envelope-cell tiling."""
    cap = DCEGM(savings_grid=_GRID, envelope=envelope).capabilities
    assert ("envelope_cell" in cap.axis_names) is expected


def test_nested_inner_axis_configuration_is_preserved() -> None:
    """A nested fast-envelope solver does not advertise the exact backend's axis."""
    cap = NEGM(
        inner=DCEGM(savings_grid=_GRID, envelope=FUESEnvelope()), outer_grid=_GRID
    ).capabilities
    assert cap.axis_names == frozenset(
        {"stochastic_node", "outer_candidate", "cell", "savings_point", "euler_point"}
    )


def test_host_repetition_is_distinct_from_compiled_reduction() -> None:
    """NNBEGM describes the actual repeated graph keys without relabeling the cores."""
    assert _SOLVERS["NNBEGM"].capabilities.host_driven_programs == (
        "adjuster:main",
        "adjuster:replay",
    )


@pytest.mark.parametrize("name", tuple(_SOLVERS))
def test_capability_queries_leave_solver_configuration_unchanged(name: str) -> None:
    """Metadata queries add no fingerprinted instance fields or configuration."""
    solver = _SOLVERS[name]
    before = tuple(
        (field.name, getattr(solver, field.name))
        for field in dataclasses.fields(solver)
    )
    _ = solver.capabilities
    assert (
        tuple(
            (field.name, getattr(solver, field.name))
            for field in dataclasses.fields(solver)
        )
        == before
    )


def test_capability_axes_are_frozen_at_construction() -> None:
    """Mutating a supplied axis list cannot alter the published configuration."""
    axes = ["custom_axis"]
    cap = dataclasses.replace(_SOLVERS["GridSearch"].capabilities, reduced_axes=axes)
    axes.append("later")
    assert cap.reduced_axes == ("custom_axis",)


@pytest.mark.parametrize(
    "changes",
    [
        {"required_declaration": ""},
        {"reduced_axes": ("",)},
        {"tiled_axes": ("cell", "cell")},
        {"host_axes": ("cell",)},
    ],
)
def test_capabilities_refuse_ambiguous_or_empty_names(
    changes: dict[str, object],
) -> None:
    """Published metadata has nonempty names and unambiguous axis roles."""
    cap = _SOLVERS["GridSearch"].capabilities
    with pytest.raises(ValueError, match=r"nonempty|duplicate|exactly one"):
        dataclasses.replace(cap, **changes)


def test_reference_tables_are_generated_from_public_metadata() -> None:
    """The checked-in tables match the configured solver properties exactly."""
    assert importlib.util.find_spec("_lcm.docs") is not None
    renderer = importlib.import_module("_lcm.docs.render_solver_tables")
    text = (Path(__file__).parents[1] / "docs/reference/solvers.md").read_text(
        encoding="utf-8"
    )
    assert renderer.table_region(text=text) == renderer.render_solver_tables(
        solvers=_SOLVERS
    )
