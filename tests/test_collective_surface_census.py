"""Which test modules exercise the collective / gated-edge surface, and how far.

A module that builds a `GatedEdge` or declares `stakeholders=` can cover the
feature at either of two altitudes. Most drive it through `Model`, so the
public route the documentation describes is what runs. The rest call the engine
directly — `process_regimes`, a bare `solve`, `route_gated_edges` — which is the
right altitude for pinning a kernel's arithmetic and the wrong one for
believing the feature works, because everything between the user's `Model` call
and that kernel is skipped.

Neither altitude is a defect. What would be a defect is the second set growing
without anyone deciding it should, which is how a feature ends up with a large
test suite and an untested public surface. So the partition is pinned exactly:
adding a module to the engine-level set is a deliberate edit here, and moving
one to the public route is a deletion from the list.

The census reads the syntax tree rather than the file's text, so a `GatedEdge`
named in a docstring or commented out does not count as exercising it.
"""

import ast
from pathlib import Path

import pytest

_TESTS_ROOT = Path(__file__).parent

#: Modules that exercise the surface below `Model`, on purpose. Each drives an
#: engine entry point directly to pin something a whole-model run would only
#: cover incidentally: a compiled fold's argument provenance, a treedef, a
#: projector's vmap, a guard that raises before a model could be built.
_ENGINE_LEVEL_MODULES = frozenset(
    {
        "tests/regime_building/test_carried_state_through_gated_self_loop.py",
        "tests/regime_building/test_collective_extended_real.py",
        "tests/regime_building/test_fold_gate_guard.py",
        "tests/regime_building/test_fold_guard_complete.py",
        "tests/regime_building/test_fold_iid_shocks.py",
        "tests/regime_building/test_gated_edge_arg_provenance.py",
        "tests/regime_building/test_gated_edge_gate_process_state_interpolation.py",
        "tests/regime_building/test_gated_edge_simulate_operand_recompute.py",
        "tests/regime_building/test_route_conditions_on_ordinary_draw.py",
        "tests/regime_building/test_same_period_ref_process_state_interpolation.py",
        "tests/regime_building/test_simulate_gate_param_and_leg_selection.py",
        "tests/regime_building/test_simulate_guards.py",
        "tests/regime_building/test_terminal_collective_solve.py",
        "tests/simulation/test_leg_projector_vmap.py",
        "tests/solution/test_edge_topologies_build_each_target_once.py",
    }
)


def _calls(source: str, *, name: str) -> bool:
    """Report whether the source calls `name`, plain or as an attribute."""
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call):
            called = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            if called == name:
                return True
    return False


def _passes_keyword(source: str, *, keyword: str) -> bool:
    """Report whether the source passes `keyword=` to any call."""
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call) and any(
            kw.arg == keyword for kw in node.keywords
        ):
            return True
    return False


def _exercises_the_surface(source: str) -> bool:
    """Report whether the source builds a gated edge or declares stakeholders."""
    return _calls(source, name="GatedEdge") or _passes_keyword(
        source, keyword="stakeholders"
    )


def _census() -> tuple[frozenset[str], frozenset[str]]:
    """Split the modules exercising the surface into public-route and engine-level.

    Returns:
        Tuple of the repo-relative module paths reaching `Model` and those not.
    """
    reaching: set[str] = set()
    not_reaching: set[str] = set()
    for path in sorted(_TESTS_ROOT.rglob("*.py")):
        source = path.read_text()
        if not _exercises_the_surface(source):
            continue
        relative = str(path.relative_to(_TESTS_ROOT.parent))
        target = reaching if _calls(source, name="Model") else not_reaching
        target.add(relative)
    return frozenset(reaching), frozenset(not_reaching)


def test_the_engine_level_set_is_exactly_the_pinned_one():
    """No module joins or leaves the below-`Model` set without saying so here."""
    _reaching, not_reaching = _census()

    assert not_reaching == _ENGINE_LEVEL_MODULES


def test_most_of_the_surface_is_covered_through_the_public_route():
    """The engine-level set is the minority, not the way the feature is tested."""
    reaching, not_reaching = _census()

    assert len(reaching) > len(not_reaching)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("GatedEdge(legs=())", True),
        ("lcm.GatedEdge(legs=())", True),
        ("Regime(stakeholders=('f', 'm'))", True),
        ('"""A docstring naming GatedEdge and stakeholders."""', False),
        ("# GatedEdge(legs=()) commented out\nx = 1", False),
        ("stakeholders = ('f', 'm')", False),
    ],
)
def test_the_census_reads_syntax_not_text(*, source: str, expected: bool):
    """The detector answers both ways, so a negative census means something."""
    assert _exercises_the_surface(source) is expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("Model(regimes={})", True),
        ("lcm.Model(regimes={})", True),
        ('"""Builds no Model, only mentions one."""', False),
    ],
)
def test_reaching_model_is_a_call_not_a_mention(*, source: str, expected: bool):
    """A module is on the public route only if it actually calls `Model`."""
    assert _calls(source, name="Model") is expected
