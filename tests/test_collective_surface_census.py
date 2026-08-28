"""Which test modules exercise the collective / gated-edge surface, and how far.

A module that declares a `CollectiveUtility`, a `ValueDependentConstraint` or a
`ValueDependentTransition` can cover the feature at either of two altitudes.
Most drive it through `Model`, so the public route the documentation describes
is what runs. The rest call the engine directly — `process_regimes`, a bare
`solve`, `route_gated_edges` — which is the right altitude for pinning a
kernel's arithmetic and the wrong one for believing the feature works, because
everything between the user's `Model` call and that kernel is skipped.

Neither altitude is a defect. What would be a defect is the second set growing
without anyone deciding it should, which is how a feature ends up with a large
test suite and an untested public surface. So the partition is pinned exactly:
adding a module to the engine-level set is a deliberate edit here, and moving
one to the public route is a deletion from the list.

The census reads the syntax tree rather than the file's text, so a declaration
named in a docstring or commented out does not count as exercising it.
"""

import ast
from pathlib import Path, PurePath, PureWindowsPath

import pytest

_TESTS_ROOT = Path(__file__).parent

#: Modules that exercise the surface below `Model`, on purpose. Each drives an
#: engine entry point directly to pin something a whole-model run would only
#: cover incidentally: a compiled fold's argument provenance, a treedef, a
#: projector's vmap, a guard that raises before a model could be built, or the
#: decomposition a declaration takes apart into, which is a property of the
#: `Regime` alone and so has no model in it to build.
_ENGINE_LEVEL_MODULES = frozenset(
    {
        "tests/regime_building/test_carried_state_through_gated_self_loop.py",
        "tests/regime_building/test_collective_extended_real.py",
        "tests/regime_building/test_decomposed_views.py",
        "tests/regime_building/test_fold_gate_guard.py",
        "tests/regime_building/test_fold_guard_complete.py",
        "tests/regime_building/test_fold_iid_shocks.py",
        "tests/regime_building/test_gated_edge_arg_provenance.py",
        "tests/regime_building/test_gated_edge_gate_process_state_interpolation.py",
        "tests/regime_building/test_gated_edge_simulate_operand_recompute.py",
        "tests/regime_building/test_route_conditions_on_ordinary_draw.py",
        "tests/regime_building/test_same_period_ref_process_state_interpolation.py",
        "tests/regime_building/test_same_period_ref_projection_free_params.py",
        "tests/regime_building/test_simulate_gate_param_and_leg_selection.py",
        "tests/regime_building/test_simulate_guards.py",
        "tests/regime_building/test_terminal_collective_solve.py",
        "tests/regime_building/test_with_engine_functions.py",
        "tests/simulation/test_leg_projector_vmap.py",
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


#: What a module writes when it exercises the collective / gated-edge surface.
#: There is one vocabulary now: a regime says who its stakeholders are, what a
#: value-reading constraint is, and where a value-dependent transition routes,
#: each in the slot it already has.
_DECLARATIONS = (
    "CollectiveUtility",
    "ValueDependentConstraint",
    "ValueDependentTransition",
)


def _exercises_the_surface(source: str) -> bool:
    """Report whether the source declares a household, a value constraint or an edge."""
    return any(_calls(source, name=name) for name in _DECLARATIONS)


def _module_name(*, path: PurePath, root: PurePath) -> str:
    """Name one module the way the pinned set spells it: relative, `/`-separated."""
    return path.relative_to(root).as_posix()


def _census() -> tuple[frozenset[str], frozenset[str]]:
    """Split the modules exercising the surface into public-route and engine-level.

    Returns:
        Tuple of the repo-relative module paths reaching `Model` and those not.
    """
    reaching: set[str] = set()
    not_reaching: set[str] = set()
    for path in sorted(_TESTS_ROOT.rglob("*.py")):
        # Named, not defaulted: house style puts literal UTF-8 in sources, and a
        # platform whose default codec is not UTF-8 cannot decode them.
        source = path.read_text(encoding="utf-8")
        if not _exercises_the_surface(source):
            continue
        relative = _module_name(path=path, root=_TESTS_ROOT.parent)
        target = reaching if _calls(source, name="Model") else not_reaching
        target.add(relative)
    return frozenset(reaching), frozenset(not_reaching)


def test_sources_are_read_as_utf_8_whatever_the_platform_default_is():
    """The census decodes every source as UTF-8, not as the locale's codec.

    House style puts literal `—`, `→` and `μ` in source files, and a platform
    whose default codec is not UTF-8 cannot decode them: the read raises before
    the census can parse anything. Naming the encoding at the read makes the
    census independent of the machine it runs on.
    """
    recorded: list[str | None] = []
    original = Path.read_text

    def recording_read_text(self: Path, *args: object, **kwargs: object) -> str:
        recorded.append(kwargs.get("encoding"))  # ty: ignore[invalid-argument-type]
        return original(self, *args, **kwargs)  # ty: ignore[invalid-argument-type]

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(Path, "read_text", recording_read_text)
        _census()

    assert set(recorded) == {"utf-8"}


def test_module_names_are_posix_whatever_the_platform_separator_is():
    """A module's census name uses `/`, so the pinned set is platform-independent.

    The pinned set is written with forward slashes. A name built from the
    platform separator spells the same module `tests\\solution\\x.py` on Windows,
    so every entry would read as both added and removed at once.
    """
    windows_path = PureWindowsPath(r"D:\a\pylcm\pylcm\tests\solution\test_x.py")

    assert (
        _module_name(path=windows_path, root=PureWindowsPath(r"D:\a\pylcm\pylcm"))
        == "tests/solution/test_x.py"
    )


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
        ("CollectiveUtility(utilities={})", True),
        ("lcm.CollectiveUtility(utilities={})", True),
        ("ValueDependentConstraint(predicate=p)", True),
        ("ValueDependentTransition(gate=g)", True),
        ('"""A docstring naming CollectiveUtility and stakeholders."""', False),
        ("# CollectiveUtility(utilities={}) commented out\nx = 1", False),
        ("collective_utility = 1", False),
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


#: What a regime's three declarations decompose into. A model author never
#: writes one: `stakeholders`, `pareto_objective`, `value_constraints`,
#: `same_period_refs` and `gated_edges` are read off a regime, and `GatedEdge`
#: is the engine's own form of an edge.
_DECOMPOSED_NAMES = (
    "stakeholders",
    "pareto_objective",
    "value_constraints",
    "same_period_refs",
    "gated_edges",
)


def _writes_a_decomposed_name(source: str) -> list[str]:
    """Return the decomposed names this source writes as a keyword or parameter.

    Reading one back off a regime stays legal — that is what they are for — so
    only the writing forms count: a keyword argument, and a helper parameter
    that would forward one into a `Regime`.
    """
    written = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call):
            called = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            if called in _ENGINE_CONSTRUCTORS:
                continue
            written += [kw.arg for kw in node.keywords if kw.arg in _DECOMPOSED_NAMES]
        elif isinstance(node, ast.FunctionDef):
            written += [
                arg.arg
                for arg in node.args.args + node.args.kwonlyargs
                if arg.arg in _DECOMPOSED_NAMES
            ]
    return written


#: Callables that take a decomposed name legitimately: the engine helpers a
#: unit test drives directly.
_ENGINE_CONSTRUCTORS = frozenset({"MockRegime", "_MockRegime", "build_pareto_weights"})

#: Modules that build the ENGINE's regime rather than the author's, and so
#: write the engine's own vocabulary on purpose. One replaces `stakeholders` on
#: a canonical regime the solver already produced; the other mocks a canonical
#: regime outright. Neither reaches `Regime.__init__`, so neither is a second
#: way to write a model — but both are pinned here so that becoming one is a
#: deliberate edit.
_ENGINE_VOCABULARY_MODULES = frozenset(
    {
        "tests/regime_building/test_simulate_gate_param_and_leg_selection.py",
        "tests/solution/test_edge_topologies_build_each_target_once.py",
    }
)


def test_no_test_writes_what_a_declaration_decomposes_to():
    """There is one way to write a collective regime, and it is the economics.

    A test that constructed the decomposed form would be pinning a second
    spelling into the API — the thing this vocabulary exists to remove. Reading
    one back is untouched; only writing one is out.
    """
    offenders = {}
    for path in sorted(_TESTS_ROOT.rglob("test_*.py")):
        module = _module_name(path=path, root=_TESTS_ROOT.parent)
        if module in _ENGINE_VOCABULARY_MODULES:
            continue
        written = sorted(
            set(_writes_a_decomposed_name(path.read_text(encoding="utf-8")))
        )
        if written:
            offenders[module] = written

    assert offenders == {}


def test_the_engine_vocabulary_exemption_is_still_earned():
    """An exempt module that stops writing the engine's vocabulary leaves the list."""
    unearned = sorted(
        module
        for module in _ENGINE_VOCABULARY_MODULES
        if not _writes_a_decomposed_name(
            (_TESTS_ROOT.parent / module).read_text(encoding="utf-8")
        )
    )

    assert unearned == []
