import ast
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

import jax.numpy as jnp
import pandas as pd
import pytest
from dags import with_signature

from _lcm.reachability import (
    EdgeStatus,
    build_model_reachability,
    build_phase_reachability,
)
from _lcm.regime_building.fixed_regime_support import prune_fixed_regime_support
from _lcm.regime_law import bind_regime_law
from lcm import (
    AgeGrid,
    LinSpacedGrid,
    Model,
    Phased,
    Regime,
    StochasticTransition,
    Transition,
    categorical,
)
from lcm.typing import ScalarFloat, ScalarInt, UserParams

type SourceText = str


def test_an_edge_is_the_declared_support_at_its_period() -> None:
    """An edge exists exactly at the periods whose support declares its target."""
    graph = build_phase_reachability(
        n_periods=3,
        active_periods_by_regime={"source": {0}, "target": {1}},
        support_by_period={"source": {0: ("target",)}},
    )

    assert (
        graph.edge_status(period=0, source="source", target="target")
        == EdgeStatus.CONDITIONAL
    )
    assert not graph.has_edge(period=1, source="source", target="target")


def test_a_declared_target_not_covered_next_period_is_rejected() -> None:
    """The graph never drops a declared target; it refuses an uncovered one."""
    with pytest.raises(ValueError, match="covered at the next period"):
        build_phase_reachability(
            n_periods=2,
            active_periods_by_regime={"source": {0}, "target": {0}},
            support_by_period={"source": {0: ("target",)}},
        )


def test_conditional_edge_is_retained_without_runtime_resolution() -> None:
    """A declared conditional edge remains part of the static graph."""
    graph = build_phase_reachability(
        n_periods=2,
        active_periods_by_regime={"a": {0}, "b": {1}},
        support_by_period={"a": {0: ("b",)}},
    )

    assert graph.targets(period=0, source="a") == ("b",)
    assert graph.edge_status(period=0, source="a", target="b") == EdgeStatus.CONDITIONAL
    assert not hasattr(graph, "resolve")


def test_terminal_source_has_no_edge() -> None:
    """A terminal source has no outgoing edge even when support is declared."""
    graph = build_phase_reachability(
        n_periods=2,
        active_periods_by_regime={"dead": {0}, "alive": {1}},
        support_by_period={"dead": {0: ("alive",)}},
        terminal_regimes={"dead"},
    )

    assert not graph.has_edge(period=0, source="dead", target="alive")


def test_forward_closure_is_derived_from_the_static_graph() -> None:
    """Initial support propagates only through retained static edges."""
    graph = build_phase_reachability(
        n_periods=3,
        active_periods_by_regime={"a": {0}, "b": {1}, "c": {2}, "x": {1}},
        support_by_period={"a": {0: ("b",)}, "b": {1: ("c",)}, "x": {1: ("c",)}},
    )

    assert graph.reachable_from({"a"}) == (
        frozenset({"a"}),
        frozenset({"b"}),
        frozenset({"c"}),
    )


def test_unknown_target_is_rejected() -> None:
    """Every declared target belongs to the graph's regime universe."""
    with pytest.raises(ValueError, match="unknown regimes"):
        build_phase_reachability(
            n_periods=2,
            active_periods_by_regime={"a": {0}},
            support_by_period={"a": {0: ("missing",)}},
        )


def test_solution_and_simulation_graphs_use_the_same_builder() -> None:
    """Phase-specific declarations produce phase-specific temporal graphs."""
    graph = build_model_reachability(
        n_periods=2,
        active_periods_by_regime={
            "source": (0,),
            "solve_target": (0, 1),
            "simulate_target": (0, 1),
        },
        support_by_phase={
            "solution": {"source": {0: ("solve_target",)}},
            "simulation": {"source": {0: ("simulate_target",)}},
        },
        terminal_regimes={"solve_target", "simulate_target"},
    )

    assert graph.solution.targets(period=0, source="source") == ("solve_target",)
    assert graph.simulation.targets(period=0, source="source") == ("simulate_target",)


def test_engine_has_no_period_target_inference_helper() -> None:
    """Engine modules consume the graph instead of inferring period targets."""
    package_root = Path(__file__).parents[2] / "src" / "_lcm"
    definitions = [
        (path, node.lineno)
        for path in package_root.rglob("*.py")
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.FunctionDef)
        and node.name in {"get_period_targets", "_active_regimes_at_period"}
    ]

    assert definitions == []


def test_solver_runtime_does_not_import_regime_declaration_topology() -> None:
    """Solver runtime modules depend on the graph, not user declarations."""
    forbidden_modules = {
        "lcm.regime",
        "_lcm.regime_building.canonicalize",
        "_lcm.regime_building.phases",
    }
    imports = [
        (path, node.lineno, node.module)
        for path in _solver_runtime_paths()
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.ImportFrom) and node.module in forbidden_modules
    ]

    assert imports == []


def _solver_runtime_paths() -> list[Path]:
    package_root = Path(__file__).parents[2] / "src" / "_lcm"
    return [
        *sorted((package_root / "solution").rglob("*.py")),
        package_root / "simulation" / "compile.py",
        package_root / "simulation" / "runtime.py",
        package_root / "simulation" / "simulate.py",
        package_root / "simulation" / "transitions.py",
    ]


def _is_simulation_program_bundle(node: ast.expr | None) -> bool:
    """Recognize the canonical engine phase's published program bundle."""
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "programs"
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "simulation"
    )


def _visible_name_bindings(tree: ast.AST) -> Counter[str]:
    """Count assignments, declarations and name captures conservatively."""
    bindings: Counter[str] = Counter()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store | ast.Del):
            bindings[node.id] += 1
        elif isinstance(node, ast.arg):
            bindings[node.arg] += 1
        elif isinstance(node, ast.alias):
            bindings[node.asname or node.name.split(".")[0]] += 1
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            bindings[node.name] += 1
        elif isinstance(node, ast.ExceptHandler | ast.MatchAs | ast.MatchStar):
            if node.name is not None:
                bindings[node.name] += 1
        elif isinstance(node, ast.MatchMapping) and node.rest is not None:
            bindings[node.rest] += 1
        elif isinstance(node, ast.Global | ast.Nonlocal):
            bindings.update(node.names)
    return bindings


def _program_bundle_names(tree: ast.AST) -> set[str]:
    """Allow a receiver only when each visible binding identifies a program bundle.

    Unknown or reassigned names stay forbidden. This deliberately conservative
    source guard does not infer arbitrary aliases or interprocedural types.
    """
    bindings = _visible_name_bindings(tree)
    bundle_bindings: Counter[str] = Counter()
    imported_types = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.ImportFrom)
            and node.module == "_lcm.simulation.program_types"
        ):
            imported_types.update(
                alias.asname or alias.name
                for alias in node.names
                if alias.name == "SimulationPrograms"
            )
    imported_types = {name for name in imported_types if bindings[name] == 1}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign | ast.AnnAssign):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if _is_simulation_program_bundle(node.value):
                bundle_bindings.update(
                    target.id for target in targets if isinstance(target, ast.Name)
                )
        elif (
            isinstance(node, ast.arg)
            and isinstance(node.annotation, ast.Name)
            and node.annotation.id in imported_types
        ):
            bundle_bindings[node.arg] += 1
    return {name for name, count in bundle_bindings.items() if count == bindings[name]}


def _raw_transition_reads(source: SourceText) -> list[ast.Attribute]:
    """Reject raw declaration reads while admitting proved program-family reads."""
    tree = ast.parse(source)
    program_names = _program_bundle_names(tree)
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and node.attr in {"transition", "state_transitions"}
        and not (
            node.attr == "transition"
            and (
                _is_simulation_program_bundle(node.value)
                or (isinstance(node.value, ast.Name) and node.value.id in program_names)
            )
        )
    ]


@pytest.mark.parametrize(
    ("source", "expected_reads"),
    [
        ("regime.transition", ["regime.transition"]),
        ("regime.state_transitions", ["regime.state_transitions"]),
        ("programs.transition", ["programs.transition"]),
        ("programs = regime\nprograms.transition", ["programs.transition"]),
        ("regime.simulation.programs.transition", []),
        ("programs = regime.simulation.programs\nprograms.transition", []),
        (
            (
                "programs = regime.simulation.programs\n"
                "programs = regime\nprograms.transition"
            ),
            ["programs.transition"],
        ),
        (
            "programs = regime.simulation.programs\nprograms.state_transitions",
            ["programs.state_transitions"],
        ),
        (
            (
                "from _lcm.simulation.program_types import "
                "SimulationPrograms as Bundle\n"
                "def dispatch(programs: Bundle):\n    return programs.transition"
            ),
            [],
        ),
        (
            "def dispatch(programs: Regime):\n    return programs.transition",
            ["programs.transition"],
        ),
        (
            (
                "from _lcm.simulation.program_types import "
                "SimulationPrograms as Bundle\n"
                "Bundle = Regime\n"
                "def dispatch(programs: Bundle):\n    return programs.transition"
            ),
            ["programs.transition"],
        ),
    ],
)
def test_raw_transition_guard_distinguishes_program_bundles(
    *, source: SourceText, expected_reads: list[str]
) -> None:
    """Only canonical program receivers pass; declaration and shadowed names fail."""
    assert [
        ast.unparse(node) for node in _raw_transition_reads(source)
    ] == expected_reads


def test_solver_runtime_does_not_inspect_regime_transition_mapping_keys() -> None:
    """Solver runtime modules never read raw transition declaration mappings.

    Which regime pairs are reachable is a graph property; a solver module
    inspecting the raw declared transition or state-transition mapping to
    decide targets would bypass the single-source-of-truth graph.
    """
    accesses = [
        (path, node.lineno)
        for path in _solver_runtime_paths()
        for node in _raw_transition_reads(path.read_text(encoding="utf-8"))
    ]

    assert accesses == []


def test_continuation_targets_are_not_derived_from_law_bundle_keys() -> None:
    """`carry_targets` (a law-bundle-keyed target set) must not reappear.

    Continuation/process-target membership is `phase_reachability.union_targets`
    (or an equivalent graph query) — never a set of regime names collected from
    `flat_nested_transitions` or other state-law-bundle keys.
    """
    package_root = Path(__file__).parents[2] / "src" / "_lcm"
    hits = [
        (path, lineno)
        for path in package_root.rglob("*.py")
        for lineno, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        )
        if "carry_targets" in line
    ]

    assert hits == []


@pytest.mark.parametrize(
    "fixed_params",
    [{"probability": 0.0}, {"edges": {"source": {"high": {"probability": 0.0}}}}],
)
def test_fixed_zero_probability_removes_target_problem(
    fixed_params: UserParams,
) -> None:
    """A fixed zero cell creates neither a physical visit nor a value problem."""

    @categorical(ordered=False)
    class RegimeId:
        source: ScalarInt
        low: ScalarInt
        high: ScalarInt

    def probability(probability: float) -> ScalarFloat:

        return jnp.asarray(probability)

    def one() -> ScalarFloat:

        return jnp.asarray(1.0)

    model = Model(
        edges={
            "source": Transition(
                targets={"low": 0, "high": 0},
                law={
                    "low": StochasticTransition(func=one),
                    "high": StochasticTransition(func=probability),
                },
            )
        },
        regimes={
            "source": Regime(
                functions={"utility": lambda: 0.0},
                state_transitions={
                    "wealth": {"high": lambda unused_entry_param: unused_entry_param}
                },
            ),
            "low": Regime(functions={"utility": lambda: 1.0}),
            "high": Regime(
                states={"wealth": LinSpacedGrid(start=0.0, stop=1.0, n_points=2)},
                functions={"utility": lambda wealth: wealth},
            ),
        },
        ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
        regime_id_class=RegimeId,
        initial_nodes={0: "source"},
        fixed_params=fixed_params,
        enable_jit=False,
    )
    assert model.reachability.solution.targets(period=0, source="source") == ("low",)
    assert model.reachability.simulation.targets(period=0, source="source") == ("low",)
    assert (1, "high") not in model.reachability.nodes
    assert (1, "high") not in model.reachability.visited_nodes
    assert "high" not in model.get_params_template()["source"]


@pytest.mark.parametrize(
    ("probability_value", "removes_edge"), [(0.0, True), (-0.0, True), (1e-20, False)]
)
def test_fixed_probability_support_uses_exact_zero(
    *, probability_value: float, removes_edge: bool
) -> None:
    """Signed zero removes an edge, while a representable positive mass retains it."""

    regime, law = (
        Regime(
            functions={"utility": lambda: 0.0},
        ),
        {
            "low": StochasticTransition(func=lambda: jnp.asarray(1.0)),
            "high": StochasticTransition(func=lambda probability: probability),
        },
    )
    reduced = prune_fixed_regime_support(
        user_regimes={"source": regime},
        laws={"source": bind_regime_law(law)},
        fixed_params={"probability": probability_value},
    )
    transition = reduced.laws["source"].transition
    assert isinstance(transition, Mapping)
    assert set(transition) == ({"low"} if removes_edge else {"low", "high"})
    assert reduced.consumed_param_keys == (
        frozenset({"probability"}) if removes_edge else frozenset()
    )


@pytest.mark.parametrize(
    "dependency",
    [
        "state",
        "action",
        "age",
        "period",
        "free",
        "state_ancestor",
        "age_ancestor",
        "free_ancestor",
    ],
)
def test_fixed_probability_retains_runtime_dependencies_without_evaluating(
    dependency: str,
) -> None:
    """A runtime input anywhere in the dependency graph prevents zero pruning."""

    evaluated: list[str] = []

    def forbidden_probability[Ignored](**_kwargs: Ignored) -> ScalarFloat:
        evaluated.append("probability")
        raise AssertionError("A runtime-dependent cell must not be evaluated.")

    arg_name = {
        "state": "wealth",
        "action": "choice",
        "age": "age",
        "period": "period",
        "free": "free_parameter",
    }.get(dependency, "helper")
    probability = with_signature(
        args={arg_name: "float", "probability": "float"}, return_annotation="float"
    )(forbidden_probability)
    helpers = {}
    if dependency.endswith("_ancestor"):
        ancestor_name = {
            "state_ancestor": "wealth",
            "age_ancestor": "age",
            "free_ancestor": "free_parameter",
        }[dependency]
        helpers["helper"] = with_signature(
            args={ancestor_name: "float"}, return_annotation="float"
        )(forbidden_probability)
    regime, law = (
        Regime(
            states={"wealth": LinSpacedGrid(start=0.0, stop=1.0, n_points=2)},
            actions={"choice": LinSpacedGrid(start=0.0, stop=1.0, n_points=2)},
            state_transitions={"wealth": lambda wealth: wealth},
            functions={"utility": lambda wealth: wealth, **helpers},
        ),
        {
            "low": StochasticTransition(func=lambda: jnp.asarray(1.0)),
            "high": StochasticTransition(func=probability),
        },
    )
    reduced = prune_fixed_regime_support(
        user_regimes={"source": regime},
        laws={"source": bind_regime_law(law)},
        fixed_params={"probability": 0.0},
    )
    transition = reduced.laws["source"].transition
    assert isinstance(transition, Mapping)
    assert set(transition) == {"low", "high"}
    assert reduced.consumed_param_keys == frozenset()
    assert evaluated == []


def test_fixed_probability_can_follow_constant_function_ancestors() -> None:
    """A constant helper DAG proves zero with its own fixed-parameter namespace."""

    regime, law = (
        Regime(
            functions={
                "utility": lambda: 0.0,
                "helper": lambda probability: probability,
            },
        ),
        {
            "low": StochasticTransition(func=lambda: jnp.asarray(1.0)),
            "high": StochasticTransition(func=lambda helper: helper),
        },
    )
    reduced = prune_fixed_regime_support(
        user_regimes={"source": regime},
        laws={"source": bind_regime_law(law)},
        fixed_params={"source": {"helper": {"probability": 0.0}}},
    )
    transition = reduced.laws["source"].transition
    assert isinstance(transition, Mapping)
    assert set(transition) == {"low"}
    assert reduced.consumed_param_keys == frozenset({"source__helper__probability"})


def test_fixed_series_probability_retains_coordinate_dependent_support() -> None:
    """A fixed Series remains conditional because its entries depend on coordinates."""

    regime, law = (
        Regime(
            functions={"utility": lambda: 0.0},
        ),
        {
            "low": StochasticTransition(func=lambda: jnp.asarray(1.0)),
            "high": StochasticTransition(func=lambda probability: probability),
        },
    )
    reduced = prune_fixed_regime_support(
        user_regimes={"source": regime},
        laws={"source": bind_regime_law(law)},
        fixed_params={
            "probability": pd.Series([0.0, 1.0], index=pd.Index([0, 1], name="age"))
        },
    )
    transition = reduced.laws["source"].transition
    assert isinstance(transition, Mapping)
    assert set(transition) == {"low", "high"}
    assert reduced.consumed_param_keys == frozenset()


def test_fixed_probability_phase_support_and_handoff_laws_are_independent() -> None:
    """Each phase drops only its zero destination and that destination's handoff."""

    regime, law = (
        Regime(
            functions={"utility": lambda: 0.0},
            state_transitions={
                "wealth": {
                    "low": lambda low_entry: low_entry,
                    "high": lambda high_entry: high_entry,
                }
            },
        ),
        Phased(
            solve={
                "low": StochasticTransition(func=lambda: jnp.asarray(1.0)),
                "high": StochasticTransition(
                    func=lambda perceived_probability: perceived_probability
                ),
            },
            simulate={
                "low": StochasticTransition(func=lambda: jnp.asarray(0.0)),
                "high": StochasticTransition(
                    func=lambda realized_probability: realized_probability
                ),
            },
        ),
    )
    reduced = prune_fixed_regime_support(
        user_regimes={"source": regime},
        laws={"source": bind_regime_law(law)},
        fixed_params={
            "perceived_probability": 0.0,
            "realized_probability": 1.0,
            "low_entry": 2.0,
            "high_entry": 3.0,
        },
    )
    source = reduced.user_regimes["source"]
    transition = reduced.laws["source"].transition
    state_law = source.state_transitions["wealth"]
    assert isinstance(transition, Phased)
    assert isinstance(state_law, Phased)
    assert isinstance(transition.solve, Mapping)
    assert isinstance(transition.simulate, Mapping)
    assert set(transition.solve) == {"low"}
    assert set(transition.simulate) == {"high"}
    assert set(state_law.solve) == {"low"}
    assert set(state_law.simulate) == {"high"}
    assert reduced.consumed_param_keys == frozenset(
        {"perceived_probability", "low_entry", "high_entry"}
    )


def test_fixed_zero_mass_row_stays_available_to_probability_validation() -> None:
    """An invalid all-zero distribution keeps its cells and nonterminal status."""

    regime, law = (
        Regime(
            functions={"utility": lambda: 0.0},
        ),
        {
            "low": StochasticTransition(func=lambda: jnp.asarray(0.0)),
            "high": StochasticTransition(func=lambda probability: probability),
        },
    )
    reduced = prune_fixed_regime_support(
        user_regimes={"source": regime},
        laws={"source": bind_regime_law(law)},
        fixed_params={"probability": 0.0},
    )
    assert not reduced.laws["source"].terminal
    transition = reduced.laws["source"].transition
    assert isinstance(transition, Mapping)
    assert set(transition) == {"low", "high"}
    assert reduced.consumed_param_keys == frozenset()


def test_shared_probability_cell_follows_each_phases_helper_dag() -> None:
    """One shared law retains different targets when its helper differs by phase."""
    regime, law = (
        Regime(
            functions={
                "utility": lambda: 0.0,
                "helper": Phased(
                    solve=lambda probability: probability,
                    simulate=lambda probability: 1.0 - probability,
                ),
            },
        ),
        {
            "low": StochasticTransition(func=lambda helper: 1.0 - helper),
            "high": StochasticTransition(func=lambda helper: helper),
        },
    )
    reduced = prune_fixed_regime_support(
        user_regimes={"source": regime},
        laws={"source": bind_regime_law(law)},
        fixed_params={"probability": 0.0},
    )
    transition = reduced.laws["source"].transition
    assert isinstance(transition, Phased)
    assert isinstance(transition.solve, Mapping)
    assert isinstance(transition.simulate, Mapping)
    assert set(transition.solve) == {"low"}
    assert set(transition.simulate) == {"high"}
    assert reduced.consumed_param_keys == frozenset({"probability"})
