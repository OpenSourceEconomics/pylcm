"""Static class invariant: a collective builder may not drop a phase role.

Round 2 found `_build_Q_and_F_per_period`'s collective branch calling
`get_Q_and_F_collective` with only `(transitions, functions)`, dropping the four
arguments that carry the solve/simulate phase split. That witness was repaired -- and
then the SAME class turned up one function away, in `get_Q_and_F_terminal_collective`.
A signature that DEFAULTS a phase argument makes its omission silent by construction,
so per-site repair keeps losing to the next collective twin.

This is the fail-closed replacement, at the level the class actually lives: the
source is parsed and every collective twin is required to expose -- and correctly
route -- every phase-role argument its singleton twin exposes. Adding a new
`get_X_collective` next to a `get_X` that takes phase arguments FAILS THIS TEST until
the twin threads them too.

The mutation catalogue below is the evidence that the checker has teeth: each
mutation is one member of the counterexample class (drop the argument from the
signature, drop it at the dispatch, pair a role with the wrong pool, drop the
diagnostic vocabulary, skip age specialization, hide a builder in a stakeholder-only
branch), and every one must be rejected. Two reproduce the defects actually found.

Checked against the REAL history, not only against synthetic mutations: at `f0f7173`
(the round-2 baseline) `get_Q_and_F_collective` exposed NONE of the four phase arguments
and the dispatch passed none; at `27bc15f` the terminal twin still lacked
`next_state_names`. The twin-pair rule rejects both trees, and accepts the head.

Adopted from the round-3 external audit's MT1 mutation suite (hardening note H1), which
returned 22/22 rejected against this tree; kept in the suite so it stays enforced.
"""

import ast
import copy
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src"

PHASE_ARGS = frozenset(
    {
        "continuation_functions",
        "flow_transitions",
        "flow_stochastic_transition_names",
        "next_state_names",
    }
)

# Files with a `stakeholders is not None` branch that is meant to be value/shape
# selection, NOT builder dispatch. The invariant pins that classification.
OTHER_PATHS = (
    "_lcm/regime_building/max_Q_over_a.py",
    "_lcm/regime_building/gated_edges.py",
    "_lcm/solution/v_topology.py",
    "_lcm/simulation/simulate.py",
    "_lcm/simulation/result_dataframe.py",
)


def normalized(path: Path) -> str:
    return re.sub(
        r"(?m)^(\s*)except ([A-Za-z_][\w.]*), ([A-Za-z_][\w.]*)\s*:",
        r"\1except (\2, \3):",
        path.read_text(),
    )


def parse(path: Path) -> ast.Module:
    return ast.parse(normalized(path), filename=str(path))


def dotted(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = dotted(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return None


def functions(tree: ast.Module) -> dict[str, ast.FunctionDef]:
    return {
        node.name: node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    }


def arg_names(node: ast.FunctionDef) -> set[str]:
    return {
        arg.arg
        for arg in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
    }


def call_named(node: ast.AST, name: str) -> list[ast.Call]:
    return [
        call
        for call in ast.walk(node)
        if isinstance(call, ast.Call) and dotted(call.func) == name
    ]


def keyword_map(call: ast.Call) -> dict[str, ast.AST]:
    return {kw.arg: kw.value for kw in call.keywords if kw.arg is not None}


def assignment_value(node: ast.AST, target_name: str) -> ast.AST:
    for child in ast.walk(node):
        if isinstance(child, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == target_name
            for target in child.targets
        ):
            return child.value
    raise AssertionError(f"missing assignment {target_name}")


def assert_name(node: ast.AST, expected: str) -> None:
    assert isinstance(node, ast.Name), ast.unparse(node)
    assert node.id == expected, ast.unparse(node)


def assert_role_default(node: ast.AST, *, default: str, override: str) -> None:
    """Pin the exact form `DEFAULT if OVERRIDE is None else OVERRIDE`.

    Anything else -- a truthiness test, a swapped branch -- is a different routing rule
    and must not pass silently.
    """
    assert isinstance(node, ast.IfExp), ast.unparse(node)
    assert isinstance(node.test, ast.Compare), ast.unparse(node)
    assert_name(node.test.left, override)
    assert len(node.test.ops) == 1
    assert isinstance(node.test.ops[0], ast.Is)
    assert len(node.test.comparators) == 1
    assert isinstance(node.test.comparators[0], ast.Constant)
    assert node.test.comparators[0].value is None
    assert_name(node.body, default)
    assert_name(node.orelse, override)


def stakeholder_test(node: ast.AST) -> bool:
    return "stakeholders" in ast.unparse(node)


def assert_every_twin_exposes_its_singleton_s_phase_roles(
    qdefs: dict[str, ast.FunctionDef],
) -> None:
    """Generic pair discovery -- this is the fail-closed part of the invariant.

    Every `get_*_collective` twin must expose every semantic phase-role / guard
    argument its singleton twin exposes. A NEW twin is caught here, without anyone
    having to remember the rule.
    """
    pairs: list[tuple[ast.FunctionDef, ast.FunctionDef]] = []
    for name, collective_def in qdefs.items():
        if not (name.startswith("get_") and name.endswith("_collective")):
            continue
        base_name = name.removesuffix("_collective")
        if base_name in qdefs:
            pairs.append((qdefs[base_name], collective_def))
    assert {c.name for _, c in pairs} == {
        "get_Q_and_F_collective",
        "get_Q_and_F_terminal_collective",
    }
    for base, twin in pairs:
        required = arg_names(base) & PHASE_ARGS
        assert required <= arg_names(twin), (base.name, twin.name, required)


def assert_the_collective_builders_route_each_role(
    qdefs: dict[str, ast.FunctionDef],
) -> None:
    """Each role must reach the consumer that role is FOR.

    Exposing the argument and then ignoring it is the same defect with a
    better-looking signature.
    """
    collective = qdefs["get_Q_and_F_collective"]
    assert_role_default(
        assignment_value(collective, "continuation_pool"),
        default="functions",
        override="continuation_functions",
    )
    assert_role_default(
        assignment_value(collective, "flow_pool"),
        default="transitions",
        override="flow_transitions",
    )
    assert_role_default(
        assignment_value(collective, "flow_stochastic_names"),
        default="stochastic_transition_names",
        override="flow_stochastic_transition_names",
    )

    det = call_named(collective, "_get_deterministic_transitions")
    assert len(det) == 1
    det_kw = keyword_map(det[0])
    assert_name(det_kw["transitions"], "flow_pool")
    assert_name(det_kw["stochastic_transition_names"], "flow_stochastic_names")

    u_calls = call_named(collective, "_get_U_and_F")
    assert len(u_calls) == 1  # one comprehension call, executed once per stakeholder
    u_kw = keyword_map(u_calls[0])
    assert_name(u_kw["functions"], "functions")
    assert_name(u_kw["stochastic_transition_names"], "flow_stochastic_names")
    assert_name(u_kw["next_state_names"], "next_state_names")

    next_calls = call_named(collective, "get_next_state_function_for_solution")
    weight_calls = call_named(collective, "get_next_stochastic_weights_function")
    assert len(next_calls) == len(weight_calls) == 1
    assert_name(keyword_map(next_calls[0])["functions"], "continuation_pool")
    assert_name(keyword_map(weight_calls[0])["functions"], "continuation_pool")

    terminal_collective = qdefs["get_Q_and_F_terminal_collective"]
    terminal_u = call_named(terminal_collective, "_get_U_and_F")
    assert len(terminal_u) == 1
    assert_name(keyword_map(terminal_u[0])["next_state_names"], "next_state_names")


def assert_every_dispatch_threads_every_role(
    processing_tree: ast.Module,
    pdefs: dict[str, ast.FunctionDef],
) -> None:
    """The call sites must pass what the signatures now expose."""
    per_period = pdefs["_build_Q_and_F_per_period"]
    collective_calls = call_named(per_period, "get_Q_and_F_collective")
    singleton_calls = call_named(per_period, "get_Q_and_F")
    assert len(collective_calls) == len(singleton_calls) == 1
    ckw = keyword_map(collective_calls[0])
    skw = keyword_map(singleton_calls[0])
    assert ckw.keys() >= PHASE_ARGS
    assert skw.keys() >= PHASE_ARGS
    # The continuation pool is age-specialized in BOTH dispatches.
    for keywords in (ckw, skw):
        assert "resolve_specialized_nodes(continuation_functions, age)" in ast.unparse(
            keywords["continuation_functions"]
        )

    # Both the solution and the simulation terminal path thread the guard vocabulary.
    tcalls = call_named(processing_tree, "get_Q_and_F_terminal_collective")
    scalls = call_named(processing_tree, "get_Q_and_F_terminal")
    assert len(tcalls) == len(scalls) == 2
    assert all("next_state_names" in keyword_map(call) for call in tcalls)
    assert all("next_state_names" in keyword_map(call) for call in scalls)


def assert_no_hidden_builder_in_a_stakeholder_branch(
    other_trees: dict[str, ast.Module],
) -> None:
    """A stakeholder-only branch outside the central dispatcher builds nothing.

    Secretly constructing another Q/F or phase-resolved transition DAG there is how
    a collective path escapes the twin-pair sweep entirely.
    """
    forbidden_prefixes = ("get_Q_and_F", "build_regime_transition")
    for rel, tree in other_trees.items():
        for branch in ast.walk(tree):
            if not isinstance(branch, ast.If | ast.IfExp):
                continue
            if not stakeholder_test(branch.test):
                continue
            names = {
                dotted(call.func)
                for call in ast.walk(branch)
                if isinstance(call, ast.Call)
            }
            hidden = {
                name
                for name in names
                if name is not None
                and (
                    name.startswith(forbidden_prefixes)
                    or name == "concatenate_functions"
                )
            }
            assert not hidden, (rel, hidden)


def check_class_invariant(
    q_tree: ast.Module,
    processing_tree: ast.Module,
    other_trees: dict[str, ast.Module],
) -> None:
    """Raise unless every collective twin exposes AND routes AND is passed its roles."""
    qdefs = functions(q_tree)
    pdefs = functions(processing_tree)

    assert_every_twin_exposes_its_singleton_s_phase_roles(qdefs)
    assert_the_collective_builders_route_each_role(qdefs)
    assert_every_dispatch_threads_every_role(processing_tree, pdefs)
    assert_no_hidden_builder_in_a_stakeholder_branch(other_trees)


@dataclass(frozen=True)
class Mutation:
    name: str
    apply: Callable[[ast.Module, ast.Module, dict[str, ast.Module]], None]


def remove_kwonly_arg(func_name: str, arg_name: str) -> Callable:
    def mutate(q_tree, _p_tree, _others):
        func = functions(q_tree)[func_name]
        index = next(
            i for i, arg in enumerate(func.args.kwonlyargs) if arg.arg == arg_name
        )
        del func.args.kwonlyargs[index]
        del func.args.kw_defaults[index]

    return mutate


def remove_call_keyword(callee: str, keyword: str, occurrence: int = 0) -> Callable:
    def mutate(_q_tree, p_tree, _others):
        call = call_named(p_tree, callee)[occurrence]
        before = len(call.keywords)
        call.keywords = [kw for kw in call.keywords if kw.arg != keyword]
        assert len(call.keywords) == before - 1

    return mutate


def replace_assignment(name: str, expression: str) -> Callable:
    def mutate(q_tree, _p_tree, _others):
        func = functions(q_tree)["get_Q_and_F_collective"]
        replacement = ast.parse(expression, mode="eval").body
        for node in ast.walk(func):
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == name for t in node.targets
            ):
                node.value = replacement
                return
        raise AssertionError(name)

    return mutate


def replace_call_keyword_in_collective(
    callee: str, keyword: str, expression: str
) -> Callable:
    def mutate(q_tree, _p_tree, _others):
        func = functions(q_tree)["get_Q_and_F_collective"]
        call = call_named(func, callee)[0]
        kw = next(kw for kw in call.keywords if kw.arg == keyword)
        kw.value = ast.parse(expression, mode="eval").body

    return mutate


def remove_terminal_forwarding(q_tree, _p_tree, _others):
    func = functions(q_tree)["get_Q_and_F_terminal_collective"]
    call = call_named(func, "_get_U_and_F")[0]
    call.keywords = [kw for kw in call.keywords if kw.arg != "next_state_names"]


def drop_age_specialization(_q_tree, p_tree, _others):
    func = functions(p_tree)["_build_Q_and_F_per_period"]
    call = call_named(func, "get_Q_and_F_collective")[0]
    kw = next(kw for kw in call.keywords if kw.arg == "continuation_functions")
    kw.value = ast.Name(id="continuation_functions", ctx=ast.Load())


def inject_hidden_builder(_q_tree, _p_tree, others):
    tree = others["_lcm/regime_building/max_Q_over_a.py"]
    branch = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If) and stakeholder_test(node.test)
    )
    branch.body.insert(
        0,
        ast.Expr(
            value=ast.Call(
                func=ast.Name(id="get_Q_and_F_collective", ctx=ast.Load()),
                args=[],
                keywords=[],
            )
        ),
    )


def mutation_catalog() -> list[Mutation]:
    mutations: list[Mutation] = []
    for arg in sorted(PHASE_ARGS):
        mutations.append(
            Mutation(
                f"drop_collective_signature_{arg}",
                remove_kwonly_arg("get_Q_and_F_collective", arg),
            )
        )
        mutations.append(
            Mutation(
                f"drop_collective_dispatch_{arg}",
                remove_call_keyword("get_Q_and_F_collective", arg),
            )
        )

    mutations.extend(
        [
            Mutation(
                "force_continuation_to_decision_pool",
                replace_assignment("continuation_pool", "functions"),
            ),
            Mutation(
                "force_flow_to_solve_transitions",
                replace_assignment("flow_pool", "transitions"),
            ),
            Mutation(
                "force_flow_stochastic_names_to_solve",
                replace_assignment(
                    "flow_stochastic_names", "stochastic_transition_names"
                ),
            ),
            Mutation(
                "bypass_flow_pool_at_deterministic_merge",
                replace_call_keyword_in_collective(
                    "_get_deterministic_transitions", "transitions", "transitions"
                ),
            ),
            Mutation(
                "bypass_continuation_pool_at_state_law",
                replace_call_keyword_in_collective(
                    "get_next_state_function_for_solution", "functions", "functions"
                ),
            ),
            Mutation(
                "bypass_continuation_pool_at_weights",
                replace_call_keyword_in_collective(
                    "get_next_stochastic_weights_function", "functions", "functions"
                ),
            ),
            Mutation(
                "drop_flow_stochastic_names_from_utility",
                replace_call_keyword_in_collective(
                    "_get_U_and_F",
                    "stochastic_transition_names",
                    "stochastic_transition_names",
                ),
            ),
            Mutation(
                "drop_next_state_guard_from_collective_utility",
                replace_call_keyword_in_collective(
                    "_get_U_and_F", "next_state_names", "frozenset()"
                ),
            ),
            Mutation(
                "drop_terminal_collective_signature_next_state_names",
                remove_kwonly_arg(
                    "get_Q_and_F_terminal_collective", "next_state_names"
                ),
            ),
            Mutation(
                "drop_terminal_collective_guard_forwarding",
                remove_terminal_forwarding,
            ),
            Mutation(
                "drop_solution_terminal_collective_call_keyword",
                remove_call_keyword(
                    "get_Q_and_F_terminal_collective", "next_state_names", occurrence=0
                ),
            ),
            Mutation(
                "drop_simulation_terminal_collective_call_keyword",
                remove_call_keyword(
                    "get_Q_and_F_terminal_collective", "next_state_names", occurrence=1
                ),
            ),
            Mutation("drop_collective_age_specialization", drop_age_specialization),
            Mutation(
                "inject_hidden_builder_in_stakeholder_branch", inject_hidden_builder
            ),
        ]
    )
    return mutations


@pytest.fixture(scope="module")
def production_trees():
    return (
        parse(SRC / "_lcm/regime_building/Q_and_F.py"),
        parse(SRC / "_lcm/regime_building/processing.py"),
        {rel: parse(SRC / rel) for rel in OTHER_PATHS},
    )


def test_the_production_source_satisfies_the_class_invariant(production_trees):
    """Guard the guard: the invariant must hold on the unmodified tree."""
    q_tree, p_tree, others = production_trees
    check_class_invariant(q_tree, p_tree, others)


@pytest.mark.parametrize(
    "mutation", mutation_catalog(), ids=lambda mutation: mutation.name
)
def test_every_member_of_the_counterexample_class_is_rejected(
    mutation, production_trees
):
    """Each mutation is one way a collective path can drop or misroute a phase role."""
    q_tree, p_tree, others = (copy.deepcopy(tree) for tree in production_trees)
    mutation.apply(q_tree, p_tree, others)

    with pytest.raises((AssertionError, KeyError, StopIteration)):
        check_class_invariant(q_tree, p_tree, others)
