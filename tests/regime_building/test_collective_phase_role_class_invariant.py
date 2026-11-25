"""Static class invariant: the Q/F builder may not drop a phase role.

The arguments carrying the solve/simulate phase split are exactly the ones a
builder can quietly fail to thread. A signature that DEFAULTS such an argument
makes its omission silent by construction, so repairing one call site does
nothing about the next builder written beside it.

This checks the class rather than a site: the source is parsed, and every Q/F
builder that prices a continuation is required to expose -- and correctly route
-- every phase-role argument. Adding a `get_Q_and_F_*` builder that reaches the
continuation without threading them FAILS THIS TEST.

One role splits the two phases today, `continuation_functions`, alongside the
complementary requirement that the flow be built from `functions` and NOT from the
continuation pool. `next_<state>` is reserved for a transition's OUTPUT, so no utility,
feasibility or value constraint may read one and the flow holds no transition node to
resolve. The invariant is generic over `PHASE_ARGS`, so it re-arms by itself if a role
is added.

The builder reaches the continuation through `_get_compute_CE`, so the pool travels
two hops and the invariant follows it over both: the builder is pinned to hand
`continuation_pool` to the aggregator, and `_build_target_continuation` is pinned to
resolve BOTH continuation roles -- the state law and the stochastic weights --
against the pool it was handed. A single-role bypass is constructible one frame
down, which is where the per-role guard sits.

The mutation catalogue below is the evidence that the checker has teeth: each
mutation is one member of the counterexample class (drop the argument from the
signature, drop it at the dispatch, pair a role with the wrong pool, skip age
specialization), and every one must be rejected.

Kept in the suite so the rule stays enforced, and pruned in step with the roles it
tracks.
"""

import ast
import copy
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pytest

from lcm.typing import FunctionName, ReferenceName

SRC = Path(__file__).resolve().parents[2] / "src"

PHASE_ARGS = frozenset({"continuation_functions"})


def normalized(path: Path) -> str:
    return re.sub(
        r"(?m)^(\s*)except ([A-Za-z_][\w.]*), ([A-Za-z_][\w.]*)\s*:",
        r"\1except (\2, \3):",
        # UTF-8 explicitly: the locale default (cp1252 on Windows) cannot decode the
        # superscript minus the sources' math notation uses.
        path.read_text(encoding="utf-8"),
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


def call_named(*, node: ast.AST, name: str) -> list[ast.Call]:
    return [
        call
        for call in ast.walk(node)
        if isinstance(call, ast.Call) and dotted(call.func) == name
    ]


def keyword_map(call: ast.Call) -> dict[str, ast.AST]:
    return {kw.arg: kw.value for kw in call.keywords if kw.arg is not None}


def _resolver_name(node: ast.AST) -> str | None:
    """Name of the `resolve_*` call inside `node`, or `None` if it resolves nothing.

    The argument is usually wrapped (`cast("...", resolve_x(pool, period))`), so walk
    rather than matching the top node. Returns the name so the invariant can require
    the SAME resolver elsewhere without hard-coding which one is current.
    """
    for child in ast.walk(node):
        if (
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Name)
            and child.func.id.startswith("resolve_")
        ):
            return child.func.id
    return None


def assignment_value(*, node: ast.AST, target_name: ReferenceName) -> ast.AST:
    for child in ast.walk(node):
        if isinstance(child, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == target_name
            for target in child.targets
        ):
            return child.value
    raise AssertionError(f"missing assignment {target_name}")


def assert_name(*, node: ast.AST, expected: str) -> None:
    assert isinstance(node, ast.Name), ast.unparse(node)
    assert node.id == expected, ast.unparse(node)


def assert_role_default(*, node: ast.AST, default: str, override: str) -> None:
    """Pin the exact form `DEFAULT if OVERRIDE is None else OVERRIDE`.

    Anything else -- a truthiness test, a swapped branch -- is a different routing rule
    and must not pass silently.
    """
    assert isinstance(node, ast.IfExp), ast.unparse(node)
    assert isinstance(node.test, ast.Compare), ast.unparse(node)
    assert_name(node=node.test.left, expected=override)
    assert len(node.test.ops) == 1
    assert isinstance(node.test.ops[0], ast.Is)
    assert len(node.test.comparators) == 1
    assert isinstance(node.test.comparators[0], ast.Constant)
    assert node.test.comparators[0].value is None
    assert_name(node=node.body, expected=default)
    assert_name(node=node.orelse, expected=override)


def assert_every_continuation_builder_exposes_the_phase_roles(
    qdefs: dict[str, ast.FunctionDef],
) -> None:
    """Every Q/F builder that prices a continuation exposes every phase role.

    This is the fail-closed part of the invariant: a NEW builder reaching
    `_get_compute_CE` is caught here, without anyone having to remember the rule;
    so is a role dropped from the one builder there is.
    """
    builders = {
        name: definition
        for name, definition in qdefs.items()
        if name.startswith("get_Q_and_F")
        and call_named(node=definition, name="_get_compute_CE")
    }
    assert set(builders) == {"get_Q_and_F_collective"}
    for name, definition in builders.items():
        assert arg_names(definition) >= PHASE_ARGS, (name, PHASE_ARGS)


def assert_the_collective_builders_route_each_role(
    qdefs: dict[str, ast.FunctionDef],
) -> None:
    """Each role must reach the consumer that role is FOR.

    Exposing the argument and then ignoring it is the same defect with a
    better-looking signature.
    """
    collective = qdefs["get_Q_and_F_collective"]
    assert_role_default(
        node=assignment_value(node=collective, target_name="continuation_pool"),
        default="functions",
        override="continuation_functions",
    )

    # The FLOW is the other half of the split: each member's within-period
    # utility and the shared feasibility must be built from `functions` -- the pool
    # of the phase being built -- never from the perceived continuation pool.
    u_calls = call_named(node=collective, name="_get_U_and_F")
    assert len(u_calls) == 1  # one call, building every member's felicity at once
    u_kw = keyword_map(u_calls[0])
    assert_name(node=u_kw["functions"], expected="functions")
    assert_name(node=u_kw["constraints"], expected="constraints")

    # The CONTINUATION prices the target's V under the perceived law, through the
    # one aggregator, `_get_compute_CE`. What the aggregator does with the pool is
    # a separate assertion, in
    # `assert_the_shared_continuation_builder_reads_its_own_pool` below, because
    # its two ROLES read the pool separately.
    ce_calls = call_named(node=collective, name="_get_compute_CE")
    assert len(ce_calls) == 1
    assert_name(
        node=keyword_map(ce_calls[0])["functions"], expected="continuation_pool"
    )


def assert_the_shared_continuation_builder_reads_its_own_pool(
    qdefs: dict[str, ast.FunctionDef],
) -> None:
    """Each continuation ROLE reads the pool the aggregator was handed.

    The state law and the stochastic weights are two roles, and the pool has to
    reach each of them. Handing the pool to `_get_compute_CE` and checking no
    further would accept a builder that resolves the state law against it and the
    weights against something else -- a SINGLE-role bypass, the harder half of the
    counterexample class.

    The weight pool may overlay the target lotteries' original probability laws
    when restoring their reduction layout. Every other law retains its phase pool.
    """
    builder = qdefs["_build_target_continuation"]
    next_calls = call_named(node=builder, name="get_next_state_function_for_solution")
    weight_calls = call_named(node=builder, name="get_next_stochastic_weights_function")
    assert len(next_calls) == len(weight_calls) == 1
    assert_name(node=keyword_map(next_calls[0])["functions"], expected="functions")
    assert_name(
        node=keyword_map(weight_calls[0])["functions"], expected="weight_functions"
    )
    expected_layouts = ast.parse(
        """{
            name: layout for name in lottery_variables
            if restore_original_layout and
            (layout := transition_plans[target_regime_name]
                .lotteries[name].original_layout)
            is not None
        }""",
        mode="eval",
    ).body
    assert ast.dump(assignment_value(node=builder, target_name="original_layouts")) == (
        ast.dump(expected_layouts)
    )
    expected_pool = ast.parse(
        """weight_functions = functions
if original_layouts:
    weight_functions = MappingProxyType(
        dict(functions) | {
            transition_plans[target_regime_name].lotteries[name].weight_name:
                layout.probabilities
            for name, layout in original_layouts.items()
        }
    )
"""
    )
    pool_statements = [
        statement
        for statement in builder.body
        if any(
            isinstance(node, ast.Name)
            and node.id == "weight_functions"
            and isinstance(node.ctx, ast.Store)
            for node in ast.walk(statement)
        )
    ]
    assert [ast.dump(node) for node in pool_statements] == [
        ast.dump(node) for node in expected_pool.body
    ]


def assert_every_dispatch_threads_every_role(
    pdefs: dict[str, ast.FunctionDef],
) -> None:
    """The call site must pass what the signature exposes."""
    per_period = pdefs["_build_Q_and_F_per_period"]
    calls = call_named(node=per_period, name="get_Q_and_F_collective")
    assert len(calls) == 1
    keywords = keyword_map(calls[0])
    assert keywords.keys() >= PHASE_ARGS
    # The continuation pool is node-resolved by the SAME resolver the dispatch uses
    # for its ordinary `functions` pool. Deriving the expected resolver from the
    # dispatch's own `functions` argument keeps the teeth (an UNRESOLVED pool, or
    # one resolved by a different rule than the flow pool, still fails) without
    # pinning a resolver name that is not part of the invariant.
    resolver = _resolver_name(keywords["functions"])
    assert resolver is not None, (
        "the dispatch no longer resolves `functions`: "
        + ast.unparse(keywords["functions"])
    )
    rendered = ast.unparse(keywords["continuation_functions"])
    assert f"{resolver}(mapping=continuation_functions" in rendered, (
        f"the dispatch does not resolve `continuation_functions` with "
        f"`{resolver}` — the continuation sub-DAG would see an unresolved marker: "
        f"{rendered}"
    )


def check_class_invariant(*, q_tree: ast.Module, processing_tree: ast.Module) -> None:
    """Raise unless the Q/F builder exposes AND routes AND is passed its roles."""
    qdefs = functions(q_tree)
    pdefs = functions(processing_tree)

    assert_every_continuation_builder_exposes_the_phase_roles(qdefs)
    assert_the_collective_builders_route_each_role(qdefs)
    assert_the_shared_continuation_builder_reads_its_own_pool(qdefs)
    assert_every_dispatch_threads_every_role(pdefs)


@dataclass(frozen=True)
class Mutation:
    name: str
    apply: Callable[..., None]


def remove_kwonly_arg(
    *, func_name: FunctionName, arg_name: ReferenceName
) -> Callable[..., None]:
    def mutate(*, q_tree: ast.Module, p_tree: ast.Module) -> None:
        _ = p_tree
        func = functions(q_tree)[func_name]
        index = next(
            i for i, arg in enumerate(func.args.kwonlyargs) if arg.arg == arg_name
        )
        del func.args.kwonlyargs[index]
        del func.args.kw_defaults[index]

    return mutate


def remove_call_keyword(
    *, callee: str, keyword: str, occurrence: int = 0
) -> Callable[..., None]:
    def mutate(*, q_tree: ast.Module, p_tree: ast.Module) -> None:
        _ = q_tree
        call = call_named(node=p_tree, name=callee)[occurrence]
        before = len(call.keywords)
        call.keywords = [kw for kw in call.keywords if kw.arg != keyword]
        assert len(call.keywords) == before - 1

    return mutate


def replace_assignment(
    *,
    name: str,
    expression: str,
    func_name: FunctionName = "get_Q_and_F_collective",
    occurrence: int = 0,
) -> Callable[..., None]:
    def mutate(*, q_tree: ast.Module, p_tree: ast.Module) -> None:
        _ = p_tree
        func = functions(q_tree)[func_name]
        replacement = ast.parse(expression, mode="eval").body
        matches = [
            node
            for node in ast.walk(func)
            if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)
        ]
        assert len(matches) > occurrence, name
        matches[occurrence].value = replacement

    return mutate


def replace_call_keyword_in(
    *, func_name: FunctionName, callee: str, keyword: str, expression: str
) -> Callable[..., None]:
    """Rebind one keyword of one call inside one function of `Q_and_F.py`.

    Takes the enclosing function by name because a role-level bypass lives in
    `_build_target_continuation`, one frame below the builder whose behaviour it
    changes.
    """

    def mutate(*, q_tree: ast.Module, p_tree: ast.Module) -> None:
        _ = p_tree
        func = functions(q_tree)[func_name]
        call = call_named(node=func, name=callee)[0]
        kw = next(kw for kw in call.keywords if kw.arg == keyword)
        kw.value = ast.parse(expression, mode="eval").body

    return mutate


def replace_call_keyword_in_collective(
    *, callee: str, keyword: str, expression: str
) -> Callable[..., None]:
    return replace_call_keyword_in(
        func_name="get_Q_and_F_collective",
        callee=callee,
        keyword=keyword,
        expression=expression,
    )


def drop_age_specialization(*, q_tree: ast.Module, p_tree: ast.Module) -> None:
    _ = q_tree
    func = functions(p_tree)["_build_Q_and_F_per_period"]
    call = call_named(node=func, name="get_Q_and_F_collective")[0]
    kw = next(kw for kw in call.keywords if kw.arg == "continuation_functions")
    kw.value = ast.Name(id="continuation_functions", ctx=ast.Load())


def mutation_catalog() -> list[Mutation]:
    mutations: list[Mutation] = []
    for arg in sorted(PHASE_ARGS):
        mutations.append(
            Mutation(
                f"drop_collective_signature_{arg}",
                remove_kwonly_arg(func_name="get_Q_and_F_collective", arg_name=arg),
            )
        )
        mutations.append(
            Mutation(
                f"drop_collective_dispatch_{arg}",
                remove_call_keyword(callee="get_Q_and_F_collective", keyword=arg),
            )
        )

    mutations.extend(
        [
            Mutation(
                "force_continuation_to_decision_pool",
                replace_assignment(name="continuation_pool", expression="functions"),
            ),
            Mutation(
                "price_the_flow_under_the_continuation_pool",
                replace_call_keyword_in_collective(
                    callee="_get_U_and_F",
                    keyword="functions",
                    expression="continuation_pool",
                ),
            ),
            Mutation(
                "price_feasibility_under_the_continuation_pool",
                replace_call_keyword_in_collective(
                    callee="_get_U_and_F",
                    keyword="constraints",
                    expression="continuation_pool",
                ),
            ),
            # Three bypasses, at the two levels the pool now travels through.
            # The aggregator-level one prices the WHOLE continuation under the
            # flow pool; the two role-level ones bypass a SINGLE role while the
            # other still reads the pool, which is the harder half of the class
            # and the half an aggregator-level check alone cannot see. Inside
            # `_build_target_continuation` the wrong pool is spelled `bundle` --
            # the target's transitions, the one other mapping in scope -- because
            # that builder is handed a single pool and has no flow pool to name.
            Mutation(
                "bypass_continuation_pool_at_the_aggregator",
                replace_call_keyword_in_collective(
                    callee="_get_compute_CE",
                    keyword="functions",
                    expression="functions",
                ),
            ),
            Mutation(
                "bypass_continuation_pool_at_state_law",
                replace_call_keyword_in(
                    func_name="_build_target_continuation",
                    callee="get_next_state_function_for_solution",
                    keyword="functions",
                    expression="bundle",
                ),
            ),
            Mutation(
                "bypass_continuation_pool_at_weights",
                replace_call_keyword_in(
                    func_name="_build_target_continuation",
                    callee="get_next_stochastic_weights_function",
                    keyword="functions",
                    expression="bundle",
                ),
            ),
            Mutation(
                "bypass_original_layout_weight_overlay",
                replace_call_keyword_in(
                    func_name="_build_target_continuation",
                    callee="get_next_stochastic_weights_function",
                    keyword="functions",
                    expression="functions",
                ),
            ),
            Mutation(
                "replace_weight_pool_base",
                replace_assignment(
                    func_name="_build_target_continuation",
                    name="weight_functions",
                    expression="bundle",
                ),
            ),
            Mutation(
                "replace_weight_pool_overlay",
                replace_assignment(
                    func_name="_build_target_continuation",
                    name="weight_functions",
                    occurrence=1,
                    expression="MappingProxyType(dict(bundle))",
                ),
            ),
            Mutation(
                "replace_original_layout_origin",
                replace_assignment(
                    func_name="_build_target_continuation",
                    name="original_layouts",
                    expression="bundle",
                ),
            ),
            Mutation("drop_collective_age_specialization", drop_age_specialization),
        ]
    )
    return mutations


@pytest.fixture(scope="module")
def production_trees() -> tuple[ast.Module, ast.Module]:
    return (
        parse(SRC / "_lcm/regime_building/Q_and_F.py"),
        parse(SRC / "_lcm/regime_building/processing.py"),
    )


def test_the_production_source_satisfies_the_class_invariant(
    production_trees: tuple[ast.Module, ast.Module],
) -> None:
    """Guard the guard: the invariant must hold on the unmodified tree."""
    q_tree, p_tree = production_trees
    check_class_invariant(q_tree=q_tree, processing_tree=p_tree)


@pytest.mark.parametrize(
    "mutation", mutation_catalog(), ids=lambda mutation: mutation.name
)
def test_every_member_of_the_counterexample_class_is_rejected(
    *,
    mutation: Mutation,
    production_trees: tuple[ast.Module, ast.Module],
) -> None:
    """Each mutation is one way the builder can drop or misroute a phase role."""
    q_tree, p_tree = (copy.deepcopy(tree) for tree in production_trees)
    mutation.apply(q_tree=q_tree, p_tree=p_tree)

    with pytest.raises((AssertionError, KeyError, StopIteration)):
        check_class_invariant(q_tree=q_tree, processing_tree=p_tree)
