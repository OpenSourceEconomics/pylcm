"""The engine reads a regime's members, never how many there are.

Every regime has members, a single the one member `"self"`, and the engine runs
one path for all of them. The only condition on how many members a regime has
is the solver capability `supports_several_members`. The other allowed
conditions validate a declaration against a rule that is itself about a member
count. The scan covers the packages `lcm` and `_lcm`.
"""

import ast
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"

type PythonSource = str

# (module, enclosing function) of every allowed condition, with its reason.
_ALLOWED: dict[tuple[str, str], str] = {
    (
        "_lcm/user_regime_validation.py",
        "_fail_if_solver_cannot_solve_the_members",
    ): "the solver capability `supports_several_members`",
    (
        "_lcm/user_regime_validation.py",
        "_validate_renegotiation",
    ): "renegotiation is a rule between two members",
    (
        "_lcm/user_regime_validation.py",
        "_validate_settlement",
    ): "a settlement is a rule between two members",
    (
        "_lcm/exit_queries.py",
        "_validate_consent_context",
    ): "consent compares the raw channels of two singles",
    (
        "lcm/composite_state.py",
        "__post_init__",
    ): "a composite state's member states, not a regime's members",
}


def test_member_count_conditions_are_only_the_allowed_ones() -> None:
    """Every condition on a member count sits in an allowed function."""
    found = {
        (path.relative_to(_SRC).as_posix(), function)
        for package in ("lcm", "_lcm")
        for path in sorted((_SRC / package).rglob("*.py"))
        for function, _line in member_count_conditions(
            source=path.read_text(encoding="utf-8")
        )
    }

    assert found == set(_ALLOWED)


def test_the_scan_finds_each_kind_of_member_count_condition() -> None:
    """A length, a member-count variable, the members of a single, `member_axis`."""
    source = """
def by_length(regime):
    return len(regime.members) > 1


def by_count(n_members):
    if n_members is None:
        return 0
    return n_members


def by_single(context):
    return context.members == ("self",)


def by_axis(regime):
    return regime.member_axis


def by_name(regime):
    return "self" in regime.members
"""

    assert [function for function, _line in member_count_conditions(source=source)] == [
        "by_length",
        "by_count",
        "by_single",
        "by_axis",
    ]


def member_count_conditions(*, source: PythonSource) -> list[tuple[str, int]]:
    """Return the enclosing function and line of each member-count condition.

    A condition compares the length of a members tuple or a variable named
    `n_members` with a constant, compares a members tuple with the members of
    a single, or reads a `member_axis`. Comparing a count with another count,
    as a shape check does, is no condition on the count.
    """
    conditions = {
        (function.name, node.lineno)
        for function in ast.walk(ast.parse(source))
        if isinstance(function, ast.FunctionDef | ast.AsyncFunctionDef)
        for node in ast.walk(function)
        if isinstance(node, ast.expr) and _is_member_count_condition(node)
    }
    return sorted(conditions, key=lambda condition: condition[1])


def _is_member_count_condition(node: ast.expr) -> bool:
    if isinstance(node, ast.Attribute) and node.attr == "member_axis":
        return True
    if not isinstance(node, ast.Compare):
        return False
    operands = [node.left, *node.comparators]
    against_a_constant = any(isinstance(operand, ast.Constant) for operand in operands)
    return any(
        (
            against_a_constant
            and (_is_length_of_members(operand) or _is_named_n_members(operand))
        )
        or _is_the_members_of_a_single(operand)
        for operand in operands
    )


def _is_length_of_members(node: ast.expr) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "len"
        and len(node.args) == 1
        and "members" in ast.unparse(node.args[0])
    )


def _is_named_n_members(node: ast.expr) -> bool:
    return (isinstance(node, ast.Name) and node.id == "n_members") or (
        isinstance(node, ast.Attribute) and node.attr == "n_members"
    )


def _is_the_members_of_a_single(node: ast.expr) -> bool:
    return (
        isinstance(node, ast.Tuple)
        and len(node.elts) == 1
        and isinstance(node.elts[0], ast.Constant)
        and node.elts[0].value == "self"
    )
