"""Construct exact native composition body, surface and declaration controls."""

import ast
import copy
from pathlib import Path
from types import ModuleType

try:
    from child_native_selection import select_refine_envelope_exact
except ModuleNotFoundError:
    from tests.candidate_certificate.child_native_selection import (
        select_refine_envelope_exact,
    )

try:
    from generate_sources import SourcePath
except ModuleNotFoundError:
    from tests.candidate_certificate.generate_sources import SourcePath

BODY_OWNERS: dict[SourcePath, tuple[str, ...]] = {
    "src/_lcm/egm/upper_envelope/__init__.py": ("_exact_backend",),
    "src/_lcm/egm/upper_envelope/segment_envelope.py": ("refine_envelope_exact",),
}
OVERLOAD_MUTATIONS = (
    "missing_stub",
    "extra_stub",
    "reordered_stubs",
    "changed_stub",
    "runtime_decorator",
    "extra_function_binding",
    "overload_alias",
    "overload_rebinding",
    "overload_import_removed",
)


def primary_diagnostic(*, name: str) -> str:
    """Name the exact new body, surface or overload owner of one control."""
    if name.startswith("child_native:body:"):
        qualname = name.split("::", maxsplit=1)[1]
        return f"child native dependency: exact callable corridor {qualname!r} changed"
    if name.startswith("child_native:surface:"):
        source = name.removeprefix("child_native:surface:")
        return f"child native dependency: {source} declaration surface changed"
    if name.startswith("child_native:overload:"):
        return (
            "child native dependency: refine_envelope_exact "
            "overload declarations changed"
        )
    raise ValueError(f"Unknown native composition control {name!r}")


def expected_errors(*, source: SourcePath, diagnostic: str) -> tuple[str, ...]:
    """Require the precise declaration refusal beside its changed surface."""
    errors = [diagnostic]
    if diagnostic == (
        "child native dependency: refine_envelope_exact overload declarations changed"
    ):
        errors.append(f"child native dependency: {source} declaration surface changed")
    return tuple(sorted(errors))


def _alter_overload(*, tree: ast.Module, mutation: str, checker: ModuleType) -> None:
    """Change one explicit ordered stub, name binding or runtime decoration."""
    runtime = select_refine_envelope_exact(tree=tree, checker=checker)
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "refine_envelope_exact"
    ]
    if mutation == "missing_stub":
        tree.body.remove(functions[0])
    elif mutation == "extra_stub":
        tree.body.insert(tree.body.index(functions[0]), copy.deepcopy(functions[0]))
    elif mutation == "reordered_stubs":
        first, second = tree.body.index(functions[0]), tree.body.index(functions[1])
        tree.body[first], tree.body[second] = tree.body[second], tree.body[first]
    elif mutation == "changed_stub":
        functions[0].returns = ast.Name(id="changed_annotation", ctx=ast.Load())
    elif mutation == "runtime_decorator":
        runtime.decorator_list.append(ast.Name(id="property", ctx=ast.Load()))
    elif mutation == "extra_function_binding":
        tree.body.extend(ast.parse("refine_envelope_exact = None").body)
    elif mutation == "overload_rebinding":
        tree.body.extend(ast.parse("overload = None").body)
    else:
        _alter_overload_import(tree=tree, mutation=mutation)


def _alter_overload_import(*, tree: ast.Module, mutation: str) -> None:
    """Alter only the genuine unique typing declaration used by every stub."""
    imports = [
        node
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module == "typing"
    ]
    aliases = [
        (node, alias)
        for node in imports
        for alias in node.names
        if alias.name == "overload"
    ]
    if len(aliases) != 1:
        raise ValueError("The exact typing overload declaration is not unique")
    node, alias = aliases[0]
    if mutation == "overload_alias":
        alias.asname = "changed_overload"
    elif mutation == "overload_import_removed":
        node.names.remove(alias)
        if not node.names:
            tree.body.remove(node)
    else:
        raise ValueError(f"Unknown native overload mutation {mutation!r}")


def child_native_mutation_specs(
    *, repo_root: Path, checker: ModuleType
) -> dict[str, dict[str, str]]:
    """Construct exactly two bodies, two surfaces and nine declaration attacks."""
    specifications = {}
    for source, names in BODY_OWNERS.items():
        original = (repo_root / source).read_text(encoding="utf-8")
        for qualname in names:
            tree = ast.parse(original, filename=source)
            function = (
                select_refine_envelope_exact(tree=tree, checker=checker)
                if qualname == "refine_envelope_exact"
                else checker._definition(tree=tree, name=qualname)
            )
            before = checker._callable_ast_sha256(function)
            function.body = ast.parse("return None").body
            if checker._callable_ast_sha256(function) == before:
                raise ValueError(f"Native body mutation did not alter {qualname!r}")
            name = f"child_native:body:{source}::{qualname}"
            specifications[name] = {"path": source, "source": ast.unparse(tree) + "\n"}
        tree = ast.parse(original, filename=source)
        before = checker._transport_module_surface(tree)
        tree.body.extend(ast.parse("import math as _child_native_surface_probe").body)
        if checker._transport_module_surface(tree) == before:
            raise ValueError(f"Native surface mutation did not alter {source!r}")
        specifications[f"child_native:surface:{source}"] = {
            "path": source,
            "source": ast.unparse(tree) + "\n",
        }
    source = "src/_lcm/egm/upper_envelope/segment_envelope.py"
    original = (repo_root / source).read_text(encoding="utf-8")
    for mutation in OVERLOAD_MUTATIONS:
        tree = ast.parse(original, filename=source)
        _alter_overload(tree=tree, mutation=mutation, checker=checker)
        try:
            select_refine_envelope_exact(tree=tree, checker=checker)
        except ValueError:
            pass
        else:
            raise ValueError(
                f"Native overload mutation was not constructed: {mutation}"
            )
        specifications[f"child_native:overload:{mutation}"] = {
            "path": source,
            "source": ast.unparse(tree) + "\n",
        }
    if len(specifications) != 13:
        raise ValueError("The declared native composition population changed")
    return specifications
