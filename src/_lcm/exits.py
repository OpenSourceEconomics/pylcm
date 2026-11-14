"""Phase-resolved exit declarations and their ordinary source-domain callables."""

from collections.abc import Mapping
from dataclasses import replace
from types import MappingProxyType
from typing import cast

from _lcm.output_roles import OutputRole, in_role
from lcm.exits import Consent, Exit
from lcm.phased import Phased
from lcm.typing import Phase, RegimeName, UserFunction

type ExitFunctionPath = tuple[str, ...]
type ExitFunctionDemand = Mapping[
    RegimeName, Mapping[Phase, frozenset[ExitFunctionPath]]
]


def exit_for_phase(*, declaration: Exit | None, phase: Phase) -> Exit | None:
    """Resolve settlement phases without changing the exit's information set."""
    if declaration is None:
        return None
    targets = {
        role: replace(
            target, projection=_for_phase(value=target.projection, phase=phase)
        )
        for role, target in declaration.targets.items()
    }
    rule = declaration.rule
    if isinstance(rule, Consent):
        counterparty = rule.counterparty
        rule = replace(
            rule,
            counterparty=replace(
                counterparty,
                projection=_for_phase(value=counterparty.projection, phase=phase),
            ),
        )
    return replace(declaration, targets=targets, rule=rule)


def exit_functions(
    *, declaration: Exit | None, phase: Phase
) -> Mapping[ExitFunctionPath, UserFunction]:
    """Name every callable by its public declaration path for binding and demand."""
    declaration = exit_for_phase(declaration=declaration, phase=phase)
    if declaration is None:
        return MappingProxyType({})
    functions = {}
    if declaration.available is not None:
        functions["exit", "available"] = in_role(
            func=declaration.available, role=OutputRole.BOOL, kind="Exit availability"
        )
    for role, target in declaration.targets.items():
        target_path = ("exit", "targets", role)
        if target.utility_adjustment is not None:
            functions[*target_path, "utility_adjustment"] = in_role(
                func=target.utility_adjustment,
                role=OutputRole.FLOAT,
                kind=f"Exit utility adjustment of '{role}', function",
            )
        functions.update(
            {
                (*target_path, "projection", state): function
                for state, function in cast(
                    "Mapping[str, UserFunction]", target.projection
                ).items()
            }
        )
    if isinstance(declaration.rule, Consent):
        functions.update(
            {
                ("exit", "counterparty", "projection", state): function
                for state, function in cast(
                    "Mapping[str, UserFunction]",
                    declaration.rule.counterparty.projection,
                ).items()
            }
        )
    return MappingProxyType(functions)


def _for_phase[T](*, value: T | Phased, phase: Phase) -> T:
    return cast("T", getattr(value, phase)) if isinstance(value, Phased) else value
