"""Phase-resolved exit declarations and their ordinary source-domain callables."""

from collections.abc import Mapping
from dataclasses import replace
from types import MappingProxyType
from typing import cast

from lcm.exits import Consent, Exit
from lcm.phased import Phased
from lcm.typing import Phase, RegimeName, UserFunction

type ExitFunctionPath = tuple[str, ...]
type ExitFunctionDemand = Mapping[
    RegimeName, Mapping[Phase, frozenset[ExitFunctionPath]]
]


def rule_operands(*, stakeholders: tuple[str, ...] | None) -> frozenset[str]:
    """Name exactly the engine-owned channels of the public custom-rule grammar."""
    return frozenset(
        {
            "exit_payoffs",
            "availability",
            "alternative_ids",
            "pareto_weights",
            *(f"Q_{member}" for member in stakeholders or ("self",)),
        }
    )


def exits_for_phase(*, exits: Mapping[str, Exit], phase: Phase) -> Mapping[str, Exit]:
    """Resolve settlement phases without changing an exit's information set."""
    resolved = {}
    for name, declaration in exits.items():
        targets = {
            role: replace(
                target,
                projection=getattr(target.projection, phase)
                if isinstance(target.projection, Phased)
                else target.projection,
            )
            for role, target in declaration.targets.items()
        }
        rule = declaration.rule
        if isinstance(rule, Consent):
            counterparty = rule.counterparty
            projection = counterparty.projection
            rule = replace(
                rule,
                counterparty=replace(
                    counterparty,
                    projection=getattr(projection, phase)
                    if isinstance(projection, Phased)
                    else projection,
                ),
            )
        resolved[name] = replace(declaration, targets=targets, rule=rule)
    return MappingProxyType(resolved)


def exit_functions(
    *, exits: Mapping[str, Exit], phase: Phase
) -> Mapping[tuple[str, ...], UserFunction]:
    """Name every callable by its public declaration path for binding and demand."""
    functions = {}
    for name, declaration in exits_for_phase(exits=exits, phase=phase).items():
        prefix = ("exits", name)
        if declaration.available is not None:
            functions[*prefix, "available"] = declaration.available
        if callable(declaration.rule):
            functions[*prefix, "rule"] = cast("UserFunction", declaration.rule)
        for role, target in declaration.targets.items():
            target_path = (*prefix, "targets", role)
            if target.utility_adjustment is not None:
                functions[*target_path, "utility_adjustment"] = (
                    target.utility_adjustment
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
                    (*prefix, "counterparty", "projection", state): function
                    for state, function in cast(
                        "Mapping[str, UserFunction]",
                        declaration.rule.counterparty.projection,
                    ).items()
                }
            )
    return MappingProxyType(functions)
