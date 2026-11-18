"""Phase-resolved exit declarations and their ordinary source-domain callables."""

from collections.abc import Mapping
from dataclasses import replace
from types import MappingProxyType
from typing import cast

from _lcm.output_roles import OutputRole, in_role
from lcm.exits import Consent, Exit
from lcm.phased import Phased
from lcm.typing import (
    Phase,
    RegimeName,
    StateName,
    StateOrActionName,
    UserFunction,
)

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


def exit_without_pruned_states(
    *,
    declaration: Exit | None,
    pruned: Mapping[RegimeName, frozenset[StateOrActionName]],
) -> Exit | None:
    """Drop each projection onto a state that its destination prunes.

    A projection names every state its destination declares, model-level ones
    included. Where the destination reads no such state, demand prunes it, and its
    value has no axis to land on; the projection onto it is dropped, as a law of
    motion toward a target that prunes the law's state is.

    Args:
        declaration: The source's exit, if any.
        pruned: Per regime, the model-level states and actions demand pruned.

    Returns:
        The exit, unchanged where no destination prunes a projected state.

    """
    if declaration is None:
        return None
    rule = declaration.rule
    projections = [(t.regime, t.projection) for t in declaration.targets.values()]
    if isinstance(rule, Consent):
        projections.append((rule.counterparty.regime, rule.counterparty.projection))
    if not any(
        pruned.get(destination, frozenset())
        & (
            {*projection.solve, *projection.simulate}
            if isinstance(projection, Phased)
            else set(projection)
        )
        for destination, projection in projections
    ):
        return declaration

    def kept(
        *, value: Mapping[StateName, UserFunction] | Phased, destination: RegimeName
    ) -> Mapping[StateName, UserFunction] | Phased:
        if isinstance(value, Phased):
            return Phased(
                solve=kept(value=value.solve, destination=destination),
                simulate=kept(value=value.simulate, destination=destination),
            )
        dropped = pruned.get(destination, frozenset())
        return {state: f for state, f in value.items() if state not in dropped}

    if isinstance(rule, Consent):
        counterparty = rule.counterparty
        rule = replace(
            rule,
            counterparty=replace(
                counterparty,
                projection=kept(
                    value=counterparty.projection, destination=counterparty.regime
                ),
            ),
        )
    return replace(
        declaration,
        rule=rule,
        targets={
            role: replace(
                target,
                projection=kept(value=target.projection, destination=target.regime),
            )
            for role, target in declaration.targets.items()
        },
    )


def _for_phase[T](*, value: T | Phased, phase: Phase) -> T:
    return cast("T", getattr(value, phase)) if isinstance(value, Phased) else value
