"""Composition actions resolved per phase, and their callables in the source domain."""

from collections.abc import Mapping
from dataclasses import replace
from types import MappingProxyType
from typing import cast

from _lcm.output_roles import OutputRole, in_role
from lcm.composition_actions import (
    CompositionAction,
    CompositionRule,
    Consent,
    MutualConsent,
    PCOnActions,
)
from lcm.phased import Phased
from lcm.typing import (
    Phase,
    ReferenceName,
    RegimeName,
    StateName,
    StateOrActionName,
    UserFunction,
)

type CompositionActionFunctionPath = tuple[str, ...]
type CompositionActionFunctionDemand = Mapping[
    RegimeName, Mapping[Phase, frozenset[CompositionActionFunctionPath]]
]


def compares_at_optimum(rule: CompositionRule | None) -> bool:
    """Return whether a rule compares at the members' unconstrained optimum.

    Every divorce rule does but `PCOnActions`, which compares at every stay action.
    """
    return rule is not None and not isinstance(rule, PCOnActions | Consent)


def transfer_names(declaration: CompositionAction | None) -> frozenset[ReferenceName]:
    """Return the argument the transfer binds; it is never a parameter."""
    if declaration is None or not isinstance(declaration.rule, MutualConsent):
        return frozenset()
    return frozenset({declaration.rule.transfer.name})


def composition_action_for_phase(
    *, declaration: CompositionAction | None, phase: Phase
) -> CompositionAction | None:
    """Resolve the phases of the projections without changing the information set."""
    if declaration is None:
        return None
    targets = {
        role: replace(
            target, projection=_for_phase(value=target.projection, phase=phase)
        )
        for role, target in declaration.destinations.items()
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
    return replace(declaration, destinations=targets, rule=rule)


def composition_action_functions(
    *, declaration: CompositionAction | None, phase: Phase
) -> Mapping[CompositionActionFunctionPath, UserFunction]:
    """Name every callable by its public declaration path for binding and demand."""
    declaration = composition_action_for_phase(declaration=declaration, phase=phase)
    if declaration is None:
        return MappingProxyType({})
    functions = {}
    if declaration.available is not None:
        functions["composition_action", "available"] = in_role(
            func=declaration.available,
            role=OutputRole.BOOL,
            kind="Composition action availability",
        )
    for role, target in declaration.destinations.items():
        target_path = ("composition_action", "destinations", role)
        if target.utility_adjustment is not None:
            functions[*target_path, "utility_adjustment"] = in_role(
                func=target.utility_adjustment,
                role=OutputRole.FLOAT,
                kind=f"Composition action utility adjustment of '{role}', function",
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
                ("composition_action", "counterparty", "projection", state): function
                for state, function in cast(
                    "Mapping[str, UserFunction]",
                    declaration.rule.counterparty.projection,
                ).items()
            }
        )
    if isinstance(declaration.rule, MutualConsent):
        functions["composition_action", "transfer", "default"] = in_role(
            func=declaration.rule.transfer.default,
            role=OutputRole.FLOAT,
            kind="Transfer default, function",
        )
    return MappingProxyType(functions)


def composition_action_without_pruned_states(
    *,
    declaration: CompositionAction | None,
    pruned: Mapping[RegimeName, frozenset[StateOrActionName]],
) -> CompositionAction | None:
    """Drop each projection onto a state that its destination prunes.

    A projection names every state its destination declares, model-level ones
    included. Where the destination reads no such state, demand prunes it, and its
    value has no axis to land on; the projection onto it is dropped, as a law of
    motion toward a target that prunes the law's state is.

    Args:
        declaration: The source's composition action, if any.
        pruned: Per regime, the model-level states and actions demand pruned.

    Returns:
        The composition action, unchanged where no destination prunes a projected state.

    """
    if declaration is None:
        return None
    rule = declaration.rule
    projections = [(t.regime, t.projection) for t in declaration.destinations.values()]
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
        destinations={
            role: replace(
                target,
                projection=kept(value=target.projection, destination=target.regime),
            )
            for role, target in declaration.destinations.items()
        },
    )


def _for_phase[T](*, value: T | Phased, phase: Phase) -> T:
    return cast("T", getattr(value, phase)) if isinstance(value, Phased) else value
