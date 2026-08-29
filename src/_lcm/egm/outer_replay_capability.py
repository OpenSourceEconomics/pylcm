"""What a simulation replay may assume about one regime-period's outer margin.

Both NNBEGM outer searches publish a policy that a simulation reader replays,
and every replay asks the same questions of the declared outer post-decision
map: what coefficient must be divided out to recover the outer action, which
stocks the recovered action may land on, whether the functions the recovery
needs can be bound at a realized state, and whether the published rows can be
addressed there. Asking them inside a reader lets the two searches disagree,
and lets simulation re-decide what the solve already settled.

`resolve_outer_replay_capability` answers them once per regime-period, from the
declared structure alone, before either search runs. The answer rides on the
published policy, so every replay path reads one verdict instead of deriving
its own. A verdict the adaptive nested reader cannot serve stops publication
(`fail_if_continuous_outer_replay_is_unsupported`); the finite reader addresses
a wider row layout and recovers the action by the map's own coefficient, so it
serves verdicts the nested reader refuses. That is one answer set with two
support envelopes, not two answers.
"""

import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from fractions import Fraction

from _lcm.egm.outer_inversion import DeclaredOuterInverse
from lcm.exceptions import RegimeInitializationError
from lcm.typing import ActionName, FunctionName, RegimeName, StateName

__all__ = [
    "OuterReplayCapability",
    "fail_if_continuous_outer_replay_is_unsupported",
    "resolve_outer_replay_capability",
]


@dataclass(frozen=True)
class OuterReplayCapability:
    """One regime-period's settled answer to what a replay may assume.

    Every verdict field is empty when replay is unobstructed, and names the
    offending declarations otherwise, so a refusal can report what it found
    rather than that something was wrong.
    """

    inverse: DeclaredOuterInverse
    """The certified coefficient of the declared map and the stock domain."""

    undeclared_functions: tuple[FunctionName, ...]
    """Replay functions the regime does not declare at all."""

    unbindable_functions: tuple[tuple[FunctionName, tuple[str, ...]], ...]
    """Each declared replay function paired with the arguments a replay cannot
    supply from a realized state, a parameter, `period`, or `age`."""

    unavailable_keeper_states: tuple[StateName, ...]
    """The outer state, when keeping means holding it and replay cannot read
    it; empty whenever a no-adjustment candidate is declared instead."""

    unaddressable_passive_states: tuple[StateName, ...]
    """Passive continuous row axes beyond the single one a nested read
    brackets; empty when there is at most one."""

    unaddressable_discrete_actions: tuple[ActionName, ...]
    """Discrete-action row axes, which a nested read has no address for."""

    @property
    def continuous_replay_is_supported(self) -> bool:
        """Whether the adaptive nested reader can replay this declaration."""
        return not (
            self.undeclared_functions
            or self.unbindable_functions
            or self.unavailable_keeper_states
            or self.unaddressable_passive_states
            or self.unaddressable_discrete_actions
            or self.inverse.coefficient != Fraction(1)
        )


def resolve_outer_replay_capability(
    *,
    inverse: DeclaredOuterInverse,
    functions: Mapping[FunctionName, Callable[..., object]],
    bindable_names: frozenset[str],
    outer_post_decision_name: FunctionName,
    outer_action_name: ActionName,
    outer_no_adjustment_name: FunctionName | None,
    outer_state_name: StateName,
    state_names: frozenset[StateName],
    row_passive_state_names: tuple[StateName, ...],
    row_discrete_action_names: tuple[ActionName, ...],
) -> OuterReplayCapability:
    """Answer, from the declared structure, what a replay of this period may assume.

    Reads names and signatures only, never values, so the answer is the same
    for every state a simulation later arrives at and for either outer search.

    Args:
        inverse: The period's certified inverse of the declared outer map.
        functions: The regime's declared functions, by name.
        bindable_names: Everything a replay can supply at a realized state --
            the simulated states, the regime's flat parameter names (from the
            params template, so the answer does not depend on one call's
            params), and `period`/`age`.
        outer_post_decision_name: The map whose inverse recovers the action.
        outer_action_name: The action a replay binds into that map itself.
        outer_no_adjustment_name: The keeper's declared candidate, or `None`
            when keeping means holding the outer state unchanged.
        outer_state_name: The stock the keeper holds.
        state_names: The states a replay reads at each subject.
        row_passive_state_names: Passive continuous row axes of the published
            branch rows, in axis order.
        row_discrete_action_names: Discrete-action row axes of those rows.

    Returns:
        The `OuterReplayCapability` for this regime-period.
    """
    undeclared: list[FunctionName] = []
    unbindable: list[tuple[FunctionName, tuple[str, ...]]] = []
    for name, supplied in (
        (outer_post_decision_name, frozenset({outer_action_name})),
        *(
            ()
            if outer_no_adjustment_name is None
            else ((outer_no_adjustment_name, frozenset()),)
        ),
    ):
        func = functions.get(name)
        if func is None:
            undeclared.append(name)
            continue
        unbound = tuple(
            arg
            for arg in inspect.signature(func).parameters
            if arg not in bindable_names and arg not in supplied
        )
        if unbound:
            unbindable.append((name, unbound))
    keeper_holds_the_outer_state = outer_no_adjustment_name is None
    return OuterReplayCapability(
        inverse=inverse,
        undeclared_functions=tuple(undeclared),
        unbindable_functions=tuple(unbindable),
        unavailable_keeper_states=(
            (outer_state_name,)
            if keeper_holds_the_outer_state and outer_state_name not in state_names
            else ()
        ),
        unaddressable_passive_states=(
            row_passive_state_names if len(row_passive_state_names) > 1 else ()
        ),
        unaddressable_discrete_actions=row_discrete_action_names,
    )


def fail_if_continuous_outer_replay_is_unsupported(
    *,
    capability: OuterReplayCapability,
    regime_name: RegimeName,
    outer_action_name: ActionName,
) -> None:
    """Refuse to publish a continuous-outer payload replay cannot answer.

    The adaptive mesh's reader rebuilds the keeper/adjuster comparison at each
    subject's own state: it recovers the outer action by subtracting the
    declared map's offset, evaluates that map and the keeper's candidate there,
    and addresses each branch row by the subject's discrete states plus one
    bracketed passive continuous state. A declaration outside that envelope
    leaves the reader publishing the generic action-grid winner, which ranks a
    different candidate set and reports no error for doing so -- so it stops
    here, before any policy object exists.

    Raises:
        RegimeInitializationError: If any verdict blocks the nested reader.
    """
    if capability.continuous_replay_is_supported:
        return
    findings = []
    if capability.inverse.coefficient != Fraction(1):
        findings.append(
            f"the declared map moves {capability.inverse.coefficient} units of "
            f"outer stock per unit of {outer_action_name!r}, and replay "
            "recovers the action by subtracting that map's offset, which "
            "inverts it only at one unit per unit"
        )
    if capability.undeclared_functions:
        findings.append(
            f"replay needs {list(capability.undeclared_functions)}, which this "
            "regime does not declare"
        )
    for name, unbound in capability.unbindable_functions:
        findings.append(
            f"{name!r} reads {list(unbound)}, which replay cannot supply from a "
            "simulated state, a parameter, `period`, or `age`"
        )
    if capability.unavailable_keeper_states:
        findings.append(
            f"keeping means holding {list(capability.unavailable_keeper_states)} "
            "unchanged, which replay does not read at each subject"
        )
    if capability.unaddressable_passive_states:
        findings.append(
            "the published branch rows carry passive continuous state axes "
            f"{list(capability.unaddressable_passive_states)}, and a nested read "
            "brackets exactly one"
        )
    if capability.unaddressable_discrete_actions:
        findings.append(
            "the published branch rows carry discrete-action axes "
            f"{list(capability.unaddressable_discrete_actions)}, which a nested "
            "read has no address for"
        )
    msg = (
        f"Continuous-outer replay of regime {regime_name!r} cannot reproduce the "
        "declared problem: "
        + "; ".join(findings)
        + ". Solve the regime with `NNBEGM(outer_search=FiniteOuterGrid(...))`, "
        "whose replay reads the recorded candidate targets, recovers the action "
        "by the map's own coefficient, and addresses the full row layout; or "
        "declare the outer margin inside the envelope above."
    )
    raise RegimeInitializationError(msg)
