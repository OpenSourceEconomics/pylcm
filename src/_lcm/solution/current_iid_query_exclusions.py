"""Reuse normalized model declarations to exclude non-Euler finite-bank readers."""

from collections.abc import Mapping

from _lcm.solution.contract import SolverBuildContext
from lcm.exceptions import UnsupportedOperationError
from lcm.exits import Consent
from lcm.typing import RegimeName


def _declared_policy_reads(
    *, context: SolverBuildContext, target: RegimeName
) -> set[str]:
    """Read original normalized same-period/exit declarations, not query guesses."""
    incoming = set()
    for source, regime in context.user_regimes.items():
        if any(ref.regime == target for ref in regime.same_period_refs.values()):
            incoming.add(f"same-period value read from {source}")
        if source == target and regime.exits:
            incoming.add("own exit action problem")
        for name, exit_ in regime.exits.items():
            if any(item.regime == target for item in exit_.targets.values()):
                incoming.add(f"exit policy read {source}.{name}")
            if isinstance(exit_.rule, Consent) and (
                exit_.rule.counterparty.regime == target
            ):
                incoming.add(f"exit counterparty read {source}.{name}")
    return incoming


def _declared_edge_reads(
    *, context: SolverBuildContext, target: RegimeName
) -> set[str]:
    """Read actual scheduled arrival and gate operands before later installation."""
    incoming = set()
    for source, law in context.laws.items():
        for scheduled, gate in law.gated_edges.items():
            if scheduled == target or target in gate.reference_regimes(
                phases=("solve",)
            ):
                incoming.add(f"gated value read {source}->{scheduled}")
        for scheduled, arrival in law.arrivals.items():
            if (
                scheduled == target
                or any(read.regime == target for read in arrival.reads.values())
                or any(
                    route.solve_fallback.regime == target
                    for route in arrival.routes.values()
                )
            ):
                incoming.add(f"arrival policy/value read {source}->{scheduled}")
    return incoming


def require_transition_only_projected_reads(
    *, target: RegimeName, contexts: Mapping[RegimeName, SolverBuildContext]
) -> None:
    """Authenticate the first finite class before authorizing even an empty bank.

    These are the same normalized declarations the later query installers use.
    This pass does not lower queries or filter them by realized availability,
    branch probability or discount; shadow and conditional readers count too.
    """
    context = contexts[target]
    if set(contexts) != set(context.user_regimes):
        raise TypeError("Exact projected closure requires every model source context.")
    incoming = _declared_policy_reads(context=context, target=target)
    incoming |= _declared_edge_reads(context=context, target=target)
    # Keep already prepared exclusions if called after query installation.
    for source, owner in contexts.items():
        if target in owner.same_period_ref_regimes:
            incoming.add(f"prepared same-period read from {source}")
        if target in (*owner.edge_target_regimes, *owner.edge_reference_regimes):
            incoming.add(f"prepared gated value read from {source}")
        if owner.arrival_queries:
            incoming.add(f"installed arrival query graph from {source}")
    if incoming:
        raise UnsupportedOperationError(
            f"Exact-only projected producer '{target}' cannot close "
            + "; ".join(sorted(incoming))
            + "."
        )
