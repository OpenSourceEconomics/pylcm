"""Send a household with no feasible action to its gate's dissolution route.

A gated edge decides at the end of period `t` whether a household enters its
target regime at `t + 1`. When the gate reads the target's dissolution flag
`D_target`, the question it asks is whether the target has a feasible action at
the state the household lands on. The solve answers that only at the target's
grid nodes, and an off-grid landing can differ from both neighbouring nodes in
either direction. Simulation therefore leaves `D_target` open at the gate
(`_lcm.regime_building.gated_edges._SimulateGateEvaluator`) and answers it where
the household actually is: the target's own decision at `t + 1` evaluates every
action at the realised state, exactly as it does when choosing one.

A household that finds no feasible action there takes, in that same period, the
dissolution route of the gate it came through: the fallback regime of its own
leg, at that leg's projection of its realised state, carrying the fallback's
role. That is the closed branch the gate would have taken with the realised
flag, so routing on `D_target = False` followed by this step reproduces the gate
evaluated at the realised `D` for every predicate. It is also how a household
that passed a gate not reading `D_target` (a consent gate) leaves a regime that
has no admissible action for it.

Only the gate a household actually passed through its open branch hands it a
route, and only once. A household that landed in the regime through a closed
branch has already taken its gate's route; one that reached it through no gated
edge (an initial condition, or an ungated transition) never had one; and one
that already took a route this period has used it. Regime identity cannot tell
these apart (a closed branch of one gate can land in the target of another gate
from the same source), so the router's own closed-branch masks decide, and each
row carries the id of the source whose gate it passed open in `entered_through`,
or `NO_GATED_ENTRY`. The caller raises `NoFeasibleActionError` for an empty row
without one.
"""

import dataclasses
import logging
from collections.abc import Mapping
from types import MappingProxyType

import jax
import jax.numpy as jnp
import numpy as np

from _lcm.engine import Regime
from _lcm.regime_building.gated_edges import bind_edge_period_context
from _lcm.simulation.gated_routing import (
    _call_vmapped_with_accepted_kwargs,
    _per_row_leg_outcomes,
    bind_provenance_params,
)
from _lcm.simulation.transitions import _advance_states_for_subjects
from _lcm.typing import FlatParams, RegimeName, RegimeNamesToIds, StatesPerRegime
from lcm.typing import Bool1D, Int1D, ScalarFloat, ScalarInt

# The `entered_through` value of a row that holds no unused open-branch entry.
NO_GATED_ENTRY = -1


def empty_gated_entries(*, regime_ids: Int1D) -> Int1D:
    """Build a period's entry carrier, in which no row holds an open-branch entry."""
    return jnp.full_like(regime_ids, NO_GATED_ENTRY)


@dataclasses.dataclass(frozen=True, kw_only=True)
class DissolutionFallback:
    """One household sent to its dissolution route for want of a feasible action."""

    subject_id: int
    """The subject's row in the simulated population."""
    period: int
    """The period in which the household found no feasible action."""
    regime: RegimeName
    """The regime it had entered."""
    state: MappingProxyType[str, float | str]
    """Its full realised state in that regime, in declared coordinates."""
    source: RegimeName
    """The regime whose gate the household passed open, which supplies the route."""
    fallback: RegimeName
    """The fallback regime of its own leg, where it acts this period instead."""


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class RerouteOutcome:
    """The carriers after rerouting, and which rows moved.

    A pytree of its array fields, so a held outcome is measured buffer by buffer.
    """

    states: StatesPerRegime
    subject_regime_ids: Int1D
    own_stakeholder: Int1D
    new_subject_regime_ids: Int1D
    new_own_stakeholder: Int1D
    entered_through: Int1D
    """Open-branch entries after rerouting: every rerouted row's is used up."""
    rerouted: Bool1D
    """Rows sent to a fallback regime this period."""
    fallback_ids: Int1D
    """Each rerouted row's fallback regime id; meaningless elsewhere."""


def reroute_rows_without_feasible_action(
    *,
    regime_name: RegimeName,
    regimes: Mapping[RegimeName, Regime],
    empty: Bool1D,
    entered_through: Int1D,
    source_roles: Int1D,
    states: StatesPerRegime,
    subject_regime_ids: Int1D,
    own_stakeholder: Int1D,
    new_subject_regime_ids: Int1D,
    new_own_stakeholder: Int1D,
    regime_names_to_ids: RegimeNamesToIds,
    flat_params: FlatParams,
    period: int,
    age: ScalarInt | ScalarFloat,
) -> RerouteOutcome:
    """Move every empty row that passed a gate into this regime open to its fallback.

    Args:
        regime_name: The regime whose decision found no feasible action.
        regimes: Every regime of the model.
        empty: Rows of this regime with no feasible action this period.
        entered_through: Each row's source regime id when it entered this period
            through the open branch of that source's gate and has not used the
            route yet, else `NO_GATED_ENTRY`.
        source_roles: Each row's role in the previous period, which selects its
            leg of the gate.
        states: The current-period state carrier.
        subject_regime_ids: Current-period regime membership.
        own_stakeholder: Current-period roles.
        new_subject_regime_ids: Next-period regime membership as accumulated.
        new_own_stakeholder: Next-period roles as accumulated.
        regime_names_to_ids: Mapping from regime names to ids.
        flat_params: Model parameters for all regimes.
        period: The current period, which is the edge's fold period.
        age: The age of the current period.

    Returns:
        The updated carriers and the rows that moved.
    """
    # Every leg projects from the same original target state. A self-target
    # leg may overwrite the carrier below; later legs must not compose their
    # projections with that write.
    original_target_states = dict(states[regime_name])
    rerouted = jnp.zeros_like(empty)
    fallback_ids = jnp.full_like(subject_regime_ids, -1)
    for source_name, source in regimes.items():
        edge = source.gated_edges.get(regime_name)
        if edge is None:
            continue
        rows = empty & (entered_through == regime_names_to_ids[source_name])
        own_fallback_id, _open_role, closed_role = _per_row_leg_outcomes(
            legs=edge.legs,
            own_stakeholder=source_roles,
            regime_names_to_ids=regime_names_to_ids,
            role_ids=source.stakeholder_names_to_ids,
        )
        for leg in edge.legs:
            projector = leg.fallback_state_projector
            projected = _call_vmapped_with_accepted_kwargs(
                func=projector,
                batched_kwargs=original_target_states,
                static_kwargs={
                    **bind_provenance_params(
                        provenance=projector.arg_provenance,  # ty: ignore[unresolved-attribute]
                        flat_params=flat_params,
                        source_name=source_name,
                        target_name=regime_name,
                    ),
                    **bind_edge_period_context(
                        func=projector, fold_period=period, fold_age=age
                    ),
                },
                axis_size=int(empty.shape[0]),
            )
            states = _advance_states_for_subjects(
                states_per_regime=states,
                next_states_per_regime=MappingProxyType(
                    {
                        leg.realized_fallback.regime: MappingProxyType(
                            dict(projected)  # ty: ignore[no-matching-overload]
                        )
                    }
                ),
                subject_indices=rows,
            )
        subject_regime_ids = jnp.where(rows, own_fallback_id, subject_regime_ids)
        new_subject_regime_ids = jnp.where(
            rows, own_fallback_id, new_subject_regime_ids
        )
        own_stakeholder = jnp.where(rows, closed_role, own_stakeholder)
        new_own_stakeholder = jnp.where(rows, closed_role, new_own_stakeholder)
        fallback_ids = jnp.where(rows, own_fallback_id, fallback_ids)
        rerouted = rerouted | rows
    return RerouteOutcome(
        states=states,
        subject_regime_ids=subject_regime_ids,
        own_stakeholder=own_stakeholder,
        new_subject_regime_ids=new_subject_regime_ids,
        new_own_stakeholder=new_own_stakeholder,
        entered_through=jnp.where(rerouted, NO_GATED_ENTRY, entered_through),
        rerouted=rerouted,
        fallback_ids=fallback_ids,
    )


def record_open_entries(
    *,
    source_id: int,
    target_ids: tuple[int, ...],
    closed_masks: tuple[Bool1D, ...],
    routed_ids: Int1D,
    subjects_in_regime: Bool1D,
    new_entered_through: Int1D,
) -> Int1D:
    """Mark the rows this source's gates let through their open branch.

    A row entered a gated target through the open branch exactly when it lands
    in a target whose gate was evaluated this period and no evaluated gate
    closed for it: a row is eligible only for the gate of its ordinary draw, so
    a closed branch that lands in another gate's target is excluded by its own
    closed mask. Every other row of the source holds no entry next period.

    Args:
        source_id: The id of the source regime whose gates were evaluated.
        target_ids: The ids of the gated targets, one per closed mask.
        closed_masks: Each gated target's closed-branch mask, in `target_ids` order.
        routed_ids: Next-period regime membership after routing.
        subjects_in_regime: Rows of the source regime this period.
        new_entered_through: The next period's entries as accumulated so far.

    Returns:
        The next period's entries with this source's rows written.
    """
    entered = jnp.zeros_like(subjects_in_regime)
    closed = jnp.zeros_like(subjects_in_regime)
    for target_id, mask in zip(target_ids, closed_masks, strict=True):
        entered = entered | (routed_ids == target_id)
        closed = closed | mask
    return jnp.where(
        subjects_in_regime,
        jnp.where(entered & ~closed, source_id, NO_GATED_ENTRY),
        new_entered_through,
    )


def log_dissolution_fallbacks(
    *,
    logger: logging.Logger,
    regime_name: RegimeName,
    period: int,
    n_rows: int,
    recorded: bool,
) -> None:
    """Report how many households took the route in one regime-period.

    Under a `~D_target` gate this is every dissolution the realised state
    forces, an ordinary event of the model rather than an anomaly, so it is
    reported once per regime-period at info level. The per-household record is
    on `SimulationResult.dissolution_fallbacks` when runtime checks are on;
    without them the count is raised to a warning, since nothing else records
    the households.
    """
    log = logger.info if recorded else logger.warning
    log(
        "%d subject(s) in regime '%s', period %d have no feasible action at their "
        "realised state and take their gate's dissolution route this period%s.",
        n_rows,
        regime_name,
        period,
        "" if recorded else " (enable runtime checks to record each one)",
    )


def rows_of(mask: Bool1D) -> np.ndarray:
    """Host row indices where a device mask is set."""
    return np.flatnonzero(np.asarray(mask))
