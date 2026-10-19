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

A household that reached the regime through no gated edge (an initial
condition, or an ungated transition) has no route to take, and the caller
raises `NoFeasibleActionError` for it as before.
"""

import dataclasses
import logging
from collections.abc import Mapping
from types import MappingProxyType

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
    """The regime it entered from, whose gated edge supplies the route."""
    fallback: RegimeName
    """The fallback regime of its own leg, where it acts this period instead."""


@dataclasses.dataclass(frozen=True, kw_only=True)
class RerouteOutcome:
    """The carriers after rerouting, and which rows moved."""

    states: StatesPerRegime
    subject_regime_ids: Int1D
    own_stakeholder: Int1D
    new_subject_regime_ids: Int1D
    new_own_stakeholder: Int1D
    rerouted: Bool1D
    """Rows sent to a fallback regime this period."""
    fallback_ids: Int1D
    """Each rerouted row's fallback regime id; meaningless elsewhere."""


def reroute_rows_without_feasible_action(
    *,
    regime_name: RegimeName,
    regimes: Mapping[RegimeName, Regime],
    empty: Bool1D,
    source_regime_ids: Int1D,
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
    """Move every empty row that entered through a gated edge to its fallback.

    Args:
        regime_name: The regime whose decision found no feasible action.
        regimes: Every regime of the model.
        empty: Rows of this regime with no feasible action this period.
        source_regime_ids: Each row's regime in the previous period.
        source_roles: Each row's role in the previous period.
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
    rerouted = jnp.zeros_like(empty)
    fallback_ids = jnp.full_like(subject_regime_ids, -1)
    for source_name, source in regimes.items():
        edge = source.gated_edges.get(regime_name)
        if edge is None:
            continue
        rows = empty & (source_regime_ids == regime_names_to_ids[source_name])
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
                batched_kwargs=dict(states[regime_name]),
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
        rerouted=rerouted,
        fallback_ids=fallback_ids,
    )


def log_dissolution_fallbacks(
    *,
    logger: logging.Logger,
    records: tuple[DissolutionFallback, ...],
    regime_name: RegimeName,
    period: int,
    n_rows: int,
) -> None:
    """Report every use of the route, never silently."""
    if records:
        for record in records:
            logger.warning(
                "Subject %d in regime '%s', period %d, state %s has no feasible "
                "action; it takes the dissolution route of the gated edge from "
                "'%s' into '%s' this period.",
                record.subject_id,
                record.regime,
                record.period,
                dict(record.state),
                record.source,
                record.fallback,
            )
        return
    logger.warning(
        "%d subject(s) in regime '%s', period %d have no feasible action and take "
        "their gate's dissolution route this period. Enable runtime checks to "
        "record each one.",
        n_rows,
        regime_name,
        period,
    )


def rows_of(mask: Bool1D) -> np.ndarray:
    """Host row indices where a device mask is set."""
    return np.flatnonzero(np.asarray(mask))
