"""Forward-simulation value router for gated edges.

The forward-simulation counterpart to the solve-side gated-edge fold
(`_lcm.regime_building.gated_edges`). pylcm's forward simulation recomputes
argmaxes against the stored solution rather than storing policies; a source
regime declaring `gated_edges` needs two things this module provides, both
built from the ALREADY-SOLVED next-period arrays (no new solve-time work):

1. **Value substitution** (`substitute_gated_edge_continuations`) — exactly
   mirrors the solve-side kernel's `_with_edge_substitution`: a source's
   OWN action choice this period must be informed by the gated continuation
   `Wbar`, not the target's raw (ungated) value, or the simulated argmax
   would systematically differ from the one the solved V embeds.
2. **Regime routing** (`route_gated_edges`) — genuinely new relative to
   solve: forward simulation must decide, for each REALIZED subject, which
   regime it actually occupies next period and with what states. The gate is
   RECOMPUTED at the subject's candidate target-state draw
   (the states `calculate_next_states` already computed for the target via
   the regime's ordinary `transition` declaration — a gated edge's target is
   always ALSO an ordinary Markov transition target, so those candidate
   states already exist): each VALUE operand the gate predicate reads (the
   target's own value components, every declared `gate_refs` entry) is
   interpolated at that realized point and the SAME predicate the solve-side
   fold uses is re-applied — never by interpolating the fold's baked boolean
   `gate` array and thresholding the interpolated float, which does not
   commute with a nonlinear predicate (see `_lcm.regime_building.gated_edges
   .get_edge_simulate_gate_evaluator`'s docstring for the full rationale).
   The household is routed to the target when the recomputed gate is open,
   or to a leg's FALLBACK regime when closed; the non-taken branch's
   coordinates are discarded.

**Scope fence: one subject row stays one subject row.** A COLLECTIVE
source's gated edge (e.g. a married couple's dissolution edge) declares one
leg per stakeholder, each with its OWN fallback regime (wife -> her own
single regime, husband -> his). pylcm's forward simulation is a single
fixed-size population pass: one subject ROW cannot occupy two regimes at
once, so a dissolution does NOT split a row into two independently tracked
households. That splitting behaviour — LINKED mode, in which both partners
are separately tracked rows that unlink from each other on dissolution — is
not available; it would be a future engine feature.

What this router does implement, faithfully and tested, is SYNTHETIC mode:
it recomputes the gate at the realized state, computes EVERY leg's own
fallback state coordinates via each leg's `fallback_state_projector` and
writes each into its OWN fallback regime's per-subject state slot (so a
dissolved household's row-level record of "what regime and state would each
partner have started at" is complete and correct for every stakeholder),
and then picks ONE of them as the row's OWN continuing regime membership:
the leg whose `source_stakeholder` is the role THAT ROW carries. The role
is per subject — seeded in the initial conditions, and set from the branch
each gated edge actually takes — so one population may hold wives and
husbands side by side, each following her or his own dissolution path. What
stays synthetic is the partner: a dissolution writes every leg's fallback
coordinates but continues only the row's own, so the partner's household is
imputed rather than tracked as a row of its own. Two separately tracked
rows that must unlink into each other on dissolution are what LINKED mode
would add.

**Absent: a generic between-period state-reassignment hook.** A collective
lifecycle model may want to rewrite designated state components between
simulated periods from auxiliaries that only simulation tracks — children
ageing out of the household, say. This router offers no such callback. The
row-splitting fence above is the piece that actually blocks a faithful
full-scale collective simulate, and a GENERIC reassignment-hook API
designed in isolation — without a second, independent consumer to validate
its shape against — risks guessing wrong about what such a hook needs (e.g.
whether the externally tracked auxiliary is itself subject to the same
row-per-couple vs. row-per-stakeholder question the fence above raises).
Adding one waits on the row-reallocation question being settled; a
particular study's child-age bookkeeping belongs in that study's
replication code either way, not in this generic engine layer.
"""

from collections.abc import Callable, Mapping
from inspect import signature
from types import MappingProxyType
from typing import cast
from weakref import WeakKeyDictionary

import jax
import jax.numpy as jnp

from _lcm.engine import Regime, StateActionSpace
from _lcm.regime_building.collective import NO_ROLE
from _lcm.regime_building.gated_edges import (
    SOURCE_PARAMS,
    TARGET_PARAMS,
    EdgeArgProvenance,
    ResolvedEdgeLeg,
    bind_edge_period_context,
    build_reference_params_mapping_for_fold,
    build_same_period_mapping_for_fold,
    edge_may_fold_at_period,
)
from _lcm.regime_building.Q_and_F import SAME_PERIOD_PARAMS_ARG, SAME_PERIOD_V_ARG
from _lcm.simulation.transitions import _advance_states_for_subjects
from _lcm.solution.backward_induction import _evaluate_edge_fold, _states_for_period
from _lcm.typing import FlatParams, RegimeName, RegimeNamesToIds, StatesPerRegime
from lcm.typing import (
    Bool1D,
    BoolND,
    ContinuousState,
    DiscreteState,
    FloatND,
    Int1D,
)


def substitute_gated_edge_continuations(
    *,
    regime: Regime,
    regime_name: RegimeName,
    regimes: Mapping[RegimeName, Regime],
    period: int,
    next_regime_to_V_arr: Mapping[RegimeName, FloatND],
    base_state_action_spaces: Mapping[RegimeName, StateActionSpace],
    period_to_regime_to_V_arr: Mapping[int, Mapping[RegimeName, FloatND]],
    period_to_regime_to_dissolution_flags: Mapping[int, Mapping[RegimeName, BoolND]],
    flat_params: FlatParams,
    fold_age: float | None = None,
) -> tuple[
    MappingProxyType[RegimeName, FloatND],
    MappingProxyType[RegimeName, MappingProxyType[RegimeName, FloatND]],
]:
    """Substitute each declared edge's `Wbar` for the raw target V.

    A no-op (returns `next_regime_to_V_arr` unchanged and no same-period
    mappings) when `regime` declares no `gated_edges` — the default
    simulate path is untouched.

    `period_to_regime_to_V_arr` is the
    SPARSE per-period solution `backward_induction.solve` returns (only the
    regimes actually active — i.e. solved — in that period; see
    `solution[period] = MappingProxyType(period_solution)` there). A
    REPEATING edge (the source active over a range of ages, not just a
    single "one-shot" period) folds at every period it is active, including
    the source's own last active period — whose `period + 1` may not include
    the target at all, e.g. a self-looping edge whose target IS the source,
    past the source's own `active` boundary.

    Whether each edge fires is `edge_may_fold_at_period`'s answer, the same
    one backward induction's roll consults. A target not solved at
    `period + 1` skips (no substitution, no gate) — a one-shot edge's target
    is always present there, that being the point of a one-shot edge. A
    reference regime (a leg's `fallback` or a `gate_refs` entry) missing
    while the target is present is rejected rather than silently falling back
    to the raw (ungated) target V, which would leave the edge's routing
    undetectably disabled. Backward induction tolerates that same absence at
    a period whose `Wbar` no source reads; this call site never can, because
    the source reading it is the very regime being simulated at `period`.

    The fold lands on the target regime's grid at the period whose value it
    folds, `period + 1`. For a target carrying an `AgeSpecializedGrid` state
    that grid moves with age while keeping its number of nodes, so the
    representative axis of `base_state_action_spaces` has the right shape at
    every period and the wrong nodes at all but one: gate and projection would
    be evaluated at coordinates the target was never solved on, and every
    shape check would still pass. `_states_for_period` overrides the
    age-varying axes with period `t + 1`'s nodes, exactly as backward
    induction does for its own fold.

    Args:
        regime: The source regime declaring the edges.
        regime_name: Name of the source regime.
        regimes: Every regime, for the target's own period-`t + 1` state axes.
        period: The source's period; the fold reads `period + 1`'s arrays.
        next_regime_to_V_arr: The continuation values the source's decision
            reads, whose edge targets are substituted.
        base_state_action_spaces: Every regime's params-completed state-action
            space.
        period_to_regime_to_V_arr: Value arrays per period and regime.
        period_to_regime_to_dissolution_flags: Each collective regime's
            dissolution flag `D` per period.
        flat_params: Model parameters for all regimes.
        fold_age: Age corresponding to `period + 1`, supplied only to edge
            callables that declare `age`.

    Raises:
        ModelInitializationError: A declared edge's target is solved at
            `period + 1` but one or more of its reference regimes
            (fallbacks / gate refs) are not.

    Returns:
        Tuple `(substituted_next_regime_to_V_arr, same_period_mappings)` —
        the first has every declared edge target's slot replaced by `Wbar`
        (for edges that fired this period); the second maps target regime
        name to the SAME same-period value mapping the fold itself was
        evaluated on (target V, `D`-as-float, and every reference regime's
        V — `build_same_period_mapping_for_fold`'s output), consumed by
        `route_gated_edges` to RECOMPUTE the gate from interpolated value
        operands at the realized candidate state — only
        for edges that fired.
    """
    if not regime.gated_edges:
        return MappingProxyType(dict(next_regime_to_V_arr)), MappingProxyType({})
    next_period_V = period_to_regime_to_V_arr.get(period + 1, MappingProxyType({}))
    next_period_D = period_to_regime_to_dissolution_flags.get(
        period + 1, MappingProxyType({})
    )
    substituted = dict(next_regime_to_V_arr)
    same_period_mappings: dict[RegimeName, MappingProxyType[RegimeName, FloatND]] = {}
    for target_name, edge in regime.gated_edges.items():
        if not edge_may_fold_at_period(
            edge=edge,
            source_name=regime_name,
            fold_period=period + 1,
            solved_regimes=next_period_V,
            # `_simulate_regime_in_period` reaches this only for a regime active
            # at `period`, so the source is always there to read the `period + 1`
            # fold. Backward induction has to ask whether its own consumer exists;
            # here the caller IS that consumer, which is why an unsolved reference
            # can never be the boundary no-op it is on the solve side.
            source_reads_wbar=True,
        ):
            continue
        same_period_mapping = build_same_period_mapping_for_fold(
            edge=edge,
            period_solution=next_period_V,
            period_dissolution_flags=next_period_D,
        )
        wbar = _evaluate_edge_fold(
            # The fold compiled for `period + 1` — the period whose arrays it
            # folds, one ahead of the source standing at `period`. The gate
            # references and leg fallbacks are interpolated on their own
            # regimes' grids as of that period, which an `AgeSpecializedGrid`
            # moves without changing their shape.
            fold=edge.fold_at(period=period + 1),
            fold_period=period + 1,
            fold_age=fold_age,
            target_states=cast(
                "Mapping[str, ContinuousState | DiscreteState]",
                _states_for_period(
                    regime=regimes[target_name],
                    state_action_space=base_state_action_spaces[target_name],
                    period=period + 1,
                ),
            ),
            same_period_mapping=same_period_mapping,
            source_flat_params=flat_params[regime_name],
            reference_flat_params=build_reference_params_mapping_for_fold(
                edge=edge, flat_params=flat_params
            ),
        )
        substituted[target_name] = wbar
        # The fold's own boolean `gate` output is deliberately not captured:
        # `route_gated_edges` recomputes the gate itself from this SAME
        # same-period mapping, at the realized candidate state.
        same_period_mappings[target_name] = same_period_mapping
    return MappingProxyType(substituted), MappingProxyType(same_period_mappings)


def route_gated_edges(
    *,
    regime: Regime,
    fold_period: int,
    same_period_mappings: Mapping[RegimeName, Mapping[RegimeName, FloatND]],
    next_states: StatesPerRegime,
    regime_names_to_ids: RegimeNamesToIds,
    new_subject_regime_ids: Int1D,
    subjects_in_regime: Bool1D,
    flat_params: FlatParams,
    own_stakeholder: Int1D,
    new_own_stakeholder: Int1D,
    fold_age: float | None = None,
) -> tuple[StatesPerRegime, Int1D, Int1D]:
    """Route each subject through its regime's declared gated edges.

    A no-op (returns the inputs unchanged) when `regime` declares no
    `gated_edges`.

    For each declared edge: RECOMPUTES the gate at the
    candidate target states `calculate_next_states` already computed for the
    target (the regime's ordinary `transition` declaration always
    structurally reaches a gated edge's target — see module docstring) via
    the edge's own `simulate_gate_evaluator` — which
    interpolates the gate predicate's VALUE operands (the target's own value
    components, every declared `gate_refs` entry) at that realized point from
    `same_period_mappings[target_name]` (the target V / `D` / reference-V
    arrays, from `substitute_gated_edge_continuations`) and re-applies the
    SAME predicate the solve-side fold uses — never by interpolating the
    fold's baked boolean `gate` array and thresholding the result, which does
    not commute with a nonlinear predicate (see
    `_lcm.regime_building.gated_edges.get_edge_simulate_gate_evaluator`'s
    docstring). A subject is eligible for the edge when it is
    `subjects_in_regime` AND its ORDINARY next-regime draw
    (`new_subject_regime_ids`, snapshotted before any edge is processed)
    selected this edge's target. For each eligible subject the router then:

    - sets its own continuing regime membership to the target (gate open) or
      to its OWN leg's fallback (gate closed) — the leg is the one whose
      `source_stakeholder` is the role the row itself carries;
    - sets the role the row carries onward: the leg's `target_stakeholder` on
      an open branch, its fallback's `stakeholder` on a closed one, and the
      no-role sentinel wherever the destination is a singleton regime;
    - writes every leg's own projected fallback state coordinates into that
      leg's fallback regime, for the eligible subjects the gate TURNED AWAY
      only. A gate-open row continues in the target and enters no fallback
      regime, so its fallback state slots keep the values they held.

    A row whose ordinary draw selected a different regime — this edge's
    target was never reached — keeps that draw untouched by this edge: this
    is also what makes several gated edges declared on one source
    order-independent, since each edge's eligibility is decided from the same
    pre-loop snapshot rather than from the (successively overwritten) routing
    result of an earlier-processed edge.

    Args:
        fold_period: The period whose value arrays `same_period_mappings`
            holds, i.e. the source's `period + 1` — one AHEAD of the period
            the source is being simulated in, because the gate is decided on
            the value the subject would enter next period. It selects the
            edge's gate evaluator, which interpolates the target's and the
            gate references' value arrays on THEIR grids as of that period;
            under an `AgeSpecializedGrid` those nodes move with age while
            their shape does not, so passing the source's own period reads
            every operand one period's nodes off with no shape error.
        fold_age: Age corresponding to `fold_period`, supplied only to edge
            callables that declare `age`.
        own_stakeholder: Each subject's role THIS period, as a code in the
            model's role vocabulary, or the no-role sentinel for a subject
            occupying none. A collective source's legs carry roles, so this is
            what decides which leg a row follows — the wife's fallback for her,
            the husband's for him, in one cohort.
        new_own_stakeholder: The next period's roles as accumulated by the
            regimes already routed this period, written into for the subjects
            this edge moves. Mirrors `new_subject_regime_ids`: one array is
            filled in regime by regime, so a later regime never resets an
            earlier one's rows.

    Returns:
        Tuple of the routed states, the routed regime ids, and the role each
        subject carries onward.
    """
    if not regime.gated_edges:
        return next_states, new_subject_regime_ids, new_own_stakeholder

    # An IMMUTABLE snapshot of the ordinary (gate-blind) draw, taken
    # BEFORE the edge loop below ever writes into `routed_ids`. Each edge's
    # mask below reads THIS snapshot, never the successively-updated
    # `routed_ids` — so which rows an edge is even eligible to touch does not
    # depend on what an EARLIER-processed edge did, making multiple edges
    # declaration-order-independent (see the routing-mask comment below).
    ordinary_draw_ids = new_subject_regime_ids

    states = next_states
    routed_ids = new_subject_regime_ids
    routed_roles = new_own_stakeholder
    role_ids = regime.stakeholder_names_to_ids
    for target_name, edge in regime.gated_edges.items():
        # `same_period_mappings` only carries
        # an entry for a target `substitute_gated_edge_continuations`
        # actually folded this period — absent for a REPEATING edge's target
        # that is not itself solved at `period + 1` (e.g. a self-looping
        # edge past the source's own `active` boundary; see that function's
        # docstring). The edge cannot possibly gate into a target that does
        # not exist next period, so it is a no-op here too: the ordinary
        # (ungated) transition draw and candidate states already computed
        # upstream stand, unrouted and unoverridden. Byte-identical for
        # every one-shot edge, whose target is always present at
        # `period + 1`.
        if target_name not in same_period_mappings:
            continue
        candidate_target_states = dict(next_states[target_name])
        reference_params = build_reference_params_mapping_for_fold(
            edge=edge, flat_params=flat_params
        )

        # Recompute the gate from interpolated VALUE
        # operands at the realized candidate state, instead of interpolating
        # the fold's baked boolean `gate` array and thresholding it.
        #
        # Every param the evaluator needs is bound from the regime that
        # OWNS it, as recorded argument by argument in the evaluator's published
        # `arg_provenance`. Filtering two dicts by unqualified names and merging
        # them is not merely untidy but wrong, and no merge ORDER fixes
        # it: an interpolator's runtime grid helper is named after the STATE
        # alone (`x__points`), so a source and a target that both declare a state
        # `x` contribute the same qname, the filtered dicts intersect, and one
        # keyword cannot carry both regimes' points. Whichever dict is applied
        # last wins the collision and the other regime's read is silently
        # resolved on the wrong grid. The evaluator therefore exposes each
        # namespace's params under distinct, qualified leaves, and the reference
        # regimes' own grid params never reach this signature at all — they ride
        # in `SAME_PERIOD_PARAMS_ARG`.
        simulate_gate_evaluator = edge.simulate_gate_evaluator_at(period=fold_period)
        gate_bool = jnp.asarray(
            _call_vmapped_with_accepted_kwargs(
                simulate_gate_evaluator,
                batched_kwargs=candidate_target_states,
                static_kwargs={
                    **_bind_provenance_params(
                        simulate_gate_evaluator.arg_provenance,  # ty: ignore[unresolved-attribute]
                        flat_params=flat_params,
                        source_name=regime.name,
                        target_name=target_name,
                    ),
                    **bind_edge_period_context(
                        simulate_gate_evaluator,
                        fold_period=fold_period,
                        fold_age=fold_age,
                    ),
                    SAME_PERIOD_V_ARG: same_period_mappings[target_name],
                    SAME_PERIOD_PARAMS_ARG: reference_params,
                },
                # A stateless target leaves `batched_kwargs` empty; `vmap`
                # cannot infer a batch size from nothing.
                axis_size=int(subjects_in_regime.shape[0]),
            )
        )

        target_id = regime_names_to_ids[target_name]
        # Only a row whose ORDINARY draw actually selected THIS
        # edge's target is eligible for this edge's gate at all — a row
        # whose ordinary draw picked some other (edge-unrelated, or another
        # edge's target) regime must keep that draw untouched, not be
        # force-routed through a gate it never reached. Reading the
        # immutable `ordinary_draw_ids` snapshot (not `routed_ids`) is what
        # makes multiple edges from one source order-independent: each
        # edge's eligibility is decided from the SAME pre-loop draw,
        # regardless of what an earlier edge in this loop already wrote.
        edge_mask = subjects_in_regime & (ordinary_draw_ids == target_id)

        legs = edge.legs
        # Which leg a row follows is the row's own business: the wife's
        # fallback for her, the husband's for his, in one cohort. A singleton
        # source's sole leg carries no role and answers for every row.
        own_fallback_id, open_role, closed_role = _per_row_leg_outcomes(
            legs=legs,
            own_stakeholder=own_stakeholder,
            regime_names_to_ids=regime_names_to_ids,
            role_ids=role_ids,
        )
        candidate_id = jnp.where(gate_bool, target_id, own_fallback_id)
        routed_ids = jnp.where(edge_mask, candidate_id, routed_ids)
        candidate_role = jnp.where(gate_bool, open_role, closed_role)
        routed_roles = jnp.where(edge_mask, candidate_role, routed_roles)

        # The rows this edge sends into a leg's fallback regime: eligible for
        # the edge AND turned away by the gate. A row whose gate OPENED
        # continues in the target regime and dissolves into nothing, so no
        # leg's projected coordinates belong in its record — its fallback
        # slots keep whatever they held. Making the write itself gated keeps
        # that invariant local, rather than resting on a later regime-draw
        # overwrite that never reaches the state slots.
        dissolving_mask = edge_mask & jnp.logical_not(gate_bool)

        for leg in legs:
            projector = leg.fallback_state_projector
            # The projector's free parameters are bound from the regime
            # that owns them — the SOURCE for a source-declared projection, as
            # recorded in `arg_provenance` — never from the target by merging
            # its params in wholesale. The solve-side fold projects
            # this very coordinate with `flat_params[source]`
            # (`backward_induction._evaluate_edge_fold`), so any other binding
            # writes the row into the right fallback REGIME at a STATE the
            # solved policy never priced, and carries it into the next period.
            projector_provenance = projector.arg_provenance  # ty: ignore[unresolved-attribute]
            # A projection is an ordinary model function, written for ONE
            # household's scalar state — the solve-side fold evaluates it at one
            # target grid cell — so the population is mapped over, exactly as the
            # gate evaluator above is. Handing it the whole state column instead
            # agrees only for an elementwise projection; a table lookup
            # (`settlement[health]`) or any other rank-changing read returns one
            # number for the whole cohort.
            fallback_states = cast(
                "Mapping[str, FloatND]",
                _call_vmapped_with_accepted_kwargs(
                    projector,
                    batched_kwargs=candidate_target_states,
                    static_kwargs={
                        **_bind_provenance_params(
                            projector_provenance,
                            flat_params=flat_params,
                            source_name=regime.name,
                            target_name=target_name,
                        ),
                        **bind_edge_period_context(
                            projector,
                            fold_period=fold_period,
                            fold_age=fold_age,
                        ),
                    },
                    axis_size=int(subjects_in_regime.shape[0]),
                ),
            )
            states = _advance_states_for_subjects(
                states_per_regime=states,
                next_states_per_regime=MappingProxyType(
                    {
                        leg.realized_fallback.regime: MappingProxyType(
                            dict(fallback_states)
                        )
                    }
                ),
                subject_indices=dissolving_mask,
            )

    return states, routed_ids, routed_roles


def _per_row_leg_outcomes(
    *,
    legs: tuple[ResolvedEdgeLeg, ...],
    own_stakeholder: Int1D,
    regime_names_to_ids: RegimeNamesToIds,
    role_ids: Mapping[str, int],
) -> tuple[Int1D, Int1D, Int1D]:
    """Read each row's own leg and both of that leg's complete destinations.

    A leg owns four facts about where its rows go — the open branch's regime
    and role, and the closed branch's — and all four have to travel together:
    routing a row to the right regime while leaving it carrying the role it had
    in the household it just left makes its next dissolution follow the wrong
    partner's leg.

    A singleton source declares one role-less leg, which answers for every row.

    Args:
        legs: The edge's resolved legs, in source stakeholder order.
        own_stakeholder: Each subject's role code.
        regime_names_to_ids: Immutable mapping of regime names to codes.
        role_ids: The model's role vocabulary.

    Returns:
        Tuple of each subject's closed-branch regime id, its open-branch role,
        and its closed-branch role.
    """

    def _role_of(name: str | None) -> int:
        return NO_ROLE if name is None else role_ids[name]

    first = legs[0]
    fallback_id = jnp.full_like(
        own_stakeholder, regime_names_to_ids[first.realized_fallback.regime]
    )
    open_role = jnp.full_like(own_stakeholder, _role_of(first.target_stakeholder))
    closed_role = jnp.full_like(
        own_stakeholder, _role_of(first.realized_fallback.stakeholder)
    )
    for leg in legs[1:]:
        mine = own_stakeholder == _role_of(leg.source_stakeholder)
        fallback_id = jnp.where(
            mine, regime_names_to_ids[leg.realized_fallback.regime], fallback_id
        )
        open_role = jnp.where(mine, _role_of(leg.target_stakeholder), open_role)
        closed_role = jnp.where(
            mine, _role_of(leg.realized_fallback.stakeholder), closed_role
        )
    return fallback_id, open_role, closed_role


def _bind_provenance_params(
    provenance: EdgeArgProvenance,
    *,
    flat_params: FlatParams,
    source_name: RegimeName,
    target_name: RegimeName,
) -> dict[str, object]:
    """Bind an edge callable's params, each from the regime that OWNS it.

    The router holds every regime's flat params and the realized candidate
    target states; `provenance` (published by the callable's builder in
    `_lcm.regime_building.gated_edges`) is what says which of them resolves a
    given argument. Both merge orders of two name-filtered dicts are wrong — one
    keyword cannot carry two regimes' identically named arrays, and the target
    and the source genuinely can contribute the same qname (`x__points` for a
    state `x` on a runtime irregular grid, in both regimes) — so nothing is
    merged here: each exposed leaf is looked up in exactly one namespace.

    Raises:
        KeyError: A namespace does not carry a qname the callable declares.
    """
    regime_of_namespace = {SOURCE_PARAMS: source_name, TARGET_PARAMS: target_name}
    bound: dict[str, object] = {}
    for exposed, (namespace, qname) in provenance.params.items():
        regime_name = regime_of_namespace[namespace]
        regime_params = flat_params[regime_name]
        if qname not in regime_params:
            msg = (
                f"A gated edge into '{target_name}' needs the {namespace} "
                f"regime '{regime_name}''s parameter '{qname}', which is not in "
                f"flat_params['{regime_name}'] (present: "
                f"{sorted(regime_params)})."
            )
            raise KeyError(msg)
        bound[exposed] = regime_params[qname]
    return bound


def _call_vmapped_with_accepted_kwargs(
    func: Callable,
    *,
    batched_kwargs: Mapping[str, object],
    static_kwargs: Mapping[str, object],
    axis_size: int,
) -> object:
    """Call a per-subject-scalar `func` over a whole population via `vmap`.

    `func` here is always a `_lcm.regime_building.V.get_V_interpolator`
    product (discrete-index lookup + `map_coordinates` interpolation): it is
    written to be evaluated at a single subject's SCALAR discrete indices and
    continuous coordinates, then `vmap`-ped or `productmap`-ped by its
    caller — the same idiom `_lcm.simulation.transitions
    .calculate_next_states` uses for `regime.simulation.next_state`. Calling
    it directly on whole-population `(n_subjects,)` arrays only happens to
    work when the target has zero discrete state axes (`map_coordinates`
    natively batches over query points when `len(coordinates) ==
    input.ndim`); the moment the target has >=1 discrete axis, the discrete
    lookup's fancy indexing collapses those axes into a leading batch
    dimension while leaving the continuous axes trailing, which breaks that
    invariant. `vmap`-ing here makes every call genuinely scalar-per-subject
    for both discrete and continuous axes, which is correct in both cases.

    `batched_kwargs` (mapped over subject axis 0 — e.g. the candidate target
    states) and `static_kwargs` (held fixed across subjects — e.g. regime
    params and the raw grid-level array being read) are filtered down to the
    names `func` accepts; on a name collision, `static_kwargs` wins. That
    precedence is safe here: `static_kwargs` carries the provenance-bound
    params, whose exposed names are namespace-qualified and so cannot collide
    with a target state name.

    `axis_size` is REQUIRED, not inferred. A STATELESS gated target (no
    continuous or discrete states of its own — e.g. a terminal scrap-value
    regime) leaves `batched` empty after filtering, and `vmap` cannot infer
    a batch size from zero batched arguments: it raises "vmap wrapped
    function must be passed at least one argument containing an array".
    Passing the population size explicitly makes the stateless case a
    legal broadcast instead of a crash.
    """
    accepted = _accepted_arg_names(func)
    static = {name: value for name, value in static_kwargs.items() if name in accepted}
    batched = {
        name: value
        for name, value in batched_kwargs.items()
        if name in accepted and name not in static
    }
    return _population_call(func, axis_size=axis_size)(batched, static)


# What each edge callable was built once and for all with. Weakly keyed, so a
# model's compiled artifacts are released together with the model.
_ACCEPTED_ARG_NAMES: WeakKeyDictionary[Callable, frozenset[str]] = WeakKeyDictionary()
_POPULATION_CALLS: WeakKeyDictionary[Callable, dict[int, Callable]] = (
    WeakKeyDictionary()
)


def _accepted_arg_names(func: Callable) -> frozenset[str]:
    """Return the argument names `func` accepts, read off its signature once.

    An edge callable's signature is fixed at model build, while the router asks
    for it once per edge per period per subject chunk.
    """
    accepted = _ACCEPTED_ARG_NAMES.get(func)
    if accepted is None:
        accepted = frozenset(signature(func).parameters)
        _ACCEPTED_ARG_NAMES[func] = accepted
    return accepted


def _population_call(func: Callable, *, axis_size: int) -> Callable:
    """Return `func` mapped over a population of `axis_size` subjects, compiled.

    Built ONCE per `(func, axis_size)` and reused for every later call.
    Rebuilding it per call would leave the compiled executable unreachable —
    `jax.jit` keys its cache on the wrapped function object, so a fresh
    closure each time recompiles each time — and every period and every
    subject chunk calls the same edge callable again over the same population
    size.

    The per-subject and the shared arguments enter as two POSITIONAL
    arguments, because `in_axes` addresses positional arguments only: a
    keyword argument is always mapped along axis 0, which is exactly what the
    shared arguments (regime params, whole-grid value arrays) must not be.
    Passing them through rather than closing over them is what makes the built
    call reusable across periods, whose params and value arrays differ.

    `jax.jit` wraps the `vmap` rather than the scalar function, so the whole
    population is one compiled executable and one dispatch instead of an
    op-by-op eager traversal per call. Its cache keys on the arguments'
    shapes, dtypes and pytree structure, all of which are properties of the
    model rather than of the period or chunk, so the compilation is paid once.
    """
    per_axis_size = _POPULATION_CALLS.setdefault(func, {})
    call = per_axis_size.get(axis_size)
    if call is None:

        def _call_one_subject(
            one_subject_kwargs: Mapping[str, object],
            shared_kwargs: Mapping[str, object],
        ) -> object:
            return func(**one_subject_kwargs, **shared_kwargs)

        call = jax.jit(
            jax.vmap(_call_one_subject, in_axes=(0, None), axis_size=axis_size)
        )
        per_axis_size[axis_size] = call
    return call
