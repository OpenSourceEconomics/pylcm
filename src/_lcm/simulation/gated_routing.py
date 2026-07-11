"""Forward-simulation value router for gated edges (E4).

The design-doc §2 E4 counterpart to the solve-side E3' fold
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
   regime it actually occupies next period and with what states. The same
   boolean `gate` the fold evaluated (per E3') is interpolated at the
   subject's candidate target-state draw (the states `calculate_next_states`
   already computed for the target via the regime's ordinary `transition`
   declaration — a gated edge's target is always ALSO an ordinary Markov
   transition target, so those candidate states already exist) and
   thresholded; the household is routed to the target when open, or to a
   leg's FALLBACK regime when closed, discarding the other branch's
   coordinates — precisely the source of the wording in the design doc and
   the implementation plan ("discard the non-taken branch's coordinates").

**Scope fence (documented, not silently dropped).** A COLLECTIVE source's
gated edge (e.g. a married couple's divorce edge) declares one leg per
stakeholder, each with its OWN fallback regime (wife -> her own single
regime, husband -> his). pylcm's forward simulation is a single fixed-size
population pass: one subject ROW cannot occupy two regimes at once, so a
genuine second forward-simulated row per additional stakeholder — subject
population reallocation on divorce — is NOT implemented here; it is a
follow-up engine feature (see `pylcm-extension-implementation-plan.md`,
slice 6 notes), analogous in spirit to the deferred child-age
state-reassignment hook. What IS implemented, faithfully and tested: the
router recomputes the gate at the realized state, computes EVERY leg's own
fallback state coordinates via `Regime.gated_edge_leg_projectors` and writes
each into its OWN fallback regime's per-subject state slot (so a divorced
household's row-level record of "what regime and state would each partner
have started at" is complete and correct for every stakeholder), and then
picks ONE of them — deterministically, the FIRST declared leg (source
stakeholder order) — as the row's own continuing regime membership. This
mirrors the ENGINE's existing pattern elsewhere (e.g. `calculate_next_states`
already computes states for every structurally reachable target regime
regardless of which one is stochastically drawn as `next_regime_id`) rather
than inventing a new convention.

**Deferred: the generic between-period state-reassignment hook.** The
design doc (§2 E4) and EKL's App. E.2 additionally motivate a callback that
rewrites designated state components between simulated periods from
externally tracked auxiliaries (e.g. EKL's child-age bookkeeping: children
exit the household at 19, tracked only in simulation). This is NOT
implemented in this slice: the row-splitting scope fence above is the piece
that actually blocks a faithful EKL-scale collective simulate, and a
GENERIC reassignment-hook API designed in isolation — without a second,
independent consumer to validate its shape against — risks guessing wrong
about what such a hook needs (e.g. whether the externally tracked auxiliary
is itself subject to the same row-per-couple vs. row-per-stakeholder
question the scope fence above raises). Left for a follow-up slice once the
row-reallocation question is settled; EKL's specific child-age logic
belongs in an EKL replication module either way, not in this generic engine
layer.
"""

from collections.abc import Callable, Mapping
from inspect import signature
from types import MappingProxyType
from typing import cast

import jax.numpy as jnp

from _lcm.engine import Regime, StateActionSpace
from _lcm.regime_building.gated_edges import (
    GATE_ARR_NAME,
    GATE_THRESHOLD,
    build_same_period_mapping_for_fold,
)
from _lcm.simulation.transitions import _advance_states_for_subjects
from _lcm.solution.backward_induction import _evaluate_edge_fold
from _lcm.typing import FlatParams, RegimeName, RegimeNamesToIds, StatesPerRegime
from lcm.typing import Bool1D, BoolND, FloatND, Int1D


def substitute_gated_edge_continuations(
    *,
    regime: Regime,
    regime_name: RegimeName,
    period: int,
    next_regime_to_V_arr: Mapping[RegimeName, FloatND],
    base_state_action_spaces: Mapping[RegimeName, StateActionSpace],
    period_to_regime_to_V_arr: Mapping[int, Mapping[RegimeName, FloatND]],
    period_to_regime_to_divorce_flags: Mapping[int, Mapping[RegimeName, BoolND]],
    flat_params: FlatParams,
) -> tuple[MappingProxyType[RegimeName, FloatND], MappingProxyType[RegimeName, BoolND]]:
    """Substitute each declared edge's ``Wbar`` for the raw target V (E4).

    A no-op (returns `next_regime_to_V_arr` unchanged and no gate arrays)
    when `regime` declares no `gated_edges` — the default simulate path is
    untouched.

    Returns:
        Tuple `(substituted_next_regime_to_V_arr, gate_arrays)` — the first
        has every declared edge target's slot replaced by `Wbar`; the second
        maps target regime name to the fold's raw grid-level `gate` array
        (consumed by `route_gated_edges`).
    """
    if not regime.gated_edges:
        return MappingProxyType(dict(next_regime_to_V_arr)), MappingProxyType({})
    next_period_V = period_to_regime_to_V_arr.get(period + 1, MappingProxyType({}))
    next_period_D = period_to_regime_to_divorce_flags.get(
        period + 1, MappingProxyType({})
    )
    substituted = dict(next_regime_to_V_arr)
    gate_arrays: dict[RegimeName, BoolND] = {}
    for target_name, edge in regime.gated_edges.items():
        same_period_mapping = build_same_period_mapping_for_fold(
            edge=edge,
            period_solution=next_period_V,
            period_divorce_flags=next_period_D,
        )
        wbar, gate = _evaluate_edge_fold(
            fold=regime.gated_edge_folds[target_name],
            target_states=base_state_action_spaces[target_name].states,
            same_period_mapping=same_period_mapping,
            source_flat_params=flat_params[regime_name],
        )
        substituted[target_name] = wbar
        gate_arrays[target_name] = gate
    return MappingProxyType(substituted), MappingProxyType(gate_arrays)


def _call_with_accepted_kwargs(func: Callable, kwargs: Mapping[str, object]) -> object:
    """Call `func` with only the keyword arguments its signature declares."""
    accepted = set(signature(func).parameters)
    return func(**{name: value for name, value in kwargs.items() if name in accepted})


def route_gated_edges(
    *,
    regime: Regime,
    gate_arrays: Mapping[RegimeName, BoolND],
    next_states: StatesPerRegime,
    regime_names_to_ids: RegimeNamesToIds,
    new_subject_regime_ids: Int1D,
    subjects_in_regime: Bool1D,
    flat_params: FlatParams,
) -> tuple[StatesPerRegime, Int1D]:
    """Route each subject through its regime's declared gated edges (E4).

    A no-op (returns the inputs unchanged) when `regime` declares no
    `gated_edges`.

    For each declared edge: interpolates the target's raw `gate` array
    (`gate_arrays[target_name]`, from `substitute_gated_edge_continuations`)
    at the candidate target states `calculate_next_states` already computed
    for the target (the regime's ordinary `transition` declaration always
    structurally reaches a gated edge's target — see module docstring), then
    for `subjects_in_regime`: writes EVERY leg's own fallback state
    coordinates into its fallback regime's state slot, and sets this row's
    own continuing regime membership to the target (gate open) or the FIRST
    leg's fallback (gate closed) — see the module docstring's scope fence for
    a collective (multi-leg) source.
    """
    if not regime.gated_edges:
        return next_states, new_subject_regime_ids

    states = next_states
    routed_ids = new_subject_regime_ids
    for target_name, edge in regime.gated_edges.items():
        candidate_target_states = dict(next_states[target_name])
        target_kwargs = {**candidate_target_states, **flat_params[target_name]}

        gate_interpolator = regime.gated_edge_gate_interpolators[target_name]
        gate_float = _call_with_accepted_kwargs(
            gate_interpolator,
            {
                **target_kwargs,
                GATE_ARR_NAME: jnp.asarray(gate_arrays[target_name], dtype=float),
            },
        )
        gate_bool = jnp.asarray(gate_float) > GATE_THRESHOLD

        target_id = regime_names_to_ids[target_name]
        legs = edge.legs
        primary_fallback_id = regime_names_to_ids[legs[0].fallback.regime]
        candidate_id = jnp.where(gate_bool, target_id, primary_fallback_id)
        routed_ids = jnp.where(subjects_in_regime, candidate_id, routed_ids)

        projectors = regime.gated_edge_leg_projectors[target_name]
        for leg, projector in zip(legs, projectors, strict=True):
            fallback_states = cast(
                "Mapping[str, FloatND]",
                _call_with_accepted_kwargs(projector, target_kwargs),
            )
            states = _advance_states_for_subjects(
                states_per_regime=states,
                next_states_per_regime=MappingProxyType(
                    {leg.fallback.regime: MappingProxyType(dict(fallback_states))}
                ),
                subject_indices=subjects_in_regime,
            )

    return states, routed_ids
