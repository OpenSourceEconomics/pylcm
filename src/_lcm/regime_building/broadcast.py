"""Model-level regime slots: merge and DAG-reachability pruning.

`merge_model_slots` merges `Model(functions=..., constraints=..., states=...,
state_transitions=..., actions=...)` into every regime under the
exactly-one-level rule — a name is defined at model level or regime level,
never both — with regime-level `None` masking the model entry.

`prune_broadcast_variables` then weeds the broadcast states and actions per
regime by DAG reachability: a broadcast variable survives in a regime only if
it is a transitive input of that regime's root computations in either phase
slice. Regime-level declarations are never pruned. The needed-set is a
cross-regime fixed point: a state unused inside a regime is still required
when a candidate target keeps it and the law of motion toward that target
reads it.

`root_functions` is the single definition of those root computations. The
pruning walk here and the variable-usage check in `_lcm.model_processing`
both take their roots from it, so the two cannot disagree about what counts
as a read.
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Literal, cast

from dags import get_ancestors, with_signature

from _lcm.grids import Grid
from _lcm.processes import _ContinuousStochasticProcess
from _lcm.reachability import PhaseName, candidate_targets_from_transition
from _lcm.regime_building.age_specialization import resolve_node
from _lcm.regime_building.phases import (
    PhasedRegimeSpec,
    RegimePhaseSpec,
    normalize_regime_phases,
)
from _lcm.typing import RegimeName, StateName, StateOrActionName
from _lcm.utils.error_messages import format_messages
from lcm.ages import AgeGrid
from lcm.collective import CollectiveUtility
from lcm.consumption_savings_regime import NetOfAdjustmentCost
from lcm.exceptions import ModelInitializationError
from lcm.phased import Phased
from lcm.regime import Regime as UserRegime
from lcm.transition import AgeSpecializedFunction, JointTransition
from lcm.typing import UserFunction

# Which `Phased` side each `PhasedRegimeSpec` slice is built from.
_PHASE_OF_SLICE: Mapping[PhaseName, Literal["solve", "simulate"]] = MappingProxyType(
    {"solution": "solve", "simulation": "simulate"}
)

_BROADCASTABLE_SLOTS = (
    "functions",
    "constraints",
    "states",
    "state_transitions",
    "actions",
)


def merge_model_slots(
    *,
    user_regimes: Mapping[RegimeName, UserRegime],
    model_slots: Mapping[str, Mapping[str, object]],
) -> tuple[
    MappingProxyType[RegimeName, UserRegime],
    MappingProxyType[RegimeName, frozenset[StateOrActionName]],
]:
    """Merge model-level slots into every regime.

    Args:
        user_regimes: Mapping of regime names to user-provided `Regime`
            instances.
        model_slots: Mapping of slot names (`functions`, `constraints`,
            `states`, `state_transitions`, `actions`) to model-level entries.

    Returns:
        Tuple of the merged regimes and, per regime, the names of broadcast
        states and actions (the pruning candidates).

    Raises:
        ModelInitializationError: If a name is defined at both levels, or a
            regime masks a name no model-level slot provides.

    """
    errors: list[str] = []
    merged_regimes: dict[RegimeName, UserRegime] = {}
    broadcast_variables: dict[RegimeName, frozenset[StateOrActionName]] = {}

    for regime_name, user_regime in user_regimes.items():
        replacements: dict[str, Mapping[str, object]] = {}
        variable_names: set[StateOrActionName] = set()
        for slot_name in _BROADCASTABLE_SLOTS:
            regime_slot = dict(getattr(user_regime, slot_name))
            model_slot = dict(model_slots.get(slot_name, {}))
            if slot_name == "state_transitions" and user_regime.terminal:
                # Terminal regimes consume no laws of motion; broadcast laws
                # are inert there and must not violate the empty-transitions
                # rule.
                model_slot = {}
            if slot_name == "functions":
                # A household names its own utilities. A model-level entry
                # under one of those names is there for the regimes that
                # declare no household, so it does not reach this one.
                model_slot = {
                    name: value
                    for name, value in model_slot.items()
                    if name not in _names_the_household_writes(user_regime=user_regime)
                }
            errors.extend(
                _merge_one_slot(
                    slot_name=slot_name,
                    regime_name=regime_name,
                    regime_slot=regime_slot,
                    model_slot=model_slot,
                )
            )
            if slot_name == "states":
                # Sharding is a cross-regime device-layout property; one
                # model-level declaration keeps every regime consistent.
                errors.extend(
                    f"states['{name}'] in regime '{regime_name}' has "
                    f"`distributed=True` — sharding is declared at the model "
                    f"level (`Model(states=...)`)."
                    for name, grid in regime_slot.items()
                    if isinstance(grid, Grid) and grid.distributed
                )
            if slot_name in ("states", "actions"):
                variable_names |= model_slot.keys() & regime_slot.keys()
            replacements[slot_name] = {**model_slot, **regime_slot}
        # A masked state's broadcast law is dropped with it.
        masked_states = {
            name
            for name, value in user_regime.states.items()
            if value is None and name in model_slots.get("states", {})
        }
        for slot_name in _BROADCASTABLE_SLOTS:
            replacements[slot_name] = {
                name: value
                for name, value in replacements[slot_name].items()
                if value is not None
                and not (slot_name == "state_transitions" and name in masked_states)
            }
        if not errors:
            merged_regimes[regime_name] = user_regime.replace(**replacements)
            broadcast_variables[regime_name] = frozenset(
                (
                    set(model_slots.get("states", {}))
                    | set(model_slots.get("actions", {}))
                )
                & (set(replacements["states"]) | set(replacements["actions"]))
            )

    if errors:
        raise ModelInitializationError(format_messages(errors))

    return MappingProxyType(merged_regimes), MappingProxyType(broadcast_variables)


def prune_broadcast_variables(
    *,
    user_regimes: Mapping[RegimeName, UserRegime],
    broadcast_variables: Mapping[RegimeName, frozenset[StateOrActionName]],
    koopmans_aggregator: UserFunction,
    ages: AgeGrid | None = None,
    active_periods_by_regime: Mapping[RegimeName, tuple[int, ...]] | None = None,
) -> tuple[
    MappingProxyType[RegimeName, UserRegime],
    MappingProxyType[RegimeName, frozenset[StateOrActionName]],
]:
    """Weed broadcast states and actions per regime by DAG reachability.

    A broadcast variable is pruned from a regime when no root computation of
    either phase slice transitively reads it — in that regime or through a
    law of motion toward a candidate target that keeps it. Pruning drops the
    variable's grid, and for states also the regime's law entry for it.

    Args:
        user_regimes: Mapping of regime names to merged `Regime` instances.
        broadcast_variables: Per regime, the broadcast state/action names.
        koopmans_aggregator: The model-level Bellman aggregator, used as a
            reachability root in every non-terminal regime that declares none
            of its own.
        ages: The model's `AgeGrid`, used to convert a representative period
            to an age before resolving `AgeSpecializedFunction` markers.
            `None` skips resolution (no marker can appear then).
        active_periods_by_regime: The canonical per-regime activity mapping
            (from `compute_active_periods_by_regime`), used to pick each
            regime's representative active period before resolving
            `AgeSpecializedFunction` markers to a representative age so a
            broadcast variable read only through a marker is not misread as
            unused. `None` skips resolution (no marker can appear then).

    Returns:
        Tuple of the pruned regimes and, per regime, the pruned names.

    Raises:
        ModelInitializationError: If a `distributed=True` model-level state is
            pruned from a non-terminal regime.

    """
    specs = {
        regime_name: normalize_regime_phases(user_regime)
        for regime_name, user_regime in user_regimes.items()
    }
    all_regime_names = frozenset(user_regimes)

    kept: dict[RegimeName, frozenset[StateOrActionName]] = {}
    for regime_name, user_regime in user_regimes.items():
        declared = (
            set(user_regime.states) | set(user_regime.actions)
        ) - broadcast_variables[regime_name]
        kept[regime_name] = frozenset(declared)

    for phase_name in ("solution", "simulation"):
        kept = _phase_fixed_point(
            specs=specs,
            user_regimes=user_regimes,
            broadcast_variables=broadcast_variables,
            koopmans_aggregator=koopmans_aggregator,
            kept=kept,
            phase_name=phase_name,
            all_regime_names=all_regime_names,
            ages=ages,
            active_periods_by_regime=active_periods_by_regime,
        )

    pruned_regimes: dict[RegimeName, UserRegime] = {}
    pruned_variables: dict[RegimeName, frozenset[StateOrActionName]] = {}
    errors: list[str] = []
    for regime_name, user_regime in user_regimes.items():
        pruned = broadcast_variables[regime_name] - kept[regime_name]
        pruned_variables[regime_name] = frozenset(pruned)
        if not pruned:
            pruned_regimes[regime_name] = user_regime
            continue
        errors.extend(
            _sharded_pruned_errors(
                user_regime=user_regime, regime_name=regime_name, pruned=pruned
            )
        )
        pruned_regimes[regime_name] = user_regime.replace(
            states={
                name: grid
                for name, grid in user_regime.states.items()
                if name not in pruned
            },
            actions={
                name: grid
                for name, grid in user_regime.actions.items()
                if name not in pruned
            },
            state_transitions={
                name: law
                for name, law in user_regime.state_transitions.items()
                if name not in pruned
            },
        )

    if errors:
        raise ModelInitializationError(format_messages(errors))

    return MappingProxyType(pruned_regimes), MappingProxyType(pruned_variables)


def root_functions(
    *,
    regime_name: RegimeName,
    regime: UserRegime,
    all_regimes: Mapping[RegimeName, UserRegime],
    phase: Literal["solve", "simulate"],
    koopmans_aggregator: UserFunction | None = None,
) -> MappingProxyType[str, UserFunction]:
    """Collect one regime's root computations, keyed under reserved names.

    A root is a computation the model evaluates on *this* regime's grid, so
    every name it transitively reads is a genuine input of the regime. The
    roots are

    - `utility`, or one `utility_<stakeholder>` per stakeholder of a
      collective regime;
    - every derived-categorical function;
    - every constraint;
    - the Koopmans aggregator, in a non-terminal regime;
    - the regime transition — every cell of a per-target dict, or the coarse
      callable;
    - every value-constraint predicate;
    - every Pareto weight declared as a function;
    - every `same_period_refs` projection;
    - for each gated edge *whose target is this regime*, the edge's gate, its
      gate-reference projections, and its legs' fallback projections.

    That last group is why `all_regimes` is an argument: a gated edge is
    declared on the SOURCE regime but gate, gate references and fallbacks are
    all evaluated on the TARGET regime's grid, so nothing in the target's own
    slots mentions them and a walk over one regime in isolation cannot see the
    read.

    Laws of motion are deliberately absent: the two consumers ask different
    questions of them. The pruning walk roots a law toward a target that keeps
    the state, because the value has to be handed over; the usage check roots
    every law that is not an identity, because handing a state to itself is
    not a use of it. Each consumer adds its own law roots on top of these.

    Args:
        regime_name: Name of the regime whose roots are collected.
        regime: The regime itself.
        all_regimes: Mapping of regime names to every regime of the model,
            scanned for gated edges pointing at `regime_name`.
        phase: Which side of a `Phased` slot the roots are taken from.
        koopmans_aggregator: The model-level aggregator, used when the regime
            declares none of its own. `None` leaves the aggregator out, which
            is what a regime that already carries its own needs.

    Returns:
        Immutable mapping of reserved root name to the callable it roots.

    """
    return MappingProxyType(
        _valuation_roots(
            regime=regime, phase=phase, koopmans_aggregator=koopmans_aggregator
        )
        | _transition_roots(regime=regime, phase=phase)
        | _value_aware_roots(regime=regime)
        | _incoming_edge_roots(regime_name=regime_name, all_regimes=all_regimes)
    )


def _valuation_roots(
    *,
    regime: UserRegime,
    phase: Literal["solve", "simulate"],
    koopmans_aggregator: UserFunction | None,
) -> dict[str, UserFunction]:
    """Key the payoff side of the regime — utility, categoricals, constraints, W."""
    functions = {
        name: cast("UserFunction", _for_phase(value=func, phase=phase))
        for name, func in regime.decomposed_functions.items()
    }
    utility_names = (
        tuple(f"utility_{stakeholder}" for stakeholder in regime.stakeholders)
        if regime.stakeholders is not None
        else ("utility",)
    )
    roots = {
        f"__utility__{name}": functions[name]
        for name in utility_names
        if name in functions
    }
    roots |= {
        f"__derived_categorical__{name}": functions[name]
        for name in regime.derived_categoricals
        if name in functions
    }
    roots |= {
        f"__constraint__{name}": cast(
            "UserFunction", _for_phase(value=value, phase=phase)
        )
        for name, value in regime.decomposed_constraints.items()
    }
    if not regime.terminal:
        aggregator = regime.get_koopmans_aggregator(phase=phase) or koopmans_aggregator
        if aggregator is not None:
            roots["__koopmans_aggregator"] = aggregator
    return roots


def _transition_roots(
    *, regime: UserRegime, phase: Literal["solve", "simulate"]
) -> dict[str, UserFunction]:
    """Key regime routing plus every edge-local joint probability and output law."""
    roots: dict[str, UserFunction] = {}
    transition = _for_phase(value=regime.decomposed_transition, phase=phase)
    if isinstance(transition, Mapping):
        roots |= {
            f"__next_regime__{target_regime_name}": cast("UserFunction", cell)
            for target_regime_name, cell in transition.items()
        }
    elif transition is not None:
        roots["__next_regime"] = cast("UserFunction", transition)

    for target_name, kernels in regime.joint_transitions.items():
        for kernel_name, raw in kernels.items():
            kernel = cast("JointTransition", _for_phase(value=raw, phase=phase))
            roots[f"__joint_probability__{target_name}__{kernel_name}"] = cast(
                "UserFunction", kernel.probabilities
            )
            roots |= {
                f"__joint_output__{target_name}__{state_name}": output
                for state_name, output in kernel.outputs.items()
            }
    return roots


def _value_aware_roots(*, regime: UserRegime) -> dict[str, UserFunction]:
    """Key a collective regime's household declarations.

    Its value constraints, its same-period projections, and its Pareto weights
    — a weight is evaluated on this regime's grid at every cell, so a state it
    reads is as much an input of the regime as one the utility reads.
    """
    roots = {
        f"__value_constraint__{name}": predicate
        for name, predicate in regime.value_constraints.items()
    }
    if regime.pareto_objective is not None:
        roots |= {
            f"__pareto_weight__{name}": cast("UserFunction", weight)
            for name, weight in regime.pareto_objective.weights.items()
            if callable(weight)
        }
    roots |= {
        f"__same_period_ref__{ref_name}__{state_name}": projection
        for ref_name, ref in regime.same_period_refs.items()
        for state_name, projection in ref.projection.items()
    }
    return roots


def _incoming_edge_roots(
    *, regime_name: RegimeName, all_regimes: Mapping[RegimeName, UserRegime]
) -> dict[str, UserFunction]:
    """Key the gated-edge functions other regimes evaluate on this regime's grid."""
    roots: dict[str, UserFunction] = {}
    for source_name, source in all_regimes.items():
        edge = source.gated_edges.get(regime_name)
        if edge is None:
            continue
        roots[f"__incoming_gate__{source_name}"] = edge.gate
        roots |= {
            f"__incoming_gate_ref__{source_name}__{ref_name}__{state_name}": projection
            for ref_name, ref in edge.gate_refs.items()
            for state_name, projection in ref.projection.items()
        }
        # Both phases of a `Phased` fallback are rooted: a state read only by
        # the settlement projection is still read, and pruning it would leave
        # the simulate leg with a coordinate it cannot form.
        roots |= {
            f"__incoming_fallback__{source_name}__{leg_name}__{phase}__{state_name}": (
                projection
            )
            for leg_name, leg in edge.legs.items()
            for phase, ref in (
                ("solve", leg.solve_fallback),
                ("simulate", leg.simulate_fallback),
            )
            for state_name, projection in ref.projection.items()
        }
    return roots


def _for_phase(*, value: object, phase: Literal["solve", "simulate"]) -> object:
    """Take the side of a `Phased` slot value this phase runs, else the value."""
    if isinstance(value, Phased):
        return value.solve if phase == "solve" else value.simulate
    return value


def _phase_fixed_point(
    *,
    specs: Mapping[RegimeName, PhasedRegimeSpec],
    user_regimes: Mapping[RegimeName, UserRegime],
    broadcast_variables: Mapping[RegimeName, frozenset[StateOrActionName]],
    koopmans_aggregator: UserFunction,
    kept: Mapping[RegimeName, frozenset[StateOrActionName]],
    phase_name: PhaseName,
    all_regime_names: frozenset[RegimeName],
    ages: AgeGrid | None,
    active_periods_by_regime: Mapping[RegimeName, tuple[int, ...]] | None,
) -> dict[RegimeName, frozenset[StateOrActionName]]:
    """Grow the kept-sets to this phase slice's least fixed point.

    Per iteration, each regime's needed-set is the DAG ancestry of its root
    computations plus the laws of motion toward candidate targets that
    currently keep the law's state; a broadcast variable joins the kept-set
    when needed. Monotone in the kept-sets, so iteration terminates. The
    input mapping is left untouched; the grown kept-sets are returned.
    """
    candidates_by_source = {
        regime_name: frozenset(
            candidate_targets_from_transition(
                transition=getattr(spec, phase_name).regime_transition,
                all_regime_names=all_regime_names,
            )
        )
        for regime_name, spec in specs.items()
    }

    grown = dict(kept)
    while True:
        changed = False
        for regime_name, user_regime in user_regimes.items():
            candidates = broadcast_variables[regime_name] - grown[regime_name]
            if not candidates:
                continue
            phase_slice = getattr(specs[regime_name], phase_name)
            needed = _needed_names(
                phase_slice=phase_slice,
                regime_name=regime_name,
                user_regime=user_regime,
                user_regimes=user_regimes,
                phase_name=phase_name,
                koopmans_aggregator=koopmans_aggregator,
                candidate_targets=candidates_by_source[regime_name],
                kept=grown,
                ages=ages,
                active_periods=(
                    None
                    if active_periods_by_regime is None
                    else active_periods_by_regime.get(regime_name, ())
                ),
            )
            needed |= _state_conditioned_names(
                user_regime=user_regime,
                reachable_targets=candidates_by_source[regime_name] & all_regime_names,
                user_regimes=user_regimes,
                candidates=broadcast_variables[regime_name],
                grown_here=grown[regime_name],
            )
            newly_kept = candidates & needed
            if newly_kept:
                grown[regime_name] = grown[regime_name] | newly_kept
                changed = True
        if not changed:
            return grown


def _state_conditioned_names(
    *,
    user_regime: UserRegime,
    reachable_targets: frozenset[RegimeName],
    user_regimes: Mapping[RegimeName, UserRegime],
    candidates: frozenset[StateOrActionName],
    grown_here: frozenset[StateOrActionName],
) -> frozenset[StateName]:
    """Collect the conditioning states this regime's Q reads through a process.

    A state-conditioned process reads `state_conditioned.on`: the generated solve
    weights and simulation draw both take it as an argument. It is declared as
    *metadata* on the grid rather than in a user function, so `_needed_names`'s
    callable-DAG ancestry cannot see it. Naming it here is what keeps a conditioning
    state declared at model level — a broadcast candidate reaching nothing else — from
    being pruned before the process can ask for it, which would otherwise surface as a
    misleading "must name a DiscreteGrid state in the same regime" build error.

    Two sources contribute:

    - **local processes** — declared in this regime. Only a *retained* process forces
      its conditioner (a process that is itself a pruned broadcast candidate reads
      nothing), so this stays monotone in `grown_here` and the caller's fixed point
      terminates.
    - **reachable-target processes** — a conditioned process in a regime this one can
      transition into has its transition weight built into *this* regime's Q, evaluated
      at *this* regime's `on` state. So the conditioner is needed in the source too,
      not only the process's own regime.

    Keeping the state also keeps its law of motion, which the caller filters by the
    same pruned-set.
    """
    names: set[StateName] = set()
    for name, grid in user_regime.states.items():
        if not isinstance(grid, _ContinuousStochasticProcess):
            continue
        if grid.state_conditioned is None:
            continue
        if name in candidates and name not in grown_here:
            continue  # the process itself is not (yet) retained
        names.add(grid.state_conditioned.on)
    for target in reachable_targets:
        for grid in user_regimes[target].states.values():
            if (
                isinstance(grid, _ContinuousStochasticProcess)
                and grid.state_conditioned is not None
            ):
                names.add(grid.state_conditioned.on)
    return frozenset(names)


def _resolved_at_representative_age(
    *,
    mapping: Mapping[str, UserFunction],
    ages: AgeGrid | None,
    active_periods: tuple[int, ...] | None,
) -> Mapping[str, UserFunction]:
    """Resolve `AgeSpecializedFunction` markers in `mapping` at a representative age.

    The dependency structure `get_ancestors` needs is age-invariant, so any active
    period serves. Returns `mapping` unchanged when `ages` is `None` (no marker can
    appear then) or `active_periods` is empty (the regime is about to fail with a
    more specific error once age normalization runs).
    """
    if (
        ages is None
        or not active_periods
        or not any(
            isinstance(value, AgeSpecializedFunction) for value in mapping.values()
        )
    ):
        return mapping
    representative_age = float(ages.period_to_age(active_periods[0]))
    return {
        name: cast("UserFunction", resolve_node(node=value, age=representative_age))
        for name, value in mapping.items()
    }


def _needed_names(
    *,
    phase_slice: RegimePhaseSpec,
    regime_name: RegimeName,
    user_regime: UserRegime,
    user_regimes: Mapping[RegimeName, UserRegime],
    phase_name: PhaseName,
    koopmans_aggregator: UserFunction,
    candidate_targets: frozenset[RegimeName],
    kept: Mapping[RegimeName, frozenset[StateOrActionName]],
    ages: AgeGrid | None,
    active_periods: tuple[int, ...] | None,
) -> set[str]:
    """Collect every name this phase slice's root computations read.

    The roots are `root_functions`' — utility, derived categoricals,
    constraints, the Koopmans aggregator, the regime transition, the
    value-constraint predicates, the `same_period_refs` projections and the
    incoming gated edges' gates, gate references and fallbacks — plus the laws
    of motion toward candidate targets that keep the law's state, which are
    pruning's own: whatever such a law reads has to stay alive here so the
    target can be handed its value.

    The whole pool is resolved at a representative active period before the DAG
    walk, so `get_ancestors` sees a marked node's real argument names instead of
    `AgeSpecializedFunction.__call__`'s generic `(*args, **kwargs)`.
    """
    pool: dict[str, UserFunction] = dict(phase_slice.functions)

    pool |= _composed_resources_edge(user_regime=user_regime, pool=pool)

    roots = dict(
        root_functions(
            regime_name=regime_name,
            regime=user_regime,
            all_regimes=user_regimes,
            phase=_PHASE_OF_SLICE[phase_name],
            koopmans_aggregator=koopmans_aggregator,
        )
    )
    roots |= _law_roots(
        phase_slice=phase_slice, candidate_targets=candidate_targets, kept=kept
    )
    pool |= roots

    targets = list(roots)
    if not targets:
        return set()
    resolved_pool = _resolved_at_representative_age(
        mapping=pool, ages=ages, active_periods=active_periods
    )
    return set(get_ancestors(resolved_pool, targets=targets, include_targets=True))


def _composed_resources_edge(
    *, user_regime: UserRegime, pool: Mapping[str, UserFunction]
) -> dict[str, UserFunction]:
    """Supply the resources node a `NetOfAdjustmentCost` regime composes later.

    Model finalization installs `<resources> = <before_cost> - <cost>`, but
    pruning runs first and would otherwise see `resources` as a leaf, reaching
    neither operand. A variable read only by the cost would then be pruned as
    unused and the composition would fail on the missing name. The stub carries
    the composition's argument names and no body: reachability is what is being
    asked, so only the edge matters.
    """
    resources = getattr(getattr(user_regime, "liquid", None), "resources", None)
    if not isinstance(resources, NetOfAdjustmentCost) or resources.output in pool:
        return {}

    @with_signature(args=[resources.before_cost, resources.cost])
    def composed_resources(*args: object, **kwargs: object) -> None: ...

    return {resources.output: cast("UserFunction", composed_resources)}


def _law_roots(
    *,
    phase_slice: RegimePhaseSpec,
    candidate_targets: frozenset[RegimeName],
    kept: Mapping[RegimeName, frozenset[StateOrActionName]],
) -> dict[str, UserFunction]:
    """Key the laws of motion that rescue a state as pruning roots.

    A law counts only toward a candidate target that currently keeps the
    law's state — that target needs the handed-over value, so whatever the
    law reads stays alive in this regime.
    """
    roots: dict[str, UserFunction] = {}
    for state_name, raw in phase_slice.state_transitions.items():
        laws: dict[RegimeName, object] = (
            dict(cast("Mapping[RegimeName, object]", raw))
            if isinstance(raw, Mapping)
            else dict.fromkeys(candidate_targets, raw)
        )
        for target_regime_name, law in laws.items():
            if (
                law is not None
                and target_regime_name in candidate_targets
                and state_name in kept.get(target_regime_name, frozenset())
            ):
                roots[f"__law_{state_name}_{target_regime_name}"] = cast(
                    "UserFunction", law
                )
    return roots


def _sharded_pruned_errors(
    *,
    user_regime: UserRegime,
    regime_name: RegimeName,
    pruned: frozenset[StateOrActionName],
) -> list[str]:
    """Reject pruning a `distributed=True` state from a non-terminal regime."""
    if user_regime.terminal:
        return []
    return [
        f"Sharded state '{name}' is pruned from non-terminal regime "
        f"'{regime_name}' — its DAG never reads the state, so the sharded "
        f"V-array axis would disappear there. Remove `distributed=True` "
        f"from the model-level declaration, or make the regime use the state."
        for name in sorted(pruned)
        if isinstance(grid := user_regime.states.get(name), Grid) and grid.distributed
    ]


def _merge_one_slot(
    *,
    slot_name: str,
    regime_name: RegimeName,
    regime_slot: Mapping[str, object],
    model_slot: Mapping[str, object],
) -> list[str]:
    """Apply the exactly-one-level rule to one slot of one regime.

    Args:
        slot_name: Which regime slot is being merged.
        regime_name: Name of the regime the slot belongs to.
        regime_slot: The regime's own entries.
        model_slot: The model-level entries that reach this regime.

    Returns:
        List of error messages, empty when the slot merges cleanly.

    """
    errors: list[str] = []
    for name, value in regime_slot.items():
        if value is None:
            if name not in model_slot:
                errors.append(
                    f"{slot_name}['{name}'] in regime '{regime_name}' is "
                    f"`None`, but no model-level entry provides '{name}' — "
                    f"there is nothing to mask.",
                )
        elif name in model_slot:
            errors.append(
                f"Ambiguous specification for {slot_name}['{name}'] in "
                f"regime '{regime_name}': defined at model level and regime "
                f"level. Remove one, or mask the model entry with `None`.",
            )
    return errors


def _names_the_household_writes(*, user_regime: UserRegime) -> frozenset[str]:
    """Return the function names this regime's household supplies for itself.

    A collective regime declares `utility` as a household, and with it a body
    for every stakeholder it does not delegate. Those names are the household's
    and a model-level entry under one of them belongs to the other regimes. A
    stakeholder the household *does* delegate is left out, which is exactly how
    a model-level body reaches her.
    """
    declaration = user_regime.functions.get("utility")
    if not isinstance(declaration, CollectiveUtility):
        return frozenset()
    return frozenset(
        {
            "utility",
            *(
                f"utility_{stakeholder}"
                for stakeholder, body in declaration.utilities.items()
                if body is not None
            ),
        }
    )


def _model_slot_value_errors(
    *,
    model_slots: Mapping[str, Mapping[str, object]],
) -> list[str]:
    """Reject `None` values in model-level slots (masks are regime-level).

    Per-value grammar (grids, callables, law vocabulary, `Phased` placement)
    is validated when the merged regimes are constructed; only the
    merge-specific vocabulary is checked here.
    """
    errors: list[str] = []
    for slot_name, slot in model_slots.items():
        for name, value in slot.items():
            if value is None:
                errors.append(
                    f"Model-level {slot_name}['{name}'] cannot be `None` — "
                    f"masks are regime-level.",
                )
    return errors


def validate_model_slots(*, model_slots: Mapping[str, Mapping[str, object]]) -> None:
    """Raise on merge-specific vocabulary errors in model-level slots."""
    errors = _model_slot_value_errors(model_slots=model_slots)
    if errors:
        raise ModelInitializationError(format_messages(errors))
