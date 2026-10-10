"""Model initialization helpers: validation, template creation, fixed-param handling.

Extracted from `model.py` to keep the `Model` class focused on its public API.

"""

import dataclasses
import functools
import inspect
from collections.abc import Mapping
from types import MappingProxyType
from typing import cast

from dags import get_ancestors
from dags.tree import QNAME_DELIMITER, qname_from_tree_path
from jax import Array

from _lcm.constraints.bounds import lower_bound_declaration
from _lcm.constraints.processed import ConstraintLike, normalize_constraints
from _lcm.execution.execution_plan import ResolvedExecution
from _lcm.grids import DiscreteGrid
from _lcm.pandas_utils import convert_series_in_params
from _lcm.params.edges import (
    EDGES,
    edge_params,
    flat_namespaces,
    regime_kernel_params,
)
from _lcm.params.processing import (
    broadcast_to_template,
    cast_params_to_canonical_dtypes,
    create_params_template,
    materialize_granular_transition_params,
)
from _lcm.params.sequence_leaf import SequenceLeaf
from _lcm.processes.base import _ContinuousStochasticProcess
from _lcm.regime_building.age_normalization import (
    _regime_has_markers,
    normalize_age_specialization,
)
from _lcm.regime_building.age_specialization import resolve_node
from _lcm.regime_building.broadcast import (
    root_functions,
    states_read_through_their_draw,
)
from _lcm.regime_building.finalize import FinalizedUserRegime
from _lcm.regime_building.max_Q_over_a import TASTE_SHOCK_SCALE_PARAM
from _lcm.regime_building.phases import (
    normalize_all_regime_phases,
    phase_variation_paths,
)
from _lcm.regime_building.processing import (
    PreparedModelStructure,
    Regime,
    process_regimes,
)
from _lcm.regime_law import RegimeLaws
from _lcm.simulation.policy_programs import declare_finite_replay_programs
from _lcm.solution.contract import Solver, SolverModelContext
from _lcm.solution.shipped_solvers import fail_if_solver_is_not_shipped
from _lcm.time import TimeAxis, coordinate_kind, specialization_coordinate_at
from _lcm.typing import (
    EconFunctionKwargs,
    EGMCarryProducer,
    FlatParams,
    FlatRegimeParams,
    FunctionName,
    ParamsLeaf,
    ParamsTemplate,
    RegimeName,
    RegimeNamesToIds,
    RegimeParamsTemplateNode,
    RegimeTransitionFunction,
    StateName,
    VmappedRegimeTransitionFunction,
)
from _lcm.utils.containers import get_field_names_and_values
from _lcm.utils.error_messages import format_messages, path_segment_name_errors
from lcm.exceptions import InvalidParamsError, ModelInitializationError
from lcm.params import MappingLeaf
from lcm.phased import Phased
from lcm.regime import Regime as UserRegime
from lcm.transition import JointTransition, Transition
from lcm.typing import Phase, UserFunction, UserParams


def build_regimes_and_template(
    *,
    ages: TimeAxis,
    user_regimes: Mapping[RegimeName, FinalizedUserRegime],
    regime_names_to_ids: RegimeNamesToIds,
    enable_jit: bool,
    fixed_params: UserParams,
    params_already_consumed: frozenset[str],
    prepared_structure: PreparedModelStructure,
    phase_transitions: Mapping[Phase, Mapping[RegimeName, Transition]],
    execution: ResolvedExecution | None = None,
) -> tuple[MappingProxyType[RegimeName, Regime], ParamsTemplate]:
    """Build canonical regimes and params template in a single pass.

    Compose regime processing, template creation, and optional fixed-param partialling
    so that each result is computed exactly once.

    Args:
        ages: Age grid for the model.
        user_regimes: Mapping of regime names to finalized regimes.
        regime_names_to_ids: Immutable mapping from regime names to integer
            indices.
        enable_jit: Whether to JIT-compile regime functions.
        fixed_params: Parameters to fix at model initialization.
        params_already_consumed: Flat keys the process-law binder resolved and
            acted on. They are broadcasts, so they stay in `fixed_params` for
            the slots they may still serve; naming them here keeps a broadcast
            that served only a bound process from reading as an unknown key.
        execution: The hardware-local facts the model resolved, or `None` to
            resolve the inert configuration against every visible device.

    Returns:
        Tuple of (regimes, params_template).

    """
    if not fixed_params:
        regimes = process_regimes(
            ages=ages,
            user_regimes=user_regimes,
            regime_names_to_ids=regime_names_to_ids,
            enable_jit=enable_jit,
            prepared_structure=prepared_structure,
            execution=execution,
        )
        params_template = create_params_template(regimes)
    else:
        regimes, params_template = _build_regimes_and_template_with_fixed_params(
            ages=ages,
            user_regimes=user_regimes,
            regime_names_to_ids=regime_names_to_ids,
            enable_jit=enable_jit,
            fixed_params=fixed_params,
            params_already_consumed=params_already_consumed,
            prepared_structure=prepared_structure,
            phase_transitions=phase_transitions,
            execution=execution,
        )

    # Replay bodies must consume the canonical functions after fixed parameters
    # have been bound, exactly as the ordinary forward decision does.
    regimes = MappingProxyType(
        {
            name: declare_finite_replay_programs(regime)
            for name, regime in regimes.items()
        }
    )
    return regimes, params_template


def _build_regimes_and_template_with_fixed_params(
    *,
    ages: TimeAxis,
    user_regimes: Mapping[RegimeName, FinalizedUserRegime],
    regime_names_to_ids: RegimeNamesToIds,
    enable_jit: bool,
    fixed_params: UserParams,
    params_already_consumed: frozenset[str],
    prepared_structure: PreparedModelStructure,
    phase_transitions: Mapping[Phase, Mapping[RegimeName, Transition]],
    execution: ResolvedExecution | None = None,
) -> tuple[MappingProxyType[RegimeName, Regime], ParamsTemplate]:
    """Build canonical regimes and template, then partial in fixed params.

    Args:
        ages: Age grid for the model.
        user_regimes: Mapping of regime names to finalized regimes.
        regime_names_to_ids: Immutable mapping from regime names to integer
            indices.
        enable_jit: Whether to JIT-compile regime functions.
        fixed_params: Parameters to fix at model initialization.
        params_already_consumed: Flat keys the process-law binder resolved and
            acted on.
        execution: The hardware-local facts the model resolved, or `None` to
            resolve the inert configuration against every visible device.

    Returns:
        Tuple of regimes and params_template with fixed params
        partialled in.

    """
    raw_regimes = process_regimes(
        ages=ages,
        user_regimes=user_regimes,
        regime_names_to_ids=regime_names_to_ids,
        enable_jit=enable_jit,
        prepared_structure=prepared_structure,
        execution=execution,
    )
    raw_params_template = create_params_template(raw_regimes)

    fixed_flat_params = _resolve_fixed_params(
        fixed_params=dict(fixed_params),
        template=raw_params_template,
        already_consumed=params_already_consumed,
    )
    fixed_flat_params = convert_series_in_params(
        flat_params=fixed_flat_params,
        ages=ages,
        user_regimes=user_regimes,
        laws=prepared_structure.laws,
        regime_names_to_ids=regime_names_to_ids,
        declared_transitions=prepared_structure.declared_transitions,
        phase_transitions=phase_transitions,
        declared_vocabulary=prepared_structure.declared_edge_vocabulary,
        reachability=prepared_structure.reachability,
        required_periods_by_regime={
            name: tuple(
                sorted(
                    set(regime.active_periods)
                    | {
                        period
                        for period, active in enumerate(
                            regime.simulation.reachability.active_regimes_by_period
                        )
                        if name in active
                    }
                )
            )
            for name, regime in raw_regimes.items()
        },
    )
    fixed_flat_params = cast_params_to_canonical_dtypes(fixed_flat_params)
    _validate_param_types(fixed_flat_params)

    # The template trim works on the template-shaped (user-coarse) form;
    # partialling needs the granular form the compiled functions bind.
    granular_fixed_flat_params = materialize_granular_transition_params(
        flat_params=fixed_flat_params,
        expansions={
            regime_name: regime.granular_param_expansions
            for regime_name, regime in raw_regimes.items()
        },
    )

    return (
        _partial_fixed_params_into_regimes(
            raw_regimes=raw_regimes,
            fixed_flat_params=granular_fixed_flat_params,
        ),
        _remove_fixed_params_from_template(
            template=raw_params_template,
            fixed_flat_params=fixed_flat_params,
        ),
    )


def validate_model_inputs(
    *,
    user_regimes: Mapping[RegimeName, UserRegime],
    laws: RegimeLaws,
    regime_id_class: type,
    broadcast_variables: Mapping[RegimeName, frozenset[str]],
    ages: TimeAxis,
    active_periods_by_regime: Mapping[RegimeName, tuple[int, ...]],
    visited_periods_by_regime: Mapping[RegimeName, tuple[int, ...]] | None = None,
    removed_edge_reads: Mapping[
        RegimeName, Mapping[str, tuple[RegimeName, ...]]
    ] = MappingProxyType({}),
) -> None:
    """Validate model constructor inputs.

    `regimes` is typed via beartype on `Model.__init__` and reaches this
    function with its declared type. This function focuses on value and
    cross-field rules.

    `ages` lets the used-variable check resolve `AgeSpecializedFunction` functions at
    each regime's representative age, so a state read only by a policy-specialized
    function still counts as used. A marker only the simulate phase reads is
    resolved at the first of `visited_periods_by_regime`, where a subject can be;
    they default to the active periods.

    `removed_edge_reads` name, per regime, the variables read by declarations
    that fixed-zero pruning removed with their edges, and those edges' targets.
    Such a variable is unused like any other; the names only explain why.
    """

    # DC-EGM contract checks run before the generic checks below: a contract
    # violation (e.g. a missing resources function) typically also leaves
    # variables unused, and the contract-specific message is the actionable
    # one.
    #
    # They read the *representative* regimes, in which every `AgeSpecializedGrid`
    # state is already the concrete representative-age grid. The solver contract is
    # about a state's kind and shape, both invariant across ages by the
    # `AgeSpecializedGrid` contract, so the representative grid answers it exactly —
    # whereas the raw marker is not a `Grid` at all and would be dropped from every
    # type-filtered collection of continuous states, rejecting a valid model.
    solver_validation_regimes = _representative_for_validation(
        user_regimes=user_regimes,
        laws=laws,
        ages=ages,
        active_periods_by_regime=active_periods_by_regime,
        visited_periods_by_regime=visited_periods_by_regime,
    )
    solver_validation_phase_specs = normalize_all_regime_phases(
        user_regimes=solver_validation_regimes, laws=laws
    )
    for regime_name, user_regime in solver_validation_regimes.items():
        fail_if_solver_is_not_shipped(
            solver=user_regime.solver, regime_name=regime_name
        )
        user_regime.solver.validate_model(
            context=SolverModelContext(
                regime_name=regime_name,
                user_regimes=solver_validation_regimes,
                laws=laws,
                solve_functions=solver_validation_phase_specs[
                    regime_name
                ].solution.functions,
                phase_variation_paths=phase_variation_paths(
                    user_regime=user_regime, law=laws[regime_name]
                ),
            )
        )

    error_messages = _reserved_age_errors(
        user_regimes=solver_validation_regimes, laws=laws, ages=ages
    )

    if not user_regimes:
        error_messages.append("At least one terminal regime must be provided.")

    error_messages.extend(path_segment_name_errors(kind="Regime", names=user_regimes))

    # Assume all items in regimes are lcm.Regime instances beyond this point
    terminal_regimes = [name for name in user_regimes if laws[name].terminal]
    if len(terminal_regimes) < 1:
        error_messages.append("lcm.Model must have at least one terminal regime.")

    regime_id_fields = sorted(get_field_names_and_values(regime_id_class).keys())
    regime_names = sorted(user_regimes.keys())
    if regime_id_fields != regime_names:
        error_messages.append(
            f"regime_id_cls fields must match regime names.\nGot:\n"
            "regime_id_cls fields:\n"
            f"    {regime_id_fields}\n"
            "regime names:\n"
            f"    {regime_names}."
        )
    error_messages.extend(
        _validate_all_variables_used(
            user_regimes=user_regimes,
            laws=laws,
            broadcast_variables=broadcast_variables,
            ages=ages,
            active_periods_by_regime=active_periods_by_regime,
            visited_periods_by_regime=visited_periods_by_regime,
            removed_edge_reads=removed_edge_reads,
        )
    )
    error_messages.extend(
        _validate_constraint_phase_invariance(
            user_regimes=user_regimes,
            laws=laws,
            ages=ages,
            active_periods_by_regime=active_periods_by_regime,
        )
    )

    for name, user_regime in user_regimes.items():
        if user_regime.taste_shocks is not None and not any(
            isinstance(grid, DiscreteGrid) for grid in user_regime.actions.values()
        ):
            error_messages.append(
                f"Regime '{name}' declares taste_shocks but has no discrete "
                f"action. EV1 taste shocks are drawn per discrete-action "
                f"combination, so at least one discrete action is required."
            )

    if error_messages:
        msg = format_messages(error_messages)
        raise ModelInitializationError(msg)


def _reserved_age_errors(
    *, user_regimes: Mapping[RegimeName, UserRegime], laws: RegimeLaws, ages: TimeAxis
) -> list[str]:
    """Prevent unresolved ages from becoming runtime parameters in period mode."""
    errors: list[str] = []
    if coordinate_kind(ages) != "period":
        return errors
    for regime_name, regime in user_regimes.items():
        for phase in ("solve", "simulate"):
            functions = dict(
                regime.get_all_functions(phase=phase, law=laws[regime_name])
            )
            functions.update(
                root_functions(
                    regime_name=regime_name, regime=regime, laws=laws, phase=phase
                )
            )
            for target, kernels in regime.joint_transitions.items():
                for name, raw in kernels.items():
                    joint = cast(
                        "JointTransition",
                        (raw.solve if phase == "solve" else raw.simulate)
                        if isinstance(raw, Phased)
                        else raw,
                    )
                    if callable(joint.support):
                        functions[f"__joint_support__{target}__{name}"] = joint.support
            if "age" in functions:
                continue
            consumers = sorted(
                name
                for name, func in functions.items()
                if "age" in inspect.signature(func).parameters
            )
            if consumers:
                errors.append(
                    f"Period model regime {regime_name!r} has an unresolved age "
                    f"dependency in {phase} functions {consumers}. Supply ages "
                    "or define a separately named biological_age from explicit inputs."
                )
    return errors


def _representative_for_validation(
    *,
    user_regimes: Mapping[RegimeName, UserRegime],
    laws: RegimeLaws,
    ages: TimeAxis,
    active_periods_by_regime: Mapping[RegimeName, tuple[int, ...]],
    visited_periods_by_regime: Mapping[RegimeName, tuple[int, ...]] | None = None,
) -> Mapping[RegimeName, UserRegime]:
    """Resolve age markers to their representatives, for solver-contract validation.

    Runs the same normalization boundary the build runs — phases first (local to one
    regime), then age specialization (needs the model `AgeGrid` and each regime's
    active periods) — and returns its representative regimes, which are the declared
    input to age-invariant validation.

    Returns the input unchanged when no regime carries a marker, so an age-invariant
    model neither pays for the walk nor changes behaviour.
    """
    if not any(_regime_has_markers(regime) for regime in user_regimes.values()):
        return user_regimes
    phased_specs = normalize_all_regime_phases(user_regimes=user_regimes, laws=laws)
    return normalize_age_specialization(
        user_regimes=user_regimes,
        phased_specs=phased_specs,
        ages=ages,
        active_periods_by_regime=active_periods_by_regime,
        visited_periods_by_regime=visited_periods_by_regime,
    ).representative_user_regimes


def _model_wide_conditioning_names(
    user_regimes: Mapping[RegimeName, UserRegime],
) -> frozenset[StateName]:
    """Every conditioning state read by any state-conditioned process in the model.

    `state_conditioned.on` is a real dependency of the generated weights/draw functions,
    but it lives in grid metadata rather than in a user function, so a callable-DAG scan
    misses it. A conditioned process's transition weight is also built into the Q of
    every *source* regime that can reach the process's regime, evaluated at that
    source's current `on` state — so the dependency is not local to the process's regime
    We collect the conditioners model-wide and, at the call site,
    credit each regime for those it actually carries: a conservative over-approximation
    of reachability whose only cost is not flagging a genuinely unused state, never a
    wrong policy.
    """
    return frozenset(
        grid.state_conditioned.on
        for user_regime in user_regimes.values()
        for grid in user_regime.states.values()
        if isinstance(grid, _ContinuousStochasticProcess)
        and grid.state_conditioned is not None
    )


def _validate_all_variables_used(
    *,
    user_regimes: Mapping[RegimeName, UserRegime],
    laws: RegimeLaws,
    broadcast_variables: Mapping[RegimeName, frozenset[str]],
    ages: TimeAxis,
    active_periods_by_regime: Mapping[RegimeName, tuple[int, ...]],
    visited_periods_by_regime: Mapping[RegimeName, tuple[int, ...]] | None = None,
    removed_edge_reads: Mapping[
        RegimeName, Mapping[str, tuple[RegimeName, ...]]
    ] = MappingProxyType({}),
) -> list[str]:
    """Validate that all states and actions are used somewhere in each regime.

    Each state or action must be read by one of the regime's root computations
    (`root_functions`) or by a law of motion that is not the identity. That
    covers, per regime:

    - the concurrent valuation — utility, or a collective regime's
      per-stakeholder utilities, and the constraints;
    - a derived-categorical function;
    - the Koopmans aggregator, directly or through a regime function whose
      output it consumes at the Bellman step;
    - the regime transition;
    - a value-constraint predicate or a `same_period_refs` projection;
    - the gate, gate references or fallback projections of a gated edge whose
      target is this regime — those are declared on the source regime but
      evaluated here, so this regime is where the state they read must live;
    - a law of motion, unless it hands the state to itself;
    - for a process state, any of the above reading its next-period draw
      `next_<state>`, which is taken from the state.

    A declaration removed with a fixed-zero edge is not a use: pruning removes
    as much as it can up front, so a variable read only across such an edge is
    unused, exactly as in the model declared without that edge.

    Broadcast variables are exempt: DAG pruning already weeded the unused
    ones, and a retained broadcast variable may be used only through a law
    of motion toward a candidate target (which this per-regime check cannot
    see).

    Args:
        user_regimes: Mapping of regime names to user-provided `Regime`
            instances.
        laws: Each regime's law, bound from `Model(edges=...)`, by regime name.
        broadcast_variables: Per regime, the model-level broadcast state and
            action names to exempt.
        removed_edge_reads: Per regime, each state or action read by a
            declaration removed with its fixed-zero edge, and the targets of
            those edges; named in the error to explain why a variable is unused.

    Returns:
        A list of error messages. Empty list if validation passes.

    """
    error_messages = []
    conditioning_names = _model_wide_conditioning_names(user_regimes)

    for regime_name, user_regime in user_regimes.items():
        variable_names = set(user_regime.states) | set(user_regime.actions)
        variable_names -= broadcast_variables.get(regime_name, frozenset())
        user_functions = dict(
            user_regime.get_all_functions(phase="solve", law=laws[regime_name])
        )
        # `root_functions` is the single definition of what a root computation
        # is, shared with the broadcast pruning walk so the two cannot disagree
        # about what counts as a read. It also supplies the reads no per-regime
        # walk can see: a gated edge declared on another regime whose gate,
        # gate references and fallbacks are evaluated on *this* regime's grid.
        solve_roots = root_functions(
            regime_name=regime_name, regime=user_regime, laws=laws, phase="solve"
        )
        simulate_roots = root_functions(
            regime_name=regime_name, regime=user_regime, laws=laws, phase="simulate"
        )
        # A `Phased` slot may consume a variable in only one phase, and the
        # variable is used either way, so a simulate variant that is a different
        # object joins the pool under its own key.
        roots: dict[FunctionName, UserFunction] = dict(solve_roots) | {
            f"{key}__simulate": func
            for key, func in simulate_roots.items()
            if solve_roots.get(key) is not func
        }
        user_functions |= roots
        active_periods = active_periods_by_regime.get(regime_name, ())
        if not active_periods and _regime_has_markers(user_regime):
            # Without coverage the markers cannot be resolved, and
            # `get_ancestors` would see only `AgeSpecializedFunction.__call__`'s
            # generic `(*args, **kwargs)` signature, misreporting a variable
            # used only through a marker as unused — so skip this regime's
            # variable-usage check.
            continue
        if active_periods:
            # Resolve any `AgeSpecializedFunction` marker to its concrete
            # function at a representative active age so `get_ancestors` sees
            # the real argument dependencies. The dependency structure is
            # age-invariant, so any active age serves; a stateful factory
            # could in principle be validated as one object and installed as
            # another (see the `_AgeSpecialized` docstring), so this relies on
            # `build` being pure — its result is not cached or reused
            # elsewhere.
            # A simulate-only root resolves where the regime is simulated.
            visited_periods = (
                active_periods
                if visited_periods_by_regime is None
                else visited_periods_by_regime.get(regime_name, ())
            ) or active_periods
            representative_age = specialization_coordinate_at(
                ages=ages, period=active_periods[0]
            )
            simulated_age = specialization_coordinate_at(
                ages=ages, period=visited_periods[0]
            )
            user_functions = cast(
                "dict[FunctionName, UserFunction]",
                {
                    name: resolve_node(
                        node=func,
                        age=simulated_age
                        if name.endswith("__simulate")
                        else representative_age,
                    )
                    for name, func in user_functions.items()
                },
            )

        targets = [
            *roots,
            # A law of motion is a use of what it reads, except when it is the
            # identity: handing a state to itself says nothing about the state
            # being needed. This is where the two consumers of `root_functions`
            # part ways — the pruning walk roots the identity hand-over, because
            # a target that keeps the state has to be given its value.
            *(
                name
                for name in user_functions
                if name.startswith("next_")
                and not getattr(user_functions[name], "_is_auto_identity", False)
            ),
        ]
        reachable = get_ancestors(
            user_functions, targets=targets, include_targets=False
        )
        # A state-conditioned process reads `state_conditioned.on`: the generated solve
        # weights and the simulation draw both take it as an argument. But it is
        # declared as grid *metadata* rather than in a user function, so the ancestry
        # above cannot see it. A process conditioned in one regime is also drawn into a
        # *source* regime's Q when that source can reach it, evaluated at the source's
        # own `on` state — so a conditioner is credited to every regime that carries it,
        # not only the process's own regime.
        reachable = set(reachable) | (conditioning_names & variable_names)
        # A process state has no `next_<state>` function node, so a computation
        # that reads its next-period draw ends the walk at a leaf. The draw is
        # taken from the state, so reading it is a use of the state.
        reachable |= states_read_through_their_draw(regime=user_regime, reads=reachable)
        unused_variables = sorted(variable_names - reachable)

        if unused_variables:
            unused_states = [v for v in unused_variables if v in user_regime.states]
            unused_actions = [v for v in unused_variables if v in user_regime.actions]

            msg_parts = []
            if unused_states:
                state_word = "state" if len(unused_states) == 1 else "states"
                msg_parts.append(f"{state_word} {unused_states}")
            if unused_actions:
                action_word = "action" if len(unused_actions) == 1 else "actions"
                msg_parts.append(f"{action_word} {unused_actions}")

            error_messages.append(
                f"The following variables are defined but never used in regime "
                f"'{regime_name}': {' and '.join(msg_parts)}. "
                f"Each state and action must be used in at least one of: "
                f"utility, constraints, or transition functions."
                + _removed_edge_explanation(
                    regime_name=regime_name,
                    unused=unused_variables,
                    removed_edge_reads=removed_edge_reads.get(regime_name, {}),
                )
            )

    return error_messages


def _removed_edge_explanation(
    *,
    regime_name: RegimeName,
    unused: list[str],
    removed_edge_reads: Mapping[str, tuple[RegimeName, ...]],
) -> str:
    """Name the removed fixed-zero edges an unused variable was read across."""
    return "".join(
        f" '{name}' is read only across the edge(s) "
        + ", ".join(f"'{regime_name}' -> '{target}'" for target in targets)
        + ", removed during construction because their probability is fixed "
        "at zero."
        for name in unused
        if (targets := removed_edge_reads.get(name))
    )


def _law_phase_varies(*, solve_obj: UserFunction, sim_obj: UserFunction | None) -> bool:
    """Whether a name's `solve` and `simulate` resolutions are different laws.

    Object identity is the test: `get_all_functions` returns the raw user
    callables, so a phase-invariant value is one object in both phases while a
    `Phased` yields two distinct ones. `fixed_transition` is the exception — its
    identity law is rebuilt on every collection, so the two phases hold distinct
    objects standing for the same law. Treating those as different would falsely
    reject a constraint that reads a fixed `next_<state>`.
    """
    if solve_obj is sim_obj:
        return False
    if getattr(solve_obj, "_is_auto_identity", False) and getattr(
        sim_obj, "_is_auto_identity", False
    ):
        return getattr(solve_obj, "_state_name", None) != getattr(
            sim_obj, "_state_name", object()
        )
    return True


def _post_decision_function_of_solver(solver: Solver) -> str | None:
    """Return the bound liquid post-decision role of an EGM-family solver."""
    current: Solver | None = solver
    while current is not None:
        post_decision = getattr(current, "post_decision_function", None)
        if isinstance(post_decision, str):
            return post_decision
        inner = getattr(current, "inner", None)
        if inner is current:
            break
        current = inner
    return None


def _is_solve_proved_post_decision_lower_bound(
    *, constraint_name: str, user_regime: UserRegime
) -> bool:
    """Whether the solve grid proves this exact structural lower bound.

    This is the one supported phase-resolved feasibility declaration: its solve
    disposition is a grid proof, while simulation evaluates the declaration
    against the simulation function pool.
    """
    post_decision = _post_decision_function_of_solver(user_regime.solver)
    declaration = user_regime.decomposed_constraints[constraint_name]
    if post_decision is None or declaration is None:
        return False
    processed = normalize_constraints(
        constraints={constraint_name: cast("ConstraintLike", declaration)}
    )[constraint_name]
    bound = lower_bound_declaration(constraint=processed)
    return bound is not None and bound[0] == post_decision


def _validate_constraint_phase_invariance(
    *,
    user_regimes: Mapping[RegimeName, UserRegime],
    laws: RegimeLaws,
    ages: TimeAxis,
    active_periods_by_regime: Mapping[RegimeName, tuple[int, ...]],
) -> list[str]:
    """Reject a constraint whose dependency ancestry contains a phase-varying node.

    The feasible set is a primitive of the model the agent solved, so it may not
    differ across phases: a phase-specific feasible set would let the simulated
    agent choose actions its value function was never computed for. A `Phased`
    constraint is already rejected when the regime is built, but a plain constraint
    can reach a `Phased` helper or law of motion further up its dependency chain
    and become phase-specific that way. This walks the whole chain.

    A name is phase-varying when its solve and simulate resolutions are different
    laws (`_law_phase_varies`). A structural lower bound on an EGM-family
    post-decision state is the deliberate exception: the solve grid proves it,
    while simulation evaluates it against the phase-resolved function pool.
    Two other cases need care:

    - A per-target law is keyed `next_<state>__<target>`, while a constraint reads
      the unqualified `next_<state>`. If any target's law varies by phase, so does
      the unqualified name, so the qualified entries are aliased onto it.
    - A carried state is not phase-varying: both phases read the same solve-phase
      imputation, which is also the value its decision is taken at. Reading such a
      state's *next* value is a different matter — the solve phase has no producer
      for it — and is rejected separately.

    Args:
        user_regimes: Mapping of finalized regime names to `Regime` instances.
        laws: Each regime's law, bound from `Model(edges=...)`, by regime name.
        ages: The model's age grid.
        active_periods_by_regime: Immutable mapping of regime names to their
            active periods, as resolved from the declarations.

    Returns:
        A list of error messages. Empty list if validation passes.

    """
    error_messages = []
    for regime_name, user_regime in user_regimes.items():
        law = laws[regime_name]
        solve_funcs = dict(user_regime.get_all_functions(phase="solve", law=law))
        sim_funcs = user_regime.get_all_functions(phase="simulate", law=law)
        phase_varying = frozenset(
            name
            for name in solve_funcs
            if _law_phase_varies(
                solve_obj=solve_funcs[name], sim_obj=sim_funcs.get(name)
            )
        )
        # Alias each phase-varying per-target law onto the unqualified
        # `next_<state>` a constraint actually reads.
        phase_varying = phase_varying | frozenset(
            name.rsplit(QNAME_DELIMITER, 1)[0]
            for name in phase_varying
            if QNAME_DELIMITER in name
        )
        # The solve phase imputes a carried state, so its *next* value has no
        # producer there. A constraint reading `next_<carried>` would fail deep in
        # the solve build with an unsupplied argument; reject it here instead.
        # Reading the carried state's current value is fine.
        carried_next = frozenset(
            f"next_{name}"
            for name, spec in user_regime.states.items()
            if isinstance(spec, Phased)
        )
        if not phase_varying and not carried_next:
            continue
        # Resolve every age-specialized function to a concrete per-age function
        # before walking constraint ancestry (as `_validate_all_variables_used`
        # does). Unresolved, such a function exposes only a generic
        # `(*args, **kwargs)` signature, so the ancestry stops there and a
        # phase-varying helper reached below it would escape this check. The
        # dependency structure is age-invariant, so any active age serves.
        ancestry_funcs = solve_funcs
        # Read the prepared coverage rather than recomputing it: the schedules
        # resolved once at model construction are the single canonical source.
        active_periods = active_periods_by_regime.get(regime_name, ())
        if active_periods:
            representative_age = specialization_coordinate_at(
                ages=ages, period=active_periods[0]
            )
            ancestry_funcs = cast(
                "dict[FunctionName, UserFunction]",
                {
                    name: resolve_node(node=func, age=representative_age)
                    for name, func in solve_funcs.items()
                },
            )

        for constraint_name in user_regime.decomposed_constraints:
            ancestors = get_ancestors(
                ancestry_funcs, targets=[constraint_name], include_targets=False
            )
            offending = sorted(ancestors & phase_varying)
            if offending and not _is_solve_proved_post_decision_lower_bound(
                constraint_name=constraint_name, user_regime=user_regime
            ):
                error_messages.append(
                    f"Constraint '{constraint_name}' in regime '{regime_name}' "
                    f"depends on phase-varying function(s) {offending}. "
                    f"Constraints must be phase-invariant through their whole "
                    f"dependency chain: a phase-specific feasible set would let "
                    f"the simulated agent choose actions its value function was "
                    f"never computed for. Make the constraint's dependencies "
                    f"phase-invariant, or keep the phase variance out of the "
                    f"feasibility path."
                )
            offending_carried = sorted(ancestors & carried_next)
            if offending_carried:
                error_messages.append(
                    f"Constraint '{constraint_name}' in regime '{regime_name}' "
                    f"reads the next value of a carried state {offending_carried}. "
                    f"A carried state is imputed in the solve phase, so its next "
                    f"value has no solve-phase producer and the solve feasibility "
                    f"DAG would be left with an unsupplied argument. Read the "
                    f"carried state's current value instead, or make it an "
                    f"ordinary (non-carried) state."
                )
    return error_messages


def _resolve_fixed_params(
    *,
    fixed_params: UserParams,
    template: ParamsTemplate,
    already_consumed: frozenset[str],
) -> FlatParams:
    """Resolve fixed_params against the params template.

    Like `process_params`, support model/regime/function level specification, but
    do NOT require all template keys to be present — only match what's provided.

    Args:
        fixed_params: Parameters fixed at model initialization.
        template: The params template to resolve against.
        already_consumed: Flat keys the process-law binder acted on, which the
            template no longer holds a slot for.

    Returns:
        The resolved flat params.

    """
    return broadcast_to_template(
        params=fixed_params,
        template=template,
        required=False,
        already_consumed=already_consumed,
    )


def _remove_fixed_params_from_template(
    *,
    template: ParamsTemplate,
    fixed_flat_params: FlatParams,
) -> ParamsTemplate:
    """Remove fixed params from the params template.

    After partialling fixed params into compiled functions, remove them from the
    template so users don't need to supply them at solve/simulate time.

    """

    # Template subtrees: `_trim_fixed_params` copies nodes of any depth.
    trimmed: dict[RegimeName, MappingProxyType[str, RegimeParamsTemplateNode]] = {
        regime_name: MappingProxyType(
            _trim_fixed_params(
                branch=regime_template,
                prefix=(),
                fixed=regime_kernel_params(fixed_flat_params, regime_name=regime_name),
            )
        )
        for regime_name, regime_template in template.items()
        if regime_name != EDGES
    }
    # The edge branch keeps only the sources with a slot left to supply.
    edge_branch = {
        source: MappingProxyType(trimmed_source)
        for source, source_template in template.get(EDGES, {}).items()
        if (
            trimmed_source := _trim_fixed_params(
                branch=source_template,
                prefix=(),
                fixed=edge_params(fixed_flat_params, source=source),
            )
        )
    }
    if edge_branch:
        trimmed[EDGES] = MappingProxyType(edge_branch)
    return cast("ParamsTemplate", MappingProxyType(trimmed))


def _trim_fixed_params(
    *,
    branch: Mapping[str, RegimeParamsTemplateNode],
    prefix: tuple[str, ...],
    fixed: FlatRegimeParams,
) -> dict[str, RegimeParamsTemplateNode]:
    """Copy `branch` without the leaves whose qualified name is in `fixed`."""
    trimmed: dict[str, RegimeParamsTemplateNode] = {}
    for key, value in branch.items():
        if isinstance(value, Mapping):
            inner = _trim_fixed_params(
                branch=value,
                prefix=(*prefix, key),
                fixed=fixed,
            )
            if inner:
                trimmed[key] = MappingProxyType(inner)
        elif qname_from_tree_path((*prefix, key)) not in fixed:
            trimmed[key] = value
    return trimmed


def _partial_fixed_params_into_regimes(
    *,
    raw_regimes: MappingProxyType[RegimeName, Regime],
    fixed_flat_params: FlatParams,
) -> MappingProxyType[RegimeName, Regime]:
    """Partial fixed params into all compiled functions on each Regime."""
    result: dict[RegimeName, Regime] = {}
    for regime_name, regime in raw_regimes.items():
        regime_fixed = dict(
            regime_kernel_params(fixed_flat_params, regime_name=regime_name)
        )
        # A DC-EGM source carrying into a *different* target regime also binds
        # that target's fixed params (it reads the target's resources /
        # transition functions in its per-asset-node solve). Gate the rebuild on
        # whether any fixed param reachable from this regime — its own or a
        # transition target's — exists; the per-adapter `with_fixed_params`
        # decides which of them actually reach each core.
        reachable_fixed = bool(regime_fixed) or any(
            regime_kernel_params(fixed_flat_params, regime_name=target_name)
            for target_name in regime.solution.transitions
        )
        if not reachable_fixed:
            result[regime_name] = regime
            continue

        # Build new solution phase with partialled functions. The resolved
        # fixed params also land on the phase itself — its
        # `state_action_space` consults them for runtime grid substitution.
        #
        # Each period adapter owns its solver's binding rule: a grid-search
        # adapter binds the regime's own fixed params into its core; a DC-EGM
        # adapter binds the union of the regime's and its carry targets' fixed
        # params (a source reads a different target's params in its per-asset
        # solve); a terminal carry-producing adapter binds the regime's fixed
        # params into both its base core and the carry producer. So the engine
        # threads fixed params through `with_fixed_params` without a solver-type
        # switch.
        solution = regime.solution
        new_solve = dataclasses.replace(
            solution,
            resolved_fixed_params=MappingProxyType(regime_fixed),
            period_kernels=MappingProxyType(
                {
                    period: kernel.with_fixed_params(
                        fixed_flat_params=fixed_flat_params
                    )
                    for period, kernel in solution.period_kernels.items()
                }
            ),
            compute_regime_transition_probs=(
                functools.partial(
                    solution.compute_regime_transition_probs,
                    **_filter_kwargs_for_func(
                        func=solution.compute_regime_transition_probs,
                        kwargs=regime_fixed,
                    ),
                )
                if solution.compute_regime_transition_probs is not None
                else None
            ),
            validation_regime_transition_probs=(
                functools.partial(
                    solution.validation_regime_transition_probs,
                    **_filter_kwargs_for_func(
                        func=solution.validation_regime_transition_probs,
                        kwargs=regime_fixed,
                    ),
                )
                if solution.validation_regime_transition_probs is not None
                else None
            ),
        )

        # Build new simulation phase with partialled functions
        simulation = regime.simulation
        new_simulate = dataclasses.replace(
            simulation,
            programs=dataclasses.replace(
                simulation.programs,
                **{
                    family: MappingProxyType(
                        {
                            period: dataclasses.replace(
                                program,
                                function=functools.partial(
                                    program.function, **regime_fixed
                                ),
                            )
                            for period, program in getattr(
                                simulation.programs, family
                            ).items()
                        }
                    )
                    for family in (
                        "decision",
                        "type_local_decision",
                        "action_values",
                        "transition",
                        "route",
                    )
                },
            ),
            Q_and_F=MappingProxyType(
                {
                    period: functools.partial(func, **regime_fixed)
                    for period, func in simulation.Q_and_F.items()
                }
            ),
            next_state=MappingProxyType(
                {
                    period: functools.partial(func, **regime_fixed)
                    for period, func in simulation.next_state.items()
                }
            ),
            compute_regime_transition_probs=(
                functools.partial(
                    simulation.compute_regime_transition_probs,
                    **_filter_kwargs_for_func(
                        func=simulation.compute_regime_transition_probs,
                        kwargs=regime_fixed,
                    ),
                )
                if simulation.compute_regime_transition_probs is not None
                else None
            ),
            validation_regime_transition_probs=(
                functools.partial(
                    simulation.validation_regime_transition_probs,
                    **_filter_kwargs_for_func(
                        func=simulation.validation_regime_transition_probs,
                        kwargs=regime_fixed,
                    ),
                )
                if simulation.validation_regime_transition_probs is not None
                else None
            ),
        )

        result[regime_name] = dataclasses.replace(
            regime,
            solution=new_solve,
            simulation=new_simulate,
            resolved_fixed_params=MappingProxyType(regime_fixed),
        )
    return MappingProxyType(result)


def _filter_kwargs_for_func(
    *,
    func: RegimeTransitionFunction | VmappedRegimeTransitionFunction | EGMCarryProducer,
    kwargs: EconFunctionKwargs,
) -> EconFunctionKwargs:
    """Filter kwargs to only those accepted by func's signature."""
    try:
        sig = inspect.signature(func)
    except ValueError, TypeError:
        # If we can't inspect the signature, pass all kwargs through
        return kwargs
    params = sig.parameters
    # If the function accepts **kwargs, pass everything
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in params}


def _validate_param_types(flat_params: FlatParams) -> None:
    """Raise if any param leaf is not a JAX `Array` or container leaf.

    Defense-in-depth check after `cast_params_to_canonical_dtypes`: by the
    time this runs, every leaf must be a JAX `Array`, or a `MappingLeaf` /
    `SequenceLeaf` whose contents recursively satisfy the same rule.
    """
    for path, regime_params in flat_namespaces(flat_params):
        for key, value in regime_params.items():
            _check_leaf(value=value, path=qname_from_tree_path((*path, key)))


def fail_if_nonpositive_taste_shock_scale(flat_params: FlatParams) -> None:
    """Raise if any regime's taste-shock scale is not strictly positive.

    Declaring taste shocks means opting into smoothing, so the scale must be
    positive. The hard maximum is the no-taste-shocks model, reached by not
    declaring taste shocks — not by `scale = 0`.
    """
    for regime_name, regime_params in flat_params.items():
        if regime_name == EDGES:
            continue
        scale = cast("FlatRegimeParams", regime_params).get(TASTE_SHOCK_SCALE_PARAM)
        if isinstance(scale, Array) and float(scale) <= 0:
            msg = (
                f"The taste-shock scale of regime {regime_name!r} is "
                f"{float(scale)}, but it must be strictly positive."
            )
            raise InvalidParamsError(msg)


def _check_leaf(*, value: ParamsLeaf, path: str) -> None:
    """Check a single leaf, recursing into `MappingLeaf` / `SequenceLeaf`."""
    if isinstance(value, MappingLeaf):
        for k, v in value.data.items():
            _check_leaf(value=v, path=f"{path}.{k}")
        return
    if isinstance(value, SequenceLeaf):
        for i, v in enumerate(value.data):
            _check_leaf(value=v, path=f"{path}[{i}]")
        return
    if isinstance(value, Array):
        return
    type_name = type(value).__module__ + "." + type(value).__name__
    msg = f"Parameter {path!r} is a {type_name}, expected a JAX Array."
    raise InvalidParamsError(msg)
