import functools
import inspect
import math
from collections import defaultdict
from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass
from dataclasses import replace as dataclass_replace
from types import MappingProxyType
from typing import Any, Literal, cast

import jax
import numpy as np
from dags import concatenate_functions, get_annotations, with_signature
from dags.signature import rename_arguments
from dags.tree import QNAME_DELIMITER, qname_from_tree_path, tree_path_from_qname
from jax import numpy as jnp

from _lcm.certainty_equivalent import CertaintyEquivalent
from _lcm.coarse_transition import _CoarseTransitionCell
from _lcm.egm.budget import (
    DCEGM_BUDGET_CONSTRAINT_NAME,
    get_intrinsic_budget_constraint,
)
from _lcm.egm.carry import EGMCarry, build_template_egm_carry, shard_carry_template
from _lcm.egm.negm_validation import validate_negm_regimes
from _lcm.egm.terminal import (
    N_STATELESS_CARRY_ROWS,
    get_brute_child_carry_producer,
    get_stateless_terminal_carry_producer,
    get_terminal_wealth_carry_producer,
)
from _lcm.egm.validation import (
    savings_stage_reads_euler_state,
    validate_dcegm_regimes,
)
from _lcm.engine import (
    EGMPolicyRead,
    Regime,
    SimulationPhase,
    SolutionPhase,
    StateActionSpace,
    Variables,
)
from _lcm.grids import (
    DiscreteGrid,
    Grid,
    IrregSpacedGrid,
)
from _lcm.grids.continuous import ContinuousGrid
from _lcm.grids.coordinates import get_irreg_coordinate
from _lcm.identity_transition import _IdentityTransition
from _lcm.params.processing import get_flat_param_names
from _lcm.params.regime_template import create_regime_params_template
from _lcm.processes import _ContinuousStochasticProcess
from _lcm.processes.ar1 import TauchenAR1Process
from _lcm.processes.iid import NormalIIDProcess
from _lcm.processes.state_conditioned import (
    Family,
    StateConditioned,
    conditioned_row,
    gather_sigma,
    sigma_array_by_code,
)
from _lcm.regime_building.canonicalize import canonicalize_regimes
from _lcm.regime_building.diagnostics import _build_compute_intermediates_per_period
from _lcm.regime_building.finalize import FinalizedUserRegime
from _lcm.regime_building.max_Q_over_a import (
    get_argmax_and_max_Q_over_a,
)
from _lcm.regime_building.ndimage import map_coordinates
from _lcm.regime_building.next_state import get_next_state_function_for_simulation
from _lcm.regime_building.phases import (
    PhasedRegimeSpec,
    RegimePhaseSpec,
)
from _lcm.regime_building.Q_and_F import (
    get_period_targets,
    get_Q_and_F,
    get_Q_and_F_terminal,
)
from _lcm.regime_building.stochastic_state_transitions import (
    collect_stochastic_state_transitions,
)
from _lcm.regime_building.V import VInterpolationInfo, create_v_interpolation_info
from _lcm.solution.contract import (
    ContinuationPayload,
    KernelResult,
    PeriodKernel,
    SolverBuildContext,
)
from _lcm.state_action_space import create_state_action_space
from _lcm.typing import (
    ArgmaxQOverAFunction,
    ConstraintFunctionsMapping,
    EconFunction,
    EconFunctionsMapping,
    EGMCarryProducer,
    FlatParams,
    FunctionName,
    MappingLeaf,
    NextStateSimulationFunction,
    ProcessName,
    QAndFFunction,
    RegimeName,
    RegimeNamesToIds,
    RegimeParamsTemplate,
    RegimeTransitionFunction,
    SequenceLeaf,
    StateName,
    StateOrActionName,
    TransitionFunction,
    TransitionFunctionName,
    TransitionFunctionsMapping,
    VmappedRegimeTransitionFunction,
)
from _lcm.utils.containers import ensure_containers_are_immutable
from _lcm.utils.dispatchers import simulation_spacemap, vmap_1d
from _lcm.utils.error_messages import format_messages
from _lcm.utils.namespace import flatten_regime_namespace, unflatten_regime_namespace
from _lcm.variables import (
    from_regime,
    get_grids,
    simulate_variables_from_regime,
)
from lcm.ages import AgeGrid
from lcm.exceptions import ModelInitializationError
from lcm.phased import Phased
from lcm.regime import Regime as UserRegime
from lcm.solvers import DCEGM, NEGM, Solver
from lcm.transition import (
    AgeSpecializedFunction,
    MarkovTransition,
)
from lcm.typing import BoolND, Float1D, FloatND, Int1D, IntND, UserFunction

type _TransitionBundles = dict[
    RegimeName, dict[TransitionFunctionName, UserFunction | _CoarseTransitionCell]
]


def _resolve_age_specialized_state_grids(
    *,
    user_regimes: Mapping[RegimeName, FinalizedUserRegime],
    ages: AgeGrid,
) -> tuple[
    MappingProxyType[RegimeName, FinalizedUserRegime],
    MappingProxyType[int, MappingProxyType[RegimeName, VInterpolationInfo]] | None,
]:
    """Resolve `AgeSpecializedGrid` states to representative + per-period grids.

    Returns:
        - `representative_user_regimes`: each regime with its `AgeSpecializedGrid`
          states replaced by the concrete grid at the regime's first active age
          (used by all age-invariant machinery). Unchanged when a regime has no
          age-varying grid.
        - `period_to_regime_v_interp`: `{period: {regime: VInterpolationInfo}}`
          with each regime's grids resolved at that period's age — used by the
          per-period Q-and-F so the continuation `V_{t+1}` interpolates on the
          *next* period's grid. `None` when no state anywhere is age-varying, so an
          age-invariant model builds exactly as before.
    """
    any_age_varying = any(
        has_age_specialized_grid(regime.states) for regime in user_regimes.values()
    )

    representative: dict[RegimeName, FinalizedUserRegime] = {}
    active_by_regime: dict[RegimeName, frozenset[int]] = {}
    for name, regime in user_regimes.items():
        active = ages.get_periods_where(regime.active)
        active_by_regime[name] = frozenset(active)
        if not active and has_age_specialized_grid(regime.states):
            # No active age means no age to resolve the builder at, so the marker would
            # travel unresolved into the ordinary grid machinery (audit F3). The regime
            # is inert, but an age-specialized grid on it is a modelling error, not a
            # thing to silently paper over.
            msg = (
                f"Regime '{name}' declares an AgeSpecializedGrid but is active at no "
                f"age, so there is no age at which to build its grid. Either give the "
                f"regime an active age or drop the age specialization."
            )
            raise RegimeInitializationError(msg)
        if active and has_age_specialized_grid(regime.states):
            # Validate shape-invariance only over the regime's ACTIVE ages — a builder
            # may be deliberately undefined (raise) outside them (audit F2).
            validate_age_specialized_grids(regime.states, ages, active_periods=active)
            rep_age = ages.period_to_age(active[0])
            representative[name] = regime.replace(
                states=dict(resolve_state_grids(regime.states, rep_age))
            )
        else:
            representative[name] = regime

    if not any_age_varying:
        return MappingProxyType(representative), None

    period_map: dict[int, MappingProxyType[RegimeName, VInterpolationInfo]] = {}
    for period in range(ages.n_periods):
        age = ages.period_to_age(period)
        period_map[period] = MappingProxyType(
            {
                # Resolve a regime's age-varying grids only where it is ACTIVE. A
                # regime's V_p is consumed as a continuation target only at ages where
                # the regime exists, so at inactive ages we reuse the representative
                # grid (built at the first active age) — this is never read as a
                # continuation and keeps an age-limited/terminal-only builder from being
                # called outside its domain (audit F2).
                name: create_v_interpolation_info(
                    regime.replace(states=dict(resolve_state_grids(regime.states, age)))
                    if (
                        has_age_specialized_grid(regime.states)
                        and period in active_by_regime[name]
                    )
                    else representative[name]
                )
                for name, regime in user_regimes.items()
            }
        )
    return MappingProxyType(representative), MappingProxyType(period_map)


def process_regimes(
    *,
    user_regimes: Mapping[RegimeName, FinalizedUserRegime],
    ages: AgeGrid,
    regime_names_to_ids: RegimeNamesToIds,
    enable_jit: bool,
) -> MappingProxyType[RegimeName, Regime]:
    """Process finalized regimes into canonical regimes.

    Canonicalizes every regime's laws into target-granular form
    (`canonicalize_regimes`), then compiles the per-phase function sets.
    Stochastic process transitions are generated from the grid's intrinsic
    transition logic.

    Args:
        user_regimes: Mapping of regime names to finalized regimes.
        ages: The AgeGrid for the model.
        regime_names_to_ids: Immutable mapping of regime names to integer indices.
        enable_jit: Whether to jit the functions of the canonical regime.

    Returns:
        The processed canonical regimes.

    """
    # DC-EGM regimes must satisfy the EGM model contract before any kernel
    # is built. `Model.__init__` validates earlier (so contract violations
    # beat the generic unused-variable check); this call covers direct
    # `process_regimes` callers.
    validate_dcegm_regimes(user_regimes=user_regimes)
    validate_negm_regimes(user_regimes=user_regimes)

    # The canonical specs hold every law in target-granular form, resolved per
    # phase: the simulate slice additionally holds every carried-only state
    # and its law of motion, so the canonical mapping carries the law toward
    # each reachable target that carries the state — including targets reached
    # through nothing but the carried state.
    specs = canonicalize_regimes(user_regimes=representative_user_regimes)
    solve_nested_transitions = {
        regime_name: _extract_phase_transitions(phase_slice=spec.solution)
        for regime_name, spec in specs.items()
    }
    simulate_nested_transitions = {
        regime_name: _extract_phase_transitions(phase_slice=spec.simulation)
        for regime_name, spec in specs.items()
    }
    _validate_categoricals(representative_user_regimes)

    regime_to_variables = MappingProxyType(
        {
            regime_name: from_regime(user_regime)
            for regime_name, user_regime in representative_user_regimes.items()
        }
    )
    all_grids = MappingProxyType(
        {
            regime_name: get_grids(user_regime)
            for regime_name, user_regime in representative_user_regimes.items()
        }
    )

    _fail_if_action_has_batch_size(user_regimes)

    regime_to_v_interpolation_info = MappingProxyType(
        {
            regime_name: create_v_interpolation_info(user_regime)
            for regime_name, user_regime in representative_user_regimes.items()
        }
    )
    state_action_spaces = MappingProxyType(
        {
            regime_name: create_state_action_space(
                variables=regime_to_variables[regime_name],
                grids=all_grids[regime_name],
            )
            for regime_name in user_regimes
        }
    )
    regimes_to_active_periods = MappingProxyType(
        {
            regime_name: ages.get_periods_where(user_regime.active)
            for regime_name, user_regime in user_regimes.items()
        }
    )

    model_has_egm_regime = any(
        user_regime.solver.requires_continuation
        for user_regime in user_regimes.values()
    )

    # Each regime's flat param names in the engine's binding vocabulary, keyed
    # by regime. A DC-EGM source regime that carries into a *different* target
    # regime evaluates that target's resources / transition functions in its
    # per-asset-node solve, so it must know the target's param leaves (e.g. a
    # pension factor the source itself never reads); the kernel binds them from
    # the union of the source and its reachable carry targets' fixed params.
    regime_to_params_template = MappingProxyType(
        {
            regime_name: create_regime_params_template(user_regime)
            for regime_name, user_regime in user_regimes.items()
        }
    )
    regime_to_granular_param_expansions = MappingProxyType(
        {
            regime_name: _granular_param_expansions(
                nested_transitions_by_phase=(
                    solve_nested_transitions[regime_name],
                    simulate_nested_transitions[regime_name],
                ),
                regime_params_template=regime_to_params_template[regime_name],
            )
            for regime_name in user_regimes
        }
    )
    regime_to_flat_param_names = MappingProxyType(
        {
            regime_name: _engine_flat_param_names(
                regime_params_template=regime_to_params_template[regime_name],
                granular_param_expansions=regime_to_granular_param_expansions[
                    regime_name
                ],
            )
            for regime_name in user_regimes
        }
    )

    canonical_regimes: dict[RegimeName, Regime] = {}
    # Iterate the representative-resolved regimes: identical to the user regimes
    # except that any `AgeSpecializedGrid` state is a concrete representative-age
    # grid, so every grid-derived call below is age-invariant.
    for regime_name, user_regime in representative_user_regimes.items():
        spec = specs[regime_name]
        regime_params_template = regime_to_params_template[regime_name]
        granular_param_expansions = regime_to_granular_param_expansions[regime_name]

        solution = _build_solution_phase(
            spec=spec,
            regime_name=regime_name,
            user_regimes=user_regimes,
            nested_transitions=solve_nested_transitions[regime_name],
            all_grids=all_grids,
            regime_params_template=regime_params_template,
            granular_param_expansions=granular_param_expansions,
            regime_to_flat_param_names=regime_to_flat_param_names,
            regime_names_to_ids=regime_names_to_ids,
            variables=regime_to_variables[regime_name],
            regimes_to_active_periods=regimes_to_active_periods,
            regime_to_v_interpolation_info=regime_to_v_interpolation_info,
            period_to_regime_v_interp=period_to_regime_v_interp,
            state_action_space=state_action_spaces[regime_name],
            ages=ages,
            enable_jit=enable_jit,
            certainty_equivalent=user_regime.certainty_equivalent,
            solver=user_regime.solver,
            model_has_egm_regime=model_has_egm_regime,
            has_taste_shocks=user_regime.taste_shocks is not None,
        )

        simulation = _build_simulation_phase(
            spec=spec,
            user_regime=user_regime,
            regime_name=regime_name,
            nested_transitions=simulate_nested_transitions[regime_name],
            all_grids=all_grids,
            regime_params_template=regime_params_template,
            granular_param_expansions=granular_param_expansions,
            regime_names_to_ids=regime_names_to_ids,
            variables=regime_to_variables[regime_name],
            simulation_variables=simulate_variables_from_regime(user_regime),
            regimes_to_active_periods=regimes_to_active_periods,
            regime_to_v_interpolation_info=regime_to_v_interpolation_info,
            period_to_regime_v_interp=period_to_regime_v_interp,
            state_action_space=state_action_spaces[regime_name],
            ages=ages,
            enable_jit=enable_jit,
            solve_transitions=solution.transitions,
            solve_stochastic_transition_names=solution.stochastic_transition_names,
            solve_compute_regime_transition_probs=solution.compute_regime_transition_probs,
            has_taste_shocks=user_regime.taste_shocks is not None,
            solver=user_regime.solver,
            certainty_equivalent=user_regime.certainty_equivalent,
        )

        stochastic_state_transitions = collect_stochastic_state_transitions(
            user_regime=user_regime,
            user_regimes=representative_user_regimes,
        )

        canonical_regimes[regime_name] = Regime(
            name=regime_name,
            terminal=spec.terminal,
            active_periods=tuple(regimes_to_active_periods[regime_name]),
            regime_params_template=regime_params_template,
            solution=solution,
            simulation=simulation,
            stochastic_state_transitions=stochastic_state_transitions,
            granular_param_expansions=granular_param_expansions,
            has_taste_shocks=user_regime.taste_shocks is not None,
            certainty_equivalent=user_regime.certainty_equivalent,
        )

    return ensure_containers_are_immutable(canonical_regimes)


def _build_solution_phase(
    *,
    spec: PhasedRegimeSpec,
    regime_name: RegimeName,
    user_regimes: Mapping[RegimeName, UserRegime],
    nested_transitions: _TransitionBundles,
    all_grids: MappingProxyType[RegimeName, MappingProxyType[StateOrActionName, Grid]],
    regime_params_template: RegimeParamsTemplate,
    granular_param_expansions: MappingProxyType[FunctionName, tuple[str, ...]],
    regime_to_flat_param_names: MappingProxyType[RegimeName, frozenset[str]],
    regime_names_to_ids: RegimeNamesToIds,
    variables: Variables,
    regimes_to_active_periods: MappingProxyType[RegimeName, tuple[int, ...]],
    regime_to_v_interpolation_info: MappingProxyType[RegimeName, VInterpolationInfo],
    period_to_regime_v_interp: (
        MappingProxyType[int, MappingProxyType[RegimeName, VInterpolationInfo]] | None
    ) = None,
    state_action_space: StateActionSpace,
    ages: AgeGrid,
    enable_jit: bool,
    certainty_equivalent: CertaintyEquivalent | None,
    solver: Solver,
    model_has_egm_regime: bool,
    has_taste_shocks: bool,
) -> SolutionPhase:
    """Build all compiled functions for the backward-induction (solve) phase.

    Args:
        spec: The regime's per-phase specification.
        user_regime: The finalized user regime, scanned for `Phased`
            declarations by the policy-replay gate.
        regime_name: The name of the regime.
        user_regimes: Mapping of regime names to user-provided `Regime`
            instances.
        nested_transitions: Per-target transition bundles for internal
            processing.
        all_grids: Immutable mapping of regime names to Grid spec objects.
        regime_params_template: The regime's parameter template.
        granular_param_expansions: Immutable mapping of coarse-template law
            keys to granular qname prefixes.
        regime_to_flat_param_names: Immutable mapping of every regime name to
            its flat param names in the engine's binding vocabulary. A DC-EGM
            source carrying into a different target regime reads the target's
            params in its per-asset-node solve, so the kernel build needs the
            whole mapping, not only the source regime's own params.
        regime_names_to_ids: Immutable mapping of regime names to integer indices.
        variables: States and actions of the regime with kind/topology/process tags.
        regimes_to_active_periods: Mapping of regime names to active period tuples.
        regime_to_v_interpolation_info: Mapping of regime names to state space info.
        state_action_space: The state-action space for this regime.
        ages: The AgeGrid for the model.
        enable_jit: Whether to jit the internal functions.
        certainty_equivalent: Nonlinear certainty equivalent declared by the
            regime, or `None`.
        solver: The regime's solver; the engine calls `validate` then
            `build_period_kernels` on it to obtain the per-period kernels.
        model_has_egm_regime: Whether any regime of the model uses a solver
            that reads continuation carries (an endogenous-grid solver);
            terminal regimes then produce their closed-form carries.
        has_taste_shocks: Whether the regime declares EV1 taste shocks on its
            discrete actions.

    Returns:
        Complete solve functions container.

    """
    core = _process_regime_core(
        functions=spec.solution.functions,
        constraints=spec.solution.constraints,
        state_transitions=spec.solution.state_transitions,
        nested_transitions=nested_transitions,
        all_grids=all_grids,
        regime_params_template=regime_params_template,
        variables=variables,
    )

    flat_param_names = _engine_flat_param_names(
        regime_params_template=regime_params_template,
        granular_param_expansions=granular_param_expansions,
    )

    # Fixed, distributed states are co-mapped with the continuation V so the solve
    # kernel reads only its device-local slice (no all-gather). Terminal regimes have
    # no continuation, so the set is empty there.
    co_map_state_names: tuple[StateName, ...] = ()
    co_map_v_arr_in_axes: tuple[MappingProxyType[RegimeName, int | None], ...] = ()

    if spec.terminal:
        compute_regime_transition_probs = None
        terminal_func = get_Q_and_F_terminal(
            flat_param_names=flat_param_names,
            functions=core.functions,
            constraints=core.constraints,
        )
        Q_and_F_functions = MappingProxyType(
            dict.fromkeys(range(ages.n_periods), terminal_func)
        )
        compute_intermediates: MappingProxyType[int, Callable] = MappingProxyType({})
    else:
        compute_regime_transition_probs = build_regime_transition_probs_functions(
            functions=core.functions,
            compute_regime_transition_probs=core.next_regime_func,
            grids=all_grids[regime_name],
            regime_names_to_ids=regime_names_to_ids,
            flat_param_names=flat_param_names,
            is_stochastic=spec.solution.stochastic_regime_transition,
            enable_jit=enable_jit,
            phase="solve",
            next_regime_cells=core.next_regime_cells,
        )
        co_map_state_names = _co_map_state_names(
            state_names=state_action_space.state_names,
            grids=all_grids[regime_name],
            transitions=core.transitions,
        )
        # A co-mapped state's axis is sliced only off the leaves that carry it; a
        # target regime where the state is pruned keeps its full leaf (`None`).
        co_map_v_arr_in_axes = tuple(
            MappingProxyType(
                {
                    target: 0
                    if state in regime_to_v_interpolation_info[target].state_names
                    else None
                    for target in regime_to_v_interpolation_info
                }
            )
            for state in co_map_state_names
        )
        Q_and_F_functions = _build_Q_and_F_per_period(
            regimes_to_active_periods=regimes_to_active_periods,
            functions=core.functions,
            constraints=core.constraints,
            transitions=core.transitions,
            stochastic_transition_names=core.stochastic_transition_names,
            compute_regime_transition_probs=compute_regime_transition_probs,
            regime_to_v_interpolation_info=regime_to_v_interpolation_info,
            ages=ages,
            flat_param_names=flat_param_names,
            co_map_state_names=co_map_state_names,
            certainty_equivalent=certainty_equivalent,
        )
        compute_intermediates = _build_compute_intermediates_per_period(
            flat_param_names=flat_param_names,
            regimes_to_active_periods=regimes_to_active_periods,
            functions=core.functions,
            constraints=core.constraints,
            transitions=core.transitions,
            stochastic_transition_names=core.stochastic_transition_names,
            compute_regime_transition_probs=compute_regime_transition_probs,
            regime_to_v_interpolation_info=regime_to_v_interpolation_info,
            state_action_space=state_action_space,
            grids=all_grids[regime_name],
            ages=ages,
            enable_jit=enable_jit,
            certainty_equivalent=certainty_equivalent,
            # F4: diagnostics recompute on the SAME period-specific target grid as the
            # primary solve (not the representative grid).
            period_to_regime_v_interp=period_to_regime_v_interp,
            continuation_grid_signature=_continuation_grid_signature,
        )

    # Dispatch the per-period kernel build polymorphically on the regime's
    # solver: `validate` rejects out-of-scope configurations at build time,
    # then `build_period_kernels` returns one uniform period adapter per period
    # plus the regime's continuation template. `GridSearch` wraps the
    # max-Q-over-a grid search; `DCEGM` wraps the EGM step.
    context = SolverBuildContext(
        regime_name=regime_name,
        user_regimes=user_regimes,
        state_action_space=state_action_space,
        Q_and_F_functions=Q_and_F_functions,
        grids=all_grids[regime_name],
        functions=core.functions,
        constraints=core.constraints,
        transitions=core.transitions,
        stochastic_transition_names=core.stochastic_transition_names,
        compute_regime_transition_probs=compute_regime_transition_probs,
        regime_to_v_interpolation_info=regime_to_v_interpolation_info,
        regimes_to_active_periods=regimes_to_active_periods,
        flat_param_names=flat_param_names,
        regime_to_flat_param_names=regime_to_flat_param_names,
        enable_jit=enable_jit,
        has_taste_shocks=has_taste_shocks,
        certainty_equivalent=certainty_equivalent,
        co_map_state_names=co_map_state_names,
        co_map_v_arr_in_axes=co_map_v_arr_in_axes,
    )
    solver.validate(context=context)
    solver_kernels = solver.build_period_kernels(context=context)

    # The terminal continuation publisher is a cross-solver concern, not the
    # grid search's: a terminal regime in a model with a DC-EGM regime must
    # publish a closed-form carry so a DC-EGM parent can interpolate its value
    # and marginal utility. Build the producer engine-side and compose it as an
    # output decorator around each period adapter, so the solver stays unaware
    # of the continuation it is being asked to emit.
    egm_carry_producer, egm_carry_template = _build_egm_child_carry_producer(
        user_regime=user_regimes[regime_name],
        functions=core.functions,
        variables=variables,
        grids=all_grids[regime_name],
        model_has_egm_regime=model_has_egm_regime,
        solver_produces_carry=solver_kernels.continuation_template is not None,
        enable_jit=enable_jit,
    )
    period_kernels = solver_kernels.period_kernels
    continuation_template = solver_kernels.continuation_template
    if egm_carry_producer is not None:
        period_kernels = MappingProxyType(
            {
                period: _TerminalCarryPeriodKernel(
                    base=kernel,
                    carry_producer=egm_carry_producer,
                    regime_name=regime_name,
                )
                for period, kernel in period_kernels.items()
            }
        )
        continuation_template = egm_carry_template

    # The published function set is consumed unresolved by feasibility checks and
    # additional-target computation, so resolve any `AgeSpecializedFunction`
    # marker to its representative-age concrete function here (the per-period
    # Q_and_F build keeps
    # resolving the marker-bearing `core.functions` per age).
    solution_active_periods = regimes_to_active_periods[regime_name]
    published_solution_functions = (
        cast(
            "EconFunctionsMapping",
            resolve_specialized_nodes(
                core.functions,
                float(ages.period_to_age(solution_active_periods[0])),
            ),
        )
        if solution_active_periods
        else core.functions
    )

    period_state_axes = _build_period_state_axes(
        regime_name=regime_name,
        period_to_regime_v_interp=period_to_regime_v_interp,
        active_periods=regimes_to_active_periods[regime_name],
    )

    return SolutionPhase(
        _variables=variables,
        grids=all_grids[regime_name],
        functions=published_solution_functions,
        constraints=core.constraints,
        transitions=core.transitions,
        stochastic_transition_names=core.stochastic_transition_names,
        compute_regime_transition_probs=compute_regime_transition_probs,
        period_kernels=period_kernels,
        compute_intermediates=compute_intermediates,
        continuation_template=continuation_template,
        _base_state_action_space=state_action_space,
        period_state_axes=period_state_axes,
    )


def _filter_kwargs_for_func(
    *, func: Callable, kwargs: Mapping[str, object]
) -> Mapping[str, object]:
    """Filter kwargs to only those accepted by func's signature."""
    try:
        sig = inspect.signature(func)
    except ValueError, TypeError:
        return kwargs
    params = sig.parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in params}


@dataclass(frozen=True, kw_only=True)
class _TerminalCarryPeriodKernel:
    """Engine-owned output decorator publishing a terminal regime's carry.

    Wraps a grid-search period adapter so that, after the base kernel computes
    the value-function array, the regime's closed-form carry producer turns that
    array into the continuation a DC-EGM parent interpolates. Publishing a carry
    is a cross-solver concern, so the wrapped solver stays unaware of it.

    `core` and `build_lower_args` delegate to the base adapter: the carry
    producer is a separately built (and jitted) closure invoked inline, not part
    of the AOT-compiled core, so AOT compilation deduplicates and lowers exactly
    the base grid-search core.
    """

    base: PeriodKernel
    """The wrapped grid-search period adapter."""

    carry_producer: EGMCarryProducer
    """Closed-form producer mapping the value array to the regime's carry."""

    regime_name: RegimeName
    """Name of the terminal regime whose flat params the producer reads."""

    @property
    def core(self) -> Callable:
        """The base adapter's shared jitted core, for any single-core reader."""
        return self.base.core

    def cores(self) -> Mapping[str, Callable]:
        """Delegate to the base adapter's cores (a terminal regime is single-core)."""
        return self.base.cores()

    def with_fixed_params(
        self, *, fixed_flat_params: FlatParams
    ) -> _TerminalCarryPeriodKernel:
        """Bind fixed params into both the base core and the carry producer.

        A terminal regime's carry producer evaluates the regime's own bequest
        utility on the wealth grid; that utility may reach a model-level fixed
        param (e.g. a consumption-equivalence scale) through a helper. The solve
        loop invokes the producer with only the live (free) params, so bind the
        regime's fixed params here — matching the base adapter's core binding.
        """
        regime_fixed = dict(
            fixed_flat_params.get(self.regime_name, MappingProxyType({}))
        )
        base = self.base.with_fixed_params(fixed_flat_params=fixed_flat_params)
        carry_producer = self.carry_producer
        if regime_fixed:
            carry_producer = functools.partial(
                carry_producer,
                **_filter_kwargs_for_func(func=carry_producer, kwargs=regime_fixed),
            )
        return dataclass_replace(self, base=base, carry_producer=carry_producer)

    def build_lower_args(
        self,
        *,
        core_key: str = "main",
        state_action_space: StateActionSpace,
        next_regime_to_V_arr: Mapping[RegimeName, FloatND],
        next_regime_to_continuation: Mapping[RegimeName, ContinuationPayload],
        flat_params: FlatParams,
        period: int,
        ages: AgeGrid,
    ) -> Mapping[str, object]:
        """Build the base core's lowering arguments (the carry producer is jitted
        separately at build time, so it is not part of the AOT-compiled core)."""
        return self.base.build_lower_args(
            core_key=core_key,
            state_action_space=state_action_space,
            next_regime_to_V_arr=next_regime_to_V_arr,
            next_regime_to_continuation=next_regime_to_continuation,
            flat_params=flat_params,
            period=period,
            ages=ages,
        )

    def __call__(
        self,
        *,
        compiled_cores: Mapping[str, Callable],
        state_action_space: StateActionSpace,
        next_regime_to_V_arr: Mapping[RegimeName, FloatND],
        next_regime_to_continuation: Mapping[RegimeName, ContinuationPayload],
        flat_params: FlatParams,
        period: int,
        ages: AgeGrid,
    ) -> KernelResult:
        """Run the base kernel, then publish the regime's continuation carry."""
        result = self.base(
            compiled_cores=compiled_cores,
            state_action_space=state_action_space,
            next_regime_to_V_arr=next_regime_to_V_arr,
            next_regime_to_continuation=next_regime_to_continuation,
            flat_params=flat_params,
            period=period,
            ages=ages,
        )
        carry = self.carry_producer(
            V_arr=result.V_arr,
            **state_action_space.states,
            **flat_params[self.regime_name],
            period=jnp.int32(period),
            age=ages.values[period],
        )
        return KernelResult(V_arr=result.V_arr, continuation=carry)


def _build_egm_child_carry_producer(
    *,
    user_regime: UserRegime,
    functions: EconFunctionsMapping,
    variables: Variables,
    grids: MappingProxyType[StateOrActionName, Grid],
    model_has_egm_regime: bool,
    solver_produces_carry: bool,
    enable_jit: bool,
) -> tuple[EGMCarryProducer | None, EGMCarry | None]:
    """Build the carry producer and template for an EGM regime's carry target.

    A regime an endogenous-grid regime transitions into must publish its value
    and marginal value of resources so the parent can interpolate them. The
    regime's own solver already builds its carry when it is endogenous-grid
    (`solver_produces_carry`); this engine-side producer covers the brute
    (`GridSearch`) targets that do not. Cases:

    - no states (terminal) ⇒ constant-value, zero-marginal-utility broadcast rows
    - terminal with ≥1 continuous state, no actions, and discrete states only of
      the fixed (non-process) kind ⇒ terminal utility and its wealth gradient on
      the regime's own state grid, the discrete states leading
    - living brute regime with a continuous Euler state ⇒ its solved value array
      and the array's Euler-state gradient, discrete states (process states
      included) and passive continuous states leading
    - anything else ⇒ no producer (an EGM regime targeting an unsupported shape
      is rejected by the EGM kernel builder)

    Returns:
        Tuple of the producer and the regime's carry template, both `None` for
        models without an endogenous-grid regime, for regimes whose own solver
        already produces a carry, and for unsupported shapes.

    """
    if not model_has_egm_regime or solver_produces_carry:
        return None, None
    producer: EGMCarryProducer
    discrete_state_names = tuple(
        name
        for name in variables.state_names
        if name in set(variables.discrete_state_names)
    )
    continuous_state_names = tuple(variables.continuous_state_names)
    euler_state_name = next(iter(user_regime.states), None)
    if user_regime.terminal:
        has_only_fixed_discrete_states = all(
            not isinstance(grids[name], _ContinuousStochasticProcess)
            for name in discrete_state_names
        )
        if not variables.state_names:
            producer = get_stateless_terminal_carry_producer()
            template = build_template_egm_carry(n_rows=N_STATELESS_CARRY_ROWS)
        elif (
            len(continuous_state_names) >= 1
            and has_only_fixed_discrete_states
            and not user_regime.actions
            and euler_state_name in continuous_state_names
        ):
            # The parent's child read picks the terminal's Euler state as its
            # first declared state (`_get_child_state_name`); the remaining
            # continuous states are the passive (durable / outer) margins it
            # interpolates as leading carry axes — the NEGM housing-bequest shape.
            passive_state_names = tuple(
                name for name in continuous_state_names if name != euler_state_name
            )
            producer = get_terminal_wealth_carry_producer(
                functions=functions,
                state_name=euler_state_name,
                discrete_state_names=discrete_state_names,
                passive_state_names=passive_state_names,
                continuous_state_order=continuous_state_names,
            )
            leading_shape = tuple(
                int(grids[name].to_jax().shape[0])
                for name in discrete_state_names + passive_state_names
            )
            template = shard_carry_template(
                template=build_template_egm_carry(
                    n_rows=int(grids[euler_state_name].to_jax().shape[0]),
                    leading_shape=leading_shape,
                ),
                grids=grids,
                leading_axis_names=discrete_state_names + passive_state_names,
            )
        else:
            return None, None
    elif (
        len(continuous_state_names) >= 1 and euler_state_name in continuous_state_names
    ):
        # A living brute target: its solved value array is the carry value and the
        # array's Euler-state gradient is the marginal value of resources. The
        # parent reads it in M-space ($R \\equiv M$) via the same child read it
        # uses for an endogenous-grid target.
        passive_state_names = tuple(
            name for name in continuous_state_names if name != euler_state_name
        )
        producer = get_brute_child_carry_producer(
            state_name=euler_state_name,
            discrete_state_names=discrete_state_names,
            passive_state_names=passive_state_names,
            continuous_state_order=continuous_state_names,
        )
        leading_shape = tuple(
            int(grids[name].to_jax().shape[0])
            for name in discrete_state_names + passive_state_names
        )
        template = shard_carry_template(
            template=build_template_egm_carry(
                n_rows=int(grids[euler_state_name].to_jax().shape[0]),
                leading_shape=leading_shape,
            ),
            grids=grids,
            leading_axis_names=discrete_state_names + passive_state_names,
        )
    else:
        return None, None
    if enable_jit:
        producer = jax.jit(producer)
    return producer, template


def _build_simulation_phase(
    *,
    spec: PhasedRegimeSpec,
    user_regime: UserRegime,
    regime_name: RegimeName,
    nested_transitions: _TransitionBundles,
    all_grids: MappingProxyType[RegimeName, MappingProxyType[StateOrActionName, Grid]],
    regime_params_template: RegimeParamsTemplate,
    granular_param_expansions: MappingProxyType[FunctionName, tuple[str, ...]],
    regime_names_to_ids: RegimeNamesToIds,
    variables: Variables,
    simulation_variables: Variables,
    regimes_to_active_periods: MappingProxyType[RegimeName, tuple[int, ...]],
    regime_to_v_interpolation_info: MappingProxyType[RegimeName, VInterpolationInfo],
    period_to_regime_v_interp: (
        MappingProxyType[int, MappingProxyType[RegimeName, VInterpolationInfo]] | None
    ) = None,
    state_action_space: StateActionSpace,
    ages: AgeGrid,
    enable_jit: bool,
    solve_transitions: TransitionFunctionsMapping,
    solve_stochastic_transition_names: frozenset[TransitionFunctionName],
    solve_compute_regime_transition_probs: RegimeTransitionFunction | None,
    has_taste_shocks: bool,
    solver: Solver,
    certainty_equivalent: CertaintyEquivalent | None,
) -> SimulationPhase:
    """Build all compiled functions for the forward-simulation phase.

    The decision functions (Q_and_F, argmax, regime-transition probs) are
    built from the simulation slice's functions plus each carried state's
    solve-phase imputation — the agent decides on the value the solved policy
    was computed for. The published function pool strips the imputations so
    every other simulate consumer reads the carried value.

    Q_and_F always uses the solve (non-vmapped) regime transition probs because
    it evaluates on the Cartesian grid, not per-subject.

    For a DC-EGM or NEGM regime, the budget constraint the EGM solve enforces
    intrinsically is synthesized and injected into the constraint set: the
    simulate-phase grid argmax needs it as a feasibility mask exactly like a
    user-declared borrowing constraint of a brute-force regime. NEGM nests the
    same inner 1-D solve, so the mask comes from its inner DC-EGM config. The
    solve phase is unaffected — the EGM kernels never see it.

    Args:
        spec: The regime's per-phase specification.
        user_regime: The finalized user regime, scanned for `Phased`
            declarations by the policy-replay gate.
        regime_name: The name of the regime.
        nested_transitions: Per-target transition bundles for internal
            processing.
        all_grids: Immutable mapping of regime names to Grid spec objects.
        regime_params_template: The regime's parameter template.
        granular_param_expansions: Immutable mapping of coarse-template law
            keys to granular qname prefixes.
        regime_names_to_ids: Immutable mapping of regime names to integer indices.
        variables: States and actions of the regime with kind/topology/process tags.
        simulation_variables: Simulate-phase variables (solve variables plus
            carried-only states, appended).
        regimes_to_active_periods: Mapping of regime names to active period tuples.
        regime_to_v_interpolation_info: Mapping of regime names to state space info.
        state_action_space: The state-action space for this regime.
        ages: The AgeGrid for the model.
        enable_jit: Whether to jit the internal functions.
        solve_transitions: Transitions from the solve phase (reused).
        solve_stochastic_transition_names: Stochastic transition names from solve
            (reused).
        solve_compute_regime_transition_probs: Solve-phase regime transition prob
            function, used for Q_and_F in both phases.
        has_taste_shocks: Whether the regime declares EV1 taste shocks on its
            discrete actions.
        solver: The regime's solver configuration; a DC-EGM or NEGM regime
            gets the synthesized intrinsic budget constraint.
        certainty_equivalent: Nonlinear certainty equivalent declared by the
            regime, or `None`.

    Returns:
        Complete simulate functions container.

    """
    carried_only = spec.carried_only_state_names
    decision_functions = dict(spec.simulation.functions) | {
        name: spec.solution.functions[name] for name in carried_only
    }
    core = _process_regime_core(
        functions=decision_functions,
        constraints=spec.simulation.constraints,
        state_transitions=spec.simulation.state_transitions,
        nested_transitions=nested_transitions,
        all_grids=all_grids,
        regime_params_template=regime_params_template,
        variables=variables,
    )
    functions = core.functions
    constraints = core.constraints
    if isinstance(solver, (DCEGM, NEGM)):
        if (
            DCEGM_BUDGET_CONSTRAINT_NAME in core.functions
            or DCEGM_BUDGET_CONSTRAINT_NAME in core.constraints
        ):
            msg = (
                f"Regime '{regime_name}' declares a function or constraint "
                f"named '{DCEGM_BUDGET_CONSTRAINT_NAME}'. That name is "
                "reserved for the budget constraint the simulate phase "
                "synthesizes for DC-EGM and NEGM regimes; rename it."
            )
            raise ModelInitializationError(msg)
        constraints = MappingProxyType(
            {
                **core.constraints,
                DCEGM_BUDGET_CONSTRAINT_NAME: get_intrinsic_budget_constraint(
                    solver=solver, functions=core.functions
                ),
            }
        )

    # Every published simulate-phase consumer (next_state, the realized
    # regime draw, the feasibility check, additional targets) reads each
    # carried state as its carried true value, not the solve-phase
    # imputation. Dropping the imputation turns the name into a leaf supplied
    # by the simulator, and `core.transitions` (built from the simulation
    # slice) carries every simulate-phase law — including each carried
    # state's `next_<name>` and any `Phased` law's simulate variant. Only the
    # decision functions (Q_and_F / argmax) keep the imputation — the agent
    # decides on the value the solved policy was computed for. To make the
    # realized regime draw decide on the imputation instead, declare the
    # imputation under a second name in `functions` and read that.
    if carried_only:
        simulate_functions: EconFunctionsMapping = MappingProxyType(
            {k: v for k, v in core.functions.items() if k not in carried_only}
        )
    else:
        simulate_functions = core.functions
    # Carried states are `Phased(simulate=Grid)` by the phase grammar, so the
    # isinstance check is a no-op at runtime; it narrows the type (an
    # `AgeSpecializedGrid` can never be carried-only).
    carried_grids = {
        name: grid
        for name, grid in spec.simulation.grid_states.items()
        if name in carried_only and isinstance(grid, Grid)
    }
    simulate_grids = MappingProxyType({**all_grids[regime_name], **carried_grids})

    flat_param_names = _engine_flat_param_names(
        regime_params_template=regime_params_template,
        granular_param_expansions=granular_param_expansions,
    )

    if spec.terminal:
        compute_regime_transition_probs = None
        terminal_func = get_Q_and_F_terminal(
            flat_param_names=flat_param_names,
            functions=functions,
            constraints=constraints,
        )
        Q_and_F_functions = MappingProxyType(
            dict.fromkeys(range(ages.n_periods), terminal_func)
        )
    else:
        compute_regime_transition_probs = build_regime_transition_probs_functions(
            functions=simulate_functions,
            compute_regime_transition_probs=core.next_regime_func,
            grids=simulate_grids,
            regime_names_to_ids=regime_names_to_ids,
            flat_param_names=flat_param_names,
            is_stochastic=spec.simulation.stochastic_regime_transition,
            enable_jit=enable_jit,
            phase="simulate",
            next_regime_cells=core.next_regime_cells,
        )
        # Q_and_F uses the solve (non-vmapped) regime transition probs since
        # it evaluates on the Cartesian grid, not per-subject. The solve
        # phase built that function unconditionally for non-terminal regimes.
        assert solve_compute_regime_transition_probs is not None  # noqa: S101
        Q_and_F_functions = _build_Q_and_F_per_period(
            regimes_to_active_periods=regimes_to_active_periods,
            functions=functions,
            constraints=constraints,
            transitions=solve_transitions,
            stochastic_transition_names=solve_stochastic_transition_names,
            compute_regime_transition_probs=solve_compute_regime_transition_probs,
            regime_to_v_interpolation_info=regime_to_v_interpolation_info,
            ages=ages,
            flat_param_names=flat_param_names,
            certainty_equivalent=certainty_equivalent,
            period_to_regime_v_interp=period_to_regime_v_interp,
        )

    argmax_and_max_Q_over_a = _build_argmax_and_max_Q_over_a_per_period(
        state_action_space=state_action_space,
        Q_and_F_functions=Q_and_F_functions,
        enable_jit=enable_jit,
        has_taste_shocks=has_taste_shocks,
    )

    next_state = _build_next_state_vmapped(
        functions=simulate_functions,
        transitions=core.transitions,
        stochastic_transition_names=core.stochastic_transition_names,
        all_grids=all_grids,
        variables=variables,
        flat_param_names=flat_param_names,
        ages=ages,
        enable_jit=enable_jit,
    )

    # Replaying the solve-phase EGM policy in simulation is valid only when
    # the simulate-phase decision problem is the one the solve optimized and
    # the published rows carry both the coordinates and the branch topology
    # the read interpolates over at every state dimension:
    # - a standalone `DCEGM` solver — a NEGM regime maximizes its value over
    #   keeper and adjuster candidates but publishes only the keeper's inner
    #   consumption function, so replaying it would pair an adjuster-won
    #   value with the keeper's policy;
    # - a crossing-certifying upper-envelope backend (MSS only) — RFC and LTM
    #   leave the envelope switch between two retained nodes, and FUES decides
    #   segment identity by a slope-threshold heuristic with no labels from
    #   the kernel, so its row can bridge a missed switch; only MSS inserts
    #   the exact crossing by construction (see
    #   `_envelope_publishes_crossings`);
    # - the single-post-state kernel (not asset-row mode) — when a
    #   savings-stage function reads the Euler state, DC-EGM solves per
    #   exogenous asset node and publishes one optimal point per node, not a
    #   crossing-complete resources-space row, so interpolating across nodes
    #   would mix branches wherever the winner changes between adjacent nodes;
    # - no continuous stochastic-process state — a process is stored as a
    #   node-valued row axis, but its simulation transition draws a continuous
    #   value that need not land on a node, so nearest-node row selection
    #   reads the wrong conditional policy;
    # - no passive continuous state — each row is the upper-envelope policy
    #   conditional on one passive node, and blending two rows across a
    #   passive-dimension branch switch yields an action from neither branch;
    # - no `Phased` declaration anywhere on the regime (a phase-variant
    #   utility, budget, transition, or state domain changes the
    #   simulate-phase FOC or the policy-row coordinates even under an
    #   unchanged `H`) — `Phased` is the grammar's only source of phase
    #   variance, so its absence is the exact test;
    # - no carried-only states (their simulate-phase domain has no solve-side
    #   row coordinates to read the policy on);
    # - no taste shocks (the realized discrete draw perturbs the decision).
    # The process, passive, and asset-row exclusions each lift once the read
    # publishes conditional values and re-decides the branch at the simulated
    # state; today the gate keeps those regimes on the grid-argmax path.
    phase_invariant = (
        not regime_declares_phased(user_regime) and not spec.carried_only_state_names
    )
    own_v_info = regime_to_v_interpolation_info[regime_name]
    egm_policy_read = None
    if (
        isinstance(solver, DCEGM)
        and _envelope_publishes_crossings(solver)
        and phase_invariant
        and not has_taste_shocks
        and not _regime_has_process_state(own_v_info)
        and not _regime_has_passive_state(
            v_interpolation_info=own_v_info, euler_state_name=solver.continuous_state
        )
        and not savings_stage_reads_euler_state(user_regime=user_regime, solver=solver)
    ):
        egm_policy_read = EGMPolicyRead(
            action_name=solver.continuous_action,
            resources_target=solver.resources,
            savings_lower_bound=float(solver.savings_grid.to_jax()[0]),
        )

    return SimulationPhase(
        _variables=simulation_variables,
        grids=simulate_grids,
        carried_only_state_names=frozenset(carried_grids),
        functions=published_simulate_functions,
        constraints=constraints,
        age_specialized_function_names=age_specialized_function_names,
        transitions=core.transitions,
        stochastic_transition_names=core.stochastic_transition_names,
        compute_regime_transition_probs=compute_regime_transition_probs,
        argmax_and_max_Q_over_a=argmax_and_max_Q_over_a,
        next_state=next_state,
        egm_policy_read=egm_policy_read,
    )


def _envelope_publishes_crossings(solver: DCEGM) -> bool:
    """Whether the solver's upper envelope certifies every segment crossing.

    A branch-faithful policy read interpolates a row whose envelope switches sit
    at duplicated abscissae carrying both branch records:
    - `"mss"` ⇒ yes: the refinement enumerates every envelope switch — interior
      crossings via iterated earliest-overtake between adjacent candidate
      abscissae (where the bracketing segments are full lines), switches and
      value jumps landing exactly on a candidate abscissa via two-sided node
      records — and an interval whose switch sequence exceeds the enumeration
      budget overflows loudly through `n_kept` — as does a live candidate
      point whose value no interval read can represent — so a published row
      is never a silently truncated envelope. The guarantee covers the
      live-covered domain; a row whose segment chain splits (NaN-dead
      candidates or a finite value decrease between consecutive candidates)
      is NaN-poisoned in the published policy via the kernel's read-support
      verdict, so the reader falls back to grid-argmax instead of bridging
      the gap linearly.
    - `"fues"` ⇒ no: segment identity is decided by thresholding the
      implied-savings slope (`fues_jump_thresh`) — a heuristic — and the
      DC-EGM kernel supplies no segment labels. Two value branches whose
      cross-segment slope stays below the threshold merge into one row with
      no crossing inserted; the row then bridges the switch, and neither an
      exhaustive scan nor a wider window repairs that, because the scan can
      only search within the segment identity it was given. A FUES row is
      therefore not certified crossing-complete for the read.
    - `"rfc"` / `"ltm"` ⇒ no: the switch lands between retained nodes.
    """
    return solver.upper_envelope == "mss"


def _regime_has_process_state(v_interpolation_info: VInterpolationInfo) -> bool:
    """Whether the regime carries a continuous stochastic-process state.

    A process is stored among the node-valued discrete states, so its policy
    row is a discrete axis. Its simulation transition draws a continuous value
    off the process grid, which a nearest-node row read cannot resolve.
    """
    return any(
        isinstance(grid, _ContinuousStochasticProcess)
        for grid in v_interpolation_info.discrete_states.values()
    )


def _regime_has_passive_state(
    *, v_interpolation_info: VInterpolationInfo, euler_state_name: StateName
) -> bool:
    """Whether the regime carries a passive continuous state.

    Every continuous state other than the Euler state is passive; the DC-EGM
    policy row is conditional on each passive node, so blending rows across the
    passive axis is only branch-faithful where no envelope switch lies between.
    """
    return any(
        name != euler_state_name for name in v_interpolation_info.continuous_states
    )


def regime_declares_phased(user_regime: UserRegime) -> bool:
    """Whether any regime slot carries a `Phased` (phase-variant) declaration.

    Scans `functions`, `state_transitions` (including per-target dicts), the
    regime `transition`, and `states`. `Phased` is outermost-only in every
    slot except a per-target transition dict, whose cells it may wrap.
    """

    def is_phased(value: object) -> bool:
        if isinstance(value, Phased):
            return True
        if isinstance(value, Mapping):
            return any(isinstance(cell, Phased) for cell in value.values())
        return False

    return (
        any(is_phased(value) for value in user_regime.functions.values())
        or any(is_phased(value) for value in user_regime.state_transitions.values())
        or is_phased(user_regime.transition)
        or any(is_phased(value) for value in user_regime.states.values())
    )


@dataclass(frozen=True)
class _CoreResult:
    """Result of core regime function processing for one phase."""

    functions: EconFunctionsMapping
    """User functions (utility, helpers) with params renamed to qnames."""

    constraints: ConstraintFunctionsMapping
    """Constraint functions with params renamed to qnames."""

    transitions: TransitionFunctionsMapping
    """Nested mapping of transition names to transition functions."""

    stochastic_transition_names: frozenset[TransitionFunctionName]
    """Frozenset of stochastic transition function names."""

    next_regime_func: TransitionFunction | None
    """The coarse regime transition function; `None` for terminal regimes and
    for per-target regime transitions."""

    next_regime_cells: MappingProxyType[RegimeName, EconFunction] | None
    """Per-target regime transition probability functions (params renamed),
    or `None` when the regime transition is coarse or absent."""


def _process_regime_core(
    *,
    functions: Mapping[FunctionName, UserFunction],
    constraints: Mapping[FunctionName, UserFunction],
    state_transitions: Mapping[StateName, object],
    nested_transitions: _TransitionBundles,
    all_grids: MappingProxyType[RegimeName, MappingProxyType[StateOrActionName, Grid]],
    regime_params_template: RegimeParamsTemplate,
    variables: Variables,
) -> _CoreResult:
    """Process one phase's regime functions and transitions.

    The caller supplies phase-resolved inputs (a slice of the regime's
    `PhasedRegimeSpec`, possibly augmented): rename params to qualified names,
    classify and process transitions.

    Args:
        functions: Phase-resolved regime functions for this build.
        constraints: Phase-resolved constraint functions.
        state_transitions: This phase's `state_transitions` slice, used to
            detect per-target dicts and stochastic transitions.
        nested_transitions: Per-target transition bundles for internal
            processing.
        all_grids: Immutable mapping of regime names to Grid spec objects.
        regime_params_template: The regime's parameter template.
        variables: States and actions of the regime with kind/topology/process tags.

    Returns:
        Core processing result with functions, constraints, transitions, stochastic
        transition names, and the next_regime function.

    """
    flat_grids = flatten_regime_namespace(all_grids)

    # The canonical regime transition rides in the bundles as each target's
    # `"next_regime"` cell. Cells are not state laws: split them off before
    # the flat-namespace processing, dropping bundles that held nothing else
    # so the per-target `transitions` mapping keys exactly the targets with
    # at least one state law (the period-target enumeration reads those
    # keys).
    next_regime_cells_by_target: dict[
        RegimeName, UserFunction | _CoarseTransitionCell
    ] = {}
    state_law_bundles: dict[RegimeName, dict[TransitionFunctionName, UserFunction]] = {}
    for target_regime_name, bundle in nested_transitions.items():
        if "next_regime" in bundle:
            next_regime_cells_by_target[target_regime_name] = bundle["next_regime"]
        laws = {
            law_name: cast("UserFunction", law)
            for law_name, law in bundle.items()
            if law_name != "next_regime"
        }
        if laws:
            state_law_bundles[target_regime_name] = laws

    flat_nested_transitions = flatten_regime_namespace(state_law_bundles)

    all_functions: dict[str, UserFunction] = {
        **functions,
        **constraints,
        **flat_nested_transitions,
    }

    stochastic_transition_names = _get_stochastic_transition_names(
        state_transitions=state_transitions, variables=variables
    )

    stochastic_transition_functions = {
        func_name: func
        for func_name, func in flat_nested_transitions.items()
        if tree_path_from_qname(func_name)[-1] in stochastic_transition_names
    }

    deterministic_transition_functions = {
        func_name: func
        for func_name, func in all_functions.items()
        if func_name in flat_nested_transitions
        and func_name not in stochastic_transition_functions
    }

    deterministic_functions = {
        func_name: func
        for func_name, func in all_functions.items()
        if func_name not in stochastic_transition_functions
        and func_name not in deterministic_transition_functions
    }

    processed_functions: dict[str, EconFunction] = {}

    for func_name, func in deterministic_functions.items():
        processed_functions[func_name] = _process_one_function(
            func=func,
            regime_params_template=regime_params_template,
            param_key=func_name,
        )

    for func_name, func in deterministic_transition_functions.items():
        processed_functions[func_name] = _rename_params_to_qnames(
            func=func,
            regime_params_template=regime_params_template,
            param_key=func_name,
            names_key=_extract_template_names_key(func_name, regime_params_template),
        )

    for func_name, func in stochastic_transition_functions.items():
        processed_functions[f"weight_{func_name}"] = _rename_params_to_qnames(
            func=func,
            regime_params_template=regime_params_template,
            param_key=func_name,
            names_key=_extract_template_names_key(func_name, regime_params_template),
        )
        processed_functions[func_name] = _get_discrete_markov_next_function(
            func=func,
            grid=flat_grids[func_name.replace("next_", "")].to_jax(),
        )

    # Transitions of continuous stochastic processes bypass the stub pipeline
    # entirely. Build weight and next functions for reachable target regimes
    # from each target's grid. Scope to targets already present in non-process
    # transitions to avoid spurious entries for unreachable regimes.
    process_names = variables.process_names
    reachable_targets = {
        tree_path_from_qname(k)[0]
        for k in flat_nested_transitions
        if QNAME_DELIMITER in k
    }
    target_process_grids: dict[
        tuple[RegimeName, ProcessName], _ContinuousStochasticProcess
    ] = {
        (user_regime, process): grid
        for user_regime, grids in all_grids.items()
        if user_regime in reachable_targets
        for process in process_names
        if isinstance(grid := grids.get(process), _ContinuousStochasticProcess)
    }
    for grid in target_process_grids.values():
        if grid.state_conditioned is not None:
            _validate_conditioning_codes_agree_across_regimes(
                on=grid.state_conditioned.on, all_grids=all_grids
            )
    processed_functions |= {
        f"weight_{user_regime}__next_{process}": _get_weights_func_for_process(
            name=process, grid=grid, grids=all_grids[user_regime]
        )
        for (user_regime, process), grid in target_process_grids.items()
    } | {
        f"{user_regime}__next_{process}": _get_stochastic_next_function_for_process(
            name=process, grid=grid.to_jax()
        )
        for (user_regime, process), grid in target_process_grids.items()
    }

    process_transition_keys = {
        f"{user_regime}__next_{process}"
        for user_regime, process in target_process_grids
    }
    internal_transition = {
        func_name: processed_functions[func_name]
        for func_name in flat_nested_transitions
    } | {key: processed_functions[key] for key in process_transition_keys}

    processed_constraints: ConstraintFunctionsMapping = MappingProxyType(
        {func_name: processed_functions[func_name] for func_name in constraints}
    )
    excluded_from_functions = (
        set(flat_nested_transitions) | set(constraints) | process_transition_keys
    )
    phase_functions = MappingProxyType(
        {
            func_name: processed_functions[func_name]
            for func_name in processed_functions
            if func_name not in excluded_from_functions
        }
    )

    transitions = _wrap_transitions(unflatten_regime_namespace(internal_transition))

    next_regime_func, next_regime_cells = _process_next_regime_cells(
        next_regime_cells_by_target=next_regime_cells_by_target,
        regime_params_template=regime_params_template,
    )

    return _CoreResult(
        functions=phase_functions,
        constraints=processed_constraints,
        transitions=transitions,
        stochastic_transition_names=stochastic_transition_names,
        next_regime_func=next_regime_func,
        next_regime_cells=next_regime_cells,
    )


def _process_next_regime_cells(
    *,
    next_regime_cells_by_target: Mapping[
        RegimeName, UserFunction | _CoarseTransitionCell
    ],
    regime_params_template: RegimeParamsTemplate,
) -> tuple[
    TransitionFunction | None, MappingProxyType[RegimeName, EconFunction] | None
]:
    """Process the canonical regime-transition cells of one phase.

    Dispatch on the cell type of the canonical per-target mapping:

    - empty mapping (terminal regime) ⇒ `(None, None)`
    - `_CoarseTransitionCell` cells ⇒ the shared underlying transition is
      processed once under the `next_regime` template key, so the engine
      evaluates it once and indexes per target
    - `MarkovTransition` cells (user per-target dict) ⇒ each cell is
      processed under its nested `template[target]["next_regime"]` branch

    Args:
        next_regime_cells_by_target: The canonical regime-transition cells,
            keyed by target regime name.
        regime_params_template: The regime's parameter template.

    Returns:
        Tuple of the processed coarse transition (`None` unless coarse) and
        the processed per-target cells (`None` unless per-target).

    """
    if not next_regime_cells_by_target:
        return None, None
    cells = tuple(next_regime_cells_by_target.values())
    first_cell = cells[0]
    if isinstance(first_cell, _CoarseTransitionCell):
        assert all(  # noqa: S101
            isinstance(cell, _CoarseTransitionCell)
            and cell.underlying is first_cell.underlying
            for cell in cells
        ), "Coarse regime-transition cells must share one underlying object."
        next_regime_func = _rename_params_to_qnames(
            func=cast("UserFunction", first_cell.underlying),
            regime_params_template=regime_params_template,
            param_key="next_regime",
        )
        return next_regime_func, None
    next_regime_cells = MappingProxyType(
        {
            target_regime_name: _rename_params_to_qnames(
                func=cast("UserFunction", cell),
                regime_params_template=regime_params_template,
                param_key=qname_from_tree_path((target_regime_name, "next_regime")),
            )
            for target_regime_name, cell in next_regime_cells_by_target.items()
        }
    )
    return None, next_regime_cells


def _extract_phase_transitions(*, phase_slice: RegimePhaseSpec) -> _TransitionBundles:
    """Transpose one canonical phase slice into per-target transition bundles.

    The slice's `state_transitions` values and its regime transition are
    canonical per-target mappings (`canonicalize_regimes` resolved
    reachability and desugared identities), so the extraction is a pure
    transpose: bundle each target regime's `next_<state>` laws plus its
    regime-transition cell under `"next_regime"`. A target reachable through
    the regime transition alone contributes a bundle holding only its cell.
    Stochastic process transitions are handled separately during internal
    function processing.

    Args:
        phase_slice: One canonical phase slice of the regime specification.

    Returns:
        Per-target transition bundles for internal processing.

    """
    if phase_slice.regime_transition is None:
        return {}

    per_target: _TransitionBundles = {}
    for state_name, canonical in phase_slice.state_transitions.items():
        for target_regime_name, law in cast(
            "Mapping[RegimeName, UserFunction]", canonical
        ).items():
            per_target.setdefault(target_regime_name, {})[f"next_{state_name}"] = law
    for target_regime_name, cell in cast(
        "Mapping[RegimeName, UserFunction | _CoarseTransitionCell]",
        phase_slice.regime_transition,
    ).items():
        per_target.setdefault(target_regime_name, {})["next_regime"] = cell

    return per_target


def _wrap_transitions(
    transitions: dict[RegimeName, dict[TransitionFunctionName, TransitionFunction]],
) -> TransitionFunctionsMapping:
    """Wrap nested transitions dict in MappingProxyType."""
    return MappingProxyType(
        {name: MappingProxyType(inner) for name, inner in transitions.items()}
    )


def _get_stochastic_transition_names(
    *,
    state_transitions: Mapping[StateName, object],
    variables: Variables,
) -> frozenset[TransitionFunctionName]:
    """Compute stochastic transition names from one phase's state transitions.

    Args:
        state_transitions: One phase's `state_transitions` slice.
        variables: States and actions of the regime with kind/topology/process tags.

    Returns:
        Frozenset of stochastic transition function names (e.g., "next_health").

    """
    markov_state_names: set[StateName] = set()
    for name, raw in state_transitions.items():
        if isinstance(raw, MarkovTransition) or (
            isinstance(raw, Mapping)
            and any(isinstance(v, MarkovTransition) for v in raw.values())
        ):
            markov_state_names.add(name)
    return frozenset(
        f"next_{name}" for name in markov_state_names | set(variables.process_names)
    )


def _process_one_function(
    *,
    func: UserFunction,
    regime_params_template: RegimeParamsTemplate,
    param_key: str,
    names_key: str | None = None,
) -> EconFunction:
    """Rename a function's params to qnames, or wrap an `AgeSpecializedFunction`.

    An `AgeSpecializedFunction` is wrapped as a marker.

    A plain function is renamed once. An `AgeSpecializedFunction` becomes a
    `_SpecializedEconFunction` whose `build(age)` renames the concrete function
    the wrapper produces for that age under the **same** `param_key` / `names_key`,
    so every age carries identical qnames — sound because the wrapper's call
    signature is age-invariant by contract.
    """
    if isinstance(func, AgeSpecializedFunction):
        concrete_build = func.build

        def build(age: float) -> EconFunction:
            return _rename_params_to_qnames(
                func=concrete_build(age),
                regime_params_template=regime_params_template,
                param_key=param_key,
                names_key=names_key,
            )

        return _SpecializedEconFunction(build=build, signature=func.signature)
    return _rename_params_to_qnames(
        func=func,
        regime_params_template=regime_params_template,
        param_key=param_key,
        names_key=names_key,
    )


def _rename_params_to_qnames(
    *,
    func: UserFunction,
    regime_params_template: RegimeParamsTemplate,
    param_key: str,
    names_key: str | None = None,
) -> EconFunction:
    """Rename function params to qualified names using dags.signature.rename_arguments.

    E.g., risk_aversion -> utility__risk_aversion.

    Args:
        func: The user function.
        regime_params_template: The parameter template for the regime.
        param_key: The qname prefix the renamed params carry (e.g., "utility",
            "retired__next_wealth").
        names_key: The template key under which the param names live, when it
            differs from `param_key` — a coarse law's names sit at the bare
            law name while its params bind per target. Defaults to
            `param_key`.

    Returns:
        The function with renamed parameters.

    """
    # Per-target keys are qnames (`<target>__<func>`) addressing a nested
    # template branch; walk the tree path instead of subscripting directly.
    branch: Mapping[str, object] = regime_params_template
    for part in tree_path_from_qname(names_key if names_key is not None else param_key):
        branch = cast("Mapping[str, object]", branch[part])
    param_names = list(branch)
    if not param_names:
        return cast("EconFunction", func)
    mapper = {p: qname_from_tree_path((param_key, p)) for p in param_names}

    return cast("EconFunction", rename_arguments(func, mapper=mapper))


def _engine_flat_param_names(
    *,
    regime_params_template: RegimeParamsTemplate,
    granular_param_expansions: MappingProxyType[FunctionName, tuple[str, ...]],
) -> frozenset[str]:
    """Return the regime's flat param names in the engine's binding vocabulary.

    Template names whose function key has a granular expansion are replaced
    by their per-target spellings (`<target>__<law>__<param>`); everything
    else passes through unchanged.
    """
    names: set[str] = set()
    for name in get_flat_param_names(regime_params_template):
        path = tree_path_from_qname(name)
        prefixes = granular_param_expansions.get(path[0]) if len(path) > 1 else None
        if prefixes:
            names.update(
                qname_from_tree_path((prefix, path[-1])) for prefix in prefixes
            )
        else:
            names.add(name)
    return frozenset(names)


def _granular_param_expansions(
    *,
    nested_transitions_by_phase: tuple[_TransitionBundles, ...],
    regime_params_template: RegimeParamsTemplate,
) -> MappingProxyType[FunctionName, tuple[str, ...]]:
    """Map each coarse-template law key to its granular qname prefixes.

    A state law whose params the template keys coarsely binds them per target
    in the engine; this collects, across the given phase bundles, every
    `<target>__<law>` prefix for laws whose names live at the bare law name
    (mirroring `_extract_template_names_key`) and that carry params at all.
    Canonical flat params materialize one shared leaf per prefix.
    """
    expansions: dict[FunctionName, set[str]] = {}
    for bundles in nested_transitions_by_phase:
        for target_regime_name, bundle in bundles.items():
            for law_name in bundle:
                if law_name == "next_regime":
                    continue
                qname = qname_from_tree_path((target_regime_name, law_name))
                names_key = _extract_template_names_key(qname, regime_params_template)
                if names_key != qname and regime_params_template.get(names_key):
                    expansions.setdefault(names_key, set()).add(qname)
    return MappingProxyType(
        {law_name: tuple(sorted(v)) for law_name, v in expansions.items()}
    )


def _extract_template_names_key(
    func_name: str,
    regime_params_template: RegimeParamsTemplate,
) -> str:
    """Extract the template key under which a function's param names live.

    The template mirrors the user's coarseness — a per-target dict yields
    params nested under the target (`template[target_regime][func]`), a broadcast
    law a single coarse `next_<state>` key — while the engine-side function
    names are always target-prefixed (canonical form). The template therefore
    decides where the names live:

    - "work__next_health" with `template["work"]["next_health"]` present
      (user wrote a per-target dict) ⇒ "work__next_health"
    - "work__next_wealth" without such a branch (broadcast law) ⇒
      "next_wealth"
    - unprefixed names ⇒ unchanged

    Either way the params *bind* under the engine function's qname; for
    broadcast laws the canonical flat params materialize one shared leaf
    per target (`Regime.granular_param_expansions`).
    """
    path = tree_path_from_qname(func_name)
    if len(path) > 1:
        suffix = qname_from_tree_path(path[1:])
        target_branch = regime_params_template.get(path[0])
        if isinstance(target_branch, Mapping) and isinstance(
            target_branch.get(suffix), Mapping
        ):
            return func_name
        return suffix
    return func_name


def _get_discrete_markov_next_function(
    *, func: UserFunction, grid: Int1D
) -> UserFunction:
    @with_signature(args=None, return_annotation="Int1D")
    @functools.wraps(func)
    def next_func(**kwargs: Any) -> Int1D:  # noqa: ANN401, ARG001
        return grid

    return next_func


def _get_stochastic_next_function_for_process(
    *, name: str, grid: Float1D
) -> UserFunction:
    """Get function returning the indices in the vf arr of the next process states."""

    @with_signature(args={f"{name}": "ContinuousState"}, return_annotation="Int1D")
    def next_func(**kwargs: Any) -> Int1D:  # noqa: ARG001, ANN401
        return jnp.arange(grid.shape[0], dtype=jnp.int32)

    return next_func


def _process_family(grid: _ContinuousStochasticProcess) -> Family:
    """Map a supported process to its state-conditioned family (audit F2).

    Only families whose transition CDF carries ``sigma`` can express a fixed-node,
    state-conditioned ``sigma``: CDF-binned ``NormalIIDProcess`` and
    ``TauchenAR1Process``. Gauss-Hermite IID (nodes scale with ``sigma``) and
    Rouwenhorst (``rho``-only transition) are rejected at construction.
    """
    if isinstance(grid, NormalIIDProcess | TauchenAR1Process) and grid.gauss_hermite:
        # Blanket rejection, per the stated v1 scope (code-review F6). GH-placed nodes
        # are derived from the *grid* sigma, so a state-conditioned sigma would bin a
        # different law on a quadrature rule chosen for another one; untested here.
        msg = (
            "state-conditioned sigma is not supported for Gauss-Hermite node placement "
            "(the nodes are built from the grid sigma); use gauss_hermite=False "
            "(CDF binning)."
        )
        raise ModelInitializationError(msg)
    if isinstance(grid, NormalIIDProcess):
        return "iid_normal"
    if isinstance(grid, TauchenAR1Process):
        return "tauchen"
    msg = (
        "state-conditioned sigma is only supported for CDF-binned NormalIIDProcess "
        f"and TauchenAR1Process (audit F2); got {type(grid).__name__}."
    )
    raise ModelInitializationError(msg)


def _validate_conditioning_codes_agree_across_regimes(
    *,
    on: str,
    all_grids: Mapping[RegimeName, Mapping[StateOrActionName, Grid]],
) -> None:
    """Every regime carrying the conditioning state must map categories to codes alike.

    The sigma array is ordered by the *target* regime's grid, but the code gathering
    it is the *current* (source) state's. Those agree only if the category-to-code maps
    do; a regime that relabels `{low: 0, high: 1}` to `{high: 0, low: 1}` would silently
    swap the two volatilities (code-review F4). Rather than thread the source grid
    through every seam, v1 requires one shared map and rejects otherwise.

    Round-2 review F3 noted this global scan is not edge-precise (it fails a
    state-less reachable source late rather than at the guard, and over-rejects
    disconnected components reusing the name), but established **no silent policy
    error**. Kept deliberately for v1: the conservative rule is safe, and the
    per-edge alternative would need source-reachability plumbing that could itself
    introduce the sigma-swap this prevents. Accepted limitation, not an oversight.
    """
    maps = {
        regime: dict(zip(grid.categories, grid.codes, strict=True))
        for regime, grids in all_grids.items()
        if isinstance(grid := grids.get(on), DiscreteGrid)
    }
    distinct = {tuple(sorted(m.items())) for m in maps.values()}
    if len(distinct) > 1:
        msg = (
            f"state_conditioned.on='{on}' must map categories to the same integer "
            f"codes in every regime that carries it, because the per-category sigma is "
            f"indexed by that code; got {maps}."
        )
        raise ModelInitializationError(msg)


def _validate_conditioned_sigmas(by: Mapping[str, float]) -> None:
    """Every per-category sigma must be a finite, strictly positive number.

    ``None``, ``NaN`` and ``inf`` all sail through a bare ``v <= 0`` test and then
    poison every transition row silently (code-review F3).
    """
    bad = {
        k: v
        for k, v in by.items()
        if v is None or not math.isfinite(float(v)) or float(v) <= 0.0
    }
    if bad:
        msg = f"state_conditioned.by values must be finite positive sigmas; got {bad}"
        raise ModelInitializationError(msg)


#: A CDF row bins on node midpoints, so it needs at least one interior edge.
_MIN_CONDITIONED_NODES = 2


def _validate_conditioned_grid_is_fixed(
    *, name: str, grid: _ContinuousStochasticProcess
) -> None:
    """A state-conditioned process must have every grid parameter fixed at build time.

    The conditioned branch is chosen *before* the runtime-parameter mechanism, so a
    parameter left for runtime is never bound: ``get_gridpoints()`` returns all-NaN and
    the closure captures those nodes permanently (code-review F3). Reject instead.
    """
    if not grid.is_fully_specified:
        missing = ", ".join(sorted(grid.params_to_pass_at_runtime))
        msg = (
            f"state-conditioned process '{name}' requires every grid parameter fixed "
            f"at construction (v1); {missing} would be passed at runtime. Pass them to "
            f"{type(grid).__name__}(...), or drop state_conditioned."
        )
        raise ModelInitializationError(msg)
    nodes = grid.get_gridpoints()
    if not bool(jnp.all(jnp.isfinite(nodes))):
        msg = (
            f"state-conditioned process '{name}' resolved to non-finite nodes: {nodes}"
        )
        raise ModelInitializationError(msg)
    # Finiteness is not enough. The direct-CDF row bins on the MIDPOINTS of these nodes,
    # so a strictly increasing axis of >= 2 points is what makes the CDF differences
    # probabilities at all (code-review round 2, F2). All of this is reachable from the
    # public API: `sigma=-0.3` or `n_std=-3.0` yields a DESCENDING axis (pylcm does not
    # require a positive sigma), `sigma=0.0` collapses every node, and a large `mu` in
    # float32 rounds them together. The descending case is the dangerous one — the row
    # still sums to 1.0, so a row-sum check passes while masses go negative and a
    # continuation decision flips sign.
    if nodes.ndim != 1 or nodes.shape[0] < _MIN_CONDITIONED_NODES:
        msg = (
            f"state-conditioned process '{name}' needs a 1-D axis of at least "
            f"{_MIN_CONDITIONED_NODES} nodes; got shape {nodes.shape}. A single node "
            f"leaves the CDF row no bin edges."
        )
        raise ModelInitializationError(msg)
    if not bool(jnp.all(jnp.diff(nodes) > 0)):
        msg = (
            f"state-conditioned process '{name}' resolved to a node axis that is not "
            f"strictly increasing: {nodes}. Midpoint-CDF binning would return negative "
            f"transition masses (which still sum to 1.0, so this fails silently). "
            f"Check for a negative or zero sigma / n_std, or nodes collapsed by "
            f"float32 precision at a large mu."
        )
        raise ModelInitializationError(msg)


def _get_conditioned_weights_func(
    *,
    name: str,
    grid: _ContinuousStochasticProcess,
    sc: StateConditioned,
    grids: Mapping[StateOrActionName, Grid],
) -> UserFunction:
    """Weights function for a state-conditioned process (direct-CDF, audit F1/F5/F6).

    The transition row is computed DIRECTLY at the from-value on the FIXED common nodes
    (from the scalar ``sigma``), with the per-regime ``sigma`` gathered by the
    conditioning state's integer code. No precomputed-row interpolation (F1).
    """
    conditioning_grid = grids.get(sc.on)
    if not isinstance(conditioning_grid, DiscreteGrid):
        msg = (
            f"state_conditioned.on='{sc.on}' must name a DiscreteGrid state in the "
            f"same regime as the process."
        )
        raise ModelInitializationError(msg)
    _validate_conditioned_sigmas(sc.by)
    family = _process_family(grid)
    _validate_conditioned_grid_is_fixed(name=name, grid=grid)
    nodes = grid.get_gridpoints()
    sigma_by_code = sigma_array_by_code(conditioning_grid, sc.by)
    fixed = dict(grid.params)
    mu_fixed, rho_fixed = fixed["mu"], fixed.get("rho")

    args = {name: "ContinuousState", sc.on: "DiscreteState"}

    @with_signature(args=args, return_annotation="FloatND", enforce=False)
    def weights_func_conditioned(*a: FloatND, **kwargs: FloatND) -> Float1D:  # noqa: ARG001
        sigma = gather_sigma(sigma_by_code, kwargs[sc.on])
        return conditioned_row(
            family=family,
            nodes=nodes,
            sigma=sigma,
            from_value=kwargs[name],
            mu=mu_fixed,
            rho=rho_fixed,
        )

    return weights_func_conditioned


def _get_weights_func_for_process(
    *,
    name: str,
    grid: _ContinuousStochasticProcess,
    grids: Mapping[StateOrActionName, Grid] = MappingProxyType({}),
) -> UserFunction:
    """Get function that uses linear interpolation to calculate the process weights.

    For processes whose params are supplied at runtime, the grid points and
    transition probabilities are computed inside JIT from those runtime params. For a
    state-conditioned process the row is instead computed directly at the from-value
    (``_get_conditioned_weights_func``), and `grids` is that regime's grid mapping, from
    which the conditioning state is resolved.

    """
    sc = grid.state_conditioned
    if sc is not None:
        # `sc.on` may be absent from `grids`; the callee raises a clear message naming
        # the offending state.
        return _get_conditioned_weights_func(name=name, grid=grid, sc=sc, grids=grids)

    if grid.params_to_pass_at_runtime:
        n_points = grid.n_points
        fixed_params = dict(grid.params)
        runtime_param_names = {
            qname_from_tree_path((name, p)): p for p in grid.params_to_pass_at_runtime
        }
        args = {
            name: "ContinuousState",
            **dict.fromkeys(runtime_param_names, "FloatND"),
        }

        @with_signature(args=args, return_annotation="FloatND", enforce=False)
        def weights_func_runtime(*a: FloatND, **kwargs: FloatND) -> Float1D:  # noqa: ARG001
            # `grid.params` is canonical (0-d JAX scalars) from its own
            # boundary cast; `kwargs` arrive as JAX tracers from JIT.
            process_kw: dict[str, FloatND | IntND] = {
                **fixed_params,
                **{raw: kwargs[qn] for qn, raw in runtime_param_names.items()},
            }
            gridpoints = grid.compute_gridpoints(**process_kw)
            transition_probs = grid.compute_transition_probs(**process_kw)
            coord = get_irreg_coordinate(value=kwargs[name], points=gridpoints)
            return map_coordinates(
                input=transition_probs,
                coordinates=[
                    jnp.full(n_points, fill_value=coord),
                    jnp.arange(n_points, dtype=jnp.int32),
                ],
            )

        return weights_func_runtime

    gridpoints = grid.get_gridpoints()
    transition_probs = grid.get_transition_probs()

    @with_signature(
        args={f"{name}": "ContinuousState"},
        return_annotation="FloatND",
        enforce=False,
    )
    def weights_func(*args: FloatND, **kwargs: FloatND) -> Float1D:  # noqa: ARG001
        coordinate = get_irreg_coordinate(value=kwargs[f"{name}"], points=gridpoints)
        return map_coordinates(
            input=transition_probs,
            coordinates=[
                jnp.full(grid.n_points, fill_value=coordinate),
                jnp.arange(grid.n_points, dtype=jnp.int32),
            ],
        )

    return weights_func


def _validate_categoricals(
    user_regimes: Mapping[RegimeName, UserRegime],
) -> None:
    """Validate that simple transitions don't span mismatched discrete grids.

    When a non-per-target-dict transition is used for a `DiscreteGrid` state, the same
    function is applied to all target regimes. If a target regime has a different number
    of categories for that state, JAX silently clips indices producing wrong results.

    Also validates that the `ordered` flag is consistent across regimes for the same
    discrete state variable. Mixed ordered flags (one True, one False) are not allowed.

    When both regimes are ordered with different categories, the per-regime orderings
    are merged via topological sort. If the merge is ambiguous or contradictory, an
    error is raised.

    Raises:
        ModelInitializationError: If a category count mismatch or ordered flag
            inconsistency is found.

    """
    error_messages: list[str] = []

    for source_name, source_regime in user_regimes.items():
        if source_regime.terminal:
            continue

        for state_name, raw in source_regime.state_transitions.items():
            source_grid = _get_simple_transition_discrete_grid(
                source_regime, state_name, raw
            )
            if source_grid is None:
                continue

            for target_regime_name, target_regime in user_regimes.items():
                target_grid = target_regime.states.get(state_name)
                if not isinstance(target_grid, DiscreteGrid):
                    continue

                if source_grid.categories != target_grid.categories:
                    error_messages.append(
                        f"Discrete state '{state_name}' in regime '{source_name}' "
                        f"has categories {source_grid.categories}, but regime "
                        f"'{target_regime_name}' has categories "
                        f"{target_grid.categories}. A single transition function "
                        f"cannot map between different category sets — use a "
                        f"per-target dict in state_transitions to specify the "
                        f"mapping for each target regime.",
                    )

    # Validate ordered flag consistency across regimes
    _validate_ordered_flags(user_regimes, error_messages)

    if error_messages:
        raise ModelInitializationError(format_messages(error_messages))


def compute_merged_discrete_categories(
    user_regimes: Mapping[RegimeName, UserRegime],
) -> tuple[dict[str, tuple[str, ...]], dict[str, bool]]:
    """Compute merged categories and ordered flags for all discrete variables.

    Returns:
        Tuple of (categories dict, ordered_flags dict).

    """
    var_grids: dict[str, list[tuple[str, DiscreteGrid]]] = {}
    for regime_name, user_regime in user_regimes.items():
        for var_name, grid in {**user_regime.states, **user_regime.actions}.items():
            if isinstance(grid, DiscreteGrid):
                var_grids.setdefault(var_name, []).append((regime_name, grid))

    categories: dict[str, tuple[str, ...]] = {}
    ordered_flags: dict[str, bool] = {}
    for var_name, entries in var_grids.items():
        first_grid = entries[0][1]
        ordered_flags[var_name] = first_grid.ordered

        if len(entries) == 1 or not first_grid.ordered:
            categories[var_name] = first_grid.categories
            continue

        all_cats = [grid.categories for _, grid in entries]
        if len(set(all_cats)) <= 1:
            categories[var_name] = first_grid.categories
            continue

        merged = _merge_ordered_categories(
            [(rn, grid.categories) for rn, grid in entries]
        )
        # Validation already passed, so merge must succeed
        assert merged is not None  # noqa: S101
        categories[var_name] = merged

    return categories, ordered_flags


def _validate_ordered_flags(
    user_regimes: Mapping[RegimeName, UserRegime],
    error_messages: list[str],
) -> None:
    """Validate that the ordered flag is consistent for each discrete variable.

    For each discrete state/action variable that appears in multiple regimes:
    - Mixed ordered flags (True in one, False in another) -> error.
    - Both ordered with different categories -> merge via topological sort; ambiguous
      or contradictory merges -> error.
    """
    # Collect per-variable: list of (regime_name, grid)
    var_grids: dict[str, list[tuple[str, DiscreteGrid]]] = {}
    for regime_name, user_regime in user_regimes.items():
        for var_name, grid in {**user_regime.states, **user_regime.actions}.items():
            if isinstance(grid, DiscreteGrid):
                var_grids.setdefault(var_name, []).append((regime_name, grid))

    for var_name, entries in var_grids.items():
        if len(entries) < 2:  # noqa: PLR2004
            continue

        ordered_flags = {grid.ordered for _, grid in entries}
        if len(ordered_flags) > 1:
            regime_details = ", ".join(
                f"'{rn}' (ordered={g.ordered})" for rn, g in entries
            )
            error_messages.append(
                f"Discrete variable '{var_name}' has inconsistent ordered flags "
                f"across regimes: {regime_details}. All regimes must agree on "
                f"whether the variable is ordered or unordered.",
            )
            continue

        is_ordered = next(iter(ordered_flags))
        if not is_ordered:
            continue

        # Both ordered — check if categories differ and need merging
        all_categories = [grid.categories for _, grid in entries]
        if len(set(all_categories)) <= 1:
            continue

        # Attempt topological sort merge
        merged = _merge_ordered_categories(
            [(rn, grid.categories) for rn, grid in entries]
        )
        if merged is None:
            regime_details = ", ".join(
                f"'{rn}': {list(g.categories)}" for rn, g in entries
            )
            error_messages.append(
                f"Discrete variable '{var_name}' is ordered in multiple regimes "
                f"with different categories that cannot be merged into a unique "
                f"total order. Regime orderings: {regime_details}.",
            )


def _merge_ordered_categories(
    regime_categories: list[tuple[str, tuple[str, ...]]],
) -> tuple[str, ...] | None:
    """Merge per-regime category orderings into a total order via topological sort.

    Each regime contributes a chain of ordering constraints from its field declaration
    order. Returns the unique total order if one exists, or None if ambiguous or
    contradictory.
    """
    edges, all_nodes, in_degree = _build_ordering_graph(regime_categories)
    return _unique_topological_sort(edges, all_nodes, in_degree)


def _build_ordering_graph(
    regime_categories: list[tuple[str, tuple[str, ...]]],
) -> tuple[dict[str, set[str]], set[str], dict[str, int]]:
    """Build a directed graph of ordering constraints from regime categories."""
    edges: dict[str, set[str]] = defaultdict(set)
    all_nodes: set[str] = set()
    in_degree: dict[str, int] = defaultdict(int)

    for _regime_name, categories in regime_categories:
        for cat in categories:
            all_nodes.add(cat)
            if cat not in in_degree:
                in_degree[cat] = 0
        for i in range(len(categories) - 1):
            a, b = categories[i], categories[i + 1]
            if b not in edges[a]:
                edges[a].add(b)
                in_degree[b] += 1

    return edges, all_nodes, in_degree


def _unique_topological_sort(
    edges: dict[str, set[str]],
    all_nodes: set[str],
    in_degree: dict[str, int],
) -> tuple[str, ...] | None:
    """Return the unique topological order, or None if ambiguous or cyclic."""
    queue = [n for n in all_nodes if in_degree[n] == 0]
    result: list[str] = []

    while queue:
        if len(queue) > 1:
            return None
        node = queue[0]
        queue = []
        result.append(node)
        for neighbor in sorted(edges.get(node, set())):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if len(result) != len(all_nodes):
        return None

    return tuple(result)


def _get_simple_transition_discrete_grid(
    user_regime: UserRegime,
    state_name: StateName,
    raw: object,
) -> DiscreteGrid | None:
    """Return the source DiscreteGrid for a simple transition.

    Returns None if the transition is a per-target dict, an identity law
    (fixed state), not a DiscreteGrid, or the state is not present in the
    source regime.

    """
    # Per-target dicts handle category differences explicitly
    if isinstance(raw, Mapping) and not isinstance(raw, MarkovTransition):
        return None
    # An identity law (fixed state) only maps within its own regime
    if isinstance(raw, _IdentityTransition):
        return None
    # Target-only state — no source grid to compare
    if state_name not in user_regime.states:
        return None
    source_grid = user_regime.states[state_name]
    return source_grid if isinstance(source_grid, DiscreteGrid) else None


def build_regime_transition_probs_functions(
    *,
    functions: EconFunctionsMapping,
    compute_regime_transition_probs: TransitionFunction | None,
    grids: MappingProxyType[StateOrActionName, Grid],
    regime_names_to_ids: RegimeNamesToIds,
    flat_param_names: frozenset[str],
    is_stochastic: bool,
    enable_jit: bool,
    phase: Literal["solve", "simulate"],
    next_regime_cells: MappingProxyType[RegimeName, EconFunction] | None = None,
) -> RegimeTransitionFunction | VmappedRegimeTransitionFunction:
    """Build a regime transition probability function for the given phase.

    Args:
        functions: Immutable mapping of function names to internal user functions.
        compute_regime_transition_probs: The user's coarse next_regime
            function; `None` for per-target regime transitions.
        grids: Immutable mapping of state and action variable names to grid objects.
        regime_names_to_ids: Immutable mapping of regime names to integer indices.
        flat_param_names: Frozenset of flat parameter names in the engine's
            binding vocabulary.
        is_stochastic: Whether the regime transition is stochastic.
        enable_jit: Whether to JIT-compile the functions.
        phase: Which phase to build for.
        next_regime_cells: Per-target regime transition probability functions;
            `None` for coarse regime transitions.

    """
    if next_regime_cells is not None:
        wrapped_regime_transition_probs = _assemble_granular_regime_transition_probs(
            next_regime_cells=next_regime_cells
        )
    else:
        if compute_regime_transition_probs is None:
            msg = "Either a coarse regime transition or per-target cells is required."
            raise ModelInitializationError(msg)
        # Wrap deterministic next_regime to return one-hot probability array
        if is_stochastic:
            probs_func = compute_regime_transition_probs
        else:
            probs_func = _wrap_deterministic_regime_transition(
                func=compute_regime_transition_probs,
                regime_names_to_ids=regime_names_to_ids,
            )

        # Wrap to convert array output to dict format
        wrapped_regime_transition_probs = _wrap_regime_transition_probs(
            func=probs_func, regime_names_to_ids=regime_names_to_ids
        )

    functions_pool = dict(functions) | {
        "regime_transition_probs": wrapped_regime_transition_probs
    }

    next_regime = concatenate_functions(
        functions=functions_pool,
        targets="regime_transition_probs",
        return_type="dict",
        enforce_signature=False,
        set_annotations=True,
    )
    if phase == "solve":
        return jax.jit(next_regime) if enable_jit else next_regime

    sig_args = list(inspect.signature(next_regime).parameters)

    # We do this because a transition function without any parameters will throw
    # an error with vmap
    next_regime_accepting_all = with_signature(
        next_regime,
        args=sig_args + [state for state in grids if state not in sig_args],
    )

    next_regime_vmapped = vmap_1d(
        func=next_regime_accepting_all,
        variables=_get_vmap_params(
            all_args=tuple(inspect.signature(next_regime_accepting_all).parameters),
            flat_param_names=flat_param_names,
        ),
    )

    return jax.jit(next_regime_vmapped) if enable_jit else next_regime_vmapped


def _assemble_granular_regime_transition_probs(
    *,
    next_regime_cells: MappingProxyType[RegimeName, EconFunction],
) -> Callable[..., MappingProxyType[RegimeName, FloatND]]:
    """Assemble per-target probability cells into the probs-dict contract.

    Produces the same regime-name → probability mapping that
    `_wrap_regime_transition_probs` builds from a coarse probability vector,
    restricted to the declared targets: omitted regimes are structurally
    unreachable.

    Args:
        next_regime_cells: Per-target probability functions with qname params.

    Returns:
        A function over the union of the cells' arguments returning an
        immutable mapping of declared regime names to probability scalars.

    """
    cell_arg_names = {
        target_regime_name: tuple(
            name for name in get_annotations(cell) if name != "return"
        )
        for target_regime_name, cell in next_regime_cells.items()
    }
    merged_annotations: dict[str, str] = {}
    for cell in next_regime_cells.values():
        annotations = get_annotations(cell)
        annotations.pop("return", None)
        merged_annotations |= annotations
    return_annotation = MappingProxyType[RegimeName, FloatND]

    @with_signature(args=merged_annotations, return_annotation=return_annotation)
    def regime_transition_probs(
        **kwargs: FloatND | IntND | int,
    ) -> MappingProxyType[RegimeName, FloatND]:
        return MappingProxyType(
            {
                target_regime_name: jnp.asarray(
                    cell(
                        **{
                            name: kwargs[name]
                            for name in cell_arg_names[target_regime_name]
                        }
                    )
                )
                for target_regime_name, cell in next_regime_cells.items()
            }
        )

    # Pin `__annotations__` on the final wrapper: `concatenate_functions`
    # reads `__annotations__` (not `__signature__`) to reconcile the DAG.
    regime_transition_probs.__annotations__ = {
        **merged_annotations,
        "return": return_annotation,
    }
    return regime_transition_probs


def _wrap_regime_transition_probs(
    *,
    func: TransitionFunction,
    regime_names_to_ids: RegimeNamesToIds,
) -> Callable[..., MappingProxyType[RegimeName, FloatND]]:
    """Wrap next_regime function to convert array output to dict format.

    The next_regime function returns a JAX array of probabilities indexed by
    the regime's id. This wrapper converts the array to dict format for internal
    processing.

    Args:
        func: The user's next_regime function (with qname parameters).
        regime_names_to_ids: Immutable mapping of regime names to integer indices.

    Returns:
        A wrapped function that returns an immutable mapping of regime
        names to probability scalars.

    """
    # Get regime names in index order from regime_names_to_ids. Coerce
    # `ScalarInt` ids to Python `int` so `sorted` has a comparable key.
    regime_names_by_id: list[tuple[int, str]] = sorted(
        [(int(idx), name) for name, idx in regime_names_to_ids.items()],
        key=lambda x: x[0],
    )
    regime_names = [name for _, name in regime_names_by_id]

    # `wrapped` converts `func`'s probability array into a regime-name → prob
    # mapping. The return annotation describes that mapping; `func`'s own
    # return annotation (a bare probability array) does not survive the
    # conversion and must not be carried through.
    annotations = get_annotations(func)
    annotations.pop("return", None)
    return_annotation = MappingProxyType[RegimeName, FloatND]

    @with_signature(
        args=annotations,
        return_annotation=return_annotation,
    )
    @functools.wraps(func)
    def wrapped(
        *args: FloatND | IntND | int,
        **kwargs: FloatND | IntND | int,
    ) -> MappingProxyType[RegimeName, FloatND]:
        result = func(*args, **kwargs)
        # Convert array to dict using ordering by regime id
        return MappingProxyType(
            {name: result[idx] for idx, name in enumerate(regime_names)}
        )

    # Pin `__annotations__` on the final wrapper: `concatenate_functions`
    # reads `__annotations__` (not `__signature__`) to reconcile the DAG, and
    # the decorator stack can drop them when `func` carries deferred (PEP 649)
    # annotations through `functools.wraps`.
    wrapped.__annotations__ = {**annotations, "return": return_annotation}
    return wrapped


def _wrap_deterministic_regime_transition(
    *,
    func: TransitionFunction,
    regime_names_to_ids: RegimeNamesToIds,
) -> TransitionFunction:
    """Wrap deterministic next_regime to return one-hot probability array.

    Converts a deterministic regime transition function that returns an integer
    regime ID to a function that returns a one-hot probability array, matching
    the interface of stochastic regime transitions.

    Args:
        func: The user's deterministic next_regime function (returns int).
        regime_names_to_ids: Immutable mapping of regime names to integer indices.

    Returns:
        A wrapped function that returns a one-hot probability array.

    """
    n_regimes = len(regime_names_to_ids)

    # Preserve original annotations but update return type
    annotations = {k: v for k, v in get_annotations(func).items() if k != "return"}

    @with_signature(args=annotations, return_annotation="FloatND")
    @functools.wraps(func)
    def wrapped(
        *args: FloatND | IntND | BoolND | float | MappingLeaf | SequenceLeaf,
        **kwargs: FloatND | IntND | BoolND | float | MappingLeaf | SequenceLeaf,
    ) -> FloatND:
        regime_idx = func(*args, **kwargs)
        return jax.nn.one_hot(regime_idx, n_regimes)

    # Pin `__annotations__` on the final wrapper: `concatenate_functions`
    # reads `__annotations__` (not `__signature__`) to reconcile the DAG, and
    # the decorator stack can drop them when `func` carries deferred (PEP 649)
    # annotations through `functools.wraps`.
    wrapped.__annotations__ = {**annotations, "return": "FloatND"}
    return wrapped  # ty: ignore[invalid-return-type]


def _get_vmap_params(
    *,
    all_args: tuple[str, ...],
    flat_param_names: frozenset[str],
) -> tuple[str, ...]:
    """Get parameter names that should be vmapped (states and actions)."""
    non_vmap = {"period", "age"} | flat_param_names
    return tuple(arg for arg in all_args if arg not in non_vmap)


def _co_map_state_names(
    *,
    state_names: tuple[StateName, ...],
    grids: MappingProxyType[StateOrActionName, Grid],
    transitions: TransitionFunctionsMapping,
) -> tuple[StateName, ...]:
    """Return the distributed, never-transitioning states, in state-axis order.

    A state qualifies when its grid is distributed and its law of motion is the
    identity in every target bundle that carries it — so its next value equals its
    current value, and the continuation V can be read from the device-local slice
    rather than all-gathered. Distributed states sort first in `state_names`, so the
    result is a leading prefix of it (what the co-map requires).
    """
    co_map: list[StateName] = []
    for name in state_names:
        grid = grids.get(name)
        if grid is None or not grid.distributed:
            continue
        next_key = f"next_{name}"
        carrying = [
            bundle[next_key] for bundle in transitions.values() if next_key in bundle
        ]
        if carrying and all(
            getattr(law, "_is_auto_identity", False) for law in carrying
        ):
            co_map.append(name)
    return tuple(co_map)


def _grid_identity(grid: object) -> Hashable:
    """A hashable identity of a continuous grid's actual nodes (for dedup signatures).

    Keys on the grid's resolved nodes (``to_jax()``) so grids of the same class and
    point count but different geometry get *distinct* identities: e.g. two piecewise
    grids with equal total ``n_points`` but different breakpoints, or a custom grid
    whose geometry lives only in ``to_jax()``. The concrete *class object* is included
    so distinct grid types never collide even at identical nodes. A grid that is
    constant over age yields identical node arrays across periods, hence an identical
    identity, so the age-invariant fast path (`_build_period_state_axes` returning
    ``None``, shared continuation caches) is preserved. It keys both the per-period
    current-state axes and the continuation-cache signatures
    (`_continuation_grid_signature`), so the two can never disagree.

    Audit F1: the prior ``(class, n_points)`` fallback for grids without ``start/stop``
    or a ``points`` attribute (every piecewise grid) let geometry-changing grids collide
    and silently reuse the wrong axes/kernels. Re-review: the cheap branches must key on
    the grid's *exact* built-in type, never on the mere presence of ``start``/``stop``/
    ``points`` — a custom grid may expose those attributes yet derive its nodes from
    further ones (a power-spacing exponent, say), so duck-typing reintroduced exactly
    the collision this function exists to prevent.

    Round-3 re-review F1: runtime-supplied points are a property of the *mode*, not of
    the exact class. `V._get_coordinate_finder` dispatches the runtime-points path on
    ``isinstance(grid, IrregSpacedGrid)``, so an `IrregSpacedGrid` subclass with
    ``pass_points_at_runtime`` is a supported runtime grid there; keying identity on the
    exact type alone sent it to the node branch, where its inherited ``to_jax()`` must
    raise. The runtime-mode test therefore comes first and mirrors the interpolation
    dispatch. Concrete subclasses still fall through to the node fingerprint, so an
    overridden ``to_jax()`` remains geometry-sensitive.

    Round-4 re-review F1: there is no cheap shortcut for the uniform built-ins either.
    Keying `LinSpacedGrid`/`LogSpacedGrid` on ``(start, stop, n_points)`` as Python
    floats collapsed ``-0.0`` and ``+0.0``, which ``jnp.linspace`` faithfully preserves
    as *different* endpoint bits. Only the resolved nodes decide identity now: a key
    derived from a grid's *description* is one restatement away from disagreeing with
    the array the kernel is actually handed.
    """
    # Runtime mode first, and by `isinstance` — mirrors V._get_coordinate_finder. Nodes
    # are substituted at solve time and are not build-time closure constants, so the
    # concrete class + shape is the whole of the build-time identity.
    if isinstance(grid, IrregSpacedGrid) and grid.pass_points_at_runtime:
        return (type(grid), int(grid.n_points))
    # Every concrete grid — built-in uniform, Irregular, Piecewise, or custom — carries
    # its geometry only through its resolved nodes.
    return (type(grid), *_node_fingerprint(cast("ContinuousGrid", grid).to_jax()))


def _node_fingerprint(nodes: Any) -> tuple[Hashable, ...]:  # noqa: ANN401
    """Fingerprint a node array by dtype, weak-type, shape and raw bytes.

    One bulk device-to-host transfer; iterating the array elementwise instead costs
    orders of magnitude more on realistically sized grids.

    The exact `np.dtype` *object* is the key, never ``dtype.str``: the latter is not
    injective over the extended floating types JAX supports — ``float8_e4m3fnuz`` and
    ``float8_e5m2fnuz`` both report ``'<V1'``, so same-shape arrays with identical raw
    bytes decode to different numbers while comparing equal. (Audit round-4 F1.)

    `weak_type` is read off the *JAX* array before ``np.asarray`` drops it: two axes
    can agree on dtype, shape and bytes yet promote differently in the shared trace,
    which changes the argmax. Varying it across ages violates the
    only-node-values-may-vary contract, so this is defence in depth — it turns that
    violation into a construction-time error instead of a silent mis-share. No
    built-in grid yields a weak array, so it cannot split a supported grid.
    (Audit round-5 hardening note.)
    """
    arr = np.asarray(nodes)
    weak_type = bool(getattr(nodes, "weak_type", False))
    return (arr.dtype, weak_type, arr.shape, arr.tobytes())


def _continuation_grid_signature(
    regime_to_v_interp: MappingProxyType[RegimeName, VInterpolationInfo],
    period_targets: tuple[RegimeName, ...] | None = None,
) -> Hashable:
    """Fingerprint the continuous-state grids of the period's target regimes.

    `period_targets` restricts the fingerprint to the regimes the period's kernel
    actually interpolates. Fingerprinting every regime in the global mapping instead
    (audit F4) splits otherwise identical period groups whenever an unreachable
    regime's grid moves, costing compilations for no correctness gain. `None` keeps
    the all-regime behaviour for callers without a target list.
    """
    targets = sorted(regime_to_v_interp) if period_targets is None else period_targets
    return tuple(
        (
            rname,
            sname,
            _grid_identity(regime_to_v_interp[rname].continuous_states[sname]),
        )
        for rname in sorted(targets)
        if rname in regime_to_v_interp
        for sname in sorted(regime_to_v_interp[rname].continuous_states)
    )


def _build_period_state_axes(
    *,
    regime_name: RegimeName,
    period_to_regime_v_interp: (
        MappingProxyType[int, MappingProxyType[RegimeName, VInterpolationInfo]] | None
    ),
    active_periods: tuple[int, ...],
) -> MappingProxyType[int, MappingProxyType[StateOrActionName, Float1D]] | None:
    """Per-period node arrays for this regime's age-varying continuous states.

    A continuous state is age-varying iff its resolved grid identity differs across
    the regime's active periods. Returns `{period: {state: nodes}}` for those states
    (the current period's grid nodes, so V is tabulated on the current grid), or
    `None` when nothing is age-varying (age-invariant regimes are unchanged).
    """
    if period_to_regime_v_interp is None or not active_periods:
        return None
    per_period_states = {
        period: period_to_regime_v_interp[period][regime_name].continuous_states
        for period in active_periods
    }
    first = per_period_states[active_periods[0]]
    varying = [
        name
        for name in first
        if len({_grid_identity(per_period_states[p][name]) for p in active_periods}) > 1
    ]
    if not varying:
        return None
    return MappingProxyType(
        {
            period: MappingProxyType(
                {name: per_period_states[period][name].to_jax() for name in varying}
            )
            for period in active_periods
        }
    )


def _build_Q_and_F_per_period(
    *,
    regimes_to_active_periods: MappingProxyType[RegimeName, tuple[int, ...]],
    functions: EconFunctionsMapping,
    constraints: ConstraintFunctionsMapping,
    transitions: TransitionFunctionsMapping,
    stochastic_transition_names: frozenset[TransitionFunctionName],
    compute_regime_transition_probs: RegimeTransitionFunction,
    regime_to_v_interpolation_info: MappingProxyType[RegimeName, VInterpolationInfo],
    ages: AgeGrid,
    flat_param_names: frozenset[str],
    co_map_state_names: tuple[StateName, ...] = (),
    certainty_equivalent: CertaintyEquivalent | None = None,
) -> MappingProxyType[int, QAndFFunction]:
    """Build Q-and-F closures for each period of a non-terminal regime.

    Periods sharing the same target-regime configuration reuse a single
    closure, reducing the number of distinct JIT compilations. The caller
    is responsible for handling terminal regimes.

    When `period_to_regime_v_interp` is given (a model has `AgeSpecializedGrid`
    states), period `t`'s continuation `V_{t+1}` is interpolated on the target
    regimes' grids **at period `t+1`**, and the group signature folds in those
    grids' identities so periods with different continuation grids do not
    false-share a compiled `Q_and_F`. `None` reproduces the age-invariant build
    exactly.

    Args:
        regimes_to_active_periods: Immutable mapping of regime names to
            their active period tuples.
        functions: Immutable mapping of internal user functions.
        constraints: Immutable mapping of constraint functions.
        transitions: Immutable mapping of regime-to-regime transition
            functions.
        stochastic_transition_names: Frozenset of stochastic transition
            function names.
        compute_regime_transition_probs: Regime transition probability
            function for the current regime.
        regime_to_v_interpolation_info: Mapping of regime names to
            V-interpolation info.
        ages: Age grid for the model.
        flat_param_names: Frozenset of flat parameter names for the regime.
        certainty_equivalent: Nonlinear certainty equivalent declared by the
            regime, or `None`.

    Returns:
        Immutable mapping of period index to the per-period Q-and-F closure.

    """

    # Group periods by (target configuration, per-age policy signature). The
    # signature separates ages whose `AgeSpecializedFunction` functions/constraints
    # resolve to different closures, so they never false-share a compiled
    # `Q_and_F`; with no
    # specialized node every age yields the same (all-`INVARIANT`) signature and the
    # grouping collapses to the target configuration exactly as an age-invariant model.
    def continuation_info(
        period: int,
    ) -> MappingProxyType[RegimeName, VInterpolationInfo]:
        """Target-regime interpolation info for period `t`'s continuation V_{t+1}."""
        if period_to_regime_v_interp is None:
            return regime_to_v_interpolation_info
        # V_{t+1} lives on period t+1's grids; the last period's continuation is the
        # zero template (no next period), so any info is fine there.
        return period_to_regime_v_interp.get(period + 1, regime_to_v_interpolation_info)

    configs: dict[tuple[tuple[RegimeName, ...], Hashable], list[int]] = {}
    for period in range(ages.n_periods):
        complete = get_period_targets(
            period=period,
            transitions=transitions,
            regimes_to_active_periods=regimes_to_active_periods,
        )
        age = ages.period_to_age(period)
        # Fold the continuation grids' identities (at period+1) into the signature so
        # periods with different age-varying continuation grids get distinct kernels.
        continuation_sig = _continuation_grid_signature(
            continuation_info(period), complete
        )
        signature = (
            tree_signature(functions, age),
            tree_signature(constraints, age),
            continuation_sig,
        )
        configs.setdefault((complete, signature), []).append(period)

    # Build one Q_and_F per distinct group, resolving specialized functions and
    # constraints at the group's age. Equal signature ⇒ identical closures, so any
    # period in the group is a valid representative.
    built: dict[tuple[tuple[RegimeName, ...], Hashable], QAndFFunction] = {}
    for group_key, periods in configs.items():
        period_targets = group_key[0]
        age = ages.period_to_age(periods[0])
        built[group_key] = get_Q_and_F(
            flat_param_names=flat_param_names,
            functions=cast(
                "EconFunctionsMapping", resolve_specialized_nodes(functions, age)
            ),
            constraints=cast(
                "ConstraintFunctionsMapping",
                resolve_specialized_nodes(constraints, age),
            ),
            period_targets=period_targets,
            transitions=transitions,
            stochastic_transition_names=stochastic_transition_names,
            compute_regime_transition_probs=compute_regime_transition_probs,
            regime_to_v_interpolation_info=continuation_info(periods[0]),
            co_map_state_names=co_map_state_names,
            certainty_equivalent=certainty_equivalent,
        )

    # Map each period to its group's function
    result: dict[int, QAndFFunction] = {}
    for group_key, periods in configs.items():
        for period in periods:
            result[period] = built[group_key]

    return MappingProxyType(result)


def _build_argmax_and_max_Q_over_a_per_period(
    *,
    state_action_space: StateActionSpace,
    Q_and_F_functions: MappingProxyType[int, QAndFFunction],
    enable_jit: bool,
    has_taste_shocks: bool = False,
) -> MappingProxyType[int, ArgmaxQOverAFunction]:
    """Build argmax-and-max-Q-over-a closures for each period.

    Periods sharing the same Q_and_F object reuse a single compiled function.
    With taste shocks, the per-subject Gumbel key is vmapped alongside the
    simulated states.
    """
    spacemapped_names = tuple(state_action_space.states)
    if has_taste_shocks:
        spacemapped_names = (*spacemapped_names, "taste_shock_key")

    built: dict[int, ArgmaxQOverAFunction] = {}
    result: dict[int, ArgmaxQOverAFunction] = {}
    for period, Q_and_F in Q_and_F_functions.items():
        q_id = id(Q_and_F)
        if q_id not in built:
            func = get_argmax_and_max_Q_over_a(
                Q_and_F=Q_and_F,
                action_names=state_action_space.action_names,
                state_names=state_action_space.state_names,
                n_discrete_action_axes=len(state_action_space.discrete_actions),
                has_taste_shocks=has_taste_shocks,
            )
            if enable_jit:
                func = jax.jit(func)
            built[q_id] = simulation_spacemap(
                func=func,
                action_names=(),
                state_names=spacemapped_names,
            )
        result[period] = built[q_id]
    return MappingProxyType(result)


def _build_next_state_vmapped(
    *,
    functions: EconFunctionsMapping,
    transitions: TransitionFunctionsMapping,
    stochastic_transition_names: frozenset[TransitionFunctionName],
    all_grids: MappingProxyType[RegimeName, MappingProxyType[StateOrActionName, Grid]],
    variables: Variables,
    flat_param_names: frozenset[str],
    ages: AgeGrid,
    enable_jit: bool,
) -> MappingProxyType[int, NextStateSimulationFunction]:
    """Build a per-period vmapped next-state function for simulation.

    A law of motion can read a specialized function (e.g. `next_wealth` reading
    `net_income`), so next-state is resolved per age just like `Q_and_F`. Periods
    whose functions resolve to the same closures share one compiled function; with
    no `AgeSpecializedFunction` node every period shares a single function, exactly
    as an age-invariant model.
    """
    configs: dict[Hashable, list[int]] = {}
    for period in range(ages.n_periods):
        age = ages.period_to_age(period)
        configs.setdefault(tree_signature(functions, age), []).append(period)

    built: dict[Hashable, NextStateSimulationFunction] = {}
    for signature, periods in configs.items():
        age = ages.period_to_age(periods[0])
        next_state = get_next_state_function_for_simulation(
            functions=cast(
                "EconFunctionsMapping", resolve_specialized_nodes(functions, age)
            ),
            transitions=transitions,
            stochastic_transition_names=stochastic_transition_names,
            all_grids=all_grids,
            variables=variables,
        )
        sig_args = tuple(inspect.signature(next_state).parameters)
        non_vmap = {"period", "age"} | flat_param_names
        vmap_variables = tuple(arg for arg in sig_args if arg not in non_vmap)
        next_state_vmapped = vmap_1d(func=next_state, variables=vmap_variables)
        next_state_vmapped = with_signature(
            next_state_vmapped, kwargs=sig_args, enforce=False
        )
        built[signature] = (
            jax.jit(next_state_vmapped) if enable_jit else next_state_vmapped
        )

    result: dict[int, NextStateSimulationFunction] = {}
    for signature, periods in configs.items():
        for period in periods:
            result[period] = built[signature]
    return MappingProxyType(result)


def _fail_if_action_has_batch_size(
    user_regimes: Mapping[RegimeName, UserRegime],
) -> None:
    """Raise if any action grid has a non-zero batch_size.

    Batching applies only to the outer state loop during solving, not to the
    inner action optimization. A non-zero batch_size on an action grid would be
    silently ignored, so we reject it early.

    """
    for regime_name, user_regime in user_regimes.items():
        for action_name, grid in user_regime.actions.items():
            if grid is not None and grid.batch_size != 0:
                msg = (
                    f"batch_size > 0 is not supported on action grids. Only state "
                    f"grids can be batched. Found batch_size={grid.batch_size} on "
                    f"action '{action_name}' in regime '{regime_name}'."
                )
                raise ValueError(msg)
