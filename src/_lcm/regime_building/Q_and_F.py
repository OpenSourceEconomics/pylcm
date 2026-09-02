import dataclasses
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, cast

import jax
import jax.numpy as jnp
from dags import (
    concatenate_functions,
    get_annotations,
    with_signature,
)

from _lcm.certainty_equivalent import CertaintyEquivalent, LinearExpectation
from _lcm.probability import (
    is_negative,
    is_represented_zero,
    normalized_scaled_weights,
)
from _lcm.processes import _ContinuousStochasticProcess
from _lcm.regime_building.next_state import (
    get_next_state_function_for_solution,
    get_next_stochastic_weights_function,
)
from _lcm.regime_building.V import VInterpolationInfo, get_V_interpolator
from _lcm.regime_building.w_dag import _get_build_W_kwargs

# `zero_safe_average` only, and only because it is the reduction that takes an
# `axis`: a collective regime's node reduction has to leave the trailing
# stakeholder axis standing, so it cannot reduce the whole array at once. Every
# weighted TERM in this file comes from `_lcm.zero_safe` below -- including the
# one inside `zero_safe_average` itself -- so one implementation carries the
# scale and subnormal rules for the whole engine.
from _lcm.regime_building.zero_safe import zero_safe_average
from _lcm.transition_plans import (
    LotteryLifetime,
    TargetTransitionPlans,
)
from _lcm.typing import (
    ConstraintFunction,
    ConstraintFunctionsMapping,
    EconFunction,
    EconFunctionsMapping,
    NextStateSimulationFunction,
    QAndFFunction,
    RegimeName,
    RegimeTransitionFunction,
    StateName,
    TransitionFunction,
    TransitionFunctionName,
    TransitionFunctionsMapping,
    _ParamsLeaf,
)
from _lcm.utils.dispatchers import productmap
from _lcm.utils.functools import get_union_of_args
from _lcm.zero_safe import (
    scaled_joint_weight,
    zero_safe_weighted_term,
)
from lcm.exceptions import ModelInitializationError
from lcm.typing import (
    BoolND,
    ContinuousState,
    DiscreteState,
    Float1D,
    FloatND,
    IntND,
)


def _compare_swap_values(
    *, left: FloatND, right: FloatND, ascending: bool
) -> tuple[FloatND, FloatND]:
    """Order two arrays while preserving both original operands.

    ``minimum``/``maximum`` supply the canonical numeric order, including
    ``-0 < +0`` and infinities, but each returns NaN when exactly one operand is
    NaN. Used as a compare-swap pair they would therefore duplicate the NaN and
    discard the finite operand. The two explicit NaN branches restore that
    operand before returning the pair. When both operands are NaN they are
    swapped, so even distinct payloads remain a permutation; NaNs form the
    terminal invalid-input equivalence class and still poison the final sum.
    """
    left_arr, right_arr = jnp.broadcast_arrays(jnp.asarray(left), jnp.asarray(right))
    left_is_nan = jnp.isnan(left_arr)
    right_is_nan = jnp.isnan(right_arr)

    lower = jnp.where(
        left_is_nan,
        right_arr,
        jnp.where(right_is_nan, left_arr, jnp.minimum(left_arr, right_arr)),
    )
    upper = jnp.where(
        left_is_nan,
        left_arr,
        jnp.where(right_is_nan, right_arr, jnp.maximum(left_arr, right_arr)),
    )
    return (lower, upper) if ascending else (upper, lower)


def _bitonic_value_order_network(
    n_items: int,
) -> tuple[tuple[int, int, bool], ...]:
    """Return a static O(K log² K) compare-swap network for arbitrary ``K``."""
    comparisons: list[tuple[int, int, bool]] = []

    def greatest_power_of_two_less_than(n: int) -> int:
        out = 1
        while out < n:
            out *= 2
        return out // 2

    def merge(*, lo: int, n: int, ascending: bool) -> None:
        if n <= 1:
            return
        split = greatest_power_of_two_less_than(n)
        comparisons.extend((i, i + split, ascending) for i in range(lo, lo + n - split))
        merge(lo=lo, n=split, ascending=ascending)
        merge(lo=lo + split, n=n - split, ascending=ascending)

    def sort(*, lo: int, n: int, ascending: bool) -> None:
        if n <= 1:
            return
        split = n // 2
        sort(lo=lo, n=split, ascending=not ascending)
        sort(lo=lo + split, n=n - split, ascending=ascending)
        merge(lo=lo, n=n, ascending=ascending)

    sort(lo=0, n=n_items, ascending=True)
    return tuple(comparisons)


def _sum_regime_mixture(
    *, mixture_terms: list[tuple[RegimeName, FloatND, FloatND]], like: FloatND
) -> FloatND:
    """Reduce ``E[V'] = Σ p_r V_r`` without a materialized target axis.

    Each target's zero-safe contribution is formed on its native cell shape.
    For two or more targets, a static bitonic compare-swap network orders the
    separate arrays by contribution value before a deterministic left fold. XLA
    therefore sees only cell-shaped broadcasts, selects, and additions; no
    ``(K, *cell)`` or ``(*cell, K)`` tensor exists for the target cardinality.

    The ordering is the same supported invariant as the former stack-and-sort
    spelling: it depends only on the contribution multiset, never on regime
    labels or declaration order. It is explicit rather than expressed with
    bare ``minimum``/``maximum`` because those operations turn one
    ``(NaN, finite)`` pair into two NaNs and cease to be a permutation. The
    corrected comparator preserves both operands, puts numeric values before
    NaNs, canonicalizes signed zero, and leaves equal finite values and infinities
    unchanged. A live NaN consequently remains visible in the final sum, while a
    zero probability still annihilates an admissible
    non-finite continuation inside ``zero_safe_weighted_term``.

    The one-comparison two-target network is semantically necessary, not merely
    a uniform spelling. Treating the final addition as commutative before each
    product has crossed a value comparison lets XLA contract one multiply into
    the add; swapping target declarations can then change the bits and a strict
    argmax. The compare-swap establishes the same stored contribution values and
    canonical value order as larger mixtures without a target-shaped buffer.
    One target needs no ordering. The network uses O(K log² K) comparisons rather
    than the prototype's quadratic bubble pass, keeping trace growth bounded.
    An empty terminal mixture is exactly ``zeros_like(like)``.
    """
    contributions: list[FloatND] = []
    for _, probability, value in mixture_terms:
        probability_arr = jnp.asarray(probability)
        value_arr = jnp.asarray(value)
        # Regime probabilities carry the cell axes. A collective continuation
        # adds a trailing stakeholder axis, over which one scalar probability is
        # constant; right-padding aligns that axis without stacking targets.
        if probability_arr.ndim < value_arr.ndim:
            probability_arr = probability_arr.reshape(
                probability_arr.shape + (1,) * (value_arr.ndim - probability_arr.ndim)
            )
        contributions.append(
            zero_safe_weighted_term(
                weight=probability_arr,
                value=value_arr,
                subnormal_is_accounted_for=False,
            )
        )

    if not contributions:
        return jnp.zeros_like(like)

    if len(contributions) > 1:
        for left, right, ascending in _bitonic_value_order_network(len(contributions)):
            contributions[left], contributions[right] = _compare_swap_values(
                left=contributions[left],
                right=contributions[right],
                ascending=ascending,
            )

    total = contributions[0]
    for contribution in contributions[1:]:
        total = total + contribution
    return total


def _normalized_regime_mixture(
    *,
    mixture: FloatND,
    probability_mass: FloatND,
    has_negative_probability: BoolND,
) -> FloatND:
    """Divide a summed regime mixture by the mass that was summed into it.

    A route that accumulates `Σ p·E[V]` has to divide by the represented mass to
    state the same quantity as `LinearExpectation.aggregate`. Regime-transition
    validation accepts any mass within `jnp.allclose` of one, and at the top of
    that tolerance the undivided sum reverses the Bellman argmax. Dividing is
    exact whenever the mass is exactly one, so a well-formed lottery keeps its
    floating-point association.

    Weights that are not a distribution — a mass away from one, or a negative
    probability — publish NaN rather than a finite value, at every log level:
    the arithmetic states it, so `log_level="off"` cannot skip it.

    Args:
        mixture: The summed mixture `Σ p·E[V]` over the retained targets.
        probability_mass: Those targets' probabilities, summed.
        has_negative_probability: Whether any of them carried the sign bit on a
            nonzero magnitude.

    Returns:
        The mixture divided by its mass, or NaN where the weights are not a
        probability distribution.

    """
    return mixture / _unit_regime_mass_or_nan(
        probability_mass=probability_mass,
        has_negative_probability=has_negative_probability,
    )


def get_Q_and_F(
    *,
    flat_param_names: frozenset[str],
    functions: EconFunctionsMapping,
    constraints: ConstraintFunctionsMapping,
    period_targets: tuple[RegimeName, ...],
    scalar_targets: tuple[RegimeName, ...] = (),
    transitions: TransitionFunctionsMapping,
    transition_plans: TargetTransitionPlans,
    compute_regime_transition_probs: RegimeTransitionFunction,
    regime_to_v_interpolation_info: MappingProxyType[RegimeName, VInterpolationInfo],
    koopmans_aggregator: EconFunction,
    certainty_equivalent: CertaintyEquivalent | None,
    co_map_state_names: tuple[StateName, ...] = (),
    continuation_functions: EconFunctionsMapping | None = None,
    gated_continuations: Mapping[RegimeName, GatedContinuationSpec] = MappingProxyType(
        {}
    ),
) -> QAndFFunction:
    """Get the state-action (Q) and feasibility (F) function for a non-terminal period.

    `age` and `period` are runtime arguments (via `**states_actions_params`),
    not closure constants. This allows periods with the same target
    configuration to share a single JIT-compiled function.

    Q mixes two phases when it is built for the simulate phase: the *current* flow
    (utility, feasibility, `H`) is simulate-phase, while the *continuation* is priced
    under the agent's perceived law — the solve phase. The flow is *now*, so it is
    realized under the true law; the belief is about the *future*, so it prices only
    the continuation.

    The continuation sub-DAG must be **phase-closed**: a transition law is a DAG node
    like any other, and `dags` resolves its argument names against a function pool
    transitively, so a law that depends on a `Phased` helper picks up whichever variant
    that pool holds. The continuation therefore takes a matched pair, `transitions` +
    `continuation_functions`; resolving a solve-phase law's helpers from the simulate
    pool yields a sub-DAG that is neither phase and can reverse the argmax.

    The flow needs no such pairing: `next_<state>` is reserved vocabulary a transition
    produces, so no utility or constraint reads one and the flow contains no transition
    node to resolve.

    Args:
        flat_param_names: Frozenset of flat parameter names for the regime.
        functions: Immutable mapping of function names to internal user functions.
            Supplies the current-period flow (utility, feasibility, `H`).
        constraints: Immutable mapping of constraint names to internal user functions.
        period_targets: Carry targets — reachable, active next period, and
            carrying at least one state, so their continuation is read at the
            next states their laws produce.
        scalar_targets: Graph targets active next period that carry no state.
            Their value function is rank-zero, so it enters `E[V]` as a single
            degenerate lottery node weighted only by the regime transition
            probability.
        transitions: Immutable mapping of transition names to transition functions.
        transition_plans: Immutable mapping of target regime names to their
            transition laws.
        compute_regime_transition_probs: Regime transition probability function
            for solve.
        regime_to_v_interpolation_info: Mapping of regime names to V-interpolation
            info.
        koopmans_aggregator: The regime's Bellman aggregator, combining
            utility and the certainty equivalent into `Q`.
        certainty_equivalent: Nonlinear certainty equivalent declared by the
            regime, or `None` for the linear expectation.
        co_map_state_names: Tuple of state names co-mapped with the continuation V —
            their axes are sliced off each `next_V_arr` leaf by the backward-induction
            co-map, so their coordinates are dropped from the interpolation. Only fixed
            (never-transitioning) distributed states qualify.
        certainty_equivalent: Nonlinear certainty equivalent declared by the
            regime, or `None`.
        continuation_functions: Function pool the continuation sub-DAG (the state
            transitions and the stochastic weights) is resolved against. Defaults to
            `functions`, which is correct in the solve phase, where both pools are the
            solve pool. The simulate phase must pass the SOLVE pool here so the agent
            compares actions under its perceived law while the world is realized under
            the true one.
        gated_continuations: Mapping of target regime names to the gated-edge
            continuation spec that target's leaf is read under. A target absent
            from it is read as an ordinary value function.

    Returns:
        A function that computes the state-action values (Q) and the feasibilities (F)
        for a non-terminal period.

    """
    # In the solve phase the two roles coincide; only simulate passes them apart.
    continuation_pool = (
        functions if continuation_functions is None else continuation_functions
    )
    # The flow reads no transition node at all: `next_<state>` is reserved vocabulary
    # a transition produces, never something this period's utility or a constraint may
    # read. So only the continuation needs a pool of its own, and the phase split
    # reduces to that one sub-DAG.
    U_and_F = _get_U_and_F(functions=functions, constraints=constraints)
    compute_CE, continuation_deps, continuation_arg_names = _get_compute_CE(
        # `continuation_pool`, NOT `functions`: the continuation is priced under
        # the perceived (solve-phase) law, helpers included. In the solve phase the
        # two are the same object; only simulate passes them apart.
        functions=continuation_pool,
        period_targets=period_targets,
        scalar_targets=scalar_targets,
        transitions=transitions,
        transition_plans=transition_plans,
        compute_regime_transition_probs=compute_regime_transition_probs,
        regime_to_v_interpolation_info=regime_to_v_interpolation_info,
        certainty_equivalent=certainty_equivalent,
        co_map_state_names=co_map_state_names,
        gated_continuations=gated_continuations,
    )
    _build_W_kwargs = _get_build_W_kwargs(
        functions=functions, koopmans_aggregator=koopmans_aggregator
    )

    arg_names_of_Q_and_F = _get_arg_names_of_Q_and_F(
        deps=[U_and_F, *continuation_deps],
        include=frozenset(
            {"next_regime_to_V_arr", "period", "age"}
            | flat_param_names
            | continuation_arg_names
        ),
        exclude=frozenset(),
    )

    @with_signature(
        args=arg_names_of_Q_and_F, return_annotation="tuple[FloatND, BoolND]"
    )
    def Q_and_F(
        next_regime_to_V_arr: MappingProxyType[RegimeName, FloatND],
        **states_actions_params: _ParamsLeaf,
    ) -> tuple[FloatND, BoolND]:
        """Calculate the state-action value and feasibility for a non-terminal period.

        Args:
            next_regime_to_V_arr: The next period's value function array.
            **states_actions_params: States, actions, age, period, and flat
                regime params.

        Returns:
            A tuple containing the arrays with state-action values and feasibilities.

        """
        # F_arr is built here, before and independently of Q (it never reads
        # E_next_V). A value-aware mask cannot be built at this point: it needs
        # the per-stakeholder Q^s, which is why `get_Q_and_F_collective` keeps
        # the state-independent F here and ANDs its value constraints in only
        # after computing Q^s.
        U_arr, F_arr = U_and_F(**states_actions_params)
        CE, _ = compute_CE(
            next_regime_to_V_arr=next_regime_to_V_arr,
            zero=jnp.zeros_like(U_arr),
            states_actions_params=states_actions_params,
        )

        Q_arr = koopmans_aggregator(
            utility=U_arr,
            CE=CE,
            **_build_W_kwargs(states_actions_params),
        )

        # Handle cases when there is only one state.
        # In that case, Q_arr and F_arr are scalars, but we require arrays as output.
        return jnp.asarray(Q_arr), jnp.asarray(F_arr)

    return Q_and_F


def get_compute_intermediates(
    *,
    flat_param_names: frozenset[str],
    functions: EconFunctionsMapping,
    constraints: ConstraintFunctionsMapping,
    period_targets: tuple[RegimeName, ...],
    scalar_targets: tuple[RegimeName, ...] = (),
    transitions: TransitionFunctionsMapping,
    transition_plans: TargetTransitionPlans,
    compute_regime_transition_probs: RegimeTransitionFunction,
    regime_to_v_interpolation_info: MappingProxyType[RegimeName, VInterpolationInfo],
    koopmans_aggregator: EconFunction,
    certainty_equivalent: CertaintyEquivalent | None,
    co_map_state_names: tuple[StateName, ...],
    gated_continuations: Mapping[RegimeName, GatedContinuationSpec] = MappingProxyType(
        {}
    ),
) -> Callable:
    """Build a closure that computes Q_and_F intermediates for diagnostics.

    Mirrors `get_Q_and_F` but returns all intermediates instead of just
    `(Q, F)`. The caller productmaps and JIT-compiles the closure; it runs
    only in the error path when `validate_V` detects NaN. `age` and `period`
    are runtime arguments (passed via `states_actions_params`) so that
    periods sharing the same target configuration share a single
    JIT-compiled function.

    Args:
        flat_param_names: Frozenset of flat parameter names for the regime.
        functions: Immutable mapping of function names to internal user functions.
        constraints: Immutable mapping of constraint names to constraint functions.
        period_targets: Carry targets — reachable, active next period, and
            carrying at least one state.
        scalar_targets: Graph stateless targets active next period, whose
            rank-zero value enters `E[V]` weighted only by the regime transition
            probability. Must match what `get_Q_and_F` was built with, or the
            diagnostics disagree with the solve they explain.
        transitions: Immutable mapping of target regime names to state transition
            functions.
        transition_plans: Immutable mapping of target regime names to their
            transition laws.
        compute_regime_transition_probs: Callable returning regime transition
            probabilities for the current regime.
        regime_to_v_interpolation_info: Immutable mapping of regime names to
            V-interpolation info.
        koopmans_aggregator: The regime's Bellman aggregator, combining
            utility and the certainty equivalent into `Q`.
        certainty_equivalent: Nonlinear certainty equivalent declared by the
            regime, or `None` for the linear expectation.
        co_map_state_names: Tuple of state names co-mapped with the
            continuation V in the solve kernel. The diagnostics pass an empty
            tuple: they are handed the full, un-sliced value arrays and map
            over every state, so no axis has been sliced off to compensate for.
        gated_continuations: Mapping of target regime names to the gated-edge
            continuation spec that target's leaf is read under. A target absent
            from it is read as an ordinary value function.

    Returns:
        Closure returning `(U_arr, F_arr, CE, Q_arr, active_regime_probs)`.

    """
    U_and_F = _get_U_and_F(functions=functions, constraints=constraints)
    compute_CE, continuation_deps, continuation_arg_names = _get_compute_CE(
        functions=functions,
        period_targets=period_targets,
        scalar_targets=scalar_targets,
        transitions=transitions,
        transition_plans=transition_plans,
        compute_regime_transition_probs=compute_regime_transition_probs,
        regime_to_v_interpolation_info=regime_to_v_interpolation_info,
        certainty_equivalent=certainty_equivalent,
        co_map_state_names=co_map_state_names,
        gated_continuations=gated_continuations,
    )
    _build_W_kwargs = _get_build_W_kwargs(
        functions=functions, koopmans_aggregator=koopmans_aggregator
    )

    arg_names_of_compute_intermediates = _get_arg_names_of_Q_and_F(
        deps=[U_and_F, *continuation_deps],
        include=frozenset(
            {"next_regime_to_V_arr", "period", "age"}
            | flat_param_names
            | continuation_arg_names
        ),
        exclude=frozenset(),
    )

    @with_signature(
        args=arg_names_of_compute_intermediates,
        return_annotation=(
            "tuple[FloatND, FloatND, FloatND, FloatND, "
            "MappingProxyType[RegimeName, FloatND]]"
        ),
    )
    def compute_intermediates(
        next_regime_to_V_arr: MappingProxyType[RegimeName, FloatND],
        **states_actions_params: _ParamsLeaf,
    ) -> tuple[
        FloatND, FloatND, FloatND, FloatND, MappingProxyType[RegimeName, FloatND]
    ]:
        """Compute all Q_and_F intermediates."""
        U_arr, F_arr = U_and_F(**states_actions_params)
        CE, active_regime_probs = compute_CE(
            next_regime_to_V_arr=next_regime_to_V_arr,
            zero=jnp.zeros_like(U_arr),
            states_actions_params=states_actions_params,
        )

        Q_arr = koopmans_aggregator(
            utility=U_arr,
            CE=CE,
            **_build_W_kwargs(states_actions_params),
        )

        return U_arr, F_arr, CE, Q_arr, active_regime_probs

    return compute_intermediates


def get_Q_and_F_terminal(
    *,
    flat_param_names: frozenset[str],
    functions: EconFunctionsMapping,
    constraints: ConstraintFunctionsMapping,
) -> QAndFFunction:
    """Get the state-action (Q) and feasibility (F) function for a terminal period.

    `age` and `period` are runtime arguments (via `**states_actions_params`).

    Args:
        flat_param_names: Frozenset of flat parameter names for the regime.
        functions: Immutable mapping of function names to internal user functions.
        constraints: Immutable mapping of constraint names to internal user functions.

    Returns:
        A function that computes the state-action values (Q) and the feasibilities (F)
        for a terminal period.

    """
    U_and_F = _get_U_and_F(functions=functions, constraints=constraints)

    arg_names_of_Q_and_F = _get_arg_names_of_Q_and_F(
        deps=[U_and_F],
        # While the terminal period does not depend on the value function array, we
        # include it in the signature, such that we can treat all periods uniformly
        # during the solution and simulation.
        include=frozenset({"next_regime_to_V_arr", "period", "age"} | flat_param_names),
        exclude=frozenset(),
    )

    @with_signature(
        args=arg_names_of_Q_and_F, return_annotation="tuple[FloatND, BoolND]"
    )
    def Q_and_F(
        next_regime_to_V_arr: MappingProxyType[RegimeName, FloatND],  # noqa: ARG001
        **states_actions_params: _ParamsLeaf,
    ) -> tuple[FloatND, BoolND]:
        """Calculate the state-action values and feasibilities for a terminal period.

        Args:
            next_regime_to_V_arr: Unused in the terminal period; accepted so that
                solve and simulate treat all periods uniformly.
            **states_actions_params: States, actions, age, period, and flat
                regime params.

        Returns:
            A tuple of the state-action value array (Q) and the feasibility
            mask (F).

        """
        U_arr, F_arr = U_and_F(**states_actions_params)
        return jnp.asarray(U_arr), jnp.asarray(F_arr)

    return Q_and_F


def get_Q_and_F_terminal_collective(
    *,
    flat_param_names: frozenset[str],
    functions: EconFunctionsMapping,
    constraints: ConstraintFunctionsMapping,
    stakeholders: tuple[str, ...],
    value_constraints: ConstraintFunctionsMapping = MappingProxyType({}),
    same_period_refs: MappingProxyType[
        str, ResolvedProjectedRegimeValue
    ] = MappingProxyType({}),
    same_period_v_interpolation_info: MappingProxyType[
        RegimeName, VInterpolationInfo
    ] = MappingProxyType({}),
) -> QAndFFunction:
    """Terminal (Q, F) for a collective regime — stacked per-stakeholder U + shared F.

    Separate from `get_Q_and_F_terminal` so the singleton terminal path (shared
    with the simulate / compute-intermediates machinery) carries none of the
    stakeholder handling; this builder is used only at the collective solve site.

    Builds ONE closure over every stakeholder's `utility_<s>` target plus the
    single `feasibility` target. Feasibility is regime-level: `_get_feasibility`
    takes no stakeholder input, so a household has one action set however many
    felicities it carries, and the mask is a DAG target the felicities share
    rather than one evaluated per stakeholder. The returned `Q_and_F` stacks the
    per-stakeholder utilities on a trailing stakeholder axis: for a scalar
    (state, action) cell it returns `U` of shape `(n_stakeholders,)` and a scalar
    `F`. After the action product-map in `get_max_Q_over_a`, `U` has shape
    `(*action_axes, n_stakeholders)` and `F` `(*action_axes,)`; the stakeholder
    branch there splits `U` by stakeholder and calls `collective_readout`.

    Args:
        flat_param_names: Frozenset of flat parameter names for the regime.
        functions: Immutable mapping of function names to internal user functions;
            carries `utility_<s>` for each stakeholder in place of `utility`.
        constraints: Immutable mapping of constraint names to internal user functions.
        stakeholders: Ordered stakeholder names; fixes the trailing-axis order.
        value_constraints: Immutable mapping of value-constraint names to
            predicates (params already renamed to qnames). A terminal cell's
            action value IS its utility — there is no continuation — so a
            predicate's `Q_<s>` is stakeholder `s`'s terminal payoff. An
            all-infeasible cell publishes the dissolution flag `D` and the
            `-inf` sentinel; the engine resolves no outside option in its place,
            because a terminal regime has no continuation to route into.
        same_period_refs: Immutable mapping of reference-value names to resolved
            same-period reference declarations. When non-empty, the returned
            `Q_and_F` carries `SAME_PERIOD_V_ARG` and `SAME_PERIOD_PARAMS_ARG`,
            supplied per period by the solve loop.
        same_period_v_interpolation_info: Mapping of regime names to
            V-interpolation info at THIS period, which is what a same-period
            reference reader interpolates on.

    Returns:
        A function computing the stacked per-stakeholder utilities (Q) and the
        shared feasibility mask (F) for a terminal collective period.

    """
    utilities_and_F = _get_U_and_F(
        functions=functions,
        constraints=constraints,
        utility_names=tuple(f"utility_{stakeholder}" for stakeholder in stakeholders),
    )

    value_constraint_machinery = _build_value_constraint_machinery(
        value_constraints=value_constraints,
        same_period_refs=same_period_refs,
        stakeholders=stakeholders,
        same_period_v_interpolation_info=same_period_v_interpolation_info,
        functions=functions,
    )

    arg_names_of_Q_and_F = _get_arg_names_of_Q_and_F(
        deps=[
            utilities_and_F,
            *list(value_constraint_machinery.evaluators.values()),
            *list(value_constraint_machinery.reference_readers.values()),
        ],
        include=frozenset({"next_regime_to_V_arr", "period", "age"} | flat_param_names),
        exclude=value_constraint_machinery.engine_supplied_names,
    )

    @with_signature(
        args=arg_names_of_Q_and_F, return_annotation="tuple[FloatND, BoolND]"
    )
    def Q_and_F(
        next_regime_to_V_arr: MappingProxyType[RegimeName, FloatND],  # noqa: ARG001
        **states_actions_params: _ParamsLeaf,
    ) -> tuple[FloatND, BoolND]:
        """Stacked per-stakeholder utilities and the shared feasibility mask.

        Args:
            next_regime_to_V_arr: Unused in a terminal period; accepted so solve
                treats all periods uniformly.
            **states_actions_params: States, actions, age, period, and flat
                regime params.

        Returns:
            A tuple of the stacked per-stakeholder utility array (trailing
            stakeholder axis) and the shared feasibility mask.

        """
        *stakeholder_utilities, feasibility = utilities_and_F(**states_actions_params)
        U_stack = jnp.stack(
            [jnp.asarray(U_s) for U_s in stakeholder_utilities], axis=-1
        )
        F_arr: BoolND = jnp.asarray(feasibility)

        # Value-aware feasibility on the terminal payoff. `Q^s` here IS `U^s`:
        # a terminal cell has no continuation, so the value each stakeholder
        # weighs against the reference is the payoff the cell itself delivers.
        if value_constraint_machinery.evaluators:
            F_arr = _apply_value_constraints(
                machinery=value_constraint_machinery,
                Q_arr=U_stack,
                F_arr=F_arr,
                states_actions_params=states_actions_params,
            )

        return U_stack, F_arr

    return Q_and_F


# The name under which the mapping of same-period
# reference regimes to their current-period V arrays enters the kernel
# signature. Only regimes declaring `same_period_refs` carry it.
# Kernel argument holding each edge-reference regime's value array. A gated
# edge's gate references and leg fallbacks are read where the SOURCE lands, but
# the value they name belongs to the period the source lands IN. Backward
# induction has already solved that period, so the arrays arrive here rolled
# forward — and from the RAW rolled mapping, never the edge-substituted one,
# because a reference regime may also be an edge target whose entry the
# substitution replaces with that edge's `Wbar`.
EDGE_REF_V_ARG = "edge_reference_regime_to_V_arr"

# Each edge-reference regime's OWN flat params, keyed by regime. Same split as
# `SAME_PERIOD_PARAMS_ARG`: the projection's free arguments belong to the
# reading regime, the interpolation helpers to the reference regime's own grid.
EDGE_REF_PARAMS_ARG = "edge_reference_regime_to_params"

SAME_PERIOD_V_ARG = "same_period_regime_to_V_arr"

# The name under which the mapping of
# same-period reference regimes to THEIR OWN flat params enters the kernel
# signature, alongside `SAME_PERIOD_V_ARG`. Carried by every reader built by
# `_build_same_period_ref_reader`, and hence by every regime that reads another
# regime's same-period V.
#
# A reference reader interpolates the REFERENCE regime's V over the REFERENCE
# regime's grid, so the interpolator's runtime grid helpers (an
# `IrregSpacedGrid(pass_points_at_runtime=True)` reference state's `points`, via
# `V._get_coordinate_finder`) are parameters of the REFERENCE regime: they live
# in `flat_params[ref.regime]`, never in the READING regime's own namespace.
# Coordinate VARIABLES stay prefixed (internal wiring that must not collide with
# the reading regime's own state names), but that prefixed spelling
# (`__same_period_ref__x__points`) is a name no caller supplies and no params
# template emits — `_lcm.params.regime_template._add_runtime_grid_params` emits
# `x__points`, in the reference regime's own template. PARAMETER qnames are
# therefore separated from the coordinate variables and resolved against the
# reference regime's explicit namespace through this mapping.
SAME_PERIOD_PARAMS_ARG = "same_period_regime_to_params"


# Internal argument names of the same-period reference interpolation; never
# surfaced in the kernel signature.
_REF_STATE_PREFIX = "__same_period_ref__"
_REF_V_ARR_NAME = "__same_period_ref_V_arr__"

# Engine context a gated edge's gate may name, bound to the TARGET's period.
_EDGE_CONTEXT_ARGS = frozenset({"period", "age"})

# The name under which one gated edge's stacked operand surfaces reach its
# gate. Declared here rather than beside the edge machinery because both sides
# of the split need it and the continuation reader is the lower layer.
EDGE_CHANNELS_ARG = "__edge_channels__"


@dataclass(frozen=True, kw_only=True)
class ProjectedLandingReader:
    """One projected reference or leg fallback, read AT the source's landing.

    A projection maps the target's states onto another regime's grid, so the
    value it names is `V_ref(projection(landing))`. Evaluating it here — rather
    than tabulating it on the target's grid and interpolating that surface —
    is what makes the number the source collects the number the branch pays.
    The two orders coincide only when the projection is affine.
    """

    name: str
    """Keyword the branch combiner receives this value under."""

    reader: Callable[..., FloatND]
    """Reads the referenced regime's V at the projected coordinates."""

    state_args: tuple[StateName, ...]
    """Target states the projection reads, supplied at the landing point."""

    other_args: tuple[str, ...]
    """The reader's remaining arguments — params and the same-period mappings."""


@dataclass(frozen=True, kw_only=True)
class GatedContinuationSpec:
    """What a source needs in order to read one gated target's continuation.

    Carried as plain values rather than as the resolved edge, so the
    continuation reader stays below the gated-edge machinery that builds it.
    """

    n_channels: int
    """Length of the leaf's trailing channel axis."""

    combine: Callable[..., FloatND]
    """Apply the edge's gate to the channels interpolated at the landing point."""

    gate_state_names: tuple[StateName, ...]
    """Target states the gate reads, supplied at the landing point."""

    projected_readers: tuple[ProjectedLandingReader, ...] = ()
    """Gate references and leg fallbacks, read at the landing point."""

    target_ages: Float1D
    """Age at each model period, indexed by the period the gate is read in.

    A gate is a statement about the period the source lands in, not the one it
    decides in, so the source's kernel hands it `period + 1` and the age this
    table holds there — the same context the surfaces underneath were folded
    with.
    """


@dataclass(frozen=True, kw_only=True)
class GatedContinuationSchedule:
    """One gated edge's continuation specs, keyed by the period they fold at."""

    by_period: MappingProxyType[int, GatedContinuationSpec]
    """Specs keyed by FOLD period. A source standing at `t` reads the `t + 1`
    entry: that is the period whose value arrays the edge folds, and whose
    grids its projected readers close over. Keyed by period for the reason
    `ResolvedGatedEdge.folds_by_period` is — a gate reference on an
    `AgeSpecializedGrid` is read on its own regime's nodes, which move with
    age while their shape does not."""

    reference_regimes: tuple[RegimeName, ...]
    """Regimes this edge's gate references and leg fallbacks interpolate.
    Enters the Q_and_F grouping key, so two periods whose reference grids
    differ never share one compiled kernel."""


@dataclass(frozen=True, kw_only=True)
class ResolvedProjectedRegimeValue:
    """Engine-side form of a user `ProjectedRegimeValue`, resolved at model processing.

    The user declaration names a stakeholder; the
    engine resolves it to the index on the reference regime's trailing
    stakeholder axis (`None` for a singleton reference, whose V has no such
    axis).
    """

    regime: RegimeName
    """Name of the reference regime whose same-period V is read."""

    projection: Mapping[StateName, Callable[..., Any]]
    """Per-reference-state projection functions (user vocabulary, DAG-resolved)."""

    stakeholder_index: int | None
    """Index into the reference V's trailing stakeholder axis, or `None`."""

    stakeholder: str | None = None
    """The reference regime's stakeholder as named, or `None` for a singleton.

    The index above locates the value on one regime's axis; the name is what
    the role means model-wide, and forward simulation needs the name to say
    which role a subject carries after following this reference.
    """


def projection_func_or_fail(
    *, ref: ResolvedProjectedRegimeValue, state_name: StateName
) -> Callable[..., Any]:
    """Return the coordinate function a reference's projection gives one state.

    Model build rejects an incomplete projection before any kernel exists, so
    only a caller reaching an engine entry point directly can still present
    one. Naming the reference regime, the state, and what the projection does
    supply says which declaration is short; the lookup alone would name the
    state and nothing around it.

    Args:
        ref: Resolved same-period reference whose projection is read.
        state_name: State of the reference regime a coordinate is owed on.

    Returns:
        The projection's coordinate function for that state.

    Raises:
        ModelInitializationError: If the projection has no entry for the state.

    """
    if state_name not in ref.projection:
        msg = (
            f"The projection onto reference regime '{ref.regime}' supplies no "
            f"coordinate function for state '{state_name}'. It supplies "
            f"{sorted(ref.projection)}."
        )
        raise ModelInitializationError(msg)
    return ref.projection[state_name]


def _build_same_period_ref_reader(
    *,
    ref: ResolvedProjectedRegimeValue,
    v_interpolation_info: VInterpolationInfo,
    functions: EconFunctionsMapping,
    deterministic_transitions: Mapping[TransitionFunctionName, TransitionFunction] = (
        MappingProxyType({})
    ),
    v_mapping_arg: str = SAME_PERIOD_V_ARG,
    params_mapping_arg: str = SAME_PERIOD_PARAMS_ARG,
) -> Callable[..., FloatND]:
    """Build the reader of one same-period reference value at a (state, action) cell.

    Each projection entry is concatenated with the
    regime's function DAG (so it may read states, actions, helper functions,
    and the merged deterministic `next_<state>` laws), producing one coordinate
    per reference state; the reference regime's CURRENT-period V array — passed
    per solve step under `SAME_PERIOD_V_ARG` — is then interpolated at those
    coordinates with the ordinary V-interpolation machinery
    (`get_V_interpolator`), sliced to the named stakeholder first when the
    reference is collective. The returned callable's signature carries only
    user-level names (states / actions / params reached by the projections,
    plus `SAME_PERIOD_V_ARG` and `SAME_PERIOD_PARAMS_ARG`), so the kernel
    signature stays clean.

    The projections are expressed in the READING regime's vocabulary and their
    free parameters are bound from the reading regime's own params (every caller
    passes exactly that); the INTERPOLATION helpers instead belong to the
    REFERENCE regime's grid and are resolved against `SAME_PERIOD_PARAMS_ARG`
    (see that constant). The two provenances are separated here rather
    than merged into one namespace, because a runtime irregular grid names its
    helper after the STATE alone (`x__points`), so a reading regime that happens
    to declare an identically named state would otherwise silently supply its
    OWN grid points for the reference regime's interpolation.

    A projection produces a genuine VALUE for every reference state
    (interpolation-worthy, possibly off-grid) — unlike the ordinary
    continuation-value path, which always feeds a process axis its exact
    on-grid Markov-chain index. When the reference regime carries a
    non-folded process state (`_ContinuousStochasticProcess`, classified
    `discrete_states` for the Markov-chain solve path but read here as a
    genuine value), `get_V_interpolator`'s process-aware mode
    (`interpolate_process_axes=True`) is used so that axis is linearly
    interpolated instead of integer-looked-up; a reference regime without a
    process state takes the ordinary path (`interpolate_process_axes=False`).

    Args:
        ref: Resolved same-period reference declaration.
        v_interpolation_info: V-interpolation info of the reference regime.
        functions: Immutable mapping of function names to internal user
            functions.
    """
    _reference_has_process_axis = any(
        isinstance(grid, _ContinuousStochasticProcess)
        for grid in v_interpolation_info.discrete_states.values()
    )
    interpolator = get_V_interpolator(
        v_interpolation_info=v_interpolation_info,
        state_prefix=_REF_STATE_PREFIX,
        V_arr_name=_REF_V_ARR_NAME,
        interpolate_process_axes=_reference_has_process_axis,
    )
    # Empty for a value constraint: that projection is evaluated at THIS
    # period's states, where a `next_<state>` has no value and is rejected.
    # A gated-edge fold passes the target's laws, because it projects INTO the
    # target's state space -- a transition role, where those values exist.
    dag_pool = {
        **dict(deterministic_transitions),
        **{k: v for k, v in functions.items() if k != "H"},
    }
    projection_funcs: dict[StateName, Callable[..., FloatND]] = {}
    projection_args: dict[StateName, tuple[str, ...]] = {}
    for state_name in v_interpolation_info.state_names:
        target = f"{_REF_STATE_PREFIX}{state_name}"
        projection_funcs[state_name] = concatenate_functions(
            functions={
                **dag_pool,
                target: projection_func_or_fail(ref=ref, state_name=state_name),
            },
            targets=target,
            enforce_signature=False,
            set_annotations=True,
        )
        projection_args[state_name] = tuple(
            get_union_of_args([projection_funcs[state_name]])
        )
    coordinate_names = {
        f"{_REF_STATE_PREFIX}{state}" for state in v_interpolation_info.state_names
    }
    # Extra interpolator inputs beyond the coordinates and the V array (e.g.
    # runtime-supplied irregular-grid points). These are the REFERENCE
    # regime's own parameters, so they are NOT exposed as outer arguments of
    # this reader (the reading regime's caller has no such param, and nothing
    # emits the prefixed spelling they arrive under) — they are looked up per
    # call in `SAME_PERIOD_PARAMS_ARG[ref.regime]` under their qname in the
    # reference regime's OWN namespace.
    interpolator_extra_qnames = _reference_interpolator_param_qnames(
        extra_args=get_union_of_args([interpolator])
        - coordinate_names
        - {_REF_V_ARR_NAME},
        ref=ref,
    )
    arg_names = sorted(
        {arg for args in projection_args.values() for arg in args}
        | {v_mapping_arg, params_mapping_arg}
    )

    @with_signature(args=arg_names, return_annotation="FloatND")
    def read_reference_value(**kwargs: _ParamsLeaf) -> FloatND:
        regime_to_V = cast("Mapping[RegimeName, FloatND]", kwargs[v_mapping_arg])
        V_ref = regime_to_V[ref.regime]
        if ref.stakeholder_index is not None:
            # A collective reference V carries a trailing stakeholder axis;
            # read the declared stakeholder's slice (state axes only remain).
            V_ref = V_ref[..., ref.stakeholder_index]
        coordinates = {
            f"{_REF_STATE_PREFIX}{state}": projection_funcs[state](
                **{arg: kwargs[arg] for arg in projection_args[state]}
            )
            for state in v_interpolation_info.state_names
        }
        return interpolator(
            **coordinates,
            **_lookup_reference_params(
                qnames=interpolator_extra_qnames,
                regime_to_params=kwargs[params_mapping_arg],
                ref_regime=ref.regime,
            ),
            **{_REF_V_ARR_NAME: V_ref},
        )

    return read_reference_value


def _reference_interpolator_param_qnames(
    *,
    extra_args: set[str],
    ref: ResolvedProjectedRegimeValue,
) -> MappingProxyType[str, str]:
    """Map each extra interpolator input to its qname in the REFERENCE namespace.

    `get_V_interpolator` derives its runtime
    grid-helper names from the COORDINATE VARIABLE it was given
    (`_get_coordinate_finder`: `qname_from_tree_path((in_name.removeprefix(
    "next_"), "points"))`), so with `state_prefix=_REF_STATE_PREFIX` the helper
    for reference state `x` is called `__same_period_ref__x__points` while the
    reference regime's params template calls the very same quantity `x__points`.
    Stripping the coordinate prefix is exactly the inverse of the prefixing
    `get_V_interpolator` applied, and recovers the reference regime's own qname.

    Any extra input that does NOT carry the prefix cannot be attributed to a
    reference state this way; rather than bind it from a guessed namespace,
    which would silently read another regime's parameter, fail loudly at build
    time.

    Raises:
        NotImplementedError: An interpolator input could not be attributed to a
            prefixed reference coordinate.
    """
    qnames: dict[str, str] = {}
    for arg in sorted(extra_args):
        if not arg.startswith(_REF_STATE_PREFIX):
            msg = (
                f"The same-period reference reader for regime '{ref.regime}' "
                f"needs an interpolation helper argument '{arg}' that does not "
                f"derive from a prefixed reference coordinate "
                f"('{_REF_STATE_PREFIX}...'), so pylcm cannot tell which "
                "regime's parameter namespace it belongs to. Binding it from a "
                "guessed namespace would silently read another regime's "
                "parameter; this is not supported."
            )
            raise NotImplementedError(msg)
        qnames[arg] = arg.removeprefix(_REF_STATE_PREFIX)
    return MappingProxyType(qnames)


def _lookup_reference_params(
    *,
    qnames: Mapping[str, str],
    regime_to_params: object,
    ref_regime: RegimeName,
) -> dict[str, _ParamsLeaf]:
    """Resolve a reader's interpolation helpers in the REFERENCE regime's params.

    See `SAME_PERIOD_PARAMS_ARG`.

    Raises:
        KeyError: The reference regime's params are missing from the mapping, or
            do not carry a helper the reference regime's own grid needs.
    """
    if not qnames:
        return {}
    params_per_regime = cast(
        "Mapping[RegimeName, Mapping[str, _ParamsLeaf]]", regime_to_params
    )
    if ref_regime not in params_per_regime:
        msg = (
            f"Reading regime '{ref_regime}''s same-period V requires that "
            f"regime's own params (it declares runtime grid points), but "
            f"'{ref_regime}' is missing from '{SAME_PERIOD_PARAMS_ARG}' "
            f"(present: {sorted(params_per_regime)})."
        )
        raise KeyError(msg)
    ref_params = params_per_regime[ref_regime]
    resolved: dict[str, _ParamsLeaf] = {}
    for arg, qname in qnames.items():
        if qname not in ref_params:
            msg = (
                f"Interpolating regime '{ref_regime}''s same-period V needs its "
                f"parameter '{qname}', which is not in flat_params"
                f"['{ref_regime}'] (present: {sorted(ref_params)})."
            )
            raise KeyError(msg)
        resolved[arg] = ref_params[qname]
    return resolved


def get_Q_and_F_collective(
    *,
    flat_param_names: frozenset[str],
    functions: EconFunctionsMapping,
    constraints: ConstraintFunctionsMapping,
    period_targets: tuple[RegimeName, ...],
    scalar_targets: tuple[RegimeName, ...] = (),
    transitions: TransitionFunctionsMapping,
    transition_plans: TargetTransitionPlans,
    compute_regime_transition_probs: RegimeTransitionFunction,
    regime_to_v_interpolation_info: MappingProxyType[RegimeName, VInterpolationInfo],
    same_period_v_interpolation_info: MappingProxyType[RegimeName, VInterpolationInfo],
    koopmans_aggregator: EconFunction,
    stakeholders: tuple[str, ...],
    co_map_state_names: tuple[StateName, ...] = (),
    value_constraints: ConstraintFunctionsMapping = MappingProxyType({}),
    same_period_refs: Mapping[str, ResolvedProjectedRegimeValue] = MappingProxyType({}),
    continuation_functions: EconFunctionsMapping | None = None,
    gated_continuations: Mapping[RegimeName, GatedContinuationSpec] = MappingProxyType(
        {}
    ),
) -> QAndFFunction:
    """Non-terminal (Q, F) for a collective regime — per-stakeholder continuation.

    Separate from `get_Q_and_F` for the flow alone: a collective regime carries a
    per-stakeholder `utility_<s>` rather than one `utility`, and its feasibility
    mask is value-aware, so it is built AFTER `Q^s` rather than before and
    independently of it. The continuation is not separate — both builders call
    `_get_compute_CE`, which takes the stakeholder axis as a parameter.

    Per stakeholder `s`, computes `Q^s = H(u^s, E[V'^s])` with the shared
    Bellman aggregator `H` (the default `H_linear` applies `u + beta * E[V']`
    elementwise, so every stakeholder is discounted with the SAME beta). Each
    transition target must itself be a collective regime with the identical
    `stakeholders` tuple (validated at model processing), so its
    `next_V_arr` leaf carries the trailing stakeholder axis. The continuation
    interpolates the target's V over STATE axes only: the interpolator is
    evaluated once per stakeholder on the leaf's slice `next_V_arr[..., s]` and
    the results are re-stacked on a trailing axis, so the stakeholder axis
    provably rides through the stochastic-node product-map (which stacks its
    mapped axes at the front) as the last axis. For a scalar (state, action)
    cell the returned `Q` has shape `(n_stakeholders,)` while `F` is scalar;
    after the action product-map in `get_max_Q_over_a`, `Q` is
    `(*action_axes, n_stakeholders)` and `F` `(*action_axes,)` — exactly what
    the stakeholder branch there (`collective_readout`) consumes.

    No taste shocks and no nonlinear certainty equivalent: both are rejected at
    regime construction for collective regimes.

    Args:
        flat_param_names: Frozenset of flat parameter names for the regime.
        functions: Immutable mapping of function names to internal user
            functions; carries `utility_<s>` for each stakeholder in place of
            `utility`, plus the shared `H`.
        constraints: Immutable mapping of constraint names to internal user
            functions.
        period_targets: Carry targets whose continuation enters E[V^s] this
            period (all collective with the identical stakeholder tuple), read
            at the next states their laws produce.
        scalar_targets: Graph targets active next period that carry no state.
            Their value function has the stakeholder axis alone, so there is no
            next state to interpolate at and each enters E[V^s] as a single
            degenerate node weighted only by the regime transition probability.
        transitions: Immutable mapping of transition names to transition
            functions.
        transition_plans: Immutable mapping of target regime names to their
            transition laws.
        compute_regime_transition_probs: Regime transition probability function
            for solve (stakeholder-independent — per-stakeholder gating is
            carried by the gated edges, not by these probabilities).
        regime_to_v_interpolation_info: Mapping of regime names to
            V-interpolation info of the CONTINUATION, i.e. of next period's
            value arrays (state axes only; the stakeholder axis is not an
            interpolation axis).
        same_period_v_interpolation_info: Mapping of regime names to
            V-interpolation info at THIS period, for the `same_period_refs`
            readers — they interpolate a reference regime's current-period V, so
            they read the grids that regime is tabulated on now, not the ones it
            will carry next period. Required, and distinct from
            `regime_to_v_interpolation_info` as soon as a reference regime
            carries an age-specialized state.
        koopmans_aggregator: The regime's Bellman aggregator, combining each
            stakeholder's utility and certainty equivalent into `Q^s`.
        stakeholders: Ordered stakeholder names; fixes the trailing-axis order.
        co_map_state_names: Tuple of state names co-mapped with the continuation
            V (see `get_Q_and_F`).
        value_constraints: Immutable mapping of value-constraint names to
            predicates (params already renamed to qnames). Evaluated AFTER the
            per-stakeholder `Q^s`, each predicate may
            read `Q_<s>` per stakeholder, the `same_period_refs` reference
            values, and ordinary states / actions / functions / params via the
            DAG; the results are ANDed into the feasibility mask, so the
            household argmax runs over `F ∧ g(Q^s, V_ref, ...)` and an
            all-infeasible cell publishes the dissolution flag `D` downstream.
        same_period_refs: Immutable mapping of reference-value names to resolved
            same-period reference declarations. When non-empty, the returned
            `Q_and_F` carries the extra argument `SAME_PERIOD_V_ARG` — the
            mapping of reference regime names to their CURRENT-period V arrays,
            supplied per period by the solve loop (which orders the period's
            regimes so references are solved first).
        continuation_functions: Function pool the continuation sub-DAG (per-target
            state transitions and stochastic weights) is resolved against. `None`
            (the solve phase) defaults to `functions`; the simulate phase passes
            the SOLVE pool here so each stakeholder compares actions under the
            perceived law while the world is realized under the true one. Exactly
            as `get_Q_and_F` — the collective builder must not drop the phase split.
        gated_continuations: Mapping of target regime names to the gated-edge
            continuation spec that target's leaf is read under. A target absent
            from it is read as an ordinary value function.

    Returns:
        A function computing the stacked per-stakeholder state-action values
        (trailing stakeholder axis) and the shared feasibility mask for a
        non-terminal collective period.

    """
    # Phase split, mirroring get_Q_and_F: in the solve phase the two roles
    # coincide (`None`); only the simulate phase passes them apart. The
    # continuation prices the target V under the perceived law, pairing
    # `transitions` with `continuation_pool`; a sub-DAG resolved against the
    # other pool is neither phase and can reverse the household argmax.
    #
    # The flow needs no pool of its own: `next_<state>` is reserved for a
    # transition's output, so no per-stakeholder utility, feasibility or value
    # constraint reads one and the flow holds no transition node to resolve.
    continuation_pool = (
        functions if continuation_functions is None else continuation_functions
    )
    # One DAG for every stakeholder's felicity and the single feasibility mask.
    # The mask takes no stakeholder input — `_get_feasibility` is handed the
    # regime's constraints and function pool alone — so a household has one
    # action set however many felicities it carries, and the felicities' shared
    # nodes are computed once rather than once per stakeholder.
    utilities_and_F = _get_U_and_F(
        functions=functions,
        constraints=constraints,
        utility_names=tuple(f"utility_{stakeholder}" for stakeholder in stakeholders),
    )
    n_stakeholders = len(stakeholders)

    # The engine's one continuation-aggregation site, told that every target's V
    # leaf carries a trailing stakeholder axis. `certainty_equivalent=None`
    # because a collective regime rejects a nonlinear one at construction.
    #
    # `continuation_pool`, NOT `functions`: the continuation is priced under the
    # agent's perceived law, helpers included — mirroring `get_Q_and_F`.
    compute_CE, continuation_deps, continuation_arg_names = _get_compute_CE(
        functions=continuation_pool,
        period_targets=period_targets,
        scalar_targets=scalar_targets,
        transitions=transitions,
        transition_plans=transition_plans,
        compute_regime_transition_probs=compute_regime_transition_probs,
        regime_to_v_interpolation_info=regime_to_v_interpolation_info,
        certainty_equivalent=None,
        co_map_state_names=co_map_state_names,
        n_stakeholders=n_stakeholders,
        gated_continuations=gated_continuations,
    )

    _build_W_kwargs = _get_build_W_kwargs(
        functions=functions, koopmans_aggregator=koopmans_aggregator
    )

    # Build the same-period reference readers and the
    # value-constraint evaluators once; their engine-supplied arguments —
    # `Q_<s>` and the reference-value names — are excluded from the kernel
    # signature and bound per (state, action) cell inside `Q_and_F`.
    #
    # The readers interpolate a reference regime's CURRENT-period V, so they take
    # this period's interpolation info; `regime_to_v_interpolation_info` describes
    # next period's arrays and belongs to the continuation alone.
    value_constraint_machinery = _build_value_constraint_machinery(
        value_constraints=value_constraints,
        same_period_refs=same_period_refs,
        stakeholders=stakeholders,
        same_period_v_interpolation_info=same_period_v_interpolation_info,
        functions=functions,
    )

    arg_names_of_Q_and_F = _get_arg_names_of_Q_and_F(
        deps=[
            utilities_and_F,
            *continuation_deps,
            *list(value_constraint_machinery.evaluators.values()),
            *list(value_constraint_machinery.reference_readers.values()),
        ],
        include=frozenset(
            {"next_regime_to_V_arr", "period", "age"}
            | flat_param_names
            | continuation_arg_names
        ),
        exclude=value_constraint_machinery.engine_supplied_names,
    )

    @with_signature(
        args=arg_names_of_Q_and_F, return_annotation="tuple[FloatND, BoolND]"
    )
    def Q_and_F(
        next_regime_to_V_arr: MappingProxyType[RegimeName, FloatND],
        **states_actions_params: _ParamsLeaf,
    ) -> tuple[FloatND, BoolND]:
        """Per-stakeholder state-action values and the shared feasibility mask.

        Args:
            next_regime_to_V_arr: The next period's value function arrays, each
                target leaf carrying a trailing stakeholder axis.
            **states_actions_params: States, actions, age, period, and flat
                regime params.

        Returns:
            A tuple of the stacked per-stakeholder state-action value array
            (trailing stakeholder axis) and the shared feasibility mask.

        """
        *stakeholder_utilities, feasibility = utilities_and_F(**states_actions_params)
        U_stack = jnp.stack(
            [jnp.asarray(U_s) for U_s in stakeholder_utilities], axis=-1
        )
        F_arr: BoolND = jnp.asarray(feasibility)

        # The mass is stakeholder-independent — the regime transition is — so the
        # accumulator zero carries the cell shape, and the aggregator puts the
        # stakeholder axis back on the value it builds.
        CE, _ = compute_CE(
            next_regime_to_V_arr=next_regime_to_V_arr,
            zero=jnp.zeros_like(U_stack[..., 0]),
            states_actions_params=states_actions_params,
        )

        # W applied on the stacked arrays is W per stakeholder: `utility` and
        # `CE` share the trailing stakeholder axis and the aggregator's
        # parameters (e.g. the default `LinearAggregator`'s discount factor) are shared
        # across stakeholders, so the elementwise aggregation is exactly
        # Q^s = W(u^s, CE^s, beta) with the same beta for every s.
        Q_arr = koopmans_aggregator(
            utility=U_stack,
            CE=CE,
            **_build_W_kwargs(states_actions_params),
        )

        # Value-aware feasibility. Evaluated AFTER Q^s, unlike the singleton
        # path, where F is built before and independently of Q. Interpolate each
        # declared same-period reference value at the projected coordinates, then AND
        # every predicate — reading its own `Q_<s>` gathers, the reference
        # values, and ordinary cell kwargs — into the mask. The household
        # argmax downstream runs over the masked set; an all-infeasible cell
        # sets the dissolution flag D there (`collective_readout`).
        if value_constraint_machinery.evaluators:
            F_arr = _apply_value_constraints(
                machinery=value_constraint_machinery,
                Q_arr=jnp.asarray(Q_arr),
                # A constraint-less regime's F is the Python `True` scalar.
                F_arr=jnp.asarray(F_arr),
                states_actions_params=states_actions_params,
            )

        return jnp.asarray(Q_arr), jnp.asarray(F_arr)

    return Q_and_F


@dataclass(frozen=True, kw_only=True)
class _ValueConstraintMachinery:
    """Prebuilt value-constraint machinery closed over by a collective `Q_and_F`."""

    reference_readers: Mapping[str, Callable[..., FloatND]]
    """Per reference-value name, the same-period reference reader."""

    reference_reader_args: Mapping[str, tuple[str, ...]]
    """Each reader's argument names (fetched off the cell kwargs)."""

    evaluators: Mapping[str, Callable[..., BoolND]]
    """Per value-constraint name, the DAG-concatenated predicate."""

    evaluator_args: Mapping[str, tuple[str, ...]]
    """Each evaluator's argument names (split engine-supplied vs cell kwargs)."""

    q_value_index: Mapping[str, int]
    """`Q_<s>` argument name -> index on the trailing stakeholder axis."""

    engine_supplied_names: frozenset[str]
    """Names bound by the engine per cell — excluded from the kernel signature."""


def _build_value_constraint_machinery(
    *,
    value_constraints: ConstraintFunctionsMapping,
    same_period_refs: Mapping[str, ResolvedProjectedRegimeValue],
    stakeholders: tuple[str, ...],
    same_period_v_interpolation_info: MappingProxyType[RegimeName, VInterpolationInfo],
    functions: EconFunctionsMapping,
) -> _ValueConstraintMachinery:
    """Build the same-period reference readers and value-constraint evaluators once.

    Each evaluator is the predicate concatenated with
    the regime's function DAG (so it may read helper functions, exactly like
    ordinary constraints); its engine-supplied arguments — `Q_<s>` and the
    reference-value names — are bound per (state, action) cell by
    `_apply_value_constraints`.

    A value constraint is evaluated at this period's states and actions, so no
    transition enters its pool: `next_<state>` is reserved for a transition's
    output and rejected outside one. For the same reason each reader takes the
    reference regime's interpolation info at THIS period: it reads a
    current-period value array, tabulated on the grids that regime carries now.

    Args:
        value_constraints: Immutable mapping of value-constraint names to
            predicates (params already renamed to qnames).
        same_period_refs: Immutable mapping of reference-value names to resolved
            same-period reference declarations.
        stakeholders: Ordered stakeholder names; fixes which `Q_<s>` argument
            reads which slice of the trailing stakeholder axis.
        same_period_v_interpolation_info: Mapping of regime names to their
            V-interpolation info at this period.
        functions: Immutable mapping of function names to internal user
            functions.

    Returns:
        The prebuilt readers, evaluators, and the argument bookkeeping the
        kernel needs to bind them per cell.

    """
    reference_readers: dict[str, Callable[..., FloatND]] = {}
    reference_reader_args: dict[str, tuple[str, ...]] = {}
    for ref_name, ref in same_period_refs.items():
        reader = _build_same_period_ref_reader(
            ref=ref,
            v_interpolation_info=same_period_v_interpolation_info[ref.regime],
            functions=functions,
        )
        reference_readers[ref_name] = reader
        reference_reader_args[ref_name] = tuple(get_union_of_args([reader]))

    dag_pool = {k: v for k, v in functions.items() if k != "H"}
    evaluators: dict[str, Callable[..., BoolND]] = {}
    evaluator_args: dict[str, tuple[str, ...]] = {}
    for constraint_name, predicate in value_constraints.items():
        combined = {**dag_pool, constraint_name: predicate}
        evaluator = concatenate_functions(
            functions=combined,
            targets=constraint_name,
            enforce_signature=False,
            set_annotations=True,
        )
        evaluators[constraint_name] = evaluator
        evaluator_args[constraint_name] = tuple(get_union_of_args([evaluator]))

    q_value_index = {f"Q_{s}": index for index, s in enumerate(stakeholders)}
    return _ValueConstraintMachinery(
        reference_readers=MappingProxyType(reference_readers),
        reference_reader_args=MappingProxyType(reference_reader_args),
        evaluators=MappingProxyType(evaluators),
        evaluator_args=MappingProxyType(evaluator_args),
        q_value_index=MappingProxyType(q_value_index),
        engine_supplied_names=(frozenset(q_value_index) | frozenset(reference_readers)),
    )


def _apply_value_constraints(
    *,
    machinery: _ValueConstraintMachinery,
    Q_arr: FloatND,
    F_arr: BoolND,
    # `object` values: besides ordinary `_ParamsLeaf` leaves, the cell kwargs
    # carry the same-period V mapping under `SAME_PERIOD_V_ARG`.
    states_actions_params: Mapping[str, object],
) -> BoolND:
    """AND every value constraint into the feasibility of one (state, action) cell.

    Reads each declared same-period reference value at
    the projected coordinates (the readers pull the current-period reference V
    arrays off `states_actions_params[SAME_PERIOD_V_ARG]`), then evaluates each
    predicate with its `Q_<s>` arguments gathered from the trailing stakeholder
    axis of `Q_arr`, its reference-value arguments, and its remaining arguments
    from the cell kwargs.
    """
    reference_values = {
        ref_name: reader(
            **{
                arg: states_actions_params[arg]
                for arg in machinery.reference_reader_args[ref_name]
            }
        )
        for ref_name, reader in machinery.reference_readers.items()
    }
    for constraint_name, evaluate in machinery.evaluators.items():
        predicate_kwargs: dict[str, object] = {}
        for arg in machinery.evaluator_args[constraint_name]:
            if arg in machinery.q_value_index:
                predicate_kwargs[arg] = Q_arr[..., machinery.q_value_index[arg]]
            elif arg in reference_values:
                predicate_kwargs[arg] = reference_values[arg]
            else:
                predicate_kwargs[arg] = states_actions_params[arg]
        F_arr = jnp.logical_and(F_arr, evaluate(**predicate_kwargs))
    return F_arr


def _get_stakeholder_sliced_interpolator(
    *,
    base_interpolator: Callable[..., FloatND],
    V_arr_name: str,
    n_stakeholders: int,
) -> Callable[..., FloatND]:
    """Evaluate a V-interpolator per stakeholder slice of a stacked V array.

    The target regime's `next_V_arr` leaf has
    shape `(*target_state_axes, n_stakeholders)`; the base interpolator
    interpolates over the state axes of a plain `(*target_state_axes,)` array.
    Calling it once per stakeholder on the slice `next_V_arr[..., s]` and
    re-stacking on a trailing axis keeps the interpolation semantics untouched
    and puts the stakeholder axis last by construction — no axis bookkeeping
    can reorder it. The wrapper carries the base interpolator's exact argument
    names so the stochastic-variable product-map and the extra-param discovery
    treat it like the singleton interpolator.

    Args:
        base_interpolator: The singleton V-interpolator from
            `get_V_interpolator` (state axes only).
        V_arr_name: Name of the interpolator's value-array argument.
        n_stakeholders: Number of stakeholder slices on the trailing axis.

    Returns:
        A callable with the base interpolator's signature returning the
        per-stakeholder interpolated values, stakeholder axis trailing.

    """
    arg_names = tuple(get_union_of_args([base_interpolator]))

    @with_signature(args=arg_names, return_annotation="FloatND")
    def next_V_per_stakeholder(**kwargs: _ParamsLeaf) -> FloatND:
        stacked_V_arr = cast("FloatND", kwargs.pop(V_arr_name))
        return jnp.stack(
            [
                base_interpolator(**kwargs, **{V_arr_name: stacked_V_arr[..., s]})
                for s in range(n_stakeholders)
            ],
            axis=-1,
        )

    return next_V_per_stakeholder


def evaluate_projected_readers(
    *,
    readers: tuple[ProjectedLandingReader, ...],
    landing_states: Mapping[StateName, ContinuousState | DiscreteState],
    other_values: Mapping[str, object],
) -> dict[str, FloatND]:
    """Read each projected reference at one landing point.

    A gate reference and a leg fallback name another regime's value at
    coordinates a projection produces, so both are evaluated AT the landing
    rather than tabulated on the target's grid: pre-tabulating them would make
    the source interpolate `V_ref o projection`, which equals the value the
    branch pays only where the projection is affine.

    Args:
        readers: The edge's projected readers.
        landing_states: Target states at the point the source lands on, keyed
            by the reader's own (unprefixed) state name. A projection may read
            a discrete target state, so these are not all float.
        other_values: The reader's remaining arguments — the edge-reference
            value and params mappings, the target fold's period context, and
            any source params the projections read.

    Returns:
        Dict of reader name to the value read at the landing.

    """
    return {
        reader.name: reader.reader(
            **{arg: landing_states[arg] for arg in reader.state_args},
            **{arg: other_values[arg] for arg in reader.other_args},
        )
        for reader in readers
    }


@dataclass(frozen=True, kw_only=True)
class _NodeDrawResolution:
    """One target's draw-dependent laws, resolved at a single node.

    Every consumer that needs a landing coordinate for such a law goes through
    `resolve_at_node`, so the interpolated channels and a gated edge's projected
    references are read at the same landing rather than at two coordinates that
    only coincide when the law is affine in the draw.
    """

    interpolator: Callable[..., FloatND]
    """The target's V interpolator, resolving the laws below on the node axis."""

    resolve_at_node: Callable[..., Mapping[str, FloatND]]
    """Resolve the draw-dependent laws at one node, keyed `next_<state>`."""

    resolved_names: frozenset[TransitionFunctionName]
    """The `next_<state>` names `resolve_at_node` returns."""

    arg_names: tuple[str, ...]
    """Arguments `resolve_at_node` must be called with."""


def _get_pointwise_gated_interpolator(
    *,
    base_interpolator: Callable[..., FloatND],
    V_arr_name: str,
    n_channels: int,
    combine: Callable[..., FloatND],
    gate_state_names: tuple[StateName, ...],
    projected_readers: tuple[ProjectedLandingReader, ...],
    target_ages: Float1D,
    co_map_state_names: tuple[StateName, ...],
    draw_resolution: _NodeDrawResolution | None = None,
) -> Callable[..., FloatND]:
    """Read a gated edge's operand surfaces at the landing point, then gate.

    A gated target's continuation leaf is not one value function but a stack of
    the operands its gate and its branches are built from, all tabulated on the
    target's grid. Each channel is an ordinary value function and interpolates
    like one; the gate is not, because a predicate does not commute with
    interpolation. So every channel is read at the point the source lands on and
    the gate is applied there, which is the order forward simulation routes in.

    Gating the grid first and interpolating the result instead reports, in every
    cell whose corners fall on opposite sides of the gate, a blend of the open
    and the closed branch — a number neither branch pays, and one that can rank
    a source action above the one it should have taken.

    Args:
        base_interpolator: The singleton V-interpolator over one channel.
        V_arr_name: Name of the interpolator's value-array argument.
        n_channels: Length of the stacked leaf's trailing axis.
        combine: The edge's gate, applied to the interpolated channels.
        gate_state_names: Target states the gate reads, supplied at the landing
            point under their own names.
        projected_readers: Gate references and leg fallbacks. Each is evaluated
            here, at the landing, rather than read off a channel: its projection
            maps onto ANOTHER regime's grid, so tabulating it on the target's
            grid would leave the source interpolating `V_ref o projection` where
            the branch pays `V_ref(projection(landing))`.
        target_ages: Age at each model period, indexed by the fold period.
        co_map_state_names: States whose axes the caller sliced off, so no
            coordinate for them reaches this signature.
        draw_resolution: The target's draw-dependent laws, when it has any. A law
            resolved on the node axis has no coordinate in this signature either,
            so a reference landing on such a state is resolved here through the
            same node resolution the channels interpolate under, rather than
            demanded from a caller that cannot supply it.

    Returns:
        A callable returning the gated continuation, one trailing axis per leg
        for a collective source and none for a singleton one.

    Raises:
        ValueError: The gate reads a co-mapped state, which arrives with no
            landing coordinate.

    """
    co_mapped_reads = sorted(set(gate_state_names) & set(co_map_state_names))
    if co_mapped_reads:
        msg = (
            f"A gated edge's gate reads {co_mapped_reads}, which the "
            "continuation co-maps as fixed distributed state(s). A co-mapped "
            "axis is sliced off before the continuation is read, so there is "
            "no landing coordinate to evaluate the gate at. Read the state "
            "through a gate reference, or drop `distributed=True` from it."
        )
        raise ValueError(msg)
    interpolator_args = tuple(get_union_of_args([base_interpolator]))
    combine_args = tuple(get_union_of_args([combine]))
    reader_context_args = _EDGE_CONTEXT_ARGS & {
        arg for reader in projected_readers for arg in reader.other_args
    }
    context_args = (_EDGE_CONTEXT_ARGS & set(combine_args)) | reader_context_args
    reader_names = {reader.name for reader in projected_readers}
    landing_names = {
        name: f"next_{name}"
        for name in {
            *gate_state_names,
            *(arg for reader in projected_readers for arg in reader.state_args),
        }
    }
    last_period = len(target_ages) - 1
    # A landing coordinate the node resolution produces is not an argument: the
    # law that forms it reads the draw, which only exists inside the node axis.
    resolved_names = (
        draw_resolution.resolved_names if draw_resolution is not None else frozenset()
    )
    resolver_arg_names = (
        draw_resolution.arg_names if draw_resolution is not None else ()
    )
    resolve_at_node = (
        draw_resolution.resolve_at_node if draw_resolution is not None else None
    )
    resolved_landing = {
        landing for landing in landing_names.values() if landing in resolved_names
    }
    outer_arg_names = sorted(
        (
            set(interpolator_args)
            | {
                landing_names.get(name, name)
                for name in combine_args
                if name not in context_args
            }
            | {
                landing_names[arg]
                for reader in projected_readers
                for arg in reader.state_args
            }
            | {
                arg
                for reader in projected_readers
                for arg in reader.other_args
                if arg not in context_args
            }
            | ({"period"} if context_args else set())
            | (set(resolver_arg_names) if resolved_landing else set())
        )
        - {EDGE_CHANNELS_ARG}
        - reader_names
        - resolved_landing
    )

    @with_signature(args=outer_arg_names, return_annotation="FloatND")
    def next_V_gated(**kwargs: _ParamsLeaf) -> FloatND:
        stacked = cast("FloatND", kwargs[V_arr_name])
        interpolator_kwargs = {
            name: kwargs[name] for name in interpolator_args if name != V_arr_name
        }
        channels = jnp.stack(
            [
                base_interpolator(
                    **interpolator_kwargs, **{V_arr_name: stacked[..., channel]}
                )
                for channel in range(n_channels)
            ],
            axis=-1,
        )
        # Resolved on the same node the channels were just interpolated on, from
        # the same pure laws, so a reference and the channel it is gated against
        # cannot land on two different coordinates.
        landing: dict[str, FloatND] = {}
        if resolve_at_node is not None and resolved_landing:
            landing = dict(
                resolve_at_node(**{name: kwargs[name] for name in resolver_arg_names})
            )
        context: dict[str, FloatND] = {}
        if context_args:
            # The gate speaks about the period the source LANDS in. Clipping
            # only affects a period from which no edge folds, where the value
            # is not read.
            target_period = jnp.clip(
                jnp.asarray(kwargs["period"], dtype=jnp.int32) + 1, 0, last_period
            )
            context = {"period": target_period, "age": target_ages[target_period]}
        # Each projected reference is read HERE, at the landing coordinates,
        # so the source collects the number its branch actually pays. A
        # projection declaring `period` or `age` means the TARGET fold's, the
        # same context it would have been handed inside the fold — never the
        # source's own.
        projected_values = evaluate_projected_readers(
            readers=projected_readers,
            landing_states={
                arg: cast(
                    "ContinuousState | DiscreteState",
                    landing[landing_names[arg]]
                    if landing_names[arg] in landing
                    else kwargs[landing_names[arg]],
                )
                for reader in projected_readers
                for arg in reader.state_args
            },
            other_values={
                arg: context[arg] if arg in context_args else kwargs[arg]
                for reader in projected_readers
                for arg in reader.other_args
            },
        )
        return combine(
            **{EDGE_CHANNELS_ARG: channels},
            **projected_values,
            **{
                name: context[name]
                if name in context_args
                else landing.get(
                    landing_names.get(name, name),
                    kwargs.get(landing_names.get(name, name)),
                )
                for name in combine_args
                if name != EDGE_CHANNELS_ARG and name not in reader_names
            },
        )

    return next_V_gated


def partition_continuation_targets(
    *,
    targets: tuple[RegimeName, ...],
    regime_to_v_interpolation_info: Mapping[RegimeName, VInterpolationInfo],
) -> tuple[tuple[RegimeName, ...], tuple[RegimeName, ...]]:
    """Partition canonical graph targets into stateful and stateless tuples.

    Membership comes entirely from `targets`. Interpolation metadata classifies how
    each continuation is read without adding or removing graph edges.

    Args:
        targets: Canonical graph targets for one source and period.
        regime_to_v_interpolation_info: Mapping of regime names to interpolation
            metadata whose state names determine the continuation representation.

    Returns:
        Tuple of `(stateful_targets, scalar_targets)` preserving graph order.

    """
    stateful_targets = tuple(
        target
        for target in targets
        if regime_to_v_interpolation_info[target].state_names
    )
    scalar_targets = tuple(
        target
        for target in targets
        if not regime_to_v_interpolation_info[target].state_names
    )
    return stateful_targets, scalar_targets


def _get_compute_CE(
    *,
    functions: EconFunctionsMapping,
    period_targets: tuple[RegimeName, ...],
    scalar_targets: tuple[RegimeName, ...] = (),
    transitions: TransitionFunctionsMapping,
    transition_plans: TargetTransitionPlans,
    compute_regime_transition_probs: RegimeTransitionFunction,
    regime_to_v_interpolation_info: MappingProxyType[RegimeName, VInterpolationInfo],
    certainty_equivalent: CertaintyEquivalent | None,
    co_map_state_names: tuple[StateName, ...],
    n_stakeholders: int | None = None,
    gated_continuations: Mapping[RegimeName, GatedContinuationSpec] = MappingProxyType(
        {}
    ),
) -> tuple[
    Callable[..., tuple[FloatND, MappingProxyType[RegimeName, FloatND]]],
    tuple[Callable[..., Any], ...],
    frozenset[str],
]:
    """Build the closure that aggregates next period's value into `CE`.

    The single continuation-aggregation site of the engine: the Bellman `Q` of a
    singleton regime, the per-stakeholder `Q^s` of a collective one, and the NaN
    diagnostics all call the closure this returns, so they cannot disagree. The
    continuation is a lottery over the stochastic nodes of every reachable target
    regime, weighted by that target's regime-transition probability:

    - Without a certainty equivalent, it is aggregated as the linear
      expectation `Σ_r p_r · E_w[V'_r]`.
    - With one, the whole joint lottery is handed to
      `CertaintyEquivalent.aggregate` in one piece, which applies the
      transform before every expectation and inverts exactly once. Flattening
      before aggregating is what lets `PowerMean` anchor the transform, which
      a per-target `transform -> reduce -> inverse` decomposition cannot.

    A target carrying no state joins that same lottery as a single degenerate
    node, so it is transformed with every other node rather than on its own.

    Args:
        functions: Immutable mapping of function names to internal user functions.
            This is the CONTINUATION pool, not the flow pool: every use of it here
            builds a next-period object, which is priced under the perceived
            (solve-phase) law. Callers pass `continuation_pool`. Do not add a use
            that needs the flow pool without taking it as its own argument.
        period_targets: Carry targets whose continuation is read at the next
            states their laws produce.
        scalar_targets: Targets carrying no state, whose rank-zero value enters
            as one degenerate lottery node.
        transitions: Immutable mapping of transition names to transition functions.
        transition_plans: Immutable mapping of target regime names to their
            transition laws.
        compute_regime_transition_probs: Regime transition probability function
            for solve.
        regime_to_v_interpolation_info: Immutable mapping of regime names to
            V-interpolation info.
        certainty_equivalent: Nonlinear certainty equivalent declared by the
            regime, or `None` for the linear expectation. A collective regime
            rejects a nonlinear one at construction, so it always passes `None`.
        co_map_state_names: Tuple of state names co-mapped with the continuation V.
        n_stakeholders: Length of the trailing stakeholder axis the continuation
            carries, or `None` for a singleton regime. It is the only structural
            difference a collective regime makes to the aggregation: every
            target's V leaf, and hence `CE`, gains one trailing axis, while the
            regime-transition probabilities — which are stakeholder-independent —
            keep the plain cell shape and broadcast across it.
        gated_continuations: Mapping of target regime names to the gated-edge
            continuation spec that target's leaf is read under. A target absent
            from it is read as an ordinary value function.

    Returns:
        Tuple of the closure returning `(CE, active_regime_probs)`, the
        dependencies whose arguments must enter the calling closure's signature,
        and the further argument names that signature must carry for weights
        formed inside the node axes.

    """
    continuations = {
        target_regime_name: _build_target_continuation(
            target_regime_name=target_regime_name,
            functions=functions,
            bundle=transitions.get(target_regime_name, MappingProxyType({})),
            transition_plans=transition_plans,
            v_interpolation_info=regime_to_v_interpolation_info[target_regime_name],
            co_map_state_names=co_map_state_names,
            n_stakeholders=n_stakeholders,
            gated_continuation=gated_continuations.get(target_regime_name),
        )
        for target_regime_name in period_targets
    }
    # A gated target carrying no state has no landing coordinate to interpolate
    # at, but its leaf is still the fold's channel stack rather than a rank-zero
    # value: the gate and the leg fallbacks are applied to that stack before it
    # enters the mixture, so the source pays the branch the gate selects.
    gated_scalar_readers = {
        target_regime_name: _get_pointwise_gated_interpolator(
            base_interpolator=get_V_interpolator(
                v_interpolation_info=regime_to_v_interpolation_info[target_regime_name],
                state_prefix="next_",
                V_arr_name="next_V_arr",
            ),
            V_arr_name="next_V_arr",
            n_channels=spec.n_channels,
            combine=spec.combine,
            gate_state_names=spec.gate_state_names,
            projected_readers=spec.projected_readers,
            target_ages=spec.target_ages,
            co_map_state_names=co_map_state_names,
        )
        for target_regime_name in scalar_targets
        if (spec := gated_continuations.get(target_regime_name)) is not None
    }
    gated_scalar_arg_names = {
        target_regime_name: tuple(
            arg for arg in get_union_of_args([reader]) if arg != "next_V_arr"
        )
        for target_regime_name, reader in gated_scalar_readers.items()
    }

    # The plain expectation reduces each target on its own; every other
    # certainty equivalent needs the whole joint lottery in one piece, because
    # its transform has to be applied before any expectation is taken.
    # `LinearExpectation.aggregate` states the same quantity over the flattened
    # lottery, but reducing per target is materially cheaper.
    # Exact type, not `isinstance`: a subclass overriding `aggregate` states a
    # different quantity, and the per-target route would silently discard the
    # override.
    reduces_per_target = (
        certainty_equivalent is None or type(certainty_equivalent) is LinearExpectation
    )
    ce_flat_param_names = (
        MappingProxyType({})
        if certainty_equivalent is None
        else certainty_equivalent.flat_param_names
    )

    # Co-mapped states are sliced off each `next_V_arr` leaf by the backward-
    # induction co-map, so their `next_`-prefixed coordinates are not passed to
    # the interpolator, which does not index those axes.
    co_map_next_names = frozenset(f"next_{name}" for name in co_map_state_names)

    def compute_CE(
        *,
        next_regime_to_V_arr: MappingProxyType[RegimeName, FloatND],
        zero: FloatND,
        states_actions_params: Mapping[str, Any],
    ) -> tuple[FloatND, MappingProxyType[RegimeName, FloatND]]:
        """Aggregate the continuation lottery into `CE` at one state-action point.

        Args:
            next_regime_to_V_arr: Immutable mapping of target regime names to
                next period's value function arrays.
            zero: Zero at the shape and dtype the regime-transition mass
                accumulates in — the caller's cell shape, without the trailing
                stakeholder axis, since the regime transition is
                stakeholder-independent. The value being built up carries that
                axis on top of it.
            states_actions_params: Mapping of states, actions, age, period, and
                flat regime params. Forwarded verbatim to the transition and
                probability functions, so it carries whatever the caller
                supplies — including params that never passed through
                `cast_params_to_canonical_dtypes`.

        Returns:
            Tuple of the aggregated continuation value and the regime transition
            probabilities of the reachable targets.

        """
        regime_transition_probs: MappingProxyType[RegimeName, FloatND] = (
            compute_regime_transition_probs(**states_actions_params)
        )
        active_regime_probs = MappingProxyType(
            {r: regime_transition_probs[r] for r in (*period_targets, *scalar_targets)}
        )
        # Every target's own lottery is built before any of them is weighted,
        # because the common factor that puts the whole continuation on a scale
        # the dtype can multiply depends on both halves of each node's
        # probability — the regime's and the node's. A regime probability lifted
        # only into the normal range lands back under it as soon as a quadrature
        # weight of a sixth multiplies it.
        #
        # Each target's nodes arrive carrying their own scales — a joint
        # product below the normal range travels as a number and a shift — and
        # they keep them. Forcing them onto one scale here is what no lottery
        # survives: the ratio between the rarest node and the likeliest is
        # exactly the quantity the format cannot hold, so the node that decides
        # a nonlinear continuation is the one that would be lost. Each consumer
        # below takes the pairs and reduces them where the spread costs
        # nothing.
        target_lotteries = {
            target_regime_name: continuations[target_regime_name].joint_lottery_weights(
                **continuations[target_regime_name].lottery_weights(
                    **states_actions_params
                )
            )
            for target_regime_name in period_targets
        }
        target_node_weights = {r: target_lotteries[r][0] for r in period_targets}
        target_node_shifts = {r: target_lotteries[r][1] for r in period_targets}
        # Unit mass alone does not make a collection of weights a distribution:
        # 1.5 and -0.5 sum to one. Non-negativity is tracked alongside the sum
        # so it is arithmetic too, and the two together give the whole range —
        # non-negative weights summing to one each lie in [0, 1].
        #
        # It is tracked as a decision, not as the smallest weight: reducing the
        # weights with `jnp.minimum` and testing the survivor's sign loses a
        # negative probability the dtype cannot hold as a normal number, which
        # arrives at the test as `-0` and passes it. Each target's sign is read
        # off its own bits, while it still has them.
        #
        # The accumulators are seeded with the stateless targets, which carry no
        # node axis of their own but do carry mass, a sign, and — under a
        # nonlinear certainty equivalent — one lottery node each.
        (
            mixture_terms,
            lottery_values,
            lottery_weights,
            lottery_shifts,
            probability_mass,
            has_negative_probability,
        ) = _scalar_target_contribution(
            scalar_targets=scalar_targets,
            next_regime_to_V_arr=MappingProxyType(
                {
                    **next_regime_to_V_arr,
                    **{
                        target_regime_name: reader(
                            next_V_arr=next_regime_to_V_arr[target_regime_name],
                            **{
                                arg: states_actions_params[arg]
                                for arg in gated_scalar_arg_names[target_regime_name]
                            },
                        )
                        for target_regime_name, reader in gated_scalar_readers.items()
                    },
                }
            ),
            active_regime_probs=active_regime_probs,
            as_lottery=not reduces_per_target,
            zero=zero,
        )
        for target_regime_name in period_targets:
            continuation = continuations[target_regime_name]
            next_states = continuation.next_states(**states_actions_params)
            joint_next_stochastic_states_weights = target_node_weights[
                target_regime_name
            ]

            # As we productmap'd the value function over the stochastic variables, the
            # resulting next value function gets a new dimension for each stochastic
            # variable.
            extra_kw = {
                k: states_actions_params[k] for k in continuation.extra_param_names
            }
            interpolator_coordinates = {
                name: val
                for name, val in next_states.items()
                if name not in co_map_next_names
            }
            next_V_at_stochastic_states_arr = continuation.next_V(
                **interpolator_coordinates,
                next_V_arr=next_regime_to_V_arr[target_regime_name],
                **extra_kw,
            )

            # A node the target's own lottery gives zero probability is never
            # realized, so whatever a law names there -- a value off the target's
            # support, and so a NaN out of the interpolator -- is not part of the
            # model. Both aggregation routes drop such a node rather than
            # multiplying it by its zero weight, since `0 * nan` is `nan`: the
            # per-target route in `zero_safe_average`, the lottery route in the
            # certainty equivalent's own `aggregate`.
            # The mass sum reads the weight as the arithmetic sees it: a
            # probability too small for the dtype to hold contributes nothing
            # to a total of order one, which is the right answer. Its sign is a
            # different question, and one arithmetic cannot answer at that size,
            # so it is read from the bits. The probability enters the weighted
            # term at its own size — `balanced_product` moves the exponent onto
            # it from the value it meets — so no rescaling stands between the
            # model's number and the one that is multiplied.
            target_probability = active_regime_probs[target_regime_name]
            probability_mass = probability_mass + target_probability
            has_negative_probability = has_negative_probability | is_negative(
                target_probability
            )

            if reduces_per_target:
                next_V_expected_arr = _expected_continuation_over_nodes(
                    values=next_V_at_stochastic_states_arr,
                    weights=joint_next_stochastic_states_weights,
                    shifts=target_node_shifts[target_regime_name],
                    has_lottery_axes=continuation.has_lottery_axes,
                    n_stakeholders=n_stakeholders,
                )
                # Collect the UNMULTIPLIED `(prob, expected V)`; the mixture is
                # reduced ONCE by `_sum_regime_mixture`: form each target's
                # zero-safe contribution on its native cell shape, put the
                # separate contributions in value order, then sum. See that
                # helper for why the order must not depend on regime LABELS.
                #
                # Multiplying here would put the term outside the helper's
                # zero-safe contribution and value-ordering boundaries. The
                # `0 * -inf` hazard that multiplication would raise (a
                # zero-probability target carrying an admissible `-inf`) is handled one
                # level in, by `zero_safe_weighted_term` inside `_sum_regime_mixture`.
                mixture_terms.append(
                    (
                        target_regime_name,
                        target_probability,
                        next_V_expected_arr,
                    )
                )
            else:
                values, node_weights, node_shifts = _as_lottery(
                    values=next_V_at_stochastic_states_arr,
                    weights=joint_next_stochastic_states_weights,
                    shifts=target_node_shifts[target_regime_name],
                    has_stochastic_states=continuation.has_lottery_axes,
                )
                # An impossible node is neutralized once, downstream in
                # `_aggregate_joint_lottery`, where the concatenated weights are
                # final and the stand-in can be copied from a node that carries
                # mass. Masking here would state the same rule a second time,
                # with the constant this route rejects.
                lottery_values.append(values)
                # The regime probability and the node weight are two more
                # factors of the same joint event, so their product carries the
                # same refusal as the product across the stochastic axes. The
                # node weights arrive normalized, within a factor of the node
                # count of one, so the product spans no more than they do and
                # one scale covers this arm. Whatever spread the target's
                # lottery had is in the node scales, and it passes through
                # untouched.
                weighted_nodes, product_shift = scaled_joint_weight(
                    jnp.stack(jnp.broadcast_arrays(target_probability, node_weights))
                )
                lottery_weights.append(weighted_nodes)
                lottery_shifts.append(product_shift + node_shifts)

        # ONE value-ordered reduction for the whole regime mixture, expressed
        # without a materialized target axis. Its order must not depend on regime
        # LABELS. `mixture_terms` is empty on the lottery route, where
        # `_sum_regime_mixture` returns `zeros_like(like)` -- the additive identity
        # the branches below then compose with.
        # `like` is the shape of a VALUE, so collectively it carries the trailing
        # stakeholder axis the mass-shaped `zero` does not.
        CE = _sum_regime_mixture(
            mixture_terms=mixture_terms,
            like=_value_shaped_zero(zero=zero, n_stakeholders=n_stakeholders),
        )

        if reduces_per_target and (period_targets or scalar_targets):
            CE = _normalized_regime_mixture(
                mixture=CE,
                probability_mass=probability_mass,
                has_negative_probability=has_negative_probability,
            )
        elif certainty_equivalent is not None:
            # `aggregate` normalizes by the weight sum itself, so the lottery
            # route has no division to attach the check to. Selecting between
            # the aggregate and NaN leaves the well-formed path free of any
            # arithmetic at all, which a multiplication by `1.0` would not.
            #
            # With no node collected there is nothing to hand `aggregate` — it
            # would reduce over an empty axis — so the selection falls back to
            # the initialized `CE`. The mask is `False` there regardless, since
            # a mass of zero is not unit mass.
            CE = jnp.where(
                _regime_mass_is_a_distribution(
                    probability_mass=probability_mass,
                    has_negative_probability=has_negative_probability,
                ),
                _aggregate_joint_lottery(
                    certainty_equivalent=certainty_equivalent,
                    lottery_values=lottery_values,
                    lottery_weights=lottery_weights,
                    lottery_shifts=lottery_shifts,
                    ce_flat_param_names=ce_flat_param_names,
                    states_actions_params=states_actions_params,
                )
                if lottery_values
                else CE,
                jnp.nan,
            )

        return CE, active_regime_probs

    deps = (
        compute_regime_transition_probs,
        *(c.next_states for c in continuations.values()),
        *(c.lottery_weights for c in continuations.values()),
    )
    continuation_arg_names = frozenset(
        name
        for continuation in continuations.values()
        for name in continuation.extra_param_names
    ) | frozenset(
        arg for arg_names in gated_scalar_arg_names.values() for arg in arg_names
    )
    return compute_CE, deps, continuation_arg_names


@dataclasses.dataclass(frozen=True, kw_only=True)
class _TargetContinuation:
    """Everything built once for one reachable target's continuation."""

    next_states: NextStateSimulationFunction
    """Next-period states of this target at one state-action point."""

    lottery_weights: Callable[..., dict[str, FloatND | IntND]]
    """Marginal probabilities of the target's stochastic laws."""

    joint_lottery_weights: Callable[..., tuple[FloatND, IntND]]
    """Outer product of the lottery marginals, over the node axes."""

    next_V: Callable[..., FloatND]
    """Target's value function, product-mapped over its lottery axes.

    A declared entry gets no axis: its one value is interpolated on the target's
    nodes inside the interpolator, so the surface carries genuine draws only.
    """

    extra_param_names: frozenset[str]
    """Arguments `next_V` needs beyond the next states and the value array.

    Examples are a grid whose points arrive at runtime (`wealth__points` for an
    `IrregSpacedGrid`) and a source action read by an output of a joint transition.
    """

    has_lottery_axes: bool
    """Whether the target draws anything, i.e. whether `next_V` has lottery axes."""

    lottery_axis_names: tuple[TransitionFunctionName, ...] = ()
    """Stochastic `next_<state>` names, in the order their axes appear."""

    draw_dependent_names: frozenset[TransitionFunctionName] = frozenset()
    """Laws resolved on a node axis, one value per node of a sibling draw."""


def _draw_dependencies_by_law(
    *,
    bundle: MappingProxyType[TransitionFunctionName, TransitionFunction],
    functions: EconFunctionsMapping,
    stochastic_names: tuple[TransitionFunctionName, ...],
) -> MappingProxyType[TransitionFunctionName, frozenset[TransitionFunctionName]]:
    """Return, per deterministic law, which of the target's own draws it reads.

    A law reading `next_<state>` of a stochastic sibling depends on which node the
    draw lands on, so its value is one number per node rather than one number. The
    dependence travels through helpers, so the walk is transitive.

    Which draws a law reads — not merely whether it reads any — is what the
    consumers need: the interpolator substitutes exactly those, and the
    construction-time support requirement falls on exactly those and no other
    stochastic sibling.

    Args:
        bundle: This target's unqualified `next_<state>` transition functions.
        functions: Immutable mapping of function names to internal user functions.
        stochastic_names: This target's stochastic `next_<state>` names.

    Returns:
        Immutable mapping of each draw-dependent law in `bundle` order to the
        draws it reads. Laws reading none are absent.

    """
    draws = set(stochastic_names)
    candidates = {
        name: func
        for name, func in (dict(functions) | dict(bundle)).items()
        if name not in draws
    }
    reads: dict[TransitionFunctionName, frozenset[TransitionFunctionName]] = {
        name: frozenset() for name in candidates
    }
    growing = True
    while growing:
        growing = False
        for name, func in candidates.items():
            args = [arg for arg in get_annotations(func) if arg != "return"]
            found = {arg for arg in args if arg in draws}
            found |= {draw for arg in args if arg in reads for draw in reads[arg]}
            merged = reads[name] | found
            if merged != reads[name]:
                reads[name] = merged
                growing = True
    return MappingProxyType({name: reads[name] for name in bundle if reads.get(name)})


def _fail_if_a_draw_reads_a_sibling_draw(
    *,
    target_regime_name: RegimeName,
    lottery_weights: Callable[..., dict[str, FloatND | IntND]],
    stochastic_names: tuple[TransitionFunctionName, ...],
) -> None:
    """Check that no draw's own distribution is conditioned on another draw.

    Each draw contributes its own node axis and its own weight vector, and the
    joint distribution over those axes is formed as the product of the marginals.
    A draw whose probabilities depend on where a sibling landed is not that
    product — it is a genuine joint kernel, and no product of marginals expresses
    the correlation it describes.

    The dependence is read off the weight DAG's own arguments, which is where it
    surfaces: a draw has no realized value while the expectation over it is being
    built, so a sibling's weights asking for one leaves it unbound.

    Args:
        target_regime_name: Regime whose laws are being built, named in the message.
        lottery_weights: DAG producing this target's probability weight vectors.
        stochastic_names: This target's stochastic `next_<state>` names.

    Raises:
        ModelInitializationError: If a draw's weights read a sibling draw.

    """
    read_draws = sorted(get_union_of_args([lottery_weights]) & set(stochastic_names))
    if not read_draws:
        return
    named = ", ".join(f"'{name}'" for name in read_draws)
    msg = (
        f"A draw of regime '{target_regime_name}' has probabilities that read "
        f"{named}, which are draws of the same regime. Each draw carries its own "
        f"nodes and its own probabilities, and their joint distribution is formed "
        f"as the product of those marginals — a distribution conditioned on where "
        f"a sibling landed is a joint kernel that product cannot express. Declare "
        f"the two as one state with a joint law, or condition a deterministic law "
        f"on the draw instead of the draw's own distribution."
    )
    raise ModelInitializationError(msg)


def _fail_if_a_read_draw_has_no_nodes_yet(
    *,
    target_regime_name: RegimeName,
    dependencies_by_law: MappingProxyType[
        TransitionFunctionName, frozenset[TransitionFunctionName]
    ],
    v_interpolation_info: VInterpolationInfo,
) -> None:
    """Check that every draw a law reads has a support while the model builds.

    Resolving a dependent law on the node axis reads the nodes themselves, so a
    process whose law arrives at runtime has nothing to resolve against — its
    nodes are not numbers yet. The requirement falls on the draws that are read
    and on no other stochastic sibling: a process nothing resolves against is
    free to receive its law at solve time, as any carried process is.

    Args:
        target_regime_name: Regime whose laws are being built, named in the message.
        dependencies_by_law: Immutable mapping of each draw-dependent law to the
            draws it reads.
        v_interpolation_info: The target's V-interpolation info, holding the grids.

    Raises:
        ModelInitializationError: If a read draw's process is not fully specified.

    """
    readers_by_draw: dict[TransitionFunctionName, list[TransitionFunctionName]] = {}
    for law_name, draws in dependencies_by_law.items():
        for draw in draws:
            readers_by_draw.setdefault(draw, []).append(law_name)

    for next_state_name in sorted(readers_by_draw):
        state_name = next_state_name.removeprefix("next_")
        if state_name not in v_interpolation_info.discrete_states:
            # A transition-local joint node carries its own explicit support,
            # not a target-state grid axis.
            continue
        grid = v_interpolation_info.discrete_states[state_name]
        if getattr(grid, "is_fully_specified", True):
            continue
        msg = (
            f"{', '.join(sorted(readers_by_draw[next_state_name]))} of regime "
            f"'{target_regime_name}' reads the draw '{next_state_name}', but that "
            f"process is parameterized at runtime, so its nodes are not known "
            f"while the model builds. A law reading a draw is resolved on that "
            f"draw's own nodes, which requires them to be fixed at construction — "
            f"through the process constructor or `fixed_params`."
        )
        raise ModelInitializationError(msg)


def _get_interpolator_resolving_draws(
    *,
    next_V_interpolator: Callable[..., FloatND],
    bundle: MappingProxyType[TransitionFunctionName, TransitionFunction],
    functions: EconFunctionsMapping,
    stochastic_names: tuple[TransitionFunctionName, ...],
    draw_dependent_names: tuple[TransitionFunctionName, ...],
    node_values: MappingProxyType[TransitionFunctionName, Any],
    support_provider_names: MappingProxyType[TransitionFunctionName, str],
) -> _NodeDrawResolution:
    """Wrap the interpolator so draw-dependent laws resolve on the node axis.

    The caller product-maps the result over the target's node axes, so one call
    sees one node per stochastic law. The draw's *index* is what indexes the value
    function; the draw's *value* is what a dependent law reads. Both come from the
    same node, which is why resolving them here — inside the axis the process
    already contributes — needs no second axis and no parameter for the draw.

    The resolution is published alongside the interpolator rather than sealed
    inside it, because a gated edge's projected references are read at the point
    the source lands and need the same landing coordinates. A law resolved here
    is absent from the interpolator's signature, so a consumer that only saw the
    interpolator would have to demand a coordinate no caller can supply.

    Args:
        next_V_interpolator: The target's value-function interpolator.
        bundle: This target's unqualified `next_<state>` transition functions.
        functions: Immutable mapping of function names to internal user functions.
        stochastic_names: This target's stochastic `next_<state>` names.
        draw_dependent_names: Laws to resolve here rather than ahead of the axes.
        node_values: Immutable mapping of each stochastic law to its nodes, indexed
            by the value its next-state function yields.

    Returns:
        The interpolator, which carries the interpolator's signature minus the
        laws it resolves itself plus whatever resolving them reads, bundled with
        the node resolution those laws come from.

    """
    resolve = concatenate_functions(
        functions={
            name: func
            for name, func in (dict(bundle) | dict(functions)).items()
            if name not in stochastic_names
        },
        targets=list(draw_dependent_names),
        return_type="dict",
        enforce_signature=False,
        set_annotations=True,
    )
    resolver_args = get_union_of_args([resolve])
    interpolator_args = get_union_of_args([next_V_interpolator])
    read_as_a_draw = tuple(name for name in stochastic_names if name in resolver_args)
    support_args = {
        support_provider_names[name]
        for name in read_as_a_draw
        if name in support_provider_names
    }
    arg_names = sorted(
        (interpolator_args - set(draw_dependent_names)) | resolver_args | support_args
    )

    resolver_arg_names = sorted(resolver_args | support_args)

    @with_signature(args=resolver_arg_names)
    def resolve_at_this_node(**kwargs: Any) -> Mapping[str, FloatND]:  # noqa: ANN401
        drawn: dict[str, Any] = {}
        for name in read_as_a_draw:
            index = kwargs[name].astype(jnp.int32)
            if name in support_provider_names:
                support = kwargs[support_provider_names[name]]
                drawn[name] = jax.tree_util.tree_map(
                    lambda leaf, node_index=index: leaf[node_index],
                    support,
                )
            else:
                drawn[name] = node_values[name][index]
        return resolve(
            **{
                k: v for k, v in kwargs.items() if k in resolver_args and k not in drawn
            },
            **drawn,
        )

    @with_signature(args=arg_names)
    def interpolate_at_this_node(**kwargs: Any) -> FloatND:  # noqa: ANN401
        resolved = resolve_at_this_node(
            **{k: v for k, v in kwargs.items() if k in resolver_arg_names}
        )
        return next_V_interpolator(
            **{k: v for k, v in kwargs.items() if k in interpolator_args},
            **resolved,
        )

    return _NodeDrawResolution(
        interpolator=interpolate_at_this_node,
        resolve_at_node=resolve_at_this_node,
        resolved_names=frozenset(draw_dependent_names),
        arg_names=tuple(resolver_arg_names),
    )


def _build_target_continuation(
    *,
    target_regime_name: RegimeName,
    functions: EconFunctionsMapping,
    bundle: MappingProxyType[TransitionFunctionName, TransitionFunction],
    transition_plans: TargetTransitionPlans,
    v_interpolation_info: VInterpolationInfo,
    co_map_state_names: tuple[StateName, ...],
    n_stakeholders: int | None,
    gated_continuation: GatedContinuationSpec | None = None,
) -> _TargetContinuation:
    """Build one target's continuation machinery.

    A law that carries weights is either a lottery or a declared entry, and only
    a lottery's weights are probabilities. The distinction decides what the value
    function is mapped over: a lottery gets a node axis, because its outcome is
    genuinely uncertain and every node can occur; a declared entry names one
    value, which the interpolator places on the target's nodes without the axis
    ever being formed.

    Args:
        target_regime_name: Regime the continuation leads into.
        functions: Immutable mapping of function names to internal user functions.
        bundle: This target's unqualified `next_<state>` transition functions.
        transition_plans: Immutable mapping of target regime names to their
            transition laws.
        v_interpolation_info: The target's V-interpolation info.
        co_map_state_names: Tuple of state names co-mapped with the continuation V.
        n_stakeholders: Length of the trailing stakeholder axis each target's
            `next_V_arr` leaf carries, or `None` for a singleton regime whose
            leaves carry no such axis. It is not an interpolation axis: the
            interpolator is evaluated once per stakeholder slice and the results
            are re-stacked last, so the axis rides through the node product-map
            (which stacks its mapped axes at the front) by construction.
        gated_continuation: How to turn this target's stacked operand channels
            into one value per leg at the landing point, or `None` when the
            target's leaf is an ordinary value function.

    Returns:
        The target's continuation machinery.

    """
    lottery_variables = tuple(
        key for key in bundle if transition_plans[target_regime_name].is_lottery(key)
    )
    basis_variables = tuple(
        key
        for key in bundle
        if transition_plans[target_regime_name].has_interpolation_basis(key)
    )
    # A declared entry names one value on the target's node axis, so it is
    # interpolated there rather than enumerated: only a genuine draw gets an axis
    # of its own on the continuation surface. Enumerating a declared entry instead
    # would make the surface Cartesian in the entered dimensions -- the product of
    # their node counts at every state-action point -- to state a single number.
    node_variables = lottery_variables

    V_arr_name = "next_V_arr"
    next_V_interpolator = get_V_interpolator(
        v_interpolation_info=v_interpolation_info,
        state_prefix="next_",
        V_arr_name=V_arr_name,
        co_map_state_names=co_map_state_names,
        entered_process_names=tuple(
            name.removeprefix("next_") for name in basis_variables
        ),
    )

    # A law reading one of this target's own draws has one value per node, so it
    # is resolved inside the node axes rather than once ahead of them. Which
    # consumer resolves it depends on what the law is: a law feeding a coordinate
    # is resolved by the interpolator, a declared entry by its basis weights.
    lottery_weights = get_next_stochastic_weights_function(
        functions=functions,
        transitions=bundle,
        transition_plans=transition_plans,
        regime_name=target_regime_name,
    )
    _fail_if_a_draw_reads_a_sibling_draw(
        target_regime_name=target_regime_name,
        lottery_weights=lottery_weights,
        stochastic_names=lottery_variables,
    )
    dependencies_by_law = _draw_dependencies_by_law(
        bundle=bundle, functions=functions, stochastic_names=lottery_variables
    )
    # A declared entry is a coordinate like any other, so a law reading a
    # sibling draw is resolved inside that draw's axes whether it feeds a
    # coordinate or an entry.
    dependent_coordinate_names = tuple(dependencies_by_law)
    node_values = MappingProxyType(
        {
            name: v_interpolation_info.discrete_states[
                name.removeprefix("next_")
            ].to_jax()
            for name in lottery_variables
            if transition_plans[target_regime_name].lotteries[name].lifetime
            is not LotteryLifetime.TRANSITION_LOCAL
        }
    )
    support_provider_names = MappingProxyType(
        {
            name: cast(
                "str",
                transition_plans[target_regime_name]
                .lotteries[name]
                .support_provider_name,
            )
            for name in lottery_variables
            if transition_plans[target_regime_name]
            .lotteries[name]
            .support_provider_name
            is not None
        }
    )
    if dependencies_by_law:
        _fail_if_a_read_draw_has_no_nodes_yet(
            target_regime_name=target_regime_name,
            dependencies_by_law=dependencies_by_law,
            v_interpolation_info=v_interpolation_info,
        )
    draw_resolution: _NodeDrawResolution | None = None
    if dependent_coordinate_names:
        draw_resolution = _get_interpolator_resolving_draws(
            next_V_interpolator=next_V_interpolator,
            bundle=bundle,
            functions=functions,
            stochastic_names=lottery_variables,
            draw_dependent_names=dependent_coordinate_names,
            node_values=node_values,
            support_provider_names=support_provider_names,
        )
        next_V_interpolator = draw_resolution.interpolator

    # The stakeholder axis is put on last, after every coordinate question has
    # been settled, so the slicing wrapper sees the same interpolator a singleton
    # regime gets and the two cannot drift apart.
    #
    # A gated target's leaf carries a CHANNEL axis instead of a stakeholder one:
    # its operands are interpolated separately and the gate is applied to them at
    # the landing point, which already yields one value per leg. So the two
    # wrappers are alternatives, not layers.
    if gated_continuation is not None:
        mapped_interpolator = _get_pointwise_gated_interpolator(
            base_interpolator=next_V_interpolator,
            V_arr_name=V_arr_name,
            n_channels=gated_continuation.n_channels,
            combine=gated_continuation.combine,
            gate_state_names=gated_continuation.gate_state_names,
            projected_readers=gated_continuation.projected_readers,
            target_ages=gated_continuation.target_ages,
            co_map_state_names=co_map_state_names,
            draw_resolution=draw_resolution,
        )
    elif n_stakeholders is None:
        mapped_interpolator = next_V_interpolator
    else:
        mapped_interpolator = _get_stakeholder_sliced_interpolator(
            base_interpolator=next_V_interpolator,
            V_arr_name=V_arr_name,
            n_stakeholders=n_stakeholders,
        )

    return _TargetContinuation(
        next_states=get_next_state_function_for_solution(
            functions=functions,
            transitions=bundle,
            targets=[key for key in bundle if key not in dependencies_by_law],
        ),
        lottery_weights=lottery_weights,
        joint_lottery_weights=_get_joint_weights_function(
            regime_name=target_regime_name, variables=lottery_variables
        ),
        lottery_axis_names=lottery_variables,
        next_V=productmap(
            func=mapped_interpolator,
            variables=node_variables,
            batch_sizes=dict.fromkeys(node_variables, 0),
        ),
        # Read off the MAPPED interpolator: a gated target's gate carries free
        # parameters of its own, and naming them here is how they reach the
        # kernel.
        extra_param_names=frozenset(
            get_union_of_args([mapped_interpolator]) - set(bundle) - {V_arr_name}
        ),
        has_lottery_axes=bool(lottery_variables),
        draw_dependent_names=frozenset(dependencies_by_law),
    )


def _scalar_target_contribution(
    *,
    scalar_targets: tuple[RegimeName, ...],
    next_regime_to_V_arr: Mapping[RegimeName, FloatND],
    active_regime_probs: Mapping[RegimeName, FloatND],
    as_lottery: bool,
    zero: FloatND,
) -> tuple[
    list[tuple[RegimeName, FloatND, FloatND]],
    list[FloatND],
    list[FloatND],
    list[IntND],
    FloatND,
    BoolND,
]:
    """Seed the continuation accumulators with the stateless targets.

    A target carrying no state has a rank-zero value function: there is no next
    state to evaluate it at and no stochastic node to average over, so it
    contributes exactly one node whose only weight is the probability of going
    there. Under a certainty equivalent that node joins the joint lottery, so it
    is transformed together with every other target's nodes rather than on its
    own; under the linear expectation it is added straight to the running
    certainty equivalent.

    Args:
        scalar_targets: Targets active next period that carry no state.
        next_regime_to_V_arr: Mapping of target regime names to next period's
            value function arrays.
        active_regime_probs: Mapping of target regime names to their regime
            transition probabilities.
        as_lottery: Whether a nonlinear certainty equivalent aggregates the
            continuation, so the nodes must be handed over unaggregated.
        zero: Zero at the shape and dtype the mass accumulates in — the caller's
            cell shape, which the sign flags are shaped after as well.

    Returns:
        Tuple of the linear mixture terms, the lottery values, their weights,
        the base-two scale each weight arm carries, the probability mass these
        targets represent, and whether any of their probabilities is negative.

    """
    mixture_terms: list[tuple[RegimeName, FloatND, FloatND]] = []
    values: list[FloatND] = []
    weights: list[FloatND] = []
    shifts: list[IntND] = []
    probability_mass = zero
    has_negative_probability = jnp.zeros(jnp.shape(zero), dtype=bool)
    for target_regime_name in scalar_targets:
        scalar_V = next_regime_to_V_arr[target_regime_name]
        # The mass sum and the liveness test read the weight as the arithmetic
        # sees it, exactly as the stateful route does: a probability too small
        # for the dtype to use contributes nothing to either, which is the right
        # answer for both.
        prob = active_regime_probs[target_regime_name]
        # A stateless target contributes to the represented mass on either
        # route, so the linear fast path divides by the mass of *every* target
        # it summed, not just the ones carrying state.
        probability_mass = probability_mass + prob
        # Read off the bits while the probability still has them: a negative
        # weight the dtype cannot hold as a normal number arrives at an
        # arithmetic sign test as `-0` and passes it.
        has_negative_probability = has_negative_probability | is_negative(prob)
        if as_lottery:
            # A stateless target has no stochastic node to multiply against, so
            # its regime probability is already the whole weight, and it goes on
            # a scale for the same reason a joint product does: an arm of the
            # lottery that is subnormal cannot be classified once it is read
            # inside a fused region. Broadcasting rather than multiplying by
            # ones keeps that from happening on the way in. The arm keeps the
            # scale it reached rather than being flattened onto a shared one —
            # every other arm does too, and the joint aggregator reads the pair.
            node = jnp.ravel(scalar_V)
            values.append(node)
            weighted_nodes, shift = scaled_joint_weight(
                jnp.stack([jnp.broadcast_to(prob, jnp.shape(node))])
            )
            weights.append(weighted_nodes)
            shifts.append(shift)
        else:
            # Keep the pair unmultiplied, like every carry target.
            # `_sum_regime_mixture` forms its zero-safe contribution on the native
            # cell shape, masking the value on `prob == 0`, then includes it in
            # the common value-order network. No neutralization is owed here;
            # multiplying here would raise `0 * -inf = nan` for a zero-mass
            # stateless target and bypass the canonical reduction.
            mixture_terms.append((target_regime_name, prob, scalar_V))
    return (
        mixture_terms,
        values,
        weights,
        shifts,
        probability_mass,
        has_negative_probability,
    )


def _value_shaped_zero(*, zero: FloatND, n_stakeholders: int | None) -> FloatND:
    """Return the zero a continuation VALUE has, given the mass-shaped one.

    The regime-transition mass is stakeholder-independent, so it accumulates at
    the plain cell shape; a continuation value carries one trailing axis on top
    of that whenever the regime is collective.

    Args:
        zero: Zero at the shape and dtype the mass accumulates in.
        n_stakeholders: Length of the trailing stakeholder axis, or `None` for a
            singleton regime, whose value and mass share one shape.

    Returns:
        Zero at the shape and dtype of a continuation value.

    """
    if n_stakeholders is None:
        return zero
    zero_arr = jnp.asarray(zero)
    return jnp.zeros((*zero_arr.shape, n_stakeholders), dtype=zero_arr.dtype)


def _expected_continuation_over_nodes(
    *,
    values: FloatND,
    weights: FloatND,
    shifts: IntND,
    has_lottery_axes: bool,
    n_stakeholders: int | None,
) -> FloatND:
    """Reduce one target's continuation over its lottery nodes.

    Zero-safe throughout: a zero-probability node beside an admissible on-path
    `-inf` must not turn the average into a `nan`, which dissolution makes
    routine rather than exotic on a collective regime.

    Which reduction states that depends only on whether a trailing stakeholder
    axis has to survive it:

    - **collective.** The node axes are flattened into one leading axis and
      reduced away, leaving the stakeholder axis. This covers a target with no
      lottery at all as well: the empty product is the weight `1.0` at scale
      `0`, and the reduction is then bitwise identity on the values, so the
      no-lottery case needs no branch of its own.
    - **singleton, with a lottery.** The whole array is the node surface, so it
      is reduced at once (`axis=None`) — the spelling whose bits
      `zero_safe_average` is tested against.
    - **singleton, without one.** There is no node axis and no weight to apply.

    Args:
        values: Next period's value at this target's lottery nodes, the node
            axes leading and (collectively) the stakeholder axis trailing.
        weights: The nodes' scaled joint weights, one per node.
        shifts: Each node's own base-two scale.
        has_lottery_axes: Whether the target's transition draws any stochastic
            state, and so whether `values` carries node axes at all.
        n_stakeholders: Length of the trailing stakeholder axis, or `None` for a
            singleton regime.

    Returns:
        The weighted mean over the nodes, keeping any trailing stakeholder axis.

    """
    if n_stakeholders is not None:
        return zero_safe_average(
            a=jnp.asarray(values).reshape(-1, n_stakeholders),
            axis=0,
            weights=jnp.asarray(weights).reshape(-1),
            shifts=jnp.asarray(shifts).reshape(-1),
        )
    if has_lottery_axes:
        return zero_safe_average(a=values, weights=weights, shifts=shifts)
    return jnp.average(values)


def _as_lottery(
    *,
    values: FloatND,
    weights: FloatND,
    shifts: IntND,
    has_stochastic_states: bool,
) -> tuple[Float1D, Float1D, IntND]:
    """Flatten one target regime's continuation into a unit-mass lottery.

    The nodes stay scaled pairs. Dividing them by their mass as plain numbers
    is where a lottery loses the node a nonlinear continuation is decided by:
    the mass is the size of its likeliest node, so the ratio the division has
    to represent for the rarest one is the very quantity that does not fit, and
    the scale it still carries has a zero left to restore. The scale each node
    reaches the caller with already accounts for the target's own.

    A target whose joint weights carry no mass contributes no branch. It must
    not contribute NaN either: every target's nodes are concatenated into one
    lottery, so a NaN here would destroy the certainty equivalent of the
    well-specified targets alongside it.

    Args:
        values: Next period's value at this target's stochastic nodes.
        weights: Joint weights over those nodes; ignored when the target has
            no stochastic states.
        shifts: Each node's own base-two scale; ignored likewise.
        has_stochastic_states: Whether the target's transition draws stochastic
            states.

    Returns:
        Tuple of the flattened values, their probabilities' coefficients, and
        the scale each coefficient carries. The probabilities sum to one — or
        to zero for a target whose weights carry no mass at all.

    """
    flat_values = jnp.ravel(values)
    if has_stochastic_states:
        coefficients, node_shifts = normalized_scaled_weights(
            coefficients=jnp.ravel(weights), shifts=jnp.ravel(jnp.asarray(shifts))
        )
        return flat_values, coefficients, node_shifts
    uniform = jnp.full(
        flat_values.shape, 1.0 / flat_values.size, dtype=flat_values.dtype
    )
    return flat_values, uniform, jnp.zeros(flat_values.shape, dtype=jnp.int32)


def _get_arg_names_of_Q_and_F(
    *,
    deps: list[Callable[..., Any]],
    include: frozenset[str] = frozenset(),
    exclude: frozenset[str] = frozenset(),
) -> tuple[str, ...]:
    """Get the argument names of the dependencies.

    Args:
        deps: List of dependencies.
        include: Set of argument names to include.
        exclude: Set of argument names to exclude.

    Returns:
        The union of the argument names in deps and include, except for those in
        exclude.

    """
    return tuple((get_union_of_args(deps) | include) - exclude)


def _get_joint_weights_function(
    *,
    regime_name: RegimeName,
    variables: tuple[TransitionFunctionName, ...],
) -> Callable[..., tuple[FloatND, IntND]]:
    """Get function that calculates the joint weights over one group of laws.

    This function takes the weights of the individual variables and multiplies
    them together to get the joint weights on their product space.

    The group is passed in as an ordered tuple rather than re-derived from the
    transition laws, because the caller productmaps the value function over the
    same tuple: one ordering fixes both the axes of the value surface and the
    axes of the weights, so the two cannot drift apart.

    Args:
        regime_name: Name of the target regime.
        variables: Ordered unqualified `next_<state>` names whose weights to
            multiply, in the order their axes appear on the value surface.

    Returns:
        A function that computes the outer product of the variables' weights,
        each node with the base-two scale its own product needed.

    """
    arg_names = [f"weight_{regime_name}__{key}" for key in variables]

    @with_signature(args=arg_names)
    def _outer(**kwargs: Float1D) -> tuple[FloatND, IntND]:
        # One factor per stochastic axis. Their product is the node's
        # probability, and it comes back with its own scale rather than as a
        # plain float, because a product below the normal range is not
        # something a float can carry through a fused region here.
        return scaled_joint_weight(jnp.array(list(kwargs.values())))

    variables = tuple(arg_names)
    return productmap(
        func=_outer, variables=variables, batch_sizes=dict.fromkeys(variables, 0)
    )


def _get_U_and_F(
    *,
    functions: EconFunctionsMapping,
    constraints: ConstraintFunctionsMapping,
    utility_names: tuple[str, ...] = ("utility",),
) -> Callable[..., tuple[Any, ...]]:
    """Get the instantaneous utilities and the one feasibility function.

    Note:
    -----
    U may depend on all kinds of other functions (taxes, transfers, ...), which will be
    executed if they matter for the value of U.

    Feasibility carries no stakeholder input of its own: `_get_feasibility` builds
    the mask from the regime's constraints and the shared function pool, so a
    household has ONE action set however many felicities it carries. A constraint
    may still name a stakeholder-indexed node (`utility_f`) — that names the same
    node for every stakeholder, which is why the mask is a single DAG target here
    rather than one per felicity.

    Args:
        functions: Immutable mapping of function names to internal user functions.
        constraints: Immutable mapping of constraint names to internal user functions.
        utility_names: DAG target names of the felicity functions, in the order
            their values are returned. `("utility",)` (the default) is the
            singleton case; a collective regime passes every stakeholder's
            `"utility_<s>"`, so all felicities and the shared feasibility come out
            of one DAG evaluation with every node they have in common computed once.

    Returns:
        A function returning one value per entry of `utility_names`, in that
        order, followed by the feasibility mask.

    """
    return concatenate_functions(
        functions={
            "feasibility": _get_feasibility(
                functions=functions, constraints=constraints
            ),
            **dict(functions),
        },
        targets=[*utility_names, "feasibility"],
        enforce_signature=False,
        set_annotations=True,
    )


def _get_feasibility(
    *,
    functions: EconFunctionsMapping,
    constraints: ConstraintFunctionsMapping,
) -> ConstraintFunction:
    """Create a function that combines all constraint functions into a single one.

    Args:
        functions: Immutable mapping of function names to internal user functions.
        constraints: Immutable mapping of constraint names to internal user functions.

    Returns:
        The combined constraint function (feasibility).

    """
    if constraints:
        combined_constraint = concatenate_functions(
            functions=dict(constraints) | dict(functions),
            targets=list(constraints),
            aggregator=jnp.logical_and,
            aggregator_return_type="Feasibility",
            set_annotations=True,
        )

    else:

        def combined_constraint() -> bool:
            """Dummy feasibility function that always returns True."""
            return True

    return cast("ConstraintFunction", combined_constraint)


# Gross departures from unit regime mass are a specification error, not rounding:
# every aggregation route divides the continuation by the mass it received, so the
# lost mass is divided straight back out and the survivors renormalize. The solved
# value function then comes back finite, plausible, and independent of what went
# missing. `validate_transitions` catches it, but `log_level="off"` skips that, so
# this poisons the arithmetic itself and cannot be gated away.
#
# The tolerance is deliberately loose. It is a backstop against a wrong model, not
# a numerical check: `1e-3` never fires on accumulated float error over a handful
# of targets, while a mass of 0.977 — small enough to look plausible, large enough
# to move every value in the model — becomes NaN.
_MAX_REGIME_MASS_DEVIATION = 1.0e-3


def _regime_mass_is_a_distribution(
    *, probability_mass: FloatND, has_negative_probability: BoolND
) -> BoolND:
    """Whether the retained targets carry a distribution rather than merely unit mass.

    Two conditions, both holding at every log level because they are computed
    rather than validated:

    - the represented mass is one, within tolerance;
    - no target carries a negative weight.

    Together they give the full range: non-negative weights summing to one each
    lie in `[0, 1]`. Unit mass alone does not, since 1.5 and -0.5 sum to one,
    and a NaN weight fails both tests rather than passing the first by accident.

    Non-negativity arrives as a decision rather than as a weight to inspect.
    Taken here it would have to be taken on a number the accumulation already
    reduced, and reducing with `jnp.minimum` turns a negative probability the
    dtype cannot hold as a normal number into `-0` — which is a zero, and would
    pass. The caller reads each target's sign off its own bits instead.

    Args:
        probability_mass: The retained targets' probabilities, summed.
        has_negative_probability: Whether any of them carried the sign bit on a
            nonzero magnitude.

    Returns:
        Whether the retained targets carry a probability distribution.

    """
    is_unit = jnp.abs(probability_mass - 1.0) <= _MAX_REGIME_MASS_DEVIATION
    return is_unit & ~has_negative_probability


def _aggregate_joint_lottery(
    *,
    certainty_equivalent: CertaintyEquivalent,
    lottery_values: Sequence[FloatND],
    lottery_weights: Sequence[FloatND],
    lottery_shifts: Sequence[IntND],
    ce_flat_param_names: Mapping[str, str],
    states_actions_params: Mapping[str, Any],
) -> FloatND:
    """Aggregate the continuation nodes of every retained target in one piece.

    Args:
        certainty_equivalent: The regime's certainty equivalent.
        lottery_values: Sequence of per-target continuation values.
        lottery_weights: Sequence of per-target node weights, already scaled by
            the target's regime-transition probability.
        lottery_shifts: Each target's own base-two scale, so the arms can be
            read against one another.
        ce_flat_param_names: Mapping of certainty-equivalent argument names to
            their flat parameter names.
        states_actions_params: Mapping of states, actions, age, period, and flat
            regime params.

    Returns:
        The aggregated continuation value.

    """
    values = jnp.concatenate(list(lottery_values))
    # Every node arrives as a coefficient and a scale that together state its
    # probability exactly, and the arms are read as one lottery by laying them
    # end to end in both. Bringing them onto a shared scale first is what the
    # pair exists to avoid: the shared scale can only be the one the widest
    # spread fits into, and no scale fits a lottery whose rarest node is
    # further below its likeliest than the format is wide.
    coefficients = jnp.concatenate(list(lottery_weights))
    shifts = jnp.concatenate(
        [
            jnp.broadcast_to(jnp.asarray(s), jnp.asarray(w).shape).astype(jnp.int32)
            for w, s in zip(lottery_weights, lottery_shifts, strict=True)
        ]
    )
    return certainty_equivalent.aggregate_scaled(
        values=_values_without_impossible_nodes(values=values, weights=coefficients),
        coefficients=coefficients,
        shifts=shifts,
        # The params template types every certainty-equivalent parameter as a
        # float, so its runtime values are float arrays.
        params=cast(
            "Mapping[str, FloatND]",
            {
                arg: states_actions_params[flat_name]
                for arg, flat_name in ce_flat_param_names.items()
            },
        ),
    )


def _values_without_impossible_nodes(*, values: FloatND, weights: FloatND) -> FloatND:
    """Replace the value at every zero-probability node with a live node's.

    `aggregate` is a public interface, and its implementations are entitled to
    multiply rather than mask: the ordinary weighted mean is written
    `sum(w * v) / sum(w)`. A node carrying no probability may name anything at
    all -- an entry law evaluated off the target's support returns NaN there --
    and `0 * nan` is `nan`, which would destroy every well-specified node
    beside it. Neutralizing such a node here makes that guarantee the engine's
    rather than something each certainty equivalent has to rediscover.

    The stand-in is copied from the node with the largest coefficient rather
    than being a constant, because `aggregate` may transform the values before
    averaging and an arbitrary constant need not lie in the transform's domain
    -- `log` at zero is the ordinary case. A value already in the lottery
    always does. Which live node donates it does not matter, only that one
    does, so the coefficient is read without its scale.

    Only a weight that is zero *in its bits* is replaced. `weights == 0` also
    catches every probability below the dtype's normal range, and replacing one
    of those is not a neutralization but a loss: the node can occur, and a `-inf`
    standing at a state where no action is feasible would be overwritten by a
    neighbour's finite value, turning an infinite continuation into an ordinary
    number. A negative or NaN weight is not a node that cannot occur either, and
    both stay visible.

    Args:
        values: Continuation values of the joint lottery.
        weights: Their weights, over the same axis.

    Returns:
        The values, with every zero-weight entry replaced by a live one.

    """
    stand_in = jnp.take(values, jnp.argmax(weights, axis=-1), axis=-1)
    return jnp.where(is_represented_zero(weights), stand_in, values)


def _unit_regime_mass_or_nan(
    *, probability_mass: FloatND, has_negative_probability: BoolND
) -> FloatND:
    """Return the mass itself, or NaN where the weights are not a distribution.

    For the per-target route, which divides by the mass it accumulated.
    """
    return jnp.where(
        _regime_mass_is_a_distribution(
            probability_mass=probability_mass,
            has_negative_probability=has_negative_probability,
        ),
        probability_mass,
        jnp.nan,
    )
