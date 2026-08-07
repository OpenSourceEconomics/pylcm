import dataclasses
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, cast

import jax.numpy as jnp
from dags import (
    concatenate_functions,
    get_annotations,
    with_signature,
)

from _lcm.certainty_equivalent import CertaintyEquivalent, LinearExpectation
from _lcm.processes import _ContinuousStochasticProcess
from _lcm.regime_building.next_state import (
    get_next_state_function_for_solution,
    get_next_stochastic_weights_function,
)
from _lcm.regime_building.V import VInterpolationInfo, get_V_interpolator
from _lcm.regime_building.w_dag import _get_build_W_kwargs
from _lcm.regime_building.zero_safe import zero_safe_average, zero_safe_weighted_term
from _lcm.transition_laws import (
    TransitionLaws,
    is_interpolation_basis,
    is_stochastic,
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
from lcm.exceptions import ModelInitializationError
from lcm.typing import (
    BoolND,
    Float1D,
    FloatND,
    IntND,
)


def _sum_regime_mixture(
    mixture_terms: list[tuple[RegimeName, FloatND, FloatND]], *, like: FloatND
) -> FloatND:
    """Reduce the regime mixture ``E[V']=Σ p_r·V_r`` as ONE zero-safe contraction.

    ``mixture_terms`` is a list of ``(target_name, prob_r, expected_V_r)`` — the
    UNMULTIPLIED per-target probability and expected continuation. The per-target
    probabilities and continuations are stacked along a new leading (target) axis and
    multiplied ONCE inside a single ``zero_safe_weighted_term``; the resulting
    per-target contributions ``p_r·V_r`` are then reduced by a VALUE-ORDERED
    ``jnp.sum`` — the contributions are ``jnp.sort``-ed along the target axis
    before the sum. Two
    properties this buys over the earlier sequential left-fold
    ``E = 0; for r: E += zero_safe_weighted_term(p_r, V_r)`` (round-8/round-10 external
    re-review, both MEASURED reproduce-first):

    - **Accuracy.** Stacking the OPERANDS and multiplying once inside the reduction —
      NOT stacking the already-formed products — lands on the exact-policy side of the
      round-8 pinned 5-target fixture (``> alternative`` bits ...843) where the
      left-fold and ``jnp.sum(jnp.stack(products))`` both land on the wrong side
      (bits ...842). It
      is still NOT correctly-rounded: under cancellation (Σ|p_r·V_r| ≫ |Σ p_r·V_r|) the
      error scales with Σ|p_r·V_r|, hundreds of result-ULP, so a genuine knife-edge
      argmax can still resolve either way. Deterministic resolution AT a genuine
      knife-edge would need compensated/exact summation, which is not implemented (a
      value-sorted Neumaier compensated sum WAS measured and, on the round-10
      counterexample, landed on the WRONG side of the competing action while the plain
      value-sorted reduction landed exact-side, so it was NOT adopted).
    - **Reproducibility (label-independence).** The reduction ORDER is a deterministic
      function of the contribution VALUES — economically meaningful — and NEVER of the
      arbitrary regime NAMES. The pre-round-10 code ``sorted(mixture_terms, key=name)``
      removed the transition-mapping ITERATION-ORDER dependence but made the float64
      summation order a function of the user's regime LABELS: a pure ALPHA-RENAMING of
      the regimes (same probabilities, same continuations, only the dict keys change)
      reordered the non-associative float64 sum and, MEASURED, moved the result across
      37 distinct outputs over the 120 name bijections of a valid 5-target float64
      mixture — reversing a non-tied household argmax on the round-10 counterexample.
      Sorting the CONTRIBUTION MULTISET (``jnp.sort`` along the target axis) makes the
      sum provably invariant to alpha-renaming: the multiset ``{p_r·V_r}`` is unchanged
      by relabeling, and the sorted order (hence the summation order and its bits) is a
      function of that multiset alone. The stacking order of ``mixture_terms`` is
      therefore irrelevant (the sort canonicalises it), so no name-sort is needed.

    Zero-mass safety is preserved (a zero ``p_r`` beside an admissible ``±inf`` V_r is
    masked to exactly 0 by ``zero_safe_weighted_term`` BEFORE the sort, so a zero-mass
    ``-inf`` contributes 0 and never survives the sort as ``-inf``). Cost: the K
    per-target contributions are materialised together and sorted along the (small)
    target axis, an O(K log K) sort on a tiny axis, rather than folded one at a time — K
    is the number of active next-period targets. ``mixture_terms`` is empty in a
    terminal period with no active target; the mixture is then exactly ``zeros_like``.
    """
    if not mixture_terms:
        return jnp.zeros_like(like)
    prob_list = [prob for _, prob, _ in mixture_terms]
    value_list = [value for _, _, value in mixture_terms]
    # Targets do not all arrive at the same rank: a CARRY target's expected
    # continuation carries the cell axes (and, collectively, a trailing stakeholder
    # axis), while a STATELESS target's V is rank-zero -- there is no next state to
    # evaluate it at. `jnp.stack` needs one shape, so lift the rank-deficient ones
    # to the common value shape first. `broadcast_to` replicates, it does not
    # resample, so this changes no number; it only makes the target axis stackable.
    #
    # The `len < len` guard covers the all-stateless case: with no carry target the
    # common value shape is `()` while the probabilities still carry the cell axes,
    # and `zero_safe_weighted_term` right-aligns -- it would weight a cell axis by
    # the target axis. Lifting values to the probability shape keeps axis 0 the
    # target axis, which is what the right-padding below then assumes.
    value_shape = jnp.broadcast_shapes(*(v.shape for v in value_list))
    prob_shape = jnp.broadcast_shapes(*(p.shape for p in prob_list))
    if len(value_shape) < len(prob_shape):
        value_shape = jnp.broadcast_shapes(value_shape, prob_shape)
    probs = jnp.stack(prob_list, axis=0)
    values = jnp.stack([jnp.broadcast_to(v, value_shape) for v in value_list], axis=0)
    # Right-pad the probability rank to the value rank so the per-target weight
    # broadcasts over the TARGET axis (leading, axis 0) and is constant across any
    # trailing value-only axes. The collective site carries a trailing stakeholder
    # axis on the continuation (`values` is (K, *cell, S)) that the scalar regime
    # probability (K, *cell) does not: without this alignment `zero_safe_weighted_
    # term` right-aligns and weights the STAKEHOLDER axis instead of the target axis
    # -- silently reversing a household action when K==S, leaking a zero-mass -inf,
    # or raising when K!=S. A no-op at the scalar/singleton sites (equal ranks).
    if probs.ndim < values.ndim:
        probs = probs.reshape(probs.shape + (1,) * (values.ndim - probs.ndim))
    # Reduce in VALUE order, not label order. `zero_safe_weighted_term` forms the
    # zero-mass-safe per-target contributions `p_r*V_r` (masking a zero-mass `+-inf`
    # to 0); sorting them along the target axis (axis 0) before `jnp.sum` makes the
    # non-associative float64 reduction order a deterministic function of the
    # contribution multiset -- provably invariant to an economically-inert
    # alpha-renaming of the regimes -- where the previous name-sort made the bits
    # (and a non-tied argmax) depend on the arbitrary regime labels. See the docstring.
    contributions = zero_safe_weighted_term(probs, values)
    return jnp.sum(jnp.sort(contributions, axis=0), axis=0)


def get_Q_and_F(
    *,
    flat_param_names: frozenset[str],
    functions: EconFunctionsMapping,
    constraints: ConstraintFunctionsMapping,
    period_targets: tuple[RegimeName, ...],
    scalar_targets: tuple[RegimeName, ...] = (),
    transitions: TransitionFunctionsMapping,
    transition_laws: TransitionLaws,
    compute_regime_transition_probs: RegimeTransitionFunction,
    regime_to_v_interpolation_info: MappingProxyType[RegimeName, VInterpolationInfo],
    koopmans_aggregator: EconFunction,
    certainty_equivalent: CertaintyEquivalent | None,
    co_map_state_names: tuple[StateName, ...] = (),
    continuation_functions: EconFunctionsMapping | None = None,
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
        transition_laws: Immutable mapping of target regime names to their
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
        transition_laws=transition_laws,
        compute_regime_transition_probs=compute_regime_transition_probs,
        regime_to_v_interpolation_info=regime_to_v_interpolation_info,
        certainty_equivalent=certainty_equivalent,
        co_map_state_names=co_map_state_names,
    )
    _build_W_kwargs = _get_build_W_kwargs(functions, koopmans_aggregator)

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
        # COLLECTIVE-REGIMES (E2): F_arr is built here, before and independently
        # of Q (it never reads E_next_V). A value-aware mask cannot stay here:
        # it needs per-stakeholder Q^s, so E2 splits this into (i) build the
        # state-independent F here, (ii) compute Q^s, (iii) `mask = F ∧ g(...)`
        # applied in max_Q_over_a. This site also returns the explicit dissolution
        # flag D = 1[mask empty], distinct from a numeric -inf. See design doc
        # §2 (E2) / §3.
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
    transition_laws: TransitionLaws,
    compute_regime_transition_probs: RegimeTransitionFunction,
    regime_to_v_interpolation_info: MappingProxyType[RegimeName, VInterpolationInfo],
    koopmans_aggregator: EconFunction,
    certainty_equivalent: CertaintyEquivalent | None,
    co_map_state_names: tuple[StateName, ...],
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
        transition_laws: Immutable mapping of target regime names to their
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

    Returns:
        Closure returning `(U_arr, F_arr, CE, Q_arr, active_regime_probs)`.

    """
    U_and_F = _get_U_and_F(functions=functions, constraints=constraints)
    compute_CE, continuation_deps, continuation_arg_names = _get_compute_CE(
        functions=functions,
        period_targets=period_targets,
        scalar_targets=scalar_targets,
        transitions=transitions,
        transition_laws=transition_laws,
        compute_regime_transition_probs=compute_regime_transition_probs,
        regime_to_v_interpolation_info=regime_to_v_interpolation_info,
        certainty_equivalent=certainty_equivalent,
        co_map_state_names=co_map_state_names,
    )
    _build_W_kwargs = _get_build_W_kwargs(functions, koopmans_aggregator)

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
) -> QAndFFunction:
    """Terminal (Q, F) for a collective regime — stacked per-stakeholder U + shared F.

    COLLECTIVE-REGIMES (E1). Separate from `get_Q_and_F_terminal` so the singleton
    terminal path (shared with the simulate / compute-intermediates machinery) is
    byte-identical; this builder is used only at the collective solve site.

    Builds one `U^s`-and-`F` closure per stakeholder from its own `utility_<s>`
    DAG target (feasibility is regime-level, so it is identical across
    stakeholders — the first one is kept). The returned `Q_and_F` stacks the
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

    Returns:
        A function computing the stacked per-stakeholder utilities (Q) and the
        shared feasibility mask (F) for a terminal collective period.

    """
    U_and_F_by_stakeholder = {
        stakeholder: _get_U_and_F(
            functions=functions,
            constraints=constraints,
            utility_name=f"utility_{stakeholder}",
        )
        for stakeholder in stakeholders
    }

    arg_names_of_Q_and_F = _get_arg_names_of_Q_and_F(
        deps=list(U_and_F_by_stakeholder.values()),
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
        U_arrays: list[FloatND] = []
        F_arr: BoolND | None = None
        for u_and_f in U_and_F_by_stakeholder.values():
            U_s, F_arr = u_and_f(**states_actions_params)
            U_arrays.append(jnp.asarray(U_s))
        U_stack = jnp.stack(U_arrays, axis=-1)
        return U_stack, jnp.asarray(F_arr)

    return Q_and_F


# COLLECTIVE-REGIMES (E2): the name under which the mapping of same-period
# reference regimes to their current-period V arrays enters the kernel
# signature. Only regimes declaring `same_period_refs` carry it.
SAME_PERIOD_V_ARG = "same_period_regime_to_V_arr"

# COLLECTIVE-REGIMES (E2, F4 fix): the name under which the mapping of
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
# Before this argument existed the reader exposed those helpers as extra outer
# arguments named after the PREFIXED coordinate variable
# (`__same_period_ref__x__points`), which no caller supplies and no params
# template ever emits (`_lcm.params.regime_template._add_runtime_grid_params`
# emits `x__points`, in the reference regime's own template): all four consumers
# of `_build_same_period_ref_reader` — ordinary E2 same-period refs, solve-side
# gate refs, solve-side leg-fallback value readers, and simulate-side gate refs
# — raised a missing-argument error the moment a reference regime declared a
# runtime irregular grid. Coordinate VARIABLES stay prefixed (internal wiring
# that must not collide with the reading regime's own state names); PARAMETER
# qnames are separated from them and resolved against the reference regime's
# explicit namespace through this mapping instead.
SAME_PERIOD_PARAMS_ARG = "same_period_regime_to_params"

# Internal argument names of the same-period reference interpolation; never
# surfaced in the kernel signature.
_REF_STATE_PREFIX = "__same_period_ref__"
_REF_V_ARR_NAME = "__same_period_ref_V_arr__"


@dataclass(frozen=True, kw_only=True)
class ResolvedSamePeriodRef:
    """Engine-side form of a user `SamePeriodRef`, resolved at model processing.

    COLLECTIVE-REGIMES (E2). The user declaration names a stakeholder; the
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


def _build_same_period_ref_reader(
    *,
    ref: ResolvedSamePeriodRef,
    v_interpolation_info: VInterpolationInfo,
    functions: EconFunctionsMapping,
    deterministic_transitions: Mapping[TransitionFunctionName, TransitionFunction] = (
        MappingProxyType({})
    ),
) -> Callable[..., FloatND]:
    """Build the reader of one same-period reference value at a (state, action) cell.

    COLLECTIVE-REGIMES (E2). Each projection entry is concatenated with the
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
    (F4 fix — see that constant). The two provenances are separated here rather
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
    process state is unaffected (`interpolate_process_axes=False`, the
    ordinary path, byte-identical).

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
    # Empty for an E2 value constraint: that projection is evaluated at THIS
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
            functions={**dag_pool, target: ref.projection[state_name]},
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
    # runtime-supplied irregular-grid points). F4 fix: these are the REFERENCE
    # regime's own parameters, so they are NOT exposed as outer arguments of
    # this reader (the reading regime's caller has no such param, and the
    # prefixed name they carried was unsatisfiable by anyone) — they are looked
    # up per call in `SAME_PERIOD_PARAMS_ARG[ref.regime]` under their qname in
    # the reference regime's OWN namespace.
    interpolator_extra_qnames = _reference_interpolator_param_qnames(
        extra_args=get_union_of_args([interpolator])
        - coordinate_names
        - {_REF_V_ARR_NAME},
        ref=ref,
    )
    arg_names = sorted(
        {arg for args in projection_args.values() for arg in args}
        | {SAME_PERIOD_V_ARG, SAME_PERIOD_PARAMS_ARG}
    )

    @with_signature(args=arg_names, return_annotation="FloatND")
    def read_reference_value(**kwargs: _ParamsLeaf) -> FloatND:
        same_period_V = cast("Mapping[RegimeName, FloatND]", kwargs[SAME_PERIOD_V_ARG])
        V_ref = same_period_V[ref.regime]
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
                regime_to_params=kwargs[SAME_PERIOD_PARAMS_ARG],
                ref_regime=ref.regime,
            ),
            **{_REF_V_ARR_NAME: V_ref},
        )

    return read_reference_value


def _reference_interpolator_param_qnames(
    *,
    extra_args: set[str],
    ref: ResolvedSamePeriodRef,
) -> MappingProxyType[str, str]:
    """Map each extra interpolator input to its qname in the REFERENCE namespace.

    COLLECTIVE-REGIMES (E2, F4 fix). `get_V_interpolator` derives its runtime
    grid-helper names from the COORDINATE VARIABLE it was given
    (`_get_coordinate_finder`: `qname_from_tree_path((in_name.removeprefix(
    "next_"), "points"))`), so with `state_prefix=_REF_STATE_PREFIX` the helper
    for reference state `x` is called `__same_period_ref__x__points` while the
    reference regime's params template calls the very same quantity `x__points`.
    Stripping the coordinate prefix is exactly the inverse of the prefixing
    `get_V_interpolator` applied, and recovers the reference regime's own qname.

    Any extra input that does NOT carry the prefix cannot be attributed to a
    reference state this way; rather than bind it from an arbitrary namespace
    (the defect class this whole mechanism exists to end), fail loudly at build
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

    COLLECTIVE-REGIMES (E2, F4 fix). See `SAME_PERIOD_PARAMS_ARG`.

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
    transitions: TransitionFunctionsMapping,
    transition_laws: TransitionLaws,
    compute_regime_transition_probs: RegimeTransitionFunction,
    regime_to_v_interpolation_info: MappingProxyType[RegimeName, VInterpolationInfo],
    koopmans_aggregator: EconFunction,
    stakeholders: tuple[str, ...],
    co_map_state_names: tuple[StateName, ...] = (),
    value_constraints: ConstraintFunctionsMapping = MappingProxyType({}),
    same_period_refs: Mapping[str, ResolvedSamePeriodRef] = MappingProxyType({}),
    continuation_functions: EconFunctionsMapping | None = None,
) -> QAndFFunction:
    """Non-terminal (Q, F) for a collective regime — per-stakeholder continuation.

    COLLECTIVE-REGIMES (E1, slice 2). Separate from `get_Q_and_F` so the
    singleton path is byte-identical; this builder is used only at the
    collective solve site.

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
        period_targets: Target regimes whose continuation enters E[V^s] this
            period (all collective with the identical stakeholder tuple).
        transitions: Immutable mapping of transition names to transition
            functions.
        stochastic_transition_names: Frozenset of stochastic transition function
            names.
        compute_regime_transition_probs: Regime transition probability function
            for solve (stakeholder-independent — per-stakeholder gates are E3').
        regime_to_v_interpolation_info: Mapping of regime names to
            V-interpolation info (state axes only; the stakeholder axis is not
            an interpolation axis).
        stakeholders: Ordered stakeholder names; fixes the trailing-axis order.
        co_map_state_names: Tuple of state names co-mapped with the continuation
            V (see `get_Q_and_F`).
        value_constraints: Immutable mapping of value-constraint names to
            predicates (params already renamed to qnames). COLLECTIVE-REGIMES
            (E2): evaluated AFTER the per-stakeholder `Q^s`, each predicate may
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

    Returns:
        A function computing the stacked per-stakeholder state-action values
        (trailing stakeholder axis) and the shared feasibility mask for a
        non-terminal collective period.

    """
    # Phase split, mirroring get_Q_and_F: in the solve phase the two roles
    # coincide (`None`), so this is byte-identical to the prior single-pool
    # build; only the simulate phase passes them apart. The continuation prices
    # the target V under the perceived law, pairing `transitions` with
    # `continuation_pool`. Dropping that — as the collective branch did — yields
    # a sub-DAG that is neither phase and can reverse the household argmax.
    #
    # The flow needs no pool of its own: `next_<state>` is reserved for a
    # transition's output, so no per-stakeholder utility, feasibility or E2 value
    # constraint reads one and the flow holds no transition node to resolve.
    continuation_pool = (
        functions if continuation_functions is None else continuation_functions
    )
    U_and_F_by_stakeholder = {
        stakeholder: _get_U_and_F(
            functions=functions,
            constraints=constraints,
            utility_name=f"utility_{stakeholder}",
        )
        for stakeholder in stakeholders
    }
    n_stakeholders = len(stakeholders)

    state_transitions = {}
    next_stochastic_states_weights = {}
    joint_weights_from_marginals = {}
    next_V = {}

    next_V_extra_param_names: dict[RegimeName, frozenset[str]] = {}

    for target_regime_name in period_targets:
        bundle = transitions[target_regime_name]
        # Continuation helpers read `continuation_pool` (the perceived / solve pool),
        # NOT `functions`: the continuation is priced under the agent's perceived law,
        # helpers included — mirroring get_Q_and_F.
        state_transitions[target_regime_name] = get_next_state_function_for_solution(
            functions=continuation_pool,
            transitions=bundle,
        )
        next_stochastic_states_weights[target_regime_name] = (
            get_next_stochastic_weights_function(
                functions=continuation_pool,
                transitions=bundle,
                transition_laws=transition_laws,
                regime_name=target_regime_name,
            )
        )
        # `_get_joint_weights_function` now takes the ORDERED tuple of lottery
        # variables rather than re-deriving it from the laws, so that the axes of
        # the weights and the axes the value surface is productmapped over are
        # fixed by one and the same ordering. The collective builder derives it
        # exactly as `_build_target_continuation` does on the singleton path.
        lottery_variables = tuple(
            key
            for key in bundle
            if is_stochastic(transition_laws, target_regime_name, key)
        )
        joint_weights_from_marginals[target_regime_name] = _get_joint_weights_function(
            regime_name=target_regime_name,
            variables=lottery_variables,
        )
        V_arr_name = "next_V_arr"
        next_V_interpolator = get_V_interpolator(
            v_interpolation_info=regime_to_v_interpolation_info[target_regime_name],
            state_prefix="next_",
            V_arr_name=V_arr_name,
            co_map_state_names=co_map_state_names,
        )
        next_V_extra_param_names[target_regime_name] = frozenset(
            get_union_of_args([next_V_interpolator]) - set(bundle) - {V_arr_name}
        )
        stochastic_variables = tuple(
            key
            for key in bundle
            if is_stochastic(transition_laws, target_regime_name, key)
        )
        next_V[target_regime_name] = productmap(
            func=_get_stakeholder_sliced_interpolator(
                base_interpolator=next_V_interpolator,
                V_arr_name=V_arr_name,
                n_stakeholders=n_stakeholders,
            ),
            variables=stochastic_variables,
            batch_sizes=dict.fromkeys(stochastic_variables, 0),
        )

    _build_W_kwargs = _get_build_W_kwargs(functions, koopmans_aggregator)
    _co_map_next_names = frozenset(f"next_{name}" for name in co_map_state_names)

    # COLLECTIVE-REGIMES (E2): build the same-period reference readers and the
    # value-constraint evaluators once; their engine-supplied arguments —
    # `Q_<s>` and the reference-value names — are excluded from the kernel
    # signature and bound per (state, action) cell inside `Q_and_F`.
    value_constraint_machinery = _build_value_constraint_machinery(
        value_constraints=value_constraints,
        same_period_refs=same_period_refs,
        stakeholders=stakeholders,
        regime_to_v_interpolation_info=regime_to_v_interpolation_info,
        functions=functions,
    )

    arg_names_of_Q_and_F = _get_arg_names_of_Q_and_F(
        deps=[
            *list(U_and_F_by_stakeholder.values()),
            compute_regime_transition_probs,
            *list(state_transitions.values()),
            *list(next_stochastic_states_weights.values()),
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
        regime_transition_probs: MappingProxyType[RegimeName, FloatND] = (
            compute_regime_transition_probs(**states_actions_params)
        )
        U_arrays: list[FloatND] = []
        F_arr: BoolND | None = None
        for u_and_f in U_and_F_by_stakeholder.values():
            U_s, F_arr = u_and_f(**states_actions_params)
            U_arrays.append(jnp.asarray(U_s))
        U_stack = jnp.stack(U_arrays, axis=-1)
        active_regime_probs = MappingProxyType(
            {r: regime_transition_probs[r] for r in period_targets}
        )

        mixture_terms: list[tuple[RegimeName, FloatND, FloatND]] = []
        for target_regime_name in period_targets:
            next_states = state_transitions[target_regime_name](
                **states_actions_params,
            )
            marginal_next_stochastic_states_weights = next_stochastic_states_weights[
                target_regime_name
            ](**states_actions_params)
            joint_next_stochastic_states_weights = joint_weights_from_marginals[
                target_regime_name
            ](**marginal_next_stochastic_states_weights)

            extra_kw = {
                k: states_actions_params[k]
                for k in next_V_extra_param_names[target_regime_name]
            }
            # Shape (*stochastic_axes, n_stakeholders): the product-map stacks
            # the stochastic-node axes at the front, the stakeholder axis stays
            # trailing.
            next_V_at_stochastic_states_arr = next_V[target_regime_name](
                **{
                    name: val
                    for name, val in next_states.items()
                    if name not in _co_map_next_names
                },
                next_V_arr=next_regime_to_V_arr[target_regime_name],
                **extra_kw,
            )

            # Per-stakeholder weighted average over the stochastic nodes only —
            # never over the trailing stakeholder axis. Zero-safe: see the
            # guards in `get_Q_and_F` above.
            next_V_expected_arr = zero_safe_average(
                next_V_at_stochastic_states_arr.reshape(-1, n_stakeholders),
                axis=0,
                weights=jnp.asarray(joint_next_stochastic_states_weights).reshape(-1),
            )
            mixture_terms.append(
                (
                    target_regime_name,
                    active_regime_probs[target_regime_name],
                    next_V_expected_arr,
                )
            )
        CE = _sum_regime_mixture(mixture_terms, like=U_stack)

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

        # COLLECTIVE-REGIMES (E2): value-aware feasibility. Evaluated AFTER
        # Q^s — this is the reorder the singleton path never needs (there,
        # F is built before and independently of Q). Interpolate each declared
        # same-period reference value at the projected coordinates, then AND
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
    """Prebuilt E2 evaluation machinery closed over by a collective `Q_and_F`."""

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
    same_period_refs: Mapping[str, ResolvedSamePeriodRef],
    stakeholders: tuple[str, ...],
    regime_to_v_interpolation_info: MappingProxyType[RegimeName, VInterpolationInfo],
    functions: EconFunctionsMapping,
) -> _ValueConstraintMachinery:
    """Build the E2 reference readers and value-constraint evaluators once.

    COLLECTIVE-REGIMES (E2). Each evaluator is the predicate concatenated with
    the regime's function DAG (so it may read helper functions, exactly like
    ordinary constraints); its engine-supplied arguments — `Q_<s>` and the
    reference-value names — are bound per (state, action) cell by
    `_apply_value_constraints`.

    A value constraint is evaluated at this period's states and actions, so no
    transition enters its pool: `next_<state>` is reserved for a transition's
    output and rejected outside one.
    """
    reference_readers: dict[str, Callable[..., FloatND]] = {}
    reference_reader_args: dict[str, tuple[str, ...]] = {}
    for ref_name, ref in same_period_refs.items():
        reader = _build_same_period_ref_reader(
            ref=ref,
            v_interpolation_info=regime_to_v_interpolation_info[ref.regime],
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

    COLLECTIVE-REGIMES (E2). Reads each declared same-period reference value at
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

    COLLECTIVE-REGIMES (E1, slice 2). The target regime's `next_V_arr` leaf has
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
    transition_laws: TransitionLaws,
    compute_regime_transition_probs: RegimeTransitionFunction,
    regime_to_v_interpolation_info: MappingProxyType[RegimeName, VInterpolationInfo],
    certainty_equivalent: CertaintyEquivalent | None,
    co_map_state_names: tuple[StateName, ...],
) -> tuple[
    Callable[..., tuple[FloatND, MappingProxyType[RegimeName, FloatND]]],
    tuple[Callable[..., Any], ...],
    frozenset[str],
]:
    """Build the closure that aggregates next period's value into `CE`.

    The single continuation-aggregation site of the engine: both the Bellman
    `Q` and the NaN diagnostics call the closure this returns, so they cannot
    disagree. The continuation is a lottery over the stochastic nodes of every
    reachable target regime, weighted by that target's regime-transition
    probability:

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
        transition_laws: Immutable mapping of target regime names to their
            transition laws.
        compute_regime_transition_probs: Regime transition probability function
            for solve.
        regime_to_v_interpolation_info: Immutable mapping of regime names to
            V-interpolation info.
        certainty_equivalent: Nonlinear certainty equivalent declared by the
            regime, or `None` for the linear expectation.
        co_map_state_names: Tuple of state names co-mapped with the continuation V.

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
            transition_laws=transition_laws,
            v_interpolation_info=regime_to_v_interpolation_info[target_regime_name],
            co_map_state_names=co_map_state_names,
        )
        for target_regime_name in period_targets
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
    # the interpolator (which no longer indexes those axes).
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
            zero: Zero at the shape and dtype of the value being built up.
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

        mixture_terms, lottery_values, lottery_weights, probability_mass = (
            _scalar_target_contribution(
                scalar_targets=scalar_targets,
                next_regime_to_V_arr=next_regime_to_V_arr,
                active_regime_probs=active_regime_probs,
                as_lottery=not reduces_per_target,
                zero=zero,
            )
        )
        for target_regime_name in period_targets:
            continuation = continuations[target_regime_name]
            next_states = continuation.next_states(**states_actions_params)
            joint_next_stochastic_states_weights = continuation.joint_lottery_weights(
                **continuation.lottery_weights(**states_actions_params)
            )

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
            # per-target route in `_expectation_over_stochastic_nodes`, the
            # lottery route in the certainty equivalent's own `aggregate`.
            target_probability = active_regime_probs[target_regime_name]
            probability_mass = probability_mass + target_probability

            if reduces_per_target:
                # Weighted average of the next value function at the stochastic
                # states. Zero-safe: a zero-probability stochastic node beside an
                # admissible on-path `-inf` must not turn the average into a `nan`
                # -- ordinary on the collective branch, where dissolution makes
                # `-inf` continuations routine rather than exotic. This is why the
                # reducer stays `zero_safe_average` rather than becoming
                # `_expectation_over_stochastic_nodes`: that one guards the weight
                # SUM only, so `0 * -inf` would still poison its numerator. The
                # predicate is upstream's `continuation.has_lottery_axes`, which
                # replaced the `next_V_has_stochastic_states` mapping.
                if continuation.has_lottery_axes:
                    next_V_expected_arr = zero_safe_average(
                        next_V_at_stochastic_states_arr,
                        weights=joint_next_stochastic_states_weights,
                    )
                else:
                    next_V_expected_arr = jnp.average(next_V_at_stochastic_states_arr)
                # Collect the UNMULTIPLIED `(prob, expected V)`; the mixture is
                # reduced ONCE by `_sum_regime_mixture` -- stack the operands, one
                # zero-safe contraction, value-ordered sum. See that helper for why
                # this beats a sequential left-fold on accuracy (round-8) and why the
                # order must not depend on regime LABELS (round-10 F1).
                #
                # Upstream's `_neutralize_where_unreachable` is deliberately NOT
                # applied here, and not because the hazard is absent: it is the same
                # `0 * -inf` from a zero-probability target carrying an admissible
                # `-inf`. It is already handled one level in, by
                # `zero_safe_weighted_term` inside `_sum_regime_mixture`, which masks
                # the VALUE before the multiply on the same `weight == 0` predicate
                # and for the reason upstream itself gives -- masking the product
                # instead poisons the gradient. Neutralizing here too would multiply
                # at the call site and put this term back outside the single
                # value-ordered reduction, which is the point of the unmultiplied form.
                mixture_terms.append(
                    (
                        target_regime_name,
                        active_regime_probs[target_regime_name],
                        next_V_expected_arr,
                    )
                )
            else:
                values, node_weights = _as_lottery(
                    values=next_V_at_stochastic_states_arr,
                    weights=joint_next_stochastic_states_weights,
                    has_stochastic_states=continuation.has_lottery_axes,
                )
                # Same rule, applied to the value rather than to a product: the
                # aggregate reduces values and weights together, so a node that
                # cannot occur has to be neutral before it is collected.
                #
                # The test is each node's *final* weight, not the target
                # probability alone. A target reached with certainty still
                # carries nodes of probability zero -- a Markov row with a zero
                # entry beside a state where every action is infeasible -- and
                # the aggregate cannot tell such a node from a live one.
                final_weights = target_probability * node_weights
                lottery_values.append(jnp.where(final_weights == 0, zero, values))
                lottery_weights.append(final_weights)

        # ONE reduction for the whole regime mixture: stack the operands and
        # contract once, value-ordered, rather than folding `CE = CE + p*V` per
        # target. Accuracy (round-8) and a sum order that must not depend on
        # regime LABELS (round-10 F1). Empty on the lottery route, where
        # `_sum_regime_mixture` returns `zeros_like(like)` -- the same zero the
        # upstream accumulator started from, so the branches below compose.
        CE = _sum_regime_mixture(mixture_terms, like=zero)

        if reduces_per_target and (period_targets or scalar_targets):
            # The per-target route accumulates `Σ p·E[V]`, so it has to divide by
            # the represented mass to state the same quantity as
            # `LinearExpectation.aggregate`. Regime-transition validation accepts
            # any mass within `jnp.allclose` of one, and at the top of that
            # tolerance the undivided sum reverses the Bellman argmax. Dividing is
            # exact whenever the mass is exactly one, so a well-formed lottery
            # keeps its floating-point association.
            #
            # A regime with no target at all — neither carrying state nor
            # stateless — carries no continuation, and its `CE` stays at zero
            # rather than becoming `0 / 0`, matching the lottery route, which
            # leaves `CE` at zero when it collects no nodes.
            # A represented mass of zero across targets that do exist is a
            # massless lottery, and NaN there is the same answer both routes give.
            CE = CE / _unit_regime_mass_or_nan(probability_mass)
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
                _regime_mass_is_unit(probability_mass),
                _aggregate_joint_lottery(
                    certainty_equivalent=certainty_equivalent,
                    lottery_values=lottery_values,
                    lottery_weights=lottery_weights,
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
    return compute_CE, deps, frozenset()


@dataclasses.dataclass(frozen=True, kw_only=True)
class _TargetContinuation:
    """Everything built once for one reachable target's continuation."""

    next_states: NextStateSimulationFunction
    """Next-period states of this target at one state-action point."""

    lottery_weights: Callable[..., dict[str, FloatND | IntND]]
    """Marginal probabilities of the target's stochastic laws."""

    joint_lottery_weights: Callable[..., FloatND]
    """Outer product of the lottery marginals, over the node axes."""

    next_V: Callable[..., FloatND]
    """Target's value function, product-mapped over its lottery axes.

    A declared entry gets no axis: its one value is interpolated on the target's
    nodes inside the interpolator, so the surface carries genuine draws only.
    """

    extra_param_names: frozenset[str]
    """Arguments `next_V` needs beyond the next states and the value array.

    A grid whose points arrive at runtime is the case — `wealth__points` for an
    `IrregSpacedGrid`.
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
    node_values: MappingProxyType[TransitionFunctionName, FloatND],
) -> Callable[..., FloatND]:
    """Wrap the interpolator so draw-dependent laws resolve on the node axis.

    The caller product-maps the result over the target's node axes, so one call
    sees one node per stochastic law. The draw's *index* is what indexes the value
    function; the draw's *value* is what a dependent law reads. Both come from the
    same node, which is why resolving them here — inside the axis the process
    already contributes — needs no second axis and no parameter for the draw.

    Args:
        next_V_interpolator: The target's value-function interpolator.
        bundle: This target's unqualified `next_<state>` transition functions.
        functions: Immutable mapping of function names to internal user functions.
        stochastic_names: This target's stochastic `next_<state>` names.
        draw_dependent_names: Laws to resolve here rather than ahead of the axes.
        node_values: Immutable mapping of each stochastic law to its nodes, indexed
            by the value its next-state function yields.

    Returns:
        A callable with the interpolator's signature, minus the laws it now
        resolves itself, plus whatever resolving them reads.

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
    arg_names = sorted((interpolator_args - set(draw_dependent_names)) | resolver_args)

    @with_signature(args=arg_names)
    def interpolate_at_this_node(**kwargs: FloatND) -> FloatND:
        drawn = {
            name: node_values[name][kwargs[name].astype(jnp.int32)]
            for name in read_as_a_draw
        }
        resolved = resolve(
            **{
                k: v for k, v in kwargs.items() if k in resolver_args and k not in drawn
            },
            **drawn,
        )
        return next_V_interpolator(
            **{k: v for k, v in kwargs.items() if k in interpolator_args},
            **resolved,
        )

    return interpolate_at_this_node


def _build_target_continuation(
    *,
    target_regime_name: RegimeName,
    functions: EconFunctionsMapping,
    bundle: MappingProxyType[TransitionFunctionName, TransitionFunction],
    transition_laws: TransitionLaws,
    v_interpolation_info: VInterpolationInfo,
    co_map_state_names: tuple[StateName, ...],
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
        transition_laws: Immutable mapping of target regime names to their
            transition laws.
        v_interpolation_info: The target's V-interpolation info.
        co_map_state_names: Tuple of state names co-mapped with the continuation V.

    Returns:
        The target's continuation machinery.

    """
    lottery_variables = tuple(
        key for key in bundle if is_stochastic(transition_laws, target_regime_name, key)
    )
    basis_variables = tuple(
        key
        for key in bundle
        if is_interpolation_basis(transition_laws, target_regime_name, key)
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
        transition_laws=transition_laws,
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
    # A declared entry is a coordinate like any other now, so a law reading a
    # sibling draw is resolved inside that draw's axes whether it feeds a
    # coordinate or an entry.
    dependent_coordinate_names = tuple(dependencies_by_law)
    node_values = MappingProxyType(
        {
            name: v_interpolation_info.discrete_states[
                name.removeprefix("next_")
            ].to_jax()
            for name in lottery_variables
        }
    )
    if dependencies_by_law:
        _fail_if_a_read_draw_has_no_nodes_yet(
            target_regime_name=target_regime_name,
            dependencies_by_law=dependencies_by_law,
            v_interpolation_info=v_interpolation_info,
        )
    if dependent_coordinate_names:
        next_V_interpolator = _get_interpolator_resolving_draws(
            next_V_interpolator=next_V_interpolator,
            bundle=bundle,
            functions=functions,
            stochastic_names=lottery_variables,
            draw_dependent_names=dependent_coordinate_names,
            node_values=node_values,
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
            func=next_V_interpolator,
            variables=node_variables,
            batch_sizes=dict.fromkeys(node_variables, 0),
        ),
        extra_param_names=frozenset(
            get_union_of_args([next_V_interpolator]) - set(bundle) - {V_arr_name}
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
    list[tuple[RegimeName, FloatND, FloatND]], list[FloatND], list[FloatND], FloatND
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

    Returns:
        Tuple of the linear mixture terms, the lottery values, their weights,
        and the probability mass these targets represent.

    """
    mixture_terms: list[tuple[RegimeName, FloatND, FloatND]] = []
    values: list[FloatND] = []
    weights: list[FloatND] = []
    probability_mass = zero
    for target_regime_name in scalar_targets:
        scalar_V = next_regime_to_V_arr[target_regime_name]
        prob = active_regime_probs[target_regime_name]
        # A stateless target contributes to the represented mass on either
        # route, so the linear fast path divides by the mass of *every* target
        # it summed, not just the ones carrying state.
        probability_mass = probability_mass + prob
        # Same rule on either route: a target reached with no mass contributes
        # nothing, and its value is neutralized rather than its product masked.
        # `-inf` is the ordinary value of a state where every action is
        # infeasible, so `0 * -inf` is otherwise how a single unreachable target
        # takes the reachable ones down with it.
        if as_lottery:
            node = jnp.ravel(scalar_V)
            values.append(jnp.where(prob == 0, jnp.zeros_like(node), node))
            weights.append(prob * jnp.ones_like(node))
        else:
            # UNMULTIPLIED, like every carry target: `_sum_regime_mixture` forms
            # `p_r * V_r` once inside a single zero-safe contraction, masking the
            # VALUE before the multiply. That is the same neutralization upstream's
            # `_neutralize_where_unreachable` performs, on the same `prob == 0`
            # predicate, so applying it here as well would be redundant -- and
            # multiplying here would reintroduce `0 * -inf = nan` for a zero-mass
            # stateless target and put this term outside the value-ordered reduction.
            mixture_terms.append((target_regime_name, prob, scalar_V))
    return mixture_terms, values, weights, probability_mass


def _expectation_over_stochastic_nodes(*, values: FloatND, weights: FloatND) -> FloatND:
    """Return the weighted mean of one target's continuation over its nodes.

    Normalized explicitly rather than with `jnp.average`, for the reason
    `_as_lottery` states: a target whose joint weights carry no mass
    contributes no branch, and must not contribute NaN either — every target
    enters the same continuation, so a NaN here would destroy the
    well-specified targets beside it.

    The same holds one level down, at a single node of a target that does carry
    mass: a node of probability zero contributes nothing whatever value stands
    there, so it is dropped rather than multiplied by its zero weight.

    Two details of how that node is dropped:

    - the mask sits on the **value**, so the multiplication stays a bare
      operation feeding the sum and can be contracted into a fused
      multiply-add. Selecting on the product instead forces it to round before
      the sum rounds again, which every well-specified node pays for;
    - the test is `== 0`, not `> 0`. A negative weight is a malformed
      specification and a `NaN` weight is not a probability at all; `> 0` is
      false for both and would launder either into a zero contribution, turning
      a broken transition into a plausible number.
    """
    weight_sum = jnp.sum(weights)
    safe_weight_sum = jnp.where(weight_sum > 0.0, weight_sum, 1.0)
    return jnp.sum(zero_safe_weighted_term(weights, values)) / safe_weight_sum


def _as_lottery(
    *,
    values: FloatND,
    weights: FloatND,
    has_stochastic_states: bool,
) -> tuple[Float1D, Float1D]:
    """Flatten one target regime's continuation into a unit-mass lottery.

    Args:
        values: Next period's value at this target's stochastic nodes.
        weights: Joint weights over those nodes; ignored when the target has
            no stochastic states.
        has_stochastic_states: Whether the target's transition draws stochastic
            states.

    Returns:
        Tuple of the flattened values and their probabilities, which sum to
        one — or to zero for a target whose weights carry no mass at all.

    """
    flat_values = jnp.ravel(values)
    if has_stochastic_states:
        flat_weights = jnp.ravel(weights)
        weight_sum = jnp.sum(flat_weights)
        # A target whose joint weights carry no mass contributes no branch. It
        # must not contribute NaN either: every target's nodes are concatenated
        # into one lottery, so a NaN here would destroy the certainty
        # equivalent of the well-specified targets alongside it.
        safe_weight_sum = jnp.where(weight_sum > 0.0, weight_sum, 1.0)
        return flat_values, flat_weights / safe_weight_sum
    uniform = jnp.full(
        flat_values.shape, 1.0 / flat_values.size, dtype=flat_values.dtype
    )
    return flat_values, uniform


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
) -> Callable[..., FloatND]:
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
        A function that computes the outer product of the variables' weights.

    """
    arg_names = [f"weight_{regime_name}__{key}" for key in variables]

    @with_signature(args=arg_names)
    def _outer(**kwargs: Float1D) -> FloatND:
        weights = jnp.array(list(kwargs.values()))
        return jnp.prod(weights)

    variables = tuple(arg_names)
    return productmap(
        func=_outer, variables=variables, batch_sizes=dict.fromkeys(variables, 0)
    )


def _get_U_and_F(
    *,
    functions: EconFunctionsMapping,
    constraints: ConstraintFunctionsMapping,
    utility_name: str = "utility",
) -> Callable[..., tuple[FloatND, BoolND]]:
    """Get the instantaneous utility and feasibility function.

    Note:
    -----
    U may depend on all kinds of other functions (taxes, transfers, ...), which will be
    executed if they matter for the value of U.

    Args:
        functions: Immutable mapping of function names to internal user functions.
        constraints: Immutable mapping of constraint names to internal user functions.
        utility_name: DAG target name of the felicity function. `"utility"` (the
            default) is the singleton case; a collective regime passes a
            per-stakeholder `"utility_<s>"` so this builder returns that
            stakeholder's own `U^s` alongside the shared feasibility.

    Returns:
        The instantaneous utility and feasibility function.

    """
    return concatenate_functions(
        functions={
            "feasibility": _get_feasibility(
                functions=functions, constraints=constraints
            ),
            **dict(functions),
        },
        # `utility_name`, not the literal: a collective regime builds one `U^s`
        # per stakeholder off the same shared feasibility.
        targets=[utility_name, "feasibility"],
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


def _regime_mass_is_unit(probability_mass: FloatND) -> BoolND:
    """Whether the represented regime mass is unit mass, within tolerance."""
    return jnp.abs(probability_mass - 1.0) <= _MAX_REGIME_MASS_DEVIATION


def _aggregate_joint_lottery(
    *,
    certainty_equivalent: CertaintyEquivalent,
    lottery_values: Sequence[FloatND],
    lottery_weights: Sequence[FloatND],
    ce_flat_param_names: Mapping[str, str],
    states_actions_params: Mapping[str, Any],
) -> FloatND:
    """Aggregate the continuation nodes of every retained target in one piece.

    Args:
        certainty_equivalent: The regime's certainty equivalent.
        lottery_values: Sequence of per-target continuation values.
        lottery_weights: Sequence of per-target node weights, already scaled by
            the target's regime-transition probability.
        ce_flat_param_names: Mapping of certainty-equivalent argument names to
            their flat parameter names.
        states_actions_params: Mapping of states, actions, age, period, and flat
            regime params.

    Returns:
        The aggregated continuation value.

    """
    return certainty_equivalent.aggregate(
        values=jnp.concatenate(list(lottery_values)),
        weights=jnp.concatenate(list(lottery_weights)),
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


def _unit_regime_mass_or_nan(probability_mass: FloatND) -> FloatND:
    """Return the mass itself, or NaN where it is not unit mass.

    For the per-target route, which divides by the mass it accumulated.
    """
    return jnp.where(_regime_mass_is_unit(probability_mass), probability_mass, jnp.nan)
