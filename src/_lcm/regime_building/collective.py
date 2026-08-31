"""Collective-regime readout: the stakeholder value gather at the household argmax.

A collective regime carries one per-stakeholder action-value array `Q^s` each,
chooses the action that maximizes a household *scalarization*
`O = Σ_s λ_s Q^s` over the feasible set, and then reads off *each stakeholder's
own* `Q^s` at that common argmax — NOT the scalarized value `O`. The household
maximizes the weighted objective, but the individual values are each stakeholder's
own utility stream under that joint choice.

This module is a pure, engine-topology-free building block: it takes already-computed
per-stakeholder `Q` arrays and the feasibility mask and returns the per-stakeholder
`V` plus the all-infeasible flag `D` (the dissolution / empty-feasible-set marker,
kept distinct from a numeric `-inf` that can arise on-path). The terminal and
non-terminal solve kernels call it after building their `Q^s`.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

import jax.numpy as jnp
from dags import concatenate_functions, rename_arguments, with_signature

from _lcm.regime_building.argmax import (
    _flatten_last_n_axes,
    _move_axes_to_back,
    argmax_and_max,
)
from _lcm.typing import FunctionName, RegimeName, StateName, _ParamsLeaf
from _lcm.utils.functools import get_union_of_args
from _lcm.zero_safe import sum_in_value_order, zero_safe_weighted_term
from lcm.collective import ParetoObjective
from lcm.typing import BoolND, FloatND, IntND, UserFunction

# Up to this many stakeholders, the scalarization's reduction order is not a choice:
# one term is returned as-is and two admit a single association. From three terms on,
# a declaration-order left fold lets an economically inert relabeling select a
# different reduction tree, so the sum is canonicalized by contribution value instead.
_LARGEST_ORDER_FREE_HOUSEHOLD = 2


def collective_argmax_and_readout(
    *,
    stakeholder_Q: Mapping[str, FloatND],
    feasibility: BoolND,
    weights: Mapping[str, FloatND | float],
    action_axes: tuple[int, ...],
) -> tuple[IntND, dict[str, FloatND], BoolND]:
    r"""Like `collective_readout`, but also returns the household argmax index.

    The solve-side readout (`collective_readout`) only needs the
    per-stakeholder VALUES at the shared argmax; the simulate-side value
    router additionally needs the argmax INDEX itself, so the engine can look
    up which action was actually taken (mirroring the singleton
    `argmax_and_max_Q_over_a`, whose flat index feeds
    `_lookup_values_from_indices`).

    Returns:
        Tuple `(argmax_flat, V, D)` — the flat argmax index (in the same
        flattened-action layout `argmax_and_max` produces, directly
        compatible with the singleton simulate lookup), the per-stakeholder
        value mapping, and the dissolution flag.
    """
    if not stakeholder_Q:
        msg = "collective_argmax_and_readout requires at least one stakeholder."
        raise ValueError(msg)
    if set(stakeholder_Q) != set(weights):
        msg = (
            "stakeholder_Q and weights must have identical keys; got "
            f"{sorted(stakeholder_Q)} vs {sorted(weights)}."
        )
        raise ValueError(msg)

    objective = _weighted_sum(stakeholder_Q=stakeholder_Q, weights=weights)
    argmax_flat, _ = argmax_and_max(
        a=objective, axis=action_axes, initial=-jnp.inf, where=feasibility
    )
    # With no action axis to reduce, a cell dissolves exactly where its own single
    # (state, action) cell is infeasible.
    dissolution = (
        ~jnp.any(feasibility, axis=action_axes) if action_axes else ~feasibility
    )
    values = {
        name: jnp.where(
            dissolution,
            -jnp.inf,
            _gather_along_actions(
                q=q, argmax_flat=argmax_flat, action_axes=action_axes
            ),
        )
        for name, q in stakeholder_Q.items()
    }
    return argmax_flat, values, dissolution


def collective_readout(
    *,
    stakeholder_Q: Mapping[str, FloatND],
    feasibility: BoolND,
    weights: Mapping[str, FloatND | float],
    action_axes: tuple[int, ...],
) -> tuple[dict[str, FloatND], BoolND]:
    r"""Household argmax of the scalarization, then per-stakeholder value readout.

    One household maximizes the Pareto-weighted sum of its stakeholders' action
    values over the feasible actions, and each stakeholder's value is its own
    action value read at that single choice:

    ```{math}
        a^*(x) = \arg\max_{a\,:\,F(x,a)} \sum_s \lambda_s\, Q^s(x, a),
        \qquad V^s(x) = Q^s(x, a^*(x)).
    ```

    All stakeholders share the same argmax `a*` (the joint household choice), so
    ties are broken identically for every stakeholder — `argmax_and_max` selects
    the first maximizer, and the gather uses that same flattened index for each
    `Q^s`. A cell with no feasible action yields `D = True` (the dissolution /
    empty-`F` marker); the returned `V^s` in such a cell is overwritten with the
    `-inf` sentinel — the masked argmax is arbitrary there, so the gathered
    `Q^s` would otherwise be an infeasible action's value — and must be routed
    by the caller through the dissolution fallback, never read as a value.

    Args:
        stakeholder_Q: Mapping stakeholder name -> its action-value array, each of
            shape `(*state_axes, *action_axes)` (identical shape across stakeholders).
        feasibility: Boolean mask of the same shape as each `Q^s`; `True` where
            the (state, action) is feasible.
        weights: Mapping stakeholder name -> Pareto weight `λ_s` (scalar or an array
            broadcastable to the state axes). Should sum to 1 across stakeholders,
            though this is not enforced (a caller may pass unnormalized weights).
        action_axes: The axes of `Q^s` / `feasibility` to maximize over (the
            action dimensions). The remaining axes are the state axes retained in the
            output.

    Returns:
        Tuple `(V, D)` where `V` maps each stakeholder name to its value array of
        shape `(*state_axes,)` (the action axes reduced away), and `D` is the
        boolean all-infeasible flag of shape `(*state_axes,)`.
    """
    _argmax_flat, values, dissolution = collective_argmax_and_readout(
        stakeholder_Q=stakeholder_Q,
        feasibility=feasibility,
        weights=weights,
        action_axes=action_axes,
    )
    return values, dissolution


def _weighted_sum(
    *,
    stakeholder_Q: Mapping[str, FloatND],
    weights: Mapping[str, FloatND | float],
) -> FloatND:
    """Scalarize the per-stakeholder Q into the household objective Σ_s λ_s Q^s.

    Zero-safe: a stakeholder excluded from the household scalarization via a
    zero Pareto weight may still hold an admissible on-path `-inf` `Q^s` (e.g.
    a feasible zero-consumption action for that partner); `weight * Q` would
    then be `0.0 * -inf = nan`, poisoning the objective — and, via the
    argmax, the WRONG stakeholder's `Q` would decide the household's choice.
    Each term goes through `zero_safe_weighted_term` so a zero-weight
    stakeholder contributes exactly `0.0` regardless of its own `Q^s`.
    """
    names = list(stakeholder_Q)

    # A Pareto weight may arrive as a plain Python float -- see this function's
    # own signature -- while the shared term takes arrays, so it is converted
    # here at the boundary rather than by a permissive union on the term.
    #
    # `subnormal_is_accounted_for=False`: the weight arrives at whatever size
    # the model chose and nothing upstream has put it on a scale, so the term
    # moves the exponent itself rather than let a below-normal weight flush and
    # price a stakeholder out of the household objective entirely.
    def _term(name: str) -> FloatND:
        return zero_safe_weighted_term(
            weight=jnp.asarray(weights[name]),
            value=stakeholder_Q[name],
            subnormal_is_accounted_for=False,
        )

    terms = [_term(name) for name in names]
    if len(terms) <= _LARGEST_ORDER_FREE_HOUSEHOLD:
        # Preserve the established one-/two-stakeholder arithmetic exactly.  With
        # two terms there is no association choice, so relabeling cannot select a
        # different reduction tree.
        objective = terms[0]
        for term in terms[1:]:
            objective = objective + term
        return objective

    # `stakeholders` is an ordered representation of identities, not an economic
    # ordering of summands.  For three or more terms a declaration-order left fold
    # can cross a strict action boundary under cancellation.  Canonicalise by the
    # contribution VALUES, the same invariant used for the regime mixture.
    return sum_in_value_order(values=jnp.stack(terms, axis=0), axis=0)


def _gather_along_actions(
    *, q: FloatND, argmax_flat: IntND, action_axes: tuple[int, ...]
) -> FloatND:
    """Gather `q` at the flattened action argmax, mirroring `argmax_and_max`.

    `argmax_and_max` moves `action_axes` to the back, flattens them, and argmaxes
    the last axis, so `argmax_flat` indexes into that flattened action space with
    the state axes as its shape. Reproduce the same layout on `q` and take along it.

    With no action axis to reduce, `argmax_and_max` returns `q`'s own maximum —
    the array itself — so the gather is the identity here too.
    """
    if not action_axes:
        return q

    q_moved = _move_axes_to_back(a=q, axes=action_axes)
    q_flat = _flatten_last_n_axes(a=q_moved, n=len(action_axes))
    gathered = jnp.take_along_axis(q_flat, argmax_flat[..., None], axis=-1)
    return gathered[..., 0]


# Template key the Pareto weights' free parameters live under.
PARETO_OBJECTIVE_ENTRY = "pareto_objective"

# The role a subject in a singleton regime carries: it occupies none. Negative
# so it can never collide with a declared role's code, and so an out-of-range
# read is a visible mistake rather than someone else's role.
NO_ROLE = -1


def build_role_vocabulary(
    stakeholders_by_regime: Mapping[RegimeName, tuple[str, ...] | None],
) -> MappingProxyType[str, int]:
    """Assign one code to every role any regime of the model declares.

    One vocabulary for the whole model rather than one per regime: a row moves
    between regimes, and comparing the role it carried in one against the roles
    another declares only means something if the codes agree. Regimes whose
    role names are disjoint therefore coexist, each simply using its own part
    of the vocabulary.

    Args:
        stakeholders_by_regime: Mapping of regime names to their stakeholder
            tuples, or `None` for a singleton regime.

    Returns:
        Immutable mapping of role name to code, in declaration order across
        regimes taken in their own order.
    """
    names: dict[str, None] = {}
    for stakeholders in stakeholders_by_regime.values():
        for name in stakeholders or ():
            names.setdefault(name, None)
    return MappingProxyType({name: code for code, name in enumerate(names)})


@dataclass(frozen=True, kw_only=True)
class ParetoWeights:
    """A collective regime's Pareto weights, as the kernel evaluates them.

    Built once per regime and called at every cell, so the weights are as much
    a function of the state as any other node — and the household still solves
    one argmax against one weighting, because a weight may not read the action
    it helps choose.
    """

    compute: Callable[..., dict[str, FloatND]]
    """Return the weights the objective uses at one cell, keyed by stakeholder.

    Normalized under the `"pointwise"` convention; the declared values under
    `"none"`.
    """

    declared: Callable[..., dict[str, FloatND]]
    """Return the DECLARED weights at one cell, before any normalization.

    What the admissibility check reads: normalizing turns a total of zero into
    NaN and a total of one into the answer regardless of what was declared, so
    the check that a declaration is a Pareto weighting has to see it as written.
    """

    arg_names: tuple[str, ...]
    """States, `period` / `age`, and qualified params `compute` reads."""

    param_names: tuple[str, ...]
    """The subset of `arg_names` supplied from the regime's flat params."""

    normalization: str
    """`"pointwise"` divides by the total at each cell; `"none"` does not."""


def build_pareto_weights(
    *,
    objective: ParetoObjective | None,
    stakeholders: tuple[str, ...],
    state_names: frozenset[StateName],
    carried_imputations: Mapping[StateName, UserFunction] = MappingProxyType({}),
    solve_functions: Mapping[FunctionName, UserFunction] = MappingProxyType({}),
) -> ParetoWeights:
    """Build the weight evaluator a collective regime's kernels call per cell.

    An undeclared objective is equal weights, which is the same evaluator with
    constant entries — one path through the kernel, so the symmetric couple and
    the estimated one cannot diverge.

    A weight callable's arguments are the regime's own states, the engine
    context `period` / `age`, and free parameters. The parameters are qualified
    into the regime's `pareto_objective` template entry, so every stakeholder's
    weight reads one shared namespace: `weight_f(pareto_weight)` and
    `weight_m(pareto_weight)` name the same number.

    Args:
        objective: The regime's declaration, or `None` for equal weights.
        stakeholders: Ordered stakeholder names.
        state_names: The regime's own state names.
        carried_imputations: Mapping of each carried state's name to its
            solve-phase imputation. A weight reading such a state is composed
            with the imputation here, so the evaluator the kernels call asks
            only for grid states and parameters.
        solve_functions: Complete solve-phase regime-function pool. It is used
            only while resolving a carried state's imputation, so an ordinary
            helper reached by that imputation remains a DAG dependency rather
            than being reclassified as a parameter of the carried state. Direct
            arguments of a Pareto-weight callable keep their public semantics:
            unless they are states or engine context, they are parameters even
            when their names collide with an ordinary regime function.

    Returns:
        The evaluator, its argument names, and the normalization convention.
    """
    declared: Mapping[str, UserFunction | float] = (
        MappingProxyType(dict.fromkeys(stakeholders, 1.0 / len(stakeholders)))
        if objective is None
        else objective.weights
    )
    context = frozenset({"period", "age"})
    per_stakeholder: dict[str, tuple[Callable[..., FloatND], tuple[str, ...]]] = {}
    param_names: set[str] = set()
    for name in stakeholders:
        weight = declared[name]
        if not callable(weight):
            per_stakeholder[name] = (_constant_weight(float(weight)), ())
            continue
        imputation_params: frozenset[str] = frozenset()
        if carried_imputations and _reads_a_carried_state(
            weight=weight, carried_imputations=carried_imputations
        ):
            weight, imputation_params = _compose_carried_imputations(
                weight=weight,
                carried_imputations=carried_imputations,
                solve_functions=solve_functions,
                state_names=state_names,
                context=context,
            )
        own_args = tuple(get_union_of_args([weight]))
        renamed = {
            arg: arg
            if arg in state_names or arg in context or arg in imputation_params
            else f"{PARETO_OBJECTIVE_ENTRY}__{arg}"
            for arg in own_args
        }
        param_names.update(imputation_params)
        param_names.update(
            qualified for arg, qualified in renamed.items() if qualified != arg
        )
        per_stakeholder[name] = (
            rename_arguments(func=weight, mapper=renamed),
            tuple(renamed.values()),
        )

    arg_names = tuple(
        sorted({arg for _, args in per_stakeholder.values() for arg in args})
    )
    normalization = "pointwise" if objective is None else objective.normalization

    @with_signature(args=list(arg_names), return_annotation="dict")
    def declared_weights(**kwargs: _ParamsLeaf) -> dict[str, FloatND]:
        return {
            name: jnp.asarray(func(**{arg: kwargs[arg] for arg in args}))
            for name, (func, args) in per_stakeholder.items()
        }

    @with_signature(args=list(arg_names), return_annotation="dict")
    def compute(**kwargs: _ParamsLeaf) -> dict[str, FloatND]:
        raw = declared_weights(**kwargs)
        if normalization == "pointwise" and len(raw) > 1:
            stacked = jnp.stack(jnp.broadcast_arrays(*raw.values()), axis=0)
            # Bring the weights down to (0, 1] before summing. Nothing in the
            # admissibility contract bounds how large a finite weight may be,
            # and a raw total that leaves the working format sends every share
            # to zero — which ties the stakeholders and lets the household take
            # an action no declared weighting ranks first.
            #
            # The rescale is a power of two applied with `ldexp`, not a divide
            # by the largest weight. Division would be evaluated as a multiply
            # by the reciprocal, and the reciprocal of a weight near the format
            # maximum is subnormal, which XLA:CPU flushes to zero — turning
            # every share into `0/0`. Scaling by an exponent never forms that
            # reciprocal and is exact, so an ordinary declaration keeps the
            # ratio it already had.
            _, exponent = jnp.frexp(jnp.max(stacked, axis=0))
            scaled = jnp.ldexp(stacked, -exponent)
            # Stakeholder names are economically inert, so the total is summed
            # in value order: relabelling the household must not select a
            # different reduction tree and hence a different normalizer. The
            # scale is itself order-invariant, so it does not reintroduce one.
            total = sum_in_value_order(values=scaled)
            return {
                name: share / total for name, share in zip(raw, scaled, strict=True)
            }
        return raw

    return ParetoWeights(
        compute=compute,
        declared=declared_weights,
        arg_names=arg_names,
        param_names=tuple(sorted(param_names)),
        normalization=normalization,
    )


def _constant_weight(value: float) -> Callable[..., FloatND]:
    """Wrap a declared constant as the zero-argument weight function it is."""

    @with_signature(args=[], return_annotation="FloatND")
    def weight() -> FloatND:
        return jnp.asarray(value)

    return weight


def _reads_a_carried_state(
    *,
    weight: UserFunction,
    carried_imputations: Mapping[StateName, UserFunction],
) -> bool:
    """Whether this weight names a carried state among its arguments."""
    return any(arg in carried_imputations for arg in get_union_of_args([weight]))


def _compose_carried_imputations(
    *,
    weight: UserFunction,
    carried_imputations: Mapping[StateName, UserFunction],
    solve_functions: Mapping[FunctionName, UserFunction],
    state_names: frozenset[StateName],
    context: frozenset[str],
) -> tuple[Callable[..., FloatND], frozenset[str]]:
    """Resolve a weight's carried-state reads through the solve-function DAG.

    A carried state has no solve grid axis: during backward induction its value
    is the imputation declared alongside it. The imputation is a first-class
    solve-phase regime function, so its ancestry may include ordinary helpers or
    other carried imputations. Those names must stay DAG edges. Treating an
    ordinary helper read as a free argument would qualify it as
    ``<carried-state>__<helper>`` and later fail with a missing parameter even
    though the model declared a producer for it.

    Resolution is deliberately two-stage. First, each carried imputation is
    composed against the complete solve-function pool. Second, the Pareto weight
    is composed only against those already-resolved carried imputations. This
    preserves the public weight contract: a direct weight argument whose name
    happens to collide with an ordinary regime function remains a
    ``pareto_objective`` parameter; only an argument naming a carried state is
    substituted by a regime-function value.

    Every solve function's free parameters are qualified under that function's
    own name, matching the regime parameter template. Unreachable helpers and
    their parameters are pruned by ``concatenate_functions``.

    Returns:
        Tuple of the composed weight and the qualified solve-function parameter
        names it actually reads.
    """
    function_pool = {**solve_functions, **carried_imputations}
    function_names = frozenset(function_pool)
    qualified_params: set[str] = set()
    qualified_pool: dict[str, UserFunction] = {}

    for name, func in function_pool.items():
        mapper: dict[str, str] = {}
        for arg in get_union_of_args([func]):
            if arg in function_names or arg in state_names or arg in context:
                mapper[arg] = arg
            else:
                qualified = f"{name}__{arg}"
                mapper[arg] = qualified
                qualified_params.add(qualified)
        qualified_pool[name] = rename_arguments(func=func, mapper=mapper)

    directly_read = frozenset(get_union_of_args([weight])) & frozenset(
        carried_imputations
    )
    resolved_imputations = {
        name: concatenate_functions(
            functions=dict(qualified_pool),
            targets=name,
        )
        for name in directly_read
    }

    target = "__pareto_weight__"
    functions: dict[str, UserFunction] = {
        **resolved_imputations,
        target: weight,
    }
    composed = concatenate_functions(functions=functions, targets=target)
    reached = frozenset(get_union_of_args([composed]))
    return composed, frozenset(qualified_params) & reached
