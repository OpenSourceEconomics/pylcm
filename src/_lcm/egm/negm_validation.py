"""Build-time validation of the NEGM model contract.

A regime configured with `solver=NEGM(inner=DCEGM(...), ...)` must satisfy the
nested-EGM contract on top of the inner DC-EGM contract. NEGM is a
**1-D-inner + outer-search** algorithm: the inner consumption-savings problem is
solved by DC-EGM conditional on a fixed outer (durable/illiquid) margin, and the
outer margin is a deterministic grid search. Every violation raises
`ModelInitializationError` at `Model` construction, naming the offending feature
**and** the correct alternative solver, so no rejection path silently degrades to
a different algorithm. The inner 1-D DC-EGM contract is validated separately,
against the outer-margin-bound inner view the kernel builds; this module checks
only the *outer*/nesting contract. The rules, in the order they are checked:

- the outer margin exists: `outer_action` is a continuous action of the regime
  and `outer_post_decision` is a function the regime declares — a regime with
  no outer margin is a pure 1-D consumption-savings problem and must use
  `DCEGM`,
- the two margins are distinct: the outer action is not the inner continuous
  action and the outer post-decision is not the inner post-decision (also
  guarded at `NEGM` construction; re-checked here for a single fail-loud point),
- no coupled-2-Euler structure: the outer post-decision enters the inner
  resources and/or an additively-separable utility term and the child-state
  index ONLY. If it enters the inner Euler-state transition or any other
  savings-stage function (the differentiated continuation pool), the `c` and the
  outer FOCs invert on the same continuation — the DS pension shape — and NEGM's
  deterministic outer max is invalid; the model needs the 2-D EGM foundation
  (G2EGM / multidim-RFC),
- discrete-aggregation ordering: a taste-shocked discrete choice must be the
  outermost aggregation, with the outer search living inside each discrete
  branch. The single-inner-`DCEGM` composition wraps the outer max *around* the
  inner solve (which already performs the discrete `logsumexp`), so the outer
  max sits outside the discrete aggregation — the wrong order
  (`max_{s'} logsumexp_d ≠ logsumexp_d max_{s'}`). A taste-shocked NEGM regime is
  therefore rejected,
- carry layout: the stacked outer continuation carry addresses the durable as
  the last passive row axis, so a (hard) discrete action or a passive
  continuous state declared after the durable is rejected — the per-durable
  candidate lift would otherwise address the wrong axis,
- outer-cost contract: with a declared `NEGM.outer_cost`, the resources
  function is composed by `finalize_regimes` as
  `<resources>_before_outer_cost - <outer_cost>` (affine in the cost by
  construction; a user-defined resources function is rejected at
  finalization); the declared cost may read only the durable state, the outer
  post-decision, and params, and the cost-free base must not read the outer
  post-decision (with `outer_cost=None`, resources must be independent of the
  outer post-decision) — otherwise no constant credited-cost translation onto
  a common cash-on-hand axis exists and the stacked lift would be wrong,
- the no-adjustment candidate is a unary function of the durable state — it is
  evaluated as `keep(durable)` in the credited-cost lift and the
  child-resources query map.

The coupled-2-Euler detector is structural and deliberately over-rejects:
catching the DS pension coupling (the outer post-decision feeding the inner
Euler law) and accepting the kinked-toy / housing-adjuster shape (the outer
post-decision read only by inner resources/utility and indexing the child
durable state) is the boundary it provably distinguishes. A model that couples
the two margins through a channel other than the inner Euler law and the
savings-stage functions — e.g. a non-additively-separable utility cross-term in
`(c, s')` — is rejected by the additive-separability check on utility.
"""

import inspect
from collections.abc import Mapping
from typing import cast

from _lcm.egm.validation import (
    _continuous_non_process_names,
    _dag_ancestors,
    _resolve_solve_functions,
    _savings_stage_candidates,
    _solve_grids,
    _without,
)
from _lcm.grids import ContinuousGrid
from _lcm.typing import FunctionName, RegimeName
from lcm.exceptions import ModelInitializationError
from lcm.regime import Regime as UserRegime
from lcm.solvers import DCEGM, NEGM
from lcm.typing import UserFunction


def validate_negm_regimes(
    *,
    user_regimes: Mapping[RegimeName, UserRegime],
) -> None:
    """Validate the NEGM contract for every regime with an `NEGM` solver.

    Args:
        user_regimes: Mapping of regime names to user-provided `Regime`
            instances.

    Raises:
        ModelInitializationError: If any regime with `solver=NEGM(...)` violates
            the NEGM model contract.

    """
    for regime_name, user_regime in user_regimes.items():
        if isinstance(user_regime.solver, NEGM):
            _validate_negm_regime(
                regime_name=regime_name,
                user_regime=user_regime,
            )


def _validate_negm_regime(
    *,
    regime_name: RegimeName,
    user_regime: UserRegime,
) -> None:
    """Run all NEGM contract checks for a single regime, in order."""
    solver = cast("NEGM", user_regime.solver)
    inner = solver.inner

    functions = _resolve_solve_functions(user_regime=user_regime)

    _fail_if_outer_margin_absent(
        regime_name=regime_name,
        user_regime=user_regime,
        functions=functions,
        solver=solver,
    )
    _fail_if_margins_not_distinct(regime_name=regime_name, solver=solver, inner=inner)
    _fail_if_outer_margin_euler_coupled(
        regime_name=regime_name,
        user_regime=user_regime,
        functions=functions,
        solver=solver,
    )
    _fail_if_taste_shock_ordering_violated(
        regime_name=regime_name, user_regime=user_regime
    )
    _fail_if_carry_layout_unsupported(
        regime_name=regime_name, user_regime=user_regime, solver=solver, inner=inner
    )
    _fail_if_outer_cost_contract_violated(
        regime_name=regime_name,
        user_regime=user_regime,
        functions=functions,
        solver=solver,
    )
    _fail_if_no_adjustment_candidate_not_unary(
        regime_name=regime_name, functions=functions, solver=solver
    )


def _fail_if_outer_margin_absent(
    *,
    regime_name: RegimeName,
    user_regime: UserRegime,
    functions: dict[FunctionName, UserFunction],
    solver: NEGM,
) -> None:
    """The outer durable margin must be a real action and post-decision function.

    A regime that declares no outer continuous margin (the outer action is not
    among its continuous actions, or the outer post-decision is not a declared
    function) is a pure 1-D consumption-savings problem; NEGM would silently run
    as plain DC-EGM, so it is rejected with the `DCEGM` pointer.
    """
    continuous_actions = _continuous_non_process_names(
        grids=_solve_grids(slot=user_regime.actions)
    )
    if solver.outer_action not in continuous_actions:
        msg = (
            f"NEGM.outer_action '{solver.outer_action}' is not a continuous "
            f"action of regime '{regime_name}'. NEGM nests an outer continuous "
            f"margin; this regime declares none (continuous actions: "
            f"{continuous_actions}) — use `DCEGM` for a pure 1-D "
            "consumption-savings regime."
        )
        raise ModelInitializationError(msg)
    # The outer post-decision `s'` is either a declared regime function or the
    # auto-generated transition `next_<state>` of the durable state (its law of
    # motion produces the next durable stock).
    transition_names = {f"next_{name}" for name in user_regime.states}
    if (
        solver.outer_post_decision not in functions
        and solver.outer_post_decision not in transition_names
    ):
        msg = (
            f"NEGM.outer_post_decision '{solver.outer_post_decision}' is neither "
            f"a declared function of regime '{regime_name}' nor the transition "
            "of one of its states. The outer post-decision (the next-period "
            "durable stock) must be a regime function or the durable state's "
            f"`next_<state>` law that the inner resources and the child-state "
            "index read; declare it, or use `DCEGM` for a pure 1-D "
            "consumption-savings regime."
        )
        raise ModelInitializationError(msg)


def _fail_if_margins_not_distinct(
    *,
    regime_name: RegimeName,
    solver: NEGM,
    inner: DCEGM,
) -> None:
    """The outer and inner margins must be distinct actions and post-decisions.

    Re-checks the construction-time guards at model build so every NEGM
    rejection surfaces from one validation entry point.
    """
    if solver.outer_action == inner.continuous_action:
        msg = (
            f"NEGM.outer_action '{solver.outer_action}' of regime "
            f"'{regime_name}' coincides with the inner DC-EGM continuous action "
            f"'{inner.continuous_action}'. The outer durable/illiquid margin and "
            "the inner consumption margin must be distinct actions."
        )
        raise ModelInitializationError(msg)
    if solver.outer_post_decision == inner.post_decision_function:
        msg = (
            f"NEGM.outer_post_decision '{solver.outer_post_decision}' of regime "
            f"'{regime_name}' coincides with the inner DC-EGM post-decision "
            f"function '{inner.post_decision_function}'. The outer post-decision "
            "(the next-period durable stock) and the inner post-decision (the "
            "liquid savings) must be distinct functions."
        )
        raise ModelInitializationError(msg)


def _fail_if_outer_margin_euler_coupled(
    *,
    regime_name: RegimeName,
    user_regime: UserRegime,
    functions: dict[FunctionName, UserFunction],
    solver: NEGM,
) -> None:
    """The outer margin must not enter the inner differentiated-Euler pool.

    NEGM's deterministic outer max is valid only when the inner 1-D Euler
    inversion is independent of the outer choice: the outer post-decision may be
    read by the inner resources and an additively-separable utility term, and it
    indexes the child durable state — but it must NOT enter the inner
    Euler-state transition or any other savings-stage function (the regime
    transition, stochastic weights, non-Euler state laws). Those functions feed
    the differentiated continuation the inner Euler equation inverts on, so the
    `c` and outer FOCs would invert on the same continuation — the DS pension
    coupling — which a single inverse-Euler cannot represent.

    The post-decision and resources functions are removed from the DAG (they
    become leaves), so an ancestor hit on the outer post-decision is a genuine
    decision-time coupling, not the legitimate resources read.
    """
    inner = solver.inner
    opaque_functions = _without(
        functions=functions,
        names={inner.post_decision_function, inner.resources},
    )
    coupling_msg = (
        f"the outer margin '{solver.outer_post_decision}' is Euler-coupled to "
        f"the inner state '{inner.continuous_state}' of regime '{regime_name}' "
        "through the shared continuation"
    )
    for _role, label, func in _savings_stage_candidates(
        user_regime=user_regime, solver=inner
    ):
        ancestors = _dag_ancestors(functions=opaque_functions, target_func=func)
        if solver.outer_post_decision in ancestors:
            msg = (
                f"In regime '{regime_name}', the {label} reads the outer "
                f"post-decision '{solver.outer_post_decision}', so {coupling_msg}: "
                "the inner Euler inversion is no longer independent of the outer "
                "choice. NEGM's deterministic outer max is invalid here — use the "
                "2-D EGM foundation (G2EGM / multidim-RFC), not NEGM."
            )
            raise ModelInitializationError(msg)

    # Utility may carry the outer margin only additively-separably from the
    # inner action: a cross-term in (consumption, outer post-decision) makes the
    # inner marginal utility depend on the outer choice, so the inner Euler
    # inversion couples to it.
    _fail_if_utility_couples_action_and_outer_margin(
        regime_name=regime_name,
        functions=functions,
        solver=solver,
    )


def _fail_if_utility_couples_action_and_outer_margin(
    *,
    regime_name: RegimeName,
    functions: dict[FunctionName, UserFunction],
    solver: NEGM,
) -> None:
    """Utility must read the outer margin additively-separably from consumption.

    The inner Euler inversion treats the outer margin's utility term as a
    constant (`u(c, s') = ũ(c) + g(s')`). If `utility` reaches both the inner
    continuous action and the outer post-decision through a single function that
    multiplies/composes them (a cross-term), the inner marginal utility depends
    on the outer choice and the deterministic outer max over a frozen inner
    inversion is invalid. The structural proxy: no individual utility-DAG
    function may take both the inner action and the outer post-decision (or the
    outer action) as direct arguments.
    """
    inner = solver.inner
    outer_names = {solver.outer_action, solver.outer_post_decision}
    utility_dag_names = _dag_ancestors(
        functions=functions,
        target_func=functions["utility"],
    ) | {"utility"}
    for func_name in utility_dag_names:
        func = functions.get(func_name)
        if func is None:
            continue
        arg_names = set(inspect.signature(func).parameters)
        if inner.continuous_action in arg_names and arg_names & outer_names:
            crossed = sorted(arg_names & outer_names)
            msg = (
                f"In regime '{regime_name}', the utility-DAG function "
                f"'{func_name}' takes both the inner consumption action "
                f"'{inner.continuous_action}' and the outer margin {crossed} as "
                "arguments, so utility couples the two margins (it is not "
                "additively separable in them). The inner marginal utility then "
                "depends on the outer choice, so NEGM's deterministic outer max "
                "over a frozen inner Euler inversion is invalid — use the 2-D EGM "
                "foundation (G2EGM / multidim-RFC), not NEGM."
            )
            raise ModelInitializationError(msg)


def _fail_if_taste_shock_ordering_violated(
    *,
    regime_name: RegimeName,
    user_regime: UserRegime,
) -> None:
    """A taste-shocked discrete choice must be the outermost aggregation.

    The single-inner-`DCEGM` composition wraps the outer search around the whole
    inner solve, which already performs the discrete `logsumexp` aggregation.
    The outer max therefore sits *outside* the discrete aggregation, computing
    `max_{s'} logsumexp_d` — but with EV taste shocks the correct order is
    `logsumexp_d max_{s'}`, the search nested inside each discrete branch
    (`max_{s'} logsumexp_d ≠ logsumexp_d max_{s'}`). A taste-shocked NEGM regime
    is rejected until the outer search can be pushed inside each discrete branch.
    """
    if user_regime.taste_shocks is not None:
        msg = (
            f"Regime '{regime_name}' declares EV1 taste shocks and uses the NEGM "
            "solver. NEGM wraps its outer durable-margin search around the inner "
            "DC-EGM solve, which performs the discrete-choice aggregation — so "
            "the outer max sits outside the taste-shock aggregation. With taste "
            "shocks the discrete choice must be the outermost aggregation, with "
            "the outer search nested inside each discrete branch "
            "(max over the durable margin of logsumexp over the discrete choice "
            "is not logsumexp of the max). Remove the taste shocks or use the "
            "grid-search solver for this regime."
        )
        raise ModelInitializationError(msg)


def _fail_if_outer_cost_contract_violated(
    *,
    regime_name: RegimeName,
    user_regime: UserRegime,
    functions: dict[FunctionName, UserFunction],
    solver: NEGM,
) -> None:
    """The declared outer cost must be the sole outer-margin channel of resources.

    The stacked-carry lift places every outer candidate on the keeper's
    cash-on-hand axis by crediting a constant per (durable, outer-node) cell.
    That constant exists exactly when the inner resources depend on the outer
    post-decision through a single additive cost term that itself varies only
    with the durable margin. The contract is enforced fail-closed:

    - `NEGM.outer_cost` declared ⇒ the resources function is composed by
      `finalize_regimes` as `<resources>_before_outer_cost - <outer_cost>`, so
      its affine use of the cost holds by construction (a user-defined
      resources function is rejected at finalization). Here: the cost must be
      a regime function whose DAG ancestors contain no state or action other
      than the durable state and the outer post-decision (params are fine),
      and the cost-free base must not read the outer post-decision — its only
      outer-margin channel is the subtracted cost itself,
    - `NEGM.outer_cost=None` ⇒ the inner resources must be independent of the
      outer post-decision altogether (every candidate already shares the
      keeper's axis; the shift is zero).
    """
    inner = solver.inner
    durable_state = solver.outer_post_decision.removeprefix("next_")

    if solver.outer_cost is None:
        resources_func = functions.get(inner.resources)
        if resources_func is None:
            return
        resources_ancestors = _dag_ancestors(
            functions=functions, target_func=resources_func
        )
        if solver.outer_post_decision in resources_ancestors:
            msg = (
                f"In regime '{regime_name}', the inner resources "
                f"'{inner.resources}' reads the outer post-decision "
                f"'{solver.outer_post_decision}' but the solver declares no "
                "outer cost (`NEGM.outer_cost=None`). Declare the credited-cost "
                "function via `NEGM.outer_cost` so the stacked-carry lift can "
                "place every candidate on a common cash-on-hand axis, or use "
                "`GridSearch` for this regime."
            )
            raise ModelInitializationError(msg)
        return

    cost_func = functions.get(solver.outer_cost)
    if cost_func is None:
        msg = (
            f"NEGM.outer_cost '{solver.outer_cost}' is not a declared function "
            f"of regime '{regime_name}'. The credited outer cost must be a "
            "regime function reading only the durable state, the outer "
            "post-decision, and params."
        )
        raise ModelInitializationError(msg)

    cost_ancestors = _dag_ancestors(functions=functions, target_func=cost_func)
    state_and_action_names = set(user_regime.states) | set(user_regime.actions)
    offenders = sorted((cost_ancestors & state_and_action_names) - {durable_state})
    if offenders:
        msg = (
            f"NEGM.outer_cost '{solver.outer_cost}' of regime '{regime_name}' "
            f"reads {offenders}. The declared outer cost may read only the "
            f"durable state '{durable_state}', the outer post-decision "
            f"'{solver.outer_post_decision}', and params: the credited-cost "
            "lift is a constant per (durable, outer-node) cell, so a cost that "
            "varies with the Euler state or a ride-along state/action has no "
            "constant translation onto a common cash-on-hand axis. Restructure "
            "the cost, or use `GridSearch` for this regime."
        )
        raise ModelInitializationError(msg)

    base_func = functions.get(f"{inner.resources}_before_outer_cost")
    if base_func is not None:
        base_ancestors = _dag_ancestors(functions=functions, target_func=base_func)
        if solver.outer_post_decision in base_ancestors:
            msg = (
                f"In regime '{regime_name}', the cost-free resources base "
                f"'{inner.resources}_before_outer_cost' reads the outer "
                f"post-decision '{solver.outer_post_decision}'. It must not "
                "read the outer post-decision: pylcm composes "
                f"`{inner.resources} = {inner.resources}_before_outer_cost - "
                f"{solver.outer_cost}`, so the base's only outer-margin "
                "channel is the subtracted declared cost itself. Route the "
                f"dependence through '{solver.outer_cost}', or use "
                "`GridSearch` for this regime."
            )
            raise ModelInitializationError(msg)


def _fail_if_no_adjustment_candidate_not_unary(
    *,
    regime_name: RegimeName,
    functions: dict[FunctionName, UserFunction],
    solver: NEGM,
) -> None:
    """The no-adjustment candidate must be a unary function of the durable state.

    The keeper's no-adjustment level is evaluated as `keep(durable)` in the
    credited-cost lift and in the parent's child-resources query map, so a
    candidate whose signature reads anything else — another state, an action,
    or a param — cannot be bound at those call sites.
    """
    if solver.outer_no_adjustment_candidate is None:
        return
    candidate_func = functions.get(solver.outer_no_adjustment_candidate)
    if candidate_func is None:
        return
    durable_state = solver.outer_post_decision.removeprefix("next_")
    arg_names = set(inspect.signature(candidate_func).parameters)
    if arg_names != {durable_state}:
        msg = (
            f"NEGM.outer_no_adjustment_candidate "
            f"'{solver.outer_no_adjustment_candidate}' of regime "
            f"'{regime_name}' must be a unary function of the durable state "
            f"'{durable_state}' (its signature reads {sorted(arg_names)}). The "
            "keeper's no-adjustment level is evaluated as `keep(durable)` in "
            "the credited-cost lift and the child-resources query map, so no "
            "other state, action, or param can be bound there."
        )
        raise ModelInitializationError(msg)


def _fail_if_carry_layout_unsupported(
    *,
    regime_name: RegimeName,
    user_regime: UserRegime,
    solver: NEGM,
    inner: DCEGM,
) -> None:
    """The stacked outer carry requires durable-last rows and no discrete actions.

    The published NEGM continuation retains every outer candidate on an axis
    inserted directly after the durable margin's passive axis, and lifts each
    candidate by a per-durable-state credited cost addressed through that
    layout. Two regime shapes break it and are rejected:

    - a **discrete action** (the carry's row block would gain action axes after
      the durable, so the lift would mis-identify an action axis as the
      durable) — a taste-shocked discrete choice is already rejected by the
      aggregation-ordering rule; this rejects the remaining hard case,
    - a **passive continuous state declared after the durable** (the durable
      must be the last passive axis for the per-durable lift to address it).
    """
    if user_regime.actions:
        discrete_action_names = [
            name
            for name, grid in _solve_grids(slot=user_regime.actions).items()
            if not isinstance(grid, ContinuousGrid)
        ]
        if discrete_action_names:
            msg = (
                f"Regime '{regime_name}' declares discrete action(s) "
                f"{discrete_action_names} alongside `solver=NEGM(...)`. The "
                "stacked outer continuation carry does not support "
                "discrete-action row axes (they would follow the durable "
                "margin's axis and break the per-durable candidate lift) — "
                "model the discrete choice with `GridSearch`, or fold it into "
                "the outer margin."
            )
            raise ModelInitializationError(msg)

    durable_state = solver.outer_post_decision.removeprefix("next_")
    passive_state_names = [
        name
        for name in _continuous_non_process_names(
            grids=_solve_grids(slot=user_regime.states)
        )
        if name != inner.continuous_state
    ]
    # Only meaningful when the durable is itself a passive continuous state:
    # the "last passive axis" requirement is a statement about its position
    # among the passive states. A durable that is not a passive continuous
    # state at all is a separate error caught upstream, so skip here rather
    # than emit an "after the durable" message that lists states preceding it.
    if durable_state in passive_state_names:
        after_durable = passive_state_names[
            passive_state_names.index(durable_state) + 1 :
        ]
        if after_durable:
            msg = (
                f"Regime '{regime_name}' declares passive continuous state(s) "
                f"{after_durable} after the durable margin '{durable_state}'. "
                "The stacked outer continuation carry lifts each candidate by a "
                "per-durable-state credited cost, so the durable must be the "
                "last passive continuous state the regime declares — reorder "
                f"`states` so '{durable_state}' comes last."
            )
            raise ModelInitializationError(msg)
