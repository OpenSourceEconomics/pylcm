"""Build-time validation of the DC-EGM model contract.

A regime configured with `solver=DCEGM(...)` must satisfy the structural
contract the endogenous grid method relies on. Every violation raises
`ModelInitializationError` at `Model` construction with a message naming the
offending piece. The rules, in the order they are checked:

- exactly one continuous (*Euler*) state and one continuous action; process
  states are exempt (they enter the value function as node-valued discrete
  dimensions); every other continuous state must be *passive*: deterministic,
  with a transition whose DAG ancestors include neither the continuous
  action, the resources function, nor the post-decision function (the
  current Euler state is allowed — see the savings-node-stage rule below),
  and a grid that is not distributed (`batch_size` is honored)
- the post-decision function and the resources function exist in
  `Regime.functions` (`inverse_marginal_utility` is optional — when omitted,
  the iEGM path derives a numerical inverse from `utility`)
- the regime uses the default Bellman aggregator `H`
- the post-decision function consumes the continuous action and the
  resources function (not the continuous state directly)
- no constraint touches the continuous state or action (EGM enforces the
  budget identity and the borrowing limit intrinsically)
- `utility` does not depend on the continuous state (envelope condition)
- the resources function does not depend on the continuous action
- the Euler state's transition is deterministic, names the post-decision
  function, and reaches the continuous action only through it
- everything evaluated at the savings-node stage (the Euler state's law,
  the regime transition, stochastic weights, non-Euler state transitions)
  is independent of the continuous action, the resources function, and the
  post-decision function; any of them may read the current Euler state (the
  kernel then solves per exogenous asset node, where current assets are
  known), but only through values that are CONTINUOUS in the Euler state at
  the resolution of the Euler grid — a numeric spot check on the grid nodes
  plus two levels of cell-midpoint refinement rejects values that jump
  (within a cell, a smooth function's quarter-cell increments shrink like a
  derivative bound, while a cliff's increment survives subdivision; a jump
  in the Euler law makes the child's value function discontinuous and the
  true policy bunches at the discontinuity, a corner outside EGM's
  candidate families, while a jump in the other savings-stage functions
  breaks the smoothness-at-node-resolution assumption the per-node solve
  relies on)
- grid hygiene: the Euler grid is not distributed (`batch_size` is honored),
  and the savings grid covers the Euler grid's upper region
- every declared-reachable non-terminal target regime also uses DC-EGM with
  the same Euler state (reachability is read off the regime transition: a
  granular per-target mapping declares its key set, any coarse form reaches
  every regime; brute-force regimes may target DC-EGM regimes)
- numeric spot checks on small grid samples, outside jit: consumption
  recovery `post_decision ≈ resources - action`, resources non-decreasing in
  the Euler state, and `inverse_marginal_utility` consistent with
  `jax.grad(utility)` (round-trip `(u')⁻¹(u'(c)) ≈ c` plus strictly
  decreasing `u'`). Checks whose pruned DAG needs unknown leaves (free model
  parameters) are skipped — parameter values are not available at build time.

"""

import inspect
from collections.abc import Mapping
from typing import cast

import jax
import jax.numpy as jnp
from dags import concatenate_functions, get_ancestors

from _lcm.grids import ContinuousGrid, DiscreteGrid, Grid, IrregSpacedGrid
from _lcm.processes import _ContinuousStochasticProcess
from _lcm.typing import (
    ActionName,
    FunctionName,
    RegimeName,
    StateName,
    StateOrActionName,
)
from lcm.exceptions import GridInitializationError, ModelInitializationError
from lcm.phased import Phased
from lcm.regime import Regime as UserRegime
from lcm.solvers import DCEGM
from lcm.temporal_aggregation import H_linear
from lcm.transition import MarkovTransition
from lcm.typing import Float1D, FloatND, Int1D, IntND, ScalarFloat, UserFunction

# Shrink threshold of the node-resolution continuity spot check. Within one
# Euler grid cell, a function that is smooth at node resolution has
# quarter-cell increments of roughly a quarter of the neighboring cells'
# node-level increments (a derivative bound under two midpoint subdivisions),
# while a cliff's increment survives subdivision unshrunk. The threshold sits
# between the two regimes; the criterion is scale-invariant (ratios of the
# function's own increments, with only a float-noise floor).
_CONTINUITY_SHRINK_FACTOR = 0.4


def validate_dcegm_regimes(
    *,
    user_regimes: Mapping[RegimeName, UserRegime],
) -> None:
    """Validate the DC-EGM contract for every regime with a `DCEGM` solver.

    Args:
        user_regimes: Mapping of regime names to user-provided `Regime`
            instances.

    Raises:
        ModelInitializationError: If any regime with `solver=DCEGM(...)`
            violates the DC-EGM model contract.

    """
    for regime_name, user_regime in user_regimes.items():
        if isinstance(user_regime.solver, DCEGM):
            _validate_dcegm_regime(
                regime_name=regime_name,
                user_regime=user_regime,
                user_regimes=user_regimes,
            )


def savings_stage_reads_euler_state(*, user_regime: UserRegime, solver: DCEGM) -> bool:
    """Whether any savings-stage function reads the current Euler state.

    Runs the opaque-post-decision ancestor check (`Phased` resolved to the
    solve side, per-target cells unpacked, `MarkovTransition` weights
    unwrapped) over every savings-stage function: the Euler state's law, the
    regime transition, and the non-Euler state transitions. The kernel
    builder uses the result to switch to the per-exogenous-asset-node solve
    mode, where every Euler-state read is a per-combo constant. The single
    trigger keeps the kernel-mode dispatch in lockstep with the validation
    relaxation: a savings-stage Euler-state read is admitted exactly when
    the regime is solved per node.

    Args:
        user_regime: The user-provided `Regime` instance.
        solver: The regime's DC-EGM solver configuration.

    Returns:
        `True` when any savings-stage function variant has the Euler state
        among its DAG ancestors.

    """
    functions = _resolve_solve_functions(user_regime=user_regime)
    return bool(
        _savings_stage_euler_state_readers(
            user_regime=user_regime, functions=functions, solver=solver
        )
    )


def _validate_dcegm_regime(
    *,
    regime_name: RegimeName,
    user_regime: UserRegime,
    user_regimes: Mapping[RegimeName, UserRegime],
) -> None:
    """Run all DC-EGM contract checks for a single regime, in order."""
    solver = cast("DCEGM", user_regime.solver)

    _fail_if_terminal(regime_name=regime_name, user_regime=user_regime)
    _fail_if_state_action_classification_invalid(
        regime_name=regime_name, user_regime=user_regime, solver=solver
    )
    _fail_if_required_functions_missing(
        regime_name=regime_name, user_regime=user_regime, solver=solver
    )
    _fail_if_custom_H(regime_name=regime_name, user_regime=user_regime)

    functions = _resolve_solve_functions(user_regime=user_regime)

    _fail_if_post_decision_signature_invalid(
        regime_name=regime_name, functions=functions, solver=solver
    )
    _fail_if_constraint_touches_continuous_variables(
        regime_name=regime_name,
        user_regime=user_regime,
        functions=functions,
        solver=solver,
    )
    _fail_if_utility_depends_on_continuous_state(
        regime_name=regime_name, functions=functions, solver=solver
    )
    _fail_if_resources_depend_on_continuous_action(
        regime_name=regime_name, functions=functions, solver=solver
    )
    _fail_if_passive_state_invalid(
        regime_name=regime_name,
        user_regime=user_regime,
        functions=functions,
        solver=solver,
    )
    _fail_if_euler_transition_stochastic(
        regime_name=regime_name, user_regime=user_regime, solver=solver
    )
    _fail_if_euler_transition_bypasses_post_decision(
        regime_name=regime_name,
        user_regime=user_regime,
        functions=functions,
        solver=solver,
    )
    _fail_if_savings_stage_function_depends_on_decision(
        regime_name=regime_name,
        user_regime=user_regime,
        functions=functions,
        solver=solver,
    )
    _fail_if_grid_hygiene_violated(
        regime_name=regime_name, user_regime=user_regime, solver=solver
    )
    _fail_if_target_regime_incompatible(
        regime_name=regime_name,
        user_regime=user_regime,
        user_regimes=user_regimes,
        solver=solver,
    )
    try:
        _fail_if_numeric_spot_checks_fail(
            regime_name=regime_name,
            user_regime=user_regime,
            functions=functions,
            solver=solver,
        )
        _fail_if_savings_stage_function_jumps_in_euler_state(
            regime_name=regime_name,
            user_regime=user_regime,
            functions=functions,
            solver=solver,
        )
    except GridInitializationError as error:
        msg = (
            f"A numeric spot check of the DC-EGM contract in regime "
            f"'{regime_name}' needs grid points at model construction, but a "
            f"grid supplies them only at runtime: {error}"
        )
        raise ModelInitializationError(msg) from error


def _fail_if_terminal(*, regime_name: RegimeName, user_regime: UserRegime) -> None:
    """A terminal regime has nothing to solve, so a DCEGM solver is an error."""
    if user_regime.terminal:
        msg = (
            f"Regime '{regime_name}' is terminal but configured with the DCEGM "
            "solver. Terminal regimes have no optimization problem; remove the "
            "`solver=DCEGM(...)` setting."
        )
        raise ModelInitializationError(msg)


def _fail_if_state_action_classification_invalid(
    *,
    regime_name: RegimeName,
    user_regime: UserRegime,
    solver: DCEGM,
) -> None:
    """Require exactly one continuous (Euler) state and one continuous action.

    Process states are exempt: they enter the value function as node-valued
    discrete dimensions. Other continuous states are allowed as *passive*
    states; `_fail_if_passive_state_invalid` verifies their passivity once
    the regime's solve functions are resolved.
    """
    continuous_states = _continuous_non_process_names(
        grids=_solve_grids(slot=user_regime.states)
    )
    continuous_actions = _continuous_non_process_names(
        grids=_solve_grids(slot=user_regime.actions)
    )

    if solver.continuous_state not in continuous_states:
        msg = (
            f"DCEGM `continuous_state` '{solver.continuous_state}' is not a "
            f"continuous state of regime '{regime_name}'. Continuous "
            f"(non-process) states: {continuous_states}."
        )
        raise ModelInitializationError(msg)

    if solver.continuous_action not in continuous_actions:
        msg = (
            f"DCEGM `continuous_action` '{solver.continuous_action}' is not a "
            f"continuous action of regime '{regime_name}'. Continuous "
            f"actions: {continuous_actions}."
        )
        raise ModelInitializationError(msg)

    extra_actions = [a for a in continuous_actions if a != solver.continuous_action]
    if extra_actions:
        msg = (
            f"Regime '{regime_name}' has continuous actions {extra_actions} in "
            f"addition to the DCEGM continuous action "
            f"'{solver.continuous_action}'. The DCEGM solver supports exactly "
            "one continuous action; further actions must be discrete."
        )
        raise ModelInitializationError(msg)


def _fail_if_required_functions_missing(
    *,
    regime_name: RegimeName,
    user_regime: UserRegime,
    solver: DCEGM,
) -> None:
    """Require the post-decision and resources functions.

    `inverse_marginal_utility` is *optional*: when a regime omits it, EGM derives
    a numerical inverse from `utility` (the iEGM path), so it is not required here.
    When supplied, it is validated against `jax.grad(utility)` separately.
    """
    required: dict[FunctionName, str] = {
        solver.post_decision_function: (
            "the post-decision function (`DCEGM.post_decision_function`)"
        ),
        solver.resources: "the resources function (`DCEGM.resources`)",
    }
    missing = [
        f"'{name}' — {role}"
        for name, role in required.items()
        if name not in user_regime.functions
    ]
    if missing:
        msg = (
            f"Regime '{regime_name}' uses the DCEGM solver but is missing "
            f"required entries in `functions`: {'; '.join(missing)}."
        )
        raise ModelInitializationError(msg)


def _fail_if_custom_H(*, regime_name: RegimeName, user_regime: UserRegime) -> None:
    """Require the default Bellman aggregator `H` at solve time.

    The Euler inversion hard-codes `H = utility + discount_factor * E[V']`, so a
    custom *solve-phase* `H` would silently change the meaning of the solution.
    A `Phased` `H` whose solve variant is the default aggregator is accepted —
    DC-EGM never reads the simulate variant, so a naive present-bias regime
    (`H = Phased(solve=H_linear, simulate=beta_delta_H)`) is admissible: the
    present bias enters only the simulate-phase re-optimization, outside the
    Euler inversion.
    """
    raw_H = user_regime.functions.get("H")
    solve_H = raw_H.solve if isinstance(raw_H, Phased) else raw_H
    if solve_H is not H_linear:
        msg = (
            f"Regime '{regime_name}' defines a custom solve-phase Bellman "
            "aggregator `H`. The DCEGM solver hard-codes the default aggregator "
            "`H = utility + discount_factor * E[V']` at solve time; remove the "
            "custom `H` (a `Phased` `H` whose solve variant is `H_linear` is "
            "accepted) or use the brute-force solver."
        )
        raise ModelInitializationError(msg)


def _fail_if_post_decision_signature_invalid(
    *,
    regime_name: RegimeName,
    functions: dict[FunctionName, UserFunction],
    solver: DCEGM,
) -> None:
    """The post-decision function consumes the action and the resources function."""
    arg_names = list(
        inspect.signature(functions[solver.post_decision_function]).parameters
    )
    if solver.continuous_action not in arg_names or solver.resources not in arg_names:
        msg = (
            f"The post-decision function '{solver.post_decision_function}' of "
            f"regime '{regime_name}' must take the continuous action "
            f"'{solver.continuous_action}' and the resources function "
            f"'{solver.resources}' as arguments; its arguments are "
            f"{arg_names}."
        )
        raise ModelInitializationError(msg)
    if solver.continuous_state in arg_names:
        msg = (
            f"The post-decision function '{solver.post_decision_function}' of "
            f"regime '{regime_name}' must not depend on the continuous state "
            f"'{solver.continuous_state}' directly; the state enters only "
            f"through the resources function '{solver.resources}'."
        )
        raise ModelInitializationError(msg)


def _fail_if_constraint_touches_continuous_variables(
    *,
    regime_name: RegimeName,
    user_regime: UserRegime,
    functions: dict[FunctionName, UserFunction],
    solver: DCEGM,
) -> None:
    """No constraint may touch the continuous state or action.

    EGM enforces the budget identity and the borrowing limit
    (`savings_grid.start`) intrinsically; discrete-only constraints remain
    supported.
    """
    forbidden = {solver.continuous_state, solver.continuous_action}
    for constraint_name, constraint_func in user_regime.constraints.items():
        # Constraints are phase-invariant by the slot grammar (`Phased` is
        # rejected there), so the value is always a bare callable.
        ancestors = _dag_ancestors(
            functions=functions,
            target_func=cast("UserFunction", constraint_func),
        )
        bad = sorted(ancestors & forbidden)
        if bad:
            msg = (
                f"The constraint '{constraint_name}' of regime '{regime_name}' "
                f"depends on the continuous variables {bad}. The DCEGM solver "
                "enforces the budget identity and the borrowing limit "
                "(`savings_grid.start`) intrinsically; only constraints on "
                "discrete variables are supported. Remove this constraint."
            )
            raise ModelInitializationError(msg)


def _fail_if_utility_depends_on_continuous_state(
    *,
    regime_name: RegimeName,
    functions: dict[FunctionName, UserFunction],
    solver: DCEGM,
) -> None:
    """`utility` must reach the state only through the action (envelope condition)."""
    ancestors = get_ancestors(functions, targets=["utility"], include_targets=False)
    if solver.continuous_state in ancestors:
        msg = (
            f"The utility function of regime '{regime_name}' depends on the "
            f"continuous state '{solver.continuous_state}'. The DCEGM envelope "
            "condition requires utility to reach the state only through the "
            f"continuous action '{solver.continuous_action}'."
        )
        raise ModelInitializationError(msg)


def _fail_if_resources_depend_on_continuous_action(
    *,
    regime_name: RegimeName,
    functions: dict[FunctionName, UserFunction],
    solver: DCEGM,
) -> None:
    """Resources are pre-decision: they must not depend on the continuous action."""
    ancestors = get_ancestors(
        functions, targets=[solver.resources], include_targets=False
    )
    if solver.continuous_action in ancestors:
        msg = (
            f"The resources function '{solver.resources}' of regime "
            f"'{regime_name}' depends on the continuous action "
            f"'{solver.continuous_action}'. Resources are what the continuous "
            "action is paid out of and must be known before the action is "
            "chosen."
        )
        raise ModelInitializationError(msg)


def _fail_if_passive_state_invalid(
    *,
    regime_name: RegimeName,
    user_regime: UserRegime,
    functions: dict[FunctionName, UserFunction],
    solver: DCEGM,
) -> None:
    """Every non-Euler continuous state must be passive.

    A passive state rides along as a grid axis of the value function and the
    EGM carry; its next value is computed at the savings-node stage and read
    from the child carry by interpolation on the child's passive grid. That
    requires, per passive state:

    - a deterministic transition (stochastic continuous dynamics belong in a
      process state),
    - transition DAG ancestors excluding the continuous action, the
      resources function, and the post-decision function (none of which are
      known at the savings-node stage). The current Euler state is an
      allowed ancestor: the kernel then solves per exogenous asset node,
      where the state's value is known (the read must be continuous at node
      resolution — checked by the savings-stage continuity spot check),
    - a grid that is not distributed; `batch_size` is honored (it splays the
      passive state's combo axis via productmap to shed memory), while a
      continuous axis cannot be sharded.
    """
    passive_names = [
        name
        for name in _continuous_non_process_names(
            grids=_solve_grids(slot=user_regime.states)
        )
        if name != solver.continuous_state
    ]
    opaque_functions = _without(
        functions=functions, names={solver.post_decision_function, solver.resources}
    )
    forbidden = {
        solver.continuous_action,
        solver.resources,
        solver.post_decision_function,
    }
    for state_name in passive_names:
        value = user_regime.state_transitions.get(state_name)
        if value is None:
            # Transition coverage is validated when the effective regimes are
            # built; a missing entry gets its own error there.
            continue
        is_stochastic = isinstance(value, MarkovTransition) or (
            isinstance(value, Mapping)
            and any(isinstance(v, MarkovTransition) for v in value.values())
        )
        if is_stochastic:
            msg = (
                f"The transition of the continuous state '{state_name}' in "
                f"regime '{regime_name}' is stochastic. A non-Euler continuous "
                "state in a DCEGM regime must be passive (deterministic); use "
                "a stochastic process state (e.g. Rouwenhorst or Tauchen) for "
                "stochastic continuous dynamics."
            )
            raise ModelInitializationError(msg)
        for label, transition_func in _transition_variants(value=value):
            ancestors = _dag_ancestors(
                functions=opaque_functions, target_func=transition_func
            )
            bad = sorted(ancestors & forbidden)
            if bad:
                msg = (
                    f"The continuous state '{state_name}' of regime "
                    f"'{regime_name}' is not passive: its transition{label} "
                    f"depends on {bad}. A passive continuous state's "
                    "transition must not depend on the continuous action "
                    f"'{solver.continuous_action}', the resources function "
                    f"'{solver.resources}', or the post-decision function "
                    f"'{solver.post_decision_function}' — those values are "
                    "unknown until the EGM step has run. (Reading the Euler "
                    f"state '{solver.continuous_state}' is allowed: the "
                    "kernel then solves per exogenous asset node.)"
                )
                raise ModelInitializationError(msg)
        grid = cast("ContinuousGrid", user_regime.states[state_name])
        # `batch_size` on a passive state splays its combo axis (via productmap)
        # to shed memory; `distributed` stays rejected (a continuous axis
        # cannot be sharded).
        if grid.distributed:
            msg = (
                f"The grid of the passive continuous state '{state_name}' in "
                f"regime '{regime_name}' must not be distributed in a DCEGM "
                f"regime (got distributed={grid.distributed})."
            )
            raise ModelInitializationError(msg)


def _fail_if_euler_transition_stochastic(
    *,
    regime_name: RegimeName,
    user_regime: UserRegime,
    solver: DCEGM,
) -> None:
    """The Euler state's transition must be deterministic."""
    value = user_regime.state_transitions.get(solver.continuous_state)
    is_stochastic = isinstance(value, MarkovTransition) or (
        isinstance(value, Mapping)
        and any(isinstance(v, MarkovTransition) for v in value.values())
    )
    if is_stochastic:
        msg = (
            f"The transition of the Euler state '{solver.continuous_state}' in "
            f"regime '{regime_name}' is stochastic. The DCEGM solver requires "
            "a deterministic continuous-state transition (stochastic discrete "
            "transitions and process states are fully supported)."
        )
        raise ModelInitializationError(msg)


def _fail_if_euler_transition_bypasses_post_decision(
    *,
    regime_name: RegimeName,
    user_regime: UserRegime,
    functions: dict[FunctionName, UserFunction],
    solver: DCEGM,
) -> None:
    """The Euler transition consumes the post-decision function.

    With the post-decision and resources functions removed from the DAG,
    their names become leaf inputs, so the ancestor set reveals whether the
    transition reaches the continuous action or the resources function
    through any other channel. The current Euler state is an allowed
    transition ancestor: the kernel then solves per exogenous asset node,
    where the residual is a per-combo constant
    (`savings_stage_reads_euler_state` reports this mode).
    """
    bypass_msg = (
        f"The transition of the Euler state '{solver.continuous_state}' in "
        f"regime '{regime_name}' must consume the post-decision function "
        f"'{solver.post_decision_function}' and reach "
        f"'{solver.continuous_action}' only through it."
    )
    value = user_regime.state_transitions.get(solver.continuous_state)
    if value is None:
        msg = (
            f"{bypass_msg} It is declared as a fixed state (identity "
            "transition), which bypasses the post-decision function."
        )
        raise ModelInitializationError(msg)

    opaque_functions = _without(
        functions=functions, names={solver.post_decision_function, solver.resources}
    )
    forbidden = {
        solver.continuous_action,
        solver.resources,
    }
    for label, transition_func in _transition_variants(value=value):
        ancestors = _dag_ancestors(
            functions=opaque_functions,
            target_func=transition_func,
        )
        bad = sorted(ancestors & forbidden)
        if solver.post_decision_function not in ancestors or bad:
            msg = (
                f"{bypass_msg} The transition{label} has DAG ancestors "
                f"{sorted(ancestors)}; forbidden direct dependencies: {bad}."
            )
            raise ModelInitializationError(msg)


def _fail_if_savings_stage_function_depends_on_decision(
    *,
    regime_name: RegimeName,
    user_regime: UserRegime,
    functions: dict[FunctionName, UserFunction],
    solver: DCEGM,
) -> None:
    """Savings-node-stage functions must not depend on the current decision.

    At a savings node, the continuous action and the resources are unknown
    (they are EGM outputs). The regime transition, every stochastic
    transition weight, and every non-Euler state transition must therefore
    be independent of the continuous action, the resources function, and
    the post-decision function. Reading the current Euler state is allowed:
    the kernel then solves per exogenous asset node
    (`savings_stage_reads_euler_state` reports this mode), where the read
    is a per-combo constant — its continuity at node resolution is checked
    by `_fail_if_savings_stage_function_jumps_in_euler_state`.
    """
    opaque_functions = _without(
        functions=functions, names={solver.post_decision_function, solver.resources}
    )
    forbidden = {
        solver.continuous_action,
        solver.resources,
        solver.post_decision_function,
    }
    for role, label, func in _savings_stage_candidates(
        user_regime=user_regime, solver=solver
    ):
        if role == "euler_law":
            # The Euler state's own law has its dedicated structural check
            # (`_fail_if_euler_transition_bypasses_post_decision`).
            continue
        ancestors = _dag_ancestors(functions=opaque_functions, target_func=func)
        bad = sorted(ancestors & forbidden)
        if bad:
            msg = (
                f"The {label} of regime '{regime_name}' depends on {bad}. "
                "Functions evaluated at the savings-node stage (regime "
                "transition probabilities, stochastic transition weights, and "
                "non-Euler state transitions) must not depend on the "
                f"continuous action '{solver.continuous_action}', the "
                f"resources function '{solver.resources}', or the "
                f"post-decision function '{solver.post_decision_function}' — "
                "those values are unknown until the EGM step has run."
            )
            raise ModelInitializationError(msg)


def _savings_stage_candidates(
    *,
    user_regime: UserRegime,
    solver: DCEGM,
) -> list[tuple[str, str, UserFunction]]:
    """Enumerate every savings-stage function variant of a regime.

    Coarse, `MarkovTransition`-wrapped, `Phased`, and granular per-target
    forms all unpack to plain callables via `_transition_variants`.

    Returns:
        List of `(role, label, func)` triples, with role one of
        `"euler_law"`, `"regime_transition"`, and `"state_transition"`.

    """
    candidates: list[tuple[str, str, UserFunction]] = []
    euler_value = user_regime.state_transitions.get(solver.continuous_state)
    if euler_value is not None:
        for label, transition_func in _transition_variants(value=euler_value):
            candidates.append(
                (
                    "euler_law",
                    f"transition of the Euler state '{solver.continuous_state}'{label}",
                    transition_func,
                )
            )
    if user_regime.transition is not None:
        for label, regime_transition in _transition_variants(
            value=user_regime.transition
        ):
            candidates.append(
                (
                    "regime_transition",
                    f"regime transition function{label}",
                    regime_transition,
                )
            )
    for state_name, value in user_regime.state_transitions.items():
        if state_name == solver.continuous_state or value is None:
            continue
        for label, transition_func in _transition_variants(value=value):
            candidates.append(
                (
                    "state_transition",
                    f"transition of state '{state_name}'{label}",
                    transition_func,
                )
            )
    return candidates


def _savings_stage_euler_state_readers(
    *,
    user_regime: UserRegime,
    functions: dict[FunctionName, UserFunction],
    solver: DCEGM,
) -> list[tuple[str, str, UserFunction]]:
    """Savings-stage function variants whose DAG ancestors include the Euler state.

    The post-decision and resources functions are opaque (removed from the
    DAG), so only direct decision-time reads of the Euler state count.

    Returns:
        List of `(role, label, func)` triples, filtered from
        `_savings_stage_candidates`.

    """
    opaque_functions = _without(
        functions=functions, names={solver.post_decision_function, solver.resources}
    )
    return [
        (role, label, func)
        for role, label, func in _savings_stage_candidates(
            user_regime=user_regime, solver=solver
        )
        if solver.continuous_state
        in _dag_ancestors(functions=opaque_functions, target_func=func)
    ]


def _fail_if_grid_hygiene_violated(
    *,
    regime_name: RegimeName,
    user_regime: UserRegime,
    solver: DCEGM,
) -> None:
    """Reject runtime-supplied points and distributed grids; savings grid covers
    the Euler grid.

    `batch_size` is honored on the Euler, savings, and discrete-state grids (it
    only splays combo/node axes to shed memory); it is rejected on discrete
    actions, whose logsum aggregation needs every action value at once.
    """
    # Rule 1 has already established that the Euler state's grid is a
    # (non-process) continuous grid.
    euler_grid = cast("ContinuousGrid", user_regime.states[solver.continuous_state])
    # DC-EGM builds its kernels (savings nodes, carry shapes, numeric spot
    # checks) at model-construction time, so these grids must carry their
    # points then — runtime-supplied points cannot work.
    kernel_grids = {
        "DCEGM savings grid": solver.savings_grid,
        f"grid of the Euler state '{solver.continuous_state}'": euler_grid,
        f"grid of the continuous action '{solver.continuous_action}'": (
            user_regime.actions[solver.continuous_action]
        ),
    }
    for role, grid in kernel_grids.items():
        if isinstance(grid, IrregSpacedGrid) and grid.pass_points_at_runtime:
            msg = (
                f"The {role} in regime '{regime_name}' supplies its points "
                "only at runtime via params; a DCEGM regime needs the points "
                "at model construction. Supply them via `points=...`."
            )
            raise ModelInitializationError(msg)
    # `batch_size` on a discrete state splays its combo axis: the per-combo
    # solve runs in `productmap` blocks (per-axis `lax.map`) reassembled into
    # the whole combo product before the carry is built, so carry rows still
    # carry whole discrete axes. `distributed` stays rejected — the kernel
    # selects child carry rows by integer indexing along whole discrete axes,
    # which a sharded (per-device slice) axis would break.
    for name, grid in user_regime.states.items():
        if isinstance(grid, DiscreteGrid) and grid.distributed:
            msg = (
                f"The grid of the discrete state '{name}' in regime "
                f"'{regime_name}' must not be distributed in a DCEGM regime "
                f"(got distributed={grid.distributed})."
            )
            raise ModelInitializationError(msg)
    # Discrete actions cannot be batched or distributed: the discrete-action
    # aggregation (logsum over the action axes) needs every action value at
    # once, so its axis is never split.
    for name, grid in user_regime.actions.items():
        if isinstance(grid, DiscreteGrid) and (
            grid.batch_size != 0 or grid.distributed
        ):
            msg = (
                f"The grid of the discrete action '{name}' in regime "
                f"'{regime_name}' must not be batched or distributed in a "
                f"DCEGM regime (got batch_size={grid.batch_size}, "
                f"distributed={grid.distributed})."
            )
            raise ModelInitializationError(msg)
    # `batch_size` on the Euler grid is honored: it splays the per-asset-node
    # solve into blocks (`lax.map`) to shed peak working-set memory, leaving the
    # value function unchanged. `distributed` remains disallowed — a continuous
    # axis cannot be sharded (rejected at grid construction).
    if euler_grid.distributed:
        msg = (
            f"The grid of the Euler state '{solver.continuous_state}' in "
            f"regime '{regime_name}' must not be distributed in a DCEGM regime "
            f"(got distributed={euler_grid.distributed})."
        )
        raise ModelInitializationError(msg)
    # `batch_size` on the savings grid is honored: it splays the per-savings-node
    # continuation computation into blocks (`lax.map`) to shed the dominant
    # egm_step working buffer, leaving the value function unchanged. `distributed`
    # remains disallowed — a continuous axis cannot be sharded.
    if solver.savings_grid.distributed:
        msg = (
            f"The DCEGM savings grid of regime '{regime_name}' must not be "
            f"distributed (got distributed={solver.savings_grid.distributed})."
        )
        raise ModelInitializationError(msg)
    savings_max = float(jnp.max(solver.savings_grid.to_jax()))
    euler_max = float(jnp.max(euler_grid.to_jax()))
    if savings_max < euler_max:
        msg = (
            f"The DCEGM savings grid of regime '{regime_name}' ends at "
            f"{savings_max}, below the upper end {euler_max} of the "
            f"'{solver.continuous_state}' grid. An undersized savings grid "
            "silently edge-clamps the endogenous grid at high values of "
            f"'{solver.continuous_state}'; extend `savings_grid`."
        )
        raise ModelInitializationError(msg)


def _fail_if_target_regime_incompatible(
    *,
    regime_name: RegimeName,
    user_regime: UserRegime,
    user_regimes: Mapping[RegimeName, UserRegime],
    solver: DCEGM,
) -> None:
    """Reachable non-terminal targets must be DCEGM regimes with the same state.

    Terminal targets are always allowed, and brute-force regimes may target
    DC-EGM regimes (they only consume the target's value-function array).
    """
    for target_name in sorted(
        _reachable_target_names(user_regime=user_regime, user_regimes=user_regimes)
    ):
        target = user_regimes[target_name]
        if target.terminal:
            continue
        if not isinstance(target.solver, DCEGM):
            msg = (
                f"Regime '{regime_name}' uses the DCEGM solver and can reach "
                f"the non-terminal regime '{target_name}', which uses "
                f"{type(target.solver).__name__}. Every reachable non-terminal "
                "target of a DCEGM regime must itself use the DCEGM solver "
                "(brute-force regimes may target DCEGM regimes, not the other "
                "way around)."
            )
            raise ModelInitializationError(msg)
        if target.solver.continuous_state != solver.continuous_state:
            msg = (
                f"Regime '{regime_name}' uses the DCEGM solver with Euler "
                f"state '{solver.continuous_state}' but can reach regime "
                f"'{target_name}', whose DCEGM Euler state is "
                f"'{target.solver.continuous_state}'. All mutually reachable "
                "DCEGM regimes must share the same Euler continuous state."
            )
            raise ModelInitializationError(msg)


def _fail_if_numeric_spot_checks_fail(
    *,
    regime_name: RegimeName,
    user_regime: UserRegime,
    functions: dict[FunctionName, UserFunction],
    solver: DCEGM,
) -> None:
    """Numeric spot checks on small grid samples, evaluated eagerly (no jit).

    - resources non-decreasing in the Euler state,
    - consumption recovery `post_decision ≈ resources - action`,
    - `inverse_marginal_utility` consistent with `jax.grad(utility)`:
      round-trip `(u')⁻¹(u'(c)) ≈ c`, and `u'` strictly decreasing.

    Any check whose pruned DAG requires inputs that are neither states nor
    actions (i.e. free model parameters) is skipped: parameter values are
    not available at build time.
    """
    x64_enabled = bool(jax.config.read("jax_enable_x64"))
    atol = 1e-8 if x64_enabled else 1e-4
    rtol = 1e-6 if x64_enabled else 1e-3

    grids: dict[StateOrActionName, Grid] = {
        **_solve_grids(slot=user_regime.states),
        **_solve_grids(slot=user_regime.actions),
    }
    euler_sample = _grid_sample(grid=grids[solver.continuous_state])
    action_sample = _grid_sample(grid=grids[solver.continuous_action])
    n_sample = min(euler_sample.shape[0], action_sample.shape[0])

    resources_func = concatenate_functions(functions, targets=solver.resources)
    resources_kwargs = _fixed_kwargs(
        func=resources_func,
        grids=grids,
        varied={solver.continuous_state},
    )
    if resources_kwargs is not None:
        resources_at = [
            _call_with_varied(
                func=resources_func,
                fixed=resources_kwargs,
                varied={solver.continuous_state: w},
            )
            for w in euler_sample
        ]
        # Asset-row mode (a savings-stage function reads the Euler state)
        # publishes the carry marginal by dividing by the resources slope
        # `dR/dA`, so a flat (zero-slope) resources region there would make the
        # marginal non-finite. Require strictly increasing resources in that
        # mode, not merely non-decreasing.
        require_strict = savings_stage_reads_euler_state(
            user_regime=user_regime, solver=solver
        )

        def _violates(
            r_lo: float, r_hi: float, *, strict: bool = require_strict
        ) -> bool:
            return r_hi <= r_lo + atol if strict else r_hi < r_lo - atol

        bad = [
            (float(w_lo), float(w_hi))
            for w_lo, w_hi, r_lo, r_hi in zip(
                euler_sample[:-1],
                euler_sample[1:],
                resources_at[:-1],
                resources_at[1:],
                strict=True,
            )
            if _violates(float(r_lo), float(r_hi))
        ]
        if bad:
            requirement = "strictly increasing" if require_strict else "non-decreasing"
            reason = (
                " This regime reads the Euler state at the savings stage, so the "
                "carry marginal divides by the resources slope — a flat region "
                "would make it non-finite."
                if require_strict
                else ""
            )
            msg = (
                f"The resources function '{solver.resources}' of regime "
                f"'{regime_name}' must be {requirement} in "
                f"'{solver.continuous_state}'; it violates that on the sample "
                f"intervals {bad}.{reason}"
            )
            raise ModelInitializationError(msg)

        post_func = concatenate_functions(
            functions, targets=solver.post_decision_function
        )
        post_kwargs = _fixed_kwargs(
            func=post_func,
            grids=grids,
            varied={solver.continuous_state, solver.continuous_action},
        )
        if post_kwargs is not None:
            for w, c in zip(
                euler_sample[:n_sample], action_sample[:n_sample], strict=True
            ):
                resources_value = _call_with_varied(
                    func=resources_func,
                    fixed=resources_kwargs,
                    varied={solver.continuous_state: w},
                )
                post_value = _call_with_varied(
                    func=post_func,
                    fixed=post_kwargs,
                    varied={
                        solver.continuous_state: w,
                        solver.continuous_action: c,
                    },
                )
                expected = resources_value - c
                if not _isclose(
                    actual=post_value, expected=expected, rtol=rtol, atol=atol
                ):
                    msg = (
                        f"Consumption recovery fails in regime '{regime_name}': "
                        f"the post-decision function must satisfy "
                        f"`{solver.post_decision_function} = {solver.resources} "
                        f"- {solver.continuous_action}`. At "
                        f"{solver.continuous_state}={float(w)}, "
                        f"{solver.continuous_action}={float(c)}: "
                        f"{solver.post_decision_function}={float(post_value)} "
                        f"but {solver.resources} - {solver.continuous_action}="
                        f"{float(expected)}."
                    )
                    raise ModelInitializationError(msg)

    _fail_if_inverse_marginal_utility_inconsistent(
        regime_name=regime_name,
        functions=functions,
        solver=solver,
        grids=grids,
        action_sample=action_sample,
        rtol=rtol,
        atol=atol,
    )


def _fail_if_inverse_marginal_utility_inconsistent(
    *,
    regime_name: RegimeName,
    functions: dict[FunctionName, UserFunction],
    solver: DCEGM,
    grids: dict[StateOrActionName, Grid],
    action_sample: Float1D,
    rtol: float,
    atol: float,
) -> None:
    """Check `(u')⁻¹(u'(c)) ≈ c` and strict monotonicity of `u'` on a sample.

    Skipped when the regime supplies no `inverse_marginal_utility`: the iEGM path
    inverts `u'` numerically, so there is no analytic inverse to check against.
    """
    if "inverse_marginal_utility" not in functions:
        return
    inverse_func = concatenate_functions(functions, targets="inverse_marginal_utility")
    inverse_arg_names = list(inspect.signature(inverse_func).parameters)
    if "marginal_continuation" not in inverse_arg_names:
        msg = (
            f"The function 'inverse_marginal_utility' of regime "
            f"'{regime_name}' must take an argument named "
            f"'marginal_continuation' (the marginal continuation value to "
            f"invert); its arguments are {inverse_arg_names}."
        )
        raise ModelInitializationError(msg)

    utility_func = concatenate_functions(functions, targets="utility")
    if solver.continuous_action not in inspect.signature(utility_func).parameters:
        return
    # Run the concavity and round-trip checks at a few discrete-combo contexts,
    # not just the grid's lower corner: a utility (or inverse) that reads a
    # discrete/passive state as a parameter can be consistent at one combo and
    # wrong at another. The contexts are index-aligned, so the j-th utility and
    # inverse contexts bind shared arguments to the same grid points.
    utility_contexts = _combo_contexts(
        func=utility_func, grids=grids, varied={solver.continuous_action}
    )
    inverse_contexts = _combo_contexts(
        func=inverse_func, grids=grids, varied={"marginal_continuation"}
    )
    if not utility_contexts or not inverse_contexts:
        return

    for utility_kwargs, inverse_kwargs in zip(
        utility_contexts, inverse_contexts, strict=True
    ):

        def utility_of_action(
            action_value: ScalarFloat,
            _kwargs: dict[str, object] = utility_kwargs,
        ) -> ScalarFloat:
            return utility_func(**_kwargs, **{solver.continuous_action: action_value})

        marginal_utility = [jax.grad(utility_of_action)(c) for c in action_sample]

        non_decreasing = [
            (float(c_lo), float(c_hi))
            for c_lo, c_hi, mu_lo, mu_hi in zip(
                action_sample[:-1],
                action_sample[1:],
                marginal_utility[:-1],
                marginal_utility[1:],
                strict=True,
            )
            if float(mu_hi) >= float(mu_lo)
        ]
        if non_decreasing:
            msg = (
                f"The marginal utility of '{solver.continuous_action}' in regime "
                f"'{regime_name}' (computed via `jax.grad` of the utility DAG) "
                "must be strictly decreasing — DCEGM requires strictly concave "
                f"utility. It fails to decrease on the sample intervals "
                f"{non_decreasing}."
            )
            raise ModelInitializationError(msg)

        for c, mu in zip(action_sample, marginal_utility, strict=True):
            recovered = inverse_func(**inverse_kwargs, marginal_continuation=mu)
            if not _isclose(actual=recovered, expected=c, rtol=rtol, atol=atol):
                msg = (
                    f"'inverse_marginal_utility' of regime '{regime_name}' is "
                    "inconsistent with `jax.grad` of the utility DAG: the "
                    f"round-trip `(u')⁻¹(u'(c))` at "
                    f"{solver.continuous_action}={float(c)} yields "
                    f"{float(recovered)} (marginal utility {float(mu)})."
                )
                raise ModelInitializationError(msg)


def _fail_if_savings_stage_function_jumps_in_euler_state(
    *,
    regime_name: RegimeName,
    user_regime: UserRegime,
    functions: dict[FunctionName, UserFunction],
    solver: DCEGM,
) -> None:
    """Savings-stage reads of the Euler state must be continuous at node resolution.

    The per-exogenous-asset-node solve evaluates every savings-stage
    Euler-state read at the grid nodes and publishes per-node results, which
    is exact only when the read is smooth at the resolution of the Euler
    grid. A jump in the Euler state's own law additionally makes the child's
    value function discontinuous, so the true policy bunches next-period
    wealth exactly at the discontinuity — a corner where the Euler equation
    does not hold, so EGM's candidate families (interior Euler inversions
    plus the closed-form credit-constrained segment) structurally exclude
    the optimum. Offending functions are rejected at build time rather than
    solved approximately.

    Numeric spot check: every savings-stage variant whose DAG ancestors
    include the Euler state is evaluated over the Euler grid nodes plus two
    levels of cell-midpoint refinement (the Euler law at a fixed savings
    value), in a few discrete-combo contexts sampled from the grids.
    `_find_jump_at_node_resolution` flags cells whose refined increments do
    not shrink under subdivision — a smooth band with dedicated nodes across
    it passes, a band narrower than one grid cell is a cliff at node
    resolution. Variants whose pruned DAG needs free model parameters are
    skipped (their values are unknown at build time).
    """
    x64_enabled = bool(jax.config.read("jax_enable_x64"))
    base_atol = 1e-8 if x64_enabled else 1e-4

    grids: dict[StateOrActionName, Grid] = {
        **_solve_grids(slot=user_regime.states),
        **_solve_grids(slot=user_regime.actions),
    }
    euler_points = jnp.sort(grids[solver.continuous_state].to_jax())
    euler_sample = _node_resolution_sample(grid_points=euler_points)
    savings_points = solver.savings_grid.to_jax()
    savings_value = savings_points[savings_points.shape[0] // 2]

    functions_without_post = _without(
        functions=functions, names={solver.post_decision_function}
    )
    for role, label, func in _savings_stage_euler_state_readers(
        user_regime=user_regime, functions=functions, solver=solver
    ):
        target_name = "__dcegm_validation_target__"
        law_func = concatenate_functions(
            functions={**functions_without_post, target_name: func},
            targets=target_name,
        )
        varied = {solver.continuous_state, solver.post_decision_function}
        for context in _combo_contexts(func=law_func, grids=grids, varied=varied):
            law_values = _law_values_on_sample(
                law_func=law_func,
                context=context,
                euler_state_name=solver.continuous_state,
                post_decision_name=solver.post_decision_function,
                savings_value=savings_value,
                euler_sample=euler_sample,
            )
            jump_location = _find_jump_at_node_resolution(
                grid_points=euler_points, values=law_values, atol=base_atol
            )
            if jump_location is not None:
                consequence = (
                    (
                        "A jump in the law makes the child's value function "
                        "discontinuous, and the true policy bunches "
                        f"next-period '{solver.continuous_state}' exactly at "
                        "the discontinuity — a corner where the Euler "
                        "equation does not hold, which EGM's candidate set "
                        "cannot represent."
                    )
                    if role == "euler_law"
                    else (
                        "Savings-stage reads of the Euler state are "
                        "evaluated per exogenous asset node, which is exact "
                        "only when they are smooth at the resolution of the "
                        f"'{solver.continuous_state}' grid."
                    )
                )
                msg = (
                    f"The {label} in regime '{regime_name}' reads the "
                    f"current Euler state '{solver.continuous_state}' but is "
                    "discontinuous in it at the resolution of the "
                    f"'{solver.continuous_state}' grid: its value jumps near "
                    f"{solver.continuous_state} ≈ {jump_location}. "
                    f"{consequence} If the function is a continuous band "
                    "steeper than the grid resolves, add grid nodes across "
                    "the band; otherwise make it continuous in "
                    f"'{solver.continuous_state}' (kinks are fine), e.g. by "
                    "phasing the term out instead of cutting it off."
                )
                raise ModelInitializationError(msg)


def _node_resolution_sample(*, grid_points: Float1D) -> Float1D:
    """Grid nodes plus two levels of cell-midpoint refinement (quarter points).

    Args:
        grid_points: Sorted 1d grid points (`n` nodes).

    Returns:
        Sorted sample of length `4 * (n - 1) + 1`: every node plus the
        quarter points of every cell.

    """
    left = grid_points[:-1]
    right = grid_points[1:]
    offsets = jnp.asarray([0.0, 0.25, 0.5, 0.75])
    refined = (left[:, None] + (right - left)[:, None] * offsets[None, :]).reshape(-1)
    return jnp.concatenate([refined, grid_points[-1:]])


def _find_jump_at_node_resolution(
    *,
    grid_points: Float1D,
    values: FloatND | IntND,
    atol: float,
) -> float | None:
    """Approximate location of the first cell with a sub-node-resolution jump.

    `values` are the function's values on `_node_resolution_sample` of
    `grid_points` (vector-valued outputs allowed; increments are reduced
    with the maximum over output components). For each grid cell, the
    maximum quarter-cell increment is compared against the node-level
    increments of the cell and its neighbors: a function that is smooth at
    node resolution shrinks like a derivative bound under two midpoint
    subdivisions (quarter-cell increments roughly a quarter of the
    neighborhood's node-level increments), while a cliff's increment — or a
    band without dedicated nodes across it — survives subdivision unshrunk.
    The criterion is scale-invariant up to a float-noise floor.

    Returns:
        Midpoint of the first offending cell, or `None` when every cell's
        refined increments shrink as a smooth function's would.

    """
    flat = jnp.asarray(values).reshape(values.shape[0], -1)
    quarter_increments = jnp.max(jnp.abs(jnp.diff(flat, axis=0)), axis=1)
    node_values = flat[::4]
    node_increments = jnp.max(jnp.abs(jnp.diff(node_values, axis=0)), axis=1)
    n_cells = int(grid_points.shape[0]) - 1
    max_quarter_per_cell = quarter_increments.reshape(n_cells, 4).max(axis=1)
    # Edge-pad so the first and last cells compare against their available
    # neighbors only.
    padded = jnp.concatenate(
        [node_increments[:1], node_increments, node_increments[-1:]]
    )
    neighborhood = jnp.maximum(jnp.maximum(padded[:-2], padded[1:-1]), padded[2:])
    scale = jnp.max(jnp.abs(jnp.where(jnp.isfinite(flat), flat, 0.0)))
    noise_floor = atol * (1.0 + scale)
    jumps = max_quarter_per_cell > (
        _CONTINUITY_SHRINK_FACTOR * neighborhood + noise_floor
    )
    if not bool(jnp.any(jumps)):
        return None
    index = int(jnp.argmax(jumps))
    return float(0.5 * (grid_points[index] + grid_points[index + 1]))


def _law_values_on_sample(
    *,
    law_func: UserFunction,
    context: dict[str, object],
    euler_state_name: StateName,
    post_decision_name: FunctionName,
    savings_value: ScalarFloat,
    euler_sample: Float1D,
) -> FloatND | IntND:
    """Evaluate one savings-stage variant over the Euler sample at fixed savings.

    The Euler state's law consumes the (removed, hence leaf) post-decision
    function and is fed the fixed savings value; the other savings-stage
    functions never read it (validated), so the varied savings slot is
    filtered away by their signatures. Vector-valued outputs (e.g. Markov
    weight vectors) stack along the sample axis; deterministic regime
    transitions yield integer regime ids.
    """

    def law_of_euler_state(state_value: ScalarFloat) -> FloatND | IntND:
        return _call_with_varied(
            func=law_func,
            fixed=context,
            varied={
                euler_state_name: state_value,
                post_decision_name: savings_value,
            },
        )

    return jax.vmap(law_of_euler_state)(euler_sample)


def _combo_contexts(
    *,
    func: UserFunction,
    grids: dict[StateOrActionName, Grid],
    varied: set[str],
    n_contexts: int = 3,
) -> list[dict[str, object]]:
    """A few fixed-input contexts for every non-varied argument of `func`.

    Context `j` binds each non-varied argument to the `j`-th point of its
    grid's small sample (clamped to the sample length), so the contexts span
    different discrete-combo regions. Returns no contexts when an argument
    is neither varied nor a state/action — a free model parameter whose
    value is unknown at build time, so the numeric check cannot run.
    """
    samples: dict[str, Float1D | Int1D] = {}
    for arg_name in inspect.signature(func).parameters:
        if arg_name in varied:
            continue
        if arg_name in grids:
            samples[arg_name] = _grid_sample(grid=grids[arg_name])
        else:
            return []
    return [
        {name: sample[min(j, sample.shape[0] - 1)] for name, sample in samples.items()}
        for j in range(n_contexts)
    ]


def _reachable_target_names(
    *,
    user_regime: UserRegime,
    user_regimes: Mapping[RegimeName, UserRegime],
) -> set[RegimeName]:
    """Regimes a regime can transition into, read off the declared reachability.

    The regime transition is the single source of truth for reachability:

    - a granular per-target mapping declares exactly its key set — omitted
      regimes are structurally unreachable,
    - any coarse form (bare callable or `MarkovTransition`) reaches every
      regime.
    """
    transition = user_regime.transition
    if isinstance(transition, Phased):
        transition = transition.solve
    if isinstance(transition, Mapping):
        return set(cast("Mapping[RegimeName, object]", transition).keys())
    return set(user_regimes)


def _resolve_solve_functions(
    *,
    user_regime: UserRegime,
) -> dict[FunctionName, UserFunction]:
    """Return the regime's solve-phase function pool.

    `Regime.functions` with solve-phase variants resolved, plus the solve
    imputation of every carried (`Phased`) state under the state's name. The
    imputations make DAG-ancestor checks see through carried states: a
    function reading a carried state imputed from the Euler state genuinely
    depends on the Euler state at the savings stage.
    """
    resolved: dict[FunctionName, UserFunction] = {}
    for name, func in user_regime.functions.items():
        if isinstance(func, Phased):
            resolved[name] = cast("UserFunction", func.solve)
        elif func is not None:
            resolved[name] = func
    for state_name, grid in user_regime.states.items():
        if isinstance(grid, Phased):
            resolved[state_name] = cast("UserFunction", grid.solve)
    return resolved


def _solve_grids(
    *,
    slot: Mapping[StateName, object] | Mapping[ActionName, object],
) -> dict[StateOrActionName, Grid]:
    """Solve-phase grids of a `states` or `actions` slot.

    A `Phased` state is carried: derived (no grid axis) during backward
    induction, so it contributes no solve-phase grid — exactly why carried
    states are invisible to the DC-EGM state classification. `None` entries
    (model-level broadcast masks) carry no grid either; the effective regime
    the validation runs on has them resolved away.
    """
    return {name: grid for name, grid in slot.items() if isinstance(grid, Grid)}


def _continuous_non_process_names(
    *,
    grids: Mapping[StateOrActionName, Grid],
) -> list[StateOrActionName]:
    """Names of continuous grids that are not stochastic processes."""
    return [
        name
        for name, grid in grids.items()
        if isinstance(grid, ContinuousGrid)
        and not isinstance(grid, _ContinuousStochasticProcess)
    ]


def _transition_variants(
    *,
    value: object,
) -> list[tuple[str, UserFunction]]:
    """Unpack a `state_transitions` entry into labeled callables.

    Handles bare callables, `Phased` containers (solve variant),
    `MarkovTransition` wrappers (unwrapped to the weight function), and
    per-target dicts (one entry per target regime).
    """
    if isinstance(value, Phased):
        value = value.solve
    if isinstance(value, MarkovTransition):
        return [("", cast("UserFunction", value.func))]
    if isinstance(value, Mapping):
        variants: list[tuple[str, UserFunction]] = []
        for target_name, target_value in value.items():
            func = (
                target_value.func
                if isinstance(target_value, MarkovTransition)
                else target_value
            )
            variants.append((f" (target '{target_name}')", cast("UserFunction", func)))
        return variants
    return [("", cast("UserFunction", value))]


def _without(
    *,
    functions: dict[FunctionName, UserFunction],
    names: set[FunctionName],
) -> dict[FunctionName, UserFunction]:
    """Return `functions` with `names` removed, so they become DAG leaves."""
    return {name: func for name, func in functions.items() if name not in names}


def _dag_ancestors(
    *,
    functions: dict[FunctionName, UserFunction],
    target_func: UserFunction,
) -> set[str]:
    """Ancestors (function names and leaf inputs) of a standalone callable.

    The callable is added to the DAG under a reserved name so its own name
    cannot shadow a regime function.
    """
    target_name = "__dcegm_validation_target__"
    mapping = {**functions, target_name: target_func}
    return set(get_ancestors(mapping, targets=[target_name], include_targets=False))


def _grid_sample(*, grid: Grid, n_points: int = 5) -> Float1D | Int1D:
    """A small, sorted, evenly indexed sample of grid points.

    Continuous grids yield float points; discrete grids yield their integer
    codes, so the sample dtype follows the grid kind.
    """
    points = grid.to_jax()
    n_grid = points.shape[0]
    indices = jnp.unique(
        jnp.linspace(0, n_grid - 1, num=min(n_points, n_grid)).astype(jnp.int32)
    )
    return points[indices]


def _fixed_kwargs(
    *,
    func: UserFunction,
    grids: dict[StateOrActionName, Grid],
    varied: set[str],
) -> dict[str, object] | None:
    """Fixed inputs (first grid point) for every non-varied argument of `func`.

    Returns `None` when an argument is neither varied nor a state/action —
    a free model parameter whose value is unknown at build time, so the
    numeric check cannot run.
    """
    fixed: dict[str, object] = {}
    for arg_name in inspect.signature(func).parameters:
        if arg_name in varied:
            continue
        if arg_name in grids:
            fixed[arg_name] = grids[arg_name].to_jax()[0]
        else:
            return None
    return fixed


def _call_with_varied(
    *,
    func: UserFunction,
    fixed: dict[str, object],
    varied: dict[str, object],
) -> FloatND | IntND:
    """Call `func` with fixed kwargs plus the varied values it actually takes.

    `fixed` covers exactly the non-varied arguments; varied values are
    filtered against the signature because a DAG target need not consume
    every sample variable (e.g. resources independent of the Euler state).
    """
    arg_names = set(inspect.signature(func).parameters)
    return func(**fixed, **{k: v for k, v in varied.items() if k in arg_names})


def _isclose(*, actual: object, expected: object, rtol: float, atol: float) -> bool:
    """Eager scalar closeness check, robust to 0-d JAX arrays."""
    return bool(
        jnp.isclose(jnp.asarray(actual), jnp.asarray(expected), rtol=rtol, atol=atol)
    )
