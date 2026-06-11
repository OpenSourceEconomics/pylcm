"""Build-time validation of the DC-EGM model contract.

A regime configured with `solver=DCEGM(...)` must satisfy the structural
contract the endogenous grid method relies on. Every violation raises
`ModelInitializationError` at `Model` construction with a message naming the
offending piece. The rules, in the order they are checked:

- exactly one continuous (*Euler*) state and one continuous action; process
  states are exempt (they enter the value function as node-valued discrete
  dimensions); passive continuous states are not supported yet
- the post-decision function, the resources function, and
  `inverse_marginal_utility` exist in `Regime.functions`
- the regime uses the default Bellman aggregator `H`
- the post-decision function consumes the continuous action and the
  resources function (not the continuous state directly)
- no constraint touches the continuous state or action (EGM enforces the
  budget identity and the borrowing limit intrinsically)
- `utility` does not depend on the continuous state (envelope condition)
- the resources function does not depend on the continuous action
- the Euler state's transition is deterministic, names the post-decision
  function, and reaches the state and the action only through it
- everything evaluated at the savings-node stage (regime transition,
  stochastic weights, non-Euler state transitions) is independent of the
  Euler state, the continuous action, the resources function, and the
  post-decision function
- grid hygiene: the Euler grid is neither batched nor distributed, and the
  savings grid covers the Euler grid's upper region
- every reachable non-terminal target regime also uses DC-EGM with the same
  Euler state (brute-force regimes may target DC-EGM regimes)
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
from lcm.regime import _default_H
from lcm.solvers import DCEGM
from lcm.transition import MarkovTransition
from lcm.typing import Float1D, FloatND, ScalarFloat, UserFunction


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

    functions = _resolve_solve_functions(user_regime)

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
    discrete dimensions. Any other continuous state would have to be a
    *passive* state, which the DCEGM solver does not support yet.
    """
    continuous_states = _continuous_non_process_names(_solve_grids(user_regime.states))
    continuous_actions = _continuous_non_process_names(
        _solve_grids(user_regime.actions)
    )

    if solver.continuous_state not in continuous_states:
        msg = (
            f"DCEGM `continuous_state` '{solver.continuous_state}' is not a "
            f"continuous state of regime '{regime_name}'. Continuous "
            f"(non-process) states: {continuous_states}."
        )
        raise ModelInitializationError(msg)

    extra_states = [s for s in continuous_states if s != solver.continuous_state]
    if extra_states:
        msg = (
            f"Regime '{regime_name}' has continuous states {extra_states} in "
            f"addition to the Euler continuous state "
            f"'{solver.continuous_state}'. The DCEGM solver supports exactly "
            "one continuous state; passive continuous states are not "
            "supported yet (process states are — declare them via the "
            "stochastic process grids)."
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
    """Require the post-decision, resources, and inverse-marginal-utility functions."""
    required: dict[FunctionName, str] = {
        solver.post_decision_function: (
            "the post-decision function (`DCEGM.post_decision_function`)"
        ),
        solver.resources: "the resources function (`DCEGM.resources`)",
        "inverse_marginal_utility": (
            "the inverse marginal utility of the continuous action"
        ),
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
    """Require the default Bellman aggregator `H`.

    The Euler inversion hard-codes `H = utility + discount_factor * E[V']`,
    so a custom `H` would silently change the meaning of the solution.
    """
    if user_regime.functions.get("H") is not _default_H:
        msg = (
            f"Regime '{regime_name}' defines a custom Bellman aggregator `H`. "
            "The DCEGM solver hard-codes the default aggregator "
            "`H = utility + discount_factor * E[V']`; remove the custom `H` "
            "or use the brute-force solver."
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
    """The Euler transition consumes the post-decision function — and only it.

    With the post-decision and resources functions removed from the DAG,
    their names become leaf inputs, so the ancestor set reveals whether the
    transition reaches the state or the action through any other channel.
    """
    bypass_msg = (
        f"The transition of the Euler state '{solver.continuous_state}' in "
        f"regime '{regime_name}' must consume the post-decision function "
        f"'{solver.post_decision_function}' and reach "
        f"'{solver.continuous_state}' and '{solver.continuous_action}' only "
        "through it."
    )
    value = user_regime.state_transitions.get(solver.continuous_state)
    if value is None:
        msg = (
            f"{bypass_msg} It is declared as a fixed state (identity "
            "transition), which bypasses the post-decision function."
        )
        raise ModelInitializationError(msg)

    opaque_functions = _without(
        functions, names={solver.post_decision_function, solver.resources}
    )
    forbidden = {
        solver.continuous_state,
        solver.continuous_action,
        solver.resources,
    }
    for label, transition_func in _transition_variants(value):
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

    At a savings node, current wealth and the continuous action are unknown
    (they are EGM outputs). The regime transition, every stochastic
    transition weight, and every non-Euler state transition must therefore
    be independent of the Euler state, the continuous action, the resources
    function, and the post-decision function.
    """
    candidates: list[tuple[str, UserFunction]] = []
    if user_regime.transition is not None:
        # Coarse, MarkovTransition-wrapped, Phased, and granular per-target
        # regime transitions all unpack to plain callables.
        for label, regime_transition in _transition_variants(user_regime.transition):
            candidates.append((f"regime transition function{label}", regime_transition))
    for state_name, value in user_regime.state_transitions.items():
        if state_name == solver.continuous_state or value is None:
            continue
        for label, transition_func in _transition_variants(value):
            candidates.append(
                (f"transition of state '{state_name}'{label}", transition_func)
            )

    opaque_functions = _without(
        functions, names={solver.post_decision_function, solver.resources}
    )
    forbidden = {
        solver.continuous_state,
        solver.continuous_action,
        solver.resources,
        solver.post_decision_function,
    }
    for label, func in candidates:
        ancestors = _dag_ancestors(functions=opaque_functions, target_func=func)
        bad = sorted(ancestors & forbidden)
        if bad:
            msg = (
                f"The {label} of regime '{regime_name}' depends on {bad}. "
                "Functions evaluated at the savings-node stage (regime "
                "transition probabilities, stochastic transition weights, and "
                "non-Euler state transitions) must not depend on the Euler "
                f"state '{solver.continuous_state}', the continuous action "
                f"'{solver.continuous_action}', the resources function "
                f"'{solver.resources}', or the post-decision function "
                f"'{solver.post_decision_function}' — those values are "
                "unknown until the EGM step has run."
            )
            raise ModelInitializationError(msg)


def _fail_if_grid_hygiene_violated(
    *,
    regime_name: RegimeName,
    user_regime: UserRegime,
    solver: DCEGM,
) -> None:
    """No runtime/batched/distributed grids; savings grid covers the Euler grid."""
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
    # The EGM kernel selects carry rows by integer indexing along whole
    # discrete axes, so discrete grids cannot be batched or distributed.
    for kind, name_to_grid in (
        ("state", user_regime.states),
        ("action", user_regime.actions),
    ):
        for name, grid in name_to_grid.items():
            if isinstance(grid, DiscreteGrid) and (
                grid.batch_size != 0 or grid.distributed
            ):
                msg = (
                    f"The grid of the discrete {kind} '{name}' in regime "
                    f"'{regime_name}' must not be batched or distributed in "
                    f"a DCEGM regime (got batch_size={grid.batch_size}, "
                    f"distributed={grid.distributed})."
                )
                raise ModelInitializationError(msg)
    if euler_grid.batch_size != 0 or euler_grid.distributed:
        msg = (
            f"The grid of the Euler state '{solver.continuous_state}' in "
            f"regime '{regime_name}' must not be batched or distributed in a "
            f"DCEGM regime (got batch_size={euler_grid.batch_size}, "
            f"distributed={euler_grid.distributed})."
        )
        raise ModelInitializationError(msg)
    if solver.savings_grid.batch_size != 0 or solver.savings_grid.distributed:
        msg = (
            f"The DCEGM savings grid of regime '{regime_name}' must not be "
            f"batched or distributed (got "
            f"batch_size={solver.savings_grid.batch_size}, "
            f"distributed={solver.savings_grid.distributed})."
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
    for target_name in sorted(_reachable_target_names(user_regime, user_regimes)):
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
        **_solve_grids(user_regime.states),
        **_solve_grids(user_regime.actions),
    }
    euler_sample = _grid_sample(grids[solver.continuous_state])
    action_sample = _grid_sample(grids[solver.continuous_action])
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
        decreases = [
            (float(w_lo), float(w_hi))
            for w_lo, w_hi, r_lo, r_hi in zip(
                euler_sample[:-1],
                euler_sample[1:],
                resources_at[:-1],
                resources_at[1:],
                strict=True,
            )
            if float(r_hi) < float(r_lo) - atol
        ]
        if decreases:
            msg = (
                f"The resources function '{solver.resources}' of regime "
                f"'{regime_name}' must be non-decreasing in "
                f"'{solver.continuous_state}'; it decreases on the sample "
                f"intervals {decreases}."
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
                if not _isclose(post_value, expected, rtol=rtol, atol=atol):
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
    """Check `(u')⁻¹(u'(c)) ≈ c` and strict monotonicity of `u'` on a sample."""
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
    utility_kwargs = _fixed_kwargs(
        func=utility_func,
        grids=grids,
        varied={solver.continuous_action},
    )
    inverse_kwargs = _fixed_kwargs(
        func=inverse_func,
        grids=grids,
        varied={"marginal_continuation"},
    )
    if utility_kwargs is None or inverse_kwargs is None:
        return
    if solver.continuous_action not in inspect.signature(utility_func).parameters:
        return

    def utility_of_action(action_value: ScalarFloat) -> ScalarFloat:
        return utility_func(
            **utility_kwargs, **{solver.continuous_action: action_value}
        )

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
        if not _isclose(recovered, c, rtol=rtol, atol=atol):
            msg = (
                f"'inverse_marginal_utility' of regime '{regime_name}' is "
                "inconsistent with `jax.grad` of the utility DAG: the "
                f"round-trip `(u')⁻¹(u'(c))` at "
                f"{solver.continuous_action}={float(c)} yields "
                f"{float(recovered)} (marginal utility {float(mu)})."
            )
            raise ModelInitializationError(msg)


def _reachable_target_names(
    user_regime: UserRegime,
    user_regimes: Mapping[RegimeName, UserRegime],
) -> set[RegimeName]:
    """Regimes a regime can transition into, judged from `state_transitions`.

    Mirrors the engine's reachability notion: without per-target transition
    dicts every regime is potentially reachable; with them, the reachable set
    is the union of the explicitly named targets plus any regime whose states
    are fully covered by simple (non-per-target) transitions.
    """
    per_target_keys: set[RegimeName] = set()
    has_per_target = False
    simple_state_names: set[str] = set()
    for state_name, value in user_regime.state_transitions.items():
        if isinstance(value, Mapping) and not isinstance(value, MarkovTransition):
            has_per_target = True
            per_target_keys |= set(cast("Mapping[RegimeName, object]", value).keys())
        else:
            simple_state_names.add(state_name)
    if not has_per_target:
        return set(user_regimes)
    for target_name, target in user_regimes.items():
        needed = set(target.states)
        if needed and needed.issubset(simple_state_names):
            per_target_keys.add(target_name)
    return per_target_keys


def _resolve_solve_functions(
    user_regime: UserRegime,
) -> dict[FunctionName, UserFunction]:
    """Return `Regime.functions` with solve-phase variants resolved."""
    resolved: dict[FunctionName, UserFunction] = {}
    for name, func in user_regime.functions.items():
        if isinstance(func, Phased):
            resolved[name] = cast("UserFunction", func.solve)
        elif func is not None:
            resolved[name] = func
    return resolved


def _solve_grids(
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
    functions: dict[FunctionName, UserFunction],
    *,
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


def _grid_sample(grid: Grid, *, n_points: int = 5) -> Float1D:
    """A small, sorted, evenly indexed sample of grid points."""
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
) -> FloatND:
    """Call `func` with fixed kwargs plus the varied values it actually takes.

    `fixed` covers exactly the non-varied arguments; varied values are
    filtered against the signature because a DAG target need not consume
    every sample variable (e.g. resources independent of the Euler state).
    """
    arg_names = set(inspect.signature(func).parameters)
    return func(**fixed, **{k: v for k, v in varied.items() if k in arg_names})


def _isclose(actual: object, expected: object, *, rtol: float, atol: float) -> bool:
    """Eager scalar closeness check, robust to 0-d JAX arrays."""
    return bool(
        jnp.isclose(jnp.asarray(actual), jnp.asarray(expected), rtol=rtol, atol=atol)
    )
