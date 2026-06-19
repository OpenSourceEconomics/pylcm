"""Pure spec-introspection helpers for DC-EGM regimes and their carry targets.

These read a regime's (or a carry target's) static specification — its
`VInterpolationInfo`, its `UserRegime`, its function set — and return the
names, grids, and composed functions the kernel build, the continuation
subsystem, and the kernel-scope checks all need. They hold no kernel state and
perform no numerics, so they form the dependency leaf the other `egm` step
modules import from.
"""

from collections.abc import Callable
from typing import Any, cast

from dags import concatenate_functions

from _lcm.params.regime_template import create_regime_params_template
from _lcm.processes import _ContinuousStochasticProcess
from _lcm.regime_building.V import VInterpolationInfo
from _lcm.typing import ActionName, EconFunctionsMapping, FunctionName, StateName
from _lcm.utils.functools import get_union_of_args
from _lcm.variables import from_regime, get_grids
from lcm.phased import Phased
from lcm.regime import Regime as UserRegime
from lcm.solvers import DCEGM
from lcm.typing import ScalarFloat, UserFunction


def _get_discrete_state_names(
    *,
    v_interpolation_info: VInterpolationInfo,
) -> tuple[StateName, ...]:
    """Discrete-state names of a regime in carry-axis (V state) order."""
    return tuple(
        name
        for name in v_interpolation_info.state_names
        if name in v_interpolation_info.discrete_states
    )


def _get_passive_state_names(
    *,
    v_interpolation_info: VInterpolationInfo,
    euler_state_name: StateName,
) -> tuple[StateName, ...]:
    """Passive-state names of a regime in carry-axis (V continuous-state) order.

    Every continuous state other than the Euler state is passive — DC-EGM
    validation enforces this for DC-EGM regimes, and terminal carry targets
    are restricted to a single continuous state.
    """
    return tuple(
        name
        for name in v_interpolation_info.continuous_states
        if name != euler_state_name
    )


def _get_process_state_names(
    *,
    v_interpolation_info: VInterpolationInfo,
) -> tuple[StateName, ...]:
    """Names of a regime's process states (node-valued discrete dimensions)."""
    return tuple(
        name
        for name, grid in v_interpolation_info.discrete_states.items()
        if isinstance(grid, _ContinuousStochasticProcess)
    )


def _get_child_state_name(*, user_regime: UserRegime) -> StateName:
    """Name of a carry target's continuous (Euler) state.

    For a DC-EGM target this is its configured Euler state; for a terminal
    target it is its only state (uniqueness is checked by
    `_find_unsupported_feature`).
    """
    if isinstance(user_regime.solver, DCEGM):
        return user_regime.solver.continuous_state
    return next(iter(user_regime.states))


def _get_child_discrete_actions(
    *, user_regime: UserRegime
) -> tuple[tuple[ActionName, ...], tuple[Any, ...]]:
    """Discrete-action names and grid values of a carry target, in combo order.

    The order matches the target's own kernel combos (its state-action
    space's discrete actions), so per-row bindings line up with the carry's
    action axes. Terminal targets have no actions (guarded).
    """
    variables = from_regime(user_regime)
    grids = get_grids(user_regime)
    names = tuple(variables.discrete_action_names)
    return names, tuple(grids[name].to_jax() for name in names)


def _concatenate_regime_function(
    *,
    functions: EconFunctionsMapping,
    target: FunctionName,
) -> UserFunction:
    """Concatenate one regime-function target from the H-free DAG."""
    return concatenate_functions(
        functions={name: func for name, func in functions.items() if name != "H"},
        targets=target,
        enforce_signature=False,
        set_annotations=True,
    )


def _get_child_resources_function(
    *, user_regime: UserRegime
) -> Callable[..., ScalarFloat]:
    """Build the closed-over resources map of one carry target.

    For a DC-EGM target the map is its declared resources function (resolved
    to the solve-phase variant); for a terminal target the carry lives in
    M-space and the map is the identity. The returned callable takes the
    child's state, passive, and discrete-action values as keyword arguments
    (child names) so the kernel can compose it with the state transition and
    differentiate the composition per carry row.
    """
    if isinstance(user_regime.solver, DCEGM):
        return _concatenate_child_resources(user_regime=user_regime)

    state_name = next(iter(user_regime.states))

    def identity_resources(**kwargs: ScalarFloat) -> ScalarFloat:
        return kwargs[state_name]

    return identity_resources


def _get_child_resources_arg_names(*, user_regime: UserRegime) -> set[str]:
    """Argument names of a carry target's resources map."""
    if isinstance(user_regime.solver, DCEGM):
        return set(
            get_union_of_args([_concatenate_child_resources(user_regime=user_regime)])
        )
    return {next(iter(user_regime.states))}


def _concatenate_child_resources(*, user_regime: UserRegime) -> UserFunction:
    """Concatenate a DC-EGM target's resources function from its user DAG.

    Each user function's params are renamed to their qualified names
    (`<func>__<param>`) before concatenation, matching the engine's binding
    vocabulary so the kernel feeds them straight from the combo pool (which
    carries the regime's flat params). Solve-phase imputed intermediates (a
    `Phased` function or a carried state's solve law) are resolved to their
    solve variant and baked into the DAG, so their outputs are computed from
    leaf states and params rather than demanded as leaves.
    """
    # Imported lazily: `regime_building.processing` imports the solver
    # registry, which imports this module, so a top-level import would cycle.
    from _lcm.regime_building import processing as _proc  # noqa: PLC0415

    solver = cast("DCEGM", user_regime.solver)
    regime_params_template = create_regime_params_template(user_regime)
    resolved: dict[str, UserFunction] = {}
    for name, func in user_regime.functions.items():
        if isinstance(func, Phased):
            resolved[name] = cast("UserFunction", func.solve)
        elif func is not None:
            resolved[name] = func
    # A carried state contributes a solve-phase imputation function under its
    # own name; include it so resources may read the imputed value.
    for name, value in user_regime.states.items():
        if isinstance(value, Phased) and name not in resolved:
            resolved[name] = cast("UserFunction", value.solve)
    qnamed = {
        name: _proc._rename_params_to_qnames(  # noqa: SLF001
            func=func,
            regime_params_template=regime_params_template,
            param_key=name,
        )
        for name, func in resolved.items()
    }
    return concatenate_functions(
        functions=qnamed,
        targets=solver.resources,
        enforce_signature=False,
        set_annotations=True,
    )
