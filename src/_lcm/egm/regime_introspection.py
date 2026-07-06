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

from dags import concatenate_functions, get_annotations, with_signature
from dags.annotations import ensure_annotations_are_strings

from _lcm.grids.continuous import ContinuousGrid
from _lcm.params.regime_template import create_regime_params_template
from _lcm.processes import _ContinuousStochasticProcess
from _lcm.regime_building.V import VInterpolationInfo
from _lcm.typing import ActionName, EconFunctionsMapping, FunctionName, StateName
from _lcm.utils.functools import get_union_of_args
from _lcm.variables import from_regime, get_grids
from lcm.phased import Phased
from lcm.regime import Regime as UserRegime
from lcm.solvers import DCEGM, NEGM
from lcm.typing import ScalarFloat, UserFunction


def _as_dcegm(user_regime: UserRegime) -> DCEGM | None:
    """Return the DC-EGM config a carry target solves its inner Euler with.

    A `DCEGM` regime solves directly; a `NEGM` regime nests the same 1-D
    DC-EGM consumption-savings solve, so its child read uses `solver.inner`. Any
    other solver (e.g. a terminal grid-search regime) returns `None` — the
    target's carry lives in M-space and is read with the identity map.
    """
    solver = user_regime.solver
    if isinstance(solver, DCEGM):
        return solver
    if isinstance(solver, NEGM):
        return solver.inner
    return None


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

    For a DC-EGM or NEGM target this is its configured (inner) Euler state; for a
    terminal target it is its unique non-process continuous state. A model-level
    distributed state (e.g. a permanent type sharded across devices) is broadcast
    into the terminal regime as an extra discrete axis, so the terminal regime's
    `states` need not be a singleton — the Euler state is the continuous one, not
    whichever state iterates first.
    """
    dcegm = _as_dcegm(user_regime)
    if dcegm is not None:
        return dcegm.continuous_state
    grids = get_grids(user_regime)
    continuous = tuple(
        name
        for name in user_regime.states
        if isinstance(grids[name], ContinuousGrid)
        and not isinstance(grids[name], _ContinuousStochasticProcess)
    )
    if len(continuous) != 1:
        msg = (
            f"A terminal carry target must have exactly one continuous (Euler) "
            f"state; regime states {tuple(user_regime.states)} resolve to "
            f"continuous states {continuous}."
        )
        raise ValueError(msg)
    return continuous[0]


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

    For a DC-EGM or NEGM target the map is its (inner) resources function
    (resolved to the solve-phase variant); for a terminal target the carry lives
    in M-space and the map is the identity. The returned callable takes the
    child's state, passive, and discrete-action values as keyword arguments
    (child names) so the kernel can compose it with the state transition and
    differentiate the composition per carry row.
    """
    if _as_dcegm(user_regime) is not None:
        return _concatenate_child_resources(user_regime=user_regime)

    state_name = _get_child_state_name(user_regime=user_regime)

    def identity_resources(**kwargs: ScalarFloat) -> ScalarFloat:
        return kwargs[state_name]

    return identity_resources


def _get_child_resources_arg_names(*, user_regime: UserRegime) -> set[str]:
    """Argument names of a carry target's resources map."""
    if _as_dcegm(user_regime) is not None:
        return set(
            get_union_of_args([_concatenate_child_resources(user_regime=user_regime)])
        )
    return {_get_child_state_name(user_regime=user_regime)}


def _concatenate_child_resources(*, user_regime: UserRegime) -> UserFunction:
    """Concatenate a DC-EGM / NEGM target's resources function from its user DAG.

    Each user function's params are renamed to their qualified names
    (`<func>__<param>`) before concatenation, matching the engine's binding
    vocabulary so the kernel feeds them straight from the combo pool (which
    carries the regime's flat params). Solve-phase imputed intermediates (a
    `Phased` function or a carried state's solve law) are resolved to their
    solve variant and baked into the DAG, so their outputs are computed from
    leaf states and params rather than demanded as leaves.

    For a NEGM target the published continuation is the keeper's (the durable
    stays put), so the inner resources' outer post-decision is replaced by the
    durable identity `next_<durable> = <durable>`. The child resources then read
    `<durable>` (a bound passive state) rather than demanding the outer
    post-decision as an unbound leaf.
    """
    # Imported lazily: `regime_building.processing` imports the solver
    # registry, which imports this module, so a top-level import would cycle.
    from _lcm.regime_building import processing as _proc  # noqa: PLC0415

    dcegm = cast("DCEGM", _as_dcegm(user_regime))
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
    if isinstance(user_regime.solver, NEGM):
        resolved[user_regime.solver.outer_post_decision] = _keeper_identity_function(
            outer_post_decision=user_regime.solver.outer_post_decision,
            functions=resolved,
        )
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
        targets=dcegm.resources,
        enforce_signature=False,
        set_annotations=True,
    )


def _keeper_identity_function(
    *, outer_post_decision: FunctionName, functions: dict[str, UserFunction]
) -> UserFunction:
    """Build the keeper identity `next_<durable>(durable) = durable`.

    The injected function declares the durable state as its single argument and
    copies its annotation off the first regime function that declares it, so the
    DAG's annotation-consistency check (which requires every consumer of a leaf
    to agree) stays satisfied.
    """
    durable_state = outer_post_decision.removeprefix("next_")
    annotation = _annotation_of_arg(functions=functions, arg_name=durable_state)

    @with_signature(args={durable_state: annotation}, return_annotation=annotation)
    def keep_outer_post_decision(**kwargs: ScalarFloat) -> ScalarFloat:
        return kwargs[durable_state]

    keep_outer_post_decision.__name__ = outer_post_decision
    return cast("UserFunction", keep_outer_post_decision)


def _annotation_of_arg(
    *, functions: dict[str, UserFunction], arg_name: StateName
) -> str:
    """Return the annotation the regime's functions use for one argument.

    Falls back to `"FloatND"` when no function annotates it.
    """
    for func in functions.values():
        annotations = ensure_annotations_are_strings(get_annotations(func))
        annotation = annotations.get(arg_name, "no_annotation_found")
        if annotation != "no_annotation_found":
            return annotation
    return "FloatND"
