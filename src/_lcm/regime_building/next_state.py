"""Generate function that compute the next states for solution and simulation."""

import inspect
import operator
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, no_type_check

import jax
from dags import concatenate_functions
from dags.tree import qname_from_tree_path

from _lcm.grids import DiscreteGrid, Grid
from _lcm.processes import _ContinuousStochasticProcess
from _lcm.processes.ar1 import _AR1Process
from _lcm.processes.iid import _IIDProcess
from _lcm.processes.state_conditioned import (
    StateConditioned,
    gather_sigma,
    sigma_array_by_code,
)
from _lcm.transition_plans import (
    LotteryLifetime,
    TargetTransitionPlans,
)
from _lcm.typing import (
    EconFunctionsMapping,
    NextStateSimulationFunction,
    RegimeName,
    StateName,
    StateOrActionName,
    StochasticNextFunction,
    TransitionFunction,
    TransitionFunctionName,
    TransitionFunctionsMapping,
)
from lcm.exceptions import ModelInitializationError
from lcm.typing import ContinuousState, DiscreteState, Float1D, FloatND, IntND


def get_next_state_function_for_solution(
    *,
    transitions: MappingProxyType[TransitionFunctionName, TransitionFunction],
    functions: EconFunctionsMapping,
    targets: Sequence[TransitionFunctionName] | None = None,
) -> NextStateSimulationFunction:
    """Get function that computes the next states during the solution.

    Args:
        transitions: Transitions to the next states of a regime.
        functions: Immutable mapping of auxiliary functions of a regime.
        targets: Laws to compute, defaulting to every law in `transitions`. A law
            reading one of this target's own draws is resolved on the node axis
            instead, so the caller leaves it out here.

    Returns:
        Function that computes the next states. Depends on states and actions of the
        current period, and the regime parameters (as flat kwargs). If target
        is "simulate", the function also depends on the dictionary of random keys
        ("keys"), which corresponds to the names of stochastic next functions.

    """
    functions_to_concatenate = dict(transitions) | dict(functions)

    return concatenate_functions(
        functions=functions_to_concatenate,
        targets=list(transitions.keys() if targets is None else targets),
        return_type="dict",
        enforce_signature=False,
        set_annotations=True,
    )


def get_next_state_function_for_simulation(
    *,
    transitions: TransitionFunctionsMapping,
    functions: EconFunctionsMapping,
    all_grids: MappingProxyType[RegimeName, MappingProxyType[StateOrActionName, Grid]],
    transition_plans: TargetTransitionPlans,
) -> NextStateSimulationFunction:
    """Get function that computes the next states during the simulation.

    Builds one DAG per target regime using unqualified `next_<state>` keys, mirroring
    the per-target structure of {func}`get_next_state_function_for_solution`. This
    lets a transition function or auxiliary regime function consume another
    transition's `next_<state>` output via plain name resolution within the same
    target's DAG. The combined function returns a nested mapping keyed by target
    regime name, with each inner dict using unqualified `next_<state>` keys.

    Stochastic-transition wrappers expose `key_<target>__next_<state>` and
    `weight_<target>__next_<state>` as external arguments so callers can pass a
    distinct random key and pre-computed weight per target.

    Args:
        transitions: Nested mapping of target regime names to transition functions.
        functions: Immutable mapping of auxiliary functions of a regime.
        all_grids: Immutable mapping of regime names to Grid spec objects.
        transition_plans: Immutable mapping of target regime names to their
            transition laws.

    Returns:
        Function that computes the next states. Depends on states and actions of the
        current period, and the regime parameters ("params"). The function also
        depends on the dictionary of random keys ("keys") for stochastic transitions.
        Returns `{target_regime_name: {next_<state>: array}}`.

    """
    per_target_funcs: dict[RegimeName, Callable[..., dict[str, FloatND | IntND]]] = {}
    for target_regime_name, bundle in transitions.items():
        extended = _extend_bundle_for_simulation(
            target_regime_name=target_regime_name,
            bundle=bundle,
            all_grids=all_grids,
            transition_plans=transition_plans,
        )
        per_target_funcs[target_regime_name] = concatenate_functions(
            functions=dict(extended) | dict(functions),
            targets=[
                output.next_state_name
                for output in transition_plans[target_regime_name].outputs.values()
            ],
            return_type="dict",
            enforce_signature=False,
            set_annotations=True,
        )

    return concatenate_functions(
        functions=per_target_funcs,
        targets=list(per_target_funcs.keys()),
        return_type="dict",
        enforce_signature=False,
        set_annotations=True,
    )


def get_next_stochastic_weights_function(
    *,
    regime_name: RegimeName,
    functions: EconFunctionsMapping,
    transitions: MappingProxyType[TransitionFunctionName, TransitionFunction],
    transition_plans: TargetTransitionPlans,
) -> Callable[..., dict[str, FloatND | IntND]]:
    """Get function that computes the weights for the next stochastic states.

    Args:
        regime_name: Name of the regime that the transitions target.
        functions: Immutable mapping of auxiliary functions of the model.
        transitions: Transitions to the target regime.
        transition_plans: Immutable mapping of target regime names to their
            transition laws.

    Returns:
        Function that computes the weights for the next stochastic states.

    """
    return _get_next_weights_function(
        regime_name=regime_name,
        functions=functions,
        transitions=transitions,
        transition_plans=transition_plans,
        select=_is_lottery,
    )


def get_next_interpolation_basis_weights_function(
    *,
    regime_name: RegimeName,
    functions: EconFunctionsMapping,
    transitions: MappingProxyType[TransitionFunctionName, TransitionFunction],
    transition_plans: TargetTransitionPlans,
) -> Callable[..., dict[str, FloatND | IntND]]:
    """Get function that computes the node-basis weights of the declared entries.

    These weights place one declared value on the target's nodes; they are the
    coefficients of an interpolation, not probabilities, and the caller contracts
    them into a single continuation before any certainty equivalent runs.

    Args:
        regime_name: Name of the regime that the transitions target.
        functions: Immutable mapping of auxiliary functions of the model.
        transitions: Transitions to the target regime.
        transition_plans: Immutable mapping of target regime names to their
            transition laws.

    Returns:
        Function that computes the node-basis weights of the declared entries.

    """
    return _get_next_weights_function(
        regime_name=regime_name,
        functions=functions,
        transitions=transitions,
        transition_plans=transition_plans,
        select=_has_interpolation_basis,
    )


def _get_next_weights_function(
    *,
    regime_name: RegimeName,
    functions: EconFunctionsMapping,
    transitions: MappingProxyType[TransitionFunctionName, TransitionFunction],
    transition_plans: TargetTransitionPlans,
    select: Callable[..., bool],
) -> Callable[..., dict[str, FloatND | IntND]]:
    """Build the DAG producing one kind of target-qualified weight vector.

    Args:
        regime_name: Name of the regime that the transitions target.
        functions: Immutable mapping of auxiliary functions of the model.
        transitions: Transitions to the target regime.
        transition_plans: Immutable mapping of target regime names to their
            transition laws.
        select: Predicate picking the laws whose weights to build.

    Returns:
        Function that computes the selected weight vectors.

    """
    targets = [
        f"weight_{regime_name}__{func_name}"
        for func_name in transitions
        if select(plans=transition_plans, target=regime_name, name=func_name)
    ]
    # A weight law may read another transition's `next_<state>` output within the
    # same target's DAG -- the supported transition-reads-transition composition
    # that the solution next-state builder
    # (`get_next_state_function_for_solution`) already relies on. Those producers
    # live in `transitions`, not `functions`, so the deterministic transitions
    # belong in the weight DAG; otherwise the read is left as an unsupplied
    # argument and the Q build fails with a missing input.
    #
    # Availability is decided by whether a law has a physical value to publish,
    # never by whether weights were built for it:
    #
    # - An interpolation-basis law is deterministic -- it names one value -- so it
    #   stays a producer. Its weights place that value on the target's private
    #   node axis; they do not replace it.
    # - A stochastic law realizes a draw, which has no value while the expectation
    #   over it is still being built. Excluding it leaves a dependency on one
    #   unresolved, which surfaces the unsupported composition loudly rather than
    #   pricing a conditional joint kernel the product-of-marginals form cannot
    #   represent.
    available_transitions = {
        name: func
        for name, func in transitions.items()
        if not transition_plans[regime_name].is_lottery(name)
    }
    return concatenate_functions(
        functions=dict(available_transitions) | dict(functions),
        targets=targets,
        return_type="dict",
        enforce_signature=False,
        set_annotations=True,
    )


def _is_lottery(
    *, plans: TargetTransitionPlans, target: RegimeName, name: TransitionFunctionName
) -> bool:
    """Whether `name` is a stochastic law of `target`."""
    return plans[target].is_lottery(name)


def _has_interpolation_basis(
    *, plans: TargetTransitionPlans, target: RegimeName, name: TransitionFunctionName
) -> bool:
    """Whether `name` is a declared entry placed on `target`'s nodes."""
    return plans[target].has_interpolation_basis(name)


def _extend_bundle_for_simulation(
    *,
    target_regime_name: RegimeName,
    bundle: MappingProxyType[TransitionFunctionName, Callable[..., FloatND | IntND]],
    all_grids: MappingProxyType[RegimeName, MappingProxyType[StateOrActionName, Grid]],
    transition_plans: TargetTransitionPlans,
) -> dict[TransitionFunctionName, Callable[..., FloatND | IntND]]:
    """Replace stochastic transitions for one target with realisation wrappers.

    Deterministic transitions are passed through unchanged. Stochastic transitions
    are replaced by wrappers that draw a realisation from a precomputed weight
    vector and a random key. The wrapper's external argument names use
    target-qualified form (`key_<target>__<next_state>`,
    `weight_<target>__<next_state>`) so multi-target callers can supply distinct
    random keys per target. The dict key keeps the unqualified `next_<state>` so
    other transitions or regime functions in the same target's DAG can resolve
    it by name.

    Args:
        target_regime_name: Target regime name.
        bundle: Mapping of unqualified `next_<state>` transition names
            to functions, restricted to one target regime.
        all_grids: Immutable mapping of regime names to Grid spec objects.
        transition_plans: Immutable mapping of target regime names to their
            transition laws.

    Returns:
        Extended transitions dictionary keyed by unqualified `next_<state>` names.

    """
    plan = transition_plans[target_regime_name]
    extended: dict[TransitionFunctionName, Callable[..., FloatND | IntND]] = dict(
        bundle
    )
    for next_state_name in bundle:
        lottery = plan.lotteries.get(next_state_name)
        if lottery is None:
            continue
        if lottery.lifetime is LotteryLifetime.TRANSITION_LOCAL:
            extended[next_state_name] = _create_joint_stochastic_next_func(
                target_regime_name=target_regime_name,
                lottery_name=next_state_name,
                support_provider_name=lottery.support_provider_name,
                support_size=lottery.support_signature.size,
                node_annotation=lottery.node_annotation,
            )
            continue
        state_name = next_state_name.removeprefix("next_")
        output = plan.outputs[state_name]
        if output.continuous_process:
            extended[next_state_name] = _create_continuous_stochastic_next_func(
                target_regime_name=target_regime_name,
                next_state_name=next_state_name,
                all_grids=all_grids,
            )
        else:
            extended[next_state_name] = _create_discrete_stochastic_next_func(
                target_regime_name=target_regime_name,
                next_state_name=next_state_name,
                labels=all_grids[target_regime_name][state_name].to_jax(),
            )
    return extended


def _create_joint_stochastic_next_func(
    *,
    target_regime_name: RegimeName,
    lottery_name: TransitionFunctionName,
    support_provider_name: str | None,
    support_size: int | None,
    node_annotation: str | None,
) -> Callable[..., Any]:
    """Draw one support node and publish its whole pytree to dependent outputs."""
    if support_provider_name is None or support_size is None or node_annotation is None:
        raise ModelInitializationError(
            f"Joint lottery '{target_regime_name}__{lottery_name}' lacks its "
            "support provider or static support size."
        )
    qname = qname_from_tree_path((target_regime_name, lottery_name))

    return _RealizedJointNode(
        qname=qname,
        support_provider_name=support_provider_name,
        support_size=support_size,
        node_annotation=node_annotation,
    )


@dataclass(frozen=True, kw_only=True, eq=False)
class _RealizedJointNode:
    """Draw one support node of a joint lottery and publish its whole pytree."""

    qname: str
    """Qualified `<target>__<lottery>` name of the lottery."""
    support_provider_name: str
    """Name of the DAG node supplying the lottery's support."""
    support_size: int
    """Static number of support nodes."""
    node_annotation: str
    """Annotation of one support node, also the return annotation."""

    def __post_init__(self) -> None:
        _publish_signature(
            target=self,
            args={
                f"weight_{self.qname}": "FloatND",
                f"key_{self.qname}": "PRNGKeyND",
                self.support_provider_name: self.node_annotation,
            },
            return_annotation=self.node_annotation,
            name="realized_node",
        )

    # The kernel is traced with whatever leaves its caller supplies -- tracers,
    # Python scalars, arrays of either integer width -- so its annotations
    # document the contract and are not enforced at call time.
    @no_type_check
    def __call__(self, **kwargs: Any) -> Any:  # noqa: ANN401
        index = jax.random.choice(
            key=kwargs[f"key_{self.qname}"],
            a=self.support_size,
            p=kwargs[f"weight_{self.qname}"],
        )
        return jax.tree_util.tree_map(
            operator.itemgetter(index), kwargs[self.support_provider_name]
        )


def _create_discrete_stochastic_next_func(
    *,
    target_regime_name: RegimeName,
    next_state_name: TransitionFunctionName,
    labels: DiscreteState,
) -> StochasticNextFunction:
    """Get function that simulates the next state of a stochastic variable.

    Args:
        target_regime_name: Target regime name.
        next_state_name: Transition function name with the `next_` prefix
            (e.g. `next_health`).
        labels: Category codes the discrete state can take (the DiscreteGrid
            rendered as a 1d JAX array). The simulated realisation is one of
            these, drawn via `jax.random.choice` weighted by `weight_<qname>`.

    Returns:
        A function that simulates the next state of the stochastic variable. The
        function must be called with keyword arguments:
        - weight_{qname}: 2d array of weights. The first dimension corresponds to the
          number of simulation units. The second dimension corresponds to the number of
          grid points (one slot per `labels` entry).
        - key_{qname}: PRNG key for the stochastic next function. `qname` is the
          dags-qualified `<target>__<next_state>`.

    """
    qname = qname_from_tree_path((target_regime_name, next_state_name))

    return _DiscreteStochasticNextState(qname=qname, labels=labels)


@dataclass(frozen=True, kw_only=True, eq=False)
class _DiscreteStochasticNextState:
    """Draw a discrete state's next value from its weight vector."""

    qname: str
    """Qualified `<target>__next_<state>` name of the transition."""
    labels: DiscreteState
    """Category codes the drawn value is one of."""

    def __post_init__(self) -> None:
        _publish_signature(
            target=self,
            args={f"weight_{self.qname}": "FloatND", f"key_{self.qname}": "PRNGKeyND"},
            return_annotation="DiscreteState",
            name="next_stochastic_state",
        )

    @no_type_check
    def __call__(self, **kwargs: FloatND) -> DiscreteState:
        return jax.random.choice(
            key=kwargs[f"key_{self.qname}"],
            a=self.labels,
            p=kwargs[f"weight_{self.qname}"],
        )


def _create_continuous_stochastic_next_func(
    *,
    target_regime_name: RegimeName,
    next_state_name: TransitionFunctionName,
    all_grids: MappingProxyType[RegimeName, MappingProxyType[StateOrActionName, Grid]],
) -> StochasticNextFunction:
    """Get function that simulates the next state of a stochastic variable.

    For processes whose params are supplied at runtime, the runtime params are
    accepted as additional keyword arguments and merged with fixed process
    params before calling the process calculation function.

    Args:
        target_regime_name: Target regime name.
        next_state_name: Transition function name with the `next_` prefix
            (e.g. `next_<process>`).
        all_grids: Immutable mapping of regime names to Grid spec objects.

    Returns:
        A function that simulates the next state of the stochastic variable.

    """
    state_name = next_state_name.removeprefix("next_")
    grid: _ContinuousStochasticProcess = all_grids[target_regime_name][state_name]  # ty: ignore [invalid-assignment]
    qname = qname_from_tree_path((target_regime_name, next_state_name))

    # A state-conditioned process draws with the sigma its conditioning state selects,
    # not the scalar `sigma` that places the nodes, so the simulated law is the one the
    # solution was computed under. The conditioning value is dated t; the declaration it
    # is read against is the target regime's.
    conditioned = _resolve_conditioned_sigma(
        grid=grid, grids=all_grids[target_regime_name]
    )

    if isinstance(grid, _AR1Process):
        return _create_ar1_next_func(
            qname=qname, state_name=state_name, grid=grid, conditioned=conditioned
        )
    if isinstance(grid, _IIDProcess):
        return _create_iid_next_func(
            qname=qname, state_name=state_name, grid=grid, conditioned=conditioned
        )

    msg = f"Expected _IIDProcess or _AR1Process, got {type(grid)}"
    raise TypeError(msg)


def _resolve_conditioned_sigma(
    *,
    grid: _ContinuousStochasticProcess,
    grids: Mapping[StateOrActionName, Grid],
) -> tuple[StateConditioned, Float1D] | None:
    """Resolve the code-ordered sigma array of a state-conditioned process, else None.

    Reuses `sigma_array_by_code` so simulation gathers sigma under exactly the
    category-code ordering the solve-side transition rows use.
    """
    sc = grid.state_conditioned
    if sc is None:
        return None
    conditioning_grid = grids.get(sc.on)
    if not isinstance(conditioning_grid, DiscreteGrid):
        msg = (
            f"state_conditioned.on='{sc.on}' must name a DiscreteGrid state in the "
            f"same regime as the process."
        )
        raise ModelInitializationError(msg)
    return sc, sigma_array_by_code(cond_grid=conditioning_grid, by=sc.by)


def _create_ar1_next_func(
    *,
    qname: str,
    state_name: StateName,
    grid: _AR1Process,
    conditioned: tuple[StateConditioned, Float1D] | None = None,
) -> StochasticNextFunction:
    fixed_params = dict(grid.params)
    runtime_param_names = {
        qname_from_tree_path((state_name, p)): p for p in grid.params_to_pass_at_runtime
    }
    args: dict[str, str] = {
        f"key_{qname}": "PRNGKeyND",
        state_name: "ContinuousState",
        **dict.fromkeys(runtime_param_names, "FloatND"),
    }
    if conditioned is not None:
        args[conditioned[0].on] = "DiscreteState"
    return _AR1NextState(
        qname=qname,
        state_name=state_name,
        args=args,
        fixed_params=fixed_params,
        runtime_param_names=runtime_param_names,
        conditioned=conditioned,
        draw_shock=grid.draw_shock,
    )


@dataclass(frozen=True, kw_only=True, eq=False)
class _AR1NextState:
    """Draw an AR(1) process's next value from its current value and a key."""

    qname: str
    """Qualified `<target>__next_<state>` name of the transition."""
    state_name: StateName
    """Name of the process state whose current value the draw conditions on."""
    args: dict[str, str]
    """Argument names and annotations, in signature order."""
    fixed_params: dict[str, Any]
    """Process parameters fixed at construction."""
    runtime_param_names: dict[str, str]
    """Mapping of qualified runtime-param names to the process's own names."""
    conditioned: tuple[StateConditioned, Float1D] | None
    """The state-conditioned sigma array, or `None` for an unconditioned process."""
    draw_shock: Callable[..., ContinuousState]
    """The process's own draw."""

    def __post_init__(self) -> None:
        _publish_signature(
            target=self,
            args=self.args,
            return_annotation="ContinuousState",
            name="next_stochastic_state",
        )

    @no_type_check
    def __call__(self, **kwargs: FloatND) -> ContinuousState:
        params = MappingProxyType(
            {
                **self.fixed_params,
                **{raw: kwargs[qn] for qn, raw in self.runtime_param_names.items()},
                **_conditioned_sigma(conditioned=self.conditioned, kwargs=kwargs),
            }
        )
        return self.draw_shock(
            params=params,
            key=kwargs[f"key_{self.qname}"],
            current_value=kwargs[self.state_name],
        )


def _create_iid_next_func(
    *,
    qname: str,
    state_name: StateName,
    grid: _IIDProcess,
    conditioned: tuple[StateConditioned, Float1D] | None = None,
) -> StochasticNextFunction:
    fixed_params = dict(grid.params)
    runtime_param_names = {
        qname_from_tree_path((state_name, p)): p for p in grid.params_to_pass_at_runtime
    }
    args: dict[str, str] = {
        f"key_{qname}": "PRNGKeyND",
        **dict.fromkeys(runtime_param_names, "FloatND"),
    }
    if conditioned is not None:
        args[conditioned[0].on] = "DiscreteState"
    return _IIDNextState(
        qname=qname,
        args=args,
        fixed_params=fixed_params,
        runtime_param_names=runtime_param_names,
        conditioned=conditioned,
        draw_shock=grid.draw_shock,
    )


@dataclass(frozen=True, kw_only=True, eq=False)
class _IIDNextState:
    """Draw an IID process's next value from a key."""

    qname: str
    """Qualified `<target>__next_<state>` name of the transition."""
    args: dict[str, str]
    """Argument names and annotations, in signature order."""
    fixed_params: dict[str, Any]
    """Process parameters fixed at construction."""
    runtime_param_names: dict[str, str]
    """Mapping of qualified runtime-param names to the process's own names."""
    conditioned: tuple[StateConditioned, Float1D] | None
    """The state-conditioned sigma array, or `None` for an unconditioned process."""
    draw_shock: Callable[..., ContinuousState]
    """The process's own draw."""

    def __post_init__(self) -> None:
        _publish_signature(
            target=self,
            args=self.args,
            return_annotation="ContinuousState",
            name="next_stochastic_state",
        )

    @no_type_check
    def __call__(self, **kwargs: FloatND) -> ContinuousState:
        params = MappingProxyType(
            {
                **self.fixed_params,
                **{raw: kwargs[qn] for qn, raw in self.runtime_param_names.items()},
                **_conditioned_sigma(conditioned=self.conditioned, kwargs=kwargs),
            }
        )
        return self.draw_shock(
            params=params,
            key=kwargs[f"key_{self.qname}"],
        )


def _conditioned_sigma(
    *, conditioned: tuple[StateConditioned, Float1D] | None, kwargs: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Return `{"sigma": <the value the conditioning state selects>}`, else `{}`.

    Overrides the scalar `sigma` that places the nodes, so the draw uses the same
    per-category value the solve-side transition row was built from.
    """
    if conditioned is None:
        return {}
    sc, sigma_by_code = conditioned
    return {"sigma": gather_sigma(sigma_by_code=sigma_by_code, code=kwargs[sc.on])}


def _publish_signature(
    *,
    target: object,
    args: Mapping[str, str],
    return_annotation: str,
    name: str,
) -> None:
    """Publish a DAG-facing signature, annotations, and name on a callable instance.

    The instance is a frozen dataclass, so the attributes go through
    `object.__setattr__`. `args` maps each argument name to its annotation
    string; the DAG reads both the signature and `__annotations__`.
    """
    signature = inspect.Signature(
        [
            inspect.Parameter(
                arg, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=annotation
            )
            for arg, annotation in args.items()
        ],
        return_annotation=return_annotation,
    )
    object.__setattr__(target, "__signature__", signature)
    object.__setattr__(target, "__annotations__", {**args, "return": return_annotation})
    object.__setattr__(target, "__name__", name)
