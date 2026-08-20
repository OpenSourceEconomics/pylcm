"""The one-asset EGM solver.

`EGM` runs the single-asset endogenous grid method for a regime with
one continuous (Euler) state and no discrete kinks — the specialization whose
step needs no upper envelope. The kernel-building imports are function-local
so the public `lcm.solvers` façade stays a thin re-export that pulls in no
numerical engine modules.
"""

import functools
import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import cast

import jax
import jax.numpy as jnp
from beartype import beartype

from _lcm.beartype_conf import REGIME_CONF
from _lcm.constraints.bounds import proves_the_savings_grids_lower_bound
from _lcm.constraints.routes import (
    ConstraintRoute,
    ConstraintRouteKey,
    ConstraintSite,
)
from _lcm.continuation import EGMContinuationSpec
from _lcm.egm.carry import EGMCarry
from _lcm.engine import StateActionSpace
from _lcm.grids import ContinuousGrid
from _lcm.solution.continuation_target import (
    _period_to_continuation_target,
    _union_fixed_params,
    _union_free_params,
    target_period_grid,
)
from _lcm.solution.contract import (
    ConstraintRouteContext,
    ContinuationPayload,
    KernelResult,
    OneMarginSolver,
    PeriodKernel,
    SolutionKernels,
    SolverBuildContext,
    SolverModelContext,
    _BoundLiquidMargin,
    bind_roles,
    simulation_route,
)
from _lcm.typing import (
    EconFunction,
    EconFunctionsMapping,
    FlatParams,
    RegimeName,
)
from lcm.ages import AgeGrid
from lcm.exceptions import ModelInitializationError
from lcm.typing import (
    ActionName,
    Float1D,
    FloatND,
    FunctionName,
    StateName,
)


@beartype(conf=REGIME_CONF)
@dataclass(frozen=True, kw_only=True)
class EGM(OneMarginSolver):
    """Endogenous-grid solver for a 1-D consumption--saving regime.

    A regime with exactly one continuous state (the liquid wealth), one
    continuous consumption action, and no discrete choice is a plain
    consumption--saving problem. The single continuous state needs no upper
    envelope: inverting the consumption Euler equation on the post-decision
    savings grid and mapping the resulting endogenous wealth back onto the
    regular grid solves the period exactly. The step carries the marginal
    value of liquid backward (the envelope theorem makes it exact, unlike a
    finite difference of a coarse value array), so each period both reads its
    continuation's marginal and publishes its own.
    """

    savings_grid: ContinuousGrid
    """Exogenous post-decision savings grid; its lower bound is the borrowing limit.

    Zero for a household that cannot borrow, minus the limit for one that can.
    The corner is read off this bound rather than assumed to be zero savings.
    """

    def _with_liquid_margin(self, margin: _BoundLiquidMargin) -> _BoundEGM:
        """Bind regime-owned DAG names without exposing them on public `EGM`."""
        return cast(
            "_BoundEGM",
            bind_roles(
                solver=self,
                role_type=_BoundEGM,
                continuous_state=margin.state,
                continuous_action=margin.action,
                resources=margin.resources,
                post_decision_function=margin.post_decision_state,
            ),
        )

    @property
    def requires_continuation(self) -> bool:
        """The 1-D EGM step reads its continuation's marginal value of liquid."""
        return True

    def validate_model(  # noqa: C901, PLR0912, PLR0915
        self, *, context: SolverModelContext
    ) -> None:
        """Check the preconditions the envelope-free EGM kernel assumes.

        Plain EGM is specialized to cash-on-hand as its continuous state: the
        kernel constructs its endogenous grid as `state = action + savings`, so
        `resources(state) == state` and `post_decision(state, action) ==
        state - action` are **assumptions of the method**, not properties it
        verifies and not capabilities it offers. A model whose resources map
        genuinely transforms the state belongs on DCEGM or GridSearch, or
        should redefine its state as cash-on-hand.

        What happens here is a spot check for ordinary specification mistakes,
        and it is a diagnostic rather than a proof. Resources is evaluated at
        every represented state node; the post-decision identity is sampled.
        Neither establishes an identity *between* nodes, and the kernel can
        evaluate off-grid, so a callable that agrees wherever it is checked and
        differs elsewhere will be admitted and then silently not applied. That
        is a limit of finite evaluation, not something a denser probe closes —
        so this must not be read as a guarantee that every incompatible
        callable is rejected.
        """
        bound = cast("_BoundEGM", self)
        from dags import concatenate_functions  # noqa: PLC0415

        from _lcm.egm.validation import (  # noqa: PLC0415
            _call_with_varied,
            _continuous_non_process_names,
            _dag_ancestors,
            _grid_sample,
            _isclose,
            _resolve_solve_functions,
            _solve_grids,
            fail_if_custom_koopmans_aggregator,
        )

        regime_name = context.regime_name
        user_regime = context.user_regimes[regime_name]
        fail_if_custom_koopmans_aggregator(
            regime_name=regime_name,
            user_regime=user_regime,
            solver_name="EGM",
        )
        if user_regime.terminal:
            msg = (
                f"Regime '{regime_name}' is terminal but configured with EGM. "
                "Terminal regimes have no optimization problem; remove the "
                "solver setting."
            )
            raise ModelInitializationError(msg)

        state_grids = _solve_grids(slot=user_regime.states)
        action_grids = _solve_grids(slot=user_regime.actions)
        continuous_states = _continuous_non_process_names(grids=state_grids)
        continuous_actions = _continuous_non_process_names(grids=action_grids)
        if len(continuous_states) != 1:
            msg = (
                f"EGM regime '{regime_name}' must have exactly one continuous "
                f"state, but has {len(continuous_states)}: "
                f"{list(continuous_states)}."
            )
            raise ModelInitializationError(msg)
        if len(continuous_actions) != 1:
            msg = (
                f"EGM regime '{regime_name}' must have exactly one continuous "
                f"action, but has {len(continuous_actions)}: "
                f"{list(continuous_actions)}."
            )
            raise ModelInitializationError(msg)
        discrete_states = sorted(set(state_grids) - set(continuous_states))
        if discrete_states:
            msg = (
                f"EGM regime '{regime_name}' has discrete or process states "
                f"{discrete_states}. The envelope-free one-row kernel publishes "
                "a one-dimensional continuation and cannot carry those axes; "
                "use DCEGM or GridSearch."
            )
            raise ModelInitializationError(msg)
        discrete_actions = sorted(set(action_grids) - set(continuous_actions))
        if discrete_actions:
            msg = (
                f"EGM regime '{regime_name}' has discrete actions "
                f"{discrete_actions}. A discrete choice makes the candidate "
                "correspondence multi-valued; use DCEGM."
            )
            raise ModelInitializationError(msg)

        liquid_state = continuous_states[0]
        consumption_action = continuous_actions[0]
        if bound.post_decision_function not in user_regime.functions:
            msg = (
                f"EGM regime '{regime_name}' is missing the declared "
                f"post-decision function '{bound.post_decision_function}'."
            )
            raise ModelInitializationError(msg)
        functions = _resolve_solve_functions(user_regime=user_regime)
        post_func = functions[bound.post_decision_function]
        post_ancestors = _dag_ancestors(functions=functions, target_func=post_func)
        missing_roles = sorted({liquid_state, consumption_action} - set(post_ancestors))
        if missing_roles:
            msg = (
                f"The post-decision function '{bound.post_decision_function}' "
                f"of EGM regime '{regime_name}' must depend on the state "
                f"'{liquid_state}' and action '{consumption_action}'; its DAG "
                f"does not reach {missing_roles}."
            )
            raise ModelInitializationError(msg)

        if user_regime.constraints:
            constraint_names = sorted(user_regime.constraints)
            msg = (
                f"EGM regime '{regime_name}' declares constraints "
                f"{constraint_names}. Plain EGM evaluates no user constraint; "
                "encode the borrowing limit in `savings_grid.start` and the "
                "budget identity in the post-decision function, or use "
                "GridSearch."
            )
            raise ModelInitializationError(msg)

        utility_ancestors = _dag_ancestors(
            functions=functions, target_func=functions["utility"]
        )
        if liquid_state in utility_ancestors:
            msg = (
                f"The utility DAG of EGM regime '{regime_name}' depends on "
                f"the continuous state '{liquid_state}'. The envelope-free "
                "kernel evaluates utility as a function of the continuous "
                f"action '{consumption_action}' only."
            )
            raise ModelInitializationError(msg)

        composed_post = concatenate_functions(
            functions, targets=bound.post_decision_function
        )
        post_arguments = set(inspect.signature(composed_post).parameters)
        expected_arguments = {liquid_state, consumption_action}
        if post_arguments != expected_arguments:
            msg = (
                f"The post-decision DAG '{bound.post_decision_function}' of EGM "
                f"regime '{regime_name}' must be exactly a function of "
                f"'{liquid_state}' and '{consumption_action}', but its leaf "
                f"arguments are {sorted(post_arguments)}."
            )
            raise ModelInitializationError(msg)
        state_sample = _grid_sample(grid=state_grids[liquid_state])
        action_sample = _grid_sample(grid=action_grids[consumption_action])
        x64_enabled = bool(jax.config.read("jax_enable_x64"))
        atol = 1e-8 if x64_enabled else 1e-4
        rtol = 1e-6 if x64_enabled else 1e-3
        composed_resources = concatenate_functions(functions, targets=bound.resources)
        resources_arguments = set(inspect.signature(composed_resources).parameters)
        if resources_arguments != {liquid_state}:
            msg = (
                f"The resources DAG '{bound.resources}' of EGM regime "
                f"'{regime_name}' must be exactly a function of "
                f"'{liquid_state}', but its leaf arguments are "
                f"{sorted(resources_arguments)}. The envelope-free kernel reads "
                f"resources off the liquid state itself, so anything the "
                f"declared resources add or subtract is silently ignored; use "
                f"DCEGM or GridSearch."
            )
            raise ModelInitializationError(msg)
        # Every represented state node, not a sample: resources takes exactly
        # the liquid state, so one vectorized call costs no more than a few
        # scalar ones and catches a mistake at any tabulated point. It remains
        # a diagnostic -- the kernel can evaluate off-grid, and no finite set of
        # evaluations establishes an identity between the nodes.
        all_state_nodes = state_grids[liquid_state].to_jax()
        resources_at_nodes = _call_with_varied(
            func=composed_resources, fixed={}, varied={liquid_state: all_state_nodes}
        )
        disagreeing = ~jnp.isclose(
            jnp.asarray(resources_at_nodes), all_state_nodes, rtol=rtol, atol=atol
        )
        if bool(jnp.any(disagreeing)):
            first = int(jnp.argmax(disagreeing))
            msg = (
                f"The resources function '{bound.resources}' of EGM regime "
                f"'{regime_name}' must equal the liquid state "
                f"'{liquid_state}'. At {liquid_state}="
                f"{float(all_state_nodes[first])} it returns "
                f"{float(jnp.asarray(resources_at_nodes)[first])}. The "
                f"envelope-free kernel inverts the Euler equation against the "
                f"state directly, so a resources function that transforms it is "
                f"not applied; use DCEGM or GridSearch."
            )
            raise ModelInitializationError(msg)
        for state_value in state_sample:
            for action_value in action_sample:
                actual = _call_with_varied(
                    func=composed_post,
                    fixed={},
                    varied={
                        liquid_state: state_value,
                        consumption_action: action_value,
                    },
                )
                expected = state_value - action_value
                if not _isclose(actual=actual, expected=expected, rtol=rtol, atol=atol):
                    msg = (
                        f"Consumption recovery fails in EGM regime "
                        f"'{regime_name}': '{bound.post_decision_function}' must "
                        f"equal `{liquid_state} - {consumption_action}`. At "
                        f"{liquid_state}={float(state_value)}, "
                        f"{consumption_action}={float(action_value)}, it returns "
                        f"{float(actual)} rather than {float(expected)}."
                    )
                    raise ModelInitializationError(msg)

    def build_constraint_routes(
        self, *, context: ConstraintRouteContext
    ) -> tuple[ConstraintRoute, ...]:
        """Declare the one route plain EGM walks in each phase.

        The envelope-free kernel calls no user predicate anywhere. It builds
        its endogenous grid by inverting the Euler equation on the savings
        grid, and the only requirement it honours is the borrowing limit that
        grid's lowest node already is. Its solve route is therefore a single
        site that evaluates nothing — a place a constraint can be discharged,
        never one where it can be called — and anything that site cannot
        discharge is refused rather than quietly dropped.

        Simulation is a different pipeline with a different answer. There the
        subject's own realized action is in hand, so the feasibility check runs
        over a whole candidate, including the budget constraint the phase
        synthesizes for an endogenous-grid regime.
        """
        bound = cast("_BoundEGM", self)
        proofs = (
            proves_the_savings_grids_lower_bound(
                post_decision=bound.post_decision_function
            ),
        )
        if context.phase == "simulate":
            return (simulation_route(context=context, solver_path=("egm",)),)
        return (
            ConstraintRoute(
                key=ConstraintRouteKey(
                    phase="solve", period_group=None, solver_path=("egm",)
                ),
                sites=(
                    ConstraintSite(
                        stage="savings_stage",
                        function_pool=context.functions,
                        available_names=frozenset(),
                        structural_proofs=proofs,
                    ),
                ),
            ),
        )

    def validate_build(self, *, context: SolverBuildContext) -> None:
        """Check the regime and its targets are 1-D consumption--saving problems.

        The solver's liquid role is filled positionally, which is only
        unambiguous with a single continuous state. The same is asked of each
        target, and for the same reason: with one continuous state on each side
        the correspondence is determined, whatever the two regimes call it.
        Every message reports the regimes' own state names, never the solver's
        internal role vocabulary.

        The regime must also carry no discrete action and keep the default
        Koopmans aggregator: a discrete choice folds the candidate value
        correspondence, which is the case the envelope-free step does not
        cover, and the Euler inversion the step runs is the one that
        aggregator implies.
        """
        from _lcm.egm.validation import (  # noqa: PLC0415
            fail_if_custom_koopmans_aggregator,
        )

        fail_if_custom_koopmans_aggregator(
            regime_name=context.regime_name,
            user_regime=context.user_regimes[context.regime_name],
            solver_name="EGM",
        )
        discrete_actions = tuple(context.state_action_space.discrete_actions)
        if discrete_actions:
            msg = (
                f"EGM regime '{context.regime_name}' has discrete actions "
                f"{sorted(discrete_actions)}. A discrete choice makes the "
                f"candidate value correspondence multi-valued, which the "
                f"envelope-free one-asset step does not solve. Use DCEGM, "
                f"which refines the candidates to their upper envelope."
            )
            raise ModelInitializationError(msg)
        continuous = tuple(
            context.regime_to_v_interpolation_info[
                context.regime_name
            ].continuous_states
        )
        if len(continuous) != 1:
            msg = (
                f"EGM regime '{context.regime_name}' must have exactly one "
                f"continuous state, but has {len(continuous)}: {list(continuous)}. "
                f"Use a solver that handles more than one Euler state, or move the "
                f"extra states to discrete grids."
            )
            raise ModelInitializationError(msg)
        for target in set(_period_to_continuation_target(context=context).values()):
            target_user_regime = context.user_regimes[target]
            if target_user_regime.terminal and target_user_regime.actions:
                msg = (
                    f"EGM regime '{context.regime_name}' continues into terminal "
                    f"regime '{target}', which has actions "
                    f"{list(target_user_regime.actions)}. A terminal EGM carry "
                    "is utility on the state grid and cannot represent a final-"
                    "period optimization; remove the actions or use GridSearch "
                    "for the parent."
                )
                raise ModelInitializationError(msg)
            target_states = tuple(
                context.regime_to_v_interpolation_info[target].continuous_states
            )
            if len(target_states) != 1:
                msg = (
                    f"EGM regime '{context.regime_name}' continues into target "
                    f"regime '{target}', whose continuous states are "
                    f"{sorted(target_states)}. The Euler inversion reads a single "
                    f"continuation state, so the target must declare exactly one — "
                    f"its name need not match this regime's '{continuous[0]}'."
                )
                raise ModelInitializationError(msg)
            target_discrete = tuple(
                context.regime_to_v_interpolation_info[target].discrete_states
            )
            if target_discrete:
                msg = (
                    f"EGM regime '{context.regime_name}' continues into target "
                    f"regime '{target}', whose discrete/process states are "
                    f"{sorted(target_discrete)}. The one-row continuation "
                    "kernel cannot carry those axes."
                )
                raise ModelInitializationError(msg)

    def build_period_kernels(self, *, context: SolverBuildContext) -> SolutionKernels:
        """Build one 1-D EGM period adapter per active period.

        Each period's adapter knows the single deterministic continuation
        target (the transition target whose value array and marginal-utility carry
        feed the Euler inversion).
        """
        bound = cast("_BoundEGM", self)

        savings_grid = self.savings_grid.to_jax()
        liquid_state = next(
            iter(
                context.regime_to_v_interpolation_info[
                    context.regime_name
                ].continuous_states
            )
        )
        liquid_grid = context.grids[liquid_state].to_jax()
        # The regime's single continuous action fills the consumption role, the
        # same positional reading as the liquid state above. It is the argument
        # the regime's felicity, its marginal, and its inverse are functions of.
        consumption_action = next(iter(context.state_action_space.continuous_actions))

        from _lcm.egm.declared_law import build_declared_liquid_law  # noqa: PLC0415

        variable_names = (
            frozenset(context.state_action_space.states)
            | frozenset(context.state_action_space.continuous_actions)
            | frozenset(context.state_action_space.discrete_actions)
        )
        period_to_target = _period_to_continuation_target(context=context)
        cores: dict[RegimeName, Callable] = {}
        laws: dict[RegimeName, Callable[..., tuple[Float1D, Float1D]]] = {}
        period_kernels: dict[int, PeriodKernel] = {}
        # The target's own name for its single continuous state. It is read off
        # that regime: the value grid it is tabulated on and the namespace its
        # transition params live under are both facts about the target, so
        # neither is inherited from this regime's spelling.
        target_state_names = {
            target: next(
                iter(context.regime_to_v_interpolation_info[target].continuous_states)
            )
            for target in period_to_target.values()
        }
        for period, target in period_to_target.items():
            target_state = target_state_names[target]
            if target not in cores:
                core = _build_egm_core(
                    savings_grid=savings_grid,
                    functions=context.functions,
                    koopmans_aggregator=cast(
                        "EconFunction", context.koopmans_aggregator
                    ),
                    consumption_action=consumption_action,
                )
                cores[target] = jax.jit(core) if context.enable_jit else core
                laws[target] = build_declared_liquid_law(
                    transitions=context.transitions,
                    functions=context.functions,
                    post_decision_name=bound.post_decision_function,
                    target=target,
                    target_state=target_state,
                    variable_names=variable_names,
                )
            period_kernels[period] = _EGMPeriodKernel(
                core=cores[target],
                declared_law=laws[target],
                savings_grid=savings_grid,
                regime_name=context.regime_name,
                continuation_target=target,
                liquid_state=liquid_state,
                transition_target_names=tuple(context.transitions),
                next_liquid_grid=target_period_grid(
                    context=context,
                    period=period,
                    target=target,
                    target_state_name=target_state,
                ),
            )
        return SolutionKernels(
            period_kernels=MappingProxyType(period_kernels),
            continuation_spec=EGMContinuationSpec(
                template=_build_one_asset_carry_template(liquid_grid=liquid_grid),
                layout=self.egm_continuation_layout,
            ),
        )


@dataclass(frozen=True, kw_only=True)
class _BoundEGM(EGM):
    """Internal EGM configuration with regime-resolved DAG role names.

    Public `EGM` contains numerical configuration only.  A
    `ConsumptionSavingsRegime` binds its liquid margin into this private
    subclass before model processing, allowing the numerical implementation to
    keep using explicit names without re-exposing them on the public solver.
    """

    continuous_state: StateName
    """Name of the liquid state whose resources the action is drawn from."""

    continuous_action: ActionName
    """Name of the liquid action the endogenous grid is expressed in."""

    resources: FunctionName
    """Name of the function giving resources available for that action."""

    post_decision_function: FunctionName
    """Name of the function giving the savings the exogenous grid spans."""


@dataclass(frozen=True, kw_only=True)
class _EGMPeriodKernel:
    """The 1-D EGM period adapter — wraps the shared `egm_one_asset_step` core.

    Closes over the regime name, the period's single deterministic
    continuation target (whose value array and marginal carry feed the Euler
    inversion), and the transition target names (to union their params).
    Returns a `KernelResult` carrying the value array and the marginal-value
    carry a parent EGM regime interpolates.
    """

    core: Callable
    """The shared jitted 1-D EGM-step core."""

    regime_name: RegimeName
    """Name of the regime whose flat params this adapter projects."""

    continuation_target: RegimeName
    """The regime active next period; its value and marginal continue this one."""

    liquid_state: StateName
    """The regime's own name for the state filling the kernel's liquid role.

    The core takes its state grid under the private keyword `liquid`; this is
    the name the modeller gave it, used to look the grid up in the state-action
    space and to qualify the liquid law's parameters.
    """

    transition_target_names: tuple[RegimeName, ...]
    """Names of the regime's transition targets, whose params are unioned in."""

    next_liquid_grid: Float1D
    """The continuation target's liquid nodes in the *next* period.

    The abscissae of the continuation value and marginal this adapter reads. Equal
    to this period's own grid unless the liquid state is an `AgeSpecializedGrid`.
    """

    declared_law: Callable[..., tuple[Float1D, Float1D]]
    """The regime's own law toward the target, as a function of savings."""

    savings_grid: Float1D
    """The post-decision grid the law is tabulated on."""

    bound_params: Mapping[str, FloatND] = MappingProxyType({})
    """Fixed params bound into the core, kept so the law can read them too."""

    def _law_readings(self, *, flat_params: FlatParams) -> tuple[Float1D, Float1D]:
        """Read the declared law on the savings grid and check it can be inverted.

        Evaluated here rather than inside the compiled core: the readings depend
        on the params alone, and their ordering is a Python-level decision that
        no traced value can make. The check therefore has to see concrete
        numbers, which is exactly what this side of the boundary has.
        """
        from _lcm.egm.declared_law import (  # noqa: PLC0415
            fail_if_declared_law_is_not_increasing,
        )

        next_liquid, marginal_return = self.declared_law(
            savings_grid=self.savings_grid,
            **self.bound_params,
            **_union_free_params(
                flat_params=flat_params,
                regime_name=self.regime_name,
                transition_target_names=self.transition_target_names,
            ),
        )
        fail_if_declared_law_is_not_increasing(
            next_liquid=next_liquid,
            regime_name=self.regime_name,
            target=self.continuation_target,
        )
        return next_liquid, marginal_return

    def cores(self) -> Mapping[str, Callable]:
        """Return the single EGM-step core under the `"main"` key."""
        return MappingProxyType({"main": self.core})

    def with_fixed_params(self, *, fixed_flat_params: FlatParams) -> _EGMPeriodKernel:
        """Bind the regime's and its targets' fixed params into the core."""
        bound = _union_fixed_params(
            fixed_flat_params=fixed_flat_params,
            regime_name=self.regime_name,
            transition_target_names=self.transition_target_names,
        )
        if not bound:
            return self
        return replace(
            self,
            core=functools.partial(self.core, **bound),
            bound_params=MappingProxyType(dict(bound)),
        )

    def build_lower_args(
        self,
        *,
        core_key: str = "main",  # noqa: ARG002
        state_action_space: StateActionSpace,
        next_regime_to_V_arr: Mapping[RegimeName, FloatND],
        next_regime_to_continuation: Mapping[RegimeName, ContinuationPayload],
        flat_params: FlatParams,
        period: int,  # noqa: ARG002
        ages: AgeGrid,  # noqa: ARG002
    ) -> Mapping[str, object]:
        """Build the core's lowering arguments: state, continuation, law, params."""
        next_liquid, marginal_return = self._law_readings(flat_params=flat_params)
        return {
            "liquid": state_action_space.states[self.liquid_state],
            "next_liquid_grid": self.next_liquid_grid,
            "next_liquid": next_liquid,
            "marginal_return": marginal_return,
            "next_value": next_regime_to_V_arr[self.continuation_target],
            "next_marginal": next_regime_to_continuation[
                self.continuation_target
            ].marginal_utility,
            **_union_free_params(
                flat_params=flat_params,
                regime_name=self.regime_name,
                transition_target_names=self.transition_target_names,
            ),
        }

    def __call__(
        self,
        *,
        compiled_cores: Mapping[str, Callable],
        state_action_space: StateActionSpace,
        next_regime_to_V_arr: Mapping[RegimeName, FloatND],
        next_regime_to_continuation: Mapping[RegimeName, ContinuationPayload],
        flat_params: FlatParams,
        period: int,  # noqa: ARG002
        ages: AgeGrid,  # noqa: ARG002
    ) -> KernelResult:
        """Run the 1-D EGM step and assemble the `KernelResult`."""
        next_liquid, marginal_return = self._law_readings(flat_params=flat_params)
        V_arr, carry = compiled_cores["main"](
            liquid=state_action_space.states[self.liquid_state],
            next_liquid_grid=self.next_liquid_grid,
            next_liquid=next_liquid,
            marginal_return=marginal_return,
            next_value=next_regime_to_V_arr[self.continuation_target],
            next_marginal=next_regime_to_continuation[
                self.continuation_target
            ].marginal_utility,
            **_union_free_params(
                flat_params=flat_params,
                regime_name=self.regime_name,
                transition_target_names=self.transition_target_names,
            ),
        )
        return KernelResult(V_arr=V_arr, continuation=carry)


def _build_egm_core(
    *,
    savings_grid: Float1D,
    functions: EconFunctionsMapping,
    koopmans_aggregator: EconFunction,
    consumption_action: ActionName,
) -> Callable:
    """Build the jitted-able 1-D EGM core closing over the savings grid.

    The core reads the state grid under the private role keyword `liquid`, the
    continuation value and marginal, the two readings of the declared law of
    motion (where each savings level lands, and how that landing point moves
    with savings), and the regime's scalar params. It runs `egm_one_asset_step`
    and returns the value array and the marginal-value carry on the liquid grid.

    The law's two readings arrive as arrays rather than being composed here:
    they are properties of the params alone, so the period adapter evaluates
    them once outside the compiled region — which is also what lets their
    ordering be checked, a Python-level decision no traced value can make.

    Preferences and the discount factor come from the regime itself: the
    felicity trio is bound out of `functions` at each call, and beta is read
    off the aggregator's own signature.
    """
    from _lcm.egm.one_asset_egm_step import egm_one_asset_step  # noqa: PLC0415
    from _lcm.egm.preferences import (  # noqa: PLC0415
        NEWTON_ACTION_FLOOR,
        get_discount_factor_reader,
        get_preferences_builder,
        newton_action_ceiling,
    )

    build_preferences = get_preferences_builder(
        functions=functions,
        action_name=consumption_action,
        action_lower=NEWTON_ACTION_FLOOR,
        action_upper=newton_action_ceiling(savings_grid),
    )
    read_discount_factor = get_discount_factor_reader(
        functions=functions, koopmans_aggregator=koopmans_aggregator
    )

    def core(
        *,
        liquid: Float1D,
        next_liquid_grid: Float1D,
        next_value: Float1D,
        next_marginal: Float1D,
        next_liquid: Float1D,
        marginal_return: Float1D,
        **params: FloatND,
    ) -> tuple[Float1D, EGMCarry]:
        step = egm_one_asset_step(
            next_value=next_value,
            next_marginal=next_marginal,
            liquid_grid=liquid,
            next_liquid_grid=next_liquid_grid,
            savings_grid=savings_grid,
            discount_factor=read_discount_factor(params),
            preferences=build_preferences(params),
            next_liquid=next_liquid,
            marginal_return=marginal_return,
        )
        carry = EGMCarry(
            endog_grid=liquid,
            value=step.value,
            marginal_utility=step.marginal,
            taste_shock_scale=jnp.asarray(0.0, dtype=step.value.dtype),
        )
        return step.value, carry

    return core


def _build_one_asset_carry_template(*, liquid_grid: Float1D) -> EGMCarry:
    """Build the all-finite 1-D EGM carry template on the liquid grid."""
    return EGMCarry(
        endog_grid=liquid_grid,
        value=jnp.zeros_like(liquid_grid),
        marginal_utility=jnp.zeros_like(liquid_grid),
        taste_shock_scale=jnp.asarray(0.0, dtype=liquid_grid.dtype),
    )
