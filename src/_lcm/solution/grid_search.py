"""The default grid-search solver.

`GridSearch` runs the max-Q-over-a grid search. Its `build_period_kernels`
returns one `PeriodKernel` per period. Eligible hard-max, collective, and EV1 solve
kernels declare their canonical action product for blockwise execution, and the engine
binds the block width before lowering. Folded or co-mapped states and value-dependent
edges retain the dense kernel. The adapter assembles the resulting `KernelResult`
outside JIT.

The kernel-building imports (`jax`, `get_max_Q_over_a`) are function-local so
the public `lcm.solvers` façade stays a thin re-export that pulls in no
numerical engine modules.
"""

import functools
import inspect
import logging
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import cast

import jax
import jax.numpy as jnp
from beartype import beartype

from _lcm.beartype_conf import REGIME_CONF
from _lcm.constraints.routes import (
    ConstraintRoute,
    ConstraintRouteKey,
    ConstraintSite,
)
from _lcm.continuation import EGMContinuationLayout
from _lcm.engine import StateActionSpace
from _lcm.execution.core_program import (
    CoreExecutionRequirements,
    CoreProgram,
    StreamableProductAxis,
)
from _lcm.execution.output_layout import (
    DISSOLUTION_FLAG,
    VALUE,
)
from _lcm.processes.base import _ContinuousStochasticProcess
from _lcm.solution.action_reduction import (
    COLLECTIVE_HARD_MAX_REDUCTION,
    HARD_MAX_REDUCTION,
)
from _lcm.solution.action_streaming import (
    GridSearchEV1ActionReduction,
)
from _lcm.solution.contract import (
    ConstraintRouteContext,
    ContinuationPayload,
    KernelResult,
    PeriodKernel,
    SolutionKernels,
    Solver,
    SolverBuildContext,
    simulation_route,
)
from _lcm.typing import (
    ActionName,
    FlatParams,
    MaxQOverAFunction,
    RegimeName,
    StateName,
)
from lcm.ages import AgeGrid
from lcm.typing import (
    FloatND,
)

_ACTION_AXIS_NAME = "action"
_ACTION_WIDTH_KEYWORD = "_lcm_action_block_width"


@beartype(conf=REGIME_CONF)
@dataclass(frozen=True, kw_only=True)
class GridSearch(Solver):
    """Grid-search solver over the full state-action product (the default)."""

    @property
    def supports_transition_local_lotteries(self) -> bool:
        """Grid search enumerates transition-local lotteries inside Q."""
        return True

    @property
    def egm_continuation_layout(self) -> EGMContinuationLayout:
        """A brute child publishes one action-maxed row on its state grid."""
        return EGMContinuationLayout(
            retains_discrete_action_rows=False,
            rows_share_state_grid=True,
        )

    def build_constraint_routes(
        self, *, context: ConstraintRouteContext
    ) -> tuple[ConstraintRoute, ...]:
        """Declare the one route grid search walks: whole candidates, nothing hidden.

        The search enumerates the entire state-action product, so every name a
        constraint could read is bound where it evaluates. There is no inner
        stage to fall through to and nothing its construction enforces on a
        constraint's behalf, which is why the route is one unrestricted site
        carrying neither a proof nor a compiler.

        One route, not one per period: the search does not resolve its pool
        differently at any age, so a per-period key would put an entry per
        period in the plan where there is a single fact.
        """
        if context.phase == "simulate":
            return (simulation_route(context=context, solver_path=("grid_search",)),)
        return (
            ConstraintRoute(
                key=ConstraintRouteKey(
                    phase="solve",
                    period_group=None,
                    solver_path=("grid_search",),
                ),
                sites=(
                    ConstraintSite(
                        stage="state_action",
                        function_pool=context.functions,
                        available_names=None,
                    ),
                ),
            ),
        )

    def build_period_kernels(self, *, context: SolverBuildContext) -> SolutionKernels:
        """Build one max-Q-over-a period adapter per period.

        Periods sharing the same Q_and_F object reuse the same dense and streamed
        function objects so the execution layer can deduplicate their lowerings.
        """
        from _lcm.regime_building.max_Q_over_a import (  # noqa: PLC0415
            get_max_Q_over_a,
            get_streaming_max_Q_over_a,
        )
        from _lcm.regime_building.processing import (  # noqa: PLC0415
            get_conditioned_fold_weights_by_code,
        )

        built: dict[int, MaxQOverAFunction] = {}
        unwrapped: dict[int, MaxQOverAFunction] = {}
        streamed: dict[int, MaxQOverAFunction] = {}
        result: dict[int, PeriodKernel] = {}
        # Fold weights are the folded process's own marginal distribution, a
        # plain constant computed once here at kernel-build time and never
        # inside the traced core (`_validate_fold_declarations` rejects a
        # runtime-parameterized process). Two shapes, by declaration:
        # - an unconditioned process contributes one row. Its
        #   `compute_transition_probs` returns an `(n_points, n_points)` matrix
        #   whose every row is that marginal — the "IID" part — so row 0 is it.
        # - a `StateConditioned` `sigma` contributes one row per category of
        #   the conditioning state, ordered by that categorical's integer code,
        #   which the fold reduction gathers along the conditioning axis.
        fold_weights: dict[StateName, FloatND] = {}
        fold_conditioning: dict[StateName, StateName] = {}
        for name in context.fold_state_names:
            process = cast("_ContinuousStochasticProcess", context.grids[name])
            if process.state_conditioned is None:
                fold_weights[name] = process.get_transition_probs()[0]
            else:
                fold_weights[name] = get_conditioned_fold_weights_by_code(
                    name=name, grid=process, grids=context.grids
                )
                fold_conditioning[name] = process.state_conditioned.on
        stream_actions = _supports_action_streaming(context=context)
        action_names = context.state_action_space.action_names
        action_extents = context.state_action_space.actions_grid_shapes
        for period, Q_and_F in context.Q_and_F_functions.items():
            q_id = id(Q_and_F)
            if q_id not in built:
                func = get_max_Q_over_a(
                    Q_and_F=Q_and_F,
                    batch_sizes={
                        name: grid.batch_size
                        for name, grid in context.grids.items()
                        if name in context.state_action_space.state_names
                    },
                    action_names=context.state_action_space.action_names,
                    state_names=context.state_action_space.state_names,
                    n_discrete_action_axes=len(
                        context.state_action_space.discrete_actions
                    ),
                    has_taste_shocks=context.has_taste_shocks,
                    co_map_state_names=context.co_map_state_names,
                    co_map_v_arr_in_axes=context.co_map_v_arr_in_axes,
                    stakeholders=context.stakeholders,
                    pareto_weights=context.pareto_weights,
                    fold_state_names=context.fold_state_names,
                    fold_weights=MappingProxyType(fold_weights),
                    fold_conditioning=MappingProxyType(fold_conditioning),
                )
                built[q_id] = jax.jit(func) if context.enable_jit else func
                unwrapped[q_id] = func
                if stream_actions:
                    streamed[q_id] = get_streaming_max_Q_over_a(
                        Q_and_F=Q_and_F,
                        batch_sizes={
                            name: grid.batch_size
                            for name, grid in context.grids.items()
                            if name in context.state_action_space.state_names
                        },
                        action_names=action_names,
                        state_names=context.state_action_space.state_names,
                        n_discrete_action_axes=len(
                            context.state_action_space.discrete_actions
                        ),
                        has_taste_shocks=context.has_taste_shocks,
                        co_map_state_names=context.co_map_state_names,
                        co_map_v_arr_in_axes=context.co_map_v_arr_in_axes,
                        stakeholders=context.stakeholders,
                        pareto_weights=context.pareto_weights,
                        fold_state_names=context.fold_state_names,
                        fold_weights=MappingProxyType(fold_weights),
                        fold_conditioning=MappingProxyType(fold_conditioning),
                    )
            result[period] = _GridSearchPeriodKernel(
                core=built[q_id],
                unwrapped_core=unwrapped[q_id],
                streamed_core=streamed.get(q_id),
                action_names=action_names,
                action_extents=action_extents,
                regime_name=context.regime_name,
                collective=context.stakeholders is not None,
                same_period_ref_regimes=context.same_period_ref_regimes,
                has_taste_shocks=context.has_taste_shocks,
                n_discrete_action_axes=len(context.state_action_space.discrete_actions),
                edge_reference_regimes=context.edge_reference_regimes,
                edge_target_regimes=context.edge_target_regimes,
            )
        return SolutionKernels(period_kernels=MappingProxyType(result))


def _supports_action_streaming(*, context: SolverBuildContext) -> bool:
    """Return whether this regime supports the streamed solve route."""
    action_extents = context.state_action_space.actions_grid_shapes
    return (
        bool(context.state_action_space.action_names)
        and math.prod(action_extents) > 1
        and all(
            _ACTION_WIDTH_KEYWORD not in inspect.signature(Q_and_F).parameters
            for Q_and_F in context.Q_and_F_functions.values()
        )
        and (
            not context.has_taste_shocks
            or (
                context.enable_jit
                and context.stakeholders is None
                and bool(context.state_action_space.discrete_actions)
            )
        )
        and not context.fold_state_names
        and not context.co_map_state_names
        and not context.co_map_v_arr_in_axes
        and not context.same_period_ref_regimes
        and not context.edge_reference_regimes
        and not context.edge_target_regimes
    )


@dataclass(frozen=True, kw_only=True)
class _GridSearchPeriodKernel:
    """The grid-search period adapter — wraps one max-Q-over-a core.

    Closes over the regime name and shared core. Calling the dense fallback evaluates
    Q on the full state-action product; eligible planned execution instead streams
    that product. Both publish the value array, plus the dissolution flag for a
    collective regime, and neither publishes a continuation or simulation policy.
    """

    core: Callable
    """The shared jitted max-Q-over-a core (`id`-deduped across periods)."""

    unwrapped_core: Callable | None = None
    """The same dense core before GridSearch's JIT wrapper."""

    streamed_core: Callable | None = None
    """Action-streaming core, or `None` for an unsupported route."""

    action_names: tuple[ActionName, ...] = ()
    """Canonical C-order action-coordinate names for the streamed core."""

    action_extents: tuple[int, ...] = ()
    """Static coordinate extents aligned with `action_names`."""

    regime_name: RegimeName
    """Name of the regime whose flat params this adapter projects."""

    collective: bool = False
    """Whether the core is a collective (stakeholder-valued) reduction.

    A collective core returns the pair `(V, D)` —
    the stakeholder-axis value array plus the boolean dissolution flag — instead
    of the plain V array; the adapter unpacks it into the `KernelResult`.
    `False` keeps the singleton default byte-identical.
    """

    has_taste_shocks: bool = False
    """Whether this singleton core uses EV1 branch smoothing."""

    n_discrete_action_axes: int = 0
    """Number of leading discrete coordinates in the canonical action product."""

    edge_reference_regimes: tuple[RegimeName, ...] = ()
    """Regimes a gated edge reads a projected value from, or empty.

    A gate reference and a leg fallback both name another regime's value at
    coordinates a projection produces. Neither is tabulated on the target's
    grid, so both are read where the source lands — inside the source's own
    kernel — at the value of the period the source lands in. The rolled V
    mapping already carries that array; these names are what pick it out and
    thread each reference regime's OWN grid params beside it.
    """

    same_period_ref_regimes: tuple[RegimeName, ...] = ()
    """Reference regimes whose same-period V the core reads, or empty.

    When non-empty, `__call__` forwards the solve loop's
    `same_period_regime_to_V_arr` mapping into the core, and
    `build_lower_args` supplies matching zero templates (reusing the
    period-invariant `next_regime_to_V_arr` templates — a regime's V shape does
    not change across periods, so the next-period template is also the correct
    same-period lowering shape).
    """

    edge_target_regimes: tuple[RegimeName, ...] = ()
    """Target regimes reached through a gated edge, or empty.

    When non-empty, `build_lower_args` and `__call__`
    replace each such target's entry in `next_regime_to_V_arr` with the gated
    continuation object `Wbar` supplied under `edge_regime_to_V_arr` (a
    per-source template at lowering, the freshly folded array at run time), so
    the source's continuation reads `Wbar` in place of the raw target V with
    no change to the compiled core. Empty keeps every other kernel
    byte-identical.
    """

    def _with_edge_substitution(
        self,
        *,
        next_regime_to_V_arr: Mapping[RegimeName, FloatND],
        edge_regime_to_V_arr: Mapping[RegimeName, FloatND] | None,
    ) -> Mapping[RegimeName, FloatND]:
        """Replace edge targets' raw V with their gated `Wbar`."""
        if not self.edge_target_regimes:
            return next_regime_to_V_arr
        if edge_regime_to_V_arr is None:
            msg = (
                f"Regime '{self.regime_name}' declares gated edges into "
                f"{self.edge_target_regimes} but the solve loop passed no edge "
                "continuation arrays."
            )
            raise RuntimeError(msg)
        return MappingProxyType(
            {
                name: (
                    edge_regime_to_V_arr[name]
                    if name in self.edge_target_regimes
                    else arr
                )
                for name, arr in next_regime_to_V_arr.items()
            }
        )

    def cores(self) -> Mapping[str, Callable]:
        """Return the single max-Q-over-a core under the `"main"` key."""
        return MappingProxyType({"main": self.core})

    def build_core_program(
        self,
        *,
        core_key: str,
        arguments: Mapping[str, object],
    ) -> CoreProgram | None:
        """Declare the eligible GridSearch action product for engine planning.

        Unsupported numerical routes keep using the dense core. The program
        snapshots exactly the arguments built by this adapter; the planner owns
        only the static block width and binds it before lowering.
        """
        if core_key != "main":
            msg = f"GridSearch has no core named {core_key!r}."
            raise KeyError(msg)
        if self.streamed_core is None:
            return None
        return CoreProgram(
            function=self.streamed_core,
            arguments=arguments,
            requirements=CoreExecutionRequirements(
                streamable_axes=(
                    StreamableProductAxis(
                        name=_ACTION_AXIS_NAME,
                        coordinate_names=self.action_names,
                        coordinate_extents=self.action_extents,
                        canonical_order="c",
                        reduction=(
                            COLLECTIVE_HARD_MAX_REDUCTION
                            if self.collective
                            else GridSearchEV1ActionReduction(
                                n_discrete_action_axes=self.n_discrete_action_axes
                            )
                            if self.has_taste_shocks
                            else HARD_MAX_REDUCTION
                        ),
                        width_keyword=_ACTION_WIDTH_KEYWORD,
                    ),
                )
            ),
            output_roles=((VALUE, DISSOLUTION_FLAG) if self.collective else VALUE),
        )

    def output_roles(self, *, core_key: str) -> object:
        """Name the core output leaves whose concrete layout the engine owns."""
        if core_key != "main":
            msg = f"GridSearch has no core named {core_key!r}."
            raise KeyError(msg)
        return (VALUE, DISSOLUTION_FLAG) if self.collective else VALUE

    def core_for_output_layout(self, *, core_key: str) -> Callable:
        """Return the GridSearch-owned raw core for output-sharded lowering."""
        if core_key != "main":
            msg = f"GridSearch has no core named {core_key!r}."
            raise KeyError(msg)
        if self.unwrapped_core is None:
            msg = (
                "This GridSearch adapter has no raw core for output-layout "
                "lowering. Build it through GridSearch.build_period_kernels()."
            )
            raise RuntimeError(msg)
        return self.unwrapped_core

    def with_fixed_params(
        self, *, fixed_flat_params: FlatParams
    ) -> _GridSearchPeriodKernel:
        """Bind the regime's fixed params into the core.

        The core threads its `**kwargs` into the per-combo pool, so binding the
        regime's own fixed params restores the values removed from the live
        `flat_params`; the captured functions read only the keys they need.
        """
        regime_fixed = dict(
            fixed_flat_params.get(self.regime_name, MappingProxyType({}))
        )
        if not regime_fixed:
            return self
        return replace(
            self,
            core=functools.partial(self.core, **regime_fixed),
            unwrapped_core=(
                None
                if self.unwrapped_core is None
                else functools.partial(self.unwrapped_core, **regime_fixed)
            ),
            streamed_core=(
                None
                if self.streamed_core is None
                else functools.partial(self.streamed_core, **regime_fixed)
            ),
        )

    def build_lower_args(
        self,
        *,
        core_key: str = "main",  # noqa: ARG002
        state_action_space: StateActionSpace,
        next_regime_to_V_arr: Mapping[RegimeName, FloatND],
        next_regime_to_continuation: Mapping[RegimeName, ContinuationPayload],  # noqa: ARG002
        flat_params: FlatParams,
        period: int,
        ages: AgeGrid,
        edge_regime_to_V_arr: Mapping[RegimeName, FloatND] | None = None,
    ) -> Mapping[str, object]:
        """Build the core's lowering arguments: the full state-action product.

        A regime that declares same-period references (`same_period_ref_regimes`
        non-empty) has those reference V arrays lowered with the zero templates
        already built for `next_regime_to_V_arr` — a reference regime's
        same-period array carries exactly its own (period-invariant) V shape and
        sharding. A gated-edge source (`edge_target_regimes` non-empty) has each
        edge target's continuation lowered with its `Wbar` template instead of
        the raw target V: the target's grid plus the source's stakeholder axis.

        The two slots stay independent when a regime both references and gates
        into the SAME regime: the solve loop passes that regime's own V under
        the same-period slot and its `Wbar` under the continuation slot, and
        for a collective source the two differ in rank. So the same-period
        templates read the raw mapping, never the edge-substituted one.
        """
        edge_substituted_V_arr = self._with_edge_substitution(
            next_regime_to_V_arr=next_regime_to_V_arr,
            edge_regime_to_V_arr=edge_regime_to_V_arr,
        )
        lower_args: dict[str, object] = {
            **dict(state_action_space.states),
            **dict(state_action_space.actions),
            "next_regime_to_V_arr": edge_substituted_V_arr,
            **dict(flat_params[self.regime_name]),
            "period": jnp.int32(period),
            "age": ages.values[period],
        }
        if self.same_period_ref_regimes:
            lower_args["same_period_regime_to_V_arr"] = MappingProxyType(
                {
                    regime_name: next_regime_to_V_arr[regime_name]
                    for regime_name in self.same_period_ref_regimes
                }
            )
            lower_args["same_period_regime_to_params"] = self._same_period_params(
                flat_params=flat_params
            )
        lower_args.update(
            self._edge_reference_args(
                next_regime_to_V_arr=next_regime_to_V_arr, flat_params=flat_params
            )
        )
        return lower_args

    def _edge_reference_args(
        self,
        *,
        next_regime_to_V_arr: Mapping[RegimeName, FloatND],
        flat_params: FlatParams,
    ) -> dict[str, object]:
        """Build the edge-reference channel, empty for a regime that has none.

        A gate reference and a leg fallback are read where the source lands, at
        the value of the period it lands in — which backward induction has
        already solved and rolled into `next_regime_to_V_arr`. Lowering and the
        call share this builder, so the pytree the core is compiled against is
        the pytree it is called with.

        Takes the RAW rolled mapping, never the edge-substituted one: a
        reference regime may also be an edge target, whose entry the
        substitution replaces with that edge's `Wbar`.
        """
        if not self.edge_reference_regimes:
            return {}
        return {
            "edge_reference_regime_to_V_arr": MappingProxyType(
                {
                    regime_name: next_regime_to_V_arr[regime_name]
                    for regime_name in self.edge_reference_regimes
                }
            ),
            "edge_reference_regime_to_params": MappingProxyType(
                {
                    regime_name: flat_params[regime_name]
                    for regime_name in self.edge_reference_regimes
                }
            ),
        }

    def _same_period_params(
        self, *, flat_params: FlatParams
    ) -> MappingProxyType[RegimeName, Mapping[str, object]]:
        """Each reference regime's OWN flat params, for its own grid.

        A same-period reference reader interpolates the REFERENCE regime's V over
        the REFERENCE regime's grid, so its runtime grid helpers (an
        `IrregSpacedGrid(pass_points_at_runtime=True)` reference state's points)
        are the reference regime's parameters — not this regime's, whose params
        are the only ones splatted into the core. Threaded per regime name under
        `Q_and_F.SAME_PERIOD_PARAMS_ARG`, exactly like the same-period V arrays
        beside it; see that constant for the defect this ends.
        """
        return MappingProxyType(
            {
                regime_name: flat_params[regime_name]
                for regime_name in self.same_period_ref_regimes
            }
        )

    def __call__(
        self,
        *,
        compiled_cores: Mapping[str, Callable],
        state_action_space: StateActionSpace,
        next_regime_to_V_arr: Mapping[RegimeName, FloatND],
        next_regime_to_continuation: Mapping[RegimeName, ContinuationPayload],  # noqa: ARG002
        flat_params: FlatParams,
        period: int,
        ages: AgeGrid,
        logger: logging.Logger,  # noqa: ARG002
        same_period_regime_to_V_arr: Mapping[RegimeName, FloatND] | None = None,
        edge_regime_to_V_arr: Mapping[RegimeName, FloatND] | None = None,
    ) -> KernelResult:
        """Evaluate the grid search and assemble the `KernelResult`.

        `same_period_regime_to_V_arr` is passed by the solve loop only for a
        regime declaring `same_period_refs`; `edge_regime_to_V_arr` only for
        a regime declaring `gated_edges` (substituted into
        `next_regime_to_V_arr` before the core call). Every other kernel keeps
        the uniform `PeriodKernel` call signature.
        """
        raw_next_regime_to_V_arr = next_regime_to_V_arr
        next_regime_to_V_arr = self._with_edge_substitution(
            next_regime_to_V_arr=next_regime_to_V_arr,
            edge_regime_to_V_arr=edge_regime_to_V_arr,
        )
        extra_kwargs: dict[str, object] = self._edge_reference_args(
            next_regime_to_V_arr=raw_next_regime_to_V_arr, flat_params=flat_params
        )
        if self.same_period_ref_regimes:
            if same_period_regime_to_V_arr is None:
                msg = (
                    f"Regime '{self.regime_name}' declares same_period_refs on "
                    f"{self.same_period_ref_regimes} but the solve loop passed "
                    "no same-period V arrays."
                )
                raise RuntimeError(msg)
            extra_kwargs["same_period_regime_to_V_arr"] = same_period_regime_to_V_arr
            extra_kwargs["same_period_regime_to_params"] = self._same_period_params(
                flat_params=flat_params
            )
        out = compiled_cores["main"](
            **state_action_space.states,
            **state_action_space.actions,
            next_regime_to_V_arr=next_regime_to_V_arr,
            **flat_params[self.regime_name],
            period=jnp.int32(period),
            age=ages.values[period],
            **extra_kwargs,
        )
        if self.collective:
            # The collective core returns the pair
            # (stakeholder-axis V, dissolution flag D).
            V_arr, dissolution = out
            return KernelResult(V_arr=V_arr, dissolution=dissolution)
        return KernelResult(V_arr=out)
