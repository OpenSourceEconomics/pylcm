"""Solve current-IID shocks read by an endogenous-grid regime on the node-state axis.

A group named in `Regime.shocks` by a regime whose solver is not `GridSearch` leaves
the current-IID route. Each of its shocks becomes an internal IID node state of every
regime that reads the group, so the ordinary node-state machinery solves every shape it
admits: several states, discrete states and actions, chains of dates, floors. The
parent's continuation integrates the node axis with the rule weights.

The adapter keeps the `shocks=` contract at the model boundary:

- published values integrate every node axis with its rule weights, so no value
  carries a shock axis;
- simulation realises each shock from the addressed stream of the current-IID route
  and applies `BaseInnovations` / `PhysicalValues` inputs with its override producer,
  then hands the physical paths to the node-state simulation;
- regimes, state listings and simulated panels show each group as declared: a reader
  lists the group in `shocks`, has no state for its shocks, and its panel carries the
  realised shocks beside the actions.
"""

import dataclasses
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, fields
from types import MappingProxyType
from typing import ClassVar, cast

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from _lcm.current_iid import CurrentIIDDrawPlan, _draw_base_value
from _lcm.engine import PeriodRegimeSimulationData
from _lcm.processes.base import StateConditioned
from _lcm.processes.iid import (
    LogNormalIIDProcess,
    NormalIIDProcess,
    UniformIIDProcess,
    _IIDProcess,
)
from _lcm.shock_registry import ShockBinding
from _lcm.simulation.current_iid_overrides import (
    ShockPathsInput,
    SimulationInitialConditionsInput,
    split_current_iid_inputs,
)
from _lcm.simulation.initial_conditions import resolve_initial_periods
from _lcm.simulation.random import create_simulation_key, draw_random_seed
from _lcm.simulation.random_address import RandomAddress, draw_row_keys
from _lcm.simulation.time_inputs import lower_initial_time
from _lcm.time import TimeAxis
from _lcm.typing import PRNGKeyND
from lcm._solver_api.contract import ValueArraySchema
from lcm._solver_api.identity import AxisDescriptor, AxisRole
from lcm._solver_api.result import SolutionResult
from lcm.exceptions import InvalidSimulationInputError, ShockDeclarationError
from lcm.regime import Regime
from lcm.result import SimulationResult
from lcm.shocks import (
    BaseInnovations,
    Discrete,
    EqualSpaced,
    EqualSpacedUniform,
    GaussHermite,
    Law,
    LogNormal,
    MonteCarlo,
    Normal,
    PhysicalValues,
    Product,
    QuasiMonteCarlo,
    ShockGroup,
    ShockGroupName,
    ShockName,
    ShockOverride,
    Uniform,
    _ConditionedLookup,
    _Independence,
)
from lcm.solvers import GridSearch
from lcm.typing import (
    BoolND,
    Float1D,
    FloatND,
    RegimeName,
    ScalarFloat,
    ScalarInt,
    StateName,
    UserInitialConditions,
    UserSeed,
)

# Attribute of a published result that holds the solve's node-axis result.
_NODE_AXIS_SOLUTION = "_lcm_node_axis_solution"


@dataclass(frozen=True, kw_only=True)
class NodeAxisAdapter:
    """Record which declared groups a model solves on the node-state axis."""

    declared_groups: Mapping[ShockGroupName, ShockGroup]
    """Every group declared on the model, in declaration order."""
    groups: Mapping[ShockGroupName, ShockGroup]
    """The groups solved on the node-state axis."""
    processes: Mapping[ShockName, _IIDProcess]
    """Internal node state of each adapted shock, named as the shock."""
    regime_shocks: Mapping[RegimeName, tuple[ShockGroupName, ...]]
    """Declared `shocks` of every regime that reads an adapted group."""

    def present_regimes(
        self, *, regimes: Mapping[RegimeName, Regime]
    ) -> MappingProxyType[RegimeName, Regime]:
        """Return the regimes with their declared shock groups and no node states."""
        return MappingProxyType(
            {
                name: regime.replace(
                    states={
                        state: grid
                        for state, grid in regime.states.items()
                        if state not in self.processes
                    },
                    shocks=self.regime_shocks[name],
                )
                if name in self.regime_shocks
                else regime
                for name, regime in regimes.items()
            }
        )

    def present_solution(self, *, result: SolutionResult) -> SolutionResult:
        """Return the published result: values integrated over every node axis.

        The node-axis result stays attached, so simulation replays it by reference.
        """
        schemas = result.metadata.value_schemas
        values = {
            period: {
                regime_name: self._integrate(
                    value=value, axis_names=schemas[period, regime_name].axis_names
                )
                for regime_name, value in regimes.items()
            }
            for period, regimes in result.values.items()  # noqa: PD011 - result store
        }
        published = _copy_result(
            result=result,
            values=values,
            schemas={
                key: self._published_schema(schema) for key, schema in schemas.items()
            },
        )
        object.__setattr__(published, _NODE_AXIS_SOLUTION, result)
        return published

    def node_axis_solution(
        self,
        *,
        solution: SolutionResult,
        state_names: Mapping[RegimeName, tuple[StateName, ...]],
    ) -> SolutionResult:
        """Return the node-axis result a published result replays.

        A result without its attached node-axis result (restored from an archive or
        copied) is expanded: each value is broadcast along the node axes. Replay
        reads only the node-weighted next-period value, which the broadcast
        reproduces up to the rounding of the weights' sum.
        """
        attached = getattr(solution, _NODE_AXIS_SOLUTION, None)
        if attached is not None:
            return attached
        schemas = solution.metadata.value_schemas
        if any(
            name in self.processes
            for schema in schemas.values()
            for name in schema.axis_names
        ):
            return solution
        expanded = {
            key: self._node_axis_schema(schema=schema, state_names=state_names[key[1]])
            for key, schema in schemas.items()
        }
        values = {
            period: {
                regime_name: _broadcast(
                    value=jnp.asarray(value),
                    published=schemas[period, regime_name],
                    expanded=expanded[period, regime_name],
                )
                for regime_name, value in regimes.items()
            }
            for period, regimes in solution.values.items()  # noqa: PD011 - result store
        }
        return _copy_result(result=solution, values=values, schemas=expanded)

    def present_simulation(
        self, *, result: SimulationResult, solution: SolutionResult | None
    ) -> SimulationResult:
        """Publish the simulation's values, solution and panel without node states.

        Each realised shock moves from the states to the shocks of its rows, so
        the panel carries it beside the actions. `solution` is the published
        result the caller passed, if any.
        """
        # The adapter owns this boundary of the result.
        result._raw_results = MappingProxyType(  # noqa: SLF001
            {
                regime_name: MappingProxyType(
                    {
                        period: self._shocks_out_of_states(data)
                        for period, data in periods.items()
                    }
                )
                for regime_name, periods in result.raw_results.items()
            }
        )
        metadata = result._metadata  # noqa: SLF001
        result._metadata = dataclasses.replace(  # noqa: SLF001
            metadata,
            state_names=self._hide(metadata.state_names),
            regime_to_states=MappingProxyType(
                {
                    name: self._hide(states)
                    for name, states in metadata.regime_to_states.items()
                }
            ),
        )
        replayed = result._solution  # noqa: SLF001
        if replayed is None:
            return result
        schemas = replayed.metadata.value_schemas
        result._period_to_regime_to_V_arr = MappingProxyType(  # noqa: SLF001
            {
                period: MappingProxyType(
                    {
                        regime_name: self._integrate(
                            value=value,
                            axis_names=schemas[period, regime_name].axis_names,
                        )
                        for regime_name, value in regimes.items()
                    }
                )
                for period, regimes in result.period_to_regime_to_V_arr.items()
            }
        )
        result._solution = (  # noqa: SLF001
            self.present_solution(result=replayed) if solution is None else solution
        )
        return result

    def lower_simulation_inputs(
        self,
        *,
        initial_conditions: SimulationInitialConditionsInput | pd.DataFrame,
        shock_paths: ShockPathsInput | None,
        seed: UserSeed | None,
        ages: TimeAxis,
    ) -> tuple[
        SimulationInitialConditionsInput | pd.DataFrame, ShockPathsInput, UserSeed
    ]:
        """Realise every adapted shock for every subject row and period.

        Fresh draws use the current-IID route's addressed stream: the shock's name,
        the subject's original row and the period of use. A tagged entry applies at
        the row's entry period; a tagged path applies at every period it covers and
        supersedes the entry. Both go through the current-IID override producer.
        The physical paths become the node states' given shock paths.

        Returns:
            The initial conditions without adapted shocks, the shock paths with
            every adapted shock's physical path, and the seed the simulation uses.
        """
        initial, entries, paths, state_paths = split_current_iid_inputs(
            initial=initial_conditions,
            paths=shock_paths,
            names=frozenset(self.processes),
        )
        seed = draw_random_seed() if seed is None else seed
        # Only the entry-time column is read; other columns pass through unread.
        lowered = lower_initial_time(
            initial_conditions=cast("UserInitialConditions | pd.DataFrame", initial),
            ages=ages,
        )
        if "age" not in lowered:
            raise InvalidSimulationInputError(
                "Simulating current-IID shocks requires each row's entry time."
            )
        starts, _ = resolve_initial_periods(
            ages=ages, initial_ages=np.asarray(lowered["age"])
        )
        key = create_simulation_key(seed=seed)
        physical: dict[ShockName, FloatND] = {}
        for group_name, group in self.groups.items():
            physical.update(
                _realise_group_paths(
                    group_name=group_name,
                    group=group,
                    key=key,
                    starts=np.asarray(starts),
                    ages=ages,
                    entries={k: v for k, v in entries.items() if k in group.shocks},
                    paths={k: v for k, v in paths.items() if k in group.shocks},
                )
            )
        return initial, MappingProxyType({**state_paths, **physical}), seed

    def _hide(self, names: tuple[StateName, ...]) -> tuple[StateName, ...]:
        return tuple(name for name in names if name not in self.processes)

    def _shocks_out_of_states(
        self, data: PeriodRegimeSimulationData
    ) -> PeriodRegimeSimulationData:
        moved = {
            name: value for name, value in data.states.items() if name in self.processes
        }
        if not moved:
            return data
        return dataclasses.replace(
            data,
            states=MappingProxyType(
                {
                    name: value
                    for name, value in data.states.items()
                    if name not in self.processes
                }
            ),
            shocks=MappingProxyType({**data.shocks, **moved}),
        )

    def _integrate(self, *, value: FloatND, axis_names: tuple[str, ...]) -> FloatND:
        """Integrate the node axes with their rule weights, last axis first."""
        for index in reversed(range(len(axis_names))):
            process = self.processes.get(axis_names[index])
            if process is not None:
                weights = process.get_transition_probs()[0]
                value = jnp.tensordot(value, weights, axes=([index], [0]))
        return value

    def _published_schema(self, schema: ValueArraySchema) -> ValueArraySchema:
        keep = tuple(
            index
            for index, name in enumerate(schema.axis_names)
            if name not in self.processes
        )
        if len(keep) == len(schema.axis_names):
            return schema
        return ValueArraySchema(
            shape=tuple(schema.shape[index] for index in keep),
            dtype=schema.dtype,
            axis_names=tuple(schema.axis_names[index] for index in keep),
            named_axes=tuple(schema.named_axes[index] for index in keep)
            if schema.named_axes
            else (),
            categorical_domains=schema.categorical_domains,
        )

    def _node_axis_schema(
        self, *, schema: ValueArraySchema, state_names: tuple[StateName, ...]
    ) -> ValueArraySchema:
        """Re-insert a published schema's node axes in the regime's state order."""
        nodes = tuple(
            name
            for name in state_names
            if name in self.processes and name not in schema.axis_names
        )
        if not nodes:
            return schema
        position = {name: index for index, name in enumerate(state_names)}
        lengths = dict(zip(schema.axis_names, schema.shape, strict=True))
        descriptors = {name: self._node_axis(name) for name in nodes}
        lengths.update({name: axis.length for name, axis in descriptors.items()})
        if schema.named_axes:
            descriptors.update(zip(schema.axis_names, schema.named_axes, strict=True))
        order = sorted(lengths, key=lambda name: position.get(name, len(position)))
        return ValueArraySchema(
            shape=tuple(lengths[name] for name in order),
            dtype=schema.dtype,
            axis_names=tuple(order),
            named_axes=tuple(descriptors[name] for name in order)
            if schema.named_axes
            else (),
            categorical_domains=schema.categorical_domains,
        )

    def _node_axis(self, name: ShockName) -> AxisDescriptor:
        gridpoints = np.asarray(self.processes[name].get_gridpoints())
        return AxisDescriptor(
            name=name,
            length=len(gridpoints),
            role=AxisRole.STATE,
            coordinates=tuple(gridpoints.tolist()),
        )


def adapt_declaration(
    *,
    regimes: Mapping[RegimeName, Regime],
    shocks: Mapping[ShockGroupName, ShockGroup],
) -> tuple[
    Mapping[RegimeName, Regime],
    Mapping[ShockGroupName, ShockGroup],
    NodeAxisAdapter | None,
]:
    """Move every group an endogenous-grid regime reads onto its readers' states.

    Every reader of such a group gets its shocks as node states, GridSearch readers
    included, so one group has one representation. Groups read only by GridSearch
    regimes stay on the current-IID route.

    Returns:
        The internal regimes, the groups left on the current-IID route and the
        adapter record, which is `None` when no group moves.
    """
    adapted = {
        name: shocks[name]
        for name in dict.fromkeys(
            name
            for regime in regimes.values()
            if not isinstance(regime.solver, GridSearch)
            for name in regime.shocks
        )
        if name in shocks
    }
    if not adapted:
        return regimes, shocks, None
    processes = {
        shock: node_process(group_name=name, group=group, shock=shock)
        for name, group in adapted.items()
        for shock in group.shocks
    }
    internal = {}
    for regime_name, regime in regimes.items():
        read = tuple(name for name in regime.shocks if name in adapted)
        internal[regime_name] = (
            regime.replace(
                states={
                    **regime.states,
                    **{
                        shock: processes[shock]
                        for name in read
                        for shock in adapted[name].shocks
                    },
                },
                shocks=tuple(name for name in regime.shocks if name not in adapted),
            )
            if read
            else regime
        )
    return (
        MappingProxyType(internal),
        MappingProxyType(
            {name: group for name, group in shocks.items() if name not in adapted}
        ),
        NodeAxisAdapter(
            declared_groups=MappingProxyType(dict(shocks)),
            groups=MappingProxyType(adapted),
            processes=MappingProxyType(processes),
            regime_shocks=MappingProxyType(
                {
                    name: regimes[name].shocks
                    for name, regime in internal.items()
                    if regime is not regimes[name]
                }
            ),
        ),
    )


def node_process(
    *, group_name: ShockGroupName, group: ShockGroup, shock: ShockName
) -> _IIDProcess:
    """Return the node state that solves one shock of an adapted group.

    The IID process classes cover the scalar rules they implement node for node;
    every other supported rule is realised from the group's resolved node set.
    """
    _require_supported(group_name=group_name, group=group)
    law = group.shocks[shock]
    rule = group.nodes.rules[shock] if isinstance(group.nodes, Product) else None
    if isinstance(law, Normal | LogNormal) and isinstance(
        rule, GaussHermite | EqualSpaced
    ):
        process = NormalIIDProcess if isinstance(law, Normal) else LogNormalIIDProcess
        return process(
            n_points=rule.n,
            gauss_hermite=isinstance(rule, GaussHermite),
            n_std=rule.n_std if isinstance(rule, EqualSpaced) else None,
            # A supported law's parameters are fixed scalars, stored as floats.
            mu=cast("float", law.mu),
            sigma=cast("float", law.sigma),
        )
    if isinstance(law, Uniform) and isinstance(rule, EqualSpacedUniform):
        return UniformIIDProcess(
            n_points=rule.n,
            start=cast("float", law.start),
            stop=cast("float", law.stop),
        )
    single = ShockGroup(
        shocks={shock: law},
        nodes=group.nodes if rule is None else Product(**{shock: rule}),
    )
    realisation = single.realise(
        nodes=single.resolve_nodes(
            group_name=group_name, age=0, coordinate_kind="period"
        )
    )
    realisation.raise_if_invalid()
    nodes = np.asarray(realisation.values[shock], dtype=float)  # noqa: PD011 - shock realisation
    weights = np.asarray(realisation.weights, dtype=float)
    order = np.argsort(nodes, kind="stable")
    return _ShockNodeProcess(
        n_points=len(nodes),
        group=single,
        nodes=tuple(nodes[order].tolist()),
        weights=tuple(weights[order].tolist()),
    )


@dataclass(frozen=True, kw_only=True)
class _ShockNodeProcess(_IIDProcess):
    """One shock's resolved rule as a node state, drawn from its declared law."""

    group: ShockGroup = field(compare=False)
    """One-shock group whose law places the nodes and maps base draws."""
    nodes: tuple[float, ...]
    """Physical nodes in increasing order."""
    weights: tuple[float, ...]
    """Rule weights in node order."""

    _NON_PARAM_FIELDS: ClassVar[frozenset[str]] = _IIDProcess._NON_PARAM_FIELDS | {  # noqa: SLF001 - extending the inherited process contract
        "group",
        "nodes",
        "weights",
    }

    def compute_gridpoints(self, **kwargs: ScalarFloat | ScalarInt) -> Float1D:  # noqa: ARG002
        return jnp.asarray(self.nodes)

    def compute_transition_probs(self, **kwargs: ScalarFloat | ScalarInt) -> FloatND:  # noqa: ARG002
        return jnp.full(
            (self.n_points, self.n_points), fill_value=jnp.asarray(self.weights)
        )

    def draw_shock(
        self, *, params: MappingProxyType[str, ScalarFloat | ScalarInt], key: PRNGKeyND
    ) -> ScalarFloat:
        del params
        ((name, law),) = self.group.shocks.items()
        base = _draw_base_value(key=key, uniform=isinstance(law, Uniform))
        realisation = self.group.transform(base_nodes=jnp.reshape(base, (1, 1)))
        return realisation.values[name][0]  # noqa: PD011 - shock realisation


def _require_supported(*, group_name: ShockGroupName, group: ShockGroup) -> None:
    """Refuse declarations the node-state axis does not represent."""
    unsupported = None
    if group.signed:
        unsupported = "signed node weights, which the node-state expectation refuses"
    elif group.correlation is not _Independence.DEFAULT:
        unsupported = "correlated shocks"
    elif isinstance(group.nodes, MonteCarlo | QuasiMonteCarlo):
        unsupported = "Monte Carlo rules, whose nodes change with age"
    elif not isinstance(group.nodes, Product) and len(group.shocks) > 1:
        unsupported = "a joint node rule over several shocks that is not a Product"
    elif any(isinstance(law, Discrete) for law in group.shocks.values()):
        unsupported = "discrete laws"
    elif not all(_is_fixed(law) for law in group.shocks.values()):
        unsupported = (
            "law parameters that are runtime slots, callables or state-conditioned"
        )
    if unsupported is not None:
        raise ShockDeclarationError(
            f"Shock group {group_name!r} is read by an endogenous-grid regime, which "
            "solves it on the node-state axis; that axis does not support "
            f"{unsupported}. Use fixed scalar laws with an independent node rule, "
            "or read the group only from GridSearch regimes."
        )


def _is_fixed(law: Law) -> bool:
    return not any(
        value is None
        or callable(value)
        or isinstance(value, StateConditioned | _ConditionedLookup)
        for value in (getattr(law, item.name) for item in fields(law))
    )


def _realise_group_paths(
    *,
    group_name: ShockGroupName,
    group: ShockGroup,
    key: PRNGKeyND,
    starts: np.ndarray,
    ages: TimeAxis,
    entries: Mapping[ShockName, ShockOverride],
    paths: Mapping[ShockName, ShockOverride],
) -> Mapping[ShockName, FloatND]:
    """Realise one group's physical values for every subject row and period."""
    n_subjects, n_periods = len(starts), ages.n_periods
    supplied, active, kinds = {}, {}, {}
    for name in group.shocks:
        item = paths.get(name, entries.get(name))
        if item is None:
            continue
        values, enabled = _dated_cells(
            name=name,
            item=item,
            dated=name in paths,
            starts=starts,
            n_periods=n_periods,
        )
        supplied[name] = jnp.asarray(values, dtype=jnp.result_type(0.0))
        active[name] = jnp.asarray(enabled)
        kinds[name] = (
            BaseInnovations if isinstance(item, BaseInnovations) else PhysicalValues
        )
    keys = {
        name: _addressed_keys(
            key=key, name=name, n_subjects=n_subjects, n_periods=n_periods
        )
        for name in group.shocks
    }

    physical, valid = _cell_program(
        group_name=group_name, group=group, kinds=tuple(kinds.items())
    )(
        keys,
        jnp.broadcast_to(
            jnp.arange(n_periods, dtype=jnp.int32), (n_subjects, n_periods)
        ),
        jnp.broadcast_to(jnp.asarray(ages.values), (n_subjects, n_periods)),
        supplied,
        active,
    )
    if not bool(jnp.all(valid)):
        raise InvalidSimulationInputError(
            f"Shock inputs for group {group_name!r} are invalid under its law: a "
            "supplied physical value lies outside the law's support, or a base "
            "coordinate is not finite."
        )
    return MappingProxyType(dict(physical))


# Realisation programs by shock group and override kinds. A model's groups are
# few and live as long as the model, so entries are never evicted.
# ponytail: unbounded per-process cache; key by a weak reference if models churn.
_CELL_PROGRAMS: dict[
    tuple[
        ShockGroupName,
        int,
        tuple[tuple[ShockName, type[BaseInnovations | PhysicalValues]], ...],
    ],
    tuple[ShockGroup, Callable[..., tuple[Mapping[ShockName, FloatND], BoolND]]],
] = {}


def _cell_program(
    *,
    group_name: ShockGroupName,
    group: ShockGroup,
    kinds: tuple[tuple[ShockName, type[BaseInnovations | PhysicalValues]], ...],
) -> Callable[..., tuple[Mapping[ShockName, FloatND], BoolND]]:
    """Return the compiled per-(subject, period) realisation of one shock group.

    Built once per group and override kinds, so a repeated simulate reuses the
    traced program instead of wrapping a fresh closure in `jax.jit`.
    """
    cache_key = (group_name, id(group), kinds)
    cached = _CELL_PROGRAMS.get(cache_key)
    if cached is not None and cached[0] is group:
        return cached[1]
    override_kinds = dict(kinds)
    plan = CurrentIIDDrawPlan(
        bindings={
            group_name: ShockBinding(
                group=group,
                state_inputs=frozenset(),
                state_domains=MappingProxyType({}),
                categorical_domains=MappingProxyType({}),
                functions=MappingProxyType({}),
                params_template=MappingProxyType({}),
                callable_parameters=frozenset(),
            )
        }
    )

    # keyword-only-exempt: library-callback=jax.vmap
    def cell(
        cell_keys: Mapping[ShockName, PRNGKeyND],
        period: ScalarInt,
        age: ScalarFloat | ScalarInt,
        values: Mapping[ShockName, ScalarFloat],
        enabled: Mapping[ShockName, BoolND],
    ) -> tuple[Mapping[ShockName, FloatND], BoolND]:
        frame = plan.realise(
            keys=cell_keys,
            group_params={group_name: {}},
            arguments={"period": period, "age": age},
            overrides={
                name: override_kinds[name](values=value)
                for name, value in values.items()
            },
            override_active=enabled,
        )
        return frame.physical, frame.valid

    program = jax.jit(jax.vmap(jax.vmap(cell)))
    _CELL_PROGRAMS[cache_key] = (group, program)
    return program


def _addressed_keys(
    *, key: PRNGKeyND, name: ShockName, n_subjects: int, n_periods: int
) -> PRNGKeyND:
    """Address one shock's keys by original row and period, rows by periods."""
    rows = np.arange(n_subjects, dtype=np.uint64)
    words = np.asarray(RandomAddress(kind="shock", name=name).words(), dtype=np.uint32)

    def period_keys(period: jax.Array) -> PRNGKeyND:
        return draw_row_keys(
            key=key,
            address_words=words,
            row_high=(rows >> np.uint64(32)).astype(np.uint32),
            row_low=(rows & np.uint64(0xFFFFFFFF)).astype(np.uint32),
            period=period,
        )

    return jax.vmap(period_keys, out_axes=1)(jnp.arange(n_periods, dtype=jnp.uint32))


def _dated_cells(
    *,
    name: ShockName,
    item: ShockOverride,
    dated: bool,
    starts: np.ndarray,
    n_periods: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Spread one tagged record over the subject-by-period cells it covers."""
    values = np.asarray(item.values)
    n_subjects = len(starts)
    cells = np.zeros((n_subjects, n_periods), dtype=values.dtype)
    enabled = np.zeros((n_subjects, n_periods), dtype=bool)
    if not dated:
        if values.shape != (n_subjects,):
            raise InvalidSimulationInputError(
                f"Current-IID entry {name!r} requires shape (subjects,) in original "
                f"row order; got {values.shape}."
            )
        rows = np.arange(n_subjects)
        cells[rows, starts] = values
        enabled[rows, starts] = True
        return cells, enabled
    if values.ndim == 1 and values.size:
        values = np.broadcast_to(values, (n_subjects, values.shape[0]))
    if values.ndim != 2 or values.shape[0] != n_subjects or not values.shape[1]:  # noqa: PLR2004 - row-by-period path matrix
        raise InvalidSimulationInputError(
            f"Current-IID path {name!r} requires shape (periods,) or "
            f"(subjects, periods) in original row order; got {values.shape}."
        )
    covered = min(values.shape[1], n_periods)
    cells[:, :covered] = values[:, :covered]
    enabled[:, :covered] = True
    return cells, enabled


def _broadcast(
    *, value: FloatND, published: ValueArraySchema, expanded: ValueArraySchema
) -> FloatND:
    positions = tuple(
        index
        for index, name in enumerate(expanded.axis_names)
        if name not in published.axis_names
    )
    return jnp.broadcast_to(jnp.expand_dims(value, positions), expanded.shape)


def _copy_result(
    *,
    result: SolutionResult,
    values: Mapping[int, Mapping[RegimeName, FloatND]],
    schemas: Mapping[tuple[int, RegimeName], ValueArraySchema],
) -> SolutionResult:
    """Copy a result with other values and schemas, keeping its persisted extras."""
    copied = SolutionResult(
        values=values,
        metadata=dataclasses.replace(result.metadata, value_schemas=schemas),
        retained_continuations=result.retained_continuations,
        replay_artifacts=result.replay_artifacts,
        auxiliary_artifacts=result.auxiliary_artifacts,
        omissions=result.omissions,
        diagnostics=result.diagnostics,
    )
    for name in ("_artifact_authority", "_participation_surfaces"):
        object.__setattr__(copied, name, getattr(result, name))
    return copied
