"""Physical entry frames shared by seedless helpers and seeded preflight."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Int32, UInt32

from _lcm.current_iid import (
    CORRELATION_SUPPORT_ARGUMENT,
    AdmittedGroupParams,
    CurrentIIDDrawPlan,
    _law_inputs,
)
from _lcm.engine import Regime
from _lcm.shock_overrides import supplied_group_values
from _lcm.shock_registry import ShockBinding, law_input_validity
from _lcm.simulation.current_iid_overrides import (
    CurrentIIDOverrides,
    gather_current_iid_overrides,
)
from _lcm.simulation.exits import query_producing_bindings
from _lcm.simulation.memory import (
    SimulationMemory,
    run_simulation_operation,
    run_simulation_producer,
)
from _lcm.simulation.random import create_simulation_key
from _lcm.simulation.random_address import RandomAddress, draw_row_keys
from _lcm.simulation.subject_groups import SubjectRows
from _lcm.time import TimeAxis
from _lcm.typing import FlatParams, PRNGKeyND, PytreeValue, QAndFArg
from lcm.exceptions import ShockDeclarationError, UnsupportedOperationError
from lcm.shocks import (
    BaseInnovations,
    HostArray,
    LawValue,
    PhysicalValues,
    ShockGroupName,
    ShockName,
    ShockOverride,
)
from lcm.typing import (
    BoolND,
    FloatND,
    IntND,
    ParameterName,
    PRNGKey,
    ReferenceName,
    RegimeName,
)


@dataclass(frozen=True, kw_only=True, eq=False)
class _SuppliedGroup:
    """One bound scalar law mapped over explicitly gathered original entry rows."""

    binding: ShockBinding
    kinds: tuple[tuple[ShockName, str], ...]

    def __call__(
        self,
        *,
        arguments: Mapping[ReferenceName, FloatND | IntND],
        values: Mapping[ShockName, FloatND | IntND],
        params: Mapping[ParameterName, LawValue],
    ) -> tuple[Mapping[ShockName, FloatND | IntND], BoolND]:
        support = cast("BoolND | None", params.get(CORRELATION_SUPPORT_ARGUMENT))
        law_params = {
            name: value
            for name, value in params.items()
            if name != CORRELATION_SUPPORT_ARGUMENT
        }

        # keyword-only-exempt: library-callback=jax.vmap
        def cell(
            state: Mapping[ReferenceName, FloatND | IntND],
            supplied: Mapping[ShockName, FloatND | IntND],
        ) -> tuple[Mapping[ShockName, FloatND | IntND], BoolND]:
            inputs = _law_inputs(
                binding=self.binding,
                arguments=cast("Mapping[ReferenceName, QAndFArg]", state),
                params=law_params,
            )
            overrides = {
                name: BaseInnovations(values=supplied[name])
                if kind == "base"
                else PhysicalValues(values=supplied[name])
                for name, kind in self.kinds
            }
            physical, valid = supplied_group_values(
                group=self.binding.group,
                supplied=overrides,
                params=law_params,
                inputs=inputs,
                admitted_correlation_support=support,
            )
            return dict(physical), valid & law_input_validity(
                binding=self.binding, inputs=inputs
            )

        return jax.vmap(cell)(arguments, values)


@dataclass(frozen=True, kw_only=True)
class SuppliedEntryFrames:
    """Call-local supplied bank and parameters; no PRNG root or recovered draws."""

    entries: Mapping[ShockName, ShockOverride]
    flat_params: FlatParams

    @property
    def operand_roots(self) -> PytreeValue:
        """Retain supplied arrays explicitly in preflight's residency inventory."""
        return tuple(item.values for item in self.entries.values())

    def physical_inputs(
        self,
        *,
        regime: Regime,
        regime_name: RegimeName,
        initial_states: Mapping[ReferenceName, FloatND | IntND],
        indices: list[int],
        periods: np.ndarray,
        ages: TimeAxis,
        needed: frozenset[ReferenceName],
        memory: SimulationMemory | None,
    ) -> Mapping[ReferenceName, FloatND | IntND]:
        """Validate every supplied local tag before any feasibility callback."""
        plan = regime.simulation.current_iid_draw_plan
        entries = self.entries
        local_names = frozenset() if plan is None else frozenset(plan.physical_names)
        _require_local_entry_producers(
            regime=regime,
            regime_name=regime_name,
            entries=entries,
            periods=periods,
            indices=indices,
            local_names=local_names,
        )
        if plan is None:
            return MappingProxyType({})
        required = needed & frozenset(plan.physical_names)
        if missing := required - entries.keys():
            raise UnsupportedOperationError(
                f"Seedless entry feasibility for regime {regime_name!r} requires "
                f"supplied physical information for {sorted(missing)} at entry periods "
                f"{sorted({int(periods[index]) for index in indices})}."
            )
        result = {}
        for group_name, binding in plan.bindings.items():
            names = tuple(name for name in binding.group.shocks if name in entries)
            if not names:
                continue
            if binding.group.correlation is not None and any(
                isinstance(entries[name], BaseInnovations) for name in names
            ):
                # Runtime correlation is judged by the admitted scalar producer
                # with this invocation's operands, never by a cached host proof.
                identity = binding.group._has_identity_correlation(params={})  # noqa: SLF001 - fixed metadata proof only
                if (not isinstance(identity, bool) or not identity) and len(
                    names
                ) != len(binding.group.shocks):
                    raise UnsupportedOperationError(
                        f"Seedless entry group {group_name!r} needs "
                        "a complete base frame "
                        "to determine its correlated outputs."
                    )
            arguments = {name: initial_states[name] for name in binding.state_inputs}
            gathered = run_simulation_operation(
                memory=memory,
                function=_gather_entry_values,
                arguments={
                    "arguments": arguments,
                    "values": {name: entries[name].values for name in names},
                    "indices": np.asarray(indices, dtype=np.int32),
                    "periods": np.asarray(periods, dtype=np.int32),
                    "age_values": ages.values,
                },
            )
            kinds = tuple(
                "base" if isinstance(entries[name], BaseInnovations) else "physical"
                for name in names
            )
            operation = (
                _supplied_entry_group(
                    plan=plan, group_name=group_name, names=names, kinds=kinds
                )
                if memory is None
                else memory.producers.built(
                    builder=_supplied_entry_group,
                    plan=plan,
                    group_name=group_name,
                    names=names,
                    kinds=kinds,
                )
            )
            physical, valid = run_simulation_producer(
                memory=memory,
                function=operation,
                arguments={
                    "arguments": gathered[0],
                    "values": gathered[1],
                    "params": cast("PytreeValue", self.flat_params.get(group_name, {})),
                },
            )
            if not bool(np.all(np.asarray(valid))):
                raise ShockDeclarationError(
                    f"Invalid supplied entry frame for regime {regime_name!r}, "
                    f"group {group_name!r}; original rows {indices}."
                )
            result.update(
                {name: value for name, value in physical.items() if name in needed}
            )
        return MappingProxyType(result)


@dataclass(frozen=True, kw_only=True)
class SeededEntryFrames(SuppliedEntryFrames):
    """Recompute ordinary entry frames from the final decision's named addresses."""

    seed: int | PRNGKey
    overrides: CurrentIIDOverrides | None

    @property
    def operand_roots(self) -> PytreeValue:
        """Charge the root key and normalized banks without retaining draws."""
        return (
            self.seed,
            super().operand_roots,
            ()
            if self.overrides is None
            else tuple(item.values for item in self.overrides.paths.values()),
        )

    def physical_inputs(
        self,
        *,
        regime: Regime,
        regime_name: RegimeName,
        initial_states: Mapping[ReferenceName, FloatND | IntND],
        indices: list[int],
        periods: np.ndarray,
        ages: TimeAxis,
        needed: frozenset[ReferenceName],
        memory: SimulationMemory | None,
    ) -> Mapping[ReferenceName, FloatND | IntND]:
        """Validate support before constraints, using exact original-row draws."""
        plan = regime.simulation.current_iid_draw_plan
        names = frozenset() if plan is None else frozenset(plan.physical_names)
        paths = {} if self.overrides is None else self.overrides.paths
        _require_local_entry_producers(
            regime=regime,
            regime_name=regime_name,
            entries={**self.entries, **paths},
            periods=periods,
            indices=indices,
            local_names=names,
        )
        if not ((needed & names) - self.entries.keys()) and not (paths.keys() & names):
            return super().physical_inputs(
                regime=regime,
                regime_name=regime_name,
                initial_states=initial_states,
                indices=indices,
                periods=periods,
                ages=ages,
                needed=needed,
                memory=memory,
            )
        if plan is None:
            return MappingProxyType({})
        gathered = run_simulation_operation(
            memory=memory,
            function=_gather_entry_values,
            arguments={
                "arguments": {name: initial_states[name] for name in plan.state_inputs},
                "values": {},
                "indices": np.asarray(indices, dtype=np.int32),
                "periods": np.asarray(periods, dtype=np.int32),
                "age_values": ages.values,
            },
        )[0]
        local_periods = np.asarray(periods, dtype=np.int32)[np.asarray(indices)]
        values, kinds = _entry_overrides(
            bank=self.overrides,
            names=plan.physical_names,
            indices=indices,
            periods=local_periods,
        )
        operation = (
            _seeded_entry_draw(plan=plan, names=tuple(values), kinds=kinds)
            if memory is None
            else memory.producers.built(
                builder=_seeded_entry_draw,
                plan=plan,
                names=tuple(values),
                kinds=kinds,
            )
        )
        rows = np.asarray(indices, dtype=np.uint64)
        words = np.stack((rows >> np.uint64(32), rows), axis=-1).astype(np.uint32)
        physical, valid = run_simulation_producer(
            memory=memory,
            function=operation,
            arguments={
                "key": create_simulation_key(seed=self.seed, memory=memory),
                "arguments": gathered,
                "rows": words,
                "values": values,
                "group_params": cast(
                    "PytreeValue",
                    {name: self.flat_params.get(name, {}) for name in plan.bindings},
                ),
            },
        )
        if not bool(np.all(np.asarray(valid))):
            raise ShockDeclarationError(
                f"Invalid seeded entry frame for regime {regime_name!r}; "
                f"original rows {indices}."
            )
        return MappingProxyType(
            {name: value for name, value in physical.items() if name in needed}
        )


def _entry_overrides(
    *,
    bank: CurrentIIDOverrides | None,
    names: tuple[ShockName, ...],
    indices: list[int],
    periods: np.ndarray,
) -> tuple[dict[ShockName, HostArray], tuple[str, ...]]:
    """Reuse dated gathering on admitted original rows, never cohort row numbers."""
    if bank is None:
        return {}, ()
    values = {}
    kinds = {}
    for period in np.unique(periods):
        positions = np.flatnonzero(periods == period)
        gathered = gather_current_iid_overrides(
            bank=bank,
            names=names,
            period=int(period),
            subject_slice=SubjectRows(rows=np.asarray(indices)[positions]),
        )
        for name, item in gathered.items():
            if not np.all(np.asarray(item.active)):
                raise ShockDeclarationError(
                    "An entry override must be active at entry."
                )
            if name not in values:
                values[name] = np.empty(len(indices), dtype=item.values.dtype)
                kinds[name] = item.kind
            values[name][positions] = item.values
    return values, tuple(kinds.values())


@dataclass(frozen=True, kw_only=True, eq=False)
class _SeededEntryDraw:
    """The native scalar producer projected to physical values and integrity."""

    plan: CurrentIIDDrawPlan
    names: tuple[ShockName, ...]
    kinds: tuple[str, ...]

    def __call__(
        self,
        *,
        key: PRNGKeyND,
        arguments: Mapping[ReferenceName, FloatND | IntND],
        rows: UInt32[jax.Array | np.ndarray, "rows 2"],
        values: Mapping[ShockName, FloatND | IntND | HostArray],
        group_params: AdmittedGroupParams,
    ) -> tuple[dict[ShockName, FloatND | IntND], BoolND]:
        # keyword-only-exempt: library-callback=jax.vmap
        def cell(
            state: Mapping[ReferenceName, FloatND | IntND],
            row: UInt32[jax.Array | np.ndarray, "2"],  # noqa: UP037 - jaxtyping shape string
            supplied: Mapping[ShockName, FloatND | IntND | HostArray],
        ) -> tuple[dict[ShockName, FloatND | IntND], BoolND]:
            keys = {
                name: draw_row_keys(
                    key=key,
                    address_words=np.asarray(
                        RandomAddress(kind="shock", name=name).words(), dtype=np.uint32
                    ),
                    row_high=row[:1],
                    row_low=row[1:],
                    period=jnp.asarray(state["period"], dtype=jnp.uint32),
                )[0]
                for name in self.plan.physical_names
            }
            frame = self.plan.realise(
                keys=keys,
                group_params=group_params,
                arguments=state,
                overrides={
                    name: BaseInnovations(values=supplied[name])
                    if kind == "base"
                    else PhysicalValues(values=supplied[name])
                    for name, kind in zip(self.names, self.kinds, strict=True)
                },
            )
            return dict(frame.physical), frame.valid

        return jax.vmap(cell)(arguments, rows, values)


def _supplied_entry_group(
    *,
    plan: CurrentIIDDrawPlan,
    group_name: ShockGroupName,
    names: tuple[ShockName, ...],
    kinds: tuple[str, ...],
) -> _SuppliedGroup:
    """Compose a model-owned supplied producer from immutable law/tag metadata."""
    return _SuppliedGroup(
        binding=plan.bindings[group_name], kinds=tuple(zip(names, kinds, strict=True))
    )


def _seeded_entry_draw(
    *, plan: CurrentIIDDrawPlan, names: tuple[ShockName, ...], kinds: tuple[str, ...]
) -> _SeededEntryDraw:
    """Compose once per model-owned plan and tag schema, without numeric captures."""
    return _SeededEntryDraw(plan=plan, names=names, kinds=kinds)


def _gather_entry_values(
    *,
    arguments: Mapping[ReferenceName, FloatND | IntND | HostArray],
    values: Mapping[ShockName, FloatND | IntND | HostArray],
    indices: Int32[jax.Array | np.ndarray, "..."],  # noqa: UP037 - jaxtyping shape string
    periods: Int32[jax.Array | np.ndarray, "..."],  # noqa: UP037 - jaxtyping shape string
    age_values: FloatND | IntND,
) -> tuple[dict[ReferenceName, FloatND | IntND], dict[ShockName, FloatND | IntND]]:
    """Gather original caller rows once, leaving scientific tags outside states."""
    local_periods = jnp.asarray(periods, dtype=jnp.int32)[indices]
    return (
        {
            **{name: jnp.asarray(value)[indices] for name, value in arguments.items()},
            "period": local_periods,
            "age": age_values[local_periods],
        },
        {name: jnp.asarray(value)[indices] for name, value in values.items()},
    )


def _require_local_entry_producers(
    *,
    regime: Regime,
    regime_name: RegimeName,
    entries: Mapping[ShockName, ShockOverride],
    periods: np.ndarray,
    indices: list[int],
    local_names: frozenset[ShockName],
) -> None:
    """Never silently discard a supplied frame from an unrepresented mapped producer."""
    for period in {int(periods[index]) for index in indices}:
        exit_plan = regime.simulation.exit_queries.get(period)
        if exit_plan is None or exit_plan.full is None:
            continue
        foreign = {
            name
            for binding in query_producing_bindings(queries=exit_plan.full).values()
            for name in binding.group.shocks
        } - local_names
        if unresolved := foreign & entries.keys():
            raise UnsupportedOperationError(
                f"Seedless entry feasibility for {regime_name!r} "
                f"at period {period} "
                "must resolve the mapped producer of supplied "
                f"{sorted(unresolved)}."
            )
