"""Seedless entry support: supplied physical values, never latent draw evidence."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np

from _lcm.current_iid import CORRELATION_SUPPORT_ARGUMENT, _law_inputs
from _lcm.engine import Regime
from _lcm.shock_overrides import supplied_group_values
from _lcm.shock_registry import ShockBinding, law_input_validity
from _lcm.simulation.exits import query_producing_bindings
from _lcm.simulation.memory import SimulationMemory, run_simulation_operation
from _lcm.typing import FlatParams, PytreeValue, QAndFArg
from lcm.exceptions import ShockDeclarationError, UnsupportedOperationError
from lcm.shocks import (
    BaseInnovations,
    HostArray,
    LawValue,
    PhysicalValues,
    ShockName,
    ShockOverride,
)
from lcm.typing import BoolND, FloatND, IntND, ParameterName, ReferenceName, RegimeName


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

    def physical_inputs(
        self,
        *,
        regime: Regime,
        regime_name: RegimeName,
        initial_states: Mapping[ReferenceName, FloatND | IntND],
        indices: list[int],
        periods: np.ndarray,
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
            if any(isinstance(entries[name], BaseInnovations) for name in names):
                identity = (
                    False
                    if binding.group.correlation is None
                    else binding.group._has_identity_correlation(params={})  # noqa: SLF001 - fixed metadata proof only
                )
                if (not isinstance(identity, bool) or not identity) and len(
                    names
                ) != len(binding.group.shocks):
                    raise UnsupportedOperationError(
                        f"Seedless entry group {group_name!r} needs "
                        "a complete base frame "
                        "to determine its correlated outputs."
                    )
            arguments: dict[ReferenceName, FloatND | IntND | HostArray] = {
                name: initial_states[name] for name in (*binding.state_inputs, "age")
            }
            arguments["period"] = periods
            gathered = run_simulation_operation(
                memory=memory,
                function=_gather_entry_values,
                arguments={
                    "arguments": arguments,
                    "values": {name: entries[name].values for name in names},
                    "indices": np.asarray(indices, dtype=np.int32),
                },
            )
            operation = _SuppliedGroup(
                binding=binding,
                kinds=tuple(
                    (
                        name,
                        "base"
                        if isinstance(entries[name], BaseInnovations)
                        else "physical",
                    )
                    for name in names
                ),
            )
            physical, valid = run_simulation_operation(
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


def _gather_entry_values(
    *,
    arguments: Mapping[ReferenceName, FloatND | IntND | HostArray],
    values: Mapping[ShockName, FloatND | IntND | HostArray],
    indices: np.ndarray,
) -> tuple[dict[ReferenceName, FloatND | IntND], dict[ShockName, FloatND | IntND]]:
    """Gather original caller rows once, leaving scientific tags outside states."""
    return (
        {name: jnp.asarray(value)[indices] for name, value in arguments.items()},
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
