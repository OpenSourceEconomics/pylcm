"""Exact forward diagnostic kernels and their abstract allocation bindings."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

import jax
import jax.numpy as jnp
from jaxtyping import Integer

from _lcm.simulation.host_operations import StaticArgument
from _lcm.simulation.memory import SimulationMemory
from _lcm.utils.logging import LogLevel, _owned_values, non_finite_by_regime
from lcm.exceptions import ExecutionPlanningError
from lcm.typing import BoolND, FloatND, Int1D


def period_value_flags(
    *, values: tuple[FloatND, ...], in_regime: tuple[BoolND, ...]
) -> BoolND:
    """Use the existing ownership mask and exact two-row nonfinite reduction."""
    return non_finite_by_regime(values=values, in_regime=in_regime)


def owned_value_nan_count(
    *, value: FloatND, in_regime: BoolND
) -> Integer[jax.Array, ""]:
    """Count precisely the NaNs that the existing masked validate_V receives."""
    owned = _owned_values(value=value, in_regime=in_regime)
    return jnp.sum(jnp.isnan(owned))


def transition_counts(
    *, prev_regime_ids: Int1D, new_regime_ids: Int1D, sorted_ids: tuple[int, ...]
) -> Integer[jax.Array, "regime regime"]:
    """Keep the original ordered transition count tensor expression."""
    id_array = jnp.array(sorted_ids)
    from_one_hot = prev_regime_ids[:, None] == id_array[None, :]
    to_one_hot = new_regime_ids[:, None] == id_array[None, :]
    return (from_one_hot[:, :, None] & to_one_hot[:, None, :]).sum(axis=0)


def profiled_transition_counts(
    *,
    memory: SimulationMemory,
    prev_regime_ids: Int1D,
    new_regime_ids: Int1D,
    sorted_ids: tuple[int, ...],
) -> list[list[int]]:
    """Return host counts after admitting the exact current diagnostic operation."""
    return memory.run(
        function=transition_counts,
        arguments={
            "prev_regime_ids": prev_regime_ids,
            "new_regime_ids": new_regime_ids,
        },
        subject_arg_names=("prev_regime_ids", "new_regime_ids"),
        static_arguments={"sorted_ids": sorted_ids},
    ).tolist()


@dataclass(frozen=True, kw_only=True)
class DiagnosticBinding:
    """One exact operation with abstract inputs; executable out_info owns its shape."""

    function: Callable[..., object]
    arguments: Mapping[str, object]
    subject_arg_names: tuple[str, ...]
    static_arguments: Mapping[str, StaticArgument] = field(
        default_factory=lambda: MappingProxyType({})
    )
    subject_outputs: bool = False

    def __post_init__(self) -> None:
        """Snapshot metadata and reject accidental ownership of caller arrays."""
        if any(
            not isinstance(leaf, jax.ShapeDtypeStruct)
            for leaf in jax.tree.leaves(self.arguments)
        ):
            raise ExecutionPlanningError(
                "Diagnostic bindings require abstract operands."
            )
        object.__setattr__(self, "arguments", MappingProxyType(dict(self.arguments)))
        object.__setattr__(
            self, "static_arguments", MappingProxyType(dict(self.static_arguments))
        )


def diagnostic_bindings(
    *,
    values: tuple[jax.ShapeDtypeStruct, ...],
    in_regime: tuple[jax.ShapeDtypeStruct, ...],
    prev_regime_ids: jax.ShapeDtypeStruct,
    new_regime_ids: jax.ShapeDtypeStruct,
    sorted_ids: tuple[int, ...],
    log_level: LogLevel,
) -> tuple[DiagnosticBinding, ...]:
    """Enumerate enabled diagnostics, including every possible NaN report.

    Inputs carry their actual layout. The caller lowers these same functions and
    obtains output metadata and workspace from the resulting executable, never
    from predicted output multipliers. NaN reports are conditional at runtime;
    reserving each enabled regime's report is a conservative complete-period bound.
    """
    if log_level == "off":
        return ()
    bindings = []
    if values:
        bindings.append(
            DiagnosticBinding(
                function=period_value_flags,
                arguments={"values": values, "in_regime": in_regime},
                subject_arg_names=("values", "in_regime"),
            )
        )
        bindings.extend(
            DiagnosticBinding(
                function=owned_value_nan_count,
                arguments={"value": value, "in_regime": mask},
                subject_arg_names=("value", "in_regime"),
            )
            for value, mask in zip(values, in_regime, strict=True)
        )
    if log_level == "debug":
        bindings.append(
            DiagnosticBinding(
                function=transition_counts,
                arguments={
                    "prev_regime_ids": prev_regime_ids,
                    "new_regime_ids": new_regime_ids,
                },
                subject_arg_names=("prev_regime_ids", "new_regime_ids"),
                static_arguments={"sorted_ids": sorted_ids},
            )
        )
    return tuple(bindings)
