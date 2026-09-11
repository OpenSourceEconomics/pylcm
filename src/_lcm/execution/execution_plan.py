"""Resolve a public `ExecutionConfig` against what a model declares.

The devices a model may use are read here and nowhere else in the package: a
model resolves them once when it is built and every phase reads the resolved
ids, so two phases of one model can never disagree about the hardware they run
on.
"""

import dataclasses
import operator
from collections.abc import Iterable, Mapping
from types import MappingProxyType

import jax

from _lcm.execution.core_program import CoreProgram
from _lcm.typing import StateName
from lcm.exceptions import ExecutionPlanningError
from lcm.execution import ExecutionConfig


@dataclasses.dataclass(frozen=True, kw_only=True)
class ResolvedExecution:
    """Hardware-local facts every phase of one model reads."""

    device_ids: tuple[int, ...]
    """Visible device ids the model uses, ascending."""

    sharded_states: frozenset[StateName]
    """States carrying a device axis."""

    axis_widths: MappingProxyType[str, int]
    """Fixed planner widths by axis name."""

    device_memory_bytes: int | None
    """Per-device workspace budget, or `None`."""

    donate_buffers: bool = True
    """Whether eligible solve inputs may be donated to a compiled executable."""


def resolve_execution_config(
    *,
    config: ExecutionConfig,
    visible_device_ids: tuple[int, ...],
    state_names: frozenset[StateName],
) -> ResolvedExecution:
    """Check a configuration against the model and freeze it.

    The axis names are not checked here: they are legal exactly when a core
    program declares them, and the programs do not exist until the regimes are
    built. `fail_if_axis_widths_name_undeclared_axes` is that gate, and runs
    once the programs are in hand.

    Args:
        config: The user's configuration.
        visible_device_ids: Ids of the devices JAX reports at model build.
        state_names: Every state name any regime declares.

    Returns:
        The resolved facts.

    Raises:
        ExecutionPlanningError: A state or device the model cannot serve.

    """
    for name in config.sharded_states:
        if name not in state_names:
            msg = (
                f"ExecutionConfig.sharded_states names {name!r}, which no regime "
                f"declares as a state; declared states are {sorted(state_names)!r}."
            )
            raise ExecutionPlanningError(msg)
    device_ids = visible_device_ids if config.devices is None else config.devices
    for device_id in device_ids:
        if device_id not in visible_device_ids:
            msg = (
                f"ExecutionConfig.devices: device id {device_id} is not visible; "
                f"visible ids are {visible_device_ids!r}."
            )
            raise ExecutionPlanningError(msg)
    return ResolvedExecution(
        device_ids=tuple(sorted(device_ids)),
        sharded_states=frozenset(config.sharded_states),
        axis_widths=MappingProxyType(dict(config.axis_widths)),
        device_memory_bytes=config.device_memory_bytes,
        donate_buffers=config.donate_buffers,
    )


def fail_if_axis_widths_name_undeclared_axes(
    *,
    axis_widths: Mapping[str, int],
    program_collections: tuple[Iterable[CoreProgram], ...],
) -> None:
    """Reject an axis width for a name none of the model's programs declares.

    The legal set is derived, never listed: it is the union of the axis names
    over every program in every collection, so a solver that declares a new axis
    is configurable the moment it declares it. Each phase contributes one
    collection, which is why the collections arrive as a tuple rather than
    already merged.

    Args:
        axis_widths: The widths the user declared, by axis name.
        program_collections: One collection of core programs per phase whose
            axes the widths may name.

    Raises:
        ExecutionPlanningError: A width names an axis no program declares.

    """
    if not axis_widths:
        return
    declared = frozenset(
        name
        for programs in program_collections
        for program in programs
        for name in program.requirements.axis_names
    )
    for name in axis_widths:
        if name not in declared:
            msg = (
                f"ExecutionConfig.axis_widths names {name!r}, which no core program "
                f"declares; declared axes are {sorted(declared)!r}."
            )
            raise ExecutionPlanningError(msg)


def visible_devices() -> tuple[jax.Device, ...]:
    """Return every device JAX reports, ascending by id."""
    return tuple(sorted(jax.devices(), key=operator.attrgetter("id")))


def visible_device_ids() -> tuple[int, ...]:
    """Return the ids of every device JAX reports, ascending."""
    return tuple(device.id for device in visible_devices())


def execution_over_visible_devices() -> ResolvedExecution:
    """Return the inert configuration resolved against every visible device.

    The resolution a caller that builds canonical regimes outside a `Model` —
    a test, or a tool inspecting one regime — runs under.
    """
    return resolve_execution_config(
        config=ExecutionConfig(),
        visible_device_ids=visible_device_ids(),
        state_names=frozenset(),
    )
