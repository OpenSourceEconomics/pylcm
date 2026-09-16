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
from typing import Literal

import jax

from _lcm.execution.core_program import CoreProgram
from _lcm.typing import RegimeName, StateName
from lcm.exceptions import ExecutionPlanningError
from lcm.execution import AxisWidth, ExecutionConfig


@dataclasses.dataclass(frozen=True, kw_only=True)
class ResolvedExecution:
    """Hardware-local facts every phase of one model reads."""

    device_ids: tuple[int, ...]
    """Visible device ids the model uses, ascending."""

    sharded_states: frozenset[StateName]
    """States carrying a device axis."""

    axis_widths: MappingProxyType[str, int]
    """Fixed planner widths by axis name, for every regime declaring the axis."""

    axis_widths_by_regime: MappingProxyType[RegimeName, MappingProxyType[str, int]] = (
        MappingProxyType({})
    )
    """Widths that override the model-wide ones, by regime name then axis name."""

    device_memory_bytes: int | None
    """Per-device workspace budget, or `None`."""

    simulation_sharding: Literal["legacy", "subjects"] = "legacy"
    """Forward placement and local-loop policy, separate from solve-state axes."""

    continuous_sharded_state: StateName | None = None
    """Internal capability set only after Model validates continuous GridSearch."""

    donate_buffers: bool = True
    """Whether eligible solve inputs may be donated to a compiled executable."""

    def widths_for(self, *, regime_name: RegimeName) -> MappingProxyType[str, int]:
        """Return the fixed widths one regime's programs are planned against.

        Args:
            regime_name: The regime whose programs are being planned.

        Returns:
            The model-wide widths, with this regime's overrides applied.

        """
        override = self.axis_widths_by_regime.get(regime_name)
        if not override:
            return self.axis_widths
        return MappingProxyType({**self.axis_widths, **override})


def _split_axis_widths(
    *, axis_widths: Mapping[str, AxisWidth]
) -> tuple[
    MappingProxyType[str, int], MappingProxyType[RegimeName, MappingProxyType[str, int]]
]:
    """Split one declaration into model-wide widths and per-regime overrides.

    Args:
        axis_widths: The user's declaration, already validated by `ExecutionConfig`.

    Returns:
        The widths that hold for every regime, and the per-regime overrides
        keyed by regime name so a planner serving one regime reads one mapping.

    """
    model_wide: dict[str, int] = {}
    by_regime: dict[RegimeName, dict[str, int]] = {}
    for axis_name, width in axis_widths.items():
        if isinstance(width, Mapping):
            for regime_name, regime_width in width.items():
                by_regime.setdefault(regime_name, {})[axis_name] = regime_width
        else:
            model_wide[axis_name] = width
    return (
        MappingProxyType(model_wide),
        MappingProxyType(
            {name: MappingProxyType(widths) for name, widths in by_regime.items()}
        ),
    )


def resolve_execution_config(
    *,
    config: ExecutionConfig,
    visible_device_ids: tuple[int, ...],
    state_names: frozenset[StateName],
    regime_names: frozenset[RegimeName] = frozenset(),
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
        regime_names: Every regime name the model declares, which a per-regime
            axis width may name. Empty admits no per-regime width.

    Returns:
        The resolved facts.

    Raises:
        ExecutionPlanningError: A state, regime or device the model cannot serve.

    """
    for name in config.sharded_states:
        if name not in state_names:
            msg = (
                f"ExecutionConfig.sharded_states names {name!r}, which no regime "
                f"declares as a state; declared states are {sorted(state_names)!r}."
            )
            raise ExecutionPlanningError(msg)
    model_wide_widths, widths_by_regime = _split_axis_widths(
        axis_widths=config.axis_widths
    )
    _fail_if_a_width_names_an_unknown_regime(
        widths_by_regime=widths_by_regime, regime_names=regime_names
    )
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
        axis_widths=model_wide_widths,
        axis_widths_by_regime=widths_by_regime,
        device_memory_bytes=config.device_memory_bytes,
        donate_buffers=config.donate_buffers,
        simulation_sharding=config.simulation_sharding,
    )


def _fail_if_a_width_names_an_unknown_regime(
    *,
    widths_by_regime: Mapping[RegimeName, Mapping[str, int]],
    regime_names: frozenset[RegimeName],
) -> None:
    """Reject a per-regime axis width for a regime the model does not declare."""
    for regime_name, widths in widths_by_regime.items():
        if regime_name in regime_names:
            continue
        axis_name = min(widths)
        msg = (
            f"ExecutionConfig.axis_widths[{axis_name!r}] names regime "
            f"{regime_name!r}, which the model does not declare; declared regimes "
            f"are {sorted(regime_names)!r}."
        )
        raise ExecutionPlanningError(msg)


def fail_if_per_regime_widths_name_non_solve_axes(
    *,
    axis_widths_by_regime: Mapping[RegimeName, Mapping[str, int]],
    solve_programs: Iterable[CoreProgram],
) -> None:
    """Reject a per-regime width for an axis the solve phase does not declare.

    The solve planner is the one that plans per regime, so it is the only
    consumer a per-regime override can reach. An axis only simulation programs
    declare would silently keep its planned width, so it is refused instead.

    Args:
        axis_widths_by_regime: The per-regime overrides, by regime then axis.
        solve_programs: Every core program the solve phase declares.

    Raises:
        ExecutionPlanningError: An override names an axis no solve program declares.

    """
    if not axis_widths_by_regime:
        return
    declared = frozenset(
        name for program in solve_programs for name in program.requirements.axis_names
    )
    for regime_name, widths in axis_widths_by_regime.items():
        for axis_name in sorted(widths):
            if axis_name not in declared:
                msg = (
                    f"ExecutionConfig.axis_widths[{axis_name!r}] fixes a width for "
                    f"regime {regime_name!r}, but no solve program declares that "
                    "axis; only the solve phase plans per regime, so this axis "
                    "takes a single width for the whole model."
                )
                raise ExecutionPlanningError(msg)


def fail_if_axis_widths_name_undeclared_axes(
    *,
    axis_widths: Mapping[str, AxisWidth],
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
