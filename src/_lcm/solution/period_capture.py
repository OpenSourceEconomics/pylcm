"""Capture one regime-period's kernel inputs during a solve.

Diagnosing a kernel that is slow, or whose allocation is refused, otherwise costs
a full backward induction: every period above the one in question has to be
solved before the interesting one is reached. Capturing the inputs of a single
regime-period turns each subsequent experiment into one kernel invocation.

The capture is written from the funnel every regime-period passes through, so
what a replay runs is what ran, rather than a reconstruction that might differ
from it. Selection is by `LCM_CAPTURE_PERIOD="<regime>@<period>"`, written to
`LCM_CAPTURE_DIR`; a malformed target raises rather than reading as "nothing to
capture".

The compiled cores are not part of the capture — an XLA executable is not
portable. `_lcm.solution.period_replay` rebuilds them from the captured regime,
lowering and compiling only the cores of the one period it runs.

**Arrays and executables are never written; their device layout is.** Beside the
logical kernel inputs, a capture records a `layouts` block of pure descriptors:
per array leaf its tree path, shape, dtype, weak type, sharding kind,
`PartitionSpec`, mesh axis names and sizes, memory kind and ordered device ids;
per core the lowered `out_shardings`, the compiled executable's
`input_shardings` and `output_shardings`, the resolved value-transfer plan, the
donated argument names and the variant those names select. A pickle round trip
restores the arrays at the backend's default placement, so `replay_period`
reports fidelity `logical`; `replay_period_on_recorded_layout` reinstates the
recorded layout from these descriptors and reports `layout`.
"""

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import numpy as np

from _lcm.execution.output_layout import PlannedCore
from _lcm.execution.value_transfer import ResolvedValueTransfer
from _lcm.persistence.io import _save_pkl
from _lcm.typing import RegimeName

type PeriodCaptureTarget = tuple[RegimeName, int]

_TARGET_ENV = "LCM_CAPTURE_PERIOD"
_DIR_ENV = "LCM_CAPTURE_DIR"
_PAYLOAD_NAME = "kernel_inputs.pkl"

# Payload key under which a capture records its device-layout descriptors.
LAYOUTS_KEY = "layouts"

# Variant label of a core lowered with at least one donated argument.
DONATING_VARIANT = "donating"

# Variant label of a core lowered with no donated argument.
NON_DONATING_VARIANT = "non_donating"


@dataclass(frozen=True, kw_only=True)
class ShardingDescriptor:
    """Everything needed to rebuild one array placement on a named device list."""

    kind: str
    """Sharding class name, `NamedSharding` or `SingleDeviceSharding`."""

    device_ids: tuple[int, ...]
    """Device ids the placement spans, in mesh order on a named mesh and
    ascending otherwise."""

    partition_spec: tuple[str | tuple[str, ...] | None, ...] | None
    """`PartitionSpec` entries, or `None` off a named mesh."""

    mesh_axis_names: tuple[str, ...] | None
    """Mesh axis names, or `None` off a named mesh."""

    mesh_axis_sizes: tuple[int, ...] | None
    """Mesh axis sizes in the same order, or `None` off a named mesh."""

    memory_kind: str | None
    """Memory space the buffers live in, as the backend names it."""


@dataclass(frozen=True, kw_only=True)
class LeafLayoutDescriptor:
    """Placement and abstract identity of one captured array leaf."""

    tree_path: str
    """`jax.tree_util.keystr` path of the leaf within `kernel_kwargs`."""

    shape: tuple[int, ...]
    """Absolute shape of the leaf."""

    dtype: str
    """Leaf dtype, by name."""

    weak_type: bool
    """Whether the leaf carries JAX's weak dtype."""

    committed: bool
    """Whether the leaf's placement was pinned rather than the backend's default."""

    sharding: ShardingDescriptor
    """Placement the leaf was held in when the kernel was called."""


@dataclass(frozen=True, kw_only=True)
class ValueTransferDescriptor:
    """One resolved stored-value transfer, as descriptors only."""

    kind: str
    """`ValueTransferKind` member name of the resolved operator."""

    target: str
    """Stored artifact address the transfer reads, rendered for comparison."""

    source: str
    """Core-input address the transfer writes, rendered for comparison."""

    stored_sharding: ShardingDescriptor
    """Placement the stored array is read from."""

    source_sharding: ShardingDescriptor
    """Placement the core input requires."""

    expected_shape: tuple[int, ...]
    """Absolute shape the transfer carries."""

    expected_dtype: str
    """Dtype the transfer carries, by name."""


@dataclass(frozen=True, kw_only=True)
class CoreLayoutDescriptor:
    """Lowering and compilation placement of one core of a captured period."""

    name: str
    """Graph key of the core."""

    lowered_out_shardings: tuple[ShardingDescriptor, ...]
    """Output placements requested at lowering, in output leaf order."""

    compiled_input_shardings: tuple[tuple[str, ShardingDescriptor], ...]
    """Placement the executable takes each input in, keyed by argument path."""

    compiled_output_shardings: tuple[tuple[str, ShardingDescriptor], ...]
    """Placement the executable produces each output in, keyed by leaf path."""

    input_transfer_plan: tuple[ValueTransferDescriptor, ...]
    """Resolved stored-value transfers feeding this core, in plan order."""

    donated_arguments: tuple[str, ...]
    """Arguments the executable was lowered to donate, in declaration order."""

    variant: str
    """`donating` when any argument is donated, otherwise `non_donating`."""


@dataclass(frozen=True, kw_only=True)
class PeriodLayouts:
    """The device layout a captured regime-period ran on."""

    route: str
    """Fully qualified class name of the period kernel that ran."""

    device_ids: tuple[int, ...]
    """Ids of the devices the recorded placements use, in `jax.devices()` order."""

    leaves: tuple[LeafLayoutDescriptor, ...]
    """One descriptor per array leaf of `kernel_kwargs`, in leaf order."""

    cores: Mapping[str, CoreLayoutDescriptor | None]
    """One entry per core, keyed as the kernel publishes them. The value is
    `None` for a core whose executable publishes no input/output shardings —
    the eager route's `_EagerCore` is one — which records the layout block as
    absent rather than refusing the capture."""


def resolve_capture_target() -> PeriodCaptureTarget | None:
    """Return the regime-period selected for capture, or `None`.

    Raises:
        ValueError: The target is set but is not `<regime>@<period>`.

    """
    raw = os.environ.get(_TARGET_ENV, "")
    if not raw:
        return None
    regime_name, separator, period = raw.partition("@")
    if not separator or not regime_name or not period.isdigit():
        msg = (
            f"{_TARGET_ENV} must be '<regime>@<period>', e.g. 'retiree@12'; got {raw!r}"
        )
        raise ValueError(msg)
    return regime_name, int(period)


def describe_sharding(*, sharding: object) -> ShardingDescriptor:
    """Render one concrete placement as portable data.

    A `NamedSharding` keeps its mesh shape and `PartitionSpec`, so the same
    partitioning can be rebuilt over a differently numbered device list. Every
    other concrete sharding keeps only its ordered device ids, which is what
    `jax.device_put` needs to restore it.

    Raises:
        TypeError: The object is not a concrete `jax.sharding.Sharding`.

    """
    if not isinstance(sharding, jax.sharding.Sharding):
        msg = f"A layout descriptor requires a concrete sharding; got {sharding!r}."
        raise TypeError(msg)
    memory_kind = sharding.memory_kind
    if isinstance(sharding, jax.NamedSharding):
        mesh = sharding.mesh
        return ShardingDescriptor(
            kind="NamedSharding",
            device_ids=tuple(
                int(device.id) for device in mesh.devices.reshape(-1).tolist()
            ),
            partition_spec=tuple(sharding.spec),
            mesh_axis_names=tuple(str(name) for name in mesh.axis_names),
            mesh_axis_sizes=tuple(int(size) for size in mesh.devices.shape),
            memory_kind=memory_kind,
        )
    return ShardingDescriptor(
        kind=type(sharding).__name__,
        device_ids=tuple(sorted(int(device.id) for device in sharding.device_set)),
        partition_spec=None,
        mesh_axis_names=None,
        mesh_axis_sizes=None,
        memory_kind=memory_kind,
    )


def describe_array_leaves(*, tree: object) -> tuple[LeafLayoutDescriptor, ...]:
    """Describe every `jax.Array` leaf of a pytree, in leaf order.

    Non-array leaves — a logger, a frozen set of artifact keys, a Python int —
    carry no placement and are skipped; the tree path keeps each descriptor
    addressable when the tree is rebuilt.
    """
    described: list[LeafLayoutDescriptor] = []
    for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
        if not isinstance(leaf, jax.Array):
            continue
        described.append(
            LeafLayoutDescriptor(
                tree_path=jax.tree_util.keystr(path),
                shape=tuple(int(size) for size in leaf.shape),
                dtype=str(leaf.dtype),
                weak_type=bool(leaf.weak_type),
                committed=bool(leaf.committed),
                sharding=describe_sharding(sharding=leaf.sharding),
            )
        )
    return tuple(described)


def describe_core(*, name: str, core: PlannedCore) -> CoreLayoutDescriptor | None:
    """Describe one core's lowering and compilation placement, or `None`.

    An executable that publishes no `input_shardings` or `output_shardings` —
    the eager route's `_EagerCore` is one — carries no placement to record, and
    describing it as an empty tuple would let a replay compare two empty tuples
    and call that agreement. Such a core is recorded as absent instead, which
    keeps the capture writable on the eager route and leaves the refusal to
    `replay_period_on_recorded_layout`.
    """
    input_shardings = getattr(core.compiled, "input_shardings", None)
    output_shardings = getattr(core.compiled, "output_shardings", None)
    if input_shardings is None or output_shardings is None:
        return None
    donated = tuple(core.donated_arguments)
    return CoreLayoutDescriptor(
        name=name,
        lowered_out_shardings=tuple(
            describe_sharding(sharding=sharding)
            for sharding in jax.tree.leaves(core.layout.out_shardings)
        ),
        compiled_input_shardings=_describe_named_shardings(tree=input_shardings),
        compiled_output_shardings=_describe_named_shardings(tree=output_shardings),
        input_transfer_plan=tuple(
            _describe_transfer(transfer=transfer)
            for transfer in core.input_transfer_plan
        ),
        donated_arguments=donated,
        variant=DONATING_VARIANT if donated else NON_DONATING_VARIANT,
    )


def capture_kernel_inputs(
    *,
    capture_target: PeriodCaptureTarget | None,
    regime: Any,  # noqa: ANN401 - the canonical Regime, circular to import here
    regime_name: RegimeName,
    period: int,
    kernel_kwargs: dict[str, Any],
    compiled_cores: Mapping[str, PlannedCore],
) -> None:
    """Write this regime-period's kernel inputs if it is the selected target.

    The compiled cores are deliberately absent: they are rebuilt at replay from
    the captured regime, which is what keeps the capture portable. Their selected
    tile widths are portable static choices, however, and are captured exactly so
    replay never runs the workspace planner again. Their placement is captured as
    descriptors, so a replay can reinstate the layout without carrying a buffer.
    """
    if capture_target != (regime_name, period):
        return

    directory = Path(os.environ.get(_DIR_ENV, ".")) / f"{regime_name}@{period}"
    directory.mkdir(parents=True, exist_ok=True)
    _save_pkl(
        path=directory / _PAYLOAD_NAME,
        obj={
            "regime": regime,
            "period": period,
            "kernel_kwargs": kernel_kwargs,
            "core_tile_widths": {
                core_name: dict(core.tile_widths)
                for core_name, core in compiled_cores.items()
            },
            LAYOUTS_KEY: _period_layouts(
                regime=regime,
                period=period,
                kernel_kwargs=kernel_kwargs,
                compiled_cores=compiled_cores,
            ),
        },
    )


def _period_layouts(
    *,
    regime: Any,  # noqa: ANN401 - the canonical Regime, circular to import here
    period: int,
    kernel_kwargs: dict[str, Any],
    compiled_cores: Mapping[str, PlannedCore],
) -> PeriodLayouts:
    """Collect the layout descriptors of one regime-period."""
    kernel = regime.solution.period_kernels[period]
    kernel_type = type(kernel)
    leaves = describe_array_leaves(tree=kernel_kwargs)
    cores = {
        core_name: describe_core(name=core_name, core=core)
        for core_name, core in compiled_cores.items()
    }
    return PeriodLayouts(
        route=f"{kernel_type.__module__}.{kernel_type.__qualname__}",
        device_ids=_layout_device_ids(leaves=leaves, cores=cores),
        leaves=leaves,
        cores=cores,
    )


def _layout_device_ids(
    *,
    leaves: tuple[LeafLayoutDescriptor, ...],
    cores: Mapping[str, CoreLayoutDescriptor | None],
) -> tuple[int, ...]:
    """Return the ids of the devices the recorded placements use, in host order.

    Only these devices need a stand-in at replay, so a capture taken on a subset of
    the visible devices replays on any host with as many devices. An uncommitted
    leaf is left where the backend puts it at replay and is never substituted, so
    its default device is not part of the layout.
    """
    shardings = [leaf.sharding for leaf in leaves if leaf.committed]
    for core in cores.values():
        if core is None:
            continue
        shardings.extend(core.lowered_out_shardings)
        shardings.extend(
            sharding
            for _, sharding in (
                *core.compiled_input_shardings,
                *core.compiled_output_shardings,
            )
        )
        for transfer in core.input_transfer_plan:
            shardings.extend((transfer.stored_sharding, transfer.source_sharding))
    used = {device_id for sharding in shardings for device_id in sharding.device_ids}
    return tuple(int(device.id) for device in jax.devices() if int(device.id) in used)


def _describe_named_shardings(
    *, tree: object
) -> tuple[tuple[str, ShardingDescriptor], ...]:
    """Describe every concrete sharding of a compiled executable's placement tree."""
    described: list[tuple[str, ShardingDescriptor]] = []
    for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
        if not isinstance(leaf, jax.sharding.Sharding):
            continue
        described.append((jax.tree_util.keystr(path), describe_sharding(sharding=leaf)))
    return tuple(described)


def _describe_transfer(*, transfer: ResolvedValueTransfer) -> ValueTransferDescriptor:
    """Describe one resolved stored-value transfer as portable data."""
    return ValueTransferDescriptor(
        kind=str(transfer.kind),
        target=repr(transfer.target),
        source=repr(transfer.source),
        stored_sharding=describe_sharding(sharding=transfer.stored_sharding),
        source_sharding=describe_sharding(sharding=transfer.source_sharding),
        expected_shape=tuple(int(size) for size in transfer.expected_shape),
        expected_dtype=str(transfer.expected_dtype),
    )


def rebuild_sharding(
    *,
    descriptor: ShardingDescriptor,
    device_by_recorded_id: Mapping[int, jax.Device],
) -> jax.sharding.Sharding:
    """Rebuild one recorded placement over the devices standing in for it.

    Raises:
        ValueError: The descriptor names a device the mapping does not cover, or
            a sharding kind this route cannot rebuild.

    """
    missing = [
        device_id
        for device_id in descriptor.device_ids
        if device_id not in device_by_recorded_id
    ]
    if missing:
        msg = (
            "A recorded layout names device ids that the given devices do not "
            f"stand in for: recorded={descriptor.device_ids!r}, missing={missing!r}."
        )
        raise ValueError(msg)
    devices = tuple(
        device_by_recorded_id[device_id] for device_id in descriptor.device_ids
    )
    if descriptor.kind == "NamedSharding":
        if descriptor.mesh_axis_names is None or descriptor.mesh_axis_sizes is None:
            msg = "A recorded NamedSharding must carry its mesh axis names and sizes."
            raise ValueError(msg)
        mesh = jax.sharding.Mesh(
            _device_grid(devices=devices, sizes=descriptor.mesh_axis_sizes),
            descriptor.mesh_axis_names,
        )
        return jax.NamedSharding(
            mesh=mesh,
            spec=jax.P(*descriptor.partition_spec or ()),
            memory_kind=descriptor.memory_kind,
        )
    if descriptor.kind == "SingleDeviceSharding":
        return jax.sharding.SingleDeviceSharding(
            devices[0], memory_kind=descriptor.memory_kind
        )
    msg = f"rebuild_sharding does not rebuild sharding kind {descriptor.kind!r}."
    raise ValueError(msg)


def _device_grid(*, devices: Sequence[jax.Device], sizes: Sequence[int]) -> np.ndarray:
    """Arrange devices into the recorded mesh shape, row-major."""
    return np.array(devices, dtype=object).reshape(tuple(sizes))
