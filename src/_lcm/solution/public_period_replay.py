"""Validate a numerical capture against a fresh model and replay one adapter."""

import logging
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from types import MappingProxyType, NoneType
from typing import TypedDict, TypeGuard

import jax
import numpy as np
from jax._src.lax.lax import _convert_element_type

from _lcm.engine import Regime, placed_devices_for_ids
from _lcm.execution.execution_plan import ResolvedExecution
from _lcm.execution.output_layout import PlannedCore
from _lcm.execution.workspace_planning import compiler_memory_reservation
from _lcm.persistence.period import read_period_archive
from _lcm.solution.backward_induction import (
    _build_base_state_action_spaces,
    _run_period_kernel,
)
from _lcm.solution.period_capture import (
    CoreLayoutDescriptor,
    LeafLayoutDescriptor,
    PeriodKernelContext,
    PeriodKernelKwargs,
    PeriodLayouts,
    RenderedAddress,
    ShardingDescriptor,
    ValueTransferDescriptor,
    describe_array_leaves,
    rebuild_sharding,
)
from _lcm.solution.period_replay import prepare_recorded_period
from _lcm.solution.public_period_capture import (
    load_period_capture,
    optimized_hlo_records,
    period_identity,
    validate_capture_route,
)
from _lcm.solution.v_topology import _get_regime_V_shapes_and_shardings
from _lcm.time import TimeAxis
from _lcm.typing import FlatParams, HostArray, JSONValue, RegimeName
from _lcm.utils.logging import get_logger
from lcm.period_capture import CapturedPeriodReplay


class _HostKernelKwargs(PeriodKernelContext):
    """Period kernel arguments whose value arrays are still host NumPy arrays."""

    next_regime_to_V_arr: MappingProxyType[RegimeName, HostArray]
    """Next period's persisted value array per regime."""

    logger: logging.Logger
    """Logger the adapter reports through."""

    period_solution: Mapping[RegimeName, HostArray]
    """Persisted value arrays of the regimes already solved in this period.

    A plain dict, as the solve loop passes it, so the restored leaf paths match
    the captured ones.
    """


class _ShardingRecord(TypedDict):
    """A `ShardingDescriptor` as the capture archive's JSON holds it."""

    kind: str
    device_ids: list[int]
    partition_spec: list[str | list[str] | None] | None
    mesh_axis_names: list[str] | None
    mesh_axis_sizes: list[int] | None
    memory_kind: str | None


class _LeafRecord(TypedDict):
    """A `LeafLayoutDescriptor` as the capture archive's JSON holds it."""

    tree_path: str
    shape: list[int]
    dtype: str
    weak_type: bool
    committed: bool
    sharding: _ShardingRecord


class _TransferRecord(TypedDict):
    """A `ValueTransferDescriptor` as the capture archive's JSON holds it."""

    kind: str
    target: RenderedAddress
    source: RenderedAddress
    stored_sharding: _ShardingRecord
    source_sharding: _ShardingRecord
    expected_shape: list[int]
    expected_dtype: str


class _CoreRecord(TypedDict):
    """A `CoreLayoutDescriptor` as the capture archive's JSON holds it.

    Each compiled placement is a `[path, sharding]` pair.
    """

    name: str
    lowered_out_shardings: list[_ShardingRecord]
    compiled_input_shardings: list[list[str | _ShardingRecord]]
    compiled_output_shardings: list[list[str | _ShardingRecord]]
    input_transfer_plan: list[_TransferRecord]
    donated_arguments: list[str]
    variant: str


class _LayoutsRecord(TypedDict):
    """`PeriodLayouts` as the capture archive's JSON holds it."""

    route: str
    device_ids: list[int]
    leaves: list[_LeafRecord]
    cores: dict[str, _CoreRecord | None]


class _AdmissionRecord(TypedDict):
    """One captured core's admission, as the capture archive's JSON holds it."""

    budget_bytes: int | None
    resident_bytes: int | None
    reservation_bytes: int
    peak_bytes: int


class _CaptureMetadata(TypedDict):
    """The capture-archive metadata entries replay reads, at their JSON types."""

    regime: RegimeName
    period: int
    layouts: _LayoutsRecord
    widths: dict[str, dict[str, int]]
    admission: dict[str, _AdmissionRecord]
    optimized_hlo: dict[str, dict[str, str]]
    retain_replay: bool


type _EntryCheck = Callable[[JSONValue], bool]


def replay_public_period(
    *,
    directory: Path,
    flat_params: FlatParams,
    regimes: MappingProxyType[str, Regime],
    ages: TimeAxis,
    execution: ResolvedExecution,
    enable_jit: bool,
    source_identity: Mapping[str, str],
    model_fingerprint: str,
    params_fingerprint: str,
    require_reference: bool,
) -> CapturedPeriodReplay:
    """Bind, compile and validate one captured period before dispatching it."""
    record = load_period_capture(directory=directory)
    metadata: dict[str, JSONValue] = dict(record.metadata)
    expected_identity = period_identity(
        model_fingerprint=model_fingerprint,
        params_fingerprint=params_fingerprint,
        source_identity=source_identity,
        execution=execution,
    )
    if metadata.get("identity") != expected_identity:
        raise ValueError(
            "Period capture identity is incompatible with this model and runtime."
        )
    if require_reference and not record.completed:
        raise ValueError(
            "Period capture has no completed reference; "
            "use require_reference=False for inputs only."
        )
    if not _is_capture_metadata(metadata):
        raise ValueError("Period capture metadata must be a JSON object.")
    name, period = metadata["regime"], metadata["period"]
    validate_capture_route(
        periods=((name, period),),
        regimes=regimes,
        execution=execution,
        enable_jit=enable_jit,
    )
    entry_metadata, arrays, _ = read_period_archive(path=directory / "entry.h5")
    if entry_metadata != metadata:
        raise ValueError("Period capture changed while being read.")
    layouts = _decode_layouts(metadata["layouts"])
    devices = placed_devices_for_ids(
        submesh_device_ids=layouts.device_ids, visible_device_ids=execution.device_ids
    )
    kernel_kwargs = _restore_public_inputs(
        arrays=arrays,
        regimes=regimes,
        flat_params=flat_params,
        execution=execution,
        name=name,
        period=period,
        ages=ages,
        retain_replay=metadata["retain_replay"],
    )
    kernel_kwargs = _restore_array_leaves(
        kernel_kwargs=kernel_kwargs,
        leaves=layouts.leaves,
        device_by_recorded_id={int(device.id): device for device in devices},
    )
    observed_leaves = describe_array_leaves(tree=kernel_kwargs)
    if tuple(
        (leaf.tree_path, leaf.shape, leaf.dtype, leaf.weak_type)
        for leaf in observed_leaves
    ) != tuple(
        (leaf.tree_path, leaf.shape, leaf.dtype, leaf.weak_type)
        for leaf in layouts.leaves
    ):
        raise ValueError(
            "Captured input layout identity is incompatible with reconstructed inputs."
        )
    kernel_kwargs, cores = prepare_recorded_period(
        payload={
            "regime": regimes[name],
            "period": period,
            "kernel_kwargs": kernel_kwargs,
            "layouts": layouts,
            "core_tile_widths": metadata["widths"],
        },
        directory=directory,
        devices=devices,
    )
    hlo = optimized_hlo_records(compiled_cores=cores)
    _validate_recorded_admission(
        cores=cores,
        records=metadata["admission"],
        budget=execution.device_memory_bytes,
    )
    if hlo != metadata["optimized_hlo"]:
        raise ValueError(
            "Replayed optimized HLO identity is incompatible "
            "with the captured executable."
        )
    jax.block_until_ready(
        (kernel_kwargs["next_regime_to_V_arr"], kernel_kwargs["period_solution"])
    )
    started = time.perf_counter()
    output = _run_period_kernel(
        regime=regimes[name], capture_target=None, compiled_cores=cores, **kernel_kwargs
    )
    jax.block_until_ready(output.value)
    seconds = time.perf_counter() - started
    reference_matches = None
    if record.reference is not None:
        actual = np.asarray(output.value)
        reference_matches = (
            actual.shape == record.reference.shape
            and actual.dtype == record.reference.dtype
            and actual.tobytes() == record.reference.tobytes()
            and all(
                np.array_equal(func(actual), func(record.reference))
                for func in (np.isnan, np.isposinf, np.isneginf)
            )
        )
    return CapturedPeriodReplay(
        value=output.value,
        reference_matches=reference_matches,
        optimized_hlo_matches=True,
        in_context_seconds=record.in_context_seconds,
        replay_seconds=seconds,
        capture=record,
    )


def _restore_public_inputs(
    *,
    arrays: Mapping[str, np.ndarray],
    regimes: MappingProxyType[str, Regime],
    flat_params: FlatParams,
    execution: ResolvedExecution,
    name: str,
    period: int,
    ages: TimeAxis,
    retain_replay: bool,
) -> _HostKernelKwargs:
    """Validate persisted values against fresh topology and rebuild adapter inputs."""
    topology = _get_regime_V_shapes_and_shardings(
        regimes=regimes,
        flat_params=flat_params,
        device_ids=execution.device_ids,
    )
    values: dict[str, dict[str, np.ndarray]] = {
        "next_regime_to_V_arr": {},
        "period_solution": {},
    }
    dtype = np.dtype("float64" if jax.config.jax_enable_x64 else "float32")
    for key, array in arrays.items():
        channel, regime_name = key.split("/", 1)
        if channel not in values or regime_name not in topology:
            raise ValueError("Captured input address is incompatible with this model.")
        if array.shape != topology[regime_name].shape or array.dtype != dtype:
            raise ValueError(
                "Captured input shape or dtype is incompatible with this model."
            )
        values[channel][regime_name] = array
    if set(values["next_regime_to_V_arr"]) != set(regimes):
        raise ValueError("Captured continuation value coordinates are incomplete.")
    return {
        "regime_name": name,
        "period": period,
        "state_action_space": _build_base_state_action_spaces(
            regimes=regimes, flat_params=flat_params
        )[name],
        "flat_params": flat_params,
        "ages": ages,
        # MappingProxyType's pytree children retain insertion order. The archive
        # sorts JSON keys, so reconstruct the production topology's order here.
        "next_regime_to_V_arr": MappingProxyType(
            {key: values["next_regime_to_V_arr"][key] for key in topology}
        ),
        "next_regime_to_continuation": MappingProxyType({}),
        "next_edge_to_V_arr": MappingProxyType({}),
        "period_solution": values["period_solution"],
        "logger": get_logger(log_level="off"),
        "retain_replay": retain_replay,
        "selected_artifact_keys": frozenset(),
    }


def _restore_array_leaves(
    *,
    kernel_kwargs: _HostKernelKwargs,
    leaves: tuple[LeafLayoutDescriptor, ...],
    device_by_recorded_id: Mapping[int, jax.Device],
) -> PeriodKernelKwargs:
    """Upload host values directly to recorded devices and restore weak typing."""
    descriptors = {leaf.tree_path: leaf for leaf in leaves}
    flat, treedef = jax.tree_util.tree_flatten_with_path(kernel_kwargs)
    restored = []
    for path, leaf in flat:
        value = leaf
        descriptor = descriptors.get(jax.tree_util.keystr(path))
        if isinstance(value, np.ndarray):
            # Persisted V arrays are committed in production. Fresh parameter
            # leaves are already JAX arrays and may legitimately be uncommitted.
            if (
                descriptor is None
                or not descriptor.committed
                or value.shape != descriptor.shape
                or str(value.dtype) != descriptor.dtype
            ):
                raise ValueError(
                    "Captured value layout is incompatible with its payload."
                )
            value = jax.device_put(
                value,
                rebuild_sharding(
                    descriptor=descriptor.sharding,
                    device_by_recorded_id=device_by_recorded_id,
                ),
            )
        if (
            isinstance(value, jax.Array)
            and descriptor is not None
            and value.weak_type != descriptor.weak_type
        ):
            # JAX's public conversion forces strong typing. Its internal
            # conversion preserves the dtype and values while restoring this
            # abstract property; the runtime identity pins the JAX version.
            restored.append(
                _convert_element_type(
                    operand=value,
                    new_dtype=value.dtype,
                    weak_type=descriptor.weak_type,
                )
            )
        else:
            restored.append(value)
    return jax.tree_util.tree_unflatten(treedef, restored)


def _validate_recorded_admission(
    *,
    cores: Mapping[str, PlannedCore],
    records: Mapping[str, _AdmissionRecord],
    budget: int | None,
) -> None:
    """Recheck the selected executable under its recorded production residency."""
    if set(cores) != set(records):
        raise ValueError("Captured admission core identities differ.")
    for name, core in cores.items():
        record = records[name]
        memory = compiler_memory_reservation(
            compiled=core.compiled, widths=core.tile_widths
        )
        resident = record["resident_bytes"]
        if (
            record["budget_bytes"] != budget
            or record["reservation_bytes"] != memory.reservation_bytes
            or record["peak_bytes"] != memory.peak_bytes
            or (budget is None) != (resident is None)
            or (resident is not None and resident < 0)
        ):
            raise ValueError(
                "Captured admission is incompatible with the replay executable: "
                f"core={name!r}, captured={record!r}, "
                f"replay_budget={budget!r}, "
                f"replay_reservation={memory.reservation_bytes!r}, "
                f"replay_peak={memory.peak_bytes!r}."
            )
        if (
            budget is not None
            and resident is not None
            and resident + memory.reservation_bytes > budget
        ):
            raise ValueError("Replay exceeds the recorded admission budget.")


def _is_capture_metadata(raw: JSONValue, /) -> TypeGuard[_CaptureMetadata]:
    """Whether `raw` is capture metadata; a malformed entry is refused by name."""
    return _has_entries(
        raw=raw,
        record="metadata",
        checks={
            "regime": _exact(str),
            "period": _exact(int),
            "layouts": _is_layouts_record,
            "widths": _dict_of(value=_dict_of(value=_exact(int))),
            "admission": _dict_of(value=_is_admission_record),
            "optimized_hlo": _dict_of(value=_dict_of(value=_exact(str))),
            "retain_replay": _exact(bool),
        },
    )


def _is_admission_record(raw: JSONValue, /) -> TypeGuard[_AdmissionRecord]:
    """Whether `raw` is an admission record; a malformed entry is refused by name."""
    return _has_entries(
        raw=raw,
        record="admission",
        checks={
            "budget_bytes": _exact(int, NoneType),
            "resident_bytes": _exact(int, NoneType),
            "reservation_bytes": _exact(int),
            "peak_bytes": _exact(int),
        },
    )


def _is_layouts_record(raw: JSONValue, /) -> TypeGuard[_LayoutsRecord]:
    """Whether `raw` is a layout block; a malformed entry is refused by name."""
    return _has_entries(
        raw=raw,
        record="layouts",
        checks={
            "route": _exact(str),
            "device_ids": _list_of(item=_exact(int)),
            "leaves": _list_of(item=_is_leaf_record),
            "cores": _dict_of(value=_either(_exact(NoneType), _is_core_record)),
        },
    )


def _is_core_record(raw: JSONValue, /) -> TypeGuard[_CoreRecord]:
    """Whether `raw` is a core layout record; a malformed entry is refused by name."""
    named_sharding = _list_of(item=_either(_exact(str), _is_sharding_record))
    return _has_entries(
        raw=raw,
        record="core",
        checks={
            "name": _exact(str),
            "lowered_out_shardings": _list_of(item=_is_sharding_record),
            "compiled_input_shardings": _list_of(item=named_sharding),
            "compiled_output_shardings": _list_of(item=named_sharding),
            "input_transfer_plan": _list_of(item=_is_transfer_record),
            "donated_arguments": _list_of(item=_exact(str)),
            "variant": _exact(str),
        },
    )


def _is_transfer_record(raw: JSONValue, /) -> TypeGuard[_TransferRecord]:
    """Whether `raw` is a transfer record; a malformed entry is refused by name."""
    return _has_entries(
        raw=raw,
        record="transfer",
        checks={
            "kind": _exact(str),
            "target": _exact(str),
            "source": _exact(str),
            "stored_sharding": _is_sharding_record,
            "source_sharding": _is_sharding_record,
            "expected_shape": _list_of(item=_exact(int)),
            "expected_dtype": _exact(str),
        },
    )


def _is_leaf_record(raw: JSONValue, /) -> TypeGuard[_LeafRecord]:
    """Whether `raw` is an array-leaf record; a malformed entry is refused by name."""
    return _has_entries(
        raw=raw,
        record="leaf",
        checks={
            "tree_path": _exact(str),
            "shape": _list_of(item=_exact(int)),
            "dtype": _exact(str),
            "weak_type": _exact(bool),
            "committed": _exact(bool),
            "sharding": _is_sharding_record,
        },
    )


def _is_sharding_record(raw: JSONValue, /) -> TypeGuard[_ShardingRecord]:
    """Whether `raw` is a sharding record; a malformed entry is refused by name."""
    names = _list_of(item=_exact(str))
    return _has_entries(
        raw=raw,
        record="sharding",
        checks={
            "kind": _exact(str),
            "device_ids": _list_of(item=_exact(int)),
            "partition_spec": _either(
                _exact(NoneType), _list_of(item=_either(_exact(str, NoneType), names))
            ),
            "mesh_axis_names": _either(_exact(NoneType), names),
            "mesh_axis_sizes": _either(_exact(NoneType), _list_of(item=_exact(int))),
            "memory_kind": _exact(str, NoneType),
        },
    )


def _has_entries(
    *, raw: JSONValue, record: str, checks: Mapping[str, _EntryCheck]
) -> bool:
    """Whether `raw` is a JSON object whose named entries pass their checks.

    Returns `False` for anything but a JSON object. An object that lacks a named
    entry, or holds one that fails its check, is refused naming that entry.
    """
    if not isinstance(raw, dict):
        return False
    for key, check in checks.items():
        if key not in raw:
            msg = f"The period capture's {record} record is missing {key!r}."
            raise ValueError(msg)
        if not check(raw[key]):
            msg = (
                f"The period capture's {record} record holds {key!r} at the wrong "
                f"type ({type(raw[key]).__name__})."
            )
            raise TypeError(msg)
    return True


def _exact(*types: type[bool | int | float | str | None]) -> _EntryCheck:
    """Accept a JSON value whose type is exactly one of `types`.

    Exact, so a JSON `true` is never taken for an integer.
    """
    return lambda value: type(value) in types


def _either(*checks: _EntryCheck) -> _EntryCheck:
    """Accept a JSON value that passes any of `checks`."""
    return lambda value: any(check(value) for check in checks)


def _list_of(*, item: _EntryCheck) -> _EntryCheck:
    """Accept a JSON array whose every item passes `item`."""
    return lambda value: (
        isinstance(value, list) and all(item(element) for element in value)
    )


def _dict_of(*, value: _EntryCheck) -> _EntryCheck:
    """Accept a JSON object with string keys whose every value passes `value`."""
    return lambda raw: (
        isinstance(raw, dict)
        and all(type(key) is str and value(entry) for key, entry in raw.items())
    )


def _decode_sharding(raw: _ShardingRecord) -> ShardingDescriptor:
    """Reconstruct only the fixed non-executable sharding descriptor schema."""
    partition_spec = raw["partition_spec"]
    mesh_axis_names = raw["mesh_axis_names"]
    mesh_axis_sizes = raw["mesh_axis_sizes"]
    return ShardingDescriptor(
        kind=raw["kind"],
        device_ids=tuple(raw["device_ids"]),
        partition_spec=None
        if partition_spec is None
        else tuple(
            tuple(item) if isinstance(item, list) else item for item in partition_spec
        ),
        mesh_axis_names=None if mesh_axis_names is None else tuple(mesh_axis_names),
        mesh_axis_sizes=None if mesh_axis_sizes is None else tuple(mesh_axis_sizes),
        memory_kind=raw["memory_kind"],
    )


def _decode_named_sharding(
    pair: list[str | _ShardingRecord],
) -> tuple[str, ShardingDescriptor]:
    """Decode one `[path, sharding]` pair of a compiled placement."""
    path, sharding = pair
    if not isinstance(path, str) or isinstance(sharding, str):
        raise TypeError("Invalid compiled placement in the period capture.")
    return path, _decode_sharding(sharding)


def _decode_transfer(raw: _TransferRecord) -> ValueTransferDescriptor:
    """Decode one stored-value transfer descriptor."""
    return ValueTransferDescriptor(
        kind=raw["kind"],
        target=raw["target"],
        source=raw["source"],
        stored_sharding=_decode_sharding(raw["stored_sharding"]),
        source_sharding=_decode_sharding(raw["source_sharding"]),
        expected_shape=tuple(raw["expected_shape"]),
        expected_dtype=raw["expected_dtype"],
    )


def _decode_core(raw: _CoreRecord) -> CoreLayoutDescriptor:
    """Decode the existing strict core layout and transfer descriptors."""
    if raw["donated_arguments"]:
        raise ValueError("Public capture replay does not support donated inputs.")
    return CoreLayoutDescriptor(
        name=raw["name"],
        lowered_out_shardings=tuple(
            _decode_sharding(item) for item in raw["lowered_out_shardings"]
        ),
        compiled_input_shardings=tuple(
            _decode_named_sharding(pair) for pair in raw["compiled_input_shardings"]
        ),
        compiled_output_shardings=tuple(
            _decode_named_sharding(pair) for pair in raw["compiled_output_shardings"]
        ),
        input_transfer_plan=tuple(
            _decode_transfer(item) for item in raw["input_transfer_plan"]
        ),
        donated_arguments=(),
        variant=raw["variant"],
    )


def _decode_leaf(raw: _LeafRecord) -> LeafLayoutDescriptor:
    """Decode one captured array leaf's layout descriptor."""
    return LeafLayoutDescriptor(
        tree_path=raw["tree_path"],
        shape=tuple(raw["shape"]),
        dtype=raw["dtype"],
        weak_type=raw["weak_type"],
        committed=raw["committed"],
        sharding=_decode_sharding(raw["sharding"]),
    )


def _decode_layouts(raw: _LayoutsRecord) -> PeriodLayouts:
    """Decode fixed descriptors without importing a type named by the archive."""
    return PeriodLayouts(
        route=raw["route"],
        device_ids=tuple(raw["device_ids"]),
        leaves=tuple(_decode_leaf(leaf) for leaf in raw["leaves"]),
        cores=MappingProxyType(
            {
                name: None if core is None else _decode_core(core)
                for name, core in raw["cores"].items()
            }
        ),
    )
