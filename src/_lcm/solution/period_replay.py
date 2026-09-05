"""Re-run a single captured regime-period without solving the ones above it.

`_lcm.solution.period_capture` writes the inputs; this module runs them back
through the same funnel the solve loop uses, so a replay repeats every step the
original call took rather than approximating it.

The split is what keeps the import graph acyclic: the capture side is imported
by the backward-induction loop, and the replay side imports that loop.
"""

import dataclasses
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import cloudpickle
import jax

from _lcm.engine import StateActionSpace
from _lcm.execution.compiler_memory import (
    CompilerMemoryBytes,
    compiler_memory_bytes,
)
from _lcm.execution.core_program import (
    CoreBuildContext,
    MaterializedCoreProgram,
    core_program_graph,
    materialize_core_program,
    select_programs,
)
from _lcm.execution.internal_outputs import (
    internal_input_templates,
    topological_program_order,
)
from _lcm.execution.output_layout import PlannedCore, resolve_output_layout
from _lcm.solution.backward_induction import (
    _assert_lowered_output_roles,
    _attach_resolved_output_layout,
    _edge_kwargs,
    _resolve_program_for_execution,
    _run_period_kernel,
)
from _lcm.solution.period_capture import _PAYLOAD_NAME
from _lcm.typing import RegimeName
from lcm.solver_api import KernelOutput


@dataclasses.dataclass(frozen=True)
class PeriodReplay:
    """One regime-period re-run from a capture."""

    regime_name: RegimeName
    """Name of the regime the captured kernel belongs to."""

    period: int
    """Index of the captured period in the model's age grid."""

    age: float
    """Age the captured period sits at, for reading against a solve log."""

    output: KernelOutput
    """What the kernel returned: the value array and its artifact channels."""


@dataclasses.dataclass(frozen=True)
class PeriodCoreMemoryAnalysis:
    """Compiler memory of one captured period's cores at capture-roundtrip placement.

    The capture preserves the period's logical pytrees and array shapes. Its pickle
    round trip does not preserve production sharding, so these byte counts describe
    executables lowered with the restored arrays' default backend placement. They are
    not production-layout or production-memory measurements.
    """

    regime_name: RegimeName
    """Name of the regime the captured kernel belongs to."""

    period: int
    """Index of the captured period in the model's age grid."""

    age: float
    """Age the captured period sits at, for reading against a solve log."""

    preserves_production_sharding: bool
    """Always false: period capture does not serialize production sharding."""

    core_memory_bytes: MappingProxyType[str, CompilerMemoryBytes | None]
    """Byte counts per production core, keyed as the kernel publishes them."""


def replay_period(*, directory: Path) -> PeriodReplay:
    """Re-run the regime-period captured in `directory`.

    The cores are lowered and compiled for this one period only, so the call
    costs one kernel rather than a backward induction. The returned `V_arr` is
    what the original solve produced for that regime-period.

    Args:
        directory: A capture directory written during a solve.

    Returns:
        The captured identity and the kernel's result.

    """
    payload = _load_capture_payload(directory=directory)

    regime = payload["regime"]
    period = payload["period"]
    kernel_kwargs = payload["kernel_kwargs"]
    core_tile_widths = payload["core_tile_widths"]

    output = _run_period_kernel(
        regime=regime,
        capture_target=None,
        compiled_cores=_compile_cores_for_one_period(
            regime=regime,
            period=period,
            kernel_kwargs=kernel_kwargs,
            core_tile_widths=core_tile_widths,
        ),
        **kernel_kwargs,
    )
    return PeriodReplay(
        regime_name=kernel_kwargs["regime_name"],
        period=period,
        age=float(kernel_kwargs["ages"].values[period]),
        output=output,
    )


def analyze_period_core_memory(*, directory: Path) -> PeriodCoreMemoryAnalysis:
    """Compile, but never execute, one captured period's cores and read their memory.

    The production cores are lowered against the capture's logical pytrees and
    production shapes. The capture round trip loses device sharding, so every executable
    uses the restored arrays' default backend placement. Calling `memory_analysis()`
    on the resulting executables is safe even when their estimated runtime
    allocation exceeds the available device budget because this function
    deliberately has no execution path.

    Args:
        directory: A capture directory written during a solve.

    Returns:
        The captured identity and compiler memory per core.

    """
    payload = _load_capture_payload(directory=directory)
    regime = payload["regime"]
    period = payload["period"]
    kernel_kwargs = payload["kernel_kwargs"]
    core_tile_widths = payload["core_tile_widths"]

    production = _compile_cores_for_one_period(
        regime=regime,
        period=period,
        kernel_kwargs=kernel_kwargs,
        core_tile_widths=core_tile_widths,
    )
    return PeriodCoreMemoryAnalysis(
        regime_name=kernel_kwargs["regime_name"],
        period=period,
        age=float(kernel_kwargs["ages"].values[period]),
        preserves_production_sharding=False,
        core_memory_bytes=MappingProxyType(
            {
                key: compiler_memory_bytes(compiled=core.compiled)
                for key, core in production.items()
            }
        ),
    )


def _load_capture_payload(*, directory: Path) -> dict[str, Any]:
    """Load one period's logical inputs; array sharding is not round-tripped."""
    with (directory / _PAYLOAD_NAME).open("rb") as stream:
        payload = cloudpickle.load(stream)
    if not isinstance(payload, dict):
        msg = "A period capture payload must be a dictionary."
        raise TypeError(msg)
    if "core_tile_widths" not in payload:
        msg = "Period capture is missing required 'core_tile_widths'."
        raise ValueError(msg)
    loaded = dict(payload)
    loaded["core_tile_widths"] = _normalize_core_tile_widths(
        raw=payload["core_tile_widths"]
    )
    return loaded


def _normalize_core_tile_widths(
    *, raw: object
) -> MappingProxyType[str, MappingProxyType[str, int]]:
    """Validate and freeze the portable width map stored in a capture."""
    if not isinstance(raw, Mapping):
        msg = "Period capture 'core_tile_widths' must be a mapping."
        raise TypeError(msg)

    normalized: dict[str, MappingProxyType[str, int]] = {}
    for core_name, raw_widths in raw.items():
        if not isinstance(core_name, str) or not core_name:
            msg = "Captured core names must be non-empty strings."
            raise TypeError(msg)
        if not isinstance(raw_widths, Mapping):
            msg = f"Captured tile widths for core {core_name!r} must be a mapping."
            raise TypeError(msg)

        widths: dict[str, int] = {}
        for axis_name, width in raw_widths.items():
            if not isinstance(axis_name, str) or not axis_name:
                msg = (
                    f"Captured tile-width names for core {core_name!r} must be "
                    "non-empty strings."
                )
                raise TypeError(msg)
            if type(width) is not int:
                msg = (
                    f"Captured tile width for {core_name!r}/{axis_name!r} must be "
                    "an exact integer."
                )
                raise TypeError(msg)
            if width <= 0:
                msg = (
                    f"Captured tile width for {core_name!r}/{axis_name!r} must be "
                    "positive."
                )
                raise ValueError(msg)
            widths[axis_name] = width
        normalized[core_name] = MappingProxyType(widths)
    return MappingProxyType(normalized)


def _require_exact_core_tile_widths(
    *,
    raw: object,
    core_names: tuple[str, ...],
) -> MappingProxyType[str, MappingProxyType[str, int]]:
    """Require one captured width map for every selected replay core."""
    widths = _normalize_core_tile_widths(raw=raw)
    expected = frozenset(core_names)
    actual = frozenset(widths)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        msg = (
            "Captured core_tile_widths must match the selected replay cores "
            f"exactly; missing={missing!r}, extra={extra!r}."
        )
        raise ValueError(msg)
    return widths


def _compile_cores_for_one_period(
    *,
    regime: Any,  # noqa: ANN401 - the canonical Regime, circular to import here
    period: int,
    kernel_kwargs: dict[str, Any],
    core_tile_widths: object,
) -> MappingProxyType[str, PlannedCore]:
    """Lower and compile the cores of a single period.

    The solve loop compiles every regime-period up front and deduplicates
    identical cores across them. A replay wants neither: it needs exactly the
    cores this one period calls.
    """
    period_kernel = regime.solution.period_kernels[period]
    context = _core_build_context_for_one_period(
        regime=regime, period=period, kernel_kwargs=kernel_kwargs
    )
    graph = select_programs(
        graph=core_program_graph(kernel=period_kernel),
        retain_replay=kernel_kwargs["retain_replay"],
        selected_artifact_keys=kernel_kwargs["selected_artifact_keys"],
    )
    captured_widths = _require_exact_core_tile_widths(
        raw=core_tile_widths,
        core_names=tuple(graph),
    )
    compiled: dict[str, PlannedCore] = {}
    producers: dict[str, MaterializedCoreProgram] = {}
    for core_name in topological_program_order(graph=graph):
        declaration = graph[core_name]
        materialized = materialize_core_program(program=declaration, context=context)
        producers[core_name] = materialized
        templates = internal_input_templates(program=materialized, producers=producers)
        resolved = _resolve_program_for_execution(
            program=materialized,
            tile_widths=captured_widths[core_name],
            source_value_template=context.next_regime_to_V_arr[
                kernel_kwargs["regime_name"]
            ],
            source=(kernel_kwargs["regime_name"], period, core_name),
        )
        state_action_space = cast("StateActionSpace", context.state_action_space)
        state_order = tuple(
            name
            for name in state_action_space.states
            if name not in regime.fold_state_names
        )
        layout = resolve_output_layout(
            core_key=core_name,
            value_template=context.next_regime_to_V_arr[kernel_kwargs["regime_name"]],
            state_order=state_order,
            output_roles=resolved.output_roles,
        )
        jitted = jax.jit(
            resolved.function,
            static_argnames=tuple(resolved.static_kwargs),
            out_shardings=layout.out_shardings,
        )
        lowered = jitted.lower(
            **resolved.arguments, **templates, **resolved.static_kwargs
        )
        _assert_lowered_output_roles(
            lowered=lowered,
            output_roles=resolved.output_roles,
            layout=layout,
            label=(
                f"{kernel_kwargs['regime_name']} {core_name} (replay period {period})"
            ),
        )
        executable = lowered.compile()
        compiled[core_name] = _attach_resolved_output_layout(
            compiled=executable,
            layout=layout,
            tile_widths=resolved.tile_widths,
            input_transfer_plan=resolved.input_transfer_plan,
            internal_input_templates=templates,
            name=core_name,
        )
    return MappingProxyType(compiled)


def _core_build_context_for_one_period(
    *,
    regime: Any,  # noqa: ANN401 - the canonical Regime, circular to import here
    period: int,
    kernel_kwargs: dict[str, Any],
) -> CoreBuildContext:
    """Build the immutable program context shared by replay and memory analysis."""
    # A source declaring gated edges reads its targets' folded continuation in
    # place of their raw V, so the kernel is compiled against a pytree the raw
    # mapping does not carry. The projection comes from the solve loop's own
    # builder rather than a second copy of it, which is what keeps the argument
    # tree and logical shapes a replay lowers against match production. Capture
    # serialization does not preserve sharding, so layout fidelity is explicitly
    # reported as default backend placement rather than implied here.
    edge_kwargs = _edge_kwargs(
        regime=regime,
        regime_name=kernel_kwargs["regime_name"],
        next_edge_to_V_arr=kernel_kwargs["next_edge_to_V_arr"],
    )
    return CoreBuildContext(
        state_action_space=kernel_kwargs["state_action_space"],
        next_regime_to_V_arr=kernel_kwargs["next_regime_to_V_arr"],
        next_regime_to_continuation=kernel_kwargs["next_regime_to_continuation"],
        flat_params=kernel_kwargs["flat_params"],
        period=period,
        ages=kernel_kwargs["ages"],
        edge_regime_to_V_arr=cast(
            "Mapping[str, object] | None",
            edge_kwargs.get("edge_regime_to_V_arr"),
        ),
    )
