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
from _lcm.execution.core_program import (
    CoreBuildContext,
    CoreExecutionDisposition,
    _LegacyArgumentBuilder,
    core_program_graph,
    materialize_core_program,
)
from _lcm.execution.output_layout import (
    UNPLANNED,
    resolve_output_layout,
)
from _lcm.solution.backward_induction import (
    _assert_lowered_output_roles,
    _attach_resolved_output_layout,
    _edge_kwargs,
    _resolve_program_for_execution,
    _run_period_kernel,
)
from _lcm.solution.contract import KernelResult
from _lcm.solution.period_capture import _PAYLOAD_NAME
from _lcm.typing import RegimeName


@dataclasses.dataclass(frozen=True)
class PeriodReplay:
    """One regime-period re-run from a capture."""

    regime_name: RegimeName
    """Name of the regime the captured kernel belongs to."""

    period: int
    """Index of the captured period in the model's age grid."""

    age: float
    """Age the captured period sits at, for reading against a solve log."""

    result: KernelResult
    """What the kernel returned — the value array and the optional payloads."""


@dataclasses.dataclass(frozen=True)
class CompilerMemoryBytes:
    """Backend-independent byte counts from JAX compiler memory analysis."""

    generated_code_size_in_bytes: int | None
    argument_size_in_bytes: int | None
    output_size_in_bytes: int | None
    alias_size_in_bytes: int | None
    temp_size_in_bytes: int | None
    peak_memory_in_bytes: int | None
    host_generated_code_size_in_bytes: int | None
    host_argument_size_in_bytes: int | None
    host_output_size_in_bytes: int | None
    host_alias_size_in_bytes: int | None
    host_temp_size_in_bytes: int | None


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

    split_oracle_memory_bytes: MappingProxyType[str, CompilerMemoryBytes | None]
    """Byte counts per split oracle core, for a kernel that keeps one.

    A ride-along NB-EGM kernel keeps its continuation and envelope halves as an
    oracle of the tile-local production core; the envelope half's argument bytes
    are the complete continuation stacks the production core never materializes.
    Empty for every other kernel.
    """


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

    result = _run_period_kernel(
        regime=regime,
        capture_target=None,
        compiled_cores=_compile_cores_for_one_period(
            regime=regime, period=period, kernel_kwargs=kernel_kwargs
        ),
        **kernel_kwargs,
    )
    return PeriodReplay(
        regime_name=kernel_kwargs["regime_name"],
        period=period,
        age=float(kernel_kwargs["ages"].values[period]),
        result=result,
    )


def analyze_period_core_memory(*, directory: Path) -> PeriodCoreMemoryAnalysis:
    """Compile, but never execute, one captured period's cores and read their memory.

    The production cores are lowered against the capture's logical pytrees and
    production shapes; a kernel keeping a split oracle has those cores lowered
    beside them. The capture round trip loses device sharding, so every executable
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

    production = _compile_cores_for_one_period(
        regime=regime, period=period, kernel_kwargs=kernel_kwargs
    )
    split_oracle = _compile_split_oracle_cores_for_one_period(
        regime=regime, period=period, kernel_kwargs=kernel_kwargs
    )
    return PeriodCoreMemoryAnalysis(
        regime_name=kernel_kwargs["regime_name"],
        period=period,
        age=float(kernel_kwargs["ages"].values[period]),
        preserves_production_sharding=False,
        core_memory_bytes=MappingProxyType(
            {
                key: _compiler_memory_bytes(compiled=core)
                for key, core in production.items()
            }
        ),
        split_oracle_memory_bytes=MappingProxyType(
            {
                key: _compiler_memory_bytes(compiled=core)
                for key, core in split_oracle.items()
            }
        ),
    )


def _load_capture_payload(*, directory: Path) -> dict[str, Any]:
    """Load one period's logical inputs; array sharding is not round-tripped."""
    with (directory / _PAYLOAD_NAME).open("rb") as stream:
        return cloudpickle.load(stream)


def _compiler_memory_bytes(*, compiled: Any) -> CompilerMemoryBytes | None:  # noqa: ANN401
    """Normalize a backend memory-analysis object to stable integer byte fields."""
    try:
        stats = compiled.memory_analysis()
    except Exception:  # noqa: BLE001 - analysis is optional across JAX backends
        return None
    if stats is None:
        return None

    def optional_bytes(name: str) -> int | None:
        value = getattr(stats, name, None)
        return None if value is None else int(value)

    return CompilerMemoryBytes(
        generated_code_size_in_bytes=optional_bytes("generated_code_size_in_bytes"),
        argument_size_in_bytes=optional_bytes("argument_size_in_bytes"),
        output_size_in_bytes=optional_bytes("output_size_in_bytes"),
        alias_size_in_bytes=optional_bytes("alias_size_in_bytes"),
        temp_size_in_bytes=optional_bytes("temp_size_in_bytes"),
        peak_memory_in_bytes=optional_bytes("peak_memory_in_bytes"),
        host_generated_code_size_in_bytes=optional_bytes(
            "host_generated_code_size_in_bytes"
        ),
        host_argument_size_in_bytes=optional_bytes("host_argument_size_in_bytes"),
        host_output_size_in_bytes=optional_bytes("host_output_size_in_bytes"),
        host_alias_size_in_bytes=optional_bytes("host_alias_size_in_bytes"),
        host_temp_size_in_bytes=optional_bytes("host_temp_size_in_bytes"),
    )


def _compile_cores_for_one_period(
    *,
    regime: Any,  # noqa: ANN401 - the canonical Regime, circular to import here
    period: int,
    kernel_kwargs: dict[str, Any],
) -> MappingProxyType[str, Any]:
    """Lower and compile the cores of a single period.

    The solve loop compiles every regime-period up front and deduplicates
    identical cores across them. A replay wants neither: it needs exactly the
    cores this one period calls.
    """
    period_kernel = regime.solution.period_kernels[period]
    context = _core_build_context_for_one_period(
        regime=regime, period=period, kernel_kwargs=kernel_kwargs
    )
    compiled = {}
    for core_name, declaration in core_program_graph(kernel=period_kernel).items():
        materialized = materialize_core_program(program=declaration, context=context)
        resolved = _resolve_program_for_execution(
            program=materialized,
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
        layout = (
            UNPLANNED
            if resolved.disposition is CoreExecutionDisposition.LEGACY_UNPLANNED
            else resolve_output_layout(
                core_key=core_name,
                value_template=context.next_regime_to_V_arr[
                    kernel_kwargs["regime_name"]
                ],
                state_order=state_order,
                output_roles=resolved.output_roles,
            )
        )
        jitted = (
            jax.jit(
                resolved.function,
                static_argnames=tuple(resolved.static_kwargs),
            )
            if layout is UNPLANNED
            else jax.jit(
                resolved.function,
                static_argnames=tuple(resolved.static_kwargs),
                out_shardings=layout.out_shardings,
            )
        )
        lowered = jitted.lower(**resolved.arguments, **resolved.static_kwargs)
        _assert_lowered_output_roles(
            lowered=lowered,
            output_roles=resolved.output_roles,
            value_template=context.next_regime_to_V_arr[kernel_kwargs["regime_name"]],
            layout=layout,
            label=(
                f"{kernel_kwargs['regime_name']} {core_name} (replay period {period})"
            ),
        )
        executable = lowered.compile()
        compiled[core_name] = _attach_resolved_output_layout(
            compiled=executable,
            layout=layout,
            input_transfer_plan=resolved.input_transfer_plan,
        )
    return MappingProxyType(compiled)


def _compile_split_oracle_cores_for_one_period(
    *,
    regime: Any,  # noqa: ANN401 - the canonical Regime, circular to import here
    period: int,
    kernel_kwargs: dict[str, Any],
) -> MappingProxyType[str, Any]:
    """Lower and compile a kernel's split oracle cores, if it keeps any.

    The oracle cores are not part of the production graph, so they are lowered
    through the kernel's own legacy argument builder rather than the resolver.
    """
    period_kernel = regime.solution.period_kernels[period]
    split_cores = getattr(period_kernel, "split_cores", None)
    if split_cores is None:
        return MappingProxyType({})
    context = _core_build_context_for_one_period(
        regime=regime, period=period, kernel_kwargs=kernel_kwargs
    )
    compiled = {}
    for core_name, core in split_cores().items():
        arguments = _LegacyArgumentBuilder(kernel=period_kernel, core_name=core_name)(
            context
        )
        compiled[core_name] = jax.jit(core).lower(**arguments).compile()
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
