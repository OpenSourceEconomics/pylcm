"""Re-run a single captured regime-period without solving the ones above it.

`_lcm.solution.period_capture` writes the inputs; this module runs them back
through the same funnel the solve loop uses, so a replay repeats every step the
original call took rather than approximating it.

The split is what keeps the import graph acyclic: the capture side is imported
by the backward-induction loop, and the replay side imports that loop.
"""

import dataclasses
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any

import cloudpickle
import jax

from _lcm.solution.backward_induction import _edge_kwargs, _run_period_kernel
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


class FusedNBEGMExperimentScope(StrEnum):
    """Architectural question the compile-only fused replay can answer."""

    EXISTING_FULL_STACK_ONE_JIT_LIFETIME = "existing_full_stack_one_jit_lifetime"


class ReplayInputShapeProvenance(StrEnum):
    """Origin of the logical array shapes supplied to replay lowering."""

    CAPTURED_PRODUCTION_SHAPES = "captured_production_shapes"


class ReplayInputLayoutFidelity(StrEnum):
    """Placement fidelity of arrays restored from a period capture."""

    DEFAULT_BACKEND_AFTER_CAPTURE_ROUNDTRIP = "default_backend_after_capture_roundtrip"


@dataclasses.dataclass(frozen=True)
class FusedNBEGMMemoryAnalysis:
    """Compiler memory for fused and split cores at capture-roundtrip placement.

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

    experiment_scope: FusedNBEGMExperimentScope
    """The one narrow compiler-lifetime question represented by this report."""

    input_shape_provenance: ReplayInputShapeProvenance
    """Where the replay lowering's logical array shapes came from."""

    input_layout_fidelity: ReplayInputLayoutFidelity
    """How faithfully replay lowering represents the captured run's placement."""

    preserves_production_sharding: bool
    """Always false: period capture does not serialize production sharding."""

    fused_memory_bytes: CompilerMemoryBytes | None
    """Byte counts for one continuation-to-envelope executable, if supported."""

    split_memory_bytes: MappingProxyType[str, CompilerMemoryBytes | None]
    """Byte counts for split cores lowered from the same restored capture."""


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


def analyze_fused_nbegm_memory(*, directory: Path) -> FusedNBEGMMemoryAnalysis:
    """Compile, but never execute, a captured ride-along NB-EGM fused experiment.

    The current split cores and a diagnostic continuation-to-envelope wrapper are
    lowered against the capture's logical pytrees and production shapes. The capture
    round trip loses device sharding; both forms therefore use the restored arrays'
    default backend placement. Calling `memory_analysis()` on the resulting executables
    is safe even when their estimated runtime allocation exceeds the available device
    budget because this function deliberately has no execution path.

    The experiment asks only whether keeping the existing full continuation stacks
    inside one JIT changes their compiler-visible lifetime relative to the current split
    cores. It is not a tile-local implementation and does not establish a completed
    production memory architecture.

    Args:
        directory: A ride-along NB-EGM capture directory.

    Returns:
        The captured identity and compiler memory analyses for fused and split forms.

    """
    payload = _load_capture_payload(directory=directory)
    regime = payload["regime"]
    period = payload["period"]
    kernel_kwargs = payload["kernel_kwargs"]

    split_cores = _compile_cores_for_one_period(
        regime=regime, period=period, kernel_kwargs=kernel_kwargs
    )
    fused_core = _compile_fused_nbegm_core_for_one_period(
        regime=regime, period=period, kernel_kwargs=kernel_kwargs
    )
    return FusedNBEGMMemoryAnalysis(
        regime_name=kernel_kwargs["regime_name"],
        period=period,
        age=float(kernel_kwargs["ages"].values[period]),
        experiment_scope=(
            FusedNBEGMExperimentScope.EXISTING_FULL_STACK_ONE_JIT_LIFETIME
        ),
        input_shape_provenance=(ReplayInputShapeProvenance.CAPTURED_PRODUCTION_SHAPES),
        input_layout_fidelity=(
            ReplayInputLayoutFidelity.DEFAULT_BACKEND_AFTER_CAPTURE_ROUNDTRIP
        ),
        preserves_production_sharding=False,
        fused_memory_bytes=_compiler_memory_bytes(compiled=fused_core),
        split_memory_bytes=MappingProxyType(
            {
                key: _compiler_memory_bytes(compiled=core)
                for key, core in split_cores.items()
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
    lower_arg_sources = _lower_arg_sources_for_one_period(
        regime=regime, period=period, kernel_kwargs=kernel_kwargs
    )
    compiled = {}
    for core_key, core in period_kernel.cores().items():
        lower_args = period_kernel.build_lower_args(
            core_key=core_key, **lower_arg_sources
        )
        compiled[core_key] = jax.jit(core).lower(**lower_args).compile()
    return MappingProxyType(compiled)


def _compile_fused_nbegm_core_for_one_period(
    *,
    regime: Any,  # noqa: ANN401 - the canonical Regime, circular to import here
    period: int,
    kernel_kwargs: dict[str, Any],
) -> Any:  # noqa: ANN401 - a backend-specific JAX compiled executable
    """Lower one replay-only fused core against restored capture inputs."""
    period_kernel = regime.solution.period_kernels[period]
    builder = getattr(period_kernel, "build_fused_replay_core", None)
    if builder is None:
        msg = (
            "Fused memory analysis requires a ride-along NB-EGM period capture; "
            f"got {type(period_kernel).__name__}."
        )
        raise TypeError(msg)
    lower_arg_sources = _lower_arg_sources_for_one_period(
        regime=regime, period=period, kernel_kwargs=kernel_kwargs
    )
    lower_args = period_kernel.build_lower_args(
        core_key="continuation", **lower_arg_sources
    )
    return jax.jit(builder()).lower(**lower_args).compile()


def _lower_arg_sources_for_one_period(
    *,
    regime: Any,  # noqa: ANN401 - the canonical Regime, circular to import here
    period: int,
    kernel_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Build the solve-loop lowering inputs shared by split and fused replay."""
    # A source declaring gated edges reads its targets' folded continuation in
    # place of their raw V, so the kernel is compiled against a pytree the raw
    # mapping does not carry. The projection comes from the solve loop's own
    # builder rather than a second copy of it, which is what keeps the argument
    # tree and logical shapes a replay lowers against match production. Capture
    # serialization does not preserve sharding, so layout fidelity is explicitly
    # reported as default backend placement rather than implied here.
    return {
        "state_action_space": kernel_kwargs["state_action_space"],
        "next_regime_to_V_arr": kernel_kwargs["next_regime_to_V_arr"],
        "next_regime_to_continuation": kernel_kwargs["next_regime_to_continuation"],
        "flat_params": kernel_kwargs["flat_params"],
        "period": period,
        "ages": kernel_kwargs["ages"],
        **_edge_kwargs(
            regime=regime,
            regime_name=kernel_kwargs["regime_name"],
            next_edge_to_V_arr=kernel_kwargs["next_edge_to_V_arr"],
        ),
    }
