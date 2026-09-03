"""Every shipped period kernel runs through the native core-program contract.

The solve loop executes a period kernel only through the immutable graph the kernel
publishes, lowers every program of that graph against a resolved output layout, and
publishes the period value after asserting its placement. These tests sweep the
built-in solvers so that no kernel can fall back to an engine-side adapter or repair.
"""

from collections.abc import Callable
from typing import Any

import pytest

from _lcm.execution.core_program import (
    CoreExecutionDisposition,
    CoreExecutionRequirements,
    CoreProgramGraphAware,
    _TargetValueAccess,
)
from _lcm.execution.output_layout import PlannedCore
from _lcm.execution.value_transfer import (
    ValueArtifactAddress,
    ValueArtifactKind,
    ValueConsumerAddress,
    ValueInputChannel,
)
from _lcm.solution import backward_induction
from _lcm.solution.backward_induction import (
    _classify_dispatch_value_artifacts,
    _ProgramExecutionMetadata,
)
from tests.solution.test_kernel_output import _SHIPPED_KERNELS

_SOLVES: dict[str, Callable[[], None]] = {
    case: solve for case, (solve, _regime, _channels) in _SHIPPED_KERNELS.items()
}


@pytest.mark.parametrize("case", list(_SOLVES))
def test_every_shipped_period_kernel_is_core_program_graph_aware(
    *, case: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each built-in kernel publishes its own graph of planned or dense programs."""
    kernels: list[object] = []
    original = backward_induction.core_program_graph

    def recording(*, kernel: object) -> Any:
        kernels.append(kernel)
        return original(kernel=kernel)

    monkeypatch.setattr(backward_induction, "core_program_graph", recording)
    _SOLVES[case]()

    assert kernels, "no period kernel was resolved; the sweep is inert"
    assert all(isinstance(kernel, CoreProgramGraphAware) for kernel in kernels)
    dispositions = {
        program.disposition
        for kernel in kernels
        for program in original(kernel=kernel).values()
    }
    assert dispositions <= {
        CoreExecutionDisposition.PLANNED,
        CoreExecutionDisposition.DENSE,
    }


@pytest.mark.parametrize("enable_jit", [True, False], ids=["aot", "eager"])
@pytest.mark.parametrize("case", list(_SOLVES))
def test_every_compiled_core_is_a_planned_core(
    *, case: str, enable_jit: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every compiled program, eager or ahead-of-time, carries its resolved layout."""
    attached: list[object] = []
    original_attach = backward_induction._attach_resolved_output_layout
    original_compile = backward_induction._compile_all_functions

    def recording(**kwargs: Any) -> Any:
        core = original_attach(**kwargs)
        attached.append(core)
        return core

    def with_jit_mode(**kwargs: Any) -> Any:
        kwargs.pop("enable_jit")
        return original_compile(**kwargs, enable_jit=enable_jit)

    monkeypatch.setattr(backward_induction, "_attach_resolved_output_layout", recording)
    monkeypatch.setattr(backward_induction, "_compile_all_functions", with_jit_mode)
    _SOLVES[case]()

    assert attached, "no program was compiled; the sweep is inert"
    assert all(isinstance(core, PlannedCore) for core in attached)


def _metadata(
    *, accesses: tuple[_TargetValueAccess, ...] = ()
) -> _ProgramExecutionMetadata:
    return _ProgramExecutionMetadata(
        requirements=CoreExecutionRequirements(target_value_accesses=accesses),
        disposition=CoreExecutionDisposition.DENSE,
        input_transfer_plan=(),
    )


def _access() -> _TargetValueAccess:
    return _TargetValueAccess(
        target=ValueArtifactAddress(
            kind=ValueArtifactKind.REGIME_VALUE, period=1, regime="target"
        ),
        source=ValueConsumerAddress(
            source_period=0,
            source_regime="source",
            core_key="main",
            channel=ValueInputChannel.NEXT_REGIME_VALUE,
            path=("target",),
        ),
    )


def test_dense_programs_without_declared_accesses_stay_conservatively_pinned() -> None:
    """A dense program that declares no value reads pins every reachable value."""
    planned, exact, has_unknown = _classify_dispatch_value_artifacts(
        programs={"main": _metadata()}
    )

    assert (planned, exact, has_unknown) == ((), (), True)


def test_dense_programs_with_declared_accesses_pin_exactly_those_values() -> None:
    """A dense program's declared value reads are pinned exactly, nothing else."""
    access = _access()

    planned, exact, has_unknown = _classify_dispatch_value_artifacts(
        programs={"main": _metadata(accesses=(access,))}
    )

    assert (planned, exact, has_unknown) == ((), (access.target,), False)
