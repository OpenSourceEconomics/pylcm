"""Re-run a single captured regime-period without solving the ones above it.

`_lcm.solution.period_capture` writes the inputs; this module runs them back
through the same funnel the solve loop uses, so a replay repeats every step the
original call took rather than approximating it.

The split is what keeps the import graph acyclic: the capture side is imported
by the backward-induction loop, and the replay side imports that loop.
"""

import dataclasses
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
    with (directory / _PAYLOAD_NAME).open("rb") as stream:
        payload = cloudpickle.load(stream)

    regime = payload["regime"]
    period = payload["period"]
    kernel_kwargs = payload["kernel_kwargs"]

    result = _run_period_kernel(
        regime=regime,
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
    # A source declaring gated edges reads its targets' folded continuation in
    # place of their raw V, so the kernel is compiled against a pytree the raw
    # mapping does not carry. The projection comes from the solve loop's own
    # builder rather than a second copy of it, which is what keeps the argument
    # tree a replay lowers against identical to the one production lowered.
    lower_arg_sources = {
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
    compiled = {}
    for core_key, core in period_kernel.cores().items():
        lower_args = period_kernel.build_lower_args(
            core_key=core_key, **lower_arg_sources
        )
        compiled[core_key] = jax.jit(core).lower(**lower_args).compile()
    return MappingProxyType(compiled)
