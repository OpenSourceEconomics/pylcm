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

**Sharding is not carried across the capture.** A sharded input is written as
whatever the pickle round-trip yields, so a replay on a multi-device host places
arrays by the default rules rather than by the solve's own sharding. Per-device
allocation figures taken from a replay are therefore not comparable with a
sharded run's unless the replay is confirmed to place them the same way.
"""

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from _lcm.execution.output_layout import PlannedCore
from _lcm.persistence.io import _save_pkl
from _lcm.typing import RegimeName

type PeriodCaptureTarget = tuple[RegimeName, int]

_TARGET_ENV = "LCM_CAPTURE_PERIOD"
_DIR_ENV = "LCM_CAPTURE_DIR"
_PAYLOAD_NAME = "kernel_inputs.pkl"


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
    replay never runs the workspace planner again.
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
        },
    )
