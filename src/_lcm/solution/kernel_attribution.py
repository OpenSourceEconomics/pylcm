"""Name the regime-period behind every kernel the solve loop runs.

A compiled XLA module cannot be attributed to a regime-period from the compile
log alone. Identical cores are deduplicated before lowering, so one module can
serve many `(regime, period, core)` triples while the compile label names only
the representative one. When an allocation is refused, the first questions are
which age, which regime, and how many discrete action branches — and neither
the solve log nor an XLA dump answers them.

Two lines close the gap from opposite sides:

- `[attr] serves N triples` — emitted once per lowered core, saying whether a
  module name in a dump identifies a regime-period (`N == 1`) or does not.
- `[attr] <regime> age <a> period <p>` — emitted once per regime-period the
  backward-induction loop actually runs, carrying the discrete action
  cardinalities whose product is the branch width.

Both are gated on `LCM_LOG_KERNEL_ATTRIBUTION`, independently of the solve
`log_level`, and emit at a level that always clears the logger's threshold. A
solve run at `log_level="off"` — the setting that silences the per-period
diagnostic so a kernel's true peak is not masked — still gets its attribution.
"""

import logging
import os
from collections.abc import Mapping
from types import MappingProxyType

from _lcm.engine import StateActionSpace
from _lcm.typing import RegimeName
from lcm.ages import AgeGrid

_ENV_VAR = "LCM_LOG_KERNEL_ATTRIBUTION"


def attribution_enabled() -> bool:
    """Return whether the caller should pay for attribution logging."""
    return os.environ.get(_ENV_VAR, "0") not in {"0", ""}


def log_executed_kernel(
    *,
    regime_name: RegimeName,
    period: int,
    ages: AgeGrid,
    state_action_space: StateActionSpace,
    core_keys: tuple[str, ...],
    logger: logging.Logger,
) -> None:
    """Log the identity and branch width of one regime-period about to run.

    Emitted from the single funnel every regime-period passes through, so the
    set of lines is exactly the set of executed kernels — a regime-period that
    is inactive, or that the loop never reaches because an earlier one failed,
    produces no line. That is what separates "compiled" from "executed" when
    reading a dump against a log.
    """
    if not attribution_enabled():
        return

    cardinalities = _discrete_action_cardinalities(
        discrete_actions=state_action_space.discrete_actions
    )
    branches = 1
    for cardinality in cardinalities.values():
        branches *= cardinality

    logger.log(
        _level(logger=logger),
        "  [attr] %s age %s period %d: branches=%d actions=(%s) states=(%s) cores=(%s)",
        regime_name,
        ages.values[period].item(),
        period,
        branches,
        _render(cardinalities),
        _render(_state_cardinalities(states=state_action_space.states)),
        ", ".join(sorted(core_keys)),
    )


def log_module_fanout(
    *,
    label: str,
    n_triples: int,
    logger: logging.Logger,
) -> None:
    """Log how many regime-period-core triples share one lowered module.

    A count of 1 makes the module's name in an XLA dump a safe attribution.
    Anything higher means the compile label names one representative out of
    several, and the executed identity has to come from `log_executed_kernel`.
    """
    if not attribution_enabled():
        return
    logger.log(
        _level(logger=logger),
        "  [attr] serves %d triple%s: %s",
        n_triples,
        "" if n_triples == 1 else "s",
        label,
    )


def _level(*, logger: logging.Logger) -> int:
    """Return a level that clears the logger's threshold, `"off"` included."""
    return max(logger.getEffectiveLevel(), logging.INFO)


def _discrete_action_cardinalities(
    *, discrete_actions: Mapping[str, object] | MappingProxyType
) -> dict[str, int]:
    """Return `{action name: cardinality}` for the regime's discrete actions."""
    return {name: _cardinality(value) for name, value in discrete_actions.items()}


def _state_cardinalities(
    *, states: Mapping[str, object] | MappingProxyType
) -> dict[str, int]:
    """Return `{state name: cardinality}` for the regime's states."""
    return {name: _cardinality(value) for name, value in states.items()}


def _cardinality(value: object) -> int:
    """Return the leading-axis length of a grid array, or 1 for a scalar."""
    shape = getattr(value, "shape", ())
    return int(shape[0]) if shape else 1


def _render(cardinalities: Mapping[str, int]) -> str:
    """Render `{name: n}` as a stable, sorted `name=n` list."""
    return ", ".join(f"{name}={n}" for name, n in sorted(cardinalities.items()))
