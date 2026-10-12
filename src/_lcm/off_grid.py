"""Raw continuous-state grid diagnostics, independent of clipping and verbosity."""

import logging

import jax
import jax.numpy as jnp

from _lcm.utils.logging import validation_enabled
from lcm.exceptions import OffGridStateError
from lcm.typing import (
    BoolND,
    Float1D,
    FloatND,
    OffGrid,
    RegimeName,
    ScalarInt,
    StateName,
)


@jax.jit
def count_off_grid(*, values: FloatND, points: Float1D, feasible: BoolND) -> ScalarInt:
    """Count strict exterior raw values on the economically reached population."""
    return jnp.count_nonzero(
        feasible & ((values < points[0]) | (values > points[-1]))
    ).astype(jnp.int32)


def report_off_grid(
    *,
    count: int,
    rule: OffGrid,
    regime: RegimeName,
    target: RegimeName,
    period: int,
    state: StateName,
    phase: str,
    logger: logging.Logger,
) -> None:
    """Enforce a declared raise policy and separately emit checked counts."""
    message = (
        f"Off-grid next states: regime={regime!r}, target={target!r}, "
        f"period={period}, state={state!r}, phase={phase!r}, count={count}"
    )
    if count and rule == "raise":
        raise OffGridStateError(message)
    if validation_enabled(logger):
        if count:
            logger.warning(message)
        else:
            logger.info(message)
