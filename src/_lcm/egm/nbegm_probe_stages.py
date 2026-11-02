"""Original NB host-probe numerical stages, preserving primitive evaluation order."""

from collections.abc import Callable, Mapping

import jax.numpy as jnp
from jax.typing import DTypeLike

from _lcm.egm.nbegm_breakpoints import clamp_breakpoints_to_grid
from _lcm.materialization import DatedGenerationMaterializer
from _lcm.typing import PytreeValue
from lcm.typing import (
    FloatND,
    ReferenceName,
    ScalarBool,
    ScalarFloat,
    ScalarInt,
    ValueND,
)


def evaluate_probe_stage[T: PytreeValue](
    *,
    function: Callable[..., T],
    arguments: Mapping[ReferenceName, PytreeValue],
    materializer: DatedGenerationMaterializer | None,
) -> T:
    """Evaluate the same original stage directly or through its existing owner."""
    if materializer is None:
        return function(**arguments)
    return materializer.evaluate_validation(function=function, arguments=arguments)


def probe_scalar(*, value: ValueND | float, dtype: DTypeLike) -> ScalarFloat:
    """Apply the original scalar cast and reshape, in that order."""
    return jnp.asarray(value, dtype=dtype).reshape(())


def probe_grid_scalar(*, grid: FloatND, index: int, dtype: DTypeLike) -> ScalarFloat:
    """Read the original grid endpoint before its scalar cast and reshape."""
    return jnp.asarray(grid[index], dtype=dtype).reshape(())


def probe_raw_breakpoints(*, values: tuple[FloatND, ...], dtype: DTypeLike) -> FloatND:
    """Stack original resolved thresholds or construct the original empty vector."""
    return jnp.stack(values) if values else jnp.zeros((0,), dtype=dtype)


def probe_clamped_breakpoints(
    *, raw: FloatND, liquid_grid: FloatND, dtype: DTypeLike
) -> FloatND:
    """Apply the original input casts before its original breakpoint clamp."""
    return clamp_breakpoints_to_grid(
        breakpoints=jnp.asarray(raw, dtype=dtype),
        liquid_grid=jnp.asarray(liquid_grid, dtype=dtype),
    )


def probe_sorted(*, values: FloatND) -> FloatND:
    """Keep the original sort before host iteration of resolved thresholds."""
    return jnp.sort(a=values)


def probe_law_samples(*, values: tuple[float, ...]) -> FloatND:
    """Keep the original law probe's default-dtype sample vector."""
    return jnp.asarray(values)


def probe_int_fill(*, value: int) -> ScalarInt:
    """Keep the original int32 scalar fill for an override."""
    return jnp.asarray(value, dtype=jnp.int32)


def probe_unstack(*, values: FloatND) -> tuple[FloatND, ...]:
    """Keep the original JAX-array iteration's numerical unstack stage."""
    return tuple(values)


def probe_between(
    *, value: ScalarFloat, lower: ScalarFloat, upper: ScalarFloat
) -> ScalarBool:
    """Keep the original two comparisons and conjunction before a host branch."""
    return (value > lower) & (value < upper)


def probe_greater(*, upper: ScalarFloat, lower: ScalarFloat) -> ScalarBool:
    """Keep the original positive-width comparison before a host branch."""
    return upper > lower


def probe_distinct(*, value: ScalarFloat, previous: ScalarFloat) -> ScalarBool:
    """Keep the original same-location comparison before a host branch."""
    return value != previous


def probe_midpoint(*, lower: ScalarFloat, upper: ScalarFloat) -> ScalarFloat:
    """Keep the original separately multiplied midpoint terms and addition."""
    return 0.5 * lower + 0.5 * upper


def probe_nextafter(*, lower: ScalarFloat, upper: ScalarFloat) -> ScalarFloat:
    """Keep the original directed one-step strict-interior fallback."""
    return jnp.nextafter(lower, upper)


def probe_selected_threshold(
    *, threshold: ScalarFloat, equality_above: bool, dtype: DTypeLike
) -> ScalarFloat:
    """Keep the original exact-side threshold selection and nextafter operands."""
    return jnp.where(
        equality_above,
        threshold,
        jnp.nextafter(threshold, jnp.asarray(jnp.inf, dtype=dtype)),
    )


def probe_points(
    *,
    liquid_samples: FloatND,
    interval_samples: tuple[FloatND, ...],
    with_grid_samples: bool,
    dtype: DTypeLike,
) -> FloatND:
    """Stack the same broad and strict-interior samples, without changing order."""
    parts = []
    if with_grid_samples:
        parts.append(jnp.asarray(liquid_samples, dtype=dtype).reshape(-1))
    if interval_samples:
        parts.append(
            jnp.stack(
                [jnp.asarray(s, dtype=dtype).reshape(()) for s in interval_samples]
            )
        )
    if not parts:
        return jnp.zeros((0,), dtype=dtype)
    return jnp.concatenate(parts)
