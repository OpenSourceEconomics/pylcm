"""Static time alignment and declared temporal-consumer metadata."""

import inspect
from collections.abc import Sequence
from typing import cast

import jax.numpy as jnp
import numpy as np

from _lcm.dtypes import CanonicalArrayWriter, canonical_float_dtype
from _lcm.time import TimeAxis, coordinate_kind
from _lcm.utils.ast_inspection import time_index_names
from lcm.exceptions import InvalidParamsError, ModelInitializationError
from lcm.temporal import TimeVarying
from lcm.typing import UserAge, UserFunction, ValueND


def temporal_parameter_names(func: UserFunction | None) -> frozenset[str]:
    """Read declarations through normal wrappers and scheduled sources."""
    if func is None:
        return frozenset()
    own = frozenset(getattr(func, "__lcm_time_params__", ()))
    if own:
        return own
    sources = getattr(func, "__lcm_sources__", ())
    if sources:
        return frozenset().union(
            *(temporal_parameter_names(source) for source in sources)
        )
    unwrapped = getattr(func, "__wrapped__", func)
    return temporal_parameter_names(unwrapped) if unwrapped is not func else frozenset()


def validate_temporal_variants(
    *, functions: Sequence[UserFunction], name: str
) -> frozenset[str]:
    """Require one temporal meaning for every parameter shared by variants."""
    names = frozenset().union(*(temporal_parameter_names(func) for func in functions))
    for param in names:
        for func in functions:
            if param in temporal_parameter_names(func) and time_index_names(
                func=func, array_param_name=param
            ):
                raise ModelInitializationError(
                    f"{name}.{param}: a managed temporal parameter is already sliced "
                    "at the current period; remove its age/period indexing."
                )
            if param in inspect.signature(
                func
            ).parameters and param not in temporal_parameter_names(func):
                raise ModelInitializationError(
                    f"{name}.{param}: shared parameter mixes managed temporal "
                    "and unmanaged consumers. Use distinct parameter names."
                )
    return names


def _validate_time_labels(*, labels: tuple, kind: str, name: str) -> None:
    """Validate the schema before surplus observations can be discarded."""
    if kind == "period" and any(
        isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer))
        for v in labels
    ):
        raise InvalidParamsError(
            f"{name}: period coordinates must be integers, excluding booleans."
        )
    if kind == "age":
        try:
            finite = all(
                not isinstance(v, (bool, np.bool_, str)) and np.isfinite(float(v))
                for v in labels
            )
        except TypeError, ValueError:
            finite = False
        if not finite:
            raise InvalidParamsError(
                f"{name}: age coordinates must be finite numeric labels."
            )


def time_gather_indices(
    *,
    labels: tuple,
    kind: str,
    ages: TimeAxis,
    required_periods: tuple[int, ...],
    name: str,
    check_duplicates: bool = True,
) -> tuple[int, ...]:
    """Validate all labels, discard surplus rows, then check selected coverage."""
    if kind != coordinate_kind(ages):
        raise InvalidParamsError(
            f"{name}: {kind} coordinates supplied to a {coordinate_kind(ages)} model."
        )
    _validate_time_labels(labels=labels, kind=kind, name=name)
    wanted: dict[UserAge | float, int] = {
        label: period for period, label in enumerate(ages.exact_values)
    }
    # Fraction labels have an exact public identity; support the AgeGrid's float
    # representation too, as existing named-Series input does.
    if kind == "age":
        wanted.update(
            {float(label): period for period, label in enumerate(ages.exact_values)}
        )
    positions: dict[int, int] = {}
    for index, label in enumerate(labels):
        period = wanted.get(label)
        if period is None:
            continue
        if check_duplicates and period in positions:
            raise InvalidParamsError(
                f"{name}: duplicate selected {kind} key {label!r}."
            )
        positions[period] = index
    missing = [ages.exact_values[p] for p in required_periods if p not in positions]
    if missing:
        raise InvalidParamsError(
            f"{name}: missing required {kind} coordinates {missing[:8]}."
        )
    return tuple(positions.get(period, -1) for period in range(ages.n_periods))


def align_time_varying(
    *,
    value: TimeVarying,
    ages: TimeAxis,
    required_periods: tuple[int, ...],
    name: str,
    array_writer: CanonicalArrayWriter | None = None,
) -> ValueND:
    """Gather static coordinates while keeping numeric values differentiable."""
    labels = value.periods if value.periods is not None else cast("tuple", value.ages)
    indices = time_gather_indices(
        labels=labels,
        kind="period" if value.periods is not None else "age",
        ages=ages,
        required_periods=required_periods,
        name=name,
    )
    if array_writer is not None:
        host = np.zeros(
            (ages.n_periods, *value.values.shape[1:]), dtype=canonical_float_dtype()
        )
        selected = np.asarray(indices) >= 0
        host[selected] = np.asarray(value.values)[np.asarray(indices)[selected]]
        return array_writer(value=host, dtype=host.dtype, name=name)
    # Only structurally unread slots receive zeros. No required observation is
    # synthesized; coverage was checked before constructing a numeric array.
    values = jnp.asarray(value.values)
    if not values.shape[0]:
        return jnp.zeros((ages.n_periods, *values.shape[1:]), dtype=values.dtype)
    gather = tuple(i if i >= 0 else values.shape[0] for i in indices)
    return jnp.take(values, jnp.asarray(gather), axis=0, mode="fill", fill_value=0)
