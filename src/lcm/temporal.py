"""Explicit temporal parameter declarations, independent of the model clock."""

import inspect
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from functools import wraps
from typing import Any, cast, no_type_check

import jax
import jax.numpy as jnp
import numpy as np
from beartype import beartype

from _lcm.beartype_conf import PARAMS_CONF
from lcm.typing import ValueND

# A computational period label. A boolean passes here and is refused when the
# labels are validated.
type _PeriodLabel = int | np.integer
# A numeric age label. Finiteness is checked when the labels are validated.
type _AgeLabel = int | float | Fraction | np.integer | np.floating
# The static labels of a `TimeVarying`: its periods and its ages, one of them set.
type _Labels = tuple[tuple[_PeriodLabel, ...] | None, tuple[_AgeLabel, ...] | None]


@beartype(conf=PARAMS_CONF)
@dataclass(frozen=True, kw_only=True)
class TimeVarying:
    """Values with a leading time axis and exactly one set of static labels.

    `periods` are computational indices; `ages` are actual age labels.
    Surplus labels are silently discarded. Selected duplicates and missing
    coordinates a consumer can read are errors. Values remain JAX pytree leaves.
    """

    values: ValueND | np.ndarray
    """Numeric values whose leading dimension matches the supplied coordinates."""

    # Coordinates are validated deterministically in preflight, including rows
    # outside the model grid; the type check samples one entry of each tuple.
    periods: tuple[_PeriodLabel, ...] | None = None
    """Integer computational coordinates, mutually exclusive with ages."""

    ages: tuple[_AgeLabel, ...] | None = None
    """Finite numeric age coordinates, mutually exclusive with periods."""

    def __post_init__(self) -> None:
        if (self.periods is None) == (self.ages is None):
            raise ValueError(
                "TimeVarying requires exactly one of periods and ages coordinates."
            )
        labels = (
            self.periods
            if self.periods is not None
            else cast("tuple[_AgeLabel, ...]", self.ages)
        )
        if self.values.ndim == 0 or self.values.shape[0] != len(labels):
            raise ValueError(
                "TimeVarying leading axis length must match its coordinate labels."
            )

    @classmethod
    def from_profile(
        cls,
        *,
        values: ValueND | np.ndarray,
        labels: Sequence[Hashable],
        period_to_label: Mapping[int, Hashable],
    ) -> TimeVarying:
        """Gather an economic profile by an explicit period-to-label mapping.

        Several stages may share a source label. Unused source rows are ignored;
        missing mapped labels and duplicate selected labels are errors. The result
        is period-labelled, so it serves period models; label an age model's
        profile by age with `TimeVarying(values=..., ages=...)`.
        """
        periods = tuple(period_to_label)
        if any(
            isinstance(period, (bool, np.bool_))
            or not isinstance(period, (int, np.integer))
            for period in periods
        ):
            raise ValueError("Profile period coordinates must be integers.")
        if values.ndim == 0 or values.shape[0] != len(labels):
            raise ValueError("Profile leading axis length must match its labels.")
        required = set(period_to_label.values())
        positions: dict[Hashable, int] = {}
        for index, label in enumerate(labels):
            if label not in required:
                continue
            if label in positions:
                raise ValueError(f"Profile has duplicate selected label {label!r}.")
            positions[label] = index
        missing = required - positions.keys()
        if missing:
            raise ValueError(f"Profile is missing mapped labels {list(missing)[:8]}.")
        indices = jnp.asarray(
            [positions[label] for label in period_to_label.values()], dtype=jnp.int32
        )
        return cls(values=jnp.take(values, indices, axis=0), periods=periods)


class UnlabelledTimeParameterWarning(UserWarning):
    """A manually indexed array has no labels with which to check its mapping."""


def _flatten(value: TimeVarying) -> tuple[tuple[ValueND | np.ndarray], _Labels]:
    return (value.values,), (value.periods, value.ages)


# keyword-only-exempt: library-callback=jax.tree_util.register_pytree_node
def _unflatten(labels: _Labels, values: Sequence[ValueND | np.ndarray]) -> TimeVarying:
    return TimeVarying(values=values[0], periods=labels[0], ages=labels[1])


jax.tree_util.register_pytree_node(TimeVarying, _flatten, _unflatten)


@dataclass(frozen=True, kw_only=True)
class _TemporalDecorator:
    names: tuple[str, ...]

    def __call__(self, func: Callable[..., Any]) -> Callable[..., Any]:
        from lcm.typing import Period  # noqa: PLC0415

        original = inspect.signature(func)
        if not self.names or len(set(self.names)) != len(self.names):
            raise ValueError("time_varying_params requires distinct parameter names.")
        if set(self.names) - set(original.parameters) or set(self.names) & {
            "period",
            "age",
        }:
            raise ValueError(
                "time_varying_params must name actual parameter arguments, "
                "excluding age and period."
            )
        parameters = [
            p.replace(kind=inspect.Parameter.KEYWORD_ONLY)
            for p in original.parameters.values()
        ]
        if "period" not in original.parameters:
            parameters.append(
                inspect.Parameter(
                    "period", inspect.Parameter.KEYWORD_ONLY, annotation=Period
                )
            )
        names = self.names
        parameter_names = tuple(p.name for p in parameters)
        defaults = tuple(
            (p.name, p.default)
            for p in parameters
            if p.default is not inspect.Parameter.empty
        )
        passes_period = "period" in original.parameters

        # Outer instrumentation binds away callable-object metadata; keep a function.
        @no_type_check
        @wraps(func)
        def consume(*args: Any, **kwargs: Any) -> Any:
            arguments = dict(defaults)
            arguments.update(zip(parameter_names[: len(args)], args, strict=True))
            arguments.update(kwargs)
            period = arguments["period"]
            for name in names:
                value = arguments[name]
                arguments[name] = value if np.ndim(value) == 0 else value[period]
            if not passes_period:
                del arguments["period"]
            return func(**arguments)

        object.__setattr__(
            consume, "__signature__", original.replace(parameters=parameters)
        )
        if not passes_period:
            # dags reads `__annotations__`, not `__signature__`, so `period` must
            # carry the type other regime functions annotate it with.
            object.__setattr__(
                consume,
                "__annotations__",
                {**inspect.get_annotations(func), "period": Period},
            )
        object.__setattr__(consume, "__lcm_time_params__", names)
        return consume


def time_varying_params(
    *names: str,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Declare temporal parameter slots before DAG compilation.

    Supply ``TimeVarying`` or a Series with named time coordinates. The consumer
    receives the current time slice. A scalar is a constant at every time.
    States and other DAG nodes cannot be marked as temporal parameters.
    """
    return _TemporalDecorator(names=names)
