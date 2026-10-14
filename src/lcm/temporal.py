"""Explicit temporal parameter declarations, independent of the model clock."""

import inspect
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass
from functools import wraps
from typing import TYPE_CHECKING, SupportsFloat, cast, no_type_check

import jax
import jax.numpy as jnp
import numpy as np
from beartype import beartype

from _lcm.beartype_conf import PARAMS_CONF
from lcm.exceptions import InvalidParamsError
from lcm.typing import AgeLabel, PeriodLabel, ValueND

if TYPE_CHECKING:
    from _lcm.typing import EconFunctionArg
else:
    # The engine's typing module imports `lcm.typing`, which imports this module
    # before the engine's aliases exist; the wrapper's arguments are not checked.
    type EconFunctionArg = object

# The static labels of a `TimeVarying`: its periods and its ages, one of them set.
type _Labels = tuple[tuple[PeriodLabel, ...] | None, tuple[AgeLabel, ...] | None]


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
    periods: tuple[PeriodLabel, ...] | None = None
    """Integer computational coordinates, mutually exclusive with ages."""

    ages: tuple[AgeLabel, ...] | None = None
    """Finite numeric age coordinates, mutually exclusive with periods."""

    def __post_init__(self) -> None:
        if (self.periods is None) == (self.ages is None):
            raise ValueError(
                "TimeVarying requires exactly one of periods and ages coordinates."
            )
        labels = (
            self.periods
            if self.periods is not None
            else cast("tuple[AgeLabel, ...]", self.ages)
        )
        if self.values.ndim == 0 or self.values.shape[0] != len(labels):
            raise ValueError(
                "TimeVarying leading axis length must match its coordinate labels."
            )

    @classmethod
    def from_profile[P: PeriodLabel, A: AgeLabel](
        cls,
        *,
        values: ValueND | np.ndarray,
        labels: Sequence[Hashable],
        period_to_label: Mapping[P, Hashable] | None = None,
        age_to_label: Mapping[A, Hashable] | None = None,
    ) -> TimeVarying:
        """Gather an economic profile by an explicit time-to-label mapping.

        Give exactly one mapping:

        - `period_to_label` returns a period-labelled table, which serves period
          models;
        - `age_to_label` returns an age-labelled table, which serves age models.
          Age keys must be finite numbers.

        Several periods or ages may share a source label, so a table read at the
        next age can map the last age to its own row. Unused source rows are
        ignored. A mapped label missing from `labels` is an error, and so is a
        selected label that appears more than once in `labels`.
        """
        if period_to_label is not None and age_to_label is None:
            kind, mapping = "period", period_to_label
        elif age_to_label is not None and period_to_label is None:
            kind, mapping = "age", age_to_label
        else:
            raise ValueError(
                "Profile requires exactly one of period_to_label and age_to_label."
            )
        keys = tuple(mapping)
        _validate_time_labels(labels=keys, kind=kind, name="Profile")
        if values.ndim == 0 or values.shape[0] != len(labels):
            raise ValueError("Profile leading axis length must match its labels.")
        required = set(mapping.values())
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
        indices = np.asarray([positions[label] for label in mapping.values()])
        # A NumPy profile keeps its dtype, so model validation sees the exact values.
        gathered = (
            np.take(values, indices, axis=0)
            if isinstance(values, np.ndarray)
            else jnp.take(values, indices, axis=0)
        )
        if period_to_label is not None:
            return cls(values=gathered, periods=tuple(period_to_label))
        return cls(values=gathered, ages=keys)


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

    def __call__[R](self, func: Callable[..., R]) -> Callable[..., R]:
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
        def consume(*args: EconFunctionArg, **kwargs: EconFunctionArg) -> R:
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


def time_varying_params(*names: str) -> _TemporalDecorator:
    """Declare temporal parameter slots before DAG compilation.

    Supply ``TimeVarying`` or a Series with named time coordinates. The consumer
    receives the current time slice. A scalar is a constant at every time.
    States and other DAG nodes cannot be marked as temporal parameters.
    """
    return _TemporalDecorator(names=names)


def _validate_time_labels(
    *, labels: tuple[Hashable, ...], kind: str, name: str
) -> None:
    """Validate the schema before surplus observations can be discarded."""
    if kind == "period" and any(
        isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer))
        for v in labels
    ):
        raise InvalidParamsError(
            f"{name}: period coordinates must be integers, excluding booleans."
        )
    if kind == "age":
        # Any label may arrive here; one that `float` rejects is not numeric.
        try:
            finite = all(
                not isinstance(v, (bool, np.bool_, str))
                and np.isfinite(float(cast("SupportsFloat", v)))
                for v in labels
            )
        except TypeError, ValueError:
            finite = False
        if not finite:
            raise InvalidParamsError(
                f"{name}: age coordinates must be finite numeric labels."
            )
