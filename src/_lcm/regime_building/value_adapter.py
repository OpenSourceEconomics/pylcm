"""Hand user functions the numbers of value grids instead of their codes.

A `DiscreteGrid(values=...)` variable is a code `0..n-1` everywhere inside the
engine. `adapt_user_function` wraps a user function so that it receives the
values of the value-grid variables it reads, and, for a deterministic law or a
derived categorical of such a grid, so that the value it returns becomes the
code of the node it names. A value on no node becomes `MISSING_CAT_CODE`.

The wrapper publishes `DiscreteState` for every decoded argument and encoded
return, the annotation the engine's own nodes carry for codes, so the DAG sees
one annotation per node whichever function names it.
"""

import dataclasses
import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import cast, no_type_check

from dags import get_annotations, with_signature

from _lcm.constraints.ir import Condition
from _lcm.grids import DiscreteGrid
from _lcm.identity_transition import _IdentityTransition
from _lcm.typing import EconFunctionArg, FunctionName, RegimeName
from lcm.collective import CollectiveUtility, ValueDependentConstraint
from lcm.exceptions import RegimeInitializationError
from lcm.phased import Phased
from lcm.regime import ConstraintEntry, FunctionEntry, StateTransitionEntry
from lcm.regime import Regime as UserRegime
from lcm.transition import AgeSpecializedFunction, StochasticTransition
from lcm.typing import StateOrActionName, UserFunction, ValueND

_CODE_ANNOTATION = "DiscreteState"

# One entry of a regime's `functions`, `state_transitions` or `constraints`.
type SlotEntry = FunctionEntry | StateTransitionEntry | ConstraintEntry


def collect_value_grids(
    *,
    regimes: Mapping[RegimeName, UserRegime],
    derived_categoricals: Mapping[FunctionName, DiscreteGrid],
) -> MappingProxyType[StateOrActionName, DiscreteGrid]:
    """Return the value grid of every variable a model declares by its values.

    A name declared with values in one place must be declared with the same
    values and tolerance wherever it appears, so one grid encodes a law's
    result whichever regime reads it next.

    Raises:
        RegimeInitializationError: If a name is a value grid in one declaration
            and differs in another, or a value grid sits where values are not
            handed to user functions.

    """
    declarations: dict[StateOrActionName, list[tuple[RegimeName, DiscreteGrid]]] = {}
    for regime_name, regime in regimes.items():
        for name, entry in {
            **dict(regime.states),
            **dict(regime.actions),
            **dict(regime.derived_categoricals),
        }.items():
            if isinstance(entry, Phased):
                _fail_if_phased_value_grid(
                    regime_name=regime_name, name=name, entry=entry
                )
            elif isinstance(entry, DiscreteGrid):
                declarations.setdefault(name, []).append((regime_name, entry))
    for name, grid in derived_categoricals.items():
        declarations.setdefault(name, []).append(("model", grid))
    value_grids = {
        name: grids[0][1]
        for name, grids in declarations.items()
        if any(grid.values is not None for _, grid in grids)
    }
    for name, value_grid in value_grids.items():
        mismatched = [
            regime_name
            for regime_name, grid in declarations[name]
            if not _same_values(left=grid, right=value_grid)
        ]
        if mismatched:
            raise RegimeInitializationError(
                f"'{name}' is a `DiscreteGrid(values=...)` with values "
                f"{value_grid.values} and tolerance {value_grid.tolerance}, but "
                f"{mismatched} declare it differently. Declare it with the same "
                "values and tolerance everywhere."
            )
    if value_grids:
        for regime_name, regime in regimes.items():
            _fail_if_unsupported_reader(
                regime_name=regime_name, regime=regime, value_grids=value_grids
            )
    return MappingProxyType(value_grids)


def adapt_user_function(
    *,
    func: UserFunction,
    value_grids: Mapping[StateOrActionName, DiscreteGrid],
    encode: DiscreteGrid | None,
) -> UserFunction:
    """Wrap `func` to read values and, given `encode`, to return a code.

    Returns `func` itself when it reads no value-grid variable and `encode` is
    `None`.

    Args:
        func: The user function.
        value_grids: The value grid of every variable declared by its values.
        encode: The grid whose code the function's returned value names, for a
            deterministic law or a derived categorical of a value grid.

    Returns:
        The function the engine calls with codes.

    """
    parameters = inspect.signature(func).parameters
    decoded = {
        name: value_grids[grid_name]
        for name in parameters
        if (grid_name := _value_grid_name(name=name, value_grids=value_grids))
        is not None
    }
    if not decoded and encode is None:
        return func
    annotations = {
        name: _CODE_ANNOTATION if name in decoded else annotation
        for name, annotation in get_annotations(func).items()
        if name != "return"
    }
    return_annotation = (
        _CODE_ANNOTATION
        if encode is not None
        else get_annotations(func).get("return", "FloatND")
    )
    call: UserFunction = (
        DecodingAdapter(func=func, decoded=MappingProxyType(decoded))
        if decoded
        else func
    )
    if encode is not None:
        call = EncodingAdapter(func=call, grid=encode)
    adapted = with_signature(args=annotations, return_annotation=return_annotation)(
        call
    )
    # `concatenate_functions` reads `__annotations__`, not `__signature__`.
    adapted.__annotations__ = {**annotations, "return": return_annotation}
    return adapted


def adapt_entry(
    *,
    entry: SlotEntry,
    value_grids: Mapping[StateOrActionName, DiscreteGrid],
    encode: DiscreteGrid | None,
) -> SlotEntry:
    """Adapt every user function inside one regime slot entry.

    Walks `Phased` sides, per-target mappings, the probability function of a
    `StochasticTransition` (decoded only: it returns probabilities over codes),
    and the functions an `AgeSpecializedFunction` builds.
    """
    # A model without value grids, and the declarations whose callables are
    # not user functions over variables, pass through unchanged.
    if not value_grids or isinstance(
        entry, Condition | CollectiveUtility | ValueDependentConstraint | None
    ):
        return entry
    if isinstance(entry, Phased):
        return Phased(
            solve=adapt_entry(
                entry=entry.solve, value_grids=value_grids, encode=encode
            ),
            simulate=adapt_entry(
                entry=entry.simulate, value_grids=value_grids, encode=encode
            ),
        )
    if isinstance(entry, Mapping):
        return cast(
            "SlotEntry",
            MappingProxyType(
                {
                    key: adapt_entry(
                        entry=value, value_grids=value_grids, encode=encode
                    )
                    for key, value in entry.items()
                }
            ),
        )
    if isinstance(entry, StochasticTransition):
        return dataclasses.replace(
            entry,
            func=adapt_user_function(
                func=entry.func, value_grids=value_grids, encode=None
            ),
        )
    if isinstance(entry, AgeSpecializedFunction):
        return dataclasses.replace(
            entry,
            build=_AdaptedBuild(
                build=entry.build,
                value_grids=MappingProxyType(dict(value_grids)),
                encode=encode,
            ),
        )
    return adapt_user_function(
        func=cast("UserFunction", entry), value_grids=value_grids, encode=encode
    )


@dataclass(frozen=True, kw_only=True)
class DecodingAdapter:
    """A user function called with codes for the value grids it reads.

    A frozen callable instance rather than a nested function: the beartype claw
    memoizes every function it decorates, so a function defined per model build
    would pin that build's user function after the model is dropped.
    """

    func: UserFunction
    """The user function, which reads values."""
    decoded: MappingProxyType[str, DiscreteGrid]
    """The grid of each argument the function reads as a value."""

    def __post_init__(self) -> None:
        _copy_identity(source=self.func, target=self)

    @no_type_check
    def __call__(self, **kwargs: EconFunctionArg) -> ValueND:
        return self.func(
            **{
                name: self.decoded[name].decode(value)
                if name in self.decoded
                else value
                for name, value in kwargs.items()
            }
        )


@dataclass(frozen=True, kw_only=True)
class EncodingAdapter:
    """A function returning a value, published as the code of its node."""

    func: UserFunction
    """The function returning a value of `grid`."""
    grid: DiscreteGrid
    """The grid whose code the returned value names."""

    def __post_init__(self) -> None:
        _copy_identity(source=self.func, target=self)

    @no_type_check
    def __call__(self, **kwargs: EconFunctionArg) -> ValueND:
        return self.grid.encode(self.func(**kwargs))


def _copy_identity(
    *, source: UserFunction, target: DecodingAdapter | EncodingAdapter
) -> None:
    """Give a wrapper the name, module and docstring of the function it wraps."""
    for name in ("__module__", "__name__", "__qualname__", "__doc__"):
        value = getattr(source, name, None)
        if value is not None:
            object.__setattr__(target, name, value)


@dataclass(frozen=True, kw_only=True)
class _AdaptedBuild:
    """The builder of an `AgeSpecializedFunction`, adapting what it builds."""

    build: Callable[[float], UserFunction]
    """The user's builder, called with an age."""
    value_grids: MappingProxyType[StateOrActionName, DiscreteGrid]
    """The value grid of every variable declared by its values."""
    encode: DiscreteGrid | None
    """The grid whose code the built function's returned value names, if any."""

    @no_type_check
    def __call__(self, age: float) -> UserFunction:
        return adapt_user_function(
            func=self.build(age), value_grids=self.value_grids, encode=self.encode
        )


def _value_grid_name(
    *, name: str, value_grids: Mapping[StateOrActionName, DiscreteGrid]
) -> StateOrActionName | None:
    """Return the value grid an argument reads: `x` or the next state `next_x`."""
    if name in value_grids:
        return name
    if name.startswith("next_") and name.removeprefix("next_") in value_grids:
        return name.removeprefix("next_")
    return None


def _same_values(*, left: DiscreteGrid, right: DiscreteGrid) -> bool:
    """Return whether two grids declare the same values and tolerance."""
    return (left.values, left.tolerance) == (right.values, right.tolerance)


def _fail_if_phased_value_grid(
    *, regime_name: RegimeName, name: StateOrActionName, entry: Phased
) -> None:
    if any(
        isinstance(side, DiscreteGrid) and side.values is not None
        for side in (entry.solve, entry.simulate)
    ):
        raise RegimeInitializationError(
            f"In regime '{regime_name}', '{name}' declares a `Phased` value grid. "
            "Declare a value grid the same way in both phases."
        )


def _fail_if_unsupported_reader(
    *,
    regime_name: RegimeName,
    regime: UserRegime,
    value_grids: Mapping[StateOrActionName, DiscreteGrid],
) -> None:
    """Raise where a value grid meets a declaration that would read its codes."""
    declared = {
        name
        for name in (*regime.states, *regime.actions, *regime.derived_categoricals)
        if name in value_grids
    }
    problems: list[str] = []
    if declared and regime.stakeholders is not None:
        problems.append("a collective regime cannot declare value grids")
    if declared and regime.joint_transitions:
        problems.append("joint transitions cannot read value grids")
    problems.extend(
        f"the condition constraint '{name}' reads value grids"
        for name, entry in regime.constraints.items()
        if isinstance(entry, Condition) and entry.dependencies & set(value_grids)
    )
    problems.extend(
        f"'{name}' is a fixed state"
        for name, entry in regime.state_transitions.items()
        if name in value_grids and isinstance(entry, _IdentityTransition)
    )
    if problems:
        raise RegimeInitializationError(
            f"In regime '{regime_name}', value grids are not supported here: "
            + "; ".join(problems)
            + ". Declare these variables with a category class."
        )
