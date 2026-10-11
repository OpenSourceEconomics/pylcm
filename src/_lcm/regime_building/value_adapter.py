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
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import cast, no_type_check

from dags import get_annotations

from _lcm.constraints.ir import Condition
from _lcm.grids import DiscreteGrid
from _lcm.identity_transition import _IdentityTransition
from _lcm.regime_law import DecomposedTransition, RegimeLaws
from _lcm.typing import EconFunctionArg, FunctionName, RegimeName
from lcm.collective import CollectiveUtility, ValueDependentConstraint
from lcm.exceptions import RegimeInitializationError
from lcm.phased import Phased
from lcm.regime import ConstraintEntry, FunctionEntry, StateTransitionEntry
from lcm.regime import Regime as UserRegime
from lcm.transition import AgeSpecializedFunction, JointTransition, StochasticTransition
from lcm.typing import StateOrActionName, UserFunction, ValueND

_CODE_ANNOTATION = "DiscreteState"

# One entry of a regime's `functions`, `state_transitions` or `constraints`.
type AdaptableEntry = FunctionEntry | StateTransitionEntry | ConstraintEntry


def collect_value_grids(
    *,
    regimes: Mapping[RegimeName, UserRegime],
    laws: RegimeLaws,
    derived_categoricals: Mapping[FunctionName, DiscreteGrid],
) -> MappingProxyType[StateOrActionName, DiscreteGrid]:
    """Return the value grid of every variable a model declares by its values.

    A name declared with values in one place must be declared with the same
    values and tolerance wherever it appears, so one grid encodes a law's
    result whichever regime reads it next.

    Raises:
        RegimeInitializationError: If a name is a value grid in one declaration
            and differs in another, or a value grid meets a declaration that
            would read its codes.

    """
    declarations: dict[StateOrActionName, list[tuple[RegimeName, DiscreteGrid]]] = {}
    for regime_name, regime in regimes.items():
        for name, entry in {
            **dict(regime.states),
            **dict(regime.actions),
            **dict(regime.derived_categoricals),
        }.items():
            if isinstance(entry, DiscreteGrid):
                declarations.setdefault(name, []).append((regime_name, entry))
    for name, grid in derived_categoricals.items():
        declarations.setdefault(name, []).append(("model", grid))
    value_grids = {
        name: value_grid
        for name, grids in declarations.items()
        for value_grid in [next((g for _, g in grids if g.values is not None), None)]
        if value_grid is not None
    }
    for name, value_grid in value_grids.items():
        mismatched = [
            regime_name
            for regime_name, grid in declarations[name]
            if (grid.values, grid.tolerance)
            != (value_grid.values, value_grid.tolerance)
        ]
        if mismatched:
            raise RegimeInitializationError(
                f"'{name}' is a `DiscreteGrid(values=...)` with values "
                f"{value_grid.values} and tolerance {value_grid.tolerance}, but "
                f"{mismatched} declare it differently. Declare it with the same "
                "values and tolerance everywhere."
            )
    if value_grids:
        _fail_if_unsupported_reader(regimes=regimes, laws=laws, value_grids=value_grids)
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
    decoded = {
        name: value_grids[grid_name]
        for name in _read_names(func)
        if (grid_name := _value_grid_name(name=name, value_grids=value_grids))
        is not None
    }
    if not decoded and encode is None:
        return func
    return ValueAdapter(func=func, decoded=MappingProxyType(decoded), encode=encode)


def adapt_entry(
    *,
    entry: AdaptableEntry,
    value_grids: Mapping[StateOrActionName, DiscreteGrid],
    encode: DiscreteGrid | None,
) -> AdaptableEntry:
    """Adapt every user function inside one regime slot entry.

    Walks `Phased` sides, per-target mappings and the probability function of a
    `StochasticTransition`, which is decoded only: it returns probabilities over
    codes.
    """
    # Without value grids nothing changes. Conditions and the collective
    # declarations carry no user function over variables, and an identity law
    # maps codes to codes exactly as it maps values to values.
    if not value_grids or isinstance(
        entry,
        Condition
        | CollectiveUtility
        | ValueDependentConstraint
        | _IdentityTransition
        | AgeSpecializedFunction
        | None,
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
            "AdaptableEntry",
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
    return adapt_user_function(
        func=cast("UserFunction", entry), value_grids=value_grids, encode=encode
    )


@dataclass(frozen=True, kw_only=True, eq=False)
class ValueAdapter:
    """A user function the engine calls with the codes of value grids.

    A frozen callable instance rather than a nested function: the beartype claw
    memoizes every function it decorates, so a function defined per model build
    would pin that build's user function after the model is dropped. It sets no
    `__wrapped__`, so nothing that unwraps callables reaches the function that
    reads values.
    """

    func: UserFunction
    """The user function, which reads values."""
    decoded: MappingProxyType[str, DiscreteGrid]
    """The grid of each argument the function reads as a value."""
    encode: DiscreteGrid | None
    """The grid whose code the returned value names, if any."""

    def __post_init__(self) -> None:
        for name in ("__module__", "__name__", "__qualname__", "__doc__"):
            value = getattr(self.func, name, None)
            if value is not None:
                object.__setattr__(self, name, value)
        annotations = {
            name: _CODE_ANNOTATION if name in self.decoded else annotation
            for name, annotation in get_annotations(self.func).items()
        }
        if self.encode is not None:
            annotations["return"] = _CODE_ANNOTATION
        signature = inspect.signature(self.func)
        object.__setattr__(
            self,
            "__signature__",
            signature.replace(
                parameters=[
                    parameter.replace(annotation=annotations[parameter.name])
                    if parameter.name in annotations
                    else parameter
                    for parameter in signature.parameters.values()
                ],
                return_annotation=annotations.get(
                    "return", signature.return_annotation
                ),
            ),
        )
        # `concatenate_functions` reads `__annotations__`, not `__signature__`.
        object.__setattr__(self, "__annotations__", annotations)

    @no_type_check
    def __call__(self, **kwargs: EconFunctionArg) -> ValueND:
        result = self.func(
            **{
                name: self.decoded[name].decode(value)
                if name in self.decoded
                else value
                for name, value in kwargs.items()
            }
        )
        return result if self.encode is None else self.encode.encode(result)


def _read_names(func: UserFunction) -> tuple[str, ...]:
    """Return the names a user function reads."""
    return tuple(inspect.signature(func).parameters)


def _value_grid_name(
    *, name: str, value_grids: Mapping[StateOrActionName, DiscreteGrid]
) -> StateOrActionName | None:
    """Return the value grid an argument reads: `x` or the next state `next_x`."""
    if name in value_grids:
        return name
    if name.startswith("next_") and name.removeprefix("next_") in value_grids:
        return name.removeprefix("next_")
    return None


def _fail_if_unsupported_reader(
    *,
    regimes: Mapping[RegimeName, UserRegime],
    laws: RegimeLaws,
    value_grids: Mapping[StateOrActionName, DiscreteGrid],
) -> None:
    """Raise where a value grid meets a declaration that would read its codes."""
    problems: list[str] = []
    for regime_name, regime in regimes.items():
        declared = {
            name
            for name in (*regime.states, *regime.actions, *regime.derived_categoricals)
            if name in value_grids
        }
        problems.extend(
            f"the fixed state '{name}' of '{regime_name}'"
            for name, law in regime.state_transitions.items()
            if name in value_grids and _holds_identity(law)
        )
        if declared and regime.stakeholders is not None:
            problems.append(f"'{regime_name}' is a collective regime")
        problems.extend(
            f"the joint transition '{key}' to '{target}' of '{regime_name}'"
            for target, joints in regime.joint_transitions.items()
            for key, joint in joints.items()
            if _joint_names(joint) & _value_names(value_grids)
        )
        problems.extend(
            f"the condition constraint '{name}' of '{regime_name}'"
            for name, entry in regime.constraints.items()
            if isinstance(entry, Condition) and entry.dependencies & set(value_grids)
        )
        if declared:
            problems.extend(
                f"the age-specialized '{name}' of '{regime_name}'"
                for name, entry in (
                    *regime.decomposed_functions.items(),
                    *regime.state_transitions.items(),
                )
                if isinstance(entry, AgeSpecializedFunction)
            )
    problems.extend(
        f"the regime transition law of '{regime_name}'"
        for regime_name, law in laws.items()
        if set(_law_reads(law.decomposed_transition)) & _value_names(value_grids)
    )
    if problems:
        raise RegimeInitializationError(
            "Value grids are not supported where these declarations read them: "
            + "; ".join(problems)
            + ". Declare these variables with a category class."
        )


def _value_names(value_grids: Mapping[StateOrActionName, DiscreteGrid]) -> set[str]:
    """Return every argument name that reads a value grid: `x` and `next_x`."""
    return {*value_grids, *(f"next_{name}" for name in value_grids)}


def _joint_names(joint: JointTransition | Phased) -> set[str]:
    """Return the outputs and the arguments of a joint transition's callables."""
    sides = (joint.solve, joint.simulate) if isinstance(joint, Phased) else (joint,)
    return {
        name
        for side in sides
        if isinstance(side, JointTransition)
        for name in (
            *side.outputs,
            *(
                read
                for func in (side.probabilities, side.support, *side.outputs.values())
                if callable(func)
                for read in _read_names(cast("UserFunction", func))
            ),
        )
    }


def _law_reads(law: DecomposedTransition) -> Iterator[str]:
    """Yield the arguments of every callable node of a regime transition law."""
    if isinstance(law, Phased):
        yield from _law_reads(law.solve)
        yield from _law_reads(law.simulate)
    elif isinstance(law, Mapping):
        for cell in law.values():
            yield from _law_reads(cell)
    elif isinstance(law, StochasticTransition):
        yield from _read_names(law.func)
    elif inspect.isfunction(law):
        yield from _read_names(law)


def _holds_identity(law: StateTransitionEntry) -> bool:
    """Return whether a state's law is an identity law, in any phase or target."""
    if isinstance(law, Phased):
        return _holds_identity(law.solve) or _holds_identity(law.simulate)
    if isinstance(law, Mapping):
        return any(_holds_identity(cell) for cell in law.values())
    return isinstance(law, _IdentityTransition)
