"""Lower `ConditionalAction` declarations to the pin constraints they mean.

A conditional action is active where every condition on the regime's discrete
actions holds and takes its inactive node elsewhere. That is the feasible set of
the constraint `conditions hold | action == inactive node`, which this module adds
to the regime's constraints once model-level slots are merged, so every discrete
action a condition names is known.
"""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass

import jax.numpy as jnp
from dags import get_annotations, with_signature
from dags.annotations import ensure_annotations_are_strings

from _lcm.grids.conditional_action import ConditionalAction
from _lcm.grids.discrete import DiscreteGrid
from _lcm.typing import ConstraintFunction
from lcm.collective import CollectiveUtility, ValueDependentConstraint
from lcm.phased import Phased
from lcm.regime import (
    ActionEntry,
    ConstraintEntry,
    FunctionEntry,
    StateTransitionEntry,
)
from lcm.transition import AgeSpecializedFunction
from lcm.typing import (
    ActionName,
    BoolND,
    FloatND,
    FunctionName,
    IntND,
    ReferenceName,
    RegimeName,
    UserFunction,
)

# A regime entry whose callables may read a conditional action's arguments.
type _Reader = FunctionEntry | ConstraintEntry | StateTransitionEntry


def add_conditional_action_pins(
    *,
    regime_name: RegimeName,
    actions: Mapping[ActionName, ActionEntry],
    constraints: Mapping[FunctionName, ConstraintEntry],
    readers: tuple[_Reader, ...],
) -> tuple[dict[FunctionName, ConstraintEntry], list[str]]:
    """Add one pin constraint per conditional action of a merged regime.

    Args:
        regime_name: Name of the regime, for error messages.
        actions: Mapping of the regime's merged action names to their grids.
        constraints: Mapping of the regime's merged constraint names to
            constraints.
        readers: The regime's merged functions, constraints and state
            transitions, read for the annotations the pins copy.

    Returns:
        Tuple of the constraints with the pins added and the error messages for
        conditions the regime cannot evaluate.

    """
    discrete = {
        name: grid for name, grid in actions.items() if isinstance(grid, DiscreteGrid)
    }
    pinned: dict[FunctionName, ConstraintEntry] = dict(constraints)
    errors: list[str] = []
    for action_name, grid in actions.items():
        if not isinstance(grid, ConditionalAction):
            continue
        condition_errors = _condition_errors(
            regime_name=regime_name,
            action_name=action_name,
            action=grid,
            discrete=discrete,
        )
        pin_name = f"conditional_action_{action_name}"
        if pin_name in pinned:
            condition_errors.append(
                f"Regime {regime_name!r}: the constraint name {pin_name!r} is "
                f"reserved for the conditional action {action_name!r}."
            )
        errors.extend(condition_errors)
        if not condition_errors:
            pinned[pin_name] = _create_pin(
                action_name=action_name,
                action=grid,
                readers=readers,
            )
    return pinned, errors


@dataclass(frozen=True, kw_only=True)
class _ConditionalActionPin:
    """The mask `conditions hold | action at its inactive node`, read by name.

    Every field is a plain number, string or tuple, so the model's durable
    fingerprint covers the mask completely. A shipped callable dataclass is how the
    fingerprint admits engine-built callables; a closure defined in this package is
    wrapped by beartype, which the fingerprint refuses.
    """

    action_name: ActionName
    """Name of the conditional action."""

    active_when: tuple[tuple[ActionName, tuple[int | float, ...]], ...]
    """Pairs of a discrete action name and the values that activate the action."""

    lower_midpoint: float
    """Midpoint between the inactive node and the node below it, or -inf."""

    upper_midpoint: float
    """Midpoint between the inactive node and the node above it, or +inf."""

    def __post_init__(self) -> None:
        # `with_signature` names its wrapper after the wrapped callable.
        object.__setattr__(self, "__name__", f"conditional_action_{self.action_name}")
        object.__setattr__(
            self, "__qualname__", f"conditional_action_{self.action_name}"
        )

    def __call__(self, **actions: FloatND | IntND) -> BoolND:
        active = jnp.asarray(True)  # noqa: FBT003
        for name, values in self.active_when:
            matches = jnp.asarray(False)  # noqa: FBT003
            for listed in values:
                matches = matches | (actions[name] == listed)
            active = active & matches
        # The action takes grid nodes only, so lying strictly between the midpoints
        # around the inactive node identifies that node at every precision, also when
        # the grid was declared at another precision than the solve runs at.
        action = actions[self.action_name]
        at_inactive_node = (action > self.lower_midpoint) & (
            action < self.upper_midpoint
        )
        return active | at_inactive_node


def _create_pin(
    *,
    action_name: ActionName,
    action: ConditionalAction,
    readers: tuple[_Reader, ...],
) -> ConstraintFunction:
    arg_names = (*action.active_when, action_name)
    lower_midpoint, upper_midpoint = action.inactive_midpoints
    return with_signature(  # ty: ignore[invalid-return-type]
        args={
            name: _find_annotation(
                readers=readers,
                arg_name=name,
                default="ContinuousAction" if name == action_name else "DiscreteAction",
            )
            for name in arg_names
        },
        return_annotation="BoolND",
        enforce=False,
    )(
        _ConditionalActionPin(
            action_name=action_name,
            active_when=tuple(
                (name, tuple(sorted(values)))
                for name, values in action.active_when.items()
            ),
            lower_midpoint=lower_midpoint,
            upper_midpoint=upper_midpoint,
        )
    )


def _condition_errors(
    *,
    regime_name: RegimeName,
    action_name: ActionName,
    action: ConditionalAction,
    discrete: Mapping[ActionName, DiscreteGrid],
) -> list[str]:
    errors = []
    for name, values in action.active_when.items():
        if name not in discrete:
            errors.append(
                f"Regime {regime_name!r}: the conditional action {action_name!r} "
                f"depends on {name!r}, which is not a discrete action of the "
                f"regime. Its discrete actions are {sorted(discrete)}."
            )
            continue
        taken = _values_taken(grid=discrete[name])
        missing = sorted(values - taken)
        if missing:
            errors.append(
                f"Regime {regime_name!r}: the conditional action {action_name!r} "
                f"lists {missing} for {name!r}, which does not take them. It takes "
                f"{sorted(taken)}."
            )
    return errors


def _values_taken(*, grid: DiscreteGrid) -> frozenset[int | float]:
    """Return the values user functions see for a discrete action."""
    return frozenset(grid.values if grid.values is not None else grid.codes)


def _find_annotation(
    *,
    readers: tuple[_Reader, ...],
    arg_name: ReferenceName,
    default: str,
) -> str:
    """Return the annotation the regime's callables use for one argument.

    Every reader of a variable must agree on its annotation, so the pin copies it
    from the first callable that declares the argument. A phased entry is read
    through both of its variants and a per-target mapping through its values. An
    age-specialized function builds its callable per age, so it is not read.
    """
    for func in _callables(readers):
        annotations = ensure_annotations_are_strings(get_annotations(func))
        if arg_name in annotations:
            return annotations[arg_name]
    return default


def _callables(readers: tuple[_Reader, ...]) -> Iterator[UserFunction]:
    for reader in readers:
        if isinstance(reader, Phased):
            yield from _callables((reader.solve, reader.simulate))
        elif isinstance(reader, Mapping):
            yield from _callables(tuple(reader.values()))
        elif isinstance(reader, CollectiveUtility):
            yield from _callables(tuple(reader.utilities.values()))
        elif isinstance(reader, ValueDependentConstraint):
            yield reader.predicate
        elif callable(reader) and not isinstance(reader, AgeSpecializedFunction):
            yield reader
