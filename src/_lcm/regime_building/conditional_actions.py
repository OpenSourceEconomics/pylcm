"""Lower `ConditionalAction` declarations to the pin constraints they mean.

A conditional action is active where every condition on the regime's discrete
actions holds and takes its inactive node elsewhere. That is the feasible set of
the constraint `conditions hold | action == inactive node`, which this module adds
to the regime's constraints once model-level slots are merged, so every discrete
action a condition names is known.
"""

import inspect
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType

import jax.numpy as jnp
from dags import get_annotations
from dags.annotations import ensure_annotations_are_strings

from _lcm.grids.conditional_action import ConditionalAction
from _lcm.grids.discrete import DiscreteGrid
from _lcm.regime_building.age_specialization import resolve_node
from _lcm.time import TimeAxis, specialization_coordinate_at
from lcm.collective import CollectiveUtility, ValueDependentConstraint
from lcm.phased import Phased
from lcm.regime import (
    ActionEntry,
    ConstraintEntry,
    FunctionEntry,
    StateTransitionEntry,
)
from lcm.regime import Regime as UserRegime
from lcm.transition import AgeSpecializedFunction
from lcm.typing import (
    ActionName,
    BoolND,
    Float1D,
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
        # Model build turns the grid into the action's candidate values at the
        # current precision; the pin reads its node from that same generation.
        nodes = grid.to_jax()
        condition_errors.extend(
            _repeated_node_errors(
                regime_name=regime_name,
                action_name=action_name,
                action=grid,
                nodes=nodes,
            )
        )
        errors.extend(condition_errors)
        if not condition_errors:
            pinned[pin_name] = _create_pin(
                action_name=action_name,
                action=grid,
                nodes=nodes,
                readers=readers,
            )
    return pinned, errors


@dataclass(frozen=True, kw_only=True)
class _ConditionalActionPin:
    """The mask `conditions hold | action == inactive node`, read by name.

    Every field is a plain number, string or tuple, so the model's durable
    fingerprint covers the mask completely. A shipped callable dataclass is how the
    fingerprint admits engine-built callables; a closure defined in this package is
    wrapped by beartype, which the fingerprint refuses.
    """

    action_name: ActionName
    """Name of the conditional action."""

    active_when: tuple[tuple[ActionName, tuple[int | float, ...]], ...]
    """Pairs of a discrete action name and the values that activate the action."""

    inactive_node: float
    """The inactive node as the grid generates it at the model's build precision."""

    node_dtype: str
    """Name of the float dtype the grid generated its nodes in at model build."""

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
        action = actions[self.action_name]
        _fail_if_precision_differs(
            action_name=self.action_name, action=action, node_dtype=self.node_dtype
        )
        # The action takes the grid's nodes, generated by the same call at the same
        # precision as `inactive_node`, and the nodes are strictly ascending. Equality
        # therefore holds at the inactive node and at no other.
        return active | (action == self.inactive_node)


@dataclass(frozen=True, kw_only=True)
class _SignedPin:
    """The pin under a keyword-only signature with the regime's annotations."""

    pin: _ConditionalActionPin
    """The mask."""

    annotations: tuple[tuple[ActionName, str], ...]
    """The pin's arguments with their annotations, in signature order."""

    def __post_init__(self) -> None:
        signature = inspect.Signature(
            [
                inspect.Parameter(
                    name, inspect.Parameter.KEYWORD_ONLY, annotation=annotation
                )
                for name, annotation in self.annotations
            ],
            return_annotation="BoolND",
        )
        object.__setattr__(self, "__signature__", signature)
        object.__setattr__(
            self, "__annotations__", {**dict(self.annotations), "return": "BoolND"}
        )
        name = f"conditional_action_{self.pin.action_name}"
        object.__setattr__(self, "__name__", name)
        object.__setattr__(self, "__qualname__", name)

    def __call__(self, **actions: FloatND | IntND) -> BoolND:
        return self.pin(**actions)


def _create_pin(
    *,
    action_name: ActionName,
    action: ConditionalAction,
    nodes: Float1D,
    readers: tuple[_Reader, ...],
) -> _SignedPin:
    """Return the pin, signed with the annotations the regime's readers give.

    A reader built per age or period is not a callable yet, so its arguments keep
    the default annotations here; `bind_specialized_annotations` signs them once
    the model's time axis is known.
    """
    pin = _ConditionalActionPin(
        action_name=action_name,
        active_when=tuple(
            (name, tuple(sorted(values))) for name, values in action.active_when.items()
        ),
        inactive_node=float(nodes[action.inactive_index]),
        node_dtype=str(nodes.dtype),
    )
    return _sign(pin=pin, readers=readers)


def bind_specialized_annotations(
    *,
    user_regimes: Mapping[RegimeName, UserRegime],
    ages: TimeAxis | None,
    active_periods_by_regime: Mapping[RegimeName, tuple[int, ...]] | None,
) -> Mapping[RegimeName, UserRegime]:
    """Re-sign conditional-action pins whose readers are built per age or period.

    Every concrete callable such a reader builds has the same call signature at
    every age or period, so the reader built at the regime's first active period
    gives the annotations the pin must carry for the DAG's annotation check.

    Args:
        user_regimes: Mapping of regime names to merged regimes.
        ages: The model's time axis; `None` leaves the regimes unchanged.
        active_periods_by_regime: The periods each regime is active in; `None`
            leaves the regimes unchanged.

    Returns:
        The regimes, with re-signed pins where a reader is built per age or period.

    """
    if ages is None or active_periods_by_regime is None:
        return user_regimes
    bound: dict[RegimeName, UserRegime] = {}
    for name, regime in user_regimes.items():
        pins = {
            pin_name: constraint
            for pin_name, constraint in regime.constraints.items()
            if isinstance(constraint, _SignedPin)
        }
        readers = (
            *regime.functions.values(),
            *regime.constraints.values(),
            *regime.state_transitions.values(),
        )
        periods = active_periods_by_regime.get(name, ())
        if not pins or not periods or not any(_specialized(readers)):
            bound[name] = regime
            continue
        coordinate = specialization_coordinate_at(ages=ages, period=periods[0])
        resolved = tuple(_resolved(readers=readers, coordinate=coordinate))
        bound[name] = regime.replace(
            constraints={
                **regime.constraints,
                **{
                    pin_name: _sign(pin=pin.pin, readers=resolved)
                    for pin_name, pin in pins.items()
                },
            }
        )
    return MappingProxyType(bound)


def _sign(*, pin: _ConditionalActionPin, readers: tuple[_Reader, ...]) -> _SignedPin:
    arg_names = (*(name for name, _ in pin.active_when), pin.action_name)
    return _SignedPin(
        pin=pin,
        annotations=tuple(
            (
                name,
                _find_annotation(callables=_callables(readers), arg_name=name)
                or (
                    "ContinuousAction" if name == pin.action_name else "DiscreteAction"
                ),
            )
            for name in arg_names
        ),
    )


def _resolved(*, readers: tuple[_Reader, ...], coordinate: float) -> Iterator[_Reader]:
    """Yield `readers` with every callable built per age or period built at
    `coordinate`, through phased and per-target entries."""
    for reader in readers:
        if isinstance(reader, AgeSpecializedFunction):
            yield resolve_node(node=reader, age=coordinate)
        elif isinstance(reader, Phased):
            yield from _resolved(
                readers=(reader.solve, reader.simulate), coordinate=coordinate
            )
        elif isinstance(reader, Mapping):
            yield from _resolved(readers=tuple(reader.values()), coordinate=coordinate)
        else:
            yield reader


def _repeated_node_errors(
    *,
    regime_name: RegimeName,
    action_name: ActionName,
    action: ConditionalAction,
    nodes: Float1D,
) -> list[str]:
    """Return an error when the inactive node's value occurs at another node too.

    The pin identifies the inactive node by its value, which needs the value to be
    unique among the nodes the grid generates at the build precision. A very narrow
    or very fine grid can round neighbouring nodes to one value.
    """
    count = int((nodes == nodes[action.inactive_index]).sum())
    if count == 1:
        return []
    node = float(nodes[action.inactive_index])
    return [
        (
            f"Regime {regime_name!r}: the inactive node {node!r} of the conditional "
            f"action {action_name!r} occurs {count} times among the grid's "
            f"{nodes.dtype} nodes, so it does not name one node. Use a grid whose "
            "nodes are distinct at this precision."
        )
    ]


def _fail_if_precision_differs(
    *, action_name: ActionName, action: FloatND, node_dtype: str
) -> None:
    if str(action.dtype) != node_dtype:
        raise TypeError(
            f"The conditional action {action_name!r} was built at {node_dtype} but is "
            f"evaluated at {action.dtype}. Build the model at the precision it is "
            "solved and simulated at."
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
    *, callables: Iterator[UserFunction], arg_name: ReferenceName
) -> str | None:
    """Return the annotation the first callable declaring `arg_name` gives it."""
    for func in callables:
        annotations = ensure_annotations_are_strings(get_annotations(func))
        if arg_name in annotations:
            return annotations[arg_name]
    return None


def _callables(readers: tuple[_Reader, ...]) -> Iterator[UserFunction]:
    """Yield the plain callables in `readers`, through phased and per-target entries.

    Callables built per age or period are left to `_specialized`.
    """
    for reader in readers:
        if isinstance(reader, Phased):
            yield from _callables((reader.solve, reader.simulate))
        elif isinstance(reader, Mapping):
            yield from _callables(tuple(reader.values()))
        elif isinstance(reader, CollectiveUtility):
            yield from _callables(tuple(reader.utilities.values()))
        elif isinstance(reader, ValueDependentConstraint):
            yield reader.predicate
        elif callable(reader) and not isinstance(
            reader, AgeSpecializedFunction | _SignedPin
        ):
            yield reader


def _specialized(readers: tuple[_Reader, ...]) -> Iterator[AgeSpecializedFunction]:
    """Yield the callables in `readers` built per age or period, through entries."""
    for reader in readers:
        if isinstance(reader, AgeSpecializedFunction):
            yield reader
        elif isinstance(reader, Phased):
            yield from _specialized((reader.solve, reader.simulate))
        elif isinstance(reader, Mapping):
            yield from _specialized(tuple(reader.values()))
        elif isinstance(reader, CollectiveUtility):
            yield from _specialized(tuple(reader.utilities.values()))
