"""Per-regime states + actions, with pre-computed name-tuple views.

`Variables` wraps an immutable `MappingProxyType[StateOrActionName,
VariableInfo]` and exposes 8 named tuples covering every kind/topology
cross-section that consumers need (`state_names`, `discrete_action_names`,
`shock_names`, etc.). Callers that need ad-hoc filters can iterate via the
`Mapping` interface.

Iteration order (set by `Variables.from_regime`): discrete states sorted by
`batch_size`, then continuous states sorted by `batch_size`, then actions.
Batch size 0 sorts last within its group (treated as +∞). Every named view
preserves this order.

"""

import dataclasses
import math
from collections.abc import Iterator, Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, Self

from lcm.grids import ContinuousGrid, Grid
from lcm.shocks import _ShockGrid
from lcm.typing import StateOrActionName

if TYPE_CHECKING:
    from lcm.regime import Regime


@dataclasses.dataclass(frozen=True)
class VariableInfo:
    """Kind/topology/shock tags for one state or action variable."""

    kind: Literal["state", "action"]
    """Whether the variable is a state or an action."""

    topology: Literal["continuous", "discrete"]
    """Topology as treated by pylcm's solve/simulate machinery.

    Shocks have topology `"discrete"` because their value space is
    approximated by a finite grid of nodes, even though the underlying
    random variable is mathematically continuous. Combine with `is_shock`
    when the distinction matters.

    """

    is_shock: bool
    """Whether the variable is a shock (always a state)."""


@dataclasses.dataclass(frozen=True)
class Variables(Mapping[StateOrActionName, VariableInfo]):
    """States + actions of a regime, with pre-computed name-tuple views.

    Mapping access by variable name returns the per-variable `VariableInfo`.
    Named accessors return tuples of names in iteration order. Use
    `Variables.from_regime` to construct from a regime; pass `info` directly
    only when names are already in the desired order.

    """

    info: MappingProxyType[StateOrActionName, VariableInfo]
    """Immutable mapping of variable name to its `VariableInfo`."""

    state_names: tuple[StateOrActionName, ...] = dataclasses.field(init=False)
    """Names of variables with kind='state'."""

    action_names: tuple[StateOrActionName, ...] = dataclasses.field(init=False)
    """Names of variables with kind='action'."""

    discrete_state_names: tuple[StateOrActionName, ...] = dataclasses.field(init=False)
    """Names of states with topology='discrete' (includes shocks)."""

    continuous_state_names: tuple[StateOrActionName, ...] = dataclasses.field(
        init=False
    )
    """Names of states with topology='continuous'."""

    discrete_action_names: tuple[StateOrActionName, ...] = dataclasses.field(init=False)
    """Names of actions with topology='discrete'."""

    continuous_action_names: tuple[StateOrActionName, ...] = dataclasses.field(
        init=False
    )
    """Names of actions with topology='continuous'."""

    state_and_discrete_action_names: tuple[StateOrActionName, ...] = dataclasses.field(
        init=False
    )
    """Every state plus every discrete action — the gridded variable set."""

    shock_names: tuple[StateOrActionName, ...] = dataclasses.field(init=False)
    """Names of variables with `is_shock=True`."""

    def __post_init__(self) -> None:
        items = tuple(self.info.items())
        # `object.__setattr__` is required to bypass the frozen guard.
        object.__setattr__(
            self,
            "state_names",
            tuple(name for name, info in items if info.kind == "state"),
        )
        object.__setattr__(
            self,
            "action_names",
            tuple(name for name, info in items if info.kind == "action"),
        )
        object.__setattr__(
            self,
            "discrete_state_names",
            tuple(
                name
                for name, info in items
                if info.kind == "state" and info.topology == "discrete"
            ),
        )
        object.__setattr__(
            self,
            "continuous_state_names",
            tuple(
                name
                for name, info in items
                if info.kind == "state" and info.topology == "continuous"
            ),
        )
        object.__setattr__(
            self,
            "discrete_action_names",
            tuple(
                name
                for name, info in items
                if info.kind == "action" and info.topology == "discrete"
            ),
        )
        object.__setattr__(
            self,
            "continuous_action_names",
            tuple(
                name
                for name, info in items
                if info.kind == "action" and info.topology == "continuous"
            ),
        )
        object.__setattr__(
            self,
            "state_and_discrete_action_names",
            tuple(
                name
                for name, info in items
                if info.kind == "state" or info.topology == "discrete"
            ),
        )
        object.__setattr__(
            self,
            "shock_names",
            tuple(name for name, info in items if info.is_shock),
        )

    def __getitem__(self, key: StateOrActionName) -> VariableInfo:
        return self.info[key]

    def __iter__(self) -> Iterator[StateOrActionName]:
        return iter(self.info)

    def __len__(self) -> int:
        return len(self.info)

    @classmethod
    def from_regime(cls, regime: Regime) -> Self:
        """Build `Variables` from a regime, ordering names canonically.

        Order: discrete states (by `batch_size`), continuous states (by
        `batch_size`), then actions in declaration order. Within each topology
        group, `batch_size == 0` sorts last.

        Args:
            regime: The regime as provided by the user.

        Returns:
            A `Variables` instance whose iteration order matches the canonical
            ordering described above.

        """
        raw_info = _raw_variable_info(regime)
        ordered_names = _ordered_state_action_names(regime, raw_info)
        return cls(
            info=MappingProxyType({name: raw_info[name] for name in ordered_names})
        )


def get_grids(
    regime: Regime,
) -> MappingProxyType[StateOrActionName, Grid]:
    """Create a mapping of grid objects for each variable in the regime.

    Args:
        regime: The regime as provided by the user.

    Returns:
        Immutable mapping of state and action variable names to their grid objects,
        in the canonical order used by `Variables.from_regime` (discrete states,
        continuous states, then actions).

    """
    variables = Variables.from_regime(regime)
    raw_variables = dict(regime.states) | dict(regime.actions)
    return MappingProxyType({name: raw_variables[name] for name in variables})


def _raw_variable_info(
    regime: Regime,
) -> dict[StateOrActionName, VariableInfo]:
    """Derive `VariableInfo` for every state and action variable."""
    variables = dict(regime.states) | dict(regime.actions)
    info: dict[StateOrActionName, VariableInfo] = {}
    for name, spec in variables.items():
        is_state = name in regime.states
        is_shock = isinstance(spec, _ShockGrid)
        is_continuous = isinstance(spec, ContinuousGrid) and not is_shock
        info[name] = VariableInfo(
            kind="state" if is_state else "action",
            topology="continuous" if is_continuous else "discrete",
            is_shock=is_shock,
        )
    return info


def _ordered_state_action_names(
    regime: Regime,
    info: dict[StateOrActionName, VariableInfo],
) -> list[StateOrActionName]:
    """Order variables: discrete states, continuous states, actions.

    States are sorted by `batch_size` within each topology group; batch size 0
    sorts last (treated as +inf). Actions keep declaration order.

    """

    def state_batch_size(name: StateOrActionName) -> float:
        batch_size = regime.states[name].batch_size
        return batch_size if batch_size != 0 else math.inf

    discrete_states = sorted(
        (
            name
            for name, var_info in info.items()
            if var_info.kind == "state" and var_info.topology == "discrete"
        ),
        key=state_batch_size,
    )
    continuous_states = sorted(
        (
            name
            for name, var_info in info.items()
            if var_info.kind == "state" and var_info.topology == "continuous"
        ),
        key=state_batch_size,
    )
    actions = [name for name, var_info in info.items() if var_info.kind == "action"]

    ordered = [*discrete_states, *continuous_states, *actions]
    if set(ordered) != set(info):
        raise ValueError("Order and index do not match.")
    return ordered
