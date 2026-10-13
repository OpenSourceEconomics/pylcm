"""Declare deterministic discrete states that pylcm stores on reachable tuples only."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

from lcm.exceptions import ModelInitializationError
from lcm.typing import StateName

type CompositeVariant = Literal["closed", "masked"]


@dataclass(frozen=True, init=False)
class CompositeState:
    """Replace the axes of member states by one axis over their reachable tuples.

    The members must be deterministic discrete states whose laws read only discrete
    variables, the period, the age and fixed parameters. pylcm enumerates the member
    tuples reachable from `start_codes` at the admissible initial nodes and solves
    and stores values on those tuples only. User functions keep reading the members.

    Two variants are available:

    - `"closed"`: the code set is closed under every period's laws, so no tuple in
      it steps outside it under an allowed action;
    - `"masked"`: the code set holds the forward-reachable tuples, and a generated
      constraint makes each tuple infeasible at the periods it cannot be reached in.
    """

    members: tuple[StateName, ...]
    """The member states, in tuple order."""

    start_codes: MappingProxyType[StateName, tuple[int, ...]]
    """Per member, the codes a subject can start with."""

    variant: CompositeVariant
    """How the code set serves every period."""

    def __init__(
        self,
        *,
        members: Sequence[StateName],
        start_codes: Mapping[StateName, Sequence[int]],
        variant: CompositeVariant = "closed",
    ) -> None:
        members = tuple(members)
        if len(members) < 2 or len(set(members)) != len(members):  # noqa: PLR2004
            msg = (
                "A composite state needs at least two distinct members; got "
                f"{members!r}."
            )
            raise ModelInitializationError(msg)
        if set(start_codes) != set(members):
            msg = (
                f"start_codes must name exactly the members {list(members)}; got "
                f"{sorted(start_codes)}."
            )
            raise ModelInitializationError(msg)
        if variant not in ("closed", "masked"):
            msg = f"variant must be 'closed' or 'masked'; got {variant!r}."
            raise ModelInitializationError(msg)
        object.__setattr__(self, "members", members)
        object.__setattr__(
            self,
            "start_codes",
            MappingProxyType(
                {
                    name: tuple(int(code) for code in start_codes[name])
                    for name in members
                }
            ),
        )
        object.__setattr__(self, "variant", variant)
