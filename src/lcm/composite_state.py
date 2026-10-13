"""Declare deterministic discrete states that pylcm stores on reachable tuples only."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from lcm.exceptions import ModelInitializationError
from lcm.typing import StateName

type CompositeVariant = Literal["closed", "masked"]


@dataclass(frozen=True, kw_only=True)
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

    start_codes: Mapping[StateName, tuple[int, ...]]
    """Per member, the codes a subject can start with."""

    variant: CompositeVariant = "closed"
    """How the code set serves every period."""

    def __post_init__(self) -> None:
        if len(self.members) < 2 or len(set(self.members)) != len(self.members):  # noqa: PLR2004
            msg = (
                "A composite state needs at least two distinct members; got "
                f"{self.members!r}."
            )
            raise ModelInitializationError(msg)
        if set(self.start_codes) != set(self.members):
            msg = (
                f"start_codes must name exactly the members {list(self.members)}; "
                f"got {sorted(self.start_codes)}."
            )
            raise ModelInitializationError(msg)
