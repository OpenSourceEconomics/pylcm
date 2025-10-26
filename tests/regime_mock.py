from dataclasses import dataclass, field
from typing import Any

from lcm.typing import UserFunction


@dataclass
class RegimeMock:
    """A regime mock for testing the params creation functions.

    This dataclass has the same attributes as the Regime dataclass, but does not perform
    any checks, which helps us to test the params creation functions in isolation.

    """

    active: list[int] | None = None
    actions: dict[str, Any] | None = None
    utility: UserFunction | None = None
    states: dict[str, Any] | None = None
    constraints: dict[str, UserFunction] = field(default_factory=dict)
    transitions: dict[str, UserFunction] = field(default_factory=dict)
    functions: dict[str, UserFunction] = field(default_factory=dict)

    def get_all_functions(self) -> dict[str, UserFunction | None]:
        """Get all regime functions including utility, constraints, and transitions."""
        return (
            self.functions
            | {"utility": self.utility}
            | self.constraints
            | self.transitions
        )
