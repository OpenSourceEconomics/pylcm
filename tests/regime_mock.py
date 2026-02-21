from dataclasses import dataclass, field
from typing import Any

from lcm.regime import _collect_state_transitions, _default_H
from lcm.typing import UserFunction


@dataclass
class RegimeMock:
    """A regime mock for testing the params_template creation functions.

    This dataclass has the same attributes as the Regime dataclass, but does not perform
    any checks, which helps us test the params_template creation functions in isolation.

    """

    n_periods: int | None = None
    actions: dict[str, Any] | None = None
    states: dict[str, Any] | None = None
    constraints: dict[str, UserFunction] = field(default_factory=dict)
    transition: UserFunction | None = None
    functions: dict[str, UserFunction] = field(default_factory=dict)

    @property
    def terminal(self) -> bool:
        return self.transition is None

    def __post_init__(self) -> None:
        if not self.terminal and "H" not in self.functions:
            self.functions = {**self.functions, "H": _default_H}

    def get_all_functions(self) -> dict[str, UserFunction]:
        """Get all regime functions including utility, constraints, and transitions."""
        result = dict(self.functions) | dict(self.constraints)
        if self.states:
            result |= _collect_state_transitions(
                {k: v for k, v in self.states.items() if v is not None},
            )
        if self.transition:
            result["next_regime"] = self.transition
        return result
