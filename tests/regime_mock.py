from dataclasses import dataclass, field
from typing import cast

from lcm.grids import Grid
from lcm.regime import _collect_state_transitions, _default_H, _is_phase_dict
from lcm.typing import UserFunction


@dataclass
class RegimeMock:
    """A regime mock for testing the params_template creation functions.

    This dataclass has the same attributes as the Regime dataclass, but does not perform
    any checks, which helps us test the params_template creation functions in isolation.

    """

    n_periods: int | None = None
    actions: dict[str, Grid | None] | None = None
    states: dict[str, Grid | None] | None = None
    state_transitions: dict[str, UserFunction | None] = field(default_factory=dict)
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
        result: dict[str, UserFunction] = {}
        for name, func in self.functions.items():
            result[name] = cast(
                "UserFunction",
                func["solve"] if _is_phase_dict(func) else func,  # ty: ignore[not-subscriptable]
            )
        result |= dict(self.constraints)
        if self.states:
            result |= _collect_state_transitions(
                {k: v for k, v in self.states.items() if v is not None},
                self.state_transitions,
            )
        if self.transition:
            result["next_regime"] = self.transition
        return result
