from dataclasses import dataclass, field
from typing import Any

from lcm.mark import stochastic
from lcm.regime import _default_H, _make_identity_fn
from lcm.shocks._base import _ShockGrid
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
        # Collect state transitions from grids (skip None grids used in mock tests)
        if self.states:
            for name, grid in self.states.items():
                if grid is None:
                    continue
                if isinstance(grid, _ShockGrid):
                    result[f"next_{name}"] = stochastic(lambda: None)
                    continue
                grid_transition = getattr(grid, "transition", None)
                if grid_transition is not None:
                    result[f"next_{name}"] = grid_transition
                else:
                    result[f"next_{name}"] = _make_identity_fn(name)
        # Add regime transition
        if self.transition is not None:
            result["next_regime"] = self.transition
        return result
