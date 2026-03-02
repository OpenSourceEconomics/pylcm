from dataclasses import dataclass, field

from lcm.grids import DiscreteGrid, DiscreteMarkovGrid, Grid
from lcm.input_processing.process_transitions import _IdentityTransition
from lcm.regime import _default_H
from lcm.shocks._base import _ShockGrid
from lcm.typing import ContinuousState, DiscreteState, UserFunction


@dataclass
class RegimeMock:
    """A regime mock for testing the params_template creation functions.

    This dataclass has the same attributes as the Regime dataclass, but does not perform
    any checks, which helps us test the params_template creation functions in isolation.

    """

    n_periods: int | None = None
    actions: dict[str, Grid | None] | None = None
    states: dict[str, Grid | None] | None = None
    constraints: dict[str, UserFunction] = field(default_factory=dict)
    transition: UserFunction | None = None
    functions: dict[str, UserFunction] = field(default_factory=dict)

    @property
    def terminal(self) -> bool:
        return self.transition is None

    def __post_init__(self) -> None:
        if not self.terminal and "H" not in self.functions:
            self.functions = {**self.functions, "H": _default_H}

    def get_user_functions(self) -> dict[str, UserFunction]:
        """Get all regime functions including utility, constraints, and transitions."""
        result: dict[str, UserFunction] = dict(self.functions)
        result.update(self.constraints)
        if self.states:
            for name, grid in self.states.items():
                if grid is None:
                    continue
                if isinstance(grid, _ShockGrid):
                    result[f"next_{name}"] = lambda: None
                elif isinstance(grid, DiscreteMarkovGrid):
                    transition = grid.transition
                    if callable(transition):
                        result[f"next_{name}"] = transition
                elif callable(grid_trans := getattr(grid, "transition", None)):
                    result[f"next_{name}"] = grid_trans
                elif grid_trans is None:
                    ann = (
                        DiscreteState
                        if isinstance(grid, DiscreteGrid)
                        else ContinuousState
                    )
                    result[f"next_{name}"] = _IdentityTransition(name, annotation=ann)
        if self.transition:
            result["next_regime"] = self.transition
        return result
