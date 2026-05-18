from types import MappingProxyType
from typing import Literal, cast

from lcm.grids import Grid
from lcm.interfaces import SolveSimulateFunctionPair
from lcm.regime_building.transitions import collect_state_transitions
from lcm.typing import UserFunction
from lcm.user_regime import Regime as UserRegime
from lcm.user_regime import _default_H


class MockRegime(UserRegime):
    """A mock of the user-provided `Regime` for params-template tests.

    Inherits from `UserRegime` so `isinstance(x, UserRegime)` holds at
    the beartype-checked perimeter of `create_regime_params_template`
    and friends, but bypasses `UserRegime.__init__`'s validation by
    writing fields directly via `object.__setattr__`. Tests use this to
    supply partial / loosely-typed configurations that the real
    constructor would reject.

    """

    def __init__(
        self,
        *,
        n_periods: int | None = None,
        actions: dict[str, Grid | None] | None = None,
        states: dict[str, Grid | None] | None = None,
        state_transitions: dict[str, UserFunction | None] | None = None,
        constraints: dict[str, UserFunction] | None = None,
        transition: UserFunction | None = None,
        functions: dict[str, UserFunction] | None = None,
    ) -> None:
        object.__setattr__(self, "n_periods", n_periods)
        object.__setattr__(self, "actions", actions if actions is not None else {})
        object.__setattr__(self, "states", states if states is not None else {})
        object.__setattr__(
            self,
            "state_transitions",
            state_transitions if state_transitions is not None else {},
        )
        object.__setattr__(
            self, "constraints", constraints if constraints is not None else {}
        )
        object.__setattr__(self, "transition", transition)
        object.__setattr__(
            self, "functions", functions if functions is not None else {}
        )
        # Match UserRegime's defaults for fields MockRegime callers don't touch
        object.__setattr__(self, "active", lambda _age: True)
        object.__setattr__(self, "derived_categoricals", MappingProxyType({}))
        object.__setattr__(self, "description", "")
        # `UserRegime.__post_init__` injects the default `H` for non-terminal
        # regimes; mirror that here.
        if not self.terminal and "H" not in self.functions:
            object.__setattr__(self, "functions", {**self.functions, "H": _default_H})

    @property
    def terminal(self) -> bool:
        return self.transition is None

    def get_all_functions(
        self, phase: Literal["solve", "simulate"] = "solve"
    ) -> MappingProxyType[str, UserFunction]:
        """Get all regime functions including utility, constraints, and transitions."""
        result: dict[str, UserFunction] = {}
        for name, func in self.functions.items():
            if isinstance(func, SolveSimulateFunctionPair):
                result[name] = cast(
                    "UserFunction",
                    func.solve if phase == "solve" else func.simulate,
                )
            else:
                result[name] = func
        result |= dict(self.constraints)
        if self.states:
            result |= collect_state_transitions(
                {k: v for k, v in self.states.items() if v is not None},
                self.state_transitions,
            )
        if self.transition:
            result["next_regime"] = self.transition
        return MappingProxyType(result)
