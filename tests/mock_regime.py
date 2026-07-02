from types import MappingProxyType
from typing import Literal, cast

from _lcm.grids import Grid
from lcm.aggregators import H_linear
from lcm.regime import Regime as UserRegime
from lcm.typing import UserFunction


def _noop() -> None:
    """Stand-in regime transition while collecting a transition-less mock."""


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
        # `finalize_regimes` injects the default `H` for non-terminal
        # regimes; mirror that here.
        if not self.terminal and "H" not in self.functions:
            object.__setattr__(self, "functions", {**self.functions, "H": H_linear})

    @property
    def terminal(self) -> bool:
        return self.transition is None

    def get_all_functions(
        self, phase: Literal["solve", "simulate"] = "solve"
    ) -> MappingProxyType[str, UserFunction]:
        """Delegate to the real method, tolerating the mock's loose fields.

        Mocks may carry `None`-valued states (partial configurations the
        real constructor would reject) and rely on state transitions being
        collected even without a regime transition. Normalize both, then
        reuse `Regime.get_all_functions` so the key set can never drift
        from the real regime's.
        """
        normalized = MockRegime(
            states=cast(
                "dict[str, Grid | None]",
                {k: v for k, v in self.states.items() if v is not None},
            ),
            state_transitions=cast(
                "dict[str, UserFunction | None]", self.state_transitions
            ),
            constraints=cast("dict[str, UserFunction]", self.constraints),
            transition=self.transition if callable(self.transition) else _noop,
            functions=cast("dict[str, UserFunction]", self.functions),
        )
        result = dict(UserRegime.get_all_functions(normalized, phase))
        if not callable(self.transition):
            del result["next_regime"]
        return MappingProxyType(result)
