from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Literal

from _lcm.regime_law import RegimeLaw
from lcm.koopmans_aggregation import LinearAggregator
from lcm.phased import Phased
from lcm.regime import (
    ActionEntry,
    ConstraintEntry,
    FunctionEntry,
    StateEntry,
    StateTransitionEntry,
)
from lcm.regime import Regime as UserRegime
from lcm.solvers import GridSearch, Solver
from lcm.transition import AgeSpecializedFunction
from lcm.typing import UserFunction


class MockRegime(UserRegime):
    """A mock of the user-provided `Regime` for params-template tests.

    Inherits from `UserRegime` so `isinstance(x, UserRegime)` holds at
    the beartype-checked perimeter of `create_regime_params_template`
    and friends, but bypasses `UserRegime.__init__`'s validation by
    writing fields directly via `object.__setattr__`. Tests use this to
    supply partial / loosely-typed configurations that the real
    constructor would reject. A mock carries no law between regimes, as
    no regime does; `terminal` only decides whether the model-level
    aggregator is injected, as `finalize_regimes` does for a regime without
    outgoing edges.

    """

    def __init__(
        self,
        *,
        n_periods: int | None = None,
        actions: Mapping[str, ActionEntry] | None = None,
        states: Mapping[str, StateEntry] | None = None,
        state_transitions: Mapping[str, StateTransitionEntry] | None = None,
        constraints: Mapping[str, ConstraintEntry] | None = None,
        terminal: bool = False,
        # Loosely typed on purpose: tests pass markers (`AgeSpecializedFunction`,
        # `Phased`) alongside plain callables.
        functions: Mapping[
            str, FunctionEntry | AgeSpecializedFunction | Callable[..., None]
        ]
        | None = None,
        koopmans_aggregator: UserFunction | Phased | None = None,
        solver: Solver | None = None,
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
        object.__setattr__(
            self, "functions", functions if functions is not None else {}
        )
        # `finalize_regimes` injects the model-level aggregator into
        # non-terminal regimes; mirror that here.
        object.__setattr__(
            self,
            "koopmans_aggregator",
            koopmans_aggregator
            if koopmans_aggregator is not None or terminal
            else LinearAggregator(),
        )
        object.__setattr__(
            self, "solver", solver if solver is not None else GridSearch()
        )
        # Match UserRegime's defaults for fields MockRegime callers don't touch
        object.__setattr__(self, "derived_categoricals", MappingProxyType({}))
        object.__setattr__(self, "joint_transitions", MappingProxyType({}))
        object.__setattr__(self, "description", "")
        # `value_constraints` / `same_period_refs` use
        # default_factory on the real dataclass, so no class-level fallback
        # exists — set them here.
        object.__setattr__(self, "value_constraints", MappingProxyType({}))
        object.__setattr__(self, "same_period_refs", MappingProxyType({}))

    # keyword-only-exempt: primary-argument=phase
    def get_all_functions(
        self,
        phase: Literal["solve", "simulate"] = "solve",
        *,
        law: RegimeLaw | None = None,
    ) -> MappingProxyType[str, UserFunction]:
        """Delegate to the real method, tolerating the mock's loose fields.

        Mocks may carry `None`-valued states (partial configurations the
        real constructor would reject) and rely on state transitions being
        collected even under a law that is not a single callable — a terminal
        or a per-target law. Drop the `None` states and collect such a mock's
        state laws without a regime transition, then reuse
        `Regime.get_all_functions` so the key set can never drift from the
        real regime's.
        """
        normalized = MockRegime(
            states={k: v for k, v in self.states.items() if v is not None},
            state_transitions=self.state_transitions,
            constraints=self.constraints,
            functions=self.functions,
        )
        callable_law = law if law is not None and callable(law.transition) else None
        return UserRegime.get_all_functions(normalized, phase, law=callable_law)
