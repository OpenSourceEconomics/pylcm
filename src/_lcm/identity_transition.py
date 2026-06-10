"""The identity law of motion backing `lcm.fixed_transition`.

A leaf module with no dependency on `Regime`, the validators, or the
regime-building code, so that the user-facing `lcm.transition` module and the
engine-internal collectors can both import it without an import cycle.
"""

import inspect
from typing import TypeAliasType, overload

from _lcm.typing import StateName
from lcm.typing import ContinuousState, DiscreteState


class _IdentityTransition:
    """Identity law of motion for a fixed state: next value = current value.

    Instances are produced by `lcm.fixed_transition` (no dtype annotation —
    the grid is not known at factory-call time) and rebuilt by the engine's
    transition collector with the state's grid-matched annotation. The
    `_is_auto_identity` attribute lets validation distinguish identity laws
    from user-provided transitions.

    """

    _is_auto_identity: bool = True

    def __init__(
        self, state_name: StateName, *, annotation: TypeAliasType | None = None
    ) -> None:
        self._state_name = state_name
        self.__name__ = f"next_{state_name}"
        param = inspect.Parameter(
            state_name,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation=annotation if annotation is not None else inspect.Parameter.empty,
        )
        self.__signature__ = inspect.Signature(
            [param],
            return_annotation=(
                annotation if annotation is not None else inspect.Signature.empty
            ),
        )
        if annotation is not None:
            self.__annotations__ = {state_name: annotation, "return": annotation}

    @overload
    def __call__(self, **kwargs: DiscreteState) -> DiscreteState: ...
    @overload
    def __call__(self, **kwargs: ContinuousState) -> ContinuousState: ...
    def __call__(
        self, **kwargs: DiscreteState | ContinuousState
    ) -> DiscreteState | ContinuousState:
        return kwargs[self._state_name]
