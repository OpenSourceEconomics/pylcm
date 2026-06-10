"""The user-facing `Regime` definition.

The validators and the identity transition live behind a leading underscore in
`_lcm.user_regime_validation` and `_lcm.regime_building.transitions`. This
module is intentionally thin: the public class definition plus the private
default Bellman aggregator (`_default_H`), which `Regime` injects when a
non-terminal regime supplies no `H`.

"""

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, cast

from beartype import beartype

from _lcm.beartype_conf import REGIME_CONF
from _lcm.grids import DiscreteGrid, Grid
from _lcm.regime_building.phases import normalize_regime_phases
from _lcm.regime_building.transitions import collect_state_transitions
from _lcm.typing import ActionName, ActiveFunction, FunctionName, RegimeName, StateName
from _lcm.user_regime_validation import (
    _validate_logical_consistency,
    _validate_mapping_contents,
)
from _lcm.utils.containers import (
    ensure_containers_are_immutable,
)
from lcm.exceptions import RegimeInitializationError
from lcm.phased import Phased
from lcm.taste_shocks import ExtremeValueTasteShocks
from lcm.transition import MarkovTransition
from lcm.typing import FloatND, UserFunction


@beartype(conf=REGIME_CONF)
@dataclass(frozen=True, kw_only=True)
class Regime:
    """User-facing regime definition.

    `Model` processes instances of this class into the canonical regime form
    (`_lcm.engine.Regime`) used internally by the solver and simulator.

    State transitions are specified via `state_transitions`, mapping state names to
    transition functions. A bare callable is deterministic; wrap in `MarkovTransition`
    for stochastic transitions. `fixed_transition(state_name)` marks a fixed state
    (identity law). Stochastic processes have intrinsic transitions and must not
    appear in `state_transitions`.

    The `transition` field on the regime itself is the *regime* transition function.
    A regime with `transition=None` is terminal — no separate `terminal` flag is
    needed.

    """

    transition: UserFunction | MarkovTransition | Phased | None
    """Regime transition function, or `None` for terminal regimes.

    A bare callable is deterministic. Wrap in `MarkovTransition` for stochastic
    regime transitions that return probability distributions. `Phased` gives
    each phase its own variant (matching stochasticity required).
    """

    active: ActiveFunction = lambda _age: True
    """Callable that takes age (float) and returns True if regime is active."""

    states: Mapping[StateName, Grid | Phased] = field(
        default_factory=lambda: MappingProxyType({})
    )
    """Mapping of state variable names to grids or phase-variant declarations.

    A plain `Grid` value is a state shared by both phases.
    `Phased(solve=callable, simulate=Grid)` declares a carried state: a
    derived function (no grid axis) in the solve phase and a seeded, evolved
    state in the simulate phase, whose law of motion is its regular
    `state_transitions` entry.
    """

    state_transitions: Mapping[
        StateName,
        UserFunction
        | MarkovTransition
        | Phased
        | None
        # `Phased` inside a per-target dict passes the type check so the
        # validator can reject it with the outermost-only explanation.
        | Mapping[RegimeName, UserFunction | MarkovTransition | Phased],
    ] = field(default_factory=lambda: MappingProxyType({}))
    """Mapping of state names to transition functions or per-target dicts.

    Every non-process state must have an entry — omitting a state raises an error.
    `fixed_transition(state_name)` marks a fixed state (identity law). Wrap in
    `MarkovTransition` for stochastic transitions. Per-target dicts map target
    regime names to transition functions — every reachable target must be listed.
    `Phased` gives each phase its own law of motion; it wraps the whole entry
    (outermost only, never inside a per-target dict).
    """

    actions: Mapping[ActionName, Grid] = field(
        default_factory=lambda: MappingProxyType({})
    )
    """Mapping of action variable names to grid objects."""

    functions: Mapping[FunctionName, UserFunction | Phased] = field(
        default_factory=lambda: MappingProxyType({})
    )
    """Mapping of function names to callables; must include 'utility'.

    `Phased` gives each phase its own implementation.
    """

    # `Phased` passes the type check so the validator can reject it with an
    # explanation (constraints are phase-invariant).
    constraints: Mapping[FunctionName, UserFunction | Phased] = field(
        default_factory=lambda: MappingProxyType({})
    )
    """Mapping of constraint names to constraint functions.

    Constraints are phase-invariant: a phase-specific feasible set would let
    the simulated argmax range over actions the value function was never
    computed for, so `Phased` is rejected here.
    """

    derived_categoricals: Mapping[FunctionName, DiscreteGrid] = field(
        default_factory=lambda: MappingProxyType({})
    )
    """Categorical grids for DAG function outputs not in states/actions."""

    taste_shocks: ExtremeValueTasteShocks | None = None
    """EV1 taste shocks on the regime's discrete-action combinations.

    When set, the shock scale becomes the runtime param
    `{"taste_shocks": {"scale": ...}}` and the solve aggregates discrete
    actions via the smoothed expected maximum instead of the hard maximum.
    Requires at least one discrete action.
    """

    description: str = ""
    """Description of the regime."""

    @property
    def terminal(self) -> bool:
        """Whether this is a terminal regime (derived from transition being None)."""
        return self.transition is None

    @property
    def stochastic_regime_transition(self) -> bool:
        """Whether the regime transition is stochastic (MarkovTransition).

        `Phased` variants must have matching stochasticity, so the solve
        variant is representative.
        """
        transition = (
            self.transition.solve
            if isinstance(self.transition, Phased)
            else self.transition
        )
        return isinstance(transition, MarkovTransition)

    def __post_init__(self) -> None:
        _validate_mapping_contents(self)
        _validate_logical_consistency(self)

        def make_immutable(name: str) -> None:
            value = ensure_containers_are_immutable(getattr(self, name))
            object.__setattr__(self, name, value)

        # Completeness (a `utility` entry, default-`H` injection, transition
        # coverage) is validated when the model builds its effective regimes
        # — model-level slots may still satisfy it after merging.
        make_immutable("functions")
        make_immutable("states")
        make_immutable("state_transitions")
        make_immutable("actions")
        make_immutable("constraints")
        make_immutable("derived_categoricals")

        # The phase grammar (states matrix, carried laws, regime-transition
        # variants) is validated by the normalizer; the per-phase spec it
        # builds is consumed during model processing.
        normalize_regime_phases(self)

    def get_all_functions(
        self,
        phase: Literal["solve", "simulate"] = "solve",
    ) -> MappingProxyType[str, UserFunction]:
        """Get all regime functions including utility, constraints, and transitions.

        Collect functions from four sources:
        - `self.functions` (utility, helpers, H)
        - `self.constraints`
        - State transitions from `self.state_transitions`
        - The regime transition (`self.transition`, keyed as `"next_regime"`)

        For `Phased` entries, the variant matching `phase` is used. A
        carried-state declaration in `states` (`Phased(solve=...,
        simulate=Grid)`) contributes its `solve` variant as a derived
        function under the state's name and its law of motion under
        `next_<name>`, mirroring how ordinary state transitions are keyed.

        Args:
            phase: Which variant to use for phase-variant entries.

        Returns:
            Read-only mapping of all regime functions.

        """

        def resolve(value: object) -> UserFunction:
            if isinstance(value, Phased):
                value = value.solve if phase == "solve" else value.simulate
            return cast("UserFunction", value)

        result: dict[str, UserFunction] = {
            name: resolve(func) for name, func in self.functions.items()
        }
        for name, spec in self.states.items():
            if isinstance(spec, Phased):
                # Carried state: the solve variant is its derived-function
                # imputation; the law of motion is its regular
                # `state_transitions` entry, collected below.
                result[name] = cast("UserFunction", spec.solve)
        result |= cast("Mapping[str, UserFunction]", self.constraints)
        if callable(self.transition) or isinstance(self.transition, Phased):
            collected = collect_state_transitions(self.states, self.state_transitions)
            result |= {name: resolve(func) for name, func in collected.items()}
            result["next_regime"] = resolve(self.transition)
        return MappingProxyType(result)

    def replace(self, **kwargs: Any) -> Regime:  # noqa: ANN401
        """Replace the attributes of the regime.

        Args:
            **kwargs: Keyword arguments to replace the attributes of the regime.

        Returns:
            A new regime with the replaced attributes.

        """
        try:
            return dataclasses.replace(self, **kwargs)
        except TypeError as e:
            raise RegimeInitializationError(
                f"Failed to replace attributes of the regime. The error was: {e}"
            ) from e


def _default_H(
    utility: FloatND, E_next_V: FloatND, discount_factor: FloatND
) -> FloatND:
    """Default Bellman aggregator: `U + β · E[V_next]`."""
    return utility + discount_factor * E_next_V
