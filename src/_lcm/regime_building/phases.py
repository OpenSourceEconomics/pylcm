"""Normalize a user regime's slots into per-phase specifications.

Phase is a broadcast dimension of the user spec: a bare slot value applies to
both the solve and simulate phases, `Phased(solve=..., simulate=...)`
specifies each phase explicitly. `normalize_regime_phases` expands every slot
via that rule, applies the per-slot grammar, and aggregates violations into a
single `RegimeInitializationError`.

This is the single place that resolves phase-variant values into per-phase
slices; everything downstream consumes the slices.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, cast

from _lcm.grids import Grid
from _lcm.processes.base import _ContinuousStochasticProcess
from _lcm.typing import FunctionName, RegimeName, StateName
from _lcm.utils.error_messages import format_messages
from lcm.exceptions import RegimeInitializationError
from lcm.phased import Phased
from lcm.transition import MarkovTransition
from lcm.typing import UserFunction

if TYPE_CHECKING:
    import lcm.regime

type _PhaseStateTransition = (
    UserFunction
    | MarkovTransition
    | None
    | Mapping[RegimeName, UserFunction | MarkovTransition]
)
type _PhaseRegimeTransition = (
    UserFunction | MarkovTransition | Mapping[RegimeName, MarkovTransition] | None
)


def normalize_regime_phases(user_regime: lcm.regime.Regime) -> PhasedRegimeSpec:
    """Expand a user regime's slots into per-phase specifications.

    Every phase-variant slot is split via one rule — `Phased` assigns each
    variant to its phase, a bare value broadcasts to both — then the per-slot
    grammar is applied. All violations are aggregated into a single
    `RegimeInitializationError`.

    Args:
        user_regime: User-form `Regime` instance.

    Returns:
        The regime expanded into solution / simulation slices.

    Raises:
        RegimeInitializationError: If any slot value violates the phase
            grammar.

    """
    solve_functions, simulate_functions, function_errors = _split_functions(
        user_regime=user_regime
    )
    solve_grid_states, simulate_grid_states, carried_imputations, state_errors = (
        _split_states(user_regime=user_regime)
    )

    collision_errors = [
        f"State '{name}' is carried: its solve-phase imputation is "
        f"registered as a derived function under '{name}', colliding "
        f"with the regime function of the same name. Rename one of "
        f"the two."
        for name in carried_imputations
        if name in user_regime.functions
    ]
    solve_functions = {**solve_functions, **carried_imputations}

    solve_state_transitions, simulate_state_transitions = _split_state_transitions(
        user_regime=user_regime
    )

    carried_only = frozenset(simulate_grid_states) - frozenset(solve_grid_states)
    solve_state_transitions = {
        name: law
        for name, law in solve_state_transitions.items()
        if name not in carried_only
    }
    carried_errors = [
        message
        for name in sorted(carried_only)
        for message in _carried_law_errors(
            name=name, law=simulate_state_transitions.get(name)
        )
    ]

    solve_transition, simulate_transition, transition_errors = _split_regime_transition(
        user_regime=user_regime
    )
    terminal = user_regime.transition is None
    terminal_errors = (
        [
            f"Terminal regimes cannot declare carried states (no next period "
            f"to carry {sorted(carried_only)} into)."
        ]
        if terminal and carried_only
        else []
    )

    errors = (
        function_errors
        + state_errors
        + collision_errors
        + carried_errors
        + transition_errors
        + terminal_errors
    )
    if errors:
        raise RegimeInitializationError(format_messages(errors))

    def _phase_spec(
        *,
        functions: dict[FunctionName, UserFunction],
        grid_states: dict[StateName, Grid],
        state_transitions: dict[StateName, _PhaseStateTransition],
        regime_transition: _PhaseRegimeTransition,
    ) -> RegimePhaseSpec:
        return RegimePhaseSpec(
            functions=MappingProxyType(functions),
            # Constraints are phase-invariant pure callables by the slot
            # grammar (phase-variant containers were rejected above).
            constraints=MappingProxyType(
                cast("dict[FunctionName, UserFunction]", dict(user_regime.constraints))
            ),
            grid_states=MappingProxyType(grid_states),
            state_transitions=MappingProxyType(state_transitions),
            regime_transition=regime_transition,
            # A per-target dict is stochastic by construction (each cell is a
            # MarkovTransition-wrapped probability function).
            stochastic_regime_transition=isinstance(
                regime_transition, MarkovTransition | Mapping
            ),
        )

    return PhasedRegimeSpec(
        solution=_phase_spec(
            functions=solve_functions,
            grid_states=solve_grid_states,
            state_transitions=solve_state_transitions,
            regime_transition=solve_transition,
        ),
        simulation=_phase_spec(
            functions=simulate_functions,
            grid_states=simulate_grid_states,
            state_transitions=simulate_state_transitions,
            regime_transition=simulate_transition,
        ),
    )


@dataclass(frozen=True, kw_only=True)
class PhasedRegimeSpec:
    """A regime expanded into per-phase slices.

    Phase-invariant slots with no phase-resolved consumers (actions,
    derived categoricals) stay on the user regime; the spec carries only
    what the phase builders read.
    """

    solution: RegimePhaseSpec
    """The solve-phase slice (backward induction)."""

    simulation: RegimePhaseSpec
    """The simulate-phase slice (forward simulation)."""

    @property
    def terminal(self) -> bool:
        """Whether the regime is terminal (no regime transition in either phase).

        Terminality is phase-invariant by the slot grammar (`Phased` variants
        cannot be `None`), so the solution slice is representative.
        """
        return self.solution.regime_transition is None

    @property
    def carried_only_state_names(self) -> frozenset[StateName]:
        """States carried in simulation but derived (no grid axis) in solution."""
        return frozenset(self.simulation.grid_states) - frozenset(
            self.solution.grid_states
        )


@dataclass(frozen=True, kw_only=True)
class RegimePhaseSpec:
    """One phase's slice of a regime specification."""

    functions: MappingProxyType[FunctionName, UserFunction]
    """Phase-resolved regime functions. The solution slice additionally holds
    each carried state's imputation as a derived function under the state's
    name; the simulation slice does not — there, the name is a genuine state."""

    constraints: MappingProxyType[FunctionName, UserFunction]
    """Constraint functions (phase-invariant by the slot grammar)."""

    grid_states: MappingProxyType[StateName, Grid]
    """States that are genuine grid states in this phase."""

    state_transitions: MappingProxyType[StateName, _PhaseStateTransition]
    """Phase-resolved laws of motion, restricted to this phase's grid states
    plus target-only entries."""

    regime_transition: _PhaseRegimeTransition
    """Phase-resolved regime transition; `None` for terminal regimes."""

    stochastic_regime_transition: bool
    """Whether this phase's regime transition is a `MarkovTransition`."""


def _split_functions(
    *, user_regime: lcm.regime.Regime
) -> tuple[
    dict[FunctionName, UserFunction], dict[FunctionName, UserFunction], list[str]
]:
    """Split `functions` into per-phase mappings, validating each variant.

    Returns the solve-phase mapping, the simulate-phase mapping, and the
    grammar violations found along the way.
    """
    solve_functions: dict[FunctionName, UserFunction] = {}
    simulate_functions: dict[FunctionName, UserFunction] = {}
    errors: list[str] = []
    for name, value in user_regime.functions.items():
        if isinstance(value, Phased):
            variants = (
                ("solve", value.solve, solve_functions),
                ("simulate", value.simulate, simulate_functions),
            )
            for phase_label, variant, target in variants:
                if callable(variant):
                    target[name] = cast("UserFunction", variant)
                else:
                    errors.append(
                        f"functions['{name}'] {phase_label} variant must be a "
                        f"callable, got {variant!r}."
                    )
        elif callable(value):
            solve_functions[name] = value
            simulate_functions[name] = value
        elif value is not None:
            # `None` masks a model-level entry; bound at model build.
            errors.append(f"functions['{name}'] must be a callable, got {value!r}.")
    return solve_functions, simulate_functions, errors


def _split_states(
    *, user_regime: lcm.regime.Regime
) -> tuple[
    dict[StateName, Grid],
    dict[StateName, Grid],
    dict[StateName, UserFunction],
    list[str],
]:
    """Split `states` into per-phase grids plus carried-state imputations.

    Returns the solve-phase grid states, the simulate-phase grid states, each
    carried state's solve-phase imputation (the carried law of motion is its
    regular `state_transitions` entry), and the grammar violations found
    along the way.
    """
    solve_grid_states: dict[StateName, Grid] = {}
    simulate_grid_states: dict[StateName, Grid] = {}
    carried_imputations: dict[StateName, UserFunction] = {}
    errors: list[str] = []
    for name, spec in user_regime.states.items():
        if isinstance(spec, Phased):
            imputation, carried_grid, state_errors = _normalize_phased_state(
                name=name, phased=spec
            )
            errors += state_errors
            if imputation is not None and carried_grid is not None:
                carried_imputations[name] = imputation
                simulate_grid_states[name] = carried_grid
        elif isinstance(spec, Grid):
            solve_grid_states[name] = spec
            simulate_grid_states[name] = spec
        elif spec is not None:
            # `None` masks a model-level entry; bound at model build.
            errors.append(
                f"states['{name}'] must be an LCM grid or `Phased`, got {spec!r}."
            )
    return solve_grid_states, simulate_grid_states, carried_imputations, errors


def _normalize_phased_state(
    *, name: StateName, phased: Phased
) -> tuple[UserFunction | None, Grid | None, list[str]]:
    """Apply the states matrix to one `Phased` state declaration.

    The only valid cell is `Phased(solve=callable, simulate=Grid)` — the
    carried state: derived (no grid axis) during backward induction, a genuine
    seeded-and-evolved state during simulation. Every other combination is
    rejected:

    - `(Grid, Grid)` ⇒ identical grids are a bare `Grid`; differing grid
      domains across phases have no defined semantics
    - `(Grid, callable)` ⇒ not yet supported
    - `(callable, callable)` ⇒ a both-phase derived function belongs in
      `functions`
    - a stochastic-process grid on either side ⇒ processes have intrinsic
      transitions and cannot be phase-variant

    Returns the carried state's solve-phase imputation and simulate-phase
    grid (both `None` when the declaration is rejected), and the violations.
    """
    solve_side, simulate_side = phased.solve, phased.simulate
    if isinstance(solve_side, _ContinuousStochasticProcess) or isinstance(
        simulate_side, _ContinuousStochasticProcess
    ):
        return (
            None,
            None,
            [
                f"states['{name}']: stochastic-process grids have intrinsic "
                f"transitions and cannot be phase-variant."
            ],
        )
    solve_is_grid = isinstance(solve_side, Grid)
    simulate_is_grid = isinstance(simulate_side, Grid)
    if solve_is_grid and simulate_is_grid:
        message = (
            f"states['{name}']: `Phased(solve=Grid, simulate=Grid)` has no "
            f"semantics — identical grids are a bare Grid, and differing "
            f"solve/simulate grid domains are not supported."
        )
    elif solve_is_grid and callable(simulate_side):
        message = (
            f"states['{name}']: a state that is a grid in the solve phase but "
            f"derived in the simulate phase is not yet supported."
        )
    elif callable(solve_side) and simulate_is_grid:
        grid = cast("Grid", simulate_side)
        if grid.batch_size > 0 or grid.distributed:
            message = (
                f"states['{name}']: the grid of a carried state is the "
                f"simulate-phase domain of a per-subject value — `batch_size` "
                f"and `distributed` apply only to solve grid axes and must "
                f"not be set on it."
            )
        else:
            return cast("UserFunction", solve_side), grid, []
    elif callable(solve_side) and callable(simulate_side):
        message = (
            f"states['{name}']: a function derived in both phases belongs in "
            f"`functions`, not `states`."
        )
    else:
        message = (
            f"states['{name}']: `Phased` state variants must be a callable "
            f"(solve) and an LCM grid (simulate), got solve={solve_side!r}, "
            f"simulate={simulate_side!r}."
        )
    return None, None, [message]


def _split_state_transitions(
    *, user_regime: lcm.regime.Regime
) -> tuple[
    dict[StateName, _PhaseStateTransition], dict[StateName, _PhaseStateTransition]
]:
    """Split `state_transitions` into per-phase mappings."""
    solve_state_transitions: dict[StateName, _PhaseStateTransition] = {}
    simulate_state_transitions: dict[StateName, _PhaseStateTransition] = {}
    for name, value in user_regime.state_transitions.items():
        if isinstance(value, Phased):
            solve_state_transitions[name] = cast("_PhaseStateTransition", value.solve)
            simulate_state_transitions[name] = cast(
                "_PhaseStateTransition", value.simulate
            )
        else:
            # `Phased` inside a per-target dict is rejected by validation, so
            # the bare value is phase-invariant.
            solve_state_transitions[name] = cast("_PhaseStateTransition", value)
            simulate_state_transitions[name] = cast("_PhaseStateTransition", value)
    return solve_state_transitions, simulate_state_transitions


def _carried_law_errors(*, name: StateName, law: _PhaseStateTransition) -> list[str]:
    """Validate the law of motion of a carried state declared via `Phased`.

    A carried state is a genuine simulate-phase state, so it needs a
    `state_transitions` entry like any other state (the missing-entry check
    runs with the regular state-transition validation); the supported form is
    a plain deterministic callable, including `fixed_transition` identities.
    """
    if isinstance(law, MarkovTransition):
        return [
            f"State '{name}' is carried only in the simulate phase; a "
            f"stochastic (`MarkovTransition`) law of motion for it is not "
            f"yet supported."
        ]
    if isinstance(law, Mapping):
        return [
            f"State '{name}' is carried only in the simulate phase; a "
            f"per-target dict law of motion for it is not yet supported."
        ]
    return []


def _split_regime_transition(
    *, user_regime: lcm.regime.Regime
) -> tuple[_PhaseRegimeTransition, _PhaseRegimeTransition, list[str]]:
    """Split the regime `transition` into per-phase variants.

    Returns the solve-phase variant, the simulate-phase variant, and the
    grammar violations found along the way.
    """
    raw = user_regime.transition
    if not isinstance(raw, Phased):
        return (
            cast("_PhaseRegimeTransition", raw),
            cast("_PhaseRegimeTransition", raw),
            [],
        )
    errors: list[str] = []
    sides = (("solve", raw.solve), ("simulate", raw.simulate))
    for phase_label, side in sides:
        if side is None:
            errors.append(
                "Regime transition variants cannot be `None` — terminality is "
                "phase-invariant; use `transition=None` for a terminal regime."
            )
        elif not callable(side) and not isinstance(side, Mapping):
            errors.append(
                f"Regime transition {phase_label} variant must be a callable, "
                f"`MarkovTransition`, or a per-target dict, got {side!r}."
            )
    if not errors:
        solve_granular = isinstance(raw.solve, Mapping)
        simulate_granular = isinstance(raw.simulate, Mapping)
        if solve_granular != simulate_granular or (
            not solve_granular
            and isinstance(raw.solve, MarkovTransition)
            != isinstance(raw.simulate, MarkovTransition)
        ):
            errors.append(
                "Regime transition variants must have matching forms: both "
                "coarse with matching stochasticity, or both per-target dicts."
            )
        elif solve_granular and set(cast("Mapping", raw.solve)) != set(
            cast("Mapping", raw.simulate)
        ):
            errors.append(
                "Per-target regime transition variants must declare identical "
                "key sets — phase-variant reachability would let the "
                "simulation realize a jump into a regime whose continuation "
                "value was never planned over."
            )
    return (
        cast("_PhaseRegimeTransition", raw.solve),
        cast("_PhaseRegimeTransition", raw.simulate),
        errors,
    )
