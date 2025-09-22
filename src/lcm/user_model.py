"""Collection of classes that are used by the user to define the model and grids."""

from __future__ import annotations

import dataclasses
from dataclasses import KW_ONLY, dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

from lcm.exceptions import ModelInitilizationError, format_messages
from lcm.grids import Grid
from lcm.logging import get_logger
from lcm.model_initialization import initialize_regime_components
from lcm.simulation.simulate import simulate
from lcm.solution.solve_brute import solve

if TYPE_CHECKING:
    import pandas as pd
    from jax import Array

    from lcm.interfaces import InternalModel, StateActionSpace, StateSpaceInfo
    from lcm.typing import (
        ArgmaxQOverAFunction,
        FloatND,
        MaxQOverAFunction,
        ParamsDict,
        UserFunction,
    )


@dataclass(frozen=True, kw_only=True)
class Regime:
    """A modular component defining a consistent state-action space and functions.

    Each Regime represents a distinct behavioral environment where the agent
    has a specific set of available states, actions, and functions.

    Args:
        name: Unique identifier for this regime.
        description: Optional description of what this regime represents.
        active: Range of periods when this regime is active. If None, the regime
            will be active in all periods (requires Model.n_periods to be specified).
        actions: Dictionary of action variables and their grids for this regime.
        states: Dictionary of state variables and their grids for this regime.
        functions: Dictionary of functions specific to this regime.
        regime_transitions: Dictionary mapping target regime names to
            transition functions.
    """

    name: str
    description: str | None = None
    active: range | None = None
    actions: dict[str, Grid] = field(default_factory=dict)
    states: dict[str, Grid] = field(default_factory=dict)
    functions: dict[str, UserFunction] = field(default_factory=dict)
    regime_transitions: dict[str, Callable[..., Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_attribute_types(self)
        _validate_logical_consistency(self)
        initialize_regime_components(regime=self)


@dataclass(frozen=True)
class Model:
    """A user model which can be processed into an internal model.

    Supports regime-based models (new API) and legacy single-regime models.

    The interaction between n_periods and Regime.active works as follows:

    - If n_periods is None: Derives total periods from max(regime.active.stop)
      across all regimes. At least one regime must have active specified.
    - If n_periods is not None: Validates that regime.active ranges don't exceed
      n_periods. Regimes with active=None will be set to active in all periods.

    Args:
        regimes: List of Regime objects defining the model structure.
        n_periods: Total number of periods. If None, derived from regime active ranges.
        description: Optional description of the model.
        enable_jit: Whether to enable JIT compilation (default: True).

    Legacy args (deprecated):
        actions: Dictionary of action variables (single-regime models only).
        states: Dictionary of state variables (single-regime models only).
        functions: Dictionary of functions (single-regime models only).
    """

    regimes: Regime | list[Regime]
    _: KW_ONLY
    n_periods: int
    description: str | None = None

    enable_jit: bool = True

    # Computed model components (set in __post_init__)
    internal_model: InternalModel = field(init=False)
    params_template: ParamsDict = field(init=False)
    state_action_spaces: dict[int, StateActionSpace] = field(init=False)
    state_space_infos: dict[int, StateSpaceInfo] = field(init=False)
    max_Q_over_a_functions: dict[int, MaxQOverAFunction] = field(init=False)
    argmax_and_max_Q_over_a_functions: dict[int, ArgmaxQOverAFunction] = field(
        init=False
    )

    # Additional computed components for regime models
    regime_transition_dag: dict[str, dict[str, Callable[..., Any]]] = field(init=False)
    next_regime_state_function: Callable[..., Any] | None = field(init=False)

    def __post_init__(self) -> None:
        if isinstance(self.regimes, Regime):
            object.__setattr__(self, "regimes", [self.regimes])
        _validate_regime_period_coverage(regimes=self.regimes, n_periods=self.n_periods)
        if len(self.regimes) == 1:
            self._initialize_regime_model()

    def _initialize_regime_model(self) -> None:
        """Initialize regime-based model."""
        if not self.regimes:
            raise ModelInitilizationError("Regime model must have at least one regime")

        # Step 1: Determine n_periods based on interaction logic
        n_periods = self.n_periods
        if n_periods is None:
            regimes_with_active = [r for r in self.regimes if r.active is not None]
            if not regimes_with_active:
                raise ModelInitilizationError(
                    "When n_periods is None, at least one regime must have "
                    "an active range specified"
                )
            n_periods = max(
                regime.active.stop
                for regime in regimes_with_active
                if regime.active is not None
            )
            object.__setattr__(self, "n_periods", n_periods)
        else:
            # Update regimes with active=None to be active in all periods
            for regime in self.regimes:
                if regime.active is None:
                    object.__setattr__(regime, "active", range(n_periods))
            # Validate that explicit active ranges align with n_periods
            for regime in self.regimes:
                if regime.active is not None and regime.active.stop > n_periods:
                    raise ModelInitilizationError(
                        f"Regime '{regime.name}' has active range extending "
                        f"beyond n_periods ({regime.active.stop} > {n_periods})"
                    )

        # Step 2: Validate regime coverage
        self._validate_regime_coverage(n_periods)

        # Initialize regime transition components (placeholder for now)
        object.__setattr__(self, "regime_transition_dag", {})
        object.__setattr__(self, "next_regime_state_function", None)

        raise NotImplementedError("Regime models are not yet fully implemented")

    def solve(
        self,
        params: ParamsDict,
        *,
        debug_mode: bool = True,
    ) -> dict[int, FloatND]:
        """Solve the model using the pre-computed functions.

        Args:
            params: Model parameters matching the template from self.params_template
            debug_mode: Whether to enable debug logging

        Returns:
            Dictionary mapping period to value function arrays
        """
        return solve(
            params=params,
            state_action_spaces=self.state_action_spaces,
            max_Q_over_a_functions=self.max_Q_over_a_functions,
            logger=get_logger(debug_mode=debug_mode),
        )

    def simulate(
        self,
        params: ParamsDict,
        initial_states: dict[str, Array],
        V_arr_dict: dict[int, FloatND],
        *,
        additional_targets: list[str] | None = None,
        seed: int | None = None,
        debug_mode: bool = True,
    ) -> pd.DataFrame:
        """Simulate the model forward using pre-computed functions.

        Args:
            params: Model parameters
            initial_states: Initial state values
            V_arr_dict: Value function arrays from solve()
            additional_targets: Additional targets to compute
            seed: Random seed
            debug_mode: Whether to enable debug logging

        Returns:
            Simulation results as DataFrame
        """
        logger = get_logger(debug_mode=debug_mode)

        return simulate(
            params=params,
            initial_states=initial_states,
            argmax_and_max_Q_over_a_functions=self.argmax_and_max_Q_over_a_functions,
            internal_model=self.internal_model,
            logger=logger,
            V_arr_dict=V_arr_dict,
            additional_targets=additional_targets,
            seed=seed,
        )

    def solve_and_simulate(
        self,
        params: ParamsDict,
        initial_states: dict[str, Array],
        *,
        additional_targets: list[str] | None = None,
        seed: int | None = None,
        debug_mode: bool = True,
    ) -> pd.DataFrame:
        """Solve and then simulate the model in one call.

        Args:
            params: Model parameters
            initial_states: Initial state values
            additional_targets: Additional targets to compute
            seed: Random seed
            debug_mode: Whether to enable debug logging

        Returns:
            Simulation results as DataFrame
        """
        V_arr_dict = self.solve(params, debug_mode=debug_mode)
        return self.simulate(
            params=params,
            initial_states=initial_states,
            V_arr_dict=V_arr_dict,
            additional_targets=additional_targets,
            seed=seed,
            debug_mode=debug_mode,
        )

    def replace(self, **kwargs: Any) -> Model:  # noqa: ANN401
        """Replace the attributes of the model.

        Args:
            **kwargs: Keyword arguments to replace the attributes of the model.

        Returns:
            A new model with the replaced attributes.

        """
        try:
            return dataclasses.replace(self, **kwargs)
        except TypeError as e:
            raise ModelInitilizationError(
                f"Failed to replace attributes of the model. The error was: {e}"
            ) from e


def _validate_attribute_types(regime: Regime) -> None:  # noqa: C901
    """Validate the types of the model attributes."""
    error_messages = []

    # Validate types of states and actions
    # ----------------------------------------------------------------------------------
    for attr_name in ("actions", "states"):
        attr = getattr(regime, attr_name)
        if isinstance(attr, dict):
            for k, v in attr.items():
                if not isinstance(k, str):
                    error_messages.append(f"{attr_name} key {k} must be a string.")
                if not isinstance(v, Grid):
                    error_messages.append(f"{attr_name} value {v} must be an LCM grid.")
        else:
            error_messages.append(f"{attr_name} must be a dictionary.")

    # Validate types of functions
    # ----------------------------------------------------------------------------------
    if isinstance(regime.functions, dict):
        for k, v in regime.functions.items():
            if not isinstance(k, str):
                error_messages.append(f"function keys must be a strings, but is {k}.")
            if not callable(v):
                error_messages.append(
                    f"function values must be a callable, but is {v}."
                )
    else:
        error_messages.append("functions must be a dictionary.")

    if error_messages:
        msg = format_messages(error_messages)
        raise ModelInitilizationError(msg)


def _validate_logical_consistency(regime: Regime) -> None:
    """Validate the logical consistency of the regime."""
    error_messages = []

    if regime.active is not None and not isinstance(regime.active, range):
        error_messages.append("Active must be a range object or None.")

    if "utility" not in regime.functions:
        error_messages.append(
            "Utility function is not defined. LCM expects a function called 'utility' "
            "in the functions dictionary.",
        )

    states_without_next_func = [
        state for state in regime.states if f"next_{state}" not in regime.functions
    ]
    if states_without_next_func:
        error_messages.append(
            "Each state must have a corresponding next state function. For the "
            "following states, no next state function was found: "
            f"{states_without_next_func}.",
        )

    states_and_actions_overlap = set(regime.states) & set(regime.actions)
    if states_and_actions_overlap:
        error_messages.append(
            "States and actions cannot have overlapping names. The following names "
            f"are used in both states and actions: {states_and_actions_overlap}.",
        )

    if error_messages:
        msg = format_messages(error_messages)
        raise ModelInitilizationError(msg)


def _validate_regime_period_coverage(regimes: list[Regime], n_periods: int) -> None:
    """Validate that regimes cover all periods."""
    all_periods: set[int] = set()

    for regime in regimes:
        if regime is not None:
            all_periods.update(set(regime.active))

    expected_periods = set(range(n_periods))
    if all_periods != expected_periods:
        missing = expected_periods - all_periods
        extra = all_periods - expected_periods
        msg = (
            f"Regime period coverage mismatch for model with {n_periods=}:\n"
            f"- Missing periods from regimes: {missing}\n"
            f"- Extra periods in regimes: {extra}"
        )
        raise ModelInitilizationError(msg)
