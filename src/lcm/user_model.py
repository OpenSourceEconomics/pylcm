"""Collection of classes that are used by the user to define the model and grids."""

from __future__ import annotations

import dataclasses
import warnings
from dataclasses import KW_ONLY, dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

from lcm.exceptions import ModelInitilizationError, format_messages
from lcm.grids import Grid
from lcm.logging import get_logger
from lcm.model_initialization import initialize_model_components
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
        # Basic validation for now
        if not self.name:
            raise ValueError("name cannot be empty")
        _validate_attribute_types(self)
        _validate_logical_consistency(self)

    def to_model(
        self,
        *,
        n_periods: int | None = None,
        description: str | None = None,
        enable_jit: bool = True,
    ) -> Model:
        """Create a single-regime Model from this Regime.

        This provides a fluent interface for the common case of single-regime models,
        eliminating the need to manually wrap the regime in a list.

        Args:
            n_periods: Number of periods. If None, derived from regime.active.
                If regime.active is also None, an error will be raised.
            description: Optional model description. If None, uses regime.description.
            enable_jit: Whether to enable JIT compilation.

        Returns:
            A Model containing only this regime.

        Example:
            >>> regime = Regime(name="simple", active=range(10), actions={...})
            >>> model = regime.to_model()  # Creates 10-period model
        """
        return Model(
            n_periods=n_periods,
            actions=self.actions,
            states=self.states,
            functions=self.functions,
            description=description or self.description,
            enable_jit=enable_jit,
        )


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

    # Model specification information (provided by the User)
    description: str | None = None
    _: KW_ONLY

    # New regime-based API (preferred)
    regimes: list[Regime] = field(default_factory=list)
    n_periods: int | None = None  # Used by both regime and legacy APIs

    # Legacy single-regime API (with deprecation warning)
    actions: dict[str, Grid] = field(default_factory=dict)
    states: dict[str, Grid] = field(default_factory=dict)
    functions: dict[str, UserFunction] = field(default_factory=dict)

    enable_jit: bool = True

    # Computed model components (set in __post_init__)
    is_regime_model: bool = field(init=False)
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
        # Determine model type
        is_regime_model = bool(self.regimes)
        is_legacy_model = bool(
            self.n_periods or self.actions or self.states or self.functions
        )

        object.__setattr__(self, "is_regime_model", is_regime_model)

        # Handle different model types
        if is_regime_model:
            self._initialize_regime_model()
        elif is_legacy_model:
            self._initialize_legacy_model()
        else:
            raise ModelInitilizationError(
                "Model must specify either regimes or legacy parameters"
            )

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

    def _validate_regime_coverage(self, n_periods: int) -> None:
        """Validate that regimes cover all periods without overlap.

        Args:
            n_periods: The number of periods the model should cover.

        Raises:
            ModelInitilizationError: If regime coverage is invalid.
        """
        all_periods: set[int] = set()

        for regime in self.regimes:
            if regime.active is None:
                raise ModelInitilizationError(
                    f"Regime '{regime.name}' has active=None after resolution - "
                    "this should not happen"
                )

            regime_periods = set(regime.active)
            overlap = all_periods & regime_periods
            if overlap:
                raise ModelInitilizationError(
                    f"Overlapping periods {overlap} between regimes"
                )
            all_periods.update(regime_periods)

        expected_periods = set(range(n_periods))
        if all_periods != expected_periods:
            missing = expected_periods - all_periods
            extra = all_periods - expected_periods
            msg = f"Period coverage mismatch. Missing: {missing}, Extra: {extra}"
            raise ModelInitilizationError(msg)

    def _initialize_legacy_model(self) -> None:
        """Initialize legacy single-regime model."""
        warnings.warn(
            "Legacy Model API lcm.Model(n_periods, actions, states, functions) is "
            "deprecated and will be removed in version 0.1.0. "
            "Use Regime API instead: lcm.Model(regimes=[lcm.Regime(name='default', "
            "active=range(n_periods), ...)])",
            DeprecationWarning,
            stacklevel=3,
        )

        if self.n_periods is None:
            raise ModelInitilizationError("Legacy model must specify n_periods")

        # Original single-regime model path
        initialize_model_components(self)

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
            model=self.internal_model,
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

    # if regime.n_periods is not None and regime.n_periods < 1:
    #     error_messages.append("Number of periods must be a positive integer.")
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
