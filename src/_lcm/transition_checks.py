"""Pre-flight numerical checks on user-supplied transition functions.

Called from `Model.solve()` and `Model.simulate()` before backward induction
runs. Two families:

- **Regime transition probability check** keyed on
  `validate_regime_transitions_all_periods`. Iterates active non-terminal
  regimes across periods, evaluates the regime transition function on the
  Cartesian product of its accepted grid variables, and verifies finiteness,
  [0, 1] range, sum-to-1, and no probability mass to inactive regimes.
  The construction-time graph supplies the allowed target set; state-law
  coverage of retained targets is validated at model build.
- **State transition probability check** keyed on
  `validate_state_transitions_all_periods`. Sweeps every `MarkovTransition`
  state transition (incl. per-target dict entries), evaluates the user
  function on the Cartesian product of the function's accepted grid
  variables, and verifies outcome-axis size, [0, 1] range, and sum-to-1.

Both checks read their policy off the `logger`: `log_level="off"` skips the
check, `"warning"` / `"progress"` log each failure and let the run continue,
`"debug"` raises on the first failure.

These are runtime checks: they need a fully-built `Regime` plus user
`flat_params` and evaluate the transition functions numerically. The
construction-time regime-spec validators (`Regime.__post_init__`, which
inspect grids, signatures, and Python source) are a separate concern.

"""

import inspect
import logging
import struct
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import partial
from math import prod
from types import MappingProxyType
from typing import Any, cast, no_type_check

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from dags.tree import tree_path_from_qname

from _lcm.engine import Regime, StateActionSpace, _StochasticStateTransition
from _lcm.execution.workspace_planning import plan_workspace
from _lcm.processes.grid_resolution import ProcessGridResolver
from _lcm.regime_building.next_state import get_next_stochastic_weights_function
from _lcm.simulation.host_operations import StaticArgument
from _lcm.simulation.memory import SimulationMemory, run_simulation_operation
from _lcm.simulation.operand_placement import place_simulation_arguments
from _lcm.simulation.residency import (
    measure_buffer_footprint,
    resident_bytes_by_device,
    union_buffer_footprints,
)
from _lcm.simulation.value_placement import simulation_value_sharding
from _lcm.transition_plans import LotteryLifetime
from _lcm.typing import FlatParams, FlatRegimeParams, RegimeName, StateOrActionName
from _lcm.utils.logging import raise_or_warn, validation_enabled
from _lcm.utils.namespace import ParamsQnameDepth
from lcm.ages import AgeGrid
from lcm.exceptions import (
    ExecutionPlanningError,
    InvalidRegimeTransitionProbabilitiesError,
    InvalidStateTransitionProbabilitiesError,
    RegimeInitializationError,
)
from lcm.typing import BoolND, FloatND, IntND, ScalarFloat, ScalarInt

_NO_EXTRA_GRIDS: Mapping[StateOrActionName, FloatND | IntND] = MappingProxyType({})
type _RegimeProbabilityOutput = tuple[
    Mapping[RegimeName, FloatND], Mapping[StateOrActionName, FloatND | IntND]
]
type _SupportSchema = tuple[str, int, object, tuple[tuple[tuple[int, ...], str], ...]]


class _SerialValidationRequired(Exception):  # noqa: N818
    """Select the original diagnostic route before publishing any warning."""


@dataclass(frozen=True, kw_only=True)
class _StateProbabilitySummary:
    """A call-owned reduced result; never retain the probability grid itself."""

    shape: tuple[int, ...]
    flag: jax.Array
    bound_inputs: tuple[object, ...] = field(repr=False)
    """Keep identity-keyed immutable operands alive until the summary closes."""


@dataclass(kw_only=True)
class _ValidationSummary:
    """Call-owned flags and bindings, never validity retained across calls."""

    memory: SimulationMemory | None = None
    process_grid_resolver: ProcessGridResolver | None = None
    flags: list[jax.Array] = field(default_factory=list)
    spaces: dict[tuple[RegimeName, tuple[tuple[str, int], ...]], StateActionSpace] = (
        field(default_factory=dict, repr=False)
    )
    state_probabilities: dict[tuple[object, ...], _StateProbabilitySummary] = field(
        default_factory=dict, repr=False
    )

    def state_action_space(
        self, *, regime: Regime, params: FlatRegimeParams
    ) -> StateActionSpace:
        """Reuse one concrete space only within this call and exact binding."""
        key = (
            regime.name,
            tuple(sorted((name, id(value)) for name, value in params.items())),
        )
        if key not in self.spaces:
            space = regime.solution.state_action_space(
                regime_params=params, process_grid_resolver=self.process_grid_resolver
            )
            self.spaces[key] = space
            if self.memory is not None:
                self.memory.hold(tree=(space.states, space.actions))
        return self.spaces[key]

    def append(
        self,
        *,
        function: Callable[..., jax.Array],
        arguments: Mapping[str, object],
        static_arguments: Mapping[str, StaticArgument] = MappingProxyType({}),
    ) -> None:
        """Admit and retain only one check's reduced output."""
        self.flags.append(
            run_simulation_operation(
                memory=self.memory,
                function=function,
                arguments=arguments,
                static_arguments=static_arguments,
            )
        )

    def valid(self) -> bool:
        """Read the packed reduced flags once on the valid numerical path."""
        if not self.flags:
            return True
        packed = run_simulation_operation(
            memory=self.memory,
            function=_pack_validation_flags,
            arguments={"flags": tuple(self.flags)},
        )
        return not np.asarray(packed).any()

    def close(self) -> None:
        """Release all concrete temporary roots after validation completes."""
        jax.block_until_ready(self.flags)
        self.flags.clear()
        self.spaces.clear()
        self.state_probabilities.clear()
        if self.memory is not None:
            self.memory.close_unit()


@jax.jit
def _pack_validation_flags(*, flags: tuple[jax.Array, ...]) -> jax.Array:
    """Concatenate reduced flags in legacy diagnostic order."""
    return jnp.concatenate([jnp.atleast_1d(flag).astype(jnp.int32) for flag in flags])


@partial(jax.jit, static_argnames=("inactive_indices",))
def _regime_probability_flags(
    *, probabilities: tuple[jax.Array, ...], inactive_indices: tuple[int, ...]
) -> jax.Array:
    """Use exactly the regime validator's existing numerical predicates."""
    all_probs = jnp.stack(probabilities)
    return jnp.stack(
        (
            jnp.any(~jnp.isfinite(all_probs)),
            jnp.any((all_probs < 0) | (all_probs > 1)),
            jnp.any(_unit_mass_violations(jnp.sum(all_probs, axis=0))),
            *(jnp.any(probabilities[index] > 0) for index in inactive_indices),
        )
    )


@jax.jit
def _state_probability_flags(*, probabilities: jax.Array) -> jax.Array:
    """Keep the separate state-law mass rule, including its existing rtol."""
    return jnp.stack(
        (
            jnp.any((probabilities < 0) | (probabilities > 1)),
            ~jnp.allclose(jnp.sum(probabilities, axis=-1), 1.0, atol=1e-6),
        )
    )


@jax.jit
def _joint_probability_flags(*, probabilities: jax.Array) -> jax.Array:
    """Reduce joint numerical failures without retaining the probability grid."""
    return jnp.stack(
        (
            jnp.any(~jnp.isfinite(probabilities)),
            jnp.any((probabilities < 0) | (probabilities > 1)),
            jnp.any(_unit_mass_violations(jnp.sum(probabilities, axis=-1))),
        )
    )


@jax.jit
def _support_finiteness_flags(*, leaves: tuple[jax.Array, ...]) -> jax.Array:
    """Report each concrete support leaf without reading its payload on the host."""
    return jnp.stack([~jnp.all(jnp.isfinite(leaf)) for leaf in leaves])


def validate_transitions(
    *,
    regimes: MappingProxyType[RegimeName, Regime],
    flat_params: FlatParams,
    ages: AgeGrid,
    logger: logging.Logger,
    summary: _ValidationSummary | None = None,
    process_grid_resolver: ProcessGridResolver | None = None,
) -> None:
    """Validate regime and state transition probabilities before solve / simulate.

    Runs the regime-transition check then the state-transition check. Both
    self-gate on the logger's runtime-validation policy (`log_level="off"`
    skips, `"warning"` / `"progress"` warn, `"debug"` raises).

    Args:
        regimes: Immutable mapping of regime names to regimes.
        flat_params: Immutable mapping of regime names to flat parameter mappings.
        ages: Age grid for the model.
        logger: Logger carrying the runtime-validation policy.

    """
    if not validation_enabled(logger):
        return
    if summary is not None:
        _validate_transition_sequence(
            regimes=regimes,
            flat_params=flat_params,
            ages=ages,
            logger=logger,
            summary=summary,
            process_grid_resolver=process_grid_resolver,
        )
        return
    pending = _ValidationSummary(process_grid_resolver=process_grid_resolver)
    try:
        try:
            _validate_transition_sequence(
                regimes=regimes,
                flat_params=flat_params,
                ages=ages,
                logger=logger,
                summary=pending,
                process_grid_resolver=process_grid_resolver,
            )
            accepted = pending.valid()
        except ExecutionPlanningError, MemoryError, jax.errors.JaxRuntimeError:
            raise
        # Speculation publishes nothing; the serial retry preserves the first
        # user-law diagnostic while resource failures above propagate directly.
        except Exception:  # noqa: BLE001
            accepted = False
    finally:
        pending.close()
    if not accepted:
        _validate_transition_sequence(
            regimes=regimes,
            flat_params=flat_params,
            ages=ages,
            logger=logger,
            summary=None,
            process_grid_resolver=process_grid_resolver,
        )


def _validate_transition_sequence(
    *,
    regimes: MappingProxyType[RegimeName, Regime],
    flat_params: FlatParams,
    ages: AgeGrid,
    logger: logging.Logger,
    summary: _ValidationSummary | None,
    process_grid_resolver: ProcessGridResolver | None = None,
    simulation_memory: SimulationMemory | None = None,
) -> None:
    """Preserve family and item ordering for collection and serial diagnostics."""
    memory = (
        summary.memory
        if summary is not None and summary.memory is not None
        else simulation_memory
    )
    validate_regime_transitions_all_periods(
        regimes=regimes,
        flat_params=flat_params,
        ages=ages,
        logger=logger,
        summary=summary,
        process_grid_resolver=process_grid_resolver,
        memory=memory,
    )
    validate_state_transitions_all_periods(
        regimes=regimes,
        flat_params=flat_params,
        ages=ages,
        logger=logger,
        summary=summary,
        process_grid_resolver=process_grid_resolver,
        memory=memory,
    )
    validate_joint_transitions_all_periods(
        regimes=regimes,
        flat_params=flat_params,
        ages=ages,
        logger=logger,
        summary=summary,
        process_grid_resolver=process_grid_resolver,
        memory=memory,
    )


def _params_callable_for_state_transition(
    *,
    regime: Regime,
    flat_params_for_regime: FlatRegimeParams,
    transition: _StochasticStateTransition,
) -> FlatRegimeParams:
    """Return un-qualified params for calling a state-transition function.

    Both `regime.resolved_fixed_params` and `flat_params_for_regime` key
    every transition-law param granularly (`<target>__next_<state>__<param>`),
    matching the engine's target-prefixed function qnames:

    - per-target dicts ⇒ one entry per target, possibly distinct values
    - coarse laws      ⇒ one entry per reachable carrying target, all
      sharing the same leaf — any target's binding yields the law's params

    The `MarkovTransition`'s user function is called with the raw
    parameter names from its signature, so the validator must strip
    the same qualifier before lookup. Without the strip, every
    transition-function parameter that isn't a grid axis falls through
    to the "not numerically validated" skip branch and the
    per-transition numerical check never runs.
    """
    merged = {**regime.resolved_fixed_params, **flat_params_for_regime}

    if transition.target_regime_name is None:
        # Coarse law: prefer any target's shared-leaf granular binding.
        law_name = f"next_{transition.state_name}"
        parts_by_name = {name: tree_path_from_qname(name) for name in merged}
        granular = {
            parts[2]: merged[name]
            for name, parts in parts_by_name.items()
            if len(parts) == ParamsQnameDepth.TARGETREGIME__FUNC__PARAM
            and parts[1] == law_name
        }
        if granular:
            return MappingProxyType(granular)

        # A law with no temporally retained carrying target keeps its original
        # template-qualified binding. It may still be checked as a user declaration.
        prefix = f"{law_name}__"
        return MappingProxyType(
            {
                name.removeprefix(prefix): value
                for name, value in merged.items()
                if name.startswith(prefix)
            }
        )

    prefix = f"{transition.target_regime_name}__next_{transition.state_name}__"
    return MappingProxyType(
        {
            name.removeprefix(prefix): value
            for name, value in merged.items()
            if name.startswith(prefix)
        }
    )


def validate_regime_transitions_all_periods(
    *,
    regimes: MappingProxyType[RegimeName, Regime],
    flat_params: FlatParams,
    ages: AgeGrid,
    logger: logging.Logger,
    summary: _ValidationSummary | None = None,
    process_grid_resolver: ProcessGridResolver | None = None,
    memory: SimulationMemory | None = None,
) -> None:
    """Validate regime transition probabilities for all periods before solve.

    For each period (except the last), for each active non-terminal regime, evaluate
    the regime transition function on all grid points and check that inactive regimes
    receive zero probability.

    Args:
        regimes: Immutable mapping of regime names to regimes.
        flat_params: Immutable mapping of regime names to flat parameter mappings.
        ages: Age grid for the model.
        logger: Logger carrying the runtime-validation policy. `log_level="off"`
            returns immediately; `"warning"` / `"progress"` log each failure and
            continue; `"debug"` raises on the first failure.

    Raises:
        InvalidRegimeTransitionProbabilitiesError: If a regime transition produces
            invalid probabilities and the logger implies raise mode.

    """
    # Skipped entirely at `log_level="off"`. What that costs is the diagnosis
    # rather than the answer: the continuation aggregator measures the mass its
    # retained targets represent and returns NaN unless it is one and no weight
    # is negative, so a misspecification survives as a NaN rather than as a
    # plausible number. These checks name the regime, the period and the
    # offending target instead, which a NaN cannot.
    if not validation_enabled(logger):
        return

    last_period = ages.n_periods - 1
    non_terminal_active_at_last = [
        regime_name
        for regime_name, regime in regimes.items()
        if not regime.terminal and last_period in regime.active_periods
    ]
    if non_terminal_active_at_last:
        if summary is not None:
            raise _SerialValidationRequired
        raise_or_warn(
            logger=logger,
            error=InvalidRegimeTransitionProbabilitiesError(
                f"Non-terminal regime(s) {non_terminal_active_at_last} are active at "
                f"the last period (age {ages.exact_values[last_period]}). Non-terminal "
                "regimes must not be active at the last period because there is no "
                "next period to transition to. Adjust the 'active' function on these "
                "regimes to exclude the last age."
            ),
        )

    for period in range(ages.n_periods - 1):
        for regime_name, regime in regimes.items():
            if period not in regime.active_periods:
                continue
            if regime.terminal:
                continue

            try:
                _validate_regime_transition_single(
                    regimes=regimes,
                    regime_params=flat_params[regime_name],
                    active_regimes_next_period=(
                        regime.solution.reachability.targets(
                            period=period, source=regime_name
                        )
                    ),
                    regime_name=regime_name,
                    period=period,
                    ages=ages,
                    summary=summary,
                    process_grid_resolver=process_grid_resolver,
                    memory=memory,
                )
            except InvalidRegimeTransitionProbabilitiesError as error:
                if summary is not None:
                    raise _SerialValidationRequired from error
                raise_or_warn(logger=logger, error=error)


def _validate_regime_transition_single(
    *,
    regimes: MappingProxyType[RegimeName, Regime],
    regime_params: FlatRegimeParams,
    active_regimes_next_period: tuple[RegimeName, ...],
    regime_name: RegimeName,
    period: int,
    ages: AgeGrid,
    summary: _ValidationSummary | None = None,
    process_grid_resolver: ProcessGridResolver | None = None,
    memory: SimulationMemory | None = None,
) -> None:
    """Validate regime transition probabilities for a single regime and period.

    Evaluate the regime transition function on the Cartesian product of all grid
    variables it accepts, using `jax.vmap` for vectorised evaluation.

    """
    regime = regimes[regime_name]
    # Non-None guaranteed: only called for non-terminal regimes
    regime_transition_func = regime.solution.validation_regime_transition_probs

    state_action_space = (
        regime.solution.state_action_space(
            regime_params=regime_params, process_grid_resolver=process_grid_resolver
        )
        if summary is None
        else summary.state_action_space(regime=regime, params=regime_params)
    )

    # Filter params to only those accepted by the transition function
    accepted_params = set(inspect.signature(regime_transition_func).parameters)  # ty: ignore[invalid-argument-type]
    filtered_params = {k: v for k, v in regime_params.items() if k in accepted_params}

    # Collect only grid variables the transition function accepts
    grids: dict[StateOrActionName, FloatND | IntND] = {
        k: v for k, v in state_action_space.states.items() if k in accepted_params
    } | {k: v for k, v in state_action_space.actions.items() if k in accepted_params}

    # Pin to int32: a Python-int `period` traced through `jax.vmap` becomes
    # int64 under x64, breaking any int32 `period` contract downstream.
    period_int32 = jnp.int32(period)
    current_memory = (
        summary.memory if summary is not None and summary.memory is not None else memory
    )
    regime_transition_probs, point = _evaluate_regime_probability_law(
        func=cast(
            "Callable[..., Mapping[RegimeName, FloatND]]", regime_transition_func
        ),
        grid_args=MappingProxyType(grids),
        scalar_kwargs=MappingProxyType(
            {
                **filtered_params,
                "period": period_int32,
                "age": ages.values[period],  # noqa: PD011
            }
        ),
        memory=current_memory,
    )
    _check_and_release_regime_probability(
        regime_transition_probs=regime_transition_probs,
        active_regimes_next_period=active_regimes_next_period,
        regime_name=regime_name,
        age=ages.values[period],  # noqa: PD011
        next_age=ages.values[period + 1],  # noqa: PD011
        period=period,
        state_action_values=point,
        summary=summary,
        memory=current_memory,
    )


def _evaluate_regime_probability_law(
    *,
    func: Callable[..., Mapping[RegimeName, FloatND]],
    grid_args: Mapping[StateOrActionName, FloatND | IntND],
    scalar_kwargs: Mapping[str, object],
    memory: SimulationMemory | None,
) -> tuple[
    MappingProxyType[RegimeName, FloatND],
    MappingProxyType[StateOrActionName, FloatND | IntND],
]:
    """Admit one regime-law mapping and its diagnostic grid coordinates."""
    function = partial(
        _regime_probability_law,
        grid_names=tuple(grid_args),
        func=func,
    )
    arguments = MappingProxyType(
        {"grid_args": grid_args, "scalar_kwargs": scalar_kwargs}
    )
    produced = (
        function(grid_args=grid_args, scalar_kwargs=scalar_kwargs)
        if memory is None
        else _evaluate_admitted_transition_producer(
            function=function,
            arguments=arguments,
            memory=memory,
        )
    )
    probabilities, point = cast("_RegimeProbabilityOutput", produced)
    return MappingProxyType(probabilities), MappingProxyType(point)


def _regime_probability_law(
    *,
    grid_args: Mapping[StateOrActionName, FloatND | IntND],
    scalar_kwargs: Mapping[str, object],
    grid_names: tuple[StateOrActionName, ...],
    func: Callable[..., Mapping[RegimeName, FloatND]],
) -> tuple[Mapping[RegimeName, FloatND], Mapping[StateOrActionName, FloatND | IntND]]:
    """Evaluate one regime law and retain its Cartesian diagnostic coordinates."""
    if not grid_names:
        return func(**scalar_kwargs), {}
    mesh = jnp.meshgrid(*(grid_args[name] for name in grid_names), indexing="ij")
    flat_arrays = tuple(array.ravel() for array in mesh)
    probabilities = jax.vmap(
        _GridPointCall(names=grid_names, scalar_kwargs=scalar_kwargs, func=func)
    )(*flat_arrays)
    return probabilities, dict(zip(grid_names, flat_arrays, strict=True))


def _check_and_release_regime_probability(
    *,
    regime_transition_probs: MappingProxyType[RegimeName, FloatND],
    active_regimes_next_period: tuple[RegimeName, ...],
    regime_name: RegimeName,
    age: float | ScalarInt | ScalarFloat,
    next_age: float | ScalarInt | ScalarFloat,
    period: int,
    state_action_values: MappingProxyType[StateOrActionName, FloatND | IntND],
    summary: _ValidationSummary | None,
    memory: SimulationMemory | None,
) -> None:
    """Own completed regime-law outputs through their admitted validation."""
    _set_transition_outputs(
        memory=memory, outputs=(regime_transition_probs, state_action_values)
    )
    try:
        _validate_regime_transition_probs(
            regime_transition_probs=regime_transition_probs,
            active_regimes_next_period=active_regimes_next_period,
            regime_name=regime_name,
            age=age,
            next_age=next_age,
            period=period,
            state_action_values=state_action_values,
            summary=summary,
            memory=memory,
        )
    finally:
        _set_transition_outputs(memory=memory, outputs=())


def _validate_regime_transition_probs(
    *,
    regime_transition_probs: MappingProxyType[RegimeName, FloatND],
    active_regimes_next_period: tuple[RegimeName, ...],
    regime_name: RegimeName,
    age: float | ScalarInt | ScalarFloat,
    next_age: float | ScalarInt | ScalarFloat,
    period: int | None = None,
    state_action_values: MappingProxyType[StateOrActionName, FloatND | IntND]
    | None = None,
    summary: _ValidationSummary | None = None,
    memory: SimulationMemory | None = None,
) -> None:
    """Validate regime transition probabilities.

    Check that probabilities are finite, sum to 1 across all regimes, and that
    inactive regimes have zero probability.

    Args:
        regime_transition_probs: Immutable mapping of regime names to probability
            arrays.
        active_regimes_next_period: Tuple of regime names active in the next period.
        regime_name: Name of the source regime (for error messages).
        age: Current age (for error messages).
        next_age: Next age (for error messages).
        period: Optional source-period index for graph diagnostics.
        state_action_values: Optional immutable mapping of state/action names to arrays,
            included in error messages to help diagnose which inputs cause violations.

    Raises:
        InvalidRegimeTransitionProbabilitiesError: If probabilities are non-finite,
            outside [0, 1], don't sum to 1, or assign positive probability to inactive
            regimes.

    """
    names = tuple(regime_transition_probs)
    inactive = tuple(set(names) - set(active_regimes_next_period))
    flag_arguments = {"probabilities": tuple(regime_transition_probs.values())}
    static_arguments = {
        "inactive_indices": tuple(names.index(name) for name in inactive)
    }
    if summary is not None:
        summary.append(
            function=_regime_probability_flags,
            arguments=flag_arguments,
            static_arguments=static_arguments,
        )
        return
    flags = (
        _regime_probability_flags(**flag_arguments, **static_arguments)
        if memory is None
        else run_simulation_operation(
            memory=memory,
            function=_regime_probability_flags,
            arguments=flag_arguments,
            static_arguments=static_arguments,
        )
    )
    nonfinite, outside_bounds, invalid_mass, *inactive_flags = np.asarray(
        flags
    ).tolist()
    if nonfinite:
        raise InvalidRegimeTransitionProbabilitiesError(
            f"Non-finite values in regime transition probabilities from "
            f"'{regime_name}' between ages {age} and {next_age}. Check the "
            f"'next_regime' function of the '{regime_name}' regime."
        )

    if outside_bounds:
        raise InvalidRegimeTransitionProbabilitiesError(
            f"Regime transition probabilities from '{regime_name}' between ages {age} "
            f"and {next_age} contain values outside [0, 1]. Check the 'next_regime' "
            f"function of the '{regime_name}' regime."
        )

    if invalid_mass:
        sum_all = jnp.sum(jnp.stack(list(regime_transition_probs.values())), axis=0)
        detail = _format_sum_violation(
            sum_all=sum_all,
            state_action_values=state_action_values,
        )
        raise InvalidRegimeTransitionProbabilitiesError(
            f"Regime transition probabilities from '{regime_name}' between ages {age} "
            f"and {next_age} do not sum to 1.0. {detail}\n"
            f"Check the 'next_regime' function of the '{regime_name}' regime."
        )

    for r, has_mass in zip(inactive, inactive_flags, strict=True):
        if has_mass:
            period_detail = "" if period is None else f" in period {period}"
            raise InvalidRegimeTransitionProbabilitiesError(
                f"Regime '{r}' is inactive at age {next_age} but has positive "
                f"transition probability from '{regime_name}' between ages {age} and "
                f"{next_age}{period_detail}. Its mass is not represented in the "
                f"continuation, so what the remaining targets carry is less than "
                f"unit mass and the solve returns NaN rather than a value that "
                f"does not depend on '{r}' at all. Either make '{r}' active at "
                f"that age or give it probability 0 there."
            )


def _format_sum_violation(
    *,
    sum_all: FloatND,
    state_action_values: MappingProxyType[StateOrActionName, FloatND | IntND]
    | None = None,
) -> str:
    """Format a human-readable description of probability sum violations.

    Args:
        sum_all: Array of probability sums (per-subject).
        state_action_values: Optional immutable mapping of state/action names to arrays,
            included in the output to show which inputs cause violations.

    Returns:
        Formatted string describing which sums violate the sum-to-1 constraint.

    """
    sum_all = jnp.atleast_1d(sum_all)
    if state_action_values is not None:
        state_action_values = MappingProxyType(
            {name: jnp.atleast_1d(arr) for name, arr in state_action_values.items()}
        )
    failing_mask = _unit_mass_violations(sum_all)
    failing_indices = jnp.where(failing_mask)[0].astype(jnp.int32)
    failing_sums = sum_all[failing_mask]
    n_failing = int(failing_indices.shape[0])
    n_show = min(n_failing, 5)
    data: dict[str, list[float]] = {
        "subject": failing_indices[:n_show].tolist(),
    }
    if state_action_values is not None:
        for name, arr in state_action_values.items():
            data[name] = [float(arr[i]) for i in failing_indices[:n_show]]
    data["sum"] = failing_sums[:n_show].tolist()
    df = pd.DataFrame(data)
    return (
        f"{n_failing} of {sum_all.shape[0]} probability vectors do not sum to 1.0.\n"
        f"First failing entries:\n{df.to_string(index=False)}"
    )


def validate_state_transitions_all_periods(  # noqa: C901
    *,
    regimes: MappingProxyType[RegimeName, Regime],
    flat_params: FlatParams,
    ages: AgeGrid,
    logger: logging.Logger,
    summary: _ValidationSummary | None = None,
    process_grid_resolver: ProcessGridResolver | None = None,
    memory: SimulationMemory | None = None,
) -> None:
    """Validate every `MarkovTransition` state transition before solve.

    For each non-terminal active period of each active regime, iterate the
    regime's `stochastic_state_transitions` and evaluate each
    `MarkovTransition` function on the Cartesian product of its accepted
    grid variables. Check:

    - The output's last-axis size matches the state's outcome count.
    - All values lie in [0, 1].
    - Rows along the last axis sum to 1.

    Fast-exits when no regime in the model has any stochastic state
    transitions, so models without `MarkovTransition` states pay no cost.

    Args:
        regimes: Immutable mapping of regime names to canonical regimes.
        flat_params: Immutable mapping of regime names to flat parameter
            mappings.
        ages: Age grid for the model.
        logger: Logger carrying the runtime-validation policy. `log_level="off"`
            returns immediately; `"warning"` / `"progress"` log each failure and
            continue; `"debug"` raises on the first failure.

    Raises:
        InvalidStateTransitionProbabilitiesError: If a `MarkovTransition`
            function returns the wrong outcome-axis size, values outside
            [0, 1], or rows that don't sum to 1, and the logger implies raise
            mode.

    """
    if not validation_enabled(logger):
        return
    if not any(r.stochastic_state_transitions for r in regimes.values()):
        return

    for period in range(ages.n_periods - 1):
        for regime_name, regime in regimes.items():
            if period not in regime.active_periods:
                continue
            if regime.terminal:
                continue
            if not regime.stochastic_state_transitions:
                continue

            state_action_space = (
                regime.solution.state_action_space(
                    regime_params=flat_params[regime_name],
                    process_grid_resolver=process_grid_resolver,
                )
                if summary is None
                else summary.state_action_space(
                    regime=regime, params=flat_params[regime_name]
                )
            )
            age = ages.values[period]  # noqa: PD011
            for transition in regime.stochastic_state_transitions.values():
                if _state_transition_unused_in_period(
                    transition=transition,
                    regime=regime,
                    period=period,
                ):
                    continue
                try:
                    _validate_state_transition_single(
                        transition=transition,
                        regime_params=_params_callable_for_state_transition(
                            regime=regime,
                            flat_params_for_regime=flat_params[regime_name],
                            transition=transition,
                        ),
                        state_action_space=state_action_space,
                        regime_name=regime_name,
                        age=age,
                        period=period,
                        logger=logger,
                        summary=summary,
                        memory=memory,
                    )
                except InvalidStateTransitionProbabilitiesError as error:
                    if summary is not None:
                        raise _SerialValidationRequired from error
                    raise_or_warn(logger=logger, error=error)


def validate_joint_transitions_all_periods(
    *,
    regimes: MappingProxyType[RegimeName, Regime],
    flat_params: FlatParams,
    ages: AgeGrid,
    logger: logging.Logger,
    summary: _ValidationSummary | None = None,
    process_grid_resolver: ProcessGridResolver | None = None,
    memory: SimulationMemory | None = None,
) -> None:
    """Validate every transition-local lottery before solve or simulation."""
    if not validation_enabled(logger):
        return

    # A callable support is resolved only after params are bound, so its full
    # pytree signature cannot be checked during ``Regime`` construction.  Compare
    # every active period and both phases in this one preflight instead.  Values
    # may differ, but tree structure, event shapes, and dtypes are a static JIT/AOT
    # contract and must not depend on period or perceived/realized phase.
    support_schemas: dict[tuple[RegimeName, RegimeName, str], _SupportSchema] = {}
    current_memory = (
        summary.memory if summary is not None and summary.memory is not None else memory
    )

    for period in range(ages.n_periods - 1):
        period_int32 = jnp.int32(period)
        age = ages.values[period]  # noqa: PD011
        for regime_name, regime in regimes.items():
            if regime.terminal or period not in regime.active_periods:
                continue
            state_action_space = (
                regime.solution.state_action_space(
                    regime_params=flat_params[regime_name],
                    process_grid_resolver=process_grid_resolver,
                )
                if summary is None
                else summary.state_action_space(
                    regime=regime, params=flat_params[regime_name]
                )
            )
            # A carried state has no solve grid axis, so a simulate-phase law
            # reading one is not resolvable on the solution state-action space.
            # Sweep its simulate-phase domain alongside the solve grids instead.
            carried_only_grids = MappingProxyType(
                {
                    name: regime.simulation.grids[name].to_jax()
                    for name in sorted(regime.simulation.carried_only_state_names)
                }
            )
            for phase_name, phase in (
                ("solve", regime.solution),
                ("simulate", regime.simulation),
            ):
                targets = phase.reachability.targets(period=period, source=regime_name)
                functions = (
                    regime.solution.continuation_functions
                    if phase_name == "solve"
                    else regime.simulation.functions
                )
                for target in targets:
                    plan = phase.transition_plans.get(target)
                    if plan is None:
                        continue
                    joint_laws = {
                        name: lottery
                        for name, lottery in plan.lotteries.items()
                        if lottery.lifetime is LotteryLifetime.TRANSITION_LOCAL
                    }
                    if not joint_laws:
                        continue
                    compute_weights = get_next_stochastic_weights_function(
                        regime_name=target,
                        functions=functions,
                        transitions=phase.transitions[target],
                        transition_plans=phase.transition_plans,
                    )
                    evaluated = _evaluate_joint_weights(
                        func=compute_weights,
                        state_action_space=state_action_space,
                        extra_grids=(
                            carried_only_grids
                            if phase_name == "simulate"
                            else _NO_EXTRA_GRIDS
                        ),
                        regime_params=flat_params[regime_name],
                        period=period_int32,
                        age=age,
                        regime_name=regime_name,
                        phase_name=phase_name,
                        logger=logger,
                        summary=summary,
                        memory=current_memory,
                    )
                    if evaluated is None:
                        continue
                    weights, n_cells = evaluated
                    _validate_joint_laws(
                        joint_laws=joint_laws,
                        transitions=phase.transitions[target],
                        weights=weights,
                        n_cells=n_cells,
                        regime_params=flat_params[regime_name],
                        period=period_int32,
                        period_index=period,
                        age=age,
                        regime_name=regime_name,
                        phase_name=phase_name,
                        target=target,
                        logger=logger,
                        summary=summary,
                        support_schemas=support_schemas,
                        memory=current_memory,
                    )


@contextmanager
def _own_transition_outputs(
    *, memory: SimulationMemory | None, outputs: object, restore: object = ()
) -> Iterator[None]:
    """Publish temporary roots for admitted checks and release them reliably."""
    _set_transition_outputs(memory=memory, outputs=outputs)
    try:
        yield
    finally:
        _set_transition_outputs(memory=memory, outputs=restore)


def _set_transition_outputs(
    *, memory: SimulationMemory | None, outputs: object
) -> None:
    """Expose every temporary mapping leaf to residency accounting."""
    if memory is not None:
        memory.set_derived(_transition_owner_tree(outputs))


def _transition_owner_tree(tree: object) -> object:
    """Convert opaque mapping containers into ordinary JAX pytree nodes."""
    if isinstance(tree, Mapping):
        return {key: _transition_owner_tree(value) for key, value in tree.items()}
    if isinstance(tree, tuple):
        return tuple(_transition_owner_tree(value) for value in tree)
    if isinstance(tree, list):
        return [_transition_owner_tree(value) for value in tree]
    return tree


def _validate_joint_laws(
    *,
    joint_laws: Mapping[str, Any],
    transitions: Mapping[str, Callable[..., Any]],
    weights: Mapping[str, FloatND | IntND],
    n_cells: int | None,
    regime_params: FlatRegimeParams,
    period: ScalarInt,
    period_index: int,
    age: ScalarInt | ScalarFloat,
    regime_name: RegimeName,
    phase_name: str,
    target: RegimeName,
    logger: logging.Logger,
    summary: _ValidationSummary | None,
    support_schemas: dict[tuple[RegimeName, RegimeName, str], _SupportSchema],
    memory: SimulationMemory | None,
) -> None:
    """Validate one target's supports and weights while retaining their owners."""
    with _own_transition_outputs(memory=memory, outputs=weights):
        for kernel_name, law in joint_laws.items():
            support_provider_name = law.support_provider_name
            if support_provider_name is None:
                raise RegimeInitializationError(
                    f"Joint transition {kernel_name!r} has no support provider in "
                    "its canonical plan."
                )
            support = _evaluate_joint_support(
                func=transitions[support_provider_name],
                regime_params=regime_params,
                period=period,
                age=age,
                kernel_name=kernel_name,
                regime_name=regime_name,
                phase_name=phase_name,
                target=target,
                logger=logger,
                summary=summary,
                memory=memory,
            )
            if support is not None:
                with _own_transition_outputs(
                    memory=memory, outputs=(weights, support), restore=weights
                ):
                    valid_support = _validate_joint_support(
                        support=support,
                        support_size=law.support_signature.size,
                        kernel_name=kernel_name,
                        regime_name=regime_name,
                        phase_name=phase_name,
                        target=target,
                        age=age,
                        logger=logger,
                        summary=summary,
                        memory=memory,
                    )
                    if valid_support:
                        _check_joint_support_schema(
                            support=support,
                            kernel_name=kernel_name,
                            regime_name=regime_name,
                            phase_name=phase_name,
                            target=target,
                            period=period_index,
                            logger=logger,
                            summary=summary,
                            support_schemas=support_schemas,
                        )
            probs = weights[f"weight_{target}__{kernel_name}"]
            _validate_joint_probabilities(
                probs=probs,
                support_size=law.support_signature.size,
                n_cells=n_cells,
                kernel_name=kernel_name,
                regime_name=regime_name,
                phase_name=phase_name,
                target=target,
                age=age,
                logger=logger,
                summary=summary,
                memory=memory,
            )


def _check_joint_support_schema(
    *,
    support: object,
    kernel_name: str,
    regime_name: RegimeName,
    phase_name: str,
    target: RegimeName,
    period: int,
    logger: logging.Logger,
    summary: _ValidationSummary | None,
    support_schemas: dict[tuple[RegimeName, RegimeName, str], _SupportSchema],
) -> None:
    """Preserve one support's pytree, event shapes, dtypes, and phase identity."""
    leaves, tree = jax.tree_util.tree_flatten(support)
    leaf_schema = tuple((tuple(leaf.shape[1:]), str(leaf.dtype)) for leaf in leaves)
    signature_key = (regime_name, target, kernel_name)
    previous = support_schemas.get(signature_key)
    if previous is None:
        support_schemas[signature_key] = (phase_name, period, tree, leaf_schema)
        return
    previous_phase, previous_period, previous_tree, previous_leaves = previous
    if tree == previous_tree and leaf_schema == previous_leaves:
        return
    if summary is not None:
        raise _SerialValidationRequired
    changed_support = (
        f"Joint transition {kernel_name}.support changed its static pytree signature "
        f"between {previous_phase} period {previous_period} and {phase_name} period "
        f"{period} of regime {regime_name}, target {target}. Support values may "
        "differ, but pytree structure, leaf event shapes, and dtypes must remain "
        f"identical across periods and phases; got {previous_leaves} and "
        f"{leaf_schema}."
    )
    raise_or_warn(logger=logger, error=RegimeInitializationError(changed_support))


def _evaluate_joint_support(
    *,
    func: Callable[..., Any],
    regime_params: FlatRegimeParams,
    period: ScalarInt,
    age: ScalarInt | ScalarFloat,
    kernel_name: str,
    regime_name: RegimeName,
    phase_name: str,
    target: RegimeName,
    logger: logging.Logger,
    summary: _ValidationSummary | None = None,
    memory: SimulationMemory | None = None,
) -> Any:  # noqa: ANN401
    """Bind and admit one complete parameter-bound joint-support provider."""
    kwargs: dict[str, object] = {}
    for name in inspect.signature(func).parameters:
        if name == "period":
            kwargs[name] = period
        elif name == "age":
            kwargs[name] = age
        elif name in regime_params:
            kwargs[name] = regime_params[name]
        else:
            if summary is not None:
                raise _SerialValidationRequired
            raise_or_warn(
                logger=logger,
                error=RegimeInitializationError(
                    f"Joint transition {kernel_name}.support may read only period, "
                    f"age, and parameters; unbound argument {name} appears in the "
                    f"{phase_name} phase of regime {regime_name}, target {target}."
                ),
            )
            return None

    if memory is None:
        return func(**kwargs)
    return _evaluate_admitted_transition_producer(
        function=func,
        arguments=MappingProxyType(kwargs),
        memory=memory,
    )


def _validate_joint_support(
    *,
    support: object,
    support_size: int,
    kernel_name: str,
    regime_name: RegimeName,
    phase_name: str,
    target: RegimeName,
    age: ScalarInt | ScalarFloat,
    logger: logging.Logger,
    summary: _ValidationSummary | None,
    memory: SimulationMemory | None,
) -> bool:
    """Validate one completed support while its full pytree remains owned."""
    leaves, _ = jax.tree_util.tree_flatten(support)
    invalid_shapes = [
        getattr(leaf, "shape", None)
        for leaf in leaves
        if not hasattr(leaf, "shape") or not leaf.shape or leaf.shape[0] != support_size
    ]
    if not leaves or invalid_shapes:
        if summary is not None:
            raise _SerialValidationRequired
        raise_or_warn(
            logger=logger,
            error=RegimeInitializationError(
                f"Joint transition {kernel_name}.support must be a nonempty pytree "
                f"whose every leaf has leading axis support_size={support_size}; "
                f"invalid leaf shape(s): {invalid_shapes}."
            ),
        )
        # The caller compares static schemas only for structurally valid
        # supports. In warning mode validation continues, so returning the invalid
        # pytree here would make the comparison itself dereference missing shapes.
        return False

    if summary is not None:
        summary.append(
            function=_support_finiteness_flags,
            arguments={"leaves": tuple(leaves)},
        )
        return True
    try:
        flags = (
            _support_finiteness_flags(leaves=tuple(leaves))
            if memory is None
            else run_simulation_operation(
                memory=memory,
                function=_support_finiteness_flags,
                arguments={"leaves": tuple(leaves)},
            )
        )
        has_nonfinite = bool(np.asarray(flags).any())
    except TypeError:
        has_nonfinite = True
    if has_nonfinite:
        raise_or_warn(
            logger=logger,
            error=RegimeInitializationError(
                f"Joint transition {kernel_name}.support contains nonfinite or "
                f"unsupported leaf values ({phase_name} phase of regime "
                f"{regime_name}, target {target}, age {age})."
            ),
        )
    return True


def _validate_joint_probabilities(
    *,
    probs: FloatND | IntND,
    support_size: int,
    n_cells: int | None,
    kernel_name: str,
    regime_name: RegimeName,
    phase_name: str,
    target: RegimeName,
    age: ScalarInt | ScalarFloat,
    logger: logging.Logger,
    summary: _ValidationSummary | None,
    memory: SimulationMemory | None,
) -> None:
    """Validate one completed joint weight array while its mapping stays owned."""
    expected_shape = (support_size,) if n_cells is None else (n_cells, support_size)
    if probs.shape != expected_shape:
        if summary is not None:
            raise _SerialValidationRequired
        owes = (
            "reads no grid variable, so it owes exactly one probability vector"
            if n_cells is None
            else f"is evaluated over {n_cells} source cells"
        )
        raise_or_warn(
            logger=logger,
            error=InvalidStateTransitionProbabilitiesError(
                f"Joint transition {kernel_name}.probabilities returned shape "
                f"{probs.shape}; expected {expected_shape}. The function {owes}, and "
                f"support_size is {support_size} ({phase_name} phase of regime "
                f"{regime_name}, target {target}, age {age}). An axis beyond those "
                "has no declared source variable, so no row of it can be attributed "
                "to a source cell."
            ),
        )
    if summary is not None:
        summary.append(
            function=_joint_probability_flags,
            arguments={"probabilities": probs},
        )
        return
    flags = (
        _joint_probability_flags(probabilities=probs)
        if memory is None
        else run_simulation_operation(
            memory=memory,
            function=_joint_probability_flags,
            arguments={"probabilities": probs},
        )
    )
    if np.asarray(flags).any():
        raise_or_warn(
            logger=logger,
            error=InvalidStateTransitionProbabilitiesError(
                f"Joint transition {kernel_name}.probabilities contains nonfinite or "
                "out-of-range values, or rows that do not sum to one "
                f"({phase_name} phase of regime {regime_name}, target {target}, "
                f"age {age})."
            ),
        )


def _evaluate_joint_weights(
    *,
    func: Callable[..., Mapping[str, FloatND | IntND]],
    state_action_space: StateActionSpace,
    extra_grids: Mapping[StateOrActionName, FloatND | IntND],
    regime_params: FlatRegimeParams,
    period: ScalarInt,
    age: ScalarInt | ScalarFloat,
    regime_name: RegimeName,
    phase_name: str,
    logger: logging.Logger,
    summary: _ValidationSummary | None = None,
    memory: SimulationMemory | None = None,
) -> tuple[Mapping[str, FloatND | IntND], int | None] | None:
    """Evaluate one compiled probability DAG on its accepted grids.

    `extra_grids` carries grid axes the solution state-action space does not
    hold — the simulate-phase domain of each carried-only state. An argument
    that resolves to none of the grids or the regime's parameters leaves the
    lottery unvalidated, which `log_level="debug"` refuses rather than sampling
    from an unexamined law.

    Returns:
        Tuple of the evaluated weights and the number of source cells they were
        evaluated over — `None` cells when the DAG reads no grid variable, so
        that each lottery owes exactly one probability vector rather than one
        per cell. `None` in place of the tuple when the lottery could not be
        evaluated at all.
    """
    grid_args: dict[StateOrActionName, FloatND | IntND] = {}
    scalar_kwargs: dict[str, object] = {}
    for name in inspect.signature(func).parameters:
        if name == "period":
            scalar_kwargs[name] = period
        elif name == "age":
            scalar_kwargs[name] = age
        elif name in state_action_space.states:
            grid_args[name] = state_action_space.states[name]
        elif name in state_action_space.actions:
            grid_args[name] = state_action_space.actions[name]
        elif name in extra_grids:
            grid_args[name] = extra_grids[name]
        elif name in regime_params:
            scalar_kwargs[name] = regime_params[name]
        else:
            if summary is not None:
                raise _SerialValidationRequired
            raise_or_warn(
                logger=logger,
                error=InvalidStateTransitionProbabilitiesError(
                    f"Joint transitions in regime {regime_name!r} "
                    f"({phase_name} phase) cannot be validated numerically: "
                    f"argument {name!r} is neither a grid variable of that "
                    "phase nor a parameter of the regime, so the lottery it "
                    "weights is never examined."
                ),
            )
            return None

    function = partial(
        _joint_weight_law,
        grid_names=tuple(grid_args),
        func=func,
    )
    arguments = MappingProxyType(
        {
            "grid_args": MappingProxyType(grid_args),
            "scalar_kwargs": MappingProxyType(scalar_kwargs),
        }
    )
    produced = (
        function(grid_args=grid_args, scalar_kwargs=scalar_kwargs)
        if memory is None
        else _evaluate_admitted_transition_producer(
            function=function,
            arguments=arguments,
            memory=memory,
        )
    )
    weights = cast("Mapping[str, FloatND | IntND]", produced)
    n_cells = prod(array.size for array in grid_args.values()) if grid_args else None
    return MappingProxyType(weights), n_cells


def _joint_weight_law(
    *,
    grid_args: Mapping[StateOrActionName, FloatND | IntND],
    scalar_kwargs: Mapping[str, object],
    grid_names: tuple[StateOrActionName, ...],
    func: Callable[..., Mapping[str, FloatND | IntND]],
) -> Mapping[str, FloatND | IntND]:
    """Evaluate a joint weight DAG and its Cartesian grid in one producer."""
    if not grid_names:
        return func(**scalar_kwargs)
    mesh = jnp.meshgrid(*(grid_args[name] for name in grid_names), indexing="ij")
    flat_arrays = tuple(array.ravel() for array in mesh)
    return jax.vmap(
        _GridPointCall(names=grid_names, scalar_kwargs=scalar_kwargs, func=func)
    )(*flat_arrays)


def _state_transition_unused_in_period(
    *,
    transition: _StochasticStateTransition,
    regime: Regime,
    period: int,
) -> bool:
    """Return whether a state transition has no retained edge this period.

    A coarse (`target_regime_name is None`) state law applies regardless of
    the regime-transition graph's own target set — an empty `period_targets`
    is a fact about *which regime* is reached, not about whether a coarse
    state law still needs checking. Only a per-target state law can be
    unused, and only when its specific target isn't retained this period.
    """
    if transition.target_regime_name is None:
        return False
    period_targets = regime.solution.reachability.targets(
        period=period, source=regime.name
    )
    return transition.target_regime_name not in period_targets


def _validate_state_transition_single(
    *,
    transition: _StochasticStateTransition,
    regime_params: FlatRegimeParams,
    state_action_space: StateActionSpace,
    regime_name: RegimeName,
    age: float | ScalarInt | ScalarFloat,
    period: int,
    logger: logging.Logger,
    summary: _ValidationSummary | None = None,
    memory: SimulationMemory | None = None,
) -> None:
    """Evaluate one MarkovTransition on its grid args and validate the output."""
    func = transition.func
    sig_params = tuple(inspect.signature(func).parameters)

    binding = (
        _state_probability_binding(
            transition=transition,
            signature_names=sig_params,
            state_action_space=state_action_space,
            regime_params=regime_params,
            regime_name=regime_name,
            age=age,
            period=period,
        )
        if summary is not None and summary.memory is None
        else None
    )
    if _append_cached_state_probability(
        summary=summary,
        binding=binding,
        transition=transition,
        regime_name=regime_name,
        age=age,
    ):
        return

    grid_args: dict[StateOrActionName, FloatND | IntND] = {}
    scalar_kwargs: dict[str, object] = {}
    period_int32 = jnp.int32(period)

    for name in sig_params:
        if name == "period":
            scalar_kwargs["period"] = period_int32
        elif name == "age":
            scalar_kwargs["age"] = age
        elif name in state_action_space.states:
            grid_args[name] = state_action_space.states[name]
        elif name in state_action_space.actions:
            grid_args[name] = state_action_space.actions[name]
        elif name in regime_params:
            scalar_kwargs[name] = regime_params[name]
        else:
            # An indexing param the function expects is neither a regime
            # grid nor a param. Skip numerical validation for this
            # transition rather than raising — a raise here would conceal
            # the real error the solve step surfaces. Warn so the skip is
            # not silent. Name the phase: a `Phased` law has two variants under
            # one state name, and only one of them may be hitting this branch.
            if summary is not None:
                raise _SerialValidationRequired
            phase_suffix = (
                f" ({transition.phase} phase)" if transition.phase is not None else ""
            )
            logger.warning(
                "MarkovTransition for state '%s' in regime '%s'%s not numerically "
                "validated: parameter '%s' is not a recognized grid or model "
                "parameter.",
                transition.state_name,
                regime_name,
                phase_suffix,
                name,
            )
            return

    current_memory = (
        summary.memory if summary is not None and summary.memory is not None else memory
    )
    probs = _evaluate_state_probability_law(
        func=func,
        grid_args=MappingProxyType(grid_args),
        scalar_kwargs=MappingProxyType(scalar_kwargs),
        memory=current_memory,
    )
    _check_and_release_state_probability(
        probs=probs,
        transition=transition,
        regime_name=regime_name,
        age=age,
        summary=summary,
        memory=current_memory,
    )
    _remember_state_probability(summary=summary, binding=binding, shape=probs.shape)


def _check_and_release_state_probability(
    *,
    probs: FloatND,
    transition: _StochasticStateTransition,
    regime_name: RegimeName,
    age: float | ScalarInt | ScalarFloat,
    summary: _ValidationSummary | None,
    memory: SimulationMemory | None,
) -> None:
    """Own one completed state law until its admitted reduction is retained."""
    if memory is not None:
        memory.set_derived(probs)
    try:
        _check_state_probs(
            probs=probs,
            transition=transition,
            regime_name=regime_name,
            age=age,
            summary=summary,
            memory=memory,
        )
    finally:
        if memory is not None:
            memory.set_derived(())


def _evaluate_state_probability_law(
    *,
    func: Callable[..., FloatND],
    grid_args: Mapping[StateOrActionName, FloatND | IntND],
    scalar_kwargs: Mapping[str, object],
    memory: SimulationMemory | None,
) -> FloatND:
    """Admit a complete stochastic-state law before its first device dispatch."""
    if memory is None:
        return _state_probability_law(
            grid_args=grid_args,
            scalar_kwargs=scalar_kwargs,
            grid_names=tuple(grid_args),
            func=func,
        )
    arguments = MappingProxyType(
        {"grid_args": grid_args, "scalar_kwargs": scalar_kwargs}
    )
    placed = place_simulation_arguments(
        arguments=arguments,
        subject_arg_names=(),
        value_reads=(),
        devices=memory.subject_devices,
        budget_bytes=memory.budget_bytes,
        live_footprint=memory.snapshot(),
        budget_devices=memory.devices,
    )
    argument_buffers = measure_buffer_footprint(tree=placed)
    live = union_buffer_footprints(
        footprints=(memory.snapshot(additional=arguments), argument_buffers)
    )
    external = resident_bytes_by_device(
        live=live, arguments=argument_buffers, devices=memory.subject_devices
    )
    output_sharding = simulation_value_sharding(
        stored_sharding=jax.sharding.SingleDeviceSharding(memory.subject_devices[0]),
        devices=(memory.subject_devices[0],),
    )
    compiler = _TransitionLawCompiler(
        function=partial(
            _state_probability_law,
            grid_names=tuple(grid_args),
            func=func,
        ),
        arguments=jax.tree.map(_abstract_transition_operand, dict(placed)),
        output_sharding=output_sharding,
    )
    plan = plan_workspace(
        axes=(),
        compile_candidate=compiler,
        budget_bytes=memory.budget_bytes,
        resident_bytes=max(external.values()),
    )
    return cast("FloatND", plan.compiled(**placed).block_until_ready())


def _evaluate_admitted_transition_producer(
    *,
    function: Callable[..., object],
    arguments: Mapping[str, object],
    memory: SimulationMemory,
) -> object:
    """Place, profile, admit, and complete one transition validation producer."""
    placed = place_simulation_arguments(
        arguments=arguments,
        subject_arg_names=(),
        value_reads=(),
        devices=memory.subject_devices,
        budget_bytes=memory.budget_bytes,
        live_footprint=memory.snapshot(),
        budget_devices=memory.devices,
    )
    argument_buffers = measure_buffer_footprint(tree=placed)
    live = union_buffer_footprints(
        footprints=(memory.snapshot(additional=arguments), argument_buffers)
    )
    external = resident_bytes_by_device(
        live=live, arguments=argument_buffers, devices=memory.subject_devices
    )
    output_sharding = simulation_value_sharding(
        stored_sharding=jax.sharding.SingleDeviceSharding(memory.subject_devices[0]),
        devices=(memory.subject_devices[0],),
    )
    compiler = _TransitionLawCompiler(
        function=function,
        arguments=jax.tree.map(_abstract_transition_operand, dict(placed)),
        output_sharding=output_sharding,
    )
    plan = plan_workspace(
        axes=(),
        compile_candidate=compiler,
        budget_bytes=memory.budget_bytes,
        resident_bytes=max(external.values()),
    )
    return jax.block_until_ready(plan.compiled(**placed))


def _state_probability_law(
    *,
    grid_args: Mapping[StateOrActionName, FloatND | IntND],
    scalar_kwargs: Mapping[str, object],
    grid_names: tuple[StateOrActionName, ...],
    func: Callable[..., FloatND],
) -> FloatND:
    """Evaluate one state law and its Cartesian grid inside one compiled producer."""
    if not grid_names:
        return func(**scalar_kwargs)
    mesh = jnp.meshgrid(*(grid_args[name] for name in grid_names), indexing="ij")
    flat_arrays = tuple(array.ravel() for array in mesh)
    return jax.vmap(
        _GridPointCall(names=grid_names, scalar_kwargs=scalar_kwargs, func=func)
    )(*flat_arrays)


@dataclass(frozen=True, kw_only=True)
class _TransitionLawCompiler:
    """Keep one user law call-local while compiling its concrete producer shape."""

    function: Callable[..., object]
    """Complete user-law producer with grid construction inside its boundary."""

    arguments: Mapping[str, object]
    """Placed abstract operands retained in the compiler allocation report."""

    output_sharding: jax.sharding.Sharding
    """Selected-device placement fixed before admission and first dispatch."""

    def __call__(self, widths: Mapping[str, int]) -> jax.stages.Compiled:
        """Lower the complete producer without allocating its result."""
        if widths:
            raise ExecutionPlanningError("Transition validation declares no axes.")
        lowered = jax.jit(
            self.function,
            keep_unused=True,
            out_shardings=self.output_sharding,
        ).lower(**self.arguments)
        return lowered.compile()


def _abstract_transition_operand(value: object) -> object:
    """Preserve the placed transition operand's dtype, weak type, and layout."""
    if isinstance(value, jax.Array):
        return jax.ShapeDtypeStruct(
            value.shape,
            value.dtype,
            sharding=value.sharding,
            weak_type=getattr(value, "weak_type", False),
        )
    return value


def _append_cached_state_probability(
    *,
    summary: _ValidationSummary | None,
    binding: tuple[tuple[object, ...], tuple[object, ...]] | None,
    transition: _StochasticStateTransition,
    regime_name: RegimeName,
    age: float | ScalarInt | ScalarFloat,
) -> bool:
    """Append a call-owned flag only after checking this occurrence's shape."""
    if summary is None or binding is None:
        return False
    cached = summary.state_probabilities.get(binding[0])
    if cached is None:
        return False
    _check_state_outcome_axis(
        shape=cached.shape,
        transition=transition,
        regime_name=regime_name,
        age=age,
        summary=summary,
    )
    summary.flags.append(cached.flag)
    return True


def _remember_state_probability(
    *,
    summary: _ValidationSummary | None,
    binding: tuple[tuple[object, ...], tuple[object, ...]] | None,
    shape: tuple[int, ...],
) -> None:
    """Retain the completed reduction and immutable bindings, not probabilities."""
    if summary is None or binding is None:
        return
    key, bound_inputs = binding
    summary.state_probabilities[key] = _StateProbabilitySummary(
        shape=shape, flag=summary.flags[-1], bound_inputs=bound_inputs
    )


def _state_probability_binding(
    *,
    transition: _StochasticStateTransition,
    signature_names: tuple[str, ...],
    state_action_space: StateActionSpace,
    regime_params: FlatRegimeParams,
    regime_name: RegimeName,
    age: float | ScalarInt | ScalarFloat,
    period: int,
) -> tuple[tuple[object, ...], tuple[object, ...]] | None:
    """Identify this pure law's exact immutable inputs inside one preflight.

    Model numerical functions obey JAX's purity contract. Mutable or opaque
    explicit operands decline reuse; a callable identity alone never suffices.
    Age and period enter the key whenever consumed. Outcome count is checked
    separately for every occurrence, including when the numerical inputs repeat.
    """
    arguments: list[tuple[str, object]] = []
    bound_inputs: list[object] = [transition.func]
    actions = state_action_space.actions
    for name in signature_names:
        if name == "period":
            value: object = period
        elif name == "age":
            value = age
        elif name in state_action_space.states:
            value = state_action_space.states[name]
        elif name in actions:
            value = actions[name]
        elif name in regime_params:
            value = regime_params[name]
        else:
            return None
        if isinstance(value, jax.Array):
            token: object = (jax.Array, id(value))
        elif type(value) in (bool, int, str, type(None)):
            token = (type(value), value)
        elif type(value) is float:
            token = (float, struct.pack("!d", value))
        else:
            return None
        arguments.append((name, token))
        bound_inputs.append(value)
    key = (
        regime_name,
        transition.state_name,
        transition.target_regime_name,
        transition.phase,
        id(transition.func),
        tuple(arguments),
    )
    return key, tuple(bound_inputs)


def _check_state_probs(
    *,
    probs: FloatND,
    transition: _StochasticStateTransition,
    regime_name: RegimeName,
    age: float | ScalarInt | ScalarFloat,
    summary: _ValidationSummary | None = None,
    memory: SimulationMemory | None = None,
) -> None:
    """Assert outcome-axis size, [0, 1] range, and sum-to-1 on a probs array."""
    state_label = _check_state_outcome_axis(
        shape=probs.shape,
        transition=transition,
        regime_name=regime_name,
        age=age,
        summary=summary,
    )

    if summary is not None:
        summary.append(
            function=_state_probability_flags,
            arguments={"probabilities": probs},
        )
        return
    if memory is None:
        flags = _state_probability_flags(probabilities=probs)
    else:
        flags = run_simulation_operation(
            memory=memory,
            function=_state_probability_flags,
            arguments={"probabilities": probs},
        )
    outside_bounds, invalid_mass = np.asarray(flags).tolist()
    if outside_bounds:
        raise InvalidStateTransitionProbabilitiesError(
            f"MarkovTransition for {state_label} in regime '{regime_name}' "
            f"at age {age} returned values outside [0, 1]."
        )

    if invalid_mass:
        raise InvalidStateTransitionProbabilitiesError(
            f"MarkovTransition for {state_label} in regime '{regime_name}' "
            f"at age {age} returned rows that do not sum to 1 along the "
            f"outcome axis."
        )


def _check_state_outcome_axis(
    *,
    shape: tuple[int, ...],
    transition: _StochasticStateTransition,
    regime_name: RegimeName,
    age: float | ScalarInt | ScalarFloat,
    summary: _ValidationSummary | None,
) -> str:
    """Check each occurrence's declared shape and preserve its diagnostic label."""
    qualifiers = []
    if transition.target_regime_name is not None:
        qualifiers.append(f"target regime '{transition.target_regime_name}'")
    if transition.phase is not None:
        # A `Phased` law has two variants under one state name; without the phase the
        # message would not say which of them is malformed.
        qualifiers.append(f"{transition.phase} phase")
    state_label = f"state '{transition.state_name}'"
    if qualifiers:
        state_label += f" ({', '.join(qualifiers)})"

    if shape[-1] != transition.n_outcomes:
        if summary is not None:
            raise _SerialValidationRequired
        raise InvalidStateTransitionProbabilitiesError(
            f"MarkovTransition for {state_label} in regime '{regime_name}' "
            f"at age {age} returned an outcome axis of size "
            f"{shape[-1]}; expected {transition.n_outcomes} from the "
            f"state's DiscreteGrid."
        )

    return state_label


def _unit_mass_violations(sum_all: FloatND) -> BoolND:
    """Return the mask of total regime masses that are not unit mass.

    The single criterion for "does not sum to 1", shared by the check that
    raises and the formatter that reports which entries failed — a formatter
    with a looser criterion of its own reports an empty table alongside a
    raised error.

    Tight by design. A tolerance wide enough to admit a mass that changes the
    Bellman `argmax` is not a guard: at float32 a total mass of `1.000005` is
    enough to reverse a decision, and `jnp.allclose`'s default `rtol` of `1e-5`
    admits it. Sixteen epsilons leaves room for the rounding of a handful of
    summed probabilities and nothing else.
    """
    tolerance = 16.0 * float(jnp.finfo(sum_all.dtype).eps)
    return jnp.abs(sum_all - 1.0) > tolerance


@dataclass(frozen=True, kw_only=True, eq=False)
class _GridPointCall:
    """Call a transition at one grid point, given positionally for `jax.vmap`."""

    names: tuple[str, ...]
    """Grid variable names, in the order the positional values arrive."""
    scalar_kwargs: Mapping[str, object] = field(repr=False)
    """Arguments held fixed across grid points."""
    func: Callable[..., Any] = field(repr=False)
    """The transition function."""

    # The kernel is traced with whatever leaves its caller supplies -- tracers,
    # Python scalars, arrays of either integer width -- so its annotations
    # document the contract and are not enforced at call time.
    @no_type_check
    def __call__(self, *args: FloatND | IntND) -> Any:  # noqa: ANN401
        kwargs = dict(zip(self.names, args, strict=True))
        return self.func(**kwargs, **self.scalar_kwargs)
