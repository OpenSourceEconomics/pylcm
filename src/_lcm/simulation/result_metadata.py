"""Metadata and discrete-output dtypes for `SimulationResult`."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import pandas as pd

from _lcm.engine import PeriodRegimeSimulationData, Regime
from _lcm.regime_building.processing import compute_merged_discrete_categories
from _lcm.typing import ActionName, RegimeName, RegimeNamesToIds, StateName
from lcm.ages import AgeGrid
from lcm.regime import Regime as UserRegime


@dataclass(frozen=True)
class ResultMetadata:
    """Pre-computed metadata about a `SimulationResult`."""

    regime_names: list[RegimeName]
    """Names of all regimes in the model."""

    state_names: list[StateName]
    """Sorted union of state variable names across all regimes."""

    action_names: list[ActionName]
    """Sorted union of action variable names across all regimes."""

    n_periods: int
    """Number of periods in the simulation."""

    n_subjects: int
    """Number of subjects simulated."""

    regime_to_states: MappingProxyType[RegimeName, tuple[StateName, ...]]
    """Immutable mapping of regime names to their state variable names."""

    regime_to_actions: MappingProxyType[RegimeName, tuple[ActionName, ...]]
    """Immutable mapping of regime names to their action variable names."""

    discrete_categories: MappingProxyType[str, tuple[str, ...]]
    """Immutable mapping of discrete variable names to their category labels."""

    discrete_ordered: MappingProxyType[str, bool]
    """Immutable mapping of discrete variable names to their ordered flag."""

    regime_discrete_categories: MappingProxyType[
        tuple[RegimeName, str], tuple[str, ...]
    ]
    """Immutable mapping of (regime_name, var_name) to per-regime categories."""


def _get_output_dtypes(
    user_regimes: Mapping[RegimeName, UserRegime],
    regime_names_to_ids: RegimeNamesToIds,
) -> MappingProxyType[str, pd.CategoricalDtype]:
    """Compute pandas CategoricalDtype for all discrete output columns.

    Merge ordered categories across regimes via topological sort. This must be
    called after model validation (which guarantees merges succeed).

    Args:
        user_regimes: Mapping of regime names to user-provided `Regime` instances.
        regime_names_to_ids: Mapping of regime names to integer IDs.

    Returns:
        Immutable mapping of variable name to `pd.CategoricalDtype`. Includes
        all discrete state/action variables plus the `"regime_name"` column.

    """
    merged_categories, ordered_flags = compute_merged_discrete_categories(user_regimes)

    dtypes: dict[str, pd.CategoricalDtype] = {}
    for var_name, categories in merged_categories.items():
        dtypes[var_name] = pd.CategoricalDtype(
            categories=list(categories),
            ordered=ordered_flags[var_name],
        )

    dtypes["regime_name"] = pd.CategoricalDtype(
        categories=list(regime_names_to_ids.keys()),
        ordered=False,
    )

    return MappingProxyType(dtypes)


def _compute_metadata(
    *,
    regimes: MappingProxyType[RegimeName, Regime],
    raw_results: MappingProxyType[
        RegimeName, MappingProxyType[int, PeriodRegimeSimulationData]
    ],
    simulation_output_dtypes: Mapping[str, pd.CategoricalDtype],
    ages: AgeGrid,
) -> ResultMetadata:
    """Compute metadata from canonical regimes, raw results, and output dtypes."""
    regime_names = list(regimes.keys())

    all_states: set[StateName] = set()
    all_actions: set[ActionName] = set()
    regime_to_states: dict[RegimeName, tuple[StateName, ...]] = {}
    regime_to_actions: dict[RegimeName, tuple[ActionName, ...]] = {}

    for regime_name, regime in regimes.items():
        regime_to_states[regime_name] = regime.simulation.state_names
        regime_to_actions[regime_name] = regime.simulation.variables.action_names
        all_states.update(regime.simulation.state_names)
        all_actions.update(regime.simulation.variables.action_names)

    # Extract categories and ordered flags from simulation_output_dtypes
    discrete_categories: dict[str, tuple[str, ...]] = {}
    discrete_ordered: dict[str, bool] = {}
    for var_name, dtype in simulation_output_dtypes.items():
        if var_name == "regime_name":
            continue
        discrete_categories[var_name] = tuple(dtype.categories)
        discrete_ordered[var_name] = bool(dtype.ordered)

    # Per-regime discrete categories for correct code→label mapping. The
    # simulation grids include carried pair states, so a discrete pair's
    # output column gets labels like any other discrete state.
    regime_discrete_categories: dict[tuple[RegimeName, str], tuple[str, ...]] = {}
    for regime_name, regime in regimes.items():
        for var_name, grid in regime.simulation.discrete_grids.items():
            regime_discrete_categories[(regime_name, var_name)] = grid.categories

    n_periods = ages.n_periods
    n_subjects = _get_n_subjects(raw_results)

    return ResultMetadata(
        regime_names=regime_names,
        state_names=sorted(all_states),
        action_names=sorted(all_actions),
        n_periods=n_periods,
        n_subjects=n_subjects,
        regime_to_states=MappingProxyType(regime_to_states),
        regime_to_actions=MappingProxyType(regime_to_actions),
        discrete_categories=MappingProxyType(discrete_categories),
        discrete_ordered=MappingProxyType(discrete_ordered),
        regime_discrete_categories=MappingProxyType(regime_discrete_categories),
    )


def _get_n_subjects(
    raw_results: MappingProxyType[
        RegimeName, MappingProxyType[int, PeriodRegimeSimulationData]
    ],
) -> int:
    """Extract number of subjects from raw results."""
    for regime_results in raw_results.values():
        if regime_results:
            first_result = next(iter(regime_results.values()))
            return len(first_result.in_regime)
    return 0
