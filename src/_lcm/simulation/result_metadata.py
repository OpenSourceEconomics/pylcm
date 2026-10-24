"""Metadata and discrete-output dtypes for `SimulationResult`."""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType

import numpy as np
import pandas as pd

from _lcm.engine import PeriodRegimeSimulationData, Regime
from _lcm.grids import DiscreteGrid
from _lcm.processes.joint_normal import JointNormalIIDProcess
from _lcm.processes.joint_normal_lowering import (
    coordinate_names,
    joint_normal_declarations,
)
from _lcm.regime_building.processing import compute_merged_discrete_categories
from _lcm.time import TimeAxis
from _lcm.typing import ActionName, ProcessName, RegimeName, RegimeNamesToIds, StateName
from lcm.phased import Phased
from lcm.regime import Regime as UserRegime
from lcm.shocks import Discrete, ShockGroup, ShockGroupName

# A category of a published discrete column: a label, or the number of a value grid.
type CategoryValue = str | int | float


@dataclass(frozen=True)
class ResultMetadata:
    """Pre-computed metadata about a `SimulationResult`."""

    regime_names: tuple[RegimeName, ...]
    """Names of all regimes in the model."""

    state_names: tuple[StateName, ...]
    """Sorted union of state variable names across all regimes."""

    action_names: tuple[ActionName, ...]
    """Sorted union of action variable names across all regimes."""

    n_periods: int
    """Number of periods in the simulation."""

    n_subjects: int
    """Number of subjects simulated."""

    regime_to_states: MappingProxyType[RegimeName, tuple[StateName, ...]]
    """Immutable mapping of regime names to their state variable names."""

    regime_to_actions: MappingProxyType[RegimeName, tuple[ActionName, ...]]
    """Immutable mapping of regime names to their action variable names."""

    regime_to_publishes_nested_policy: MappingProxyType[RegimeName, bool]
    """Whether each regime can publish a nested continuous-outer policy read.

    True exactly for regimes solved by NNBEGM, the only solver that constructs a
    `NestedEGMSimPolicy`. The `nested_policy_fallback` column is emitted only
    for those regimes: elsewhere it would be a constant-False column in every
    user's dataframe.
    """

    discrete_categories: MappingProxyType[str, tuple[CategoryValue, ...]]
    """Immutable mapping of discrete variable names to their categories.

    A category-class grid's string labels, or a value grid's numbers.
    """

    discrete_ordered: MappingProxyType[str, bool]
    """Immutable mapping of discrete variable names to their ordered flag."""

    regime_discrete_categories: MappingProxyType[
        tuple[RegimeName, str], tuple[CategoryValue, ...]
    ]
    """Immutable mapping of (regime_name, var_name) to per-regime categories."""

    regime_to_stakeholders: MappingProxyType[RegimeName, tuple[str, ...] | None] = (
        MappingProxyType({})
    )
    """Immutable mapping of regime names to their ordered stakeholder names.

    `None` for a singleton regime. Captured at `SimulationResult.__init__` time
    (when `regimes` is always fully populated) so `to_dataframe` can tell a
    collective regime's recorded `value` apart from a singleton's even after
    `save()` has dropped the `Regime` objects themselves — see
    `_lcm.simulation.result_dataframe._process_regime`.

    The default is empty, so a saved artifact whose metadata payload carries no
    such mapping still loads. Every regime then reads as a singleton, which is
    the only reading available; a regime whose recorded value does carry a
    stakeholder axis is reported by `_lcm.simulation.result_dataframe`, naming
    this field, instead of being published under a wrong column layout.
    """

    stakeholder_names_to_ids: MappingProxyType[str, int] = MappingProxyType({})
    """The model's role vocabulary: every stakeholder name, as a code.

    One mapping for the whole model rather than one per regime, so the
    `own_stakeholder` column means the same thing in every row and a role can
    be compared across regimes. Empty for a model with no collective regime,
    and for a saved artifact whose metadata payload predates the field — the
    column is then published as raw codes rather than labels.
    """

    regime_to_joint_normals: MappingProxyType[
        RegimeName, MappingProxyType[ProcessName, JointNormalIIDProcess]
    ] = MappingProxyType({})
    """Physical-output laws, independent of compiled regime workspaces.

    Fixed parameter leaves are frozen into each declaration. Runtime leaves are
    read from the result's flat parameters. Empty for artifacts predating joint laws.
    """

    randomness: MappingProxyType[str, str] = MappingProxyType({})
    """Address version, PRNG implementations and floating precision of this run.

    Empty for an archive that did not record its random experiment.
    """


def _get_output_dtypes(
    *,
    user_regimes: Mapping[RegimeName, UserRegime],
    regime_names_to_ids: RegimeNamesToIds,
    shocks: Mapping[ShockGroupName, ShockGroup],
) -> MappingProxyType[str, pd.CategoricalDtype]:
    """Compute pandas CategoricalDtype for all discrete output columns.

    Merge ordered categories across regimes via topological sort. This must be
    called after model validation (which guarantees merges succeed). A variable
    declared with `values` has the same values wherever it appears, and they are
    its categories: integers or floats, in ascending order.

    Args:
        user_regimes: Mapping of regime names to user-provided `Regime` instances.
        regime_names_to_ids: Mapping of regime names to integer IDs.
        shocks: Model-level current-IID laws supplying physical-output codecs.

    Returns:
        Immutable mapping of variable name to `pd.CategoricalDtype`. Includes
        all discrete state/action/current-shock variables and `"regime_name"`.

    """
    merged_categories, ordered_flags = compute_merged_discrete_categories(user_regimes)
    declared_values = {
        var_name: grid.values  # noqa: PD011
        for user_regime in user_regimes.values()
        for var_name, entry in {**user_regime.states, **user_regime.actions}.items()
        for grid in (entry.simulate if isinstance(entry, Phased) else entry,)
        if isinstance(grid, DiscreteGrid) and grid.values is not None  # noqa: PD011
    }

    dtypes: dict[str, pd.CategoricalDtype] = {}
    for var_name, categories in {**merged_categories, **declared_values}.items():
        dtypes[var_name] = pd.CategoricalDtype(
            categories=list(categories),
            ordered=ordered_flags.get(var_name, True),
        )

    for group in shocks.values():
        for var_name, law in group.shocks.items():
            if isinstance(law, Discrete):
                grid = DiscreteGrid(category_class=law.categories)
                dtypes[var_name] = pd.CategoricalDtype(
                    categories=list(grid.categories), ordered=grid.ordered
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
    ages: TimeAxis,
    nested_policy_regimes: frozenset[RegimeName] = frozenset(),
    randomness: Mapping[str, str] = MappingProxyType({}),
) -> ResultMetadata:
    """Compute metadata from canonical regimes, raw results, and output dtypes.

    `nested_policy_regimes` names the regimes whose solve published a
    `NestedEGMSimPolicy`. It is a property of the solved MODEL, not of the
    simulated subjects, so the dataframe schema it gates stays data-independent.
    """
    regime_names = tuple(regimes.keys())

    all_states: set[StateName] = set()
    all_actions: set[ActionName] = set()
    regime_to_states: dict[RegimeName, tuple[StateName, ...]] = {}
    regime_to_actions: dict[RegimeName, tuple[ActionName, ...]] = {}

    regime_to_publishes_nested_policy: dict[RegimeName, bool] = {}

    for regime_name, regime in regimes.items():
        regime_to_states[regime_name] = regime.simulation.state_names
        regime_to_actions[regime_name] = regime.simulation.action_names
        public_states = set(regime.simulation.state_names)
        for group, process in joint_normal_declarations(
            regime.simulation.grids
        ).items():
            public_states.difference_update(
                coordinate_names(group=group, process=process)
            )
            public_states.update(process.states)
        all_states.update(public_states)
        all_actions.update(regime.simulation.action_names)
        regime_to_publishes_nested_policy[regime_name] = (
            regime_name in nested_policy_regimes
        )

    # Extract categories and ordered flags from simulation_output_dtypes
    discrete_categories: dict[str, tuple[CategoryValue, ...]] = {}
    discrete_ordered: dict[str, bool] = {}
    for var_name, dtype in simulation_output_dtypes.items():
        if var_name == "regime_name":
            continue
        discrete_categories[var_name] = tuple(dtype.categories)
        discrete_ordered[var_name] = bool(dtype.ordered)

    # Per-regime discrete categories for correct code→label mapping. The
    # simulation grids include carried pair states, so a discrete pair's
    # output column gets labels like any other discrete state.
    regime_discrete_categories: dict[
        tuple[RegimeName, str], tuple[CategoryValue, ...]
    ] = {}
    for regime_name, regime in regimes.items():
        for var_name, grid in regime.simulation.discrete_grids.items():
            values = grid.values  # noqa: PD011
            regime_discrete_categories[(regime_name, var_name)] = (
                grid.categories if values is None else values
            )

    regime_to_stakeholders: dict[RegimeName, tuple[str, ...] | None] = {
        regime_name: regime.stakeholders for regime_name, regime in regimes.items()
    }
    stakeholder_names_to_ids = next(
        (regime.stakeholder_names_to_ids for regime in regimes.values()),
        MappingProxyType({}),
    )

    n_periods = ages.n_periods
    n_subjects = _get_n_subjects(raw_results)

    return ResultMetadata(
        regime_names=regime_names,
        state_names=tuple(sorted(all_states)),
        action_names=tuple(sorted(all_actions)),
        n_periods=n_periods,
        n_subjects=n_subjects,
        regime_to_states=MappingProxyType(regime_to_states),
        regime_to_actions=MappingProxyType(regime_to_actions),
        regime_to_publishes_nested_policy=MappingProxyType(
            regime_to_publishes_nested_policy
        ),
        discrete_categories=MappingProxyType(discrete_categories),
        discrete_ordered=MappingProxyType(discrete_ordered),
        regime_discrete_categories=MappingProxyType(regime_discrete_categories),
        regime_to_stakeholders=MappingProxyType(regime_to_stakeholders),
        stakeholder_names_to_ids=stakeholder_names_to_ids,
        regime_to_joint_normals=MappingProxyType(
            {name: _output_joint_laws(regime) for name, regime in regimes.items()}
        ),
        randomness=MappingProxyType(dict(randomness)),
    )


def _output_joint_laws(
    regime: Regime,
) -> MappingProxyType[ProcessName, JointNormalIIDProcess]:
    """Capture only the physical laws needed after compiled regimes are released."""
    laws = {}
    for group, process in joint_normal_declarations(regime.simulation.grids).items():
        mean = regime.resolved_fixed_params.get(f"{group}__mean")
        cov = regime.resolved_fixed_params.get(f"{group}__cov")
        laws[group] = replace(
            process,
            mean=tuple(float(value) for value in np.asarray(mean))
            if mean is not None
            else process.mean,
            cov=tuple(tuple(float(value) for value in row) for row in np.asarray(cov))
            if cov is not None
            else process.cov,
        )
    return MappingProxyType(laws)


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
