"""User-facing `SimulationResult` with deferred DataFrame computation."""

import pickle
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Literal

import cloudpickle
import h5py
import pandas as pd

from _lcm.engine import PeriodRegimeSimulationData, Regime
from _lcm.persistence.io import _atomic_dump, _read_h5_array, _write_sharded_dataset
from _lcm.simulation.additional_targets import (
    _collect_all_available_targets,
    _resolve_targets,
)
from _lcm.simulation.result_dataframe import (
    _convert_to_categorical,
    _create_flat_dataframe,
)
from _lcm.simulation.result_metadata import _compute_metadata
from _lcm.typing import ActionName, FlatParams, RegimeName, StateName
from lcm.ages import AgeGrid

_RESULT_PKL_NAME = "simulation_result.pkl"
_RESULT_H5_NAME = "simulation_result.h5"


class SimulationResult:
    """Result object from model simulation with deferred DataFrame computation."""

    def __init__(
        self,
        *,
        raw_results: MappingProxyType[
            RegimeName, MappingProxyType[int, PeriodRegimeSimulationData]
        ],
        regimes: MappingProxyType[RegimeName, Regime],
        flat_params: FlatParams,
        ages: AgeGrid,
        simulation_output_dtypes: Mapping[str, pd.CategoricalDtype],
    ) -> None:
        self._raw_results = raw_results
        self._regimes = regimes
        self._flat_params = flat_params
        self._ages = ages
        self._simulation_output_dtypes = simulation_output_dtypes
        self._metadata = _compute_metadata(
            regimes=regimes,
            raw_results=raw_results,
            simulation_output_dtypes=simulation_output_dtypes,
            ages=ages,
        )
        self._available_targets = sorted(_collect_all_available_targets(regimes))

    @property
    def raw_results(
        self,
    ) -> MappingProxyType[
        RegimeName, MappingProxyType[int, PeriodRegimeSimulationData]
    ]:
        """Raw simulation results by regime and period."""
        return self._raw_results

    @property
    def flat_params(self) -> FlatParams:
        """Model parameters used in simulation."""
        return self._flat_params

    @property
    def regime_names(self) -> list[RegimeName]:
        """Names of all regimes."""
        return self._metadata.regime_names

    @property
    def state_names(self) -> list[StateName]:
        """Names of all state variables (union across regimes)."""
        return self._metadata.state_names

    @property
    def action_names(self) -> list[ActionName]:
        """Names of all action variables (union across regimes)."""
        return self._metadata.action_names

    @property
    def n_periods(self) -> int:
        """Number of periods in the simulation."""
        return self._metadata.n_periods

    @property
    def n_subjects(self) -> int:
        """Number of subjects simulated."""
        return self._metadata.n_subjects

    @property
    def available_targets(self) -> list[str]:
        """Names of all available additional targets.

        These can be passed to `to_dataframe(additional_targets=...)`. Includes utility
        functions, auxiliary functions, and constraints from all regimes.

        """
        return self._available_targets

    def to_dataframe(
        self,
        additional_targets: list[str] | Literal["all"] | None = None,
        *,
        use_labels: bool = True,
    ) -> pd.DataFrame:
        """Convert simulation results to a flat pandas DataFrame.

        Args:
            additional_targets: Targets to compute. Can be:
                - None (default): No additional targets
                - list[str]: Specific target names to compute
                - "all": Compute all available targets (see `available_targets`)
                Targets can be any function defined in a regime. Each target is
                computed for the regimes where it exists; rows from regimes without
                that target will have NaN.
            use_labels: If True (default), discrete variables (states, actions, and
                regime) are returned as pandas Categorical dtype with string labels.
                If False, discrete variables are returned as integer codes.

        Returns:
            DataFrame with simulation results.

        """
        resolved_targets = _resolve_targets(
            additional_targets=additional_targets,
            available_targets=self.available_targets,
        )

        df = _create_flat_dataframe(
            raw_results=self._raw_results,
            regimes=self._regimes,
            flat_params=self._flat_params,
            metadata=self._metadata,
            additional_targets=resolved_targets,
            ages=self._ages,
        )

        if use_labels:
            return _convert_to_categorical(df=df, metadata=self._metadata)

        return df

    def to_pickle(
        self,
        path: str | Path,
        *,
        protocol: int = pickle.HIGHEST_PROTOCOL,
    ) -> Path:
        """Serialize the SimulationResult to a directory.

        Writes `<path>/simulation_result.pkl` (metadata: regimes, params, ages,
        dtypes, plus the regime/period structure of the raw results) and a
        sibling `<path>/simulation_result.h5` (per-`(regime, period, key)`
        datasets, written shard-by-shard so a sharded `jax.Array` is
        materialised one device at a time rather than via an all-gather to
        a single device).

        Args:
            path: Directory to write the result to. Must already exist.
            protocol: Pickle protocol for the metadata file. See
                https://docs.python.org/3/library/pickle.html.

        Returns:
            The directory the result was written to.

        """
        directory = Path(path)
        if not directory.is_dir():
            raise NotADirectoryError(
                f"`to_pickle` expects an existing directory; got {directory!r}."
            )

        with h5py.File(directory / _RESULT_H5_NAME, "w") as fh:
            for regime_name, period_dict in self._raw_results.items():
                regime_group = fh.create_group(regime_name)
                for period, data in period_dict.items():
                    period_group = regime_group.create_group(str(period))
                    _write_sharded_dataset(period_group, "V_arr", data.V_arr)
                    _write_sharded_dataset(period_group, "in_regime", data.in_regime)
                    actions_group = period_group.create_group("actions")
                    for action_name, arr in data.actions.items():
                        _write_sharded_dataset(actions_group, action_name, arr)
                    states_group = period_group.create_group("states")
                    for state_name, arr in data.states.items():
                        _write_sharded_dataset(states_group, state_name, arr)

        metadata = {
            "regimes": self._regimes,
            "flat_params": self._flat_params,
            "ages": self._ages,
            "simulation_output_dtypes": self._simulation_output_dtypes,
        }
        _atomic_dump(metadata, directory / _RESULT_PKL_NAME, protocol=protocol)
        return directory

    @classmethod
    def from_pickle(cls, path: str | Path) -> SimulationResult:
        """Deserialize a SimulationResult from a directory.

        Args:
            path: Directory previously written by `to_pickle`.

        Returns:
            The reconstructed `SimulationResult`. Array fields land on the
            JAX default device — original sharding is dropped on load.

        """
        directory = Path(path)
        if not directory.is_dir():
            raise NotADirectoryError(
                f"`from_pickle` expects a directory; got {directory!r}."
            )

        pkl_path = directory / _RESULT_PKL_NAME
        h5_path = directory / _RESULT_H5_NAME
        with pkl_path.open("rb") as f:
            metadata = cloudpickle.load(f)

        raw_results: dict[
            RegimeName, MappingProxyType[int, PeriodRegimeSimulationData]
        ] = {}
        with h5py.File(h5_path, "r") as fh:
            for regime_name in fh:
                regime_group = fh[regime_name]
                period_dict: dict[int, PeriodRegimeSimulationData] = {}
                for period_key in regime_group:
                    period = int(period_key)
                    period_group = regime_group[period_key]
                    actions_group = period_group["actions"]
                    states_group = period_group["states"]
                    period_dict[period] = PeriodRegimeSimulationData(
                        V_arr=_read_h5_array(period_group, "V_arr"),
                        in_regime=_read_h5_array(period_group, "in_regime"),
                        actions=MappingProxyType(
                            {k: _read_h5_array(actions_group, k) for k in actions_group}
                        ),
                        states=MappingProxyType(
                            {k: _read_h5_array(states_group, k) for k in states_group}
                        ),
                    )
                raw_results[regime_name] = MappingProxyType(period_dict)

        return cls(
            raw_results=MappingProxyType(raw_results),
            regimes=metadata["regimes"],
            flat_params=metadata["flat_params"],
            ages=metadata["ages"],
            simulation_output_dtypes=metadata["simulation_output_dtypes"],
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"  n_periods={self.n_periods},\n"
            f"  n_subjects={self.n_subjects},\n"
            f"  regime_names={self.regime_names},\n"
            f"  state_names={self.state_names},\n"
            f"  action_names={self.action_names}\n"
            f")"
        )
