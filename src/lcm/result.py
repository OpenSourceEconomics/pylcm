"""User-facing `SimulationResult` with deferred DataFrame computation."""

import pickle
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Literal

import cloudpickle
import pandas as pd

from _lcm.engine import PeriodRegimeSimulationData, Regime
from _lcm.persistence.io import _atomic_dump
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
        """Serialize the SimulationResult to a file.

        Args:
            path: File path to save the pickle.
            protocol: Int which indicates which protocol should be used by the pickler,
                default HIGHEST_PROTOCOL. The possible values are 0, 1, 2, 3, 4, 5. See
                https://docs.python.org/3/library/pickle.html.

        Returns:
            The path where the object was saved.

        """
        return _atomic_dump(self, path, protocol=protocol)

    @classmethod
    def from_pickle(cls, path: str | Path) -> SimulationResult:
        """Deserialize a SimulationResult from a pickle file.

        Args:
            path: File path to read the pickle from.

        Returns:
            The unpickled SimulationResult object.

        """
        p = Path(path)
        with p.open("rb") as f:
            obj = cloudpickle.load(f)

        if not isinstance(obj, cls):
            raise TypeError(
                f"Pickle at {p} is {type(obj).__name__}, expected {cls.__name__}"
            )
        return obj

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
