"""User-facing `SimulationResult` with deferred DataFrame computation."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

import cloudpickle
import jax
import orbax.checkpoint as ocp
import pandas as pd

from _lcm.engine import PeriodRegimeSimulationData, Regime
from _lcm.simulation.additional_targets import (
    _collect_all_available_targets,
    _resolve_targets,
)
from _lcm.simulation.result_dataframe import (
    _convert_to_categorical,
    _create_flat_dataframe,
)
from _lcm.simulation.result_metadata import ResultMetadata, _compute_metadata
from _lcm.typing import ActionName, FlatParams, RegimeName, StateName
from lcm.ages import AgeGrid
from lcm.typing import FloatND


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
        period_to_regime_to_V_arr: MappingProxyType[
            int, MappingProxyType[RegimeName, FloatND]
        ],
        ages: AgeGrid,
        simulation_output_dtypes: Mapping[str, pd.CategoricalDtype],
    ) -> None:
        self._raw_results = raw_results
        self._regimes = regimes
        self._flat_params = flat_params
        self._period_to_regime_to_V_arr = period_to_regime_to_V_arr
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
    def period_to_regime_to_V_arr(
        self,
    ) -> MappingProxyType[int, MappingProxyType[RegimeName, FloatND]]:
        """Value function arrays from the solution."""
        return self._period_to_regime_to_V_arr

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

    def save(self, directory: str | Path) -> Path:
        """Persist the result to a directory.

        Arrays are written per-shard via `orbax-checkpoint` so sharded
        V-arrays never gather to a single device. Non-array fields
        (regimes, ages, metadata, parameter scaffolding) live in a
        sibling `metadata.pkl` produced via `cloudpickle`.

        Args:
            directory: Target directory. Created if it does not exist.
                Must not contain an existing orbax checkpoint at
                `directory/arrays`.

        Returns:
            The directory the result was written to.

        """
        target = Path(directory).resolve()
        target.mkdir(parents=True, exist_ok=True)

        array_tree = {
            "raw_results": _raw_results_to_array_tree(self._raw_results),
            "period_to_regime_to_V_arr": _period_v_to_array_tree(
                self._period_to_regime_to_V_arr
            ),
            "flat_params": _flat_params_to_array_tree(self._flat_params),
        }

        checkpointer = ocp.StandardCheckpointer()
        checkpointer.save(target / "arrays", array_tree)
        # `StandardCheckpointer.save` is asynchronous; block until the
        # on-disk checkpoint is complete so the sibling `metadata.pkl`
        # write and any subsequent `load` observe a consistent directory.
        checkpointer.wait_until_finished()

        metadata = _SavedMetadata(
            regimes=self._regimes,
            flat_params_scaffold=_flat_params_to_scaffold(self._flat_params),
            ages=self._ages,
            result_metadata=self._metadata,
            available_targets=self._available_targets,
        )
        with (target / "metadata.pkl").open("wb") as fh:
            cloudpickle.dump(metadata, fh)

        return target

    @classmethod
    def load(cls, directory: str | Path) -> SimulationResult:
        """Read a result previously written by `save`.

        Sharded arrays are reconstructed onto a sharding identical to the
        one used at save time, so no implicit gather happens during load.
        """
        source = Path(directory).resolve()

        with (source / "metadata.pkl").open("rb") as fh:
            metadata: _SavedMetadata = cloudpickle.load(fh)

        checkpointer = ocp.StandardCheckpointer()
        array_tree = checkpointer.restore(source / "arrays")

        raw_results = _array_tree_to_raw_results(array_tree["raw_results"])
        period_to_regime_to_V_arr = _array_tree_to_period_v(
            array_tree["period_to_regime_to_V_arr"]
        )
        flat_params = _array_tree_and_scaffold_to_flat_params(
            array_tree["flat_params"], metadata.flat_params_scaffold
        )

        instance = cls.__new__(cls)
        instance._raw_results = raw_results  # noqa: SLF001
        instance._regimes = metadata.regimes  # noqa: SLF001
        instance._flat_params = flat_params  # noqa: SLF001
        instance._period_to_regime_to_V_arr = period_to_regime_to_V_arr  # noqa: SLF001
        instance._ages = metadata.ages  # noqa: SLF001
        instance._metadata = metadata.result_metadata  # noqa: SLF001
        instance._available_targets = metadata.available_targets  # noqa: SLF001
        return instance

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


@dataclass(frozen=True)
class _SavedMetadata:
    """Non-array data persisted alongside the orbax array checkpoint."""

    regimes: MappingProxyType[RegimeName, Regime]
    """Canonical regimes used to assemble the original result."""

    flat_params_scaffold: dict[RegimeName, dict[str, Any]]
    """`flat_params` with every JAX array replaced by an `_ArrayPlaceholder`.

    Non-array leaves (e.g. `MappingLeaf`, `SequenceLeaf`) survive the
    cloudpickle round-trip directly; placeholders mark slots filled in by
    `flat_params` arrays loaded via orbax.
    """

    ages: AgeGrid
    """Lifecycle age grid of the original model."""

    result_metadata: ResultMetadata
    """Pre-computed metadata; rebuilt to avoid re-deriving from regimes."""

    available_targets: list[str]
    """Names of all additional targets exposed via `to_dataframe`."""


@dataclass(frozen=True)
class _ArrayPlaceholder:
    """Marker for a JAX array slot in a cloudpickled scaffold."""

    key: str


def _raw_results_to_array_tree(
    raw_results: MappingProxyType[
        RegimeName, MappingProxyType[int, PeriodRegimeSimulationData]
    ],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Convert raw results into a plain-dict tree of JAX arrays.

    Periods are stringified so orbax can use them as path components.
    """
    return {
        regime_name: {
            str(period): {
                "V_arr": data.V_arr,
                "actions": dict(data.actions),
                "states": dict(data.states),
                "in_regime": data.in_regime,
            }
            for period, data in regime_dict.items()
        }
        for regime_name, regime_dict in raw_results.items()
    }


def _array_tree_to_raw_results(
    tree: dict[str, dict[str, dict[str, Any]]],
) -> MappingProxyType[RegimeName, MappingProxyType[int, PeriodRegimeSimulationData]]:
    """Inverse of `_raw_results_to_array_tree`."""
    return MappingProxyType(
        {
            regime_name: MappingProxyType(
                {
                    int(period): PeriodRegimeSimulationData(
                        V_arr=period_dict["V_arr"],
                        actions=MappingProxyType(period_dict["actions"]),
                        states=MappingProxyType(period_dict["states"]),
                        in_regime=period_dict["in_regime"],
                    )
                    for period, period_dict in regime_dict.items()
                }
            )
            for regime_name, regime_dict in tree.items()
        }
    )


def _period_v_to_array_tree(
    period_to_regime_to_V_arr: MappingProxyType[
        int, MappingProxyType[RegimeName, FloatND]
    ],
) -> dict[str, dict[RegimeName, FloatND]]:
    """Convert the per-period V-array dict into orbax-friendly form."""
    return {
        str(period): dict(regime_dict)
        for period, regime_dict in period_to_regime_to_V_arr.items()
    }


def _array_tree_to_period_v(
    tree: dict[str, dict[RegimeName, FloatND]],
) -> MappingProxyType[int, MappingProxyType[RegimeName, FloatND]]:
    """Inverse of `_period_v_to_array_tree`."""
    return MappingProxyType(
        {
            int(period): MappingProxyType(regime_dict)
            for period, regime_dict in tree.items()
        }
    )


def _flat_params_to_array_tree(
    flat_params: FlatParams,
) -> dict[RegimeName, dict[str, Any]]:
    """Extract only the JAX-array leaves of `flat_params`.

    Non-array leaves (`MappingLeaf`, `SequenceLeaf`, plain Python scalars)
    are skipped here and persisted in the scaffold instead.
    """
    return {
        regime_name: {
            name: value
            for name, value in params.items()
            if isinstance(value, jax.Array)
        }
        for regime_name, params in flat_params.items()
    }


def _flat_params_to_scaffold(
    flat_params: FlatParams,
) -> dict[RegimeName, dict[str, Any]]:
    """Build a scaffold of `flat_params` with array leaves marked by placeholders."""
    return {
        regime_name: {
            name: _ArrayPlaceholder(key=name) if isinstance(value, jax.Array) else value
            for name, value in params.items()
        }
        for regime_name, params in flat_params.items()
    }


def _array_tree_and_scaffold_to_flat_params(
    array_tree: dict[RegimeName, dict[str, Any]],
    scaffold: dict[RegimeName, dict[str, Any]],
) -> FlatParams:
    """Reassemble `flat_params` from the array tree and the scaffold."""
    out: dict[RegimeName, MappingProxyType[str, Any]] = {}
    for regime_name, regime_scaffold in scaffold.items():
        regime_arrays = array_tree.get(regime_name, {})
        regime_params: dict[str, Any] = {}
        for name, value in regime_scaffold.items():
            if isinstance(value, _ArrayPlaceholder):
                regime_params[name] = regime_arrays[value.key]
            else:
                regime_params[name] = value
        out[regime_name] = MappingProxyType(regime_params)
    return MappingProxyType(out)
