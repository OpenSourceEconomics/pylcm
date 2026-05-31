"""User-facing `SimulationResult` with deferred DataFrame computation."""

import gc
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

import cloudpickle
import jax
import jax.numpy as jnp
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
        subject_batch_size: int | None = None,
    ) -> None:
        self._raw_results = raw_results
        self._regimes = regimes
        self._flat_params = flat_params
        self._period_to_regime_to_V_arr = period_to_regime_to_V_arr
        self._ages = ages
        self._subject_batch_size = subject_batch_size
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
        """Raw simulation results by regime and period.

        Leaves are `jax.Array`. When `simulate` ran with `subject_batch_size` set,
        they are host-resident (CPU-backed), since each chunk is offloaded to host
        as it completes; otherwise they live on the compute device.
        """
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
                that target will have NaN. When `simulate` ran with
                `subject_batch_size` set, target evaluation is chunked over subjects
                with that batch size (bounding device memory; values are unchanged).
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
            subject_batch_size=self._subject_batch_size,
        )

        if use_labels:
            return _convert_to_categorical(df=df, metadata=self._metadata)

        return df

    def save(
        self,
        *,
        directory: Path,
        df_additional_targets: list[str] | Literal["all"] | None = None,
        df_use_labels: bool = True,
    ) -> Path:
        """Persist the result to a directory.

        Four sibling artifacts land at the directory root:

        - `arrays/` — orbax checkpoint of the small array trees
          (`raw_results` and `flat_params`), whose individual leaves
          fit comfortably on a single device.
        - `V_arr/` — an orbax checkpoint of the solution value-function
          arrays. orbax streams each leaf's device-to-host transfer (a
          single-device leaf in place, a sharded leaf shard by shard), so
          a near-device-cap leaf does not need a second contiguous device
          buffer at save time.
        - `metadata.pkl` — cloudpickle of regimes, ages, pre-computed
          result metadata, the parameter scaffold, and the per-regime
          chunk specs needed to reassemble on `load`.
        - `simulated_data.arrow` — a feather dump of
          `self.to_dataframe(additional_targets=df_additional_targets,
          use_labels=df_use_labels)`, ready for downstream consumers
          that want the flat per-subject view without re-instantiating
          a `SimulationResult`.

        Args:
            directory: Target directory. Created if it does not exist.
                Must not contain existing `arrays/` or `V_arr/`
                subdirectories.
            df_additional_targets: Targets passed through to `to_dataframe`
                when projecting the on-disk arrow file. `None` (default)
                writes only the base columns (states, actions, regime, age,
                period, V_arr). Pass a list of target names to bake specific
                DAG outputs into the artifact, or `"all"` to include every
                available target — the latter can grow the file by an order
                of magnitude on large models with many DAG leaves.
            df_use_labels: Whether discrete variables are stored as
                pandas `Categorical` labels (default) or integer codes.
                Forwarded to `to_dataframe`.

        Returns:
            The directory the result was written to.

        Notes:
            `save` consumes the in-memory result: on exit both
            `self.period_to_regime_to_V_arr` and `self._regimes` are
            empty mappings. The grid V-array (largest device-resident
            artifact) and the regimes' compiled `simulate_functions` /
            `solve_functions` (which pin XLA program workspaces on the
            device) are released before orbax stages the per-subject
            tree; otherwise the post-V-array D2H allocations exhaust
            the device on smaller GPUs. Callers needing further
            in-memory access after `save` must reload via
            `SimulationResult.load`. `to_dataframe` and `metadata.pkl`
            are produced upfront so the regimes are still available
            for their construction.

        """
        target = directory.resolve()
        target.mkdir(parents=True, exist_ok=True)

        # Save the solution and then drop the in-memory grid immediately. The
        # post-V-array D2H transfers (`to_dataframe`, orbax staging) need the
        # device pool the V-array was occupying.
        _save_period_to_regime_to_V_arr(
            period_to_regime_to_V_arr=self._period_to_regime_to_V_arr,
            output_dir=target / "V_arr",
        )
        self._period_to_regime_to_V_arr = MappingProxyType({})
        gc.collect()

        # Snapshot metadata while `self._regimes` is still populated;
        # the on-disk pickle captures the regime objects so the
        # in-memory copy can be released afterwards.
        metadata = _SavedMetadata(
            regimes=self._regimes,
            flat_params_scaffold=_flat_params_to_scaffold(self._flat_params),
            ages=self._ages,
            result_metadata=self._metadata,
            available_targets=self._available_targets,
            subject_batch_size=self._subject_batch_size,
        )
        with (target / "metadata.pkl").open("wb") as fh:
            cloudpickle.dump(metadata, fh)

        # Drop the compiled `simulate_functions` / `solve_functions`
        # programs inside `self._regimes`; their XLA workspaces stay
        # live until the Python refs go, and the per-period D2H gathers
        # inside `to_dataframe` need a near-empty pool. When
        # `df_additional_targets` is set, the targets DAG needs the
        # compiled programs, so the drop is deferred until after the
        # dataframe is built.
        if df_additional_targets is None:
            self._regimes = MappingProxyType({})
            gc.collect()

        # Defer the arrow write until after orbax succeeds so a partial
        # save doesn't leave stale per-subject data on disk.
        df = self.to_dataframe(
            additional_targets=df_additional_targets,
            use_labels=df_use_labels,
        )
        # Feather columns must be homogeneous. `to_dataframe` can leave
        # JAX 0-d arrays in object columns (e.g. a regime whose target
        # function returns a constant gets broadcast as a 0-d JAX scalar
        # across the per-regime sub-frame); coerce them to Python scalars.
        df = df.map(_coerce_jax_scalar_for_arrow)

        if self._regimes:
            self._regimes = MappingProxyType({})
            gc.collect()

        small_array_tree = {
            "raw_results": _raw_results_to_array_tree(self._raw_results),
            "flat_params": _flat_params_to_array_tree(self._flat_params),
        }
        _log_top_array_tree_leaves(
            tree=small_array_tree, top_k=20, label="save: small_array_tree"
        )
        checkpointer = ocp.StandardCheckpointer()
        checkpointer.save(target / "arrays", small_array_tree)
        # `StandardCheckpointer.save` is asynchronous; block until the
        # on-disk checkpoint is complete so the sibling
        # `simulated_data.arrow` write and any subsequent `load`
        # observe a consistent directory.
        checkpointer.wait_until_finished()
        df.to_feather(target / "simulated_data.arrow")

        return target

    @classmethod
    def load(cls, *, directory: Path) -> SimulationResult:
        """Read a result from a directory produced by `save`.

        Reads `arrays/` (orbax, small trees), `V_arr/` (chunked `.npy`
        files), and `metadata.pkl` (cloudpickle). The `simulated_data.arrow`
        artifact is not consumed — `to_dataframe` re-derives it on
        demand. Sharded arrays inside `arrays/` are restored onto the
        same sharding they had at save time, so no implicit gather
        happens during load.
        """
        source = directory.resolve()

        with (source / "metadata.pkl").open("rb") as fh:
            metadata: _SavedMetadata = cloudpickle.load(fh)

        checkpointer = ocp.StandardCheckpointer()
        array_tree = checkpointer.restore(source / "arrays")

        raw_results = _array_tree_to_raw_results(array_tree["raw_results"])
        period_to_regime_to_V_arr = _load_period_to_regime_to_V_arr(
            input_dir=source / "V_arr"
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
        instance._subject_batch_size = metadata.subject_batch_size  # noqa: SLF001
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

    subject_batch_size: int | None = None
    """Subject chunk size from `simulate`, reused to bound `to_dataframe` targets."""


@dataclass(frozen=True)
class _ArrayPlaceholder:
    """Marker for a JAX array slot in a cloudpickled scaffold."""

    key: str


@dataclass(frozen=True)
class _ArrayTreeLeaf:
    """Size record for one `jax.Array` leaf in a save-time array tree."""

    path: str
    """Dotted path from the tree root, e.g. `raw_results.regime_A.7.V_arr`."""

    shape: tuple[int, ...]
    """Leaf array shape."""

    dtype: jnp.dtype
    """Leaf array dtype."""

    n_bytes: int
    """`prod(shape) * dtype.itemsize` — what orbax must stage to host."""


def _coerce_jax_scalar_for_arrow(value: object) -> object:
    """Convert a 0-d JAX array to a Python scalar; pass everything else through."""
    if isinstance(value, jax.Array) and value.ndim == 0:
        return value.item()
    return value


def _collect_array_tree_leaf_sizes(
    *,
    tree: dict[str, Any],
) -> list[_ArrayTreeLeaf]:
    """Walk `tree` and return one `_ArrayTreeLeaf` per `jax.Array` leaf.

    Results come back sorted by `n_bytes` descending so callers can log the
    biggest offenders first. Non-array leaves are skipped silently — orbax
    serialises only the array entries.
    """
    leaves: list[_ArrayTreeLeaf] = []
    _walk_tree(node=tree, path_parts=(), leaves=leaves)
    leaves.sort(key=lambda leaf: leaf.n_bytes, reverse=True)
    return leaves


def _walk_tree(
    *,
    node: object,
    path_parts: tuple[str, ...],
    leaves: list[_ArrayTreeLeaf],
) -> None:
    """Recursively collect `jax.Array` leaves into `leaves`."""
    if isinstance(node, jax.Array):
        leaves.append(
            _ArrayTreeLeaf(
                path=".".join(path_parts),
                shape=tuple(node.shape),
                dtype=node.dtype,
                n_bytes=int(node.size) * node.dtype.itemsize,
            )
        )
        return
    if isinstance(node, Mapping):
        for key, value in node.items():
            _walk_tree(node=value, path_parts=(*path_parts, str(key)), leaves=leaves)


def _log_top_array_tree_leaves(
    *,
    tree: dict[str, Any],
    top_k: int,
    label: str,
) -> None:
    """Emit the `top_k` biggest `jax.Array` leaves plus aggregate tree size.

    Writes directly to `sys.stderr` so the lines surface even when the
    `lcm` logger is silenced (`log_level="off"` raises it to CRITICAL).
    Each line carries path, shape, dtype, and size in GiB; the leading
    line reports total leaf count and aggregate bytes.
    """
    leaves = _collect_array_tree_leaf_sizes(tree=tree)
    total_bytes = sum(leaf.n_bytes for leaf in leaves)
    total_gib = total_bytes / (1024**3)
    print(  # noqa: T201
        f"[{label}] total: {len(leaves)} jax.Array leaves, "
        f"{total_bytes:,} bytes ({total_gib:.3f} GiB). Top {top_k}:",
        file=sys.stderr,
        flush=True,
    )
    for leaf in leaves[:top_k]:
        gib = leaf.n_bytes / (1024**3)
        print(  # noqa: T201
            f"  {gib:>8.4f} GiB  shape={leaf.shape!s:<24} "
            f"dtype={leaf.dtype!s:<10} path={leaf.path}",
            file=sys.stderr,
            flush=True,
        )


def _save_period_to_regime_to_V_arr(
    *,
    period_to_regime_to_V_arr: MappingProxyType[
        int, MappingProxyType[RegimeName, FloatND]
    ],
    output_dir: Path,
) -> None:
    """Persist the solution as a single orbax checkpoint.

    orbax serialises each leaf with a streaming device-to-host transfer — a
    single-device leaf is read in place (no second contiguous device buffer) and a
    sharded leaf is transferred shard by shard — so a near-device-cap leaf does not
    blow up at save time. Periods are stringified so orbax can use them as path
    components.
    """
    checkpointer = ocp.StandardCheckpointer()
    checkpointer.save(output_dir, _period_V_to_array_tree(period_to_regime_to_V_arr))
    checkpointer.wait_until_finished()


def _load_period_to_regime_to_V_arr(
    *,
    input_dir: Path,
) -> MappingProxyType[int, MappingProxyType[RegimeName, FloatND]]:
    """Inverse of `_save_period_to_regime_to_V_arr`.

    Each leaf is restored onto the sharding it was saved with; loading runs on the
    same backend the checkpoint was written from (a GPU box for solve/simulate).
    """
    array_tree = ocp.StandardCheckpointer().restore(input_dir)
    return _array_tree_to_period_V(array_tree)


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


def _period_V_to_array_tree(
    period_to_regime_to_V_arr: MappingProxyType[
        int, MappingProxyType[RegimeName, FloatND]
    ],
) -> dict[str, dict[RegimeName, FloatND]]:
    """Convert the per-period V-array dict into orbax-friendly form.

    Periods are stringified so orbax can use them as path components.
    """
    return {
        str(period): dict(regime_dict)
        for period, regime_dict in period_to_regime_to_V_arr.items()
    }


def _array_tree_to_period_V(
    tree: dict[str, dict[RegimeName, FloatND]],
) -> MappingProxyType[int, MappingProxyType[RegimeName, FloatND]]:
    """Inverse of `_period_V_to_array_tree`."""
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
