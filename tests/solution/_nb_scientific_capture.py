"""Capture original income-fixture outputs without replacing any computed result."""

import hashlib
import json
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import TypedDict

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.typing import ArrayLike

from _lcm import dtypes
from _lcm.egm.nbegm_step import _NBEGMQueryStep
from _lcm.params import processing as parameter_processing
from _lcm.solution import nbegm
from _lcm.typing import FlatParams
from lcm import Model
from lcm.result import SimulationResult
from lcm.solver_api import SolutionResult
from tests.solution import test_state_dependent_borrowing_limit as fixture
from tests.solution._nb_scientific_capture_admission import (
    admit_capture,
    admit_processed_params,
    bind_source_files,
    drain_owned_callbacks,
    encode_object_strings,
    source_identity,
    verify_capture_files,
)

_CAPTURE_LOCK = Lock()


type _Json = (
    str | int | float | bool | list[_Json] | tuple[_Json, ...] | dict[str, _Json] | None
)


class _ArrayMetadata(TypedDict):
    shape: list[int]
    dtype: str
    original_dtype: str
    encoding: str


class _SavedEntry(TypedDict):
    file: str
    sha256: str
    arrays: dict[str, _ArrayMetadata]
    original_query_leaf_names: list[str] | None


class _Invocation(TypedDict, total=False):
    solver: str
    operation: str
    schemas: list[dict[str, _Json]]
    returned_original_result: bool
    unpublished_GS_M_C_A_status_topology: str
    seed: int
    rows: int
    period0_rows: int


class _ProcessedLeaf(TypedDict):
    array: str
    path: list[str]
    object_id: int
    python_type: str
    native_dtype: str
    native_shape: list[int]
    bit_pattern_hex: str


class _ProcessedCall(TypedDict):
    solver: str
    operation: str
    file: str
    method: str
    model_object_id: int
    returned_mapping_id: int
    returned_original_mapping: bool
    leaves: list[_ProcessedLeaf]
    scope: str


class _CaptureManifest(TypedDict):
    meaning: str
    precision: int
    jax: str
    devices: list[str]
    source_map: str | None
    source_map_sha256: str | None
    source_root: str
    provenance_scope: str
    source_map_externally_admitted: bool
    source_files: dict[str, str]
    calls: list[_Invocation]
    rows: list[str]
    saved: list[_SavedEntry]
    processed_parameter_calls: list[_ProcessedCall]
    original_budget: float
    threshold_exceedances: int
    original_assertion_retained: bool
    original_comparison_role: str
    row_matches: list[list[int]]
    panels: dict[str, list[int]]
    weights: str
    limitations: list[str]


def _source_files(
    *, original_step, original_solve, original_simulate, original_process
):
    expected = (
        (fixture, fixture),
        (nbegm, nbegm),
        (Model, sys.modules["lcm.model"]),
        (original_step, nbegm),
        (original_solve, sys.modules["lcm.model"]),
        (original_simulate, sys.modules["lcm.model"]),
        (original_process, sys.modules["lcm.model"]),
        (parameter_processing, parameter_processing),
        (dtypes, dtypes),
    )
    return dict(source_identity(obj=obj, module=module) for obj, module in expected)


def _encode_arrays(arrays):
    converted, metadata = {}, {}
    for name, value in arrays.items():
        array = np.asarray(value)
        original_dtype = str(array.dtype)
        encoding = "identity"
        if array.dtype.kind == "O":
            encoded, dtype = encode_object_strings(list(array.flat))
            array = np.asarray(encoded, dtype=dtype).reshape(array.shape)
            encoding = "json-string-ascii-in-fixed-unicode"
        converted[name] = array
        metadata[name] = {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "original_dtype": original_dtype,
            "encoding": encoding,
        }
    return converted, metadata


def _row_matches(*, root, rows):
    matches = []
    with np.load(root / "nbegm-public-values.npz", allow_pickle=False) as public:
        for filename in rows:
            with np.load(root / filename, allow_pickle=False) as row:
                owners = [
                    [period, income]
                    for period in (0, 1)
                    for income in (0, 1)
                    if np.array_equal(
                        row["published_V"], public[f"p{period}__saving"][income]
                    )
                ]
            if len(owners) != 1:
                raise ValueError(
                    "Published row has missing or ambiguous period/income identity"
                )
            matches.append(owners[0])
    return matches


def _save_solve(*, self, result, kwargs, solver, save, calls):
    jax.effects_barrier()
    save(
        filename=f"{solver}-public-values.npz",
        arrays={
            f"p{period}__{regime}": value
            for period, regimes in result.values.items()
            for regime, value in regimes.items()
        },
    )
    coordinates = {}
    schemas = []
    for (period, regime), schema in result.metadata.value_schemas.items():
        schemas.append(
            {
                "period": period,
                "regime": regime,
                "axis_names": schema.axis_names,
                "shape": schema.shape,
            }
        )
        for axis in schema.named_axes:
            coordinates[f"p{period}__{regime}__{axis.name}"] = axis.coordinates
    save(filename=f"{solver}-value-coordinates.npz", arrays=coordinates)
    # The actual finalized public action declaration consumed by this model.
    action_grid = self.user_regimes["saving"].actions["consumption"]
    save(filename=f"{solver}-actions.npz", arrays={"consumption": action_grid.to_jax()})
    # These are supplied Python declarations, not the processed model-owned tree.
    save(filename=f"{solver}-params.npz", arrays=dict(kwargs["params"]))
    income_axis = result.metadata.value_schemas[1, "saving"].named_axes
    income_codes = np.asarray(
        next(axis.coordinates for axis in income_axis if axis.name == "income")
    )
    native_income_codes = jnp.asarray(income_codes)
    save(
        filename=f"{solver}-income-primitives.npz",
        arrays={
            "income_codes": income_codes,
            "transition_rows": np.stack(
                [np.asarray(fixture.income_transition(code)) for code in income_codes]
            ),
            "income_amounts": fixture.income_of_node(native_income_codes),
            "declared_floor": fixture.income_lower(
                income=native_income_codes, limit_share=fixture._LIMIT_SHARE
            ),
        },
    )
    calls.append(
        {
            "solver": solver,
            "operation": "solve",
            "schemas": schemas,
            "returned_original_result": True,
            "unpublished_GS_M_C_A_status_topology": "not available; not synthesized",
        }
    )


@dataclass(frozen=True, kw_only=True)
class NativeIncomeCapture:
    """Actual objects shared by native assertions and the same-invocation writer."""

    manifest: _CaptureManifest
    models: dict[str, Model]
    solutions: dict[str, SolutionResult]
    simulations: dict[str, SimulationResult]
    processed_params: dict[tuple[str, str], FlatParams]
    actions: dict[str, np.ndarray]


def _processed_leaves(result: FlatParams):
    """Walk actual qualified mapping paths without constructing parameter values."""
    arrays, leaves = {}, []

    def visit(*, value, path):
        if isinstance(value, Mapping):
            for name, child in value.items():
                if not isinstance(name, str):
                    raise TypeError(
                        "Processed parameter path contains a non-string key"
                    )
                visit(value=child, path=(*path, name))
            return
        if not isinstance(value, (jax.Array, np.ndarray, np.generic)):
            raise TypeError("Expected actual native numeric processed parameter leaf")
        if value.dtype.kind not in "bifu":
            raise TypeError("Processed parameter leaf has an unsupported native dtype")
        key = f"leaf{len(leaves):04d}"
        arrays[key] = value
        leaves.append(
            {
                "array": key,
                "path": list(path),
                "object_id": id(value),
                "python_type": f"{type(value).__module__}.{type(value).__qualname__}",
                "native_dtype": str(value.dtype),
                "native_shape": list(value.shape),
                "bit_pattern_hex": np.asarray(value).tobytes(order="C").hex(),
            }
        )

    visit(value=result, path=())
    return arrays, leaves


def _save_processed_params(*, result, model, solver, operation, save, calls):
    """Snapshot the completed canonical return while forwarding its original tree."""
    if solver not in ("nbegm", "grid") or operation not in ("solve", "simulate"):
        raise ValueError("Processed parameters have no actual model/operation owner")
    arrays, leaves = _processed_leaves(result)
    filename = f"{solver}-{operation}-processed-params.npz"
    save(filename=filename, arrays=arrays)
    calls.append(
        {
            "solver": solver,
            "operation": operation,
            "file": filename,
            "method": "lcm.model.Model._canonicalize_params",
            "model_object_id": id(model),
            "returned_mapping_id": id(result),
            "returned_original_mapping": True,
            "leaves": leaves,
            "scope": (
                "actual canonical method result the call consumes; "
                "no nominal substitution"
            ),
        }
    )


def _create_step_observer(*, original_step, active, rows, save, lock):
    """Forward each real NB step once and save only its original callback leaves."""

    def observe_step(**kwargs):
        result = original_step(**kwargs)
        assert kwargs["retain_query"] is True
        assert isinstance(result, _NBEGMQueryStep)
        names = [
            "continuation_V",
            "continuation_M",
            "liquid",
            "savings_nodes",
            "discount",
            "coh_slopes",
            "coh_intercepts",
            "breakpoints",
            "published_V",
            "published_M",
            "published_C",
        ]
        leaves = [
            kwargs["cont_value"],
            kwargs["cont_marginal"],
            kwargs["liquid_grid"],
            kwargs["savings_grid"],
            kwargs["discount_factor"],
            kwargs["coh_slopes"],
            kwargs["coh_intercepts"],
            kwargs["breakpoints"],
            result.value,
            result.marginal,
            result.policy,
        ]
        for path, leaf in result.query_data.leaves().items():
            names.append("query__" + "__".join(path))
            leaves.append(leaf)
        for name in (
            "extra_savings",
            "extra_cont_value",
            "inverse_eis",
            "feasible_interval_mask",
        ):
            if kwargs.get(name) is not None:
                names.append(name)
                leaves.append(kwargs[name])
        solver = active["solver"]

        def record(*actual_leaves) -> None:
            with lock:
                filename = f"{solver}-row-{len(rows):04d}.npz"
                save(
                    filename=filename,
                    arrays=dict(zip(names, actual_leaves, strict=True)),
                    original_query_leaf_names=sorted(
                        name for name in names if name.startswith("query__")
                    ),
                )
                rows.append(filename)

        jax.debug.callback(record, *leaves)
        return result

    return observe_step


def _create_model_observers(
    *,
    original_process,
    original_solve,
    original_simulate,
    active,
    processed,
    processed_calls,
    solutions,
    simulations,
    models,
    panels,
    calls,
    save,
):
    """Forward each model method once while retaining its original return object."""

    def observe_process(self, *args, **kwargs):
        result = original_process(self, *args, **kwargs)
        solver, operation = active["solver"], active["operation"]
        key = (solver, operation)
        if key in processed:
            raise ValueError(
                "Repeated processed-parameter call for one actual operation"
            )
        _save_processed_params(
            result=result,
            model=self,
            solver=solver,
            operation=operation,
            save=save,
            calls=processed_calls,
        )
        processed[key] = result
        return result

    def observe_solve(self, *args, **kwargs):
        active["operation"] = "solve"
        result = original_solve(self, *args, **kwargs)
        solutions[active["solver"]] = result
        models[active["solver"]] = self
        _save_solve(
            self=self,
            result=result,
            kwargs=kwargs,
            solver=active["solver"],
            save=save,
            calls=calls,
        )
        return result

    def observe_simulate(self, *args, **kwargs):
        active["operation"] = "simulate"
        solver = active["solver"]
        save(
            filename=f"{solver}-initial.npz", arrays=dict(kwargs["initial_conditions"])
        )
        result = original_simulate(self, *args, **kwargs)
        if models[solver] is not self:
            raise ValueError(
                "Simulation uses a different model than its observed solve"
            )
        simulations[solver] = result
        jax.effects_barrier()
        frame = result.to_dataframe(use_categorical_codes=True)
        save(
            filename=f"{solver}-simulation.npz",
            arrays={name: frame[name].to_numpy() for name in frame},
        )
        period0 = frame.query("period == 0").sort_values("subject_id")
        panels[solver] = period0["subject_id"].to_list()
        assert len(period0) == 60
        np.testing.assert_array_equal(period0["subject_id"].to_numpy(), np.arange(60))
        save(
            filename=f"{solver}-period0.npz",
            arrays={name: period0[name].to_numpy() for name in period0},
        )
        calls.append(
            {
                "solver": solver,
                "operation": "simulate",
                "seed": kwargs["seed"],
                "rows": len(frame),
                "period0_rows": len(period0),
                "returned_original_result": True,
            }
        )
        return result

    return observe_process, observe_solve, observe_simulate


def _finalize_capture_manifest(
    *,
    root,
    source_map,
    source_bytes,
    observed_root,
    files,
    calls,
    rows,
    saved,
    processed_calls,
    panels,
    actions,
    save,
) -> _CaptureManifest:
    """Save the unchanged diagnostic and admit all same-invocation observations."""
    assert len(rows) == 4, "Expected two income rows at each original saving period."
    save(filename="actual-period0-actions.npz", arrays=actions)
    # Retain the exact original policy comparison as a diagnostic.
    difference = np.abs(actions["nbegm"] - actions["grid"])
    budget = 3 * fixture._CONSUMPTION_STEP
    save(
        filename="original-policy-comparison.npz",
        arrays={
            "absolute_difference": difference,
            "violates_original_budget": difference > budget,
            "original_budget": budget,
        },
    )
    if source_map is not None:
        assert source_map.read_bytes() == source_bytes
    manifest: _CaptureManifest = {
        "meaning": (
            "Pass-through genuine native observations; no scientific acceptance "
            "inferred from capture"
        ),
        "precision": 64 if jax.config.x64_enabled else 32,
        "jax": jax.__version__,
        "devices": [str(device) for device in jax.devices()],
        "source_map": None if source_map is None else str(source_map),
        "source_map_sha256": None
        if source_bytes is None
        else hashlib.sha256(source_bytes).hexdigest(),
        "source_root": str(observed_root),
        "provenance_scope": "observed_module_sources"
        if source_map is None
        else "externally_admitted_whole_source",
        "source_map_externally_admitted": source_map is not None,
        "source_files": files,
        "calls": calls,
        "rows": rows,
        "saved": saved,
        "processed_parameter_calls": processed_calls,
        "original_budget": budget,
        "threshold_exceedances": int(np.sum(difference > budget)),
        "original_assertion_retained": False,
        "original_comparison_role": "retained_diagnostic",
        "row_matches": _row_matches(root=root, rows=rows),
        "panels": panels,
        "weights": (
            "Saved actual fixture transition rows at named-axis income codes; "
            "fixed panel equally weighted"
        ),
        "limitations": [
            "No GS unpublished marginal/policy topology manufactured",
            "No full internal 150x120 candidate mesh claimed",
            "No raw BoundInputStatus or QueryInputStatus summary flags at this seam",
            (
                "*-params.npz holds supplied nominal declarations; operation-specific "
                "processed files hold actual model-owned method results"
            ),
            (
                "Transition/income/floor auxiliary exports are source-bound fixture "
                "evaluations, not captured internal invocation results"
            ),
            "No resource or certified production acceptance from observer",
        ],
    }
    admit_capture(
        manifest=manifest,
        precision=manifest["precision"],
        source_sha=manifest["source_map_sha256"],
    )
    admit_processed_params(manifest=manifest, precision=manifest["precision"])
    verify_capture_files(manifest=manifest, root=root)
    (root / "CAPTURE.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def capture_original_nb_income(
    *,
    root: Path,
    source_map: Path | None = None,
    source_root: Path | None = None,
    monkeypatch: pytest.MonkeyPatch,
) -> NativeIncomeCapture:
    """Observe the original two-model solve/simulation invocation once per method."""
    if (source_map is None) != (source_root is None):
        raise ValueError("Audit capture requires both source map and source root")
    source_bytes = None if source_map is None else source_map.read_bytes()
    observed_root = (
        Path(fixture.__file__).resolve().parents[2]
        if source_root is None
        else source_root.resolve()
    )
    root.mkdir(parents=True, exist_ok=False)
    lock = Lock()
    active = {"solver": "unset", "operation": "unset"}
    rows: list[str] = []
    calls: list[_Invocation] = []
    saved: list[_SavedEntry] = []
    panels = {}
    solutions = {}
    simulations = {}
    models = {}
    processed = {}
    processed_calls = []
    original_step = nbegm._solve_ride_along_cell_step
    original_solve = Model.solve
    original_simulate = Model.simulate
    original_process = Model._canonicalize_params

    files = _source_files(
        original_step=original_step,
        original_solve=original_solve,
        original_simulate=original_simulate,
        original_process=original_process,
    )
    bind_source_files(
        files=files,
        expected_map=None if source_bytes is None else json.loads(source_bytes),
        source_root=observed_root,
    )

    def save(
        *,
        filename: str,
        arrays: Mapping[str, ArrayLike],
        original_query_leaf_names=None,
    ) -> None:
        converted, metadata = _encode_arrays(arrays)
        path = root / filename
        assert not path.exists(), filename
        np.savez(path, **converted)
        saved.append(
            {
                "file": filename,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "arrays": metadata,
                "original_query_leaf_names": original_query_leaf_names,
            }
        )

    observe_step = _create_step_observer(
        original_step=original_step, active=active, rows=rows, save=save, lock=lock
    )
    observe_process, observe_solve, observe_simulate = _create_model_observers(
        original_process=original_process,
        original_solve=original_solve,
        original_simulate=original_simulate,
        active=active,
        processed=processed,
        processed_calls=processed_calls,
        solutions=solutions,
        simulations=simulations,
        models=models,
        panels=panels,
        calls=calls,
        save=save,
    )

    actions = {}
    with _CAPTURE_LOCK, monkeypatch.context() as hooks:
        hooks.setattr(Model, "_canonicalize_params", observe_process)
        hooks.setattr(nbegm, "_solve_ride_along_cell_step", observe_step)
        hooks.setattr(Model, "solve", observe_solve)
        hooks.setattr(Model, "simulate", observe_simulate)
        with drain_owned_callbacks(jax.effects_barrier):
            for solver in ("nbegm", "grid"):
                active["solver"] = solver
                actions[solver] = fixture._period0_consumption(
                    solver=solver, limit="income", lower_kind="function", taxed=True
                )
                assert actions[solver].shape == (60,)
        jax.effects_barrier()
    manifest = _finalize_capture_manifest(
        root=root,
        source_map=source_map,
        source_bytes=source_bytes,
        observed_root=observed_root,
        files=files,
        calls=calls,
        rows=rows,
        saved=saved,
        processed_calls=processed_calls,
        panels=panels,
        actions=actions,
        save=save,
    )
    return NativeIncomeCapture(
        manifest=manifest,
        models=models,
        solutions=solutions,
        simulations=simulations,
        processed_params=processed,
        actions=actions,
    )
