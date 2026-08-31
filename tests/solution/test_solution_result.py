"""Public, labelled solve results and their artifact-retention contract."""

import ast
import hashlib
import inspect
from dataclasses import replace
from types import MappingProxyType
from typing import cast

import cloudpickle
import jax.numpy as jnp
import numpy as np
import pytest
from pandas.testing import assert_frame_equal

import lcm.model as model_module
from _lcm.egm.nested_published_policy import NestedEGMSimPolicy
from _lcm.egm.published_policy import EGMSimPolicy, NNBEGMSimPolicy
from _lcm.engine import EGMPolicyRead
from _lcm.solution import artifacts as private_artifacts
from _lcm.solution import backward_induction
from _lcm.solution.solver_diagnostics import SolverDiagnostics
from _lcm.typing import (
    FlatParams,
    PeriodToRegimeToDissolutionFlags,
    PeriodToRegimeToSimulationPolicy,
    PeriodToRegimeToVArr,
)
from lcm import LinSpacedGrid, Model
from lcm.exceptions import InvalidSimulationInputError
from lcm.solver_api import (
    DISSOLUTION_FLAG,
    EGM_CONTINUATION,
    SIMULATION_POLICY,
    SOLVER_DIAGNOSTICS,
    ArtifactKey,
    ArtifactRef,
    ArtifactStore,
    OmissionReason,
    ResultRetention,
    SolutionMetadata,
    SolutionResult,
    ValueArraySchema,
)
from lcm.typing import UserInitialConditions, UserParams
from tests.regime_building.test_collective_regime_simulate import (
    _DISSOLUTION_PARAMS,
    _make_dissolution_model,
)
from tests.simulation.test_nnbegm_split_workflow_parity import (
    _INITIAL,
    _PARAMS,
    _build,
)
from tests.solution.test_egm_published_policy import _two_period_bequest_model
from tests.test_models.deterministic.dcegm_variants import (
    get_retirement_only_params,
)
from tests.test_models.deterministic.regression import (
    RegimeId,
    get_model,
    get_params,
)


def test_solver_api_has_no_private_lcm_imports() -> None:
    """An installed solver can import the result spine without importing `_lcm`."""
    solver_api_module = inspect.getmodule(ArtifactKey)
    assert solver_api_module is not None
    module = ast.parse(inspect.getsource(solver_api_module))
    imported = {
        alias.name
        for node in ast.walk(module)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module or ""
        for node in ast.walk(module)
        if isinstance(node, ast.ImportFrom)
    }

    assert not any(name == "_lcm" or name.startswith("_lcm.") for name in imported)


def test_artifact_identity_includes_schema_version() -> None:
    policy_v1 = ArtifactKey(type_id="example.policy", schema_version=1)
    policy_v2 = ArtifactKey(type_id="example.policy", schema_version=2)

    assert policy_v1 != policy_v2
    assert ArtifactRef(period=3, regime="alive", key=policy_v1) != ArtifactRef(
        period=3, regime="alive", key=policy_v2
    )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"type_id": "", "schema_version": 1}, "type_id"),
        ({"type_id": "example.policy", "schema_version": 0}, "schema_version"),
    ],
)
def test_invalid_artifact_key_is_rejected(
    *, kwargs: dict[str, object], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        ArtifactKey(**kwargs)  # ty: ignore[invalid-argument-type]


def test_artifact_store_is_immutable_and_projects_one_artifact_type() -> None:
    policy = ArtifactKey(type_id="example.policy")
    diagnostic = ArtifactKey(type_id="example.diagnostic")
    refs = {
        ArtifactRef(period=1, regime="alive", key=policy): "p1",
        ArtifactRef(period=0, regime="alive", key=policy): "p0",
        ArtifactRef(period=0, regime="alive", key=diagnostic): "d0",
    }
    store = ArtifactStore(refs)

    assert dict(store) == refs
    projected = store.project(policy)
    assert projected == {0: {"alive": "p0"}, 1: {"alive": "p1"}}
    assert isinstance(projected, MappingProxyType)
    assert all(isinstance(inner, MappingProxyType) for inner in projected.values())
    with pytest.raises(TypeError):
        projected[0]["alive"] = "changed"  # ty: ignore[invalid-assignment]


def test_solution_result_keeps_values_explicit_and_immutable() -> None:
    values = {0: {"alive": jnp.asarray([1.0, 2.0])}}
    metadata = SolutionMetadata(
        retention=ResultRetention.VALUES,
        n_periods=1,
        regime_names=("alive",),
        solver_types={"alive": "example.Grid"},
        model_instance_id="model-1",
        params_fingerprint="0" * 64,
        value_schemas={
            (0, "alive"): ValueArraySchema(
                shape=(2,), dtype="float32", axis_names=("wealth",)
            )
        },
    )
    result = SolutionResult(values=values, metadata=metadata)

    np.testing.assert_array_equal(result.value(period=0, regime="alive"), [1.0, 2.0])
    assert isinstance(result.values, MappingProxyType)
    assert isinstance(result.values[0], MappingProxyType)  # noqa: PD011
    assert isinstance(result.omissions, MappingProxyType)
    with pytest.raises(TypeError):
        result.values[0]["alive"] = jnp.asarray([9.0])  # noqa: PD011  # ty: ignore[invalid-assignment]


def test_solve_result_records_instance_params_and_value_array_schemas() -> None:
    model, params, _ = _small_grid_search_inputs()

    result = model.solve_result(params=params, log_level="off")

    assert result.metadata.model_instance_id
    assert len(result.metadata.params_fingerprint) == 64
    value_store = result.values  # noqa: PD011
    assert set(result.metadata.value_schemas) == {
        (period, regime_name)
        for period, regime_to_value in value_store.items()
        for regime_name in regime_to_value
    }
    for coordinate, schema in result.metadata.value_schemas.items():
        period, regime_name = coordinate
        value = value_store[period][regime_name]
        assert schema.shape == value.shape
        assert schema.dtype == str(value.dtype)


def test_flat_param_fingerprint_frames_marker_like_path_components() -> None:
    array = jnp.asarray([1.0], dtype=jnp.float32)
    flat_params = cast(
        "FlatParams",
        MappingProxyType(
            {"alive": MappingProxyType({"array": array})},
        ),
    )

    def _digest(tokens: tuple[str | bytes, ...]) -> str:
        digest = hashlib.sha256()
        for token in tokens:
            payload = token.encode() if isinstance(token, str) else token
            digest.update(len(payload).to_bytes(8, byteorder="big"))
            digest.update(payload)
        return digest.hexdigest()

    canonical = np.ascontiguousarray(np.asarray(array))
    suffix: tuple[str | bytes, ...] = (
        "alive",
        "array",
        "array",
        "1",
        canonical.dtype.str,
        canonical.tobytes(order="C"),
    )
    framed = _digest(("path", "2", *suffix))
    ambiguous_legacy_encoding = _digest(("path", *suffix))

    actual = private_artifacts.fingerprint_flat_params(flat_params)

    assert actual == framed
    assert actual != ambiguous_legacy_encoding


def test_model_solve_result_retains_replay_and_labels_unretained_continuations() -> (
    None
):
    model = _two_period_bequest_model()
    params = get_retirement_only_params(n_periods=2, discount_factor=0.98)

    result = model.solve_result(params=params, log_level="off")

    assert result.metadata.retention is ResultRetention.VALUES_AND_REPLAY
    policies = result.replay_artifacts.project(SIMULATION_POLICY)
    assert 0 in policies
    assert "retirement" in policies[0]
    continuation_ref = ArtifactRef(
        period=0,
        regime="retirement",
        key=EGM_CONTINUATION,
    )
    assert result.omissions[continuation_ref] is OmissionReason.NOT_REQUESTED
    assert result.metadata.solver_api_version == 1
    assert not result.diagnostics


def test_values_only_result_drops_replay_with_an_explicit_reason() -> None:
    model = _two_period_bequest_model()
    params = get_retirement_only_params(n_periods=2, discount_factor=0.98)

    result = model.solve_result(
        params=params,
        log_level="off",
        retention=ResultRetention.VALUES,
    )

    assert not result.replay_artifacts
    policy_ref = ArtifactRef(
        period=0,
        regime="retirement",
        key=SIMULATION_POLICY,
    )
    assert result.omissions[policy_ref] is OmissionReason.NOT_REQUESTED


def test_all_persistable_marks_unretained_continuation_unsupported() -> None:
    model = _two_period_bequest_model()
    params = get_retirement_only_params(n_periods=2, discount_factor=0.98)

    result = model.solve_result(
        params=params,
        log_level="off",
        retention=ResultRetention.ALL_PERSISTABLE_ARTIFACTS,
    )

    continuation_ref = ArtifactRef(
        period=0,
        regime="retirement",
        key=EGM_CONTINUATION,
    )
    assert result.omissions[continuation_ref] is OmissionReason.UNSUPPORTED


def test_solve_result_retains_kernel_diagnostics_only_when_log_level_enables_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = backward_induction._run_period_kernel

    def _with_diagnostics(**kwargs: object):
        kernel_result = original(**kwargs)  # ty: ignore[invalid-argument-type]
        scalar = jnp.asarray(0.0)
        flag = jnp.zeros((), dtype=jnp.bool_)
        diagnostics = SolverDiagnostics(
            max_outer_interpolation_error=scalar,
            max_outer_bracket_width=scalar,
            outer_nodes_used=jnp.asarray(1, dtype=jnp.int32),
            outer_at_lower_bound=flag,
            outer_at_upper_bound=flag,
            keeper_adjuster_margin=scalar,
            best_second_best_margin=scalar,
            policy_fallback_mask=flag,
            unresolved_mask=flag,
            n_outer_all_invalid_cells=jnp.asarray(0, dtype=jnp.int32),
        )
        return replace(kernel_result, diagnostics=diagnostics)

    monkeypatch.setattr(backward_induction, "_run_period_kernel", _with_diagnostics)
    model = _two_period_bequest_model()
    params = get_retirement_only_params(n_periods=2, discount_factor=0.98)

    enabled = model.solve_result(params=params, log_level="warning")
    disabled = model.solve_result(params=params, log_level="off")

    retained = enabled.diagnostics.project(SOLVER_DIAGNOSTICS)
    assert retained
    assert not disabled.diagnostics
    assert all(
        device.platform == "cpu"
        for regime_to_diagnostics in retained.values()
        for diagnostics in regime_to_diagnostics.values()
        for device in cast(
            "SolverDiagnostics", diagnostics
        ).max_outer_interpolation_error.devices()
    )


def test_legacy_solve_does_not_retain_solver_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = backward_induction._run_period_kernel

    def _with_diagnostics(**kwargs: object):
        kernel_result = original(**kwargs)  # ty: ignore[invalid-argument-type]
        scalar = jnp.asarray(0.0)
        flag = jnp.zeros((), dtype=jnp.bool_)
        return replace(
            kernel_result,
            diagnostics=SolverDiagnostics(
                max_outer_interpolation_error=scalar,
                max_outer_bracket_width=scalar,
                outer_nodes_used=jnp.asarray(1, dtype=jnp.int32),
                outer_at_lower_bound=flag,
                outer_at_upper_bound=flag,
                keeper_adjuster_margin=scalar,
                best_second_best_margin=scalar,
                policy_fallback_mask=flag,
                unresolved_mask=flag,
                n_outer_all_invalid_cells=jnp.asarray(0, dtype=jnp.int32),
            ),
        )

    def _must_not_copy(**_kwargs: object) -> SolverDiagnostics:
        raise AssertionError("legacy solve retained a solver diagnostic")

    monkeypatch.setattr(backward_induction, "_run_period_kernel", _with_diagnostics)
    monkeypatch.setattr(
        backward_induction, "_copy_solver_diagnostics_to_host", _must_not_copy
    )
    model = _two_period_bequest_model()
    params = get_retirement_only_params(n_periods=2, discount_factor=0.98)

    model.solve(params=params, log_level="warning")


def test_builtin_artifact_keys_are_stably_versioned() -> None:
    assert {
        key.type_id: key.schema_version
        for key in (
            EGM_CONTINUATION,
            SIMULATION_POLICY,
            DISSOLUTION_FLAG,
            SOLVER_DIAGNOSTICS,
        )
    } == {
        "pylcm.egm.continuation": 1,
        "pylcm.simulation.policy": 1,
        "pylcm.collective.dissolution_flag": 1,
        "pylcm.solver.diagnostics": 1,
    }


def test_private_artifact_key_aliases_are_the_public_singletons() -> None:
    assert private_artifacts.EGM_CONTINUATION is EGM_CONTINUATION
    assert private_artifacts.SIMULATION_POLICY is SIMULATION_POLICY
    assert private_artifacts.DISSOLUTION_FLAG is DISSOLUTION_FLAG
    assert private_artifacts.SOLVER_DIAGNOSTICS is SOLVER_DIAGNOSTICS


def test_grid_search_solution_result_drives_simulation_directly() -> None:
    model, params, initial_conditions = _small_grid_search_inputs()
    solution = model.solve_result(
        params=params,
        log_level="off",
        retention=ResultRetention.VALUES,
    )

    direct = model.simulate(
        params=params,
        initial_conditions=initial_conditions,
        solution=solution,
        log_level="off",
        seed=0,
    )
    legacy = model.simulate(
        params=params,
        initial_conditions=initial_conditions,
        period_to_regime_to_V_arr=cast("PeriodToRegimeToVArr", solution.values),
        log_level="off",
        seed=0,
    )

    assert_frame_equal(direct.to_dataframe(), legacy.to_dataframe())


def test_direct_solution_simulation_processes_params_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, params, initial_conditions = _small_grid_search_inputs()
    solution = model.solve_result(params=params, log_level="off")
    original = model._process_params
    call_count = 0

    def _counted_process_params(raw_params: UserParams):
        nonlocal call_count
        call_count += 1
        return original(raw_params)

    monkeypatch.setattr(model, "_process_params", _counted_process_params)
    model.simulate(
        params=params,
        initial_conditions=initial_conditions,
        solution=solution,
        log_level="off",
        seed=0,
    )

    assert call_count == 1


def test_model_instance_id_survives_pickle_for_result_replay() -> None:
    model, params, initial_conditions = _small_grid_search_inputs()
    solution = model.solve_result(params=params, log_level="off")
    restored = cloudpickle.loads(cloudpickle.dumps(model))

    result = restored.simulate(
        params=params,
        initial_conditions=initial_conditions,
        solution=solution,
        log_level="off",
        seed=0,
    )

    assert result.n_subjects == 1


def test_legacy_model_pickle_backfills_solution_instance_id() -> None:
    model, params, _ = _small_grid_search_inputs()
    del model._solution_model_instance_id

    restored = cloudpickle.loads(cloudpickle.dumps(model))
    solution = restored.solve_result(params=params, log_level="off")

    assert restored._solution_model_instance_id
    assert solution.metadata.model_instance_id == restored._solution_model_instance_id


def test_retained_finite_nnbegm_result_replays_the_same_policy_as_legacy() -> None:
    model = _build("finite")
    solution = model.solve_result(params=_PARAMS, log_level="off")

    direct = model.simulate(
        params=_PARAMS,
        initial_conditions=dict(_INITIAL),
        solution=solution,
        log_level="off",
        seed=42,
    )
    legacy = model.simulate(
        params=_PARAMS,
        initial_conditions=dict(_INITIAL),
        period_to_regime_to_V_arr=cast("PeriodToRegimeToVArr", solution.values),
        policies=cast(
            "PeriodToRegimeToSimulationPolicy",
            solution.replay_artifacts.project(SIMULATION_POLICY),
        ),
        log_level="off",
        seed=42,
    )

    assert_frame_equal(direct.to_dataframe(), legacy.to_dataframe())


def test_retained_adaptive_nnbegm_result_replays_the_same_policy_as_legacy() -> None:
    model = _build("adaptive")
    solution = model.solve_result(params=_PARAMS, log_level="off")

    direct = model.simulate(
        params=_PARAMS,
        initial_conditions=dict(_INITIAL),
        solution=solution,
        log_level="off",
        seed=42,
    )
    legacy = model.simulate(
        params=_PARAMS,
        initial_conditions=dict(_INITIAL),
        period_to_regime_to_V_arr=cast("PeriodToRegimeToVArr", solution.values),
        policies=cast(
            "PeriodToRegimeToSimulationPolicy",
            solution.replay_artifacts.project(SIMULATION_POLICY),
        ),
        log_level="off",
        seed=42,
    )

    assert_frame_equal(direct.to_dataframe(), legacy.to_dataframe())


def test_retained_dissolution_result_replays_the_same_flags_as_legacy() -> None:
    model = _make_dissolution_model()
    solution = model.solve_result(params=_DISSOLUTION_PARAMS, log_level="off")
    initial_conditions = {
        "wage": jnp.asarray([1.0, 2.0, 3.0]),
        "age": jnp.zeros(3),
        "regime_id": jnp.full(3, model.regime_names_to_ids["married"], dtype=jnp.int32),
        "own_stakeholder": jnp.full(
            3, model.stakeholder_names_to_ids["f"], dtype=jnp.int32
        ),
    }

    direct = model.simulate(
        params=_DISSOLUTION_PARAMS,
        initial_conditions=initial_conditions,
        solution=solution,
        log_level="off",
        seed=0,
    )
    legacy = model.simulate(
        params=_DISSOLUTION_PARAMS,
        initial_conditions=initial_conditions,
        period_to_regime_to_V_arr=cast("PeriodToRegimeToVArr", solution.values),
        period_to_regime_to_dissolution_flags=cast(
            "PeriodToRegimeToDissolutionFlags",
            solution.replay_artifacts.project(DISSOLUTION_FLAG),
        ),
        log_level="off",
        seed=0,
    )

    assert_frame_equal(direct.to_dataframe(), legacy.to_dataframe())


@pytest.mark.parametrize(
    "legacy_kwargs",
    [
        {"period_to_regime_to_V_arr": MappingProxyType({})},
        {"policies": MappingProxyType({})},
        {"period_to_regime_to_dissolution_flags": MappingProxyType({})},
    ],
)
def test_solution_result_cannot_be_mixed_with_legacy_solution_inputs(
    legacy_kwargs: dict[str, object],
) -> None:
    model, params, initial_conditions = _small_grid_search_inputs()
    solution = model.solve_result(params=params, log_level="off")

    with pytest.raises(InvalidSimulationInputError, match=r"solution.*legacy"):
        model.simulate(
            params=params,
            initial_conditions=initial_conditions,
            solution=solution,
            log_level="off",
            **legacy_kwargs,  # ty: ignore[invalid-argument-type]
        )


@pytest.mark.parametrize("defect", ["metadata", "coverage"])
def test_solution_result_structure_is_checked_before_simulation(defect: str) -> None:
    model, params, initial_conditions = _small_grid_search_inputs()
    solution = model.solve_result(params=params, log_level="off")
    if defect == "metadata":
        malformed = replace(
            solution,
            metadata=replace(solution.metadata, n_periods=model.n_periods + 1),
        )
    else:
        value_store = solution.values  # noqa: PD011
        first_period = min(value_store)
        malformed = replace(
            solution,
            values={
                period: regime_to_value
                for period, regime_to_value in value_store.items()
                if period != first_period
            },
        )

    with pytest.raises(InvalidSimulationInputError, match=defect):
        model.simulate(
            params=params,
            initial_conditions=initial_conditions,
            solution=malformed,
            log_level="off",
        )


def test_solution_result_from_another_model_instance_is_refused_at_log_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, params, _ = _small_grid_search_inputs()
    target, _, initial_conditions = _small_grid_search_inputs()
    solution = source.solve_result(params=params, log_level="off")

    def _forward_loop_must_not_run(**_kwargs: object) -> None:
        raise AssertionError("forward simulation ran before identity preflight")

    monkeypatch.setattr(model_module, "simulate", _forward_loop_must_not_run)
    with pytest.raises(InvalidSimulationInputError, match="model_instance_id"):
        target.simulate(
            params=params,
            initial_conditions=initial_conditions,
            solution=solution,
            log_level="off",
        )


def test_solution_result_with_changed_canonical_params_is_refused_before_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, params, initial_conditions = _small_grid_search_inputs()
    solution = model.solve_result(params=params, log_level="off")
    changed_params = get_params(n_periods=2, discount_factor=0.9)

    def _forward_loop_must_not_run(**_kwargs: object) -> None:
        raise AssertionError("forward simulation ran before params preflight")

    monkeypatch.setattr(model_module, "simulate", _forward_loop_must_not_run)
    with pytest.raises(InvalidSimulationInputError, match="params_fingerprint"):
        model.simulate(
            params=changed_params,
            initial_conditions=initial_conditions,
            solution=solution,
            log_level="off",
        )


@pytest.mark.parametrize("defect", ["shape", "dtype", "axis_names"])
def test_solution_result_value_schema_is_checked_before_forward(
    *, defect: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    model, params, initial_conditions = _small_grid_search_inputs()
    solution = model.solve_result(params=params, log_level="off")
    coordinate = next(
        coordinate
        for coordinate, schema in solution.metadata.value_schemas.items()
        if schema.axis_names
    )
    period, regime_name = coordinate
    value_store = solution.values  # noqa: PD011
    value = value_store[period][regime_name]

    if defect == "axis_names":
        schema = solution.metadata.value_schemas[coordinate]
        schemas = dict(solution.metadata.value_schemas)
        schemas[coordinate] = replace(
            schema, axis_names=tuple(f"wrong_{name}" for name in schema.axis_names)
        )
        malformed = replace(
            solution,
            metadata=replace(solution.metadata, value_schemas=schemas),
        )
    else:
        replacement = (
            jnp.reshape(value, (*value.shape, 1))
            if defect == "shape"
            else value.astype(
                jnp.float32 if value.dtype != jnp.dtype("float32") else jnp.float64
            )
        )
        values = {
            outer_period: dict(regime_to_value)
            for outer_period, regime_to_value in value_store.items()
        }
        values[period][regime_name] = replacement
        malformed = replace(solution, values=values)

    def _forward_loop_must_not_run(**_kwargs: object) -> None:
        raise AssertionError("forward simulation ran before value-schema preflight")

    monkeypatch.setattr(model_module, "simulate", _forward_loop_must_not_run)
    with pytest.raises(InvalidSimulationInputError, match=defect):
        model.simulate(
            params=params,
            initial_conditions=initial_conditions,
            solution=malformed,
            log_level="off",
        )


def test_solution_result_rejects_an_unexpected_empty_value_period_before_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, params, initial_conditions = _small_grid_search_inputs()
    solution = model.solve_result(params=params, log_level="off")
    values = {
        period: dict(regime_to_value)
        for period, regime_to_value in solution.values.items()  # noqa: PD011
    }
    values[model.n_periods] = {}
    malformed = replace(solution, values=values)

    def _forward_loop_must_not_run(**_kwargs: object) -> None:
        raise AssertionError("forward simulation ran before value-period preflight")

    monkeypatch.setattr(model_module, "simulate", _forward_loop_must_not_run)
    with pytest.raises(InvalidSimulationInputError, match="value period coverage"):
        model.simulate(
            params=params,
            initial_conditions=initial_conditions,
            solution=malformed,
            log_level="off",
        )


@pytest.mark.parametrize(
    "channel",
    [
        "retained_continuations",
        "replay_artifacts",
        "auxiliary_artifacts",
        "diagnostics",
        "omissions",
    ],
)
def test_solution_result_rejects_unexpected_artifact_coordinates_before_forward(
    *, channel: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    model, params, initial_conditions = _small_grid_search_inputs()
    solution = model.solve_result(params=params, log_level="off")
    unexpected_ref = ArtifactRef(
        period=model.n_periods,
        regime=solution.metadata.regime_names[0],
        key=ArtifactKey(type_id="test.unexpected"),
    )
    if channel == "omissions":
        malformed = replace(
            solution,
            omissions=dict(solution.omissions)
            | {unexpected_ref: OmissionReason.NOT_REQUESTED},
        )
    else:
        malformed = _with_artifact(
            solution=solution,
            channel=channel,
            ref=unexpected_ref,
        )

    def _forward_loop_must_not_run(**_kwargs: object) -> None:
        raise AssertionError("forward simulation ran before artifact preflight")

    monkeypatch.setattr(model_module, "simulate", _forward_loop_must_not_run)
    with pytest.raises(InvalidSimulationInputError, match="unexpected"):
        model.simulate(
            params=params,
            initial_conditions=initial_conditions,
            solution=malformed,
            log_level="off",
        )


@pytest.mark.parametrize(
    "channel",
    [
        "retained_continuations",
        "replay_artifacts",
        "auxiliary_artifacts",
        "diagnostics",
    ],
)
def test_solution_result_rejects_present_and_omitted_artifacts_before_forward(
    *, channel: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    model, params, initial_conditions = _small_grid_search_inputs()
    solution = model.solve_result(params=params, log_level="off")
    period, regime_name = next(iter(solution.metadata.value_schemas))
    ref = ArtifactRef(
        period=period,
        regime=regime_name,
        key=ArtifactKey(type_id="test.overlap"),
    )
    with_artifact = _with_artifact(solution=solution, channel=channel, ref=ref)
    malformed = replace(
        with_artifact,
        omissions=dict(with_artifact.omissions) | {ref: OmissionReason.NOT_REQUESTED},
    )

    def _forward_loop_must_not_run(**_kwargs: object) -> None:
        raise AssertionError("forward simulation ran before artifact preflight")

    monkeypatch.setattr(model_module, "simulate", _forward_loop_must_not_run)
    with pytest.raises(InvalidSimulationInputError, match="both present and omitted"):
        model.simulate(
            params=params,
            initial_conditions=initial_conditions,
            solution=malformed,
            log_level="off",
        )


def test_solution_result_rejects_one_ref_in_multiple_artifact_stores_before_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, params, initial_conditions = _small_grid_search_inputs()
    solution = model.solve_result(params=params, log_level="off")
    period, regime_name = next(iter(solution.metadata.value_schemas))
    ref = ArtifactRef(
        period=period,
        regime=regime_name,
        key=ArtifactKey(type_id="test.duplicate"),
    )
    malformed = _with_artifact(
        solution=_with_artifact(
            solution=solution,
            channel="retained_continuations",
            ref=ref,
        ),
        channel="auxiliary_artifacts",
        ref=ref,
    )

    def _forward_loop_must_not_run(**_kwargs: object) -> None:
        raise AssertionError("forward simulation ran before artifact preflight")

    monkeypatch.setattr(model_module, "simulate", _forward_loop_must_not_run)
    with pytest.raises(InvalidSimulationInputError, match="multiple stores"):
        model.simulate(
            params=params,
            initial_conditions=initial_conditions,
            solution=malformed,
            log_level="off",
        )


def test_values_only_finite_nnbegm_result_is_refused_before_forward_simulation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _build("finite")
    solution = model.solve_result(
        params=_PARAMS,
        log_level="off",
        retention=ResultRetention.VALUES,
    )

    def _forward_loop_must_not_run(**_kwargs: object) -> None:
        raise AssertionError("forward simulation ran before replay preflight")

    monkeypatch.setattr(model_module, "simulate", _forward_loop_must_not_run)
    with pytest.raises(
        InvalidSimulationInputError,
        match=r"pylcm\.simulation\.policy.*not_requested",
    ):
        model.simulate(
            params=_PARAMS,
            initial_conditions=dict(_INITIAL),
            solution=solution,
            log_level="off",
        )


@pytest.mark.parametrize(
    "defect",
    [
        "candidate_shape",
        "candidate_dtype",
        "state_names",
        "outer_count",
        "discrete_metadata",
        "keeper_count",
        "all_candidate_shapes",
    ],
)
def test_malformed_finite_nnbegm_payload_is_refused_before_forward(
    *, defect: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _build("finite")
    solution = model.solve_result(params=_PARAMS, log_level="off")
    policy_ref = next(
        ref for ref in solution.replay_artifacts if ref.key == SIMULATION_POLICY
    )
    policy = cast("NNBEGMSimPolicy", solution.replay_artifacts[policy_ref])
    n_candidates = policy.candidate_value.shape[0]
    if defect == "candidate_shape":
        malformed_policy = replace(
            policy,
            candidate_value=jnp.reshape(
                policy.candidate_value, (*policy.candidate_value.shape, 1)
            ),
        )
    elif defect == "candidate_dtype":
        malformed_policy = replace(
            policy,
            candidate_value=policy.candidate_value.astype(
                jnp.float32
                if policy.candidate_value.dtype != jnp.dtype("float32")
                else jnp.float64
            ),
        )
    elif defect == "state_names":
        malformed_policy = replace(
            policy,
            state_names=("not_a_state", *policy.state_names[1:]),
        )
    elif defect == "outer_count":
        malformed_policy = replace(policy, outer_grid_values=jnp.asarray([]))
    elif defect == "discrete_metadata":
        malformed_policy = replace(
            policy,
            candidate_discrete_actions=jnp.zeros((n_candidates, 1), dtype=jnp.int32),
            discrete_action_names=(),
        )
    elif defect == "all_candidate_shapes":
        malformed_policy = replace(
            policy,
            candidate_inner_action=policy.candidate_inner_action[..., :-1],
            candidate_outer_target=policy.candidate_outer_target[..., :-1],
            candidate_value=policy.candidate_value[..., :-1],
        )
    else:
        malformed_policy = replace(policy, n_keeper_candidates=n_candidates + 1)
    entries = dict(solution.replay_artifacts)
    entries[policy_ref] = malformed_policy
    malformed = replace(solution, replay_artifacts=ArtifactStore(entries))

    def _forward_loop_must_not_run(**_kwargs: object) -> None:
        raise AssertionError("forward simulation ran before replay-payload preflight")

    monkeypatch.setattr(model_module, "simulate", _forward_loop_must_not_run)
    with pytest.raises(InvalidSimulationInputError, match="mismatched_payload"):
        model.simulate(
            params=_PARAMS,
            initial_conditions=dict(_INITIAL),
            solution=malformed,
            log_level="off",
        )


@pytest.mark.parametrize("payload", [None, "malformed_egm"])
def test_declared_egm_policy_read_requires_a_valid_egm_payload_before_forward(
    *, payload: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    model, params, initial_conditions = _small_grid_search_inputs()
    solution = model.solve_result(params=params, log_level="off")
    regime_name = "working_life"
    regime = model._regimes[regime_name]
    model._regimes = MappingProxyType(
        dict(model._regimes)
        | {
            regime_name: replace(
                regime,
                simulation=replace(
                    regime.simulation,
                    egm_policy_read=EGMPolicyRead(
                        action_name="consumption",
                        resources_target="consumption",
                        savings_lower_bound=0.0,
                    ),
                ),
            )
        }
    )
    if payload is not None:
        ref = ArtifactRef(period=0, regime=regime_name, key=SIMULATION_POLICY)
        malformed_policy = EGMSimPolicy(
            endog_grid=jnp.ones(3),
            policy=jnp.ones(3),
            value=jnp.ones(2),
            marginal_utility=jnp.ones(3),
        )
        solution = replace(
            solution,
            replay_artifacts=ArtifactStore({ref: malformed_policy}),
        )

    def _forward_loop_must_not_run(**_kwargs: object) -> None:
        raise AssertionError("forward simulation ran before EGM replay preflight")

    monkeypatch.setattr(model_module, "simulate", _forward_loop_must_not_run)
    reason = "mismatched_payload" if payload is not None else "unrecorded"
    with pytest.raises(InvalidSimulationInputError, match=reason):
        model.simulate(
            params=params,
            initial_conditions=initial_conditions,
            solution=solution,
            log_level="off",
        )


def test_adaptive_policy_omission_distinguishes_unpublished_from_dropped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _build("adaptive")
    published_then_dropped = model.solve_result(
        params=_PARAMS,
        log_level="off",
        retention=ResultRetention.VALUES,
    )
    policy_ref = next(
        ref
        for ref, reason in published_then_dropped.omissions.items()
        if ref.key == SIMULATION_POLICY and reason is OmissionReason.NOT_REQUESTED
    )

    original = backward_induction._run_period_kernel

    def _without_policy(**kwargs: object):
        return replace(
            original(**kwargs),  # ty: ignore[invalid-argument-type]
            simulation_policy=None,
        )

    monkeypatch.setattr(backward_induction, "_run_period_kernel", _without_policy)
    never_published = model.solve_result(
        params=_PARAMS,
        log_level="off",
        retention=ResultRetention.VALUES,
    )

    assert never_published.omissions[policy_ref] is OmissionReason.NOT_APPLICABLE


def test_nested_egm_payload_is_validated_recursively_before_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _build("adaptive")
    solution = model.solve_result(params=_PARAMS, log_level="off")
    policy_ref = next(
        ref for ref in solution.replay_artifacts if ref.key == SIMULATION_POLICY
    )
    policy = cast("NestedEGMSimPolicy", solution.replay_artifacts[policy_ref])
    malformed_policy = replace(
        policy,
        keeper=replace(
            policy.keeper,
            value=jnp.reshape(policy.keeper.value, (*policy.keeper.value.shape, 1)),
        ),
    )
    entries = dict(solution.replay_artifacts)
    entries[policy_ref] = malformed_policy
    malformed = replace(solution, replay_artifacts=ArtifactStore(entries))

    def _forward_loop_must_not_run(**_kwargs: object) -> None:
        raise AssertionError("forward simulation ran before nested replay preflight")

    monkeypatch.setattr(model_module, "simulate", _forward_loop_must_not_run)
    with pytest.raises(InvalidSimulationInputError, match="mismatched_payload"):
        model.simulate(
            params=_PARAMS,
            initial_conditions=dict(_INITIAL),
            solution=malformed,
            log_level="off",
        )


def test_values_only_dissolution_result_is_refused_before_forward_simulation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _make_dissolution_model()
    solution = model.solve_result(
        params=_DISSOLUTION_PARAMS,
        log_level="off",
        retention=ResultRetention.VALUES,
    )

    def _forward_loop_must_not_run(**_kwargs: object) -> None:
        raise AssertionError("forward simulation ran before dissolution preflight")

    monkeypatch.setattr(model_module, "simulate", _forward_loop_must_not_run)
    with pytest.raises(
        InvalidSimulationInputError,
        match=r"pylcm\.collective\.dissolution_flag.*not_requested",
    ):
        model.simulate(
            params=_DISSOLUTION_PARAMS,
            initial_conditions={},
            solution=solution,
            log_level="off",
        )


@pytest.mark.parametrize("defect", ["dtype", "shape"])
def test_malformed_dissolution_flag_is_refused_before_forward(
    *, defect: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _make_dissolution_model()
    solution = model.solve_result(params=_DISSOLUTION_PARAMS, log_level="off")
    flag_ref = next(
        ref for ref in solution.replay_artifacts if ref.key == DISSOLUTION_FLAG
    )
    flag = solution.replay_artifacts[flag_ref]
    malformed_flag = (
        jnp.asarray(flag, dtype=jnp.float32)
        if defect == "dtype"
        else jnp.reshape(jnp.asarray(flag), (*jnp.asarray(flag).shape, 1))
    )
    entries = dict(solution.replay_artifacts)
    entries[flag_ref] = malformed_flag
    malformed = replace(solution, replay_artifacts=ArtifactStore(entries))

    def _forward_loop_must_not_run(**_kwargs: object) -> None:
        raise AssertionError("forward simulation ran before dissolution preflight")

    monkeypatch.setattr(model_module, "simulate", _forward_loop_must_not_run)
    with pytest.raises(InvalidSimulationInputError, match="mismatched_payload"):
        model.simulate(
            params=_DISSOLUTION_PARAMS,
            initial_conditions={},
            solution=malformed,
            log_level="off",
        )


def _small_grid_search_inputs() -> tuple[Model, UserParams, UserInitialConditions]:
    model = get_model(
        n_periods=2,
        wealth_grid=LinSpacedGrid(start=1, stop=3, n_points=3),
        consumption_grid=LinSpacedGrid(start=1, stop=3, n_points=3),
    )
    params = get_params(n_periods=2)
    initial_conditions = {
        "wealth": jnp.asarray([2.0]),
        "age": jnp.asarray([18.0]),
        "regime_id": jnp.asarray([RegimeId.working_life], dtype=jnp.int32),
    }
    return model, params, initial_conditions


def _with_artifact(
    *, solution: SolutionResult, channel: str, ref: ArtifactRef
) -> SolutionResult:
    """Return a result with one test artifact added to the named store."""
    store = cast("ArtifactStore", getattr(solution, channel))
    replacement = ArtifactStore(dict(store) | {ref: object()})
    return replace(
        solution,
        **{channel: replacement},
    )
