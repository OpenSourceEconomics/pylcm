"""Public, labelled solve results and their artifact-retention contract."""

import ast
import hashlib
import inspect
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType, ModuleType
from typing import NoReturn, NotRequired, TypedDict, Unpack, cast

import cloudpickle
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pandas.testing import assert_frame_equal

import lcm.model as model_module
import lcm.solver_api as solver_api_module
from _lcm.egm.nested_published_policy import NestedEGMSimPolicy
from _lcm.egm.published_policy import EGMSimPolicy, NNBEGMSimPolicy
from _lcm.regime_building import processing as regime_processing
from _lcm.simulation import simulate as simulation_module
from _lcm.solution import artifacts as private_artifacts
from _lcm.solution import backward_induction
from _lcm.solution.contract import GENERATED_REPLAY_AUTHORITY
from _lcm.solution.solver_diagnostics import SolverDiagnostics
from _lcm.typing import (
    ArtifactPayload,
    FlatParams,
)
from lcm import (
    AgeGrid,
    CollectiveUtility,
    DiscreteGrid,
    ExecutionConfig,
    LinSpacedGrid,
    Model,
    Regime,
    categorical,
)
from lcm.exceptions import (
    ExecutionPlanningError,
    IncompatibleSolutionError,
    InvalidSimulationInputError,
    SolutionIntegrityError,
)
from lcm.persistence import load_solution
from lcm.solver_api import (
    DISSOLUTION_FLAG,
    EGM_CONTINUATION,
    SIMULATION_POLICY,
    SOLVER_DIAGNOSTICS,
    ArtifactKey,
    ArtifactRef,
    ArtifactStore,
    AxisDescriptor,
    AxisRole,
    LoadState,
    OmissionReason,
    PersistencePolicy,
    ResultRetention,
    SolutionMetadata,
    SolutionResult,
    ValueArraySchema,
    ValueStore,
)
from lcm.solvers import MSSEnvelope
from lcm.typing import FloatND, IntND, ScalarInt, UserInitialConditions, UserParams
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
from tests.test_models.deterministic import base as deterministic_base
from tests.test_models.deterministic.dcegm_variants import (
    get_full_model,
    get_full_params,
    get_graph_only_retirement_params,
)
from tests.test_models.deterministic.regression import (
    RegimeId,
    get_model,
    get_params,
)


class _ArtifactKeyKwargs(TypedDict):
    type_id: str
    schema_version: int


class _KernelKwargs(TypedDict):
    regime: backward_induction.Regime
    regime_name: backward_induction.RegimeName
    period: int
    compiled_cores: MappingProxyType[str, backward_induction.PlannedCore]
    capture_target: backward_induction.PeriodCaptureTarget | None
    state_action_space: backward_induction.StateActionSpace
    flat_params: FlatParams
    ages: backward_induction.TimeAxis
    next_regime_to_V_arr: MappingProxyType[backward_induction.RegimeName, FloatND]
    next_regime_to_continuation: MappingProxyType[
        backward_induction.RegimeName, backward_induction.ContinuationPayload
    ]
    logger: backward_induction.logging.Logger
    next_edge_to_V_arr: MappingProxyType[backward_induction._EdgeKey, FloatND]
    period_solution: Mapping[backward_induction.RegimeName, FloatND]
    retain_replay: bool
    selected_artifact_keys: frozenset[ArtifactKey]
    period_capture: NotRequired[backward_induction.CaptureContext | None]
    captured_admission: NotRequired[Mapping[str, Mapping[str, int | None]]]


class _SimulationKwargs(TypedDict):
    flat_params: FlatParams
    initial_conditions: simulation_module.InitialConditions
    regimes: MappingProxyType[simulation_module.RegimeName, simulation_module.Regime]
    regime_names_to_ids: simulation_module.RegimeNamesToIds
    logger: simulation_module.logging.Logger
    period_to_regime_to_V_arr: MappingProxyType[
        int, MappingProxyType[simulation_module.RegimeName, FloatND]
    ]
    ages: simulation_module.TimeAxis
    simulation_output_dtypes: Mapping[str, simulation_module.pd.CategoricalDtype]
    period_to_regime_to_sim_policy: NotRequired[
        simulation_module.PeriodToRegimeToSimulationPolicy | None
    ]
    period_to_regime_to_replay_reader: NotRequired[
        simulation_module._PeriodToRegimeToReplayReader
    ]
    seed: NotRequired[int | None]
    taste_shock_seed: NotRequired[int | None]
    subject_batch_size: NotRequired[int]
    original_n_subjects: NotRequired[int | None]
    period_to_regime_to_dissolution_flags: NotRequired[
        MappingProxyType[
            int,
            MappingProxyType[simulation_module.RegimeName, simulation_module.BoolND],
        ]
    ]
    device_ids: NotRequired[tuple[int, ...]]
    retained_footprint: NotRequired[simulation_module.DeviceBufferFootprint | None]
    prepared_chunks: NotRequired[simulation_module.PreparedSimulationChunks | None]
    process_grid_resolver: NotRequired[simulation_module.ProcessGridResolver | None]
    call_id: NotRequired[simulation_module.CallId | None]
    component_values: NotRequired[simulation_module.ComponentValueSource | None]


class _RaisingLazyValueEntry(solver_api_module._LazyEntry):
    """Raise one chosen decoder exception when the value is materialized."""

    def __init__(self, error: Exception) -> None:
        self._error = error

    @property
    def load_state(self) -> LoadState:
        """Report that the adversarial entry has not materialized."""
        return LoadState.UNLOADED

    def materialize(
        self,
        *,
        template: ArtifactPayload | None = None,  # noqa: ARG002
    ) -> ArtifactPayload:
        """Raise the configured decoder exception."""
        raise self._error


type PythonSource = str


def _imported_module_names(source: PythonSource) -> set[str]:
    """Return every module name `source` imports when it runs.

    Imports under `if TYPE_CHECKING:` exist only for type checkers and are skipped.
    """
    tree = ast.parse(source)
    static_only = {
        id(inner)
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == "TYPE_CHECKING"
        for statement in node.body
        for inner in ast.walk(statement)
    }
    runtime = [node for node in ast.walk(tree) if id(node) not in static_only]
    return {
        alias.name
        for node in runtime
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {node.module or "" for node in runtime if isinstance(node, ast.ImportFrom)}


def _solver_api_modules() -> list[ModuleType]:
    """Return the public facade and every module it re-exports from."""
    facade = inspect.getmodule(solver_api_module.SolutionResult)
    assert facade is not None
    modules = {facade}
    for name in dir(facade):
        defining = inspect.getmodule(getattr(facade, name))
        if defining is not None and defining.__name__.startswith("lcm._solver_api."):
            modules.add(defining)
    return sorted(modules, key=lambda module: module.__name__)


@pytest.mark.parametrize(
    "module", _solver_api_modules(), ids=lambda module: module.__name__
)
def test_solver_api_has_no_private_lcm_imports(*, module: ModuleType) -> None:
    """An installed solver can import the result spine without importing `_lcm`."""
    imported = _imported_module_names(inspect.getsource(module))

    assert not any(name == "_lcm" or name.startswith("_lcm.") for name in imported)


def test_imported_module_names_skips_only_type_checking_imports() -> None:
    """Runtime imports count, including ones beside a `TYPE_CHECKING` block."""
    source = (
        "import json\n"
        "from _lcm.grids import Grid\n"
        "if TYPE_CHECKING:\n"
        "    from _lcm.solution.artifacts import OwnedSolutionView\n"
        "else:\n"
        "    from _lcm.typing import PytreeValue\n"
    )

    assert _imported_module_names(source) == {"json", "_lcm.grids", "_lcm.typing"}


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
    *, kwargs: _ArtifactKeyKwargs, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        ArtifactKey(**kwargs)


def test_artifact_store_is_immutable_and_projects_one_artifact_type() -> None:
    policy = ArtifactKey(type_id="example.policy")
    diagnostic = ArtifactKey(type_id="example.diagnostic")
    refs = {
        ArtifactRef(period=1, regime="alive", key=policy): "p1",
        ArtifactRef(period=0, regime="alive", key=policy): "p0",
        ArtifactRef(period=0, regime="alive", key=diagnostic): "d0",
    }
    store = ArtifactStore(cast("Mapping[ArtifactRef, ArtifactPayload]", refs))

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
    assert isinstance(result.values, ValueStore)
    assert isinstance(result.values[0], Mapping)
    assert isinstance(result.omissions, MappingProxyType)
    with pytest.raises(TypeError):
        result.values[0]["alive"] = jnp.asarray([9.0])  # ty: ignore[invalid-assignment]


def test_store_lookups_reject_nonexact_coordinates_before_hashing() -> None:
    """Reject equality-compatible lookup coordinates before mapping access."""

    class _ArmedHashString(str):  # noqa: SLOT000
        armed = False

        def __hash__(self) -> int:
            if self.armed:
                raise RuntimeError("hostile lookup hash escaped")
            return super().__hash__()

    values = ValueStore({(0, "alive"): jnp.asarray([1.0], dtype=jnp.float32)})
    result = SolutionResult(
        values=values,
        metadata=SolutionMetadata(
            retention=ResultRetention.VALUES,
            n_periods=1,
            regime_names=("alive",),
            solver_types={"alive": "example.Grid"},
            model_instance_id="model-1",
            params_fingerprint="0" * 64,
            value_schemas={
                (0, "alive"): ValueArraySchema(
                    shape=(1,), dtype="float32", axis_names=("wealth",)
                )
            },
        ),
    )
    hostile_regime = _ArmedHashString("alive")
    hostile_regime.armed = True

    assert True not in values
    assert hostile_regime not in values[0]
    with pytest.raises(TypeError, match="exact ints"):
        result.value(period=True, regime="alive")
    with pytest.raises(TypeError, match="exact strs"):
        result.value(period=0, regime=hostile_regime)
    with pytest.raises(TypeError, match="exact strs"):
        values.load_state(period=0, regime=hostile_regime)

    key = ArtifactKey(type_id="example.policy")
    ref = ArtifactRef(period=0, regime="alive", key=key)
    artifacts = ArtifactStore({ref: cast("ArtifactPayload", object())})
    hostile_ref = replace(ref)
    object.__setattr__(hostile_ref, "regime", hostile_regime)

    assert hostile_ref not in artifacts
    with pytest.raises(TypeError, match="exact strs"):
        artifacts.load_state(hostile_ref)


def test_solve_records_instance_params_and_value_array_schemas() -> None:
    model, params, _ = _small_grid_search_inputs()

    result = model.solve(params=params, log_level="off")

    assert result.metadata.model_instance_id
    assert len(result.metadata.params_fingerprint) == 64
    value_store = result.values
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


def test_model_solve_omits_policy_without_replay_route() -> None:
    model = _two_period_bequest_model()
    params = get_graph_only_retirement_params(n_periods=2, discount_factor=0.98)

    result = model.solve(params=params, log_level="off")

    assert result.metadata.retention is ResultRetention.VALUES_AND_REPLAY
    assert not result.replay_artifacts.project(SIMULATION_POLICY)
    policy_ref = ArtifactRef(
        period=0,
        regime="retirement",
        key=SIMULATION_POLICY,
    )
    continuation_ref = ArtifactRef(
        period=0,
        regime="retirement",
        key=EGM_CONTINUATION,
    )
    assert result.omissions[policy_ref] is OmissionReason.NOT_APPLICABLE
    assert result.omissions[continuation_ref] is OmissionReason.NOT_REQUESTED
    assert result.metadata.solver_api_version == solver_api_module.SOLVER_API_VERSION
    assert not result.diagnostics


def test_model_rejects_a_present_inapplicable_artifact() -> None:
    """A false present payload cannot replace model authority's omission."""
    model = _two_period_bequest_model()
    params = get_graph_only_retirement_params(n_periods=2, discount_factor=0.98)
    solution = model.solve(params=params, log_level="off")
    policy_ref = ArtifactRef(
        period=0,
        regime="retirement",
        key=SIMULATION_POLICY,
    )
    malformed = replace(
        solution,
        replay_artifacts=ArtifactStore({policy_ref: cast("ArtifactPayload", object())}),
        omissions={
            ref: reason
            for ref, reason in solution.omissions.items()
            if ref != policy_ref
        },
    )

    with pytest.raises(
        InvalidSimulationInputError,
        match=r"present artifacts.*not applicable",
    ):
        model.simulate(
            params=params,
            initial_conditions={},
            solution=malformed,
            log_level="off",
        )


def test_model_rejects_a_present_artifact_not_selected_by_retention() -> None:
    """The metadata retention label constrains every present artifact channel."""
    model = _make_dissolution_model()
    solution = model.solve(params=_DISSOLUTION_PARAMS, log_level="off")
    malformed = replace(
        solution,
        metadata=replace(solution.metadata, retention=ResultRetention.VALUES),
    )

    with pytest.raises(
        InvalidSimulationInputError,
        match=r"present artifacts.*not selected by retention 'values'",
    ):
        model.simulate(
            params=_DISSOLUTION_PARAMS,
            initial_conditions={},
            solution=malformed,
            log_level="off",
        )


def test_solve_does_not_host_copy_policy_without_replay_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_device_put = backward_induction.jax.device_put

    def _reject_policy_host_copy(
        value: object,  # noqa: PAN001 - JAX device_put accepts arbitrary registered PyTrees.
        *args: object,  # noqa: PAN001 - JAX device_put accepts arbitrary placement prefix trees.
        **kwargs: object,  # noqa: PAN001 - JAX device_put accepts arbitrary option prefix trees.
    ) -> object:  # noqa: PAN001 - JAX device_put preserves arbitrary registered PyTrees.
        if isinstance(value, EGMSimPolicy):
            raise TypeError("policy without a replay route was copied to host")
        return original_device_put(value, *args, **kwargs)

    monkeypatch.setattr(backward_induction.jax, "device_put", _reject_policy_host_copy)
    model = _two_period_bequest_model()
    params = get_graph_only_retirement_params(n_periods=2, discount_factor=0.98)

    result = model.solve(params=params, log_level="off")

    assert not result.replay_artifacts.project(SIMULATION_POLICY)


def test_values_only_result_drops_replay_with_an_explicit_reason() -> None:
    model = _two_period_bequest_model()
    params = get_graph_only_retirement_params(n_periods=2, discount_factor=0.98)

    result = model.solve(
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
    assert result.omissions[policy_ref] is OmissionReason.NOT_APPLICABLE


def test_all_persistable_retains_model_verifiable_egm_continuation() -> None:
    model = _two_period_bequest_model()
    params = get_graph_only_retirement_params(n_periods=2, discount_factor=0.98)

    result = model.solve(
        params=params,
        log_level="off",
        retention=ResultRetention.ALL_PERSISTABLE_ARTIFACTS,
    )

    continuation_ref = ArtifactRef(
        period=0,
        regime="retirement",
        key=EGM_CONTINUATION,
    )
    assert continuation_ref in result.retained_continuations
    assert continuation_ref not in result.omissions
    assert (
        result.metadata.artifact_descriptors[continuation_ref].persistence
        is PersistencePolicy.MODEL_VERIFIABLE
    )


def test_solve_retains_kernel_diagnostics_only_when_log_level_enables_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = backward_induction._run_period_kernel

    def _with_diagnostics(
        **kwargs: Unpack[_KernelKwargs],
    ) -> backward_induction.KernelOutput:
        output = original(**kwargs)
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
        return replace(
            output,
            auxiliary={**output.auxiliary, SOLVER_DIAGNOSTICS: diagnostics},
        )

    monkeypatch.setattr(backward_induction, "_run_period_kernel", _with_diagnostics)
    model = _two_period_bequest_model()
    params = get_graph_only_retirement_params(n_periods=2, discount_factor=0.98)

    enabled = model.solve(params=params, log_level="warning")
    disabled = model.solve(params=params, log_level="off")

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


def test_grid_search_result_replay_matches_automatic_solve() -> None:
    model, params, initial_conditions = _small_grid_search_inputs()
    solution = model.solve(
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
    automatic = model.simulate(
        params=params,
        initial_conditions=initial_conditions,
        log_level="off",
        seed=0,
    )

    assert_frame_equal(direct.to_dataframe(), automatic.to_dataframe())


def test_direct_solution_simulation_processes_params_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, params, initial_conditions = _small_grid_search_inputs()
    solution = model.solve(params=params, log_level="off")
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
    solution = model.solve(params=params, log_level="off")
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
    solution = restored.solve(params=params, log_level="off")

    assert restored._solution_model_instance_id
    assert solution.metadata.model_instance_id == restored._solution_model_instance_id


def test_model_archive_with_dict_fixed_component_splits_restores_them_read_only() -> (
    None
):
    """An archive holding the fixed-component splits as a dict restores them frozen."""
    model, _, _ = _small_grid_search_inputs()
    model._fixed_component_splits = dict(model._fixed_component_splits)  # ty: ignore[invalid-assignment]

    restored = cloudpickle.loads(cloudpickle.dumps(model))

    assert isinstance(restored._fixed_component_splits, MappingProxyType)


def test_retained_finite_nnbegm_result_replay_matches_automatic_solve() -> None:
    model = _build("finite")
    solution = model.solve(params=_PARAMS, log_level="off")

    direct = model.simulate(
        params=_PARAMS,
        initial_conditions=dict(_INITIAL),
        solution=solution,
        log_level="off",
        seed=42,
    )
    automatic = model.simulate(
        params=_PARAMS,
        initial_conditions=dict(_INITIAL),
        log_level="off",
        seed=42,
    )

    assert_frame_equal(direct.to_dataframe(), automatic.to_dataframe())


def test_retained_adaptive_nnbegm_result_replay_matches_automatic_solve() -> None:
    model = _build("adaptive")
    solution = model.solve(params=_PARAMS, log_level="off")

    direct = model.simulate(
        params=_PARAMS,
        initial_conditions=dict(_INITIAL),
        solution=solution,
        log_level="off",
        seed=42,
    )
    automatic = model.simulate(
        params=_PARAMS,
        initial_conditions=dict(_INITIAL),
        log_level="off",
        seed=42,
    )

    assert_frame_equal(direct.to_dataframe(), automatic.to_dataframe())


def _adaptive_outer_nodes_by_ref(
    solution: SolutionResult,
) -> dict[ArtifactRef, tuple[float, ...]]:
    """The outer nodes each nested policy descriptor declares for itself."""
    return {
        ref: cast("tuple[float, ...]", axis.coordinates)
        for ref, descriptor in solution.metadata.artifact_descriptors.items()
        if ref.key == SIMULATION_POLICY
        for axis in descriptor.named_axes
        if axis.role is AxisRole.CANDIDATE
    }


def test_adaptive_authority_and_result_survive_pickle_for_valid_replay(
    tmp_path: Path,
) -> None:
    """The outer nodes an adaptive solve settled on travel inside the result."""
    model = _build("adaptive")
    solution = model.solve(params=_PARAMS, log_level="off")
    before = _adaptive_outer_nodes_by_ref(solution)
    assert before

    restored_model, restored_solution = cloudpickle.loads(
        cloudpickle.dumps((model, solution))
    )
    restored_solution.save(path=tmp_path / "restored-solution")
    after = _adaptive_outer_nodes_by_ref(restored_solution)

    assert after == before
    for ref, expected_nodes in after.items():
        policy = cast("NestedEGMSimPolicy", restored_solution.replay_artifacts[ref])
        assert expected_nodes == tuple(
            float(node) for node in np.asarray(policy.adjuster.outer_nodes)
        )

    replay = restored_model.simulate(
        params=_PARAMS,
        initial_conditions=dict(_INITIAL),
        solution=restored_solution,
        log_level="off",
        seed=42,
    )

    assert replay.n_subjects == len(_INITIAL["wealth"])


def test_retained_dissolution_result_replay_matches_automatic_solve() -> None:
    model = _make_dissolution_model()
    solution = model.solve(params=_DISSOLUTION_PARAMS, log_level="off")
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
    automatic = model.simulate(
        params=_DISSOLUTION_PARAMS,
        initial_conditions=initial_conditions,
        log_level="off",
        seed=0,
    )

    assert_frame_equal(direct.to_dataframe(), automatic.to_dataframe())


def test_obsolete_solve_and_simulate_interfaces_are_absent() -> None:
    solve_parameters = inspect.signature(Model.solve).parameters
    simulate_parameters = inspect.signature(Model.simulate).parameters

    assert not hasattr(Model, "solve_result")
    assert "return_simulation_policy" not in solve_parameters
    assert "return_dissolution_flags" not in solve_parameters
    assert "period_to_regime_to_V_arr" not in simulate_parameters
    assert "policies" not in simulate_parameters
    assert "period_to_regime_to_dissolution_flags" not in simulate_parameters


def _refuse_unadmitted_compiled_dispatch(
    *_args: object,  # noqa: PAN001 - Rejects arbitrary compiled-core argument trees.
    **_kwargs: object,  # noqa: PAN001 - Rejects arbitrary compiled-core argument trees.
) -> NoReturn:
    raise AssertionError("A one-byte budget reached compiled core dispatch")


def test_unmeetable_execution_budget_fails_closed_before_solving(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Known resident inputs already exceed one byte, before any core executes."""
    model, params, _initial_conditions = _small_grid_search_inputs(
        execution_config=ExecutionConfig(device_memory_bytes=1)
    )

    monkeypatch.setattr(
        jax.stages.Compiled, "__call__", _refuse_unadmitted_compiled_dispatch
    )
    with pytest.raises(
        ExecutionPlanningError, match=r"keeps \d+ bytes resident.*1-byte budget"
    ):
        model.solve(params=params, log_level="off")


def test_a_budgeted_model_replays_a_supplied_solution() -> None:
    """A workspace budget is the model's, so a supplied solution still simulates."""
    model, params, initial_conditions = _small_grid_search_inputs(
        execution_config=ExecutionConfig(device_memory_bytes=2**32)
    )
    solution = model.solve(params=params, log_level="off")

    replayed = model.simulate(
        params=params,
        initial_conditions=initial_conditions,
        solution=solution,
        log_level="off",
    )
    automatic = model.simulate(
        params=params,
        initial_conditions=initial_conditions,
        log_level="off",
    )

    assert_frame_equal(replayed.to_dataframe(), automatic.to_dataframe())


def test_solution_result_has_no_mapping_compatibility_bridge() -> None:
    model, params, initial_conditions = _small_grid_search_inputs()
    solution = model.solve(params=params, log_level="off")

    assert not isinstance(solution, Mapping)
    with pytest.raises(InvalidSimulationInputError):
        model.simulate(
            params=params,
            initial_conditions=initial_conditions,
            solution=solution.values,  # ty: ignore[invalid-argument-type]
            log_level="off",
        )


@pytest.mark.parametrize("defect", ["metadata", "coverage"])
def test_solution_result_structure_is_checked_before_simulation(defect: str) -> None:
    model, params, initial_conditions = _small_grid_search_inputs()
    solution = model.solve(params=params, log_level="off")
    if defect == "metadata":
        malformed = replace(
            solution,
            metadata=replace(solution.metadata, n_periods=model.n_periods + 1),
        )
    else:
        value_store = solution.values
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


def test_solution_result_rejects_constructor_bypassed_nested_value_mapping() -> None:
    """Preflight requires the value store whose materialization API it invokes."""
    model, params, initial_conditions = _small_grid_search_inputs()
    solution = model.solve(params=params, log_level="off")
    malformed = replace(solution)
    bypassed_values = MappingProxyType(
        {
            period: MappingProxyType(dict(regime_to_value))
            for period, regime_to_value in solution.values.items()
        }
    )
    object.__setattr__(malformed, "values", bypassed_values)

    with pytest.raises(InvalidSimulationInputError, match="exact ValueStore"):
        model.simulate(
            params=params,
            initial_conditions=initial_conditions,
            solution=malformed,
            log_level="off",
        )


@pytest.mark.parametrize(
    ("decoder_error", "expected_error"),
    [
        (TypeError("hostile value decoder"), InvalidSimulationInputError),
        (ValueError("hostile value decoder"), InvalidSimulationInputError),
        (SolutionIntegrityError("hostile value decoder"), SolutionIntegrityError),
        (IncompatibleSolutionError("hostile value decoder"), IncompatibleSolutionError),
    ],
    ids=(
        "type-error-is-normalized",
        "value-error-is-normalized",
        "integrity-error-is-preserved",
        "incompatibility-error-is-preserved",
    ),
)
def test_lazy_value_decoder_errors_cross_the_public_boundary(
    *, decoder_error: Exception, expected_error: type[Exception]
) -> None:
    """Normalize decoder mechanics without hiding archive-domain exceptions."""
    model, params, initial_conditions = _small_grid_search_inputs()
    solution = model.solve(params=params, log_level="off")
    entries: dict[tuple[int, str], FloatND | solver_api_module._LazyEntry] = {
        (period, regime_name): value
        for period, regime_to_value in solution.values.items()
        for regime_name, value in regime_to_value.items()
    }
    coordinate = next(iter(entries))
    entries[coordinate] = _RaisingLazyValueEntry(decoder_error)
    malformed = replace(solution, values=ValueStore(entries))

    with pytest.raises(expected_error, match="hostile value decoder"):
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
    solution = source.solve(params=params, log_level="off")

    def _forward_loop_must_not_run(**_kwargs: Unpack[_SimulationKwargs]) -> NoReturn:
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
    solution = model.solve(params=params, log_level="off")
    changed_params = get_params(n_periods=2, discount_factor=0.9)

    def _forward_loop_must_not_run(**_kwargs: Unpack[_SimulationKwargs]) -> NoReturn:
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
    solution = model.solve(params=params, log_level="off")
    coordinate = next(
        coordinate
        for coordinate, schema in solution.metadata.value_schemas.items()
        if schema.axis_names
    )
    period, regime_name = coordinate
    value_store = solution.values
    value = value_store[period][regime_name]

    if defect == "axis_names":
        schema = solution.metadata.value_schemas[coordinate]
        schemas = dict(solution.metadata.value_schemas)
        schemas[coordinate] = replace(
            schema,
            axis_names=tuple(f"wrong_{name}" for name in schema.axis_names),
            named_axes=tuple(
                replace(axis, name=f"wrong_{axis.name}") for axis in schema.named_axes
            ),
            categorical_domains={
                f"wrong_{name}": domain
                for name, domain in schema.categorical_domains.items()
            },
        )
        malformed = replace(
            solution,
            metadata=replace(solution.metadata, value_schemas=schemas),
        )
    else:
        replacement = (
            jnp.reshape(value, (*value.shape, 1))
            if defect == "shape"
            else value.astype(jnp.int32)
        )
        values = {
            outer_period: dict(regime_to_value)
            for outer_period, regime_to_value in value_store.items()
        }
        values[period][regime_name] = replacement
        malformed = replace(solution, values=values)

    def _forward_loop_must_not_run(**_kwargs: Unpack[_SimulationKwargs]) -> NoReturn:
        raise AssertionError("forward simulation ran before value-schema preflight")

    monkeypatch.setattr(model_module, "simulate", _forward_loop_must_not_run)
    with pytest.raises(InvalidSimulationInputError, match=defect):
        model.simulate(
            params=params,
            initial_conditions=initial_conditions,
            solution=malformed,
            log_level="off",
        )


def test_solution_result_normalizes_an_unexpected_empty_value_period() -> None:
    model, params, initial_conditions = _small_grid_search_inputs()
    solution = model.solve(params=params, log_level="off")
    values = {
        period: dict(regime_to_value)
        for period, regime_to_value in solution.values.items()
    }
    values[model.n_periods] = {}
    malformed = replace(solution, values=values)

    assert model.n_periods not in malformed.values
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
    solution = model.solve(params=params, log_level="off")
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

    def _forward_loop_must_not_run(**_kwargs: Unpack[_SimulationKwargs]) -> NoReturn:
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
    solution = model.solve(params=params, log_level="off")
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

    def _forward_loop_must_not_run(**_kwargs: Unpack[_SimulationKwargs]) -> NoReturn:
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
    solution = model.solve(params=params, log_level="off")
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

    def _forward_loop_must_not_run(**_kwargs: Unpack[_SimulationKwargs]) -> NoReturn:
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
    solution = model.solve(
        params=_PARAMS,
        log_level="off",
        retention=ResultRetention.VALUES,
    )

    def _forward_loop_must_not_run(**_kwargs: Unpack[_SimulationKwargs]) -> NoReturn:
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
    solution = model.solve(params=_PARAMS, log_level="off")
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
        malformed_policy = replace(policy)
        object.__setattr__(
            malformed_policy,
            "candidate_value",
            policy.candidate_value.astype(jnp.int32),
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

    def _forward_loop_must_not_run(**_kwargs: Unpack[_SimulationKwargs]) -> NoReturn:
        raise AssertionError("forward simulation ran before replay-payload preflight")

    monkeypatch.setattr(model_module, "simulate", _forward_loop_must_not_run)
    with pytest.raises(
        InvalidSimulationInputError,
        match=r"mismatched_payload|artifact payloads cannot be detached",
    ):
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
    get_full_model.cache_clear()
    monkeypatch.setattr(
        regime_processing,
        "_CROSSING_COMPLETE_ENVELOPES",
        (MSSEnvelope,),
    )
    model = get_full_model(solver="dcegm", n_periods=2, envelope="mss")
    get_full_model.cache_clear()
    params = get_full_params(n_periods=2)
    initial_conditions = {
        "wealth": jnp.asarray([2.0]),
        "age": jnp.asarray([40.0]),
        "regime_id": jnp.asarray(
            [deterministic_base.RegimeId.working_life], dtype=jnp.int32
        ),
    }
    solution = model.solve(params=params, log_level="off")
    regime_name = "working_life"
    ref = next(
        ref
        for ref in solution.replay_artifacts
        if ref.key == SIMULATION_POLICY and ref.regime == regime_name
    )
    entries = dict(solution.replay_artifacts)
    if payload is None:
        entries.pop(ref)
    else:
        policy = cast("EGMSimPolicy", entries[ref])
        entries[ref] = replace(
            policy,
            value=policy.value[..., :-1],
        )
    solution = replace(solution, replay_artifacts=ArtifactStore(entries))

    def _forward_loop_must_not_run(**_kwargs: Unpack[_SimulationKwargs]) -> NoReturn:
        raise AssertionError("forward simulation ran before EGM replay preflight")

    monkeypatch.setattr(model_module, "simulate", _forward_loop_must_not_run)
    reason = (
        r"mismatched_payload|artifact payloads cannot be detached"
        if payload is not None
        else r"unrecorded|missing accounting"
    )
    with pytest.raises(InvalidSimulationInputError, match=reason):
        model.simulate(
            params=params,
            initial_conditions=initial_conditions,
            solution=solution,
            log_level="off",
        )


def test_adaptive_policy_omission_is_derived_from_model_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _build("adaptive")
    published_then_dropped = model.solve(
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

    def _without_policy(
        **kwargs: Unpack[_KernelKwargs],
    ) -> backward_induction.KernelOutput:
        output = original(**kwargs)
        return replace(
            output,
            replay={
                key: payload
                for key, payload in output.replay.items()
                if key != SIMULATION_POLICY
            },
            auxiliary={
                key: payload
                for key, payload in output.auxiliary.items()
                if key != GENERATED_REPLAY_AUTHORITY
            },
        )

    monkeypatch.setattr(backward_induction, "_run_period_kernel", _without_policy)
    never_published = model.solve(
        params=_PARAMS,
        log_level="off",
        retention=ResultRetention.VALUES,
    )

    assert never_published.omissions[policy_ref] is OmissionReason.NOT_REQUESTED


def test_nested_egm_payload_is_validated_recursively_before_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _build("adaptive")
    solution = model.solve(params=_PARAMS, log_level="off")
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

    def _forward_loop_must_not_run(**_kwargs: Unpack[_SimulationKwargs]) -> NoReturn:
        raise AssertionError("forward simulation ran before nested replay preflight")

    monkeypatch.setattr(model_module, "simulate", _forward_loop_must_not_run)
    with pytest.raises(
        InvalidSimulationInputError,
        match=r"mismatched_payload|artifact payloads cannot be detached",
    ):
        model.simulate(
            params=_PARAMS,
            initial_conditions=dict(_INITIAL),
            solution=malformed,
            log_level="off",
        )


def test_policy_at_an_undeclared_coordinate_is_refused_before_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_model = _build("adaptive")
    source_solution = source_model.solve(params=_PARAMS, log_level="off")
    nested_policy = next(
        source_solution.replay_artifacts[ref]
        for ref in source_solution.replay_artifacts
        if ref.key == SIMULATION_POLICY
    )
    assert isinstance(nested_policy, NestedEGMSimPolicy)

    model, params, initial_conditions = _small_grid_search_inputs()
    solution = model.solve(params=params, log_level="off")
    policy_ref = ArtifactRef(
        period=0,
        regime="working_life",
        key=SIMULATION_POLICY,
    )
    malformed = replace(
        solution,
        replay_artifacts=ArtifactStore(
            dict(solution.replay_artifacts) | {policy_ref: nested_policy}
        ),
    )

    def _forward_loop_must_not_run(**_kwargs: Unpack[_SimulationKwargs]) -> NoReturn:
        raise AssertionError("forward simulation ran before replay-route preflight")

    monkeypatch.setattr(model_module, "simulate", _forward_loop_must_not_run)
    with pytest.raises(
        InvalidSimulationInputError,
        match=r"undeclared=.*pylcm\.simulation\.policy",
    ):
        model.simulate(
            params=params,
            initial_conditions=initial_conditions,
            solution=malformed,
            log_level="off",
        )


def test_values_only_dissolution_result_is_refused_before_forward_simulation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _make_dissolution_model()
    solution = model.solve(
        params=_DISSOLUTION_PARAMS,
        log_level="off",
        retention=ResultRetention.VALUES,
    )

    def _forward_loop_must_not_run(**_kwargs: Unpack[_SimulationKwargs]) -> NoReturn:
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


@pytest.mark.parametrize("defect", ["type", "dtype", "shape"])
def test_malformed_dissolution_flag_is_refused_before_forward(
    *, defect: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _make_dissolution_model()
    solution = model.solve(params=_DISSOLUTION_PARAMS, log_level="off")
    flag_ref = next(
        ref for ref in solution.replay_artifacts if ref.key == DISSOLUTION_FLAG
    )
    flag = solution.replay_artifacts[flag_ref]
    if defect == "type":
        malformed_flag = np.asarray(flag, dtype=np.bool_)
    elif defect == "dtype":
        malformed_flag = jnp.asarray(flag, dtype=jnp.float32)
    else:
        malformed_flag = jnp.reshape(jnp.asarray(flag), (*jnp.asarray(flag).shape, 1))
    entries = dict(solution.replay_artifacts)
    entries[flag_ref] = malformed_flag
    malformed = replace(solution, replay_artifacts=ArtifactStore(entries))

    def _forward_loop_must_not_run(**_kwargs: Unpack[_SimulationKwargs]) -> NoReturn:
        raise AssertionError("forward simulation ran before dissolution preflight")

    monkeypatch.setattr(model_module, "simulate", _forward_loop_must_not_run)
    with pytest.raises(
        InvalidSimulationInputError,
        match=r"mismatched_payload|artifact payloads cannot be detached",
    ):
        model.simulate(
            params=_DISSOLUTION_PARAMS,
            initial_conditions={},
            solution=malformed,
            log_level="off",
        )


def test_dissolution_flag_is_refused_from_the_wrong_artifact_channel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _make_dissolution_model()
    solution = model.solve(params=_DISSOLUTION_PARAMS, log_level="off")
    flag_ref = next(
        ref for ref in solution.replay_artifacts if ref.key == DISSOLUTION_FLAG
    )
    replay_entries = dict(solution.replay_artifacts)
    flag = replay_entries.pop(flag_ref)
    malformed = replace(
        solution,
        replay_artifacts=ArtifactStore(replay_entries),
        auxiliary_artifacts=ArtifactStore(
            dict(solution.auxiliary_artifacts) | {flag_ref: flag}
        ),
    )

    def _forward_loop_must_not_run(**_kwargs: Unpack[_SimulationKwargs]) -> NoReturn:
        raise AssertionError("forward simulation ran before dissolution preflight")

    monkeypatch.setattr(model_module, "simulate", _forward_loop_must_not_run)
    with pytest.raises(InvalidSimulationInputError, match="wrong channels"):
        model.simulate(
            params=_DISSOLUTION_PARAMS,
            initial_conditions={},
            solution=malformed,
            log_level="off",
        )


def _small_grid_search_inputs(
    *,
    execution_config: ExecutionConfig = ExecutionConfig(),  # noqa: B008
) -> tuple[Model, UserParams, UserInitialConditions]:
    model = get_model(
        n_periods=2,
        wealth_grid=LinSpacedGrid(start=1, stop=3, n_points=3),
        consumption_grid=LinSpacedGrid(start=1, stop=3, n_points=3),
        execution_config=execution_config,
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
    replacement = ArtifactStore(dict(store) | {ref: cast("ArtifactPayload", object())})
    return replace(
        solution,
        **{channel: replacement},
    )


def _policy_refs(solution: SolutionResult) -> set[ArtifactRef]:
    """Every simulation-policy coordinate the result accounts for."""
    return {
        ref
        for ref in (*solution.replay_artifacts, *solution.omissions)
        if ref.key == SIMULATION_POLICY
    }


def test_all_persistable_retains_the_adaptive_nnbegm_policy() -> None:
    """The nested policy carries its outer nodes in its own descriptor, so it
    is as self-contained as the finite bank and persists like it."""
    model = _build("adaptive")
    solution = model.solve(
        params=_PARAMS,
        log_level="off",
        retention=ResultRetention.ALL_PERSISTABLE_ARTIFACTS,
    )

    refs = _policy_refs(solution)
    assert refs
    assert all(
        isinstance(solution.replay_artifacts[ref], NestedEGMSimPolicy) for ref in refs
    )


def test_all_persistable_retains_the_finite_nnbegm_policy() -> None:
    """The finite candidate bank is self-contained, so it persists."""
    model = _build("finite")
    solution = model.solve(
        params=_PARAMS,
        log_level="off",
        retention=ResultRetention.ALL_PERSISTABLE_ARTIFACTS,
    )

    refs = _policy_refs(solution)
    assert refs
    assert all(
        isinstance(solution.replay_artifacts[ref], NNBEGMSimPolicy) for ref in refs
    )


def test_values_and_replay_retains_the_adaptive_nnbegm_policy() -> None:
    model = _build("adaptive")
    solution = model.solve(
        params=_PARAMS,
        log_level="off",
        retention=ResultRetention.VALUES_AND_REPLAY,
    )

    refs = _policy_refs(solution)
    assert refs
    assert all(
        isinstance(solution.replay_artifacts[ref], NestedEGMSimPolicy) for ref in refs
    )


@pytest.mark.parametrize("width", [1, 3])
@pytest.mark.parametrize("use_labels", [True, False])
def test_value_frame_labels_grid_nodes(
    *, width: int, use_labels: bool, tmp_path: Path
) -> None:
    """Each labelled row retains its value through blocking and persistence."""

    @categorical(ordered=True)
    class Kind:
        low: ScalarInt
        high: ScalarInt

    @categorical(ordered=False)
    class Alive:
        alive: ScalarInt

    def utility(*, wealth: FloatND, kind: ScalarInt) -> FloatND:
        return wealth + 10 * kind

    model = Model(
        regimes={
            "alive": Regime(
                states={
                    "wealth": LinSpacedGrid(start=1, stop=3, n_points=3),
                    "kind": DiscreteGrid(Kind),
                },
                functions={"utility": utility},
            )
        },
        ages=AgeGrid(exact_values=(18,)),
        edges={},
        initial_nodes=((18, "alive"),),
        execution_config=ExecutionConfig(axis_widths={"cell": width}),
        regime_id_class=Alive,
    )
    result = model.solve(params={}, log_level="off")
    frame = result.value_frame(period=0, regime="alive", use_labels=use_labels)
    assert tuple(frame.columns) == (
        *result.metadata.value_schemas[0, "alive"].axis_names,
        "V",
    )
    np.testing.assert_array_equal(frame["wealth"], [1, 2, 3, 1, 2, 3])
    np.testing.assert_array_equal(frame["V"], [1, 2, 3, 11, 12, 13])
    np.testing.assert_array_equal(
        frame["V"], np.asarray(result.value(period=0, regime="alive")).ravel()
    )
    if use_labels:
        assert frame["kind"].tolist() == ["low"] * 3 + ["high"] * 3
        assert frame["kind"].cat.ordered
    else:
        np.testing.assert_array_equal(frame["kind"], [0, 0, 0, 1, 1, 1])
    restored = load_solution(path=result.save(path=tmp_path / "labelled"))
    assert_frame_equal(
        frame, restored.value_frame(period=0, regime="alive", use_labels=use_labels)
    )
    frame["V"] = -99.0
    np.testing.assert_array_equal(
        result.value_frame(period=0, regime="alive")["V"], [1, 2, 3, 11, 12, 13]
    )


@pytest.mark.parametrize("axis_names", [("V",), ("wealth", "wealth")])
def test_value_frame_rejects_colliding_column_names(
    axis_names: tuple[str, ...],
) -> None:
    """A frame must not silently overwrite coordinates with another column."""
    shape = (2,) * len(axis_names)
    result = SolutionResult(
        values=ValueStore({(0, "alive"): jnp.ones(shape, dtype=jnp.float32)}),
        metadata=SolutionMetadata(
            retention=ResultRetention.VALUES,
            n_periods=1,
            regime_names=("alive",),
            solver_types={"alive": "example.Grid"},
            model_instance_id="model-1",
            params_fingerprint="0" * 64,
            value_schemas={
                (0, "alive"): ValueArraySchema(
                    shape=shape,
                    dtype="float32",
                    axis_names=axis_names,
                    named_axes=tuple(
                        AxisDescriptor(
                            name=name,
                            length=2,
                            role=AxisRole.STATE,
                            coordinates=(1.0, 2.0),
                        )
                        for name in axis_names
                    ),
                )
            },
        ),
    )
    np.testing.assert_array_equal(
        result.value(period=0, regime="alive"), np.ones(shape)
    )
    with pytest.raises(ValueError, match=r"unique.*'V'"):
        result.value_frame(period=0, regime="alive")


def test_value_frame_scalar_and_collective() -> None:
    """Scalar values have one row and collective values identify stakeholders."""

    @categorical(ordered=False)
    class Household:
        alone: ScalarInt
        couple: ScalarInt

    model = Model(
        regimes={
            "alone": Regime(functions={"utility": lambda: 7.0}),
            "couple": Regime(
                functions={
                    "utility": CollectiveUtility(
                        utilities={"f": lambda: 3.0, "m": lambda: 5.0}
                    )
                }
            ),
        },
        ages=AgeGrid(exact_values=(18,)),
        edges={},
        initial_nodes=((18, "alone"), (18, "couple")),
        regime_id_class=Household,
    )
    result = model.solve(params={}, log_level="off")
    assert result.value_frame(period=0, regime="alone").to_dict("list") == {"V": [7.0]}
    assert result.value_frame(period=0, regime="couple").to_dict("list") == {
        "stakeholder": ["f", "m"],
        "V": [3.0, 5.0],
    }
    with pytest.raises(KeyError):
        result.value_frame(period=2, regime="alone")


@pytest.mark.parametrize("collective", [False, True])
@pytest.mark.parametrize("categorical_state", [False, True])
def test_value_frame_preserves_state_named_stakeholder(
    *, collective: bool, categorical_state: bool
) -> None:
    """The collective axis is appended after the ordinary state axes."""

    @categorical(ordered=False)
    class Alive:
        alive: ScalarInt

    @categorical(ordered=False)
    class Kind:
        low: ScalarInt
        high: ScalarInt

    def utility(*, stakeholder: FloatND | IntND) -> FloatND:
        return stakeholder.astype(float)

    grid = (
        DiscreteGrid(Kind)
        if categorical_state
        else LinSpacedGrid(start=1, stop=3, n_points=3)
    )
    coordinates = [0, 1] if categorical_state else [1, 2, 3]

    model = Model(
        regimes={
            "alive": Regime(
                states={"stakeholder": grid},
                functions={
                    "utility": (
                        CollectiveUtility(utilities={"f": utility, "m": utility})
                        if collective
                        else utility
                    )
                },
            )
        },
        ages=AgeGrid(exact_values=(18,)),
        regime_id_class=Alive,
        edges={},
        initial_nodes=((18, "alive"),),
    )
    result = model.solve(params={}, log_level="off")
    axes = result.metadata.value_schemas[0, "alive"].named_axes
    assert axes[0].role is AxisRole.STATE
    if collective:
        assert axes[-1].role is AxisRole.STAKEHOLDER
        np.testing.assert_array_equal(
            result.value(period=0, regime="alive"),
            np.repeat(np.asarray(coordinates)[:, None], 2, axis=1),
        )
        with pytest.raises(ValueError, match="unique"):
            result.value_frame(period=0, regime="alive")
    else:
        frame = result.value_frame(period=0, regime="alive", use_labels=False)
        np.testing.assert_array_equal(frame["stakeholder"], coordinates)
        np.testing.assert_array_equal(frame["V"], coordinates)


@pytest.mark.parametrize("grid_kind", ["process", "age"])
def test_value_frame_uses_resolved_grid_coordinates(*, grid_kind: str) -> None:
    """Coordinates follow runtime parameters and the requested period's grid."""
    from lcm import AgeSpecializedGrid, NormalIIDProcess  # noqa: PLC0415

    @categorical(ordered=False)
    class Alive:
        alive: ScalarInt

    grid = (
        NormalIIDProcess(n_points=3, gauss_hermite=False)
        if grid_kind == "process"
        else AgeSpecializedGrid(
            build=lambda age: LinSpacedGrid(
                start=float(age), stop=float(age) + 2, n_points=3
            ),
            signature=lambda age: age,
        )
    )
    model = Model(
        regimes={
            "alive": Regime(
                states={"wealth": grid}, functions={"utility": lambda wealth: wealth}
            )
        },
        ages=AgeGrid(exact_values=(18, 19)),
        regime_id_class=Alive,
        edges={},
        initial_nodes=((18, "alive"), (19, "alive")),
    )
    params = (
        {"alive": {"wealth": {"mu": 1.0, "sigma": 0.5, "n_std": 2.0}}}
        if grid_kind == "process"
        else {}
    )
    result = model.solve(params=params, log_level="off")
    for period in (0, 1):
        frame = result.value_frame(period=period, regime="alive")
        expected = (
            [0.0, 1.0, 2.0]
            if grid_kind == "process"
            else [18 + period, 19 + period, 20 + period]
        )
        np.testing.assert_array_equal(frame["wealth"], expected)
        np.testing.assert_array_equal(frame["V"], expected)
        np.testing.assert_array_equal(
            frame["wealth"],
            model.state_grid(
                params=params, regime_name="alive", state_name="wealth", period=period
            ),
        )
