"""Public lower-only candidate matches a real solve's unoptimized lowering.

Source-only draft: the requested public API deliberately does not exist yet.
"""

import dataclasses
import enum
import functools
import hashlib
import json
import logging
import os
import sys
from collections.abc import Callable, Hashable, Mapping
from importlib.metadata import distribution
from pathlib import Path
from types import MappingProxyType, ModuleType
from typing import Literal, Never, Protocol, TypedDict, Unpack, overload

import jax
import numpy as np
import pytest

import hatch_build
import lcm
from _lcm.egm.upper_envelope._exact_affine.ffi import _installed_native_directory
from _lcm.regime_building.age_specialization import INVARIANT
from _lcm.solution import backward_induction as engine
from _lcm.solution import lower_candidate as candidate_lowering
from _lcm.solution.continuation_arguments import MARGINAL_ARGUMENT
from _lcm.solution.fingerprint import _semantic_fingerprint
from _lcm.solution.lowering_descriptors import describe_lowering_value
from _lcm.typing import HostArray, JSONValue, LoweringDescriptor
from lcm.exceptions import ExecutionPlanningError
from lcm.model import _SolutionPreparation
from lcm.solver_api import EGM_CONTINUATION, ResultRetention
from lcm.typing import RegimeName, UserParams
from tests.solution._callback_types import PlanningKwargs as _ContextKwargs
from tests.solution._callback_types import PlanningResult as _ResolvedContexts
from tests.solution._candidate_census import _WaveKwargs
from tests.solution.test_solution_result import _KernelKwargs
from tests.test_models import nbegm_ride_along_toy


class _ContextResolver(Protocol):
    def __call__(
        self,
        *,
        all_programs: Mapping[engine._CoreTriple, engine.CoreProgram],
        regimes: MappingProxyType[engine.RegimeName, engine.Regime],
        program_fingerprint: str,
        flat_params: engine.FlatParams,
        ages: engine.TimeAxis,
        next_regime_to_V_arr: MappingProxyType[engine.RegimeName, engine.FloatND],
        next_regime_to_continuation: MappingProxyType[
            engine.RegimeName, engine.ContinuationPayload
        ],
        next_edge_to_V_arr: MappingProxyType[engine._EdgeKey, engine.FloatND],
        budget_bytes: int | None,
        execution_widths: engine.ResolvedExecution,
        enable_jit: bool,
        continuous_sharded_state: engine.StateName | None = ...,
        donate_buffers: bool = ...,
        retain_all_artifacts: bool,
        persistable_artifact_refs: frozenset[engine.ArtifactRef],
        process_grid_resolver: engine.ProcessGridResolver | None = ...,
        structural_blueprints: engine.StructuralBlueprintCache[
            engine._StructuralBlueprint
        ]
        | None = ...,
        base_state_action_spaces: Mapping[engine.RegimeName, engine.StateActionSpace]
        | None = ...,
        logger: logging.Logger | None = ...,
    ) -> _ResolvedContexts: ...


class _PublicLowerCallback(Protocol):
    def __call__(
        self,
        *,
        params: UserParams,
        log_level: Literal["off"],
        candidate: lcm.PeriodCandidate,
        retention: ResultRetention,
    ) -> lcm.LoweredPeriodCandidate: ...


type _Manifest = MappingProxyType[str, LoweringDescriptor | JSONValue]
type _Authority = dict[str, LoweringDescriptor | JSONValue]
type _Contexts = dict[engine._CoreCandidate, _Manifest]
type _Active = list[_Manifest | None]
type _Observed = list[tuple[_Manifest, bytes]]
type _DispatchAddress = tuple[
    str, int, str, tuple[tuple[str, int], ...], tuple[str, ...]
]


class _LowerKwargs(TypedDict):
    resolved: engine.ResolvedCoreProgram
    layout: engine.ResolvedOutputLayout
    donated: tuple[str, ...]
    internal_templates: Mapping[engine.ReferenceName, engine.ShapeDtypePytree]
    label: str


class _LowerCallback(Protocol):
    def __call__(
        self,
        *,
        resolved: engine.ResolvedCoreProgram,
        layout: engine.ResolvedOutputLayout,
        donated: tuple[str, ...],
        internal_templates: Mapping[engine.ReferenceName, engine.ShapeDtypePytree],
        label: str,
    ) -> jax.stages.Lowered: ...


class _WaveCallback(Protocol):
    def __call__(
        self,
        *,
        new_lowerings: Mapping[Hashable, engine._CoreCandidate],
        resolved_programs: Mapping[engine._CoreCandidate, engine.ResolvedCoreProgram],
        all_layouts: Mapping[engine._CoreTriple, engine.ResolvedOutputLayout],
        internal_templates: Mapping[
            engine._CoreCandidate,
            Mapping[engine.ReferenceName, engine.ShapeDtypePytree],
        ],
        donations: Mapping[engine._CoreCandidate, tuple[engine.ResolvedDonation, ...]],
        ages: engine.TimeAxis,
        n_triples_per_lowering: Mapping[Hashable, int],
        log_kernel_memory: bool,
        n_workers: int,
        logger: logging.Logger,
        compiled: dict[Hashable, jax.stages.Compiled],
        labels: dict[Hashable, str],
    ) -> None: ...


class _DispatchCallback(Protocol):
    def __call__(
        self,
        *,
        regime: engine.Regime,
        regime_name: engine.RegimeName,
        period: int,
        compiled_cores: MappingProxyType[str, engine.PlannedCore],
        capture_target: engine.PeriodCaptureTarget | None,
        state_action_space: engine.StateActionSpace,
        flat_params: engine.FlatParams,
        ages: engine.TimeAxis,
        next_regime_to_V_arr: MappingProxyType[engine.RegimeName, engine.FloatND],
        next_regime_to_continuation: MappingProxyType[
            engine.RegimeName, engine.ContinuationPayload
        ],
        logger: logging.Logger,
        next_edge_to_V_arr: MappingProxyType[engine._EdgeKey, engine.FloatND],
        period_solution: Mapping[engine.RegimeName, engine.FloatND],
        retain_replay: bool,
        selected_artifact_keys: frozenset[engine.ArtifactKey],
        period_capture: engine.CaptureContext | None = ...,
        captured_admission: Mapping[str, Mapping[str, int | None]] = ...,
    ) -> engine.KernelOutput: ...


def _candidate_address(manifest: _Manifest) -> _DispatchAddress:
    regime, period, core = manifest["regime"], manifest["period"], manifest["core"]
    assert isinstance(regime, str)
    assert isinstance(period, int)
    assert isinstance(core, str)
    widths = manifest["widths"]
    assert isinstance(widths, Mapping)
    width_items: list[tuple[str, int]] = []
    for name, width in widths.items():
        assert isinstance(name, str)
        assert isinstance(width, int)
        width_items.append((name, width))
    donated = manifest["primary_donated"]
    assert isinstance(donated, tuple)
    names: list[str] = []
    for name in donated:
        assert isinstance(name, str)
        names.append(name)
    return (
        regime,
        period,
        core,
        tuple(sorted(width_items)),
        tuple(names),
    )


def _selected_artifact_keys(manifest: _Manifest) -> frozenset[LoweringDescriptor]:
    value = manifest["selected_artifact_keys"]
    assert isinstance(value, frozenset)
    return value


class _CandidateKwargs(TypedDict):
    regime: RegimeName
    period: int
    core: str
    widths: dict[str, int]


def _candidate_kwargs(manifest: _Manifest) -> _CandidateKwargs:
    regime, period, core, widths, _donated = _candidate_address(manifest=manifest)
    return {"regime": regime, "period": period, "core": core, "widths": dict(widths)}


def _manifest_text(*, manifest: _Manifest, key: str) -> str:
    value = manifest[key]
    assert isinstance(value, str)
    return value


class _PrepareKwargs(TypedDict):
    flat_params: engine.FlatParams
    log: logging.Logger
    retention: ResultRetention
    process_grid_resolver: engine.ProcessGridResolver | None
    call_id: engine.CallId | None


class _FallbackKwargs(TypedDict):
    fallback_keys: dict[engine._CoreCandidate, Hashable]
    fallback_donations: dict[engine._CoreCandidate, tuple[engine.ResolvedDonation, ...]]
    argument_keys: dict[engine._CoreTriple, Hashable]


pytestmark = [
    pytest.mark.requires(device="cpu"),
    pytest.mark.coverage(backends=("cpu",), precisions="both"),
]


def _describe(
    value: object,  # noqa: PAN001 - JAX tree-definition auxiliary metadata accepts arbitrary Python values.
) -> LoweringDescriptor:
    """Copy descriptor data; reject unknown live objects instead of retaining them."""
    if isinstance(value, enum.Enum):
        return ("enum", type(value).__module__, type(value).__qualname__, value.name)
    if value is None or (
        isinstance(value, str | int | bytes) and type(value) in (str, bool, int, bytes)
    ):
        return value
    if isinstance(value, float):
        return ("float", value.hex())
    if isinstance(value, np.dtype):
        return ("dtype", value.str)
    if isinstance(value, type):
        return ("type", value.__module__, value.__qualname__)
    return _describe_tree(value)


def _describe_jax(
    value: jax.Array
    | jax.ShapeDtypeStruct
    | HostArray
    | jax.tree_util.PyTreeDef
    | jax.sharding.Sharding
    | jax.sharding.AbstractMesh,
) -> LoweringDescriptor:
    """Copy the concrete JAX descriptor variants used by this fixture."""
    if isinstance(value, (jax.Array, jax.ShapeDtypeStruct, np.ndarray)):
        return (
            "array",
            tuple(value.shape),
            str(value.dtype),
            bool(getattr(value, "weak_type", False)),
            _describe(getattr(value, "sharding", None)),
        )
    if isinstance(value, jax.tree_util.PyTreeDef):
        return (
            "pytree",
            _describe(value.node_data()),
            tuple(_describe(child) for child in value.children()),
        )
    if isinstance(value, jax.sharding.Sharding):
        assert isinstance(
            value, (jax.sharding.NamedSharding, jax.sharding.SingleDeviceSharding)
        )
        mesh = getattr(value, "mesh", None)
        return (
            type(value).__qualname__,
            value.memory_kind,
            tuple(
                sorted((d.platform, d.process_index, d.id) for d in value.device_set)
            ),
            None if mesh is None else tuple(mesh.shape.items()),
            None if mesh is None else _describe(mesh.axis_types),
            None if mesh is None else tuple(d.id for d in mesh.devices.flat),
            None
            if not hasattr(value, "spec")
            else (
                _describe(tuple(value.spec)),
                _describe(value.spec.reduced),
                _describe(value.spec.unreduced),
            ),
        )
    if isinstance(value, jax.sharding.AbstractMesh):
        return (
            "abstract_mesh",
            _describe(value.shape_tuple),
            _describe(value.axis_types),
            _describe(value.abstract_device),
        )
    raise AssertionError(f"Unspecified JAX descriptor type: {type(value)}")


def _describe_tree(
    value: object,  # noqa: PAN001 - JAX tree-definition auxiliary metadata accepts arbitrary Python values.
) -> LoweringDescriptor:
    """Copy structural containers without saving their live leaves."""
    if value is INVARIANT:
        return ("singleton", "_lcm.regime_building.age_specialization", "INVARIANT")
    if isinstance(
        value,
        (
            jax.Array,
            jax.ShapeDtypeStruct,
            np.ndarray,
            jax.tree_util.PyTreeDef,
            jax.sharding.Sharding,
            jax.sharding.AbstractMesh,
        ),
    ):
        return _describe_jax(value)
    if isinstance(value, Mapping):
        return MappingProxyType({_describe(k): _describe(v) for k, v in value.items()})
    if dataclasses.is_dataclass(value):
        return (
            type(value).__module__,
            type(value).__qualname__,
            tuple(
                (f.name, _describe(getattr(value, f.name)))
                for f in dataclasses.fields(value)
            ),
        )
    if isinstance(value, (tuple, list)):
        return tuple(_describe(v) for v in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_describe(v) for v in value)
    raise AssertionError(f"Unspecified descriptor type: {type(value)}")


def _forbid_candidate_execution[Argument](
    *_args: Argument, **_kwargs: Argument
) -> Never:
    raise AssertionError("lower-only submitted, compiled or dispatched a candidate")


@pytest.mark.parametrize(
    "retention",
    [ResultRetention.VALUES_AND_REPLAY, ResultRetention.ALL_PERSISTABLE_ARTIFACTS],
    ids=["donated-continuation", "persisted-continuation"],
)
def test_public_lower_period_candidate_matches_solve(
    *,
    retention: ResultRetention,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A requested primary candidate preserves solve's exact lowering contract."""
    model = nbegm_ride_along_toy.build_model(
        variant="nbegm",
        n_liquid=8,
        n_savings=10,
        n_consumption=12,
    )
    params = nbegm_ride_along_toy.build_params()
    # These containers hold copied descriptors, integer identities and raw bytes only.
    authority: _Authority = {}
    contexts: _Contexts = {}
    active: _Active = []
    fallback_contexts: dict[engine._CoreCandidate, str] = {}
    observed: _Observed = []
    dispatched: set[_DispatchAddress] = set()
    prepare = lcm.Model._prepare_solution
    bind_fallbacks = engine._LazyCandidateFrontier.bind_fallbacks
    identities = _capture_source_runtime_identity()

    def observe_prepare(
        self: lcm.Model, **kwargs: Unpack[_PrepareKwargs]
    ) -> _SolutionPreparation:
        result = prepare(self, **kwargs)
        authority.update(
            model_identity=result.model_fingerprint,
            program_identity=result.program_fingerprint,
            parameter_identity=self._params_fingerprint(
                flat_params=kwargs["flat_params"]
            ),
            artifact_refs=_describe(result.persistable_artifact_refs),
        )
        return result

    observe_context = functools.partial(
        _observe_context,
        resolve=engine._resolve_output_layouts_and_lowering_keys,
        contexts=contexts,
    )

    def observe_fallbacks(
        self: engine._LazyCandidateFrontier, **kwargs: Unpack[_FallbackKwargs]
    ) -> None:
        result = bind_fallbacks(self, **kwargs)
        for candidate, key in kwargs["fallback_keys"].items():
            fallback_contexts[candidate] = _semantic_fingerprint(_describe(key))
        return result

    observe_wave = functools.partial(
        _observe_wave,
        run_wave=engine._lower_and_compile_wave,
        contexts=contexts,
        fallback_contexts=fallback_contexts,
        active=active,
    )

    observe_lower = functools.partial(
        _observe_lower,
        lower=engine._lower_resolved_candidate,
        active=active,
        authority=authority,
        identities=identities,
        retention=retention,
        observed=observed,
    )

    observe_dispatch = functools.partial(
        _observe_dispatch, dispatch=engine._run_period_kernel, dispatched=dispatched
    )

    with monkeypatch.context() as reference:
        reference.setattr(lcm.Model, "_prepare_solution", observe_prepare)
        reference.setattr(
            engine, "_resolve_output_layouts_and_lowering_keys", observe_context
        )
        reference.setattr(
            engine._LazyCandidateFrontier, "bind_fallbacks", observe_fallbacks
        )
        reference.setattr(engine, "_lower_and_compile_wave", observe_wave)
        reference.setattr(engine, "_lower_resolved_candidate", observe_lower)
        reference.setattr(engine, "_run_period_kernel", observe_dispatch)
        model.solve(params=params, log_level="off", retention=retention)
    matches = [
        (manifest, ir)
        for manifest, ir in observed
        if _candidate_address(manifest=manifest) in dispatched
        and _candidate_address(manifest=manifest)[1] < model.n_periods - 1
        and bool(manifest["primary_donated"])
        == (retention is ResultRetention.VALUES_AND_REPLAY)
    ]
    assert matches, "No genuine primary continuation candidate reached dispatch"
    expected, expected_ir = matches[0]
    assert expected["continuation"]
    if retention is ResultRetention.ALL_PERSISTABLE_ARTIFACTS:
        assert _describe(EGM_CONTINUATION) in _selected_artifact_keys(manifest=expected)
        assert expected["primary_donated"] == ()
    else:
        assert expected["primary_donated"]

    # Missing public behavior is reached only after the real reference is qualified.
    candidate = _require_public_member(owner=lcm, name="PeriodCandidate")(
        **_candidate_kwargs(manifest=expected)
    )
    with monkeypatch.context() as bounded:
        bounded.setattr(engine.CompilationWave, "_submit", _forbid_candidate_execution)
        bounded.setattr(engine, "_compile_and_log", _forbid_candidate_execution)
        bounded.setattr(engine, "_run_period_kernel", _forbid_candidate_execution)
        bounded.setattr(engine, "_run_dispatch_unit", _forbid_candidate_execution)
        result = _require_public_member(owner=model, name="lower_period_candidate")(
            params=params,
            log_level="off",
            candidate=candidate,
            retention=retention,
        )
    assert isinstance(result.stablehlo, bytes)
    assert result.stablehlo == expected_ir
    assert result.manifest == expected
    # Fail closed on live payloads, including unexpected result fields.
    _assert_immutable_result(result)


def _assert_immutable_result[Value](value: Value) -> None:
    """Accept only immutable descriptor trees; never coerce live objects away."""
    if value is None or type(value) in (str, bool, int, bytes):
        return
    if isinstance(value, MappingProxyType):
        for key, child in value.items():
            _assert_immutable_result(key)
            _assert_immutable_result(child)
        return
    if isinstance(value, (tuple, frozenset)):
        for child in value:
            _assert_immutable_result(child)
        return
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        assert getattr(value, "__dataclass_params__").frozen  # noqa: B009 - DataclassInstance omits generated parameter attributes.
        for field in dataclasses.fields(value):
            _assert_immutable_result(getattr(value, field.name))
        return
    raise AssertionError(f"Mutable or live lower-only result payload: {type(value)}")


def _capture_source_runtime_identity() -> Mapping[str, JSONValue]:
    """Read exact bytes before observation; reuse existing native/source seals."""
    root = Path(hatch_build.__file__).resolve().parent
    sources = tuple(
        (
            path.relative_to(root).as_posix(),
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        for package in ("lcm", "_lcm")
        for path in sorted((root / "src" / package).rglob("*.py"))
    )
    assert sources
    directory = _installed_native_directory()
    manifest_bytes = (directory / hatch_build.NATIVE_MANIFEST).read_bytes()
    manifest = json.loads(manifest_bytes)
    encoded_inputs = json.dumps(
        manifest["inputs"], sort_keys=True, separators=(",", ":")
    ).encode()
    assert hashlib.sha256(encoded_inputs).hexdigest() == manifest["fingerprint"]
    native_source = hatch_build.native_source_fingerprint(root=root)
    assert manifest["inputs"]["source"] == native_source
    libraries = []
    for name in manifest["libraries"]:
        assert Path(name).name == name
        libraries.append(
            (name, hashlib.sha256((directory / name).read_bytes()).hexdigest())
        )
    assert libraries
    distributions = []
    # Initial witness is CPU-only; plugin/GPU identity is a separate admission.
    for name in ("jax", "jaxlib", "numpy"):
        package = distribution(name)
        assert package.files
        files = tuple(
            (
                str(path),
                hashlib.sha256(
                    Path(str(package.locate_file(path))).read_bytes()
                ).hexdigest(),
            )
            for path in sorted(package.files, key=str)
            if Path(str(path)).suffix in (".py", ".so", ".pyd", ".dll", ".dylib")
        )
        assert files
        distributions.append((name, files))
    devices = tuple(
        (
            device.id,
            device.process_index,
            device.platform,
            device.device_kind,
            device.client.platform_version,
        )
        for device in jax.devices()
    )
    assert devices
    assert all(device[2] == "cpu" for device in devices)
    return MappingProxyType(
        {
            "source_files": sources,
            "source_identity": _semantic_fingerprint(sources),
            "native_source_identity": native_source,
            "native_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "native_build_identity": manifest["fingerprint"],
            "native_library_bytes": tuple(libraries),
            "runtime_files": tuple(distributions),
            "runtime_identity": _semantic_fingerprint(tuple(distributions)),
            "python_binary_sha256": hashlib.sha256(
                Path(sys.executable).read_bytes()
            ).hexdigest(),
            "python_abi": sys.implementation.cache_tag,
            "devices": devices,
            "xla_flags": os.environ.get("XLA_FLAGS", ""),
        }
    )


def _observe_context(
    *, resolve: _ContextResolver, contexts: _Contexts, **kwargs: Unpack[_ContextKwargs]
) -> _ResolvedContexts:
    result = resolve(**kwargs)
    _layouts, keys, programs, _internal, _liveness, donations, _metadata, frontier = (
        result
    )
    # The initial tracer chooses an actually dispatched top-ranked candidate.
    # It does not hold frontier/program/layout objects after this call returns.
    for candidate, resolved in programs.items():
        triple, _widths = candidate
        regime, period, core = triple
        if regime != "alive" or MARGINAL_ARGUMENT not in resolved.arguments:
            continue
        selected = frozenset(
            ref.key
            for ref in kwargs["persistable_artifact_refs"]
            if ref.regime == regime and ref.period == period
        )
        context: dict[str, LoweringDescriptor | JSONValue] = {
            "primary_key": _semantic_fingerprint(_describe(keys[candidate])),
            "regime": regime,
            "period": period,
            "core": core,
            "widths": _describe(resolved.tile_widths),
            "rank": frontier.candidates_by_triple[triple].index(candidate),
            "primary_donated": engine._donated_arguments(
                donations=donations[candidate]
            ),
            "selected_artifact_keys": _describe(selected),
            "continuation": _describe(resolved.arguments[MARGINAL_ARGUMENT]),
            "requirements": _describe(resolved.requirements),
            "specialization": _describe(resolved.specialization_key),
            "transfer_plan": _describe(resolved.input_transfer_plan),
            "compiler_options": _describe(resolved.compiler_options),
            "trace_settings": _describe(engine._trace_settings_key()),
            "execution": _describe(kwargs["execution_widths"]),
        }
        contexts[candidate] = MappingProxyType(context)
    return result


def _observe_wave(
    *,
    run_wave: _WaveCallback,
    contexts: _Contexts,
    fallback_contexts: dict[engine._CoreCandidate, str],
    active: _Active,
    **kwargs: Unpack[_WaveKwargs],
) -> None:
    assert not active
    for key, candidate in kwargs["new_lowerings"].items():
        context = contexts.get(candidate)
        if context is None:
            active.append(None)
            continue
        token = _semantic_fingerprint(_describe(key))
        fallback_key = fallback_contexts.get(candidate)
        assert fallback_key is None or fallback_key != context["primary_key"]
        assert token in (context["primary_key"], fallback_key)
        active.append(
            MappingProxyType(
                {
                    **context,
                    "variant": "primary"
                    if token == context["primary_key"]
                    else "fallback",
                    "dedup_key": token,
                    "dedup_fanout": kwargs["n_triples_per_lowering"][key],
                }
            )
        )
    try:
        result = run_wave(**kwargs)
        assert not active
        return result
    finally:
        active.clear()


def _observe_lower(
    *,
    lower: _LowerCallback,
    active: _Active,
    authority: _Authority,
    identities: Mapping[str, JSONValue],
    retention: ResultRetention,
    observed: _Observed,
    **kwargs: Unpack[_LowerKwargs],
) -> jax.stages.Lowered:
    result = lower(**kwargs)
    assert active, "Lowering outside the observed real wave"
    context = active.pop(0)
    if context is None or context["variant"] != "primary":
        return result
    assert kwargs["donated"] == context["primary_donated"]
    resolved = kwargs["resolved"]
    raw_ir = str(result.compiler_ir(dialect="stablehlo")).encode("utf-8")
    manifest = MappingProxyType(
        {
            **authority,
            **context,
            **identities,
            "schema": 1,
            "retention": retention.name,
            "inputs": _describe(resolved.arguments),
            "internal_inputs": _describe(kwargs["internal_templates"]),
            "static_kwargs": _describe(resolved.static_kwargs),
            "output_roles": _describe(resolved.output_roles),
            "output_layout": _describe(kwargs["layout"]),
            "ir_sha256": hashlib.sha256(raw_ir).hexdigest(),
            "compiled": False,
            "dispatched": False,
            "admission": "not_evaluated",
            "optimized_hlo": None,
            "buffer_assignment": None,
            "compiler_memory": None,
            "physical_residency": None,
        }
    )
    observed.append((manifest, raw_ir))
    return result  # SAME real Lowered; no copy, alternate lower or saved reference.


def _observe_dispatch(
    *,
    dispatch: _DispatchCallback,
    dispatched: set[_DispatchAddress],
    **kwargs: Unpack[_KernelKwargs],
) -> engine.KernelOutput:
    for core_name, core in kwargs["compiled_cores"].items():
        dispatched.add(
            (
                kwargs["regime_name"],
                kwargs["period"],
                core_name,
                tuple(sorted(core.tile_widths.items())),
                core.donated_arguments,
            )
        )
    return dispatch(**kwargs)


@overload
def _require_public_member(
    *, owner: ModuleType, name: Literal["PeriodCandidate"]
) -> type[lcm.PeriodCandidate]: ...


@overload
def _require_public_member(
    *, owner: lcm.Model, name: Literal["lower_period_candidate"]
) -> _PublicLowerCallback: ...


def _require_public_member(
    *,
    owner: ModuleType | lcm.Model,
    name: Literal["PeriodCandidate", "lower_period_candidate"],
) -> type[lcm.PeriodCandidate] | _PublicLowerCallback:
    """Require the proposed public behavior after the real reference qualifies."""
    assert hasattr(owner, name), f"Missing public API: {name}"
    member = getattr(owner, name)
    assert callable(member)
    return member


def test_public_lower_period_candidate_rejects_absent_members_before_lowering(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Invalid requests cannot substitute a real graph member or reach lowering."""
    model = nbegm_ride_along_toy.build_model(
        variant="nbegm", n_liquid=8, n_savings=10, n_consumption=12
    )
    params = nbegm_ride_along_toy.build_params()
    dispatched: set[_DispatchAddress] = set()
    observe_dispatch = functools.partial(
        _observe_dispatch, dispatch=engine._run_period_kernel, dispatched=dispatched
    )
    with monkeypatch.context() as reference:
        reference.setattr(engine, "_run_period_kernel", observe_dispatch)
        model.solve(
            params=params,
            log_level="off",
            retention=ResultRetention.VALUES_AND_REPLAY,
        )
    candidates = sorted(
        entry for entry in dispatched if entry[1] < model.n_periods - 1 and entry[4]
    )
    assert candidates, "No genuine donated continuation candidate reached dispatch"
    regime, period, core, width_items, _donated = candidates[0]
    widths = dict(width_items)
    assert widths, "The reference must expose execution-axis membership"
    absent_core = "__absent_core__"
    assert all(entry[2] != absent_core for entry in dispatched)
    incomplete_widths = dict(width_items[1:])
    assert set(incomplete_widths) < set(widths)
    requests = (
        (
            lcm.PeriodCandidate(
                regime=regime, period=model.n_periods, core=core, widths=widths
            ),
            "Candidate is absent from the selected solve graph.",
        ),
        (
            lcm.PeriodCandidate(
                regime=regime, period=period, core=absent_core, widths=widths
            ),
            "Candidate is absent from the selected solve graph.",
        ),
        (
            lcm.PeriodCandidate(
                regime=regime,
                period=period,
                core=core,
                widths=incomplete_widths,
            ),
            "Candidate widths are absent from the ranked frontier.",
        ),
    )

    def forbid_target_lowering[Argument](
        *_args: Argument, **_kwargs: Argument
    ) -> Never:
        raise AssertionError("An absent candidate reached target lowering")

    with monkeypatch.context() as bounded:
        bounded.setattr(
            candidate_lowering, "_lower_resolved_candidate", forbid_target_lowering
        )
        for candidate, message in requests:
            with pytest.raises(ExecutionPlanningError) as exc_info:
                model.lower_period_candidate(
                    params=params,
                    log_level="off",
                    candidate=candidate,
                    retention=ResultRetention.VALUES_AND_REPLAY,
                )
            assert str(exc_info.value) == message


_EXPLICIT = ("enum", "jax._src.mesh", "AxisType", "Explicit")


@pytest.mark.parametrize(
    ("mesh", "expected"),
    [
        (jax.sharding.AbstractMesh((), ()), ("abstract_mesh", (), (), None)),
        (
            jax.sharding.AbstractMesh((2, 3), ("x", "y")),
            ("abstract_mesh", (("x", 2), ("y", 3)), (_EXPLICIT, _EXPLICIT), None),
        ),
    ],
    ids=["empty-ambient", "two-axes"],
)
@pytest.mark.parametrize(
    "describe", [_describe, describe_lowering_value], ids=["tracer", "public"]
)
def test_describe_spells_abstract_mesh_by_axis_names_sizes_and_types(
    *,
    describe: Callable[[jax.sharding.AbstractMesh], LoweringDescriptor],
    mesh: jax.sharding.AbstractMesh,
    expected: LoweringDescriptor,
) -> None:
    """An abstract mesh, such as JAX's ambient trace-context mesh, is described by
    its axis names, sizes, types and abstract device."""
    assert describe(mesh) == expected
