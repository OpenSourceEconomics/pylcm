"""Genuine GPU reference for the proposed lower-only identity profile."""

import csv
import functools
import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Mapping
from importlib.metadata import distribution, distributions
from pathlib import Path
from types import MappingProxyType
from typing import Any

import jax
import pytest

import hatch_build
import lcm
from _lcm.egm.upper_envelope._exact_affine.ffi import _installed_native_directory
from _lcm.solution import backward_induction as engine
from _lcm.solution.fingerprint import _semantic_fingerprint
from lcm.solver_api import EGM_CONTINUATION, ResultRetention
from tests.solution.test_public_lower_period_candidate import (
    _assert_immutable_result,
    _describe,
    _forbid_candidate_execution,
    _observe_context,
    _observe_dispatch,
    _observe_lower,
    _observe_wave,
    _require_public_member,
)
from tests.test_models import nbegm_ride_along_toy

pytestmark = [
    pytest.mark.requires(device="gpu"),
    pytest.mark.coverage(backends=("gpu-small", "gpu-large"), precisions="both"),
    pytest.mark.isolation(process="fresh"),
    pytest.mark.ci(tier="pr"),
]


@pytest.mark.parametrize(
    "retention",
    [ResultRetention.VALUES_AND_REPLAY, ResultRetention.ALL_PERSISTABLE_ARTIFACTS],
    ids=["donated-continuation", "persisted-continuation"],
)
def test_public_gpu_lower_period_candidate_matches_solve(
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
    identities = _capture_gpu_source_runtime_identity()
    observed, dispatched, dispatched_keys, wave_fanout = _run_gpu_reference(
        model=model,
        params=params,
        retention=retention,
        monkeypatch=monkeypatch,
        identities=identities,
    )
    matches = [
        (manifest, ir)
        for manifest, ir in observed
        if (
            manifest["regime"],
            manifest["period"],
            manifest["core"],
            tuple(sorted(manifest["widths"].items())),
            manifest["primary_donated"],
        )
        in dispatched
        and dispatched_keys[
            (
                manifest["regime"],
                manifest["period"],
                manifest["core"],
                tuple(sorted(manifest["widths"].items())),
                manifest["primary_donated"],
            )
        ]
        == manifest["primary_key"]
        and manifest["period"] < model.n_periods - 1
        and bool(manifest["primary_donated"])
        == (retention is ResultRetention.VALUES_AND_REPLAY)
    ]
    assert matches, "No genuine primary continuation candidate reached dispatch"
    expected, expected_ir = matches[0]
    assert wave_fanout[expected["dedup_key"]] > 0
    assert expected["dedup_fanout"] is None
    assert expected["continuation"]
    if retention is ResultRetention.ALL_PERSISTABLE_ARTIFACTS:
        assert _describe(EGM_CONTINUATION) in expected["selected_artifact_keys"]
        assert expected["primary_donated"] == ()
    else:
        assert expected["primary_donated"]

    fresh_model = nbegm_ride_along_toy.build_model(
        variant="nbegm", n_liquid=8, n_savings=10, n_consumption=12
    )
    fresh_params = nbegm_ride_along_toy.build_params()
    assert fresh_model is not model
    # A runtime/driver identity change during the real reference is a fixture
    # prerequisite failure, never the intended missing GPU profile RED.
    assert _capture_gpu_source_runtime_identity() == identities

    # Missing public behavior is reached only after the real reference is qualified.
    candidate = _require_public_member(owner=lcm, name="PeriodCandidate")(
        regime=expected["regime"],
        period=expected["period"],
        core=expected["core"],
        widths=expected["widths"],
    )
    with monkeypatch.context() as bounded:
        bounded.setattr(engine.CompilationWave, "_submit", _forbid_candidate_execution)
        bounded.setattr(engine, "_compile_and_log", _forbid_candidate_execution)
        bounded.setattr(engine, "_run_period_kernel", _forbid_candidate_execution)
        bounded.setattr(engine, "_run_dispatch_unit", _forbid_candidate_execution)
        result = _require_public_member(
            owner=fresh_model, name="lower_period_candidate"
        )(
            params=fresh_params,
            log_level="off",
            candidate=candidate,
            retention=retention,
        )
    assert isinstance(result.stablehlo, bytes)
    assert result.stablehlo == expected_ir
    assert result.manifest == expected
    # Fail closed on live payloads, including unexpected result fields.
    _assert_immutable_result(result)


def _run_gpu_reference(
    *,
    model: lcm.Model,
    params: Any,
    retention: ResultRetention,
    monkeypatch: pytest.MonkeyPatch,
    identities: Mapping[str, Any],
) -> tuple[list[Any], set[Any], dict[Any, str], dict[str, int]]:
    # These containers hold copied descriptors, integer identities and raw bytes only.
    authority: dict[str, Any] = {}
    contexts: dict[Any, Any] = {}
    active: list[Any] = []
    fallback_contexts: dict[Any, str] = {}
    observed: list[Any] = []
    dispatched: set[Any] = set()
    prepare = lcm.Model._prepare_solution
    bind_fallbacks = engine._LazyCandidateFrontier.bind_fallbacks
    wave_fanout: dict[str, int] = {}
    lowered_ids: dict[str, int] = {}
    compiled_keys: dict[int, str] = {}
    dispatched_keys: dict[Any, str] = {}
    compile_real = engine._compile_and_log

    def observe_prepare(self: Any, **kwargs: Any) -> Any:
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

    resolve = engine._resolve_output_layouts_and_lowering_keys

    def observe_context(**kwargs: Any) -> Any:
        # Keep ordinary GPU admission enabled. This is a budgeted profile, unlike
        # the unchanged CPU witness whose default device budget resolves to None.
        assert kwargs["execution_widths"].device_memory_bytes is not None
        return _observe_context(resolve=resolve, contexts=contexts, **kwargs)

    def observe_fallbacks(self: Any, **kwargs: Any) -> Any:
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

    copy_lower = functools.partial(
        _observe_lower,
        lower=engine._lower_resolved_candidate,
        active=active,
        authority=authority,
        identities=identities,
        retention=retention,
        observed=observed,
    )

    observe_lower = functools.partial(
        _observe_gpu_lower,
        copy_lower=copy_lower,
        observed=observed,
        wave_fanout=wave_fanout,
        lowered_ids=lowered_ids,
    )

    def observe_compile(**kwargs: Any) -> Any:
        token = _semantic_fingerprint(_describe(kwargs["lowering_key"]))
        if token in lowered_ids:
            assert id(kwargs["low"]) == lowered_ids[token]
        key, compiled = compile_real(**kwargs)
        compiled_keys[id(compiled)] = _semantic_fingerprint(_describe(key))
        return key, compiled

    dispatch = engine._run_period_kernel

    observe_dispatch = functools.partial(
        _observe_gpu_dispatch,
        dispatch=dispatch,
        dispatched=dispatched,
        dispatched_keys=dispatched_keys,
        compiled_keys=compiled_keys,
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
        reference.setattr(engine, "_compile_and_log", observe_compile)
        reference.setattr(engine, "_run_period_kernel", observe_dispatch)
        solution = model.solve(params=params, log_level="off", retention=retention)
        jax.block_until_ready(solution.values)
        del solution
    return observed, dispatched, dispatched_keys, wave_fanout


def _observe_gpu_lower(
    *,
    copy_lower: Any,
    observed: list[Any],
    wave_fanout: dict[str, int],
    lowered_ids: dict[str, int],
    **kwargs: Any,
) -> Any:
    count = len(observed)
    result = copy_lower(**kwargs)
    if len(observed) != count:
        manifest, ir = observed[-1]
        fanout = manifest["dedup_fanout"]
        assert type(fanout) is int
        assert fanout > 0
        wave_fanout[manifest["dedup_key"]] = fanout
        lowered_ids[manifest["dedup_key"]] = id(result)
        # Public contract: actual wave count is unavailable precompile under
        # a configured budget. Keep the real observed integer separately.
        observed[-1] = (MappingProxyType({**manifest, "dedup_fanout": None}), ir)
    return result


def _observe_gpu_dispatch(
    *,
    dispatch: Any,
    dispatched: set[Any],
    dispatched_keys: dict[Any, str],
    compiled_keys: dict[int, str],
    **kwargs: Any,
) -> Any:
    for core_name, core in kwargs["compiled_cores"].items():
        assert isinstance(core.compiled, jax.stages.Compiled)
        shardings = jax.tree.leaves(
            (core.compiled.input_shardings, core.compiled.output_shardings)
        )
        assert any(
            isinstance(sharding, jax.sharding.Sharding)
            and any(device.platform == "gpu" for device in sharding.device_set)
            for sharding in shardings
        ), "The genuine dispatched executable must have GPU placement"
        address = (
            kwargs["regime_name"],
            kwargs["period"],
            core_name,
            tuple(sorted(core.tile_widths.items())),
            core.donated_arguments,
        )
        dispatched_keys[address] = compiled_keys[id(core.compiled)]
    return _observe_dispatch(dispatch=dispatch, dispatched=dispatched, **kwargs)


def _sha256(path: Path) -> str:
    """Stream large plugin/CUDA binaries without retaining their bytes."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _capture_gpu_source_runtime_identity() -> Mapping[str, Any]:
    """Proposed CUDA12 source-checkout identity, independent of production code."""
    root = Path(hatch_build.__file__).resolve().parent
    sources = tuple(
        (path.relative_to(root).as_posix(), _sha256(path=path))
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
    # Existing CI probe checks complete build inputs and registration first.
    assert manifest["inputs"] == hatch_build.native_build_inputs(root=root)
    assert "libcertified_affine_ffi_cuda.so" in manifest["libraries"]
    libraries = []
    for name in manifest["libraries"]:
        assert Path(name).name == name
        libraries.append((name, _sha256(path=directory / name)))
    assert libraries
    native_headers = tuple(
        (path.relative_to(root).as_posix(), _sha256(path=path))
        for path in sorted((root / hatch_build.PACKAGE_DIR).glob("*.h"))
    )
    assert native_headers
    include_root = Path(jax.ffi.include_dir())
    ffi_headers = tuple(
        (path.relative_to(include_root).as_posix(), _sha256(path=path))
        for path in sorted(include_root.rglob("*.h"))
    )
    assert ffi_headers
    native_tools = tuple(
        (name, _sha256(path=Path(manifest["inputs"][name]).resolve(strict=True)))
        for name in ("compiler", "nvcc")
    )
    runtime = []
    for name in ("jax", "jaxlib", "numpy"):
        package = distribution(name)
        assert package.files
        files = tuple(
            (str(path), _sha256(path=Path(str(package.locate_file(path)))))
            for path in sorted(package.files, key=str)
            if Path(str(path)).suffix in (".py", ".so", ".pyd", ".dll", ".dylib")
        )
        assert files
        runtime.append((name, files))
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
    assert len(devices) == 1
    assert devices[0][2] == "gpu"
    assert tuple(jax.devices()) == tuple(jax.devices("cuda"))
    assert jax.devices("cpu"), "The ordinary retention backend must be available"
    gpu_identity = _capture_cuda_identity()
    return MappingProxyType(
        {
            "source_files": sources,
            "source_identity": _semantic_fingerprint(sources),
            "native_source_identity": native_source,
            "native_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "native_build_identity": manifest["fingerprint"],
            "native_library_bytes": tuple(libraries),
            "runtime_files": tuple(runtime),
            "runtime_identity": _semantic_fingerprint(tuple(runtime)),
            "python_binary_sha256": _sha256(path=Path(sys.executable)),
            "python_abi": sys.implementation.cache_tag,
            "devices": devices,
            "xla_flags": os.environ.get("XLA_FLAGS", ""),
            "gpu_identity": gpu_identity,
            "native_header_files": native_headers,
            "jax_ffi_header_files": ffi_headers,
            "native_tool_files": native_tools,
        }
    )


def _capture_cuda_identity() -> Mapping[str, Any]:
    """Bind plugin/CUDA package bytes and actual loaded driver/device identity."""
    packages = {}
    for package in distributions():
        name = package.metadata["Name"].lower().replace("_", "-")
        if name.startswith(("jax-cuda", "nvidia-")):
            assert name not in packages
            assert package.files
            # Include versioned .so files, libdevice/PTX/resources and metadata,
            # not just unversioned shared libraries or a package version string.
            files = tuple(
                (str(path), _sha256(path=Path(str(package.locate_file(path)))))
                for path in sorted(package.files, key=str)
                if Path(str(path)).suffix != ".pyc"
                and "__pycache__" not in Path(str(path)).parts
            )
            assert files
            packages[name] = files
    assert {"jax-cuda12-plugin", "jax-cuda12-pjrt"} <= packages.keys()
    assert not any(name.startswith("jax-cuda13") for name in packages)
    package_bytes = tuple(sorted(packages.items()))
    driver_paths = set()
    for row in Path("/proc/self/maps").read_text().splitlines():
        fields = row.split(maxsplit=5)
        if len(fields) != 6:
            continue
        path = Path(fields[5])
        if path.name.startswith(("libcuda.so", "libnvidia-")):
            assert path.is_absolute()
            assert path.is_file()
            driver_paths.add(path.resolve(strict=True))
    assert any(path.name.startswith("libcuda.so") for path in driver_paths)
    driver_files = tuple(
        (str(path), _sha256(path=path)) for path in sorted(driver_paths)
    )
    nvidia_smi = shutil.which("nvidia-smi")
    assert nvidia_smi is not None
    query = subprocess.run(  # noqa: S603 - fixed query to resolved installed tool
        [
            nvidia_smi,
            "--query-gpu=uuid,pci.bus_id,name,compute_cap,driver_version",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    rows = tuple(
        tuple(item.strip() for item in row)
        for row in csv.reader(query.stdout.splitlines())
    )
    assert len(rows) == 1
    assert len(rows[0]) == 5
    assert all(rows[0])
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    assert visible in (None, "0", rows[0][0])
    assert jax.devices()[0].local_hardware_id == 0
    return MappingProxyType(
        {
            "profile": "cuda12-source-checkout-v1",
            "package_files": package_bytes,
            "package_identity": _semantic_fingerprint(package_bytes),
            "loaded_driver_files": driver_files,
            "driver_kernel_version": Path("/proc/driver/nvidia/version").read_text(),
            "physical_devices": rows,
            "cuda_visible_devices": visible,
            "precision": 64 if jax.config.x64_enabled else 32,
            "compilation_cache_enabled": jax.config.jax_enable_compilation_cache,
        }
    )
