"""Fast structural tests for the external paired GridSearch harness."""

import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import pytest

from benchmarks.grid_search_pair import (
    _assert_pair_identity,
    _assert_scenario_execution_target,
    _compare_array_manifests,
    _compare_measurement_identity,
    _compare_value_artifacts,
    _max_ulp_distance,
    _metric_summary,
    _pair_summary,
    _parser,
    _pixi_context,
    _sha256_files,
    _validate_checkout,
    _validate_required_evidence,
    _validate_worker_environment,
    _worker_command,
    _worker_env,
)
from benchmarks.grid_search_pair_scenarios import SCENARIOS, TARGET_SCENARIO_SOURCES
from benchmarks.grid_search_pair_worker import (
    _VERSION_SHIM_VERSION,
    _block_result,
    _executed_float_dtype,
    _grid_extent,
    _hlo_census,
)


def _write_npz(path: Path, **arrays: Any) -> None:
    np.savez(path, **arrays)


def _run_git(checkout: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(checkout), *arguments],
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


def _kernel(
    *,
    regime: str,
    execution_disposition: str = "legacy-unplanned",
    disposition_reason: str | None = "legacy_adapter",
    collective: bool = False,
    has_taste_shocks: bool = False,
    fold_state_names: tuple[str, ...] = (),
    action_extent: int = 10,
) -> dict[str, Any]:
    return {
        "regime": regime,
        "period": 0,
        "action_names": ["consumption"],
        "action_extents": [action_extent],
        "streamed": execution_disposition == "planned",
        "execution_disposition": execution_disposition,
        "disposition_reason": disposition_reason,
        "collective": collective,
        "has_taste_shocks": has_taste_shocks,
        "fold_state_names": list(fold_state_names),
    }


def _set_execution(
    *, row: dict[str, Any], disposition: str, reason: str | None
) -> None:
    row["streamed"] = disposition == "planned"
    row["execution_disposition"] = disposition
    row["disposition_reason"] = reason


def _routes(
    *,
    kernels: list[dict[str, Any]],
    folded_regimes: tuple[str, ...] = (),
    collective_regimes: tuple[str, ...] = (),
    distributed_regimes: tuple[str, ...] = (),
    taste_shock_regimes: tuple[str, ...] = (),
    gs_vd_regimes: tuple[str, ...] = (),
) -> dict[str, Any]:
    return {
        "kernels": kernels,
        "streamed_kernel_count": sum(row["streamed"] for row in kernels),
        "folded_regimes": list(folded_regimes),
        "collective_regimes": list(collective_regimes),
        "distributed_regimes": list(distributed_regimes),
        "taste_shock_regimes": list(taste_shock_regimes),
        "gs_vd_regimes": list(gs_vd_regimes),
    }


def _measurement_identity_fixture(*, routes: dict[str, Any]) -> dict[str, Any]:
    return {
        "dimensions": {
            "n_periods": 2,
            "regimes": {
                "working": {
                    "active_periods": [0, 1],
                    "solution_grid_extents": {"wealth": 20, "consumption": 10},
                }
            },
        },
        "routes": routes,
        "python": "3.14.0",
        "jax_version": "0.9.0",
        "jaxlib_version": "0.9.0",
        "jax_enable_x64": False,
        "version_shim": {
            "module": "_lcm.version",
            "origin": "<pylcm-grid-search-pair-version-shim>",
            "exports": [
                "__version__",
                "__version_tuple__",
                "version",
                "version_tuple",
                "__commit_id__",
                "commit_id",
            ],
            "version": _VERSION_SHIM_VERSION,
            "version_tuple": [0, "gridsearchpair"],
            "commit_id": None,
            "sha256": "stable-shim-digest",
        },
        "devices": [{"id": 0, "platform": "cpu", "kind": "cpu"}],
        "environment": {
            "JAX_COMPILATION_CACHE_DIR": "/cache/base",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "XLA_PYTHON_CLIENT_MEM_FRACTION": None,
            "XLA_FLAGS": "--xla_gpu_autotune_level=0",
        },
    }


def test_harness_digest_is_independent_of_checkout_location(tmp_path: Path) -> None:
    relative_paths = ("benchmarks/harness.py", "benchmarks/scenario.py")
    roots = (tmp_path / "local", tmp_path / "marvin")
    for root in roots:
        for relative in relative_paths:
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"stable content for {relative}\n")

    local = _sha256_files(root=roots[0], relative_paths=relative_paths)
    marvin = _sha256_files(root=roots[1], relative_paths=reversed(relative_paths))

    assert local == marvin
    (roots[1] / relative_paths[0]).write_text("changed\n")
    assert local != _sha256_files(root=roots[1], relative_paths=relative_paths)


def test_checkout_identity_rejects_revision_drift_and_dirty_sources(
    tmp_path: Path,
) -> None:
    checkout = tmp_path / "harness"
    for relative in ("pixi.lock", *TARGET_SCENARIO_SOURCES):
        path = checkout / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"tracked {relative}\n")
    _run_git(checkout, "init", "--quiet")
    _run_git(checkout, "add", ".")
    _run_git(
        checkout,
        "-c",
        "user.name=Harness Test",
        "-c",
        "user.email=harness@example.invalid",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "--quiet",
        "-m",
        "initial",
    )
    revision = _run_git(checkout, "rev-parse", "HEAD")

    identity = _validate_checkout(
        checkout=checkout,
        expected_revision=revision,
    )
    assert identity["revision"] == revision
    with pytest.raises(RuntimeError, match="expected"):
        _validate_checkout(checkout=checkout, expected_revision="0" * 40)

    (checkout / "pixi.lock").write_text("dirty\n")
    with pytest.raises(RuntimeError, match="dirty"):
        _validate_checkout(checkout=checkout, expected_revision=revision)


def test_parser_pins_external_harness_and_selects_requested_rows(
    tmp_path: Path,
) -> None:
    args = _parser().parse_args(
        [
            "--base-checkout",
            str(tmp_path / "base"),
            "--base-revision",
            "0" * 40,
            "--head-checkout",
            str(tmp_path / "head"),
            "--head-revision",
            "1" * 40,
            "--harness-revision",
            "2" * 40,
            "--scenario",
            "aca-a3-c16",
            "--scenario",
            "aca-a6-c256",
            "--output",
            str(tmp_path / "output"),
        ]
    )

    assert args.harness_revision == "2" * 40
    assert args.scenarios == ["aca-a3-c16", "aca-a6-c256"]


def test_scenario_registry_contains_closure_and_aca_frontier_rows() -> None:
    assert tuple(SCENARIOS) == (
        "singleton-hard-max",
        "singleton-ev1",
        "collective-gs-vd",
        "distributed-co-map",
        "folded-hard-max",
        "aca-a3-c16",
        "aca-a3-c64",
        "aca-a3-c256",
        "aca-a6-c16",
        "aca-a6-c64",
        "aca-a6-c256",
    )
    assert SCENARIOS["distributed-co-map"].topology == "cpu-4"
    assert SCENARIOS["distributed-co-map"].expected_distributed
    assert SCENARIOS["folded-hard-max"].expected_folded
    assert SCENARIOS["collective-gs-vd"].expected_collective
    assert SCENARIOS["singleton-hard-max"].expected_head_disposition == "planned"
    assert SCENARIOS["distributed-co-map"].expected_head_disposition == "planned"
    assert SCENARIOS["folded-hard-max"].expected_head_disposition == "planned"
    assert SCENARIOS["singleton-ev1"].expected_head_disposition == "dense"
    assert (
        SCENARIOS["singleton-ev1"].expected_head_disposition_reason
        == "deliberately_dense:ev1_canonical_reduction_order"
    )
    assert SCENARIOS["collective-gs-vd"].expected_head_disposition == "dense"
    assert (
        SCENARIOS["collective-gs-vd"].expected_head_disposition_reason
        == "deliberately_dense:collective_resource_regression"
    )
    assert {
        name: (spec.aca_assets_n_points, spec.aca_consumption_n_points)
        for name, spec in SCENARIOS.items()
        if name.startswith("aca-")
    } == {
        "aca-a3-c16": (3, 16),
        "aca-a3-c64": (3, 64),
        "aca-a3-c256": (3, 256),
        "aca-a6-c16": (6, 16),
        "aca-a6-c64": (6, 64),
        "aca-a6-c256": (6, 256),
    }


@pytest.mark.parametrize(
    ("scenario", "target", "decoy", "route_kwargs"),
    [
        (
            "singleton-hard-max",
            _kernel(regime="working"),
            _kernel(regime="ev1", has_taste_shocks=True),
            {},
        ),
        (
            "singleton-ev1",
            _kernel(regime="ev1", has_taste_shocks=True),
            _kernel(regime="ordinary"),
            {"taste_shock_regimes": ("ev1",)},
        ),
        (
            "collective-gs-vd",
            _kernel(regime="household", collective=True),
            _kernel(regime="ordinary"),
            {
                "collective_regimes": ("household",),
                "gs_vd_regimes": ("household",),
            },
        ),
        (
            "distributed-co-map",
            _kernel(regime="distributed"),
            _kernel(regime="ordinary"),
            {"distributed_regimes": ("distributed",)},
        ),
        (
            "folded-hard-max",
            _kernel(regime="folded", fold_state_names=("shock",)),
            _kernel(regime="ordinary"),
            {"folded_regimes": ("folded",)},
        ),
        (
            "aca-a3-c16",
            _kernel(regime="retiree"),
            _kernel(regime="terminal", action_extent=1),
            {},
        ),
    ],
)
def test_execution_disposition_must_cover_every_named_nontrivial_target_kernel(
    *,
    scenario: str,
    target: dict[str, Any],
    decoy: dict[str, Any],
    route_kwargs: dict[str, Any],
) -> None:
    second_target = deepcopy(target)
    second_target["period"] = 1
    base = _routes(
        kernels=[deepcopy(target), second_target, deepcopy(decoy)],
        **route_kwargs,
    )
    head = deepcopy(base)
    _set_execution(row=head["kernels"][2], disposition="planned", reason=None)
    head["streamed_kernel_count"] = sum(row["streamed"] for row in head["kernels"])

    with pytest.raises(RuntimeError, match="required execution disposition"):
        _assert_scenario_execution_target(
            scenario=scenario,
            base_routes=base,
            head_routes=head,
        )

    spec = SCENARIOS[scenario]
    _set_execution(
        row=head["kernels"][0],
        disposition=spec.expected_head_disposition,
        reason=spec.expected_head_disposition_reason,
    )
    head["streamed_kernel_count"] = sum(row["streamed"] for row in head["kernels"])
    with pytest.raises(RuntimeError, match="required execution disposition"):
        _assert_scenario_execution_target(
            scenario=scenario,
            base_routes=base,
            head_routes=head,
        )

    _set_execution(
        row=head["kernels"][1],
        disposition=spec.expected_head_disposition,
        reason=spec.expected_head_disposition_reason,
    )
    head["streamed_kernel_count"] = sum(row["streamed"] for row in head["kernels"])
    _assert_scenario_execution_target(
        scenario=scenario,
        base_routes=base,
        head_routes=head,
    )


def test_measurement_identity_allows_only_streaming_and_cache_path_differences() -> (
    None
):
    base_routes = _routes(kernels=[_kernel(regime="working")])
    head_routes = deepcopy(base_routes)
    _set_execution(row=head_routes["kernels"][0], disposition="planned", reason=None)
    head_routes["streamed_kernel_count"] = 1
    base = _measurement_identity_fixture(routes=base_routes)
    head = _measurement_identity_fixture(routes=head_routes)
    head["environment"]["JAX_COMPILATION_CACHE_DIR"] = "/cache/head"

    shared = _compare_measurement_identity(base=base, head=head)
    assert shared["dimensions"] == base["dimensions"]

    drift = deepcopy(head)
    drift["dimensions"]["n_periods"] = 3
    with pytest.raises(RuntimeError, match="dimensions differs"):
        _compare_measurement_identity(base=base, head=drift)

    drift = deepcopy(head)
    drift["routes"]["kernels"][0]["action_extents"] = [11]
    with pytest.raises(RuntimeError, match="routes differs"):
        _compare_measurement_identity(base=base, head=drift)

    drift = deepcopy(head)
    drift["jaxlib_version"] = "different"
    with pytest.raises(RuntimeError, match="runtime differs"):
        _compare_measurement_identity(base=base, head=drift)

    drift = deepcopy(head)
    drift["version_shim"]["sha256"] = "different"
    with pytest.raises(RuntimeError, match="runtime differs"):
        _compare_measurement_identity(base=base, head=drift)

    drift = deepcopy(head)
    drift["devices"][0]["kind"] = "different"
    with pytest.raises(RuntimeError, match="devices differs"):
        _compare_measurement_identity(base=base, head=drift)

    drift = deepcopy(head)
    drift["environment"]["XLA_FLAGS"] = "different"
    with pytest.raises(RuntimeError, match="behavioral_environment differs"):
        _compare_measurement_identity(base=base, head=drift)


def test_grid_extent_reads_declared_width_without_materializing_points() -> None:
    class RuntimeGrid:
        n_points = 16

        def to_jax(self) -> None:
            raise AssertionError("runtime points must not be materialized")

    assert _grid_extent(RuntimeGrid()) == 16


class _FakeJaxConfig:
    def __init__(self, *, enable_x64: bool) -> None:
        self._enable_x64 = enable_x64

    def read(self, name: str) -> bool:
        assert name == "jax_enable_x64"
        return self._enable_x64


class _FakeJax:
    def __init__(self, *, enable_x64: bool) -> None:
        self.config = _FakeJaxConfig(enable_x64=enable_x64)


@pytest.mark.parametrize(
    ("precision", "enable_x64", "expected"),
    [("32", False, "float32"), ("64", True, "float64")],
)
def test_executed_float_dtype_reports_the_dtype_the_run_used(
    *, precision: str, enable_x64: bool, expected: str
) -> None:
    arrays = {
        "value/0/working": np.zeros(2, dtype=expected),
        "dissolution/0/couple": np.zeros(2, dtype=np.int32),
    }
    assert (
        _executed_float_dtype(
            precision=precision, jax=_FakeJax(enable_x64=enable_x64), arrays=arrays
        )
        == expected
    )


def test_executed_float_dtype_rejects_a_workload_that_reconfigured_x64() -> None:
    with pytest.raises(RuntimeError, match="jax_enable_x64=True"):
        _executed_float_dtype(precision="32", jax=_FakeJax(enable_x64=True), arrays={})


def test_executed_float_dtype_rejects_published_arrays_of_another_precision() -> None:
    arrays = {"value/0/working": np.zeros(2, dtype=np.float64)}
    with pytest.raises(RuntimeError, match="float64"):
        _executed_float_dtype(
            precision="32", jax=_FakeJax(enable_x64=False), arrays=arrays
        )


def test_block_result_descends_mapping_proxy_trees() -> None:
    class PendingLeaf:
        def __init__(self) -> None:
            self.calls = 0

        def block_until_ready(self) -> None:
            self.calls += 1

    value = PendingLeaf()
    dissolution = PendingLeaf()
    result = (
        MappingProxyType({0: MappingProxyType({"working": value})}),
        MappingProxyType({0: MappingProxyType({"working": dissolution})}),
    )

    _block_result(result)

    assert value.calls == 1
    assert dissolution.calls == 1


def test_block_result_projects_solution_result_dissolution_artifacts() -> None:
    class PendingLeaf:
        def __init__(self) -> None:
            self.calls = 0

        def block_until_ready(self) -> None:
            self.calls += 1

    class ArtifactKey:
        type_id = "pylcm.collective.dissolution_flag"

    class ArtifactRef:
        key = ArtifactKey()
        period = 0
        regime = "working"

    class SolutionResult:
        values = MappingProxyType({0: MappingProxyType({"working": PendingLeaf()})})
        replay_artifacts = MappingProxyType({ArtifactRef(): PendingLeaf()})

    result = SolutionResult()
    _block_result(result)

    assert result.values[0]["working"].calls == 1
    assert next(iter(result.replay_artifacts.values())).calls == 1


def test_closure_scenarios_construct_with_their_declared_topology(
    tmp_path: Path,
) -> None:
    code = """
import sys
from math import prod
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", False)
jax.config.update("jax_num_cpu_devices", 4)
jax.config.update("jax_platform_name", "cpu")

from benchmarks.grid_search_pair_scenarios import SCENARIOS, build_scenario
from benchmarks.grid_search_pair_worker import (
    _assert_scenario_identity,
    _import_lcm_with_version_shim,
    _route_metadata,
    _scenario_dimensions,
)

target_src = (Path.cwd() / "src").resolve()
sys.path.insert(0, str(target_src))
lcm, version_shim = _import_lcm_with_version_shim(target_src=target_src)
assert lcm.__version__ == version_shim["version"]

for name, spec in SCENARIOS.items():
    if name.startswith("aca-"):
        continue
    model, params = build_scenario(name=name)
    assert isinstance(params, dict)
    routes = _route_metadata(model)
    _assert_scenario_identity(spec=spec, routes=routes)
    nontrivial = [
        row
        for row in routes["kernels"]
        if row["action_names"] and prod(row["action_extents"]) > 1
    ]
    assert any(
        row["execution_disposition"] == spec.expected_head_disposition
        and row["disposition_reason"] == spec.expected_head_disposition_reason
        for row in nontrivial
    ), (name, routes)
    dimensions = _scenario_dimensions(model)
    assert dimensions["n_periods"] > 1
    assert dimensions["regimes"]
    assert any(
        regime["solution_grid_extents"]
        for regime in dimensions["regimes"].values()
    )
    for regime in dimensions["regimes"].values():
        assert regime["active_periods"]
        assert "solution_grid_extents" in regime
        assert all(extent > 0 for extent in regime["solution_grid_extents"].values())

assert len(jax.devices()) == 4
assert {device.platform for device in jax.devices()} == {"cpu"}
"""
    repo_root = Path(__file__).resolve().parent.parent
    env = _worker_env(
        harness_root=repo_root,
        checkout=repo_root,
        cache_dir=tmp_path / "cache",
    )

    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        cwd=repo_root,
        env=env,
        timeout=180,
    )

    assert completed.returncode == 0, (
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )


@pytest.mark.parametrize("generated_version", [False, True])
def test_version_shim_handles_missing_or_generated_metadata(
    *, tmp_path: Path, generated_version: bool
) -> None:
    target_src = tmp_path / "target" / "src"
    (target_src / "lcm").mkdir(parents=True)
    (target_src / "_lcm").mkdir()
    (target_src / "lcm" / "__init__.py").write_text(
        "from _lcm.version import __version__\n"
    )
    (target_src / "_lcm" / "__init__.py").write_text("")
    if generated_version:
        (target_src / "_lcm" / "version.py").write_text(
            'raise AssertionError("generated target metadata was imported")\n'
        )
    code = """\
import sys
from pathlib import Path

from benchmarks.grid_search_pair_worker import (
    _VERSION_SHIM_VERSION,
    _import_lcm_with_version_shim,
)

target_src = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(target_src))
lcm, identity = _import_lcm_with_version_shim(target_src=target_src)
import _lcm.version as version_module
import _lcm

assert lcm.__version__ == _VERSION_SHIM_VERSION
assert version_module.__version__ == _VERSION_SHIM_VERSION
assert version_module.version == _VERSION_SHIM_VERSION
assert version_module.__version_tuple__ == (0, "gridsearchpair")
assert version_module.version_tuple == (0, "gridsearchpair")
assert version_module.__commit_id__ is None
assert version_module.commit_id is None
assert version_module.__file__ == "<pylcm-grid-search-pair-version-shim>"
assert _lcm.version is version_module
assert identity["version"] == _VERSION_SHIM_VERSION
assert len(identity["sha256"]) == 64
"""
    repo_root = Path(__file__).resolve().parent.parent
    completed = subprocess.run(
        [sys.executable, "-c", code, str(target_src)],
        capture_output=True,
        text=True,
        check=False,
        cwd=repo_root,
        env=_worker_env(
            harness_root=repo_root,
            checkout=repo_root,
            cache_dir=tmp_path / "cache",
        ),
        timeout=60,
    )

    assert completed.returncode == 0, (
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )


def test_worker_environment_selects_external_harness_and_fresh_cache(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PYTHONPATH", "/already/there")
    monkeypatch.setenv(
        "XLA_FLAGS",
        "--some-existing-flag --xla_gpu_autotune_level=4",
    )
    monkeypatch.setenv("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")
    harness = tmp_path / "harness"
    checkout = tmp_path / "checkout"
    cache = tmp_path / "cache"

    env = _worker_env(
        harness_root=harness,
        checkout=checkout,
        cache_dir=cache,
    )

    assert env["PYTHONPATH"].split(os.pathsep) == [
        str(harness),
        str(checkout / "src"),
        "/already/there",
    ]
    assert env["JAX_COMPILATION_CACHE_DIR"] == str(cache)
    assert env["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
    assert "XLA_PYTHON_CLIENT_MEM_FRACTION" not in env
    assert env["XLA_FLAGS"].split().count("--xla_gpu_autotune_level=0") == 1
    assert "--xla_gpu_autotune_level=4" not in env["XLA_FLAGS"].split()
    assert "--some-existing-flag" in env["XLA_FLAGS"]


def test_worker_command_is_pinned_to_the_active_pixi_environment(
    tmp_path: Path,
) -> None:
    pixi_exe = tmp_path / "pixi"
    pixi_exe.write_text("")
    harness = tmp_path / "project"
    harness.mkdir()
    context = _pixi_context(
        harness_root=harness,
        environment={
            "PIXI_ENVIRONMENT_NAME": "benchmarks-cuda12",
            "PIXI_EXE": str(pixi_exe),
            "PIXI_PROJECT_ROOT": str(harness),
        },
    )

    command = _worker_command(pixi=context, arguments=["--scenario", "row"])

    assert command == [
        str(pixi_exe),
        "run",
        "--frozen",
        "-e",
        "benchmarks-cuda12",
        "python",
        "-m",
        "benchmarks.grid_search_pair_worker",
        "--scenario",
        "row",
    ]


def test_pixi_context_rejects_missing_or_wrong_project(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="missing"):
        _pixi_context(harness_root=tmp_path, environment={})
    pixi_exe = tmp_path / "pixi"
    pixi_exe.write_text("")
    with pytest.raises(RuntimeError, match="Active pixi project"):
        _pixi_context(
            harness_root=tmp_path,
            environment={
                "PIXI_ENVIRONMENT_NAME": "tests-cpu",
                "PIXI_EXE": str(pixi_exe),
                "PIXI_PROJECT_ROOT": str(tmp_path / "other"),
            },
        )


def test_hlo_census_records_size_instructions_and_collectives() -> None:
    hlo = """HloModule test
ENTRY main {
  %x = f32[4] parameter(0)
  %loop = f32[4] while(%x), condition=cond, body=body
  ROOT %out = f32[4] all-reduce(%loop), replica_groups={{0,1}}
}
"""

    census = _hlo_census(hlo)

    assert census["text_bytes"] == len(hlo.encode())
    assert census["line_count"] == len(hlo.splitlines())
    assert census["instruction_count"] == 3
    assert census["op_counts"]["while"] == 1
    assert census["op_counts"]["all-reduce"] == 1
    assert census["communication_collective_count"] == 1
    assert len(census["sha256"]) == 64


def test_hlo_census_counts_async_communication_once_per_pair() -> None:
    hlo = """HloModule async
ENTRY main {
  %x = f32[4] parameter(0)
  %ar-start = f32[4] all-reduce-start(%x)
  %ar-done = f32[4] all-reduce-done(%ar-start)
  %cp-start = f32[4] collective-permute-start(%ar-done)
  %cp-done = f32[4] collective-permute-done(%cp-start)
  %cb-start = f32[4] collective-broadcast-start(%cp-done)
  %cb-done = f32[4] collective-broadcast-done(%cb-start)
  %ragged-start = f32[4] ragged-all-to-all-start(%cb-done)
  %ragged-done = f32[4] ragged-all-to-all-done(%ragged-start)
  ROOT %out = f32[4] all-gather(%ragged-done)
}
"""

    census = _hlo_census(hlo)

    assert census["op_counts"]["all-reduce"] == 1
    assert census["op_counts"]["collective-permute"] == 1
    assert census["op_counts"]["collective-broadcast"] == 1
    assert census["op_counts"]["ragged-all-to-all"] == 1
    assert census["op_counts"]["all-gather"] == 1
    assert census["communication_collective_count"] == 5


def test_required_evidence_distinguishes_cpu_na_from_gpu_peak() -> None:
    core = {
        "label": "working/0",
        "hlo": {"text_bytes": 10, "instruction_count": 1},
        "compiler_memory": None,
        "compiler_memory_status": "unavailable",
        "compiler_memory_reason": "not supported",
    }
    cpu_device = {"id": 0, "platform": "cpu", "kind": "cpu"}
    cpu_memory = {
        **cpu_device,
        "peak_bytes_in_use": None,
        "status": "not_applicable",
        "reason": "Device peak memory is only defined for GPU rows.",
    }
    metrics: dict[str, Any] = {
        "compiled_cores": [core],
        "devices": [cpu_device],
        "memory": {
            "through_warm": {"hwm_bytes": 1_000},
            "device": [cpu_memory],
        },
    }
    _validate_required_evidence(metrics)

    missing_hlo = deepcopy(metrics)
    missing_hlo["compiled_cores"] = []
    with pytest.raises(RuntimeError, match="no compiled-core"):
        _validate_required_evidence(missing_hlo)

    missing_hwm = deepcopy(metrics)
    missing_hwm["memory"]["through_warm"]["hwm_bytes"] = None
    with pytest.raises(RuntimeError, match="no host VmHWM"):
        _validate_required_evidence(missing_hwm)

    gpu = deepcopy(metrics)
    gpu["devices"][0] = {"id": 0, "platform": "gpu", "kind": "cuda"}
    gpu["memory"]["device"][0] = {
        **gpu["devices"][0],
        "peak_bytes_in_use": 2_000,
        "status": "measured",
        "reason": None,
    }
    _validate_required_evidence(gpu)
    gpu["memory"]["device"][0]["peak_bytes_in_use"] = None
    gpu["memory"]["device"][0]["status"] = "unavailable"
    gpu["memory"]["device"][0]["reason"] = "missing"
    with pytest.raises(RuntimeError, match="no measured peak-memory"):
        _validate_required_evidence(gpu)


def test_worker_environment_requires_its_exact_fresh_cache(tmp_path: Path) -> None:
    cache = tmp_path / "fresh-cache"
    metrics: dict[str, Any] = {
        "environment": {
            "JAX_COMPILATION_CACHE_DIR": str(cache),
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "XLA_PYTHON_CLIENT_MEM_FRACTION": None,
            "XLA_FLAGS": "--xla_gpu_autotune_level=0",
        }
    }
    _validate_worker_environment(metrics=metrics, expected_cache_dir=cache)

    wrong = deepcopy(metrics)
    wrong["environment"]["JAX_COMPILATION_CACHE_DIR"] = str(tmp_path / "shared")
    with pytest.raises(RuntimeError, match="compilation cache"):
        _validate_worker_environment(metrics=wrong, expected_cache_dir=cache)

    wrong = deepcopy(metrics)
    wrong["environment"]["XLA_FLAGS"] += " --xla_gpu_autotune_level=0"
    with pytest.raises(RuntimeError, match="exactly once"):
        _validate_worker_environment(metrics=wrong, expected_cache_dir=cache)


def test_value_parity_records_bitwise_and_tolerated_float_agreement(
    tmp_path: Path,
) -> None:
    base = tmp_path / "base.npz"
    head = tmp_path / "head.npz"
    _write_npz(
        base,
        value=np.array([1.0, 2.0], dtype=np.float64),
        flag=np.array([True, False]),
    )
    _write_npz(
        head,
        value=np.array([1.0 + 1e-13, 2.0], dtype=np.float64),
        flag=np.array([True, False]),
    )

    result = _compare_value_artifacts(base_path=base, head_path=head)

    assert result["all_passed"]
    assert not result["all_bitwise_equal"]
    by_key = {record["key"]: record for record in result["arrays"]}
    assert by_key["flag"]["bitwise_equal"]
    assert by_key["value"]["parity"]
    assert by_key["value"]["max_abs"] > 0
    assert by_key["value"]["max_ulp"] > 0


def test_array_manifest_requires_exact_shape_dtype_and_sharding() -> None:
    single = {
        "kind": "SingleDeviceSharding",
        "platform": "cpu",
        "device_id": 0,
        "memory_kind": "device",
    }
    base = [{"key": "value/0/r", "shape": [2], "dtype": "float64", "sharding": single}]
    result = _compare_array_manifests(base=base, head=base)
    assert result["all_passed"]

    changed = [
        {
            **base[0],
            "sharding": {**single, "device_id": 1},
        }
    ]
    with pytest.raises(AssertionError, match="sharding differs"):
        _compare_array_manifests(base=base, head=changed)


@pytest.mark.parametrize("dtype", [np.dtype("float32"), np.dtype("float64")])
def test_max_ulp_distance_uses_exact_ordered_ieee_bits(dtype: np.dtype) -> None:
    expected = np.array([-1.0, -0.0, 0.0, 1.0], dtype=dtype)
    actual = expected.copy()
    actual[-1] = np.nextafter(actual[-1], dtype.type(2.0))
    finite = np.ones(expected.shape, dtype=bool)

    assert (
        _max_ulp_distance(
            expected=expected,
            actual=actual,
            finite=finite,
        )
        == 1
    )


@pytest.mark.parametrize(
    ("head_array", "match"),
    [
        (np.array([1.0, 2.0, 3.0]), "shape differs"),
        (np.array([1.0, 2.0], dtype=np.float32), "dtype differs"),
        (np.array([1.0, 2.1]), "violates parity"),
    ],
)
def test_value_parity_fails_closed_on_structural_or_numerical_drift(
    *, tmp_path: Path, head_array: np.ndarray, match: str
) -> None:
    base = tmp_path / "base.npz"
    head = tmp_path / "head.npz"
    _write_npz(base, value=np.array([1.0, 2.0]))
    _write_npz(head, value=head_array)

    with pytest.raises(AssertionError, match=match):
        _compare_value_artifacts(base_path=base, head_path=head)


def test_pair_identity_requires_distinct_revisions_and_equal_sources() -> None:
    base = {
        "revision": "a" * 40,
        "lock_digest": "lock",
        "scenario_sources": {"model.py": "same"},
    }
    head = {
        "revision": "b" * 40,
        "lock_digest": "lock",
        "scenario_sources": {"model.py": "same"},
    }
    _assert_pair_identity(base=base, head=head)

    with pytest.raises(RuntimeError, match="must differ"):
        _assert_pair_identity(base=base, head={**head, "revision": "a" * 40})
    with pytest.raises(RuntimeError, match=r"pixi\.lock"):
        _assert_pair_identity(base=base, head={**head, "lock_digest": "other"})
    with pytest.raises(RuntimeError, match="scenario dependencies"):
        _assert_pair_identity(
            base=base,
            head={**head, "scenario_sources": {"model.py": "other"}},
        )


def test_metric_summary_preserves_raw_samples_and_uses_per_core_maxima() -> None:
    metrics = {
        "timing_ns": {
            "cold_solve": 100,
            "aot_compile_calls": [80, 5, 6, 7],
            "warm_solve": [30, 20, 40],
        },
        "memory": {
            "through_warm": {"hwm_bytes": 1_000},
            "device": [
                {"peak_bytes_in_use": 300},
                {"peak_bytes_in_use": 500},
            ],
        },
        "compiled_cores": [
            {
                "compiler_memory": {
                    "peak_memory_in_bytes": 700,
                    "temp_size_in_bytes": 600,
                },
                "hlo": {
                    "text_bytes": 11,
                    "instruction_count": 3,
                    "communication_collective_count": 0,
                },
            },
            {
                "compiler_memory": {
                    "peak_memory_in_bytes": 900,
                    "temp_size_in_bytes": 400,
                },
                "hlo": {
                    "text_bytes": 13,
                    "instruction_count": 4,
                    "communication_collective_count": 1,
                },
            },
        ],
        "routes": {"streamed_kernel_count": 2},
        "tile_plans": [{"tile_widths": {"action": 64}}],
    }

    result = _metric_summary(metrics)

    assert result["cold_aot_compile_ns"] == 80
    assert result["warm_solve_ns"] == [30, 20, 40]
    assert result["warm_solve_median_ns"] == 30
    assert result["compiler_peak_bytes"] == 900
    assert result["compiler_temp_bytes"] == 600
    assert result["device_peak_bytes"] == 500
    assert result["hlo_text_bytes"] == 24
    assert result["communication_collective_count"] == 1


def test_pair_summary_requires_dense_base_and_allows_deliberately_dense_head() -> None:
    common = {
        "cold_solve_ns": 100,
        "cold_aot_compile_ns": 80,
        "warm_solve_median_ns": 20,
        "rss_hwm_through_warm_bytes": 1_000,
        "compiler_peak_bytes": 900,
        "compiler_temp_bytes": 600,
        "device_peak_bytes": None,
        "hlo_text_bytes": 20,
        "hlo_instruction_count": 10,
    }
    result = _pair_summary(
        base={**common, "streamed_kernel_count": 0},
        head={**common, "streamed_kernel_count": 1},
    )
    assert result["head_over_base"]["cold_solve_ns"] == 1.0
    assert result["head_over_base"]["device_peak_bytes"] is None
    dense_result = _pair_summary(
        base={**common, "streamed_kernel_count": 0},
        head={**common, "streamed_kernel_count": 0},
    )
    assert dense_result["head_over_base"]["cold_solve_ns"] == 1.0

    with pytest.raises(RuntimeError, match="base unexpectedly"):
        _pair_summary(
            base={**common, "streamed_kernel_count": 1},
            head={**common, "streamed_kernel_count": 1},
        )
