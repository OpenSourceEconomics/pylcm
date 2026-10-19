"""Contract tests binding `ci-workloads.json` to `cpu.yml` and the repository.

These are inventory-freeze checks (implementation plan batch 1): they do not
change any selection, they only make sure the source-bound manifest still
matches what `cpu.yml` actually runs and what `tests/` actually contains.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from tests.ci import ci_workloads
from tests.ci.cpu_suite_invocations import cpu_suite_invocation_argvs
from tests.ci.shard_test_files import (
    general_shard_files,
    ignore_out_of_shard_collection,
)

_REPO_ROOT = Path(__file__).parents[2]
_WORKFLOW_PATH = _REPO_ROOT / ".github/workflows/cpu.yml"


def _workflow_text() -> str:
    return _WORKFLOW_PATH.read_text(encoding="utf-8")


def _workflow_run_blocks() -> list[str]:
    workflow = yaml.safe_load(_workflow_text())
    blocks: list[str] = []
    for job in workflow["jobs"].values():
        for step in job.get("steps", ()):
            run = step.get("run")
            if isinstance(run, str):
                blocks.append(run)
    return blocks


def test_manifest_file_exists_and_parses():
    manifest = ci_workloads.load_manifest()
    assert manifest["schema_version"] == 2
    assert manifest["invocations"]


def test_every_invocation_has_a_nonempty_selection():
    for inv in ci_workloads.invocations():
        assert inv["files"], f"{inv['id']} selects no files"


def test_every_invocation_id_is_unique():
    ids = [inv["id"] for inv in ci_workloads.invocations()]
    assert len(ids) == len(set(ids))


def test_every_test_file_is_in_the_manifest_or_explicitly_excluded():
    all_files = {
        p.relative_to(_REPO_ROOT).as_posix()
        for p in (_REPO_ROOT / "tests").rglob("test_*.py")
    }
    manifest_files = ci_workloads.all_manifest_files()
    excluded = ci_workloads.excluded_files()
    unaccounted = all_files - manifest_files - set(excluded)
    assert not unaccounted, (
        "test files neither selected by any invocation nor in the exclusion "
        f"list with a reason: {sorted(unaccounted)}"
    )
    # An exclusion must name a real reason, not stand in for a missing weight.
    for file_, reason in excluded.items():
        assert file_ in all_files
        assert reason
        assert len(reason) > 10


def test_no_manifest_file_is_a_phantom():
    """Every file the manifest names as selected must actually exist under tests/."""
    all_files = {
        p.relative_to(_REPO_ROOT).as_posix()
        for p in (_REPO_ROOT / "tests").rglob("test_*.py")
    }
    phantom = ci_workloads.all_manifest_files() - all_files
    assert not phantom, f"manifest names files that do not exist: {sorted(phantom)}"


def test_unweighted_files_are_listed_not_zeroed():
    manifest = ci_workloads.load_manifest()
    weighted = set(manifest["file_weights"])
    unweighted = set(manifest["unweighted_files"])
    assert weighted.isdisjoint(unweighted)
    referenced = ci_workloads.all_manifest_files()
    # Every file selected by some invocation is accounted for as weighted or
    # unweighted -- nothing silently falls through with an implicit zero.
    assert referenced <= weighted | unweighted


def test_four_device_invocations_run_at_worker_zero_in_a_fresh_process():
    for inv in ci_workloads.invocations():
        if inv["environment"]["isolation"] == "fresh-process-4-device-pin":
            assert inv["environment"]["workers"] == 0
            assert len(inv["files"]) == 1


def test_eight_device_invocations_reject_all_skips():
    for inv in ci_workloads.invocations():
        if inv["environment"]["isolation"] == "fresh-process-8-device-env":
            assert inv.get("no_skips_required") is True
            assert inv["environment"]["workers"] == 0


def test_solution_shard_files_partition_the_solution_directory_exactly():
    solution_files = {
        p.relative_to(_REPO_ROOT).as_posix()
        for p in (_REPO_ROOT / "tests/solution").rglob("test_*.py")
    }
    layout = ci_workloads.shard_layout()["solution"]
    for precision in (64, 32):
        count = layout[f"fp{precision}-solution"]["shards"]
        shard_ids = [
            f"solution-shard-{s}-fp{precision}-linux" for s in range(1, count + 1)
        ]
        union: set[str] = set()
        for sid in shard_ids:
            files = set(ci_workloads.files_for(invocation_id=sid))
            assert files, f"empty shard {sid}"
            assert union.isdisjoint(files), f"file assigned to two shards ({sid})"
            union |= files
        assert union == solution_files


def test_general_shard_files_partition_the_general_universe_exactly():
    """Every general leg's shards are a partition of one shared file universe.

    Disjoint and jointly complete, per leg: a file dropped from every shard
    would silently stop running on that platform while the lane stayed green,
    and a file in two shards would run twice and be attributed to whichever
    report was read last.
    """
    universe = set(ci_workloads.general_shard_universe())
    assert universe
    for leg, cfg in ci_workloads.shard_layout()["general"].items():
        union: set[str] = set()
        for index in range(1, cfg["shards"] + 1):
            files = set(
                ci_workloads.files_for(invocation_id=f"general-shard-{index}-{leg}")
            )
            assert files, f"empty shard {index} of {leg}"
            assert union.isdisjoint(files), f"file in two shards of {leg}"
            union |= files
        assert union == set(ci_workloads.general_leg_universe(leg=leg)), (
            f"{leg} shards do not cover the leg's general universe"
        )


def test_linux_only_files_are_general_universe_files():
    """A Linux-only entry narrows the general universe; it never adds to it."""
    linux_only = set(ci_workloads.linux_only_files())
    assert linux_only <= set(ci_workloads.general_shard_universe())


def test_linux_only_files_run_on_every_linux_general_leg():
    """Restricting a file to Linux keeps it running, on each Linux leg."""
    linux_only = set(ci_workloads.linux_only_files())
    for leg, cfg in ci_workloads.shard_layout()["general"].items():
        if not leg.endswith("-linux"):
            continue
        selected = {
            file_
            for index in range(1, cfg["shards"] + 1)
            for file_ in ci_workloads.files_for(
                invocation_id=f"general-shard-{index}-{leg}"
            )
        }
        assert linux_only <= selected, f"{leg} drops {sorted(linux_only - selected)}"


def test_no_macos_or_windows_general_shard_selects_a_linux_only_file():
    """No macOS or Windows general shard runs a file restricted to Linux.

    Lanes that select explicit node ids -- the per-OS timing witnesses -- choose
    their tests by hand for what they measure on that OS and are not narrowed.
    """
    linux_only = set(ci_workloads.linux_only_files())
    offenders = {
        inv["id"]: sorted(linux_only & set(inv["files"]))
        for inv in ci_workloads.invocations()
        if inv["id"].startswith("general-shard-")
        and inv["environment"]["runner_os"] != "Linux"
        and linux_only & set(inv["files"])
    }
    assert not offenders, offenders


@pytest.mark.parametrize(
    ("leg", "expected"),
    [
        ("fp64-linux", ("tests/test_a.py", "tests/test_b.py")),
        ("fp64-windows", ("tests/test_a.py",)),
        ("fp64-macos", ("tests/test_a.py",)),
    ],
)
def test_general_leg_universe_drops_linux_only_files_off_linux(
    *, leg: str, expected: tuple[str, ...], tmp_path: Path
):
    manifest = tmp_path / "ci-workloads.json"
    manifest.write_text(
        '{"general_shard_universe": ["tests/test_a.py", "tests/test_b.py"],'
        ' "linux_only_files": ["tests/test_b.py"]}',
        encoding="utf-8",
    )
    assert ci_workloads.general_leg_universe(leg=leg, path=manifest) == expected


def test_general_leg_universe_refuses_a_leg_without_an_os(*, tmp_path: Path):
    manifest = tmp_path / "ci-workloads.json"
    manifest.write_text(
        '{"general_shard_universe": [], "linux_only_files": []}', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="fp64-solaris"):
        ci_workloads.general_leg_universe(leg="fp64-solaris", path=manifest)


def test_the_shard_assignment_is_reproduced_by_the_sharder():
    """The recorded assignment is exactly what `shard_test_files` recomputes.

    The manifest records the layout so contract tests and the predicted-minutes
    table can read it without a collection run, but CI computes the split at
    run time. This is the test that keeps the two from drifting apart.
    """
    for leg, cfg in ci_workloads.shard_layout()["general"].items():
        for index in range(1, cfg["shards"] + 1):
            recorded = ci_workloads.files_for(
                invocation_id=f"general-shard-{index}-{leg}"
            )
            computed = general_shard_files(leg=leg, n_shards=cfg["shards"], shard=index)
            assert tuple(sorted(recorded)) == tuple(sorted(computed)), (
                f"{leg} shard {index} drifted from the sharder"
            )


def test_source_only_modules_run_only_on_the_source_contract_lane() -> None:
    """Source campaigns run once while portability remains on every general lane."""
    source_only = set(
        ci_workloads.files_for(invocation_id="source-contract-fp64-linux")
    )
    assert source_only == {
        "tests/test_simulation_candidate_program_certificate.py",
        "tests/test_uniform_process_grid_certificate.py",
        "tests/test_outer_helper_certificate.py",
        "tests/test_final_nested_aggregate_certificate.py",
        "tests/test_additional_helper_aggregate_certificate.py",
        "tests/test_child_native_aggregate_certificate.py",
    }
    universe = set(ci_workloads.general_shard_universe())
    assert not (source_only & universe), (
        "a source-only campaign is still selected by the general lanes"
    )
    assert "tests/test_source_certificate_portability.py" in universe

    for inv in ci_workloads.invocations():
        if inv["id"] == "source-contract-fp64-linux":
            continue
        assert not (source_only & set(inv["files"])), (
            f"{inv['id']} also runs a source-only campaign"
        )


def test_leg_weights_are_recorded_per_leg_not_as_a_cross_leg_sum():
    """A per-job budget must never be computed from the cross-leg CSV totals.

    `file_weights` sums the four general legs; the split-workflow parity module
    costs 13 minutes of Windows and 9 of Linux fp64, and its cross-leg entry is
    those and the other two added together. Sharding against that number would
    over-provision the cheap platform and under-provision the expensive one, so
    the layout reads `leg_weights` instead. This test pins the distinction.

    The exemplar has to be a file the general legs still run: a module that has
    moved to a single-leg lane -- as the source-only certificates did -- keeps
    a cross-leg entry but stops accruing per-leg seconds, so it would report
    this contract as broken when only its lane changed.
    """
    manifest = ci_workloads.load_manifest()
    legs = set(ci_workloads.leg_names())
    assert {"fp64-linux", "fp64-macos", "fp64-windows", "fp32-linux"} <= legs
    parity = "tests/simulation/test_nnbegm_split_workflow_parity.py"
    assert parity in set(ci_workloads.general_shard_universe())
    legs_checked = ("fp64-linux", "fp64-macos", "fp64-windows", "fp32-linux")
    per_leg: dict[str, float] = {}
    for leg in legs_checked:
        seconds = ci_workloads.leg_weights(leg=leg).get(parity)
        assert seconds is not None, f"{parity} has no recorded weight on {leg}"
        per_leg[leg] = seconds
    assert len(set(per_leg.values())) > 1, "per-leg weights collapsed to one value"
    cross_leg = float(manifest["file_weights"][parity]["seconds"])
    assert cross_leg > max(per_leg.values())


def test_receipt_env_var_is_exported_in_every_pytest_step():
    """`cpu.yml` sets `PYLCM_CI_RECEIPT` around every CPU-suite pytest invocation.

    Additive only: this does not check or constrain selection, only that the
    receipt writer (opt-in via the same env var, see `tests/ci/receipt_plugin.py`)
    is wired up so receipts get produced for every invocation.
    """
    workflow = yaml.safe_load(_workflow_text())
    for job_name, job in workflow["jobs"].items():
        for step in job.get("steps", ()):
            run = step.get("run")
            if not isinstance(run, str):
                continue
            if not cpu_suite_invocation_argvs(run):
                continue
            step_env = step.get("env", {}) or {}
            job_env = job.get("env", {}) or {}
            assert "PYLCM_CI_RECEIPT" in step_env or "PYLCM_CI_RECEIPT" in job_env, (
                f"{job_name}/{step.get('name')} runs pytest without exporting "
                "PYLCM_CI_RECEIPT"
            )


@pytest.mark.parametrize(
    "job_name",
    sorted({inv["job"] for inv in ci_workloads.invocations()}),
)
def test_receipt_upload_step_exists_for_every_job(job_name):
    workflow = yaml.safe_load(_workflow_text())
    job = workflow["jobs"][job_name]
    upload_paths = " ".join(
        (step.get("with", {}) or {}).get("path", "")
        for step in job.get("steps", ())
        if step.get("uses", "").startswith("actions/upload-artifact")
    )
    assert "ci-receipts" in upload_paths


def test_a_test_named_directory_is_never_ignored_by_the_shard_filter():
    """Sharding must not ignore `tests/test_models/`, a directory named `test_*`.

    Regression control. The first version of the filter judged any collection
    candidate whose name started with `test_`, which matched that DIRECTORY and
    so removed all 30 of its cases from every shard at once. Each shard still
    passed, and each shard's own file list still looked right, so nothing but a
    union-against-baseline collection check could see it.
    """
    directory = _REPO_ROOT / "tests/test_models"
    assert directory.is_dir()
    assert not ignore_out_of_shard_collection(
        collection_path=directory, root=_REPO_ROOT, shard_files=frozenset()
    )

    module = directory / "test_ds_app2_housing_builds.py"
    assert module.is_file()
    assert ignore_out_of_shard_collection(
        collection_path=module, root=_REPO_ROOT, shard_files=frozenset()
    )
    assert not ignore_out_of_shard_collection(
        collection_path=module,
        root=_REPO_ROOT,
        shard_files=frozenset({"tests/test_models/test_ds_app2_housing_builds.py"}),
    )


def test_every_general_shard_universe_file_exists_and_is_collectable():
    """The sharded universe names only real files under `tests/`."""
    for name in ci_workloads.general_shard_universe():
        assert name.startswith("tests/"), name
        assert (_REPO_ROOT / name).is_file(), name


def test_every_os_persistence_test_names_a_test_function_in_its_file():
    """Each entry is `<file>::<test function>` and that function exists."""
    import ast  # noqa: PLC0415

    entries = ci_workloads.os_persistence_tests()
    assert entries
    missing = []
    for entry in entries:
        file_, name = entry.split("::")
        tree = ast.parse((_REPO_ROOT / file_).read_text(encoding="utf-8"))
        names = {
            node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
        }
        if name not in names:
            missing.append(entry)
    assert not missing, missing


@pytest.mark.parametrize("runner_os", ["macOS", "Windows"])
def test_every_os_persistence_test_runs_on_each_platform_leg(*, runner_os: str):
    """A persistence test is selected by an `os_persistence` lane on that OS."""
    files = {
        file_
        for inv in ci_workloads.invocations()
        if inv["environment"]["runner_os"] == runner_os
        and "os_persistence" in inv["selection"]
        for file_ in inv["files"]
    }
    entries = ci_workloads.os_persistence_tests()
    assert {entry.split("::")[0] for entry in entries} <= files


def test_conftest_marks_exactly_the_listed_persistence_tests():
    """Collection carries `os_persistence` on the listed tests and on no other."""
    import subprocess  # noqa: PLC0415
    import sys  # noqa: PLC0415

    target = "tests/solution/test_ephemeral_identity.py"
    listed = {
        entry
        for entry in ci_workloads.os_persistence_tests()
        if entry.startswith(target)
    }
    assert listed
    completed = subprocess.run(  # noqa: S603 - fixed pytest argv on a repository file
        [
            sys.executable,
            "-m",
            "pytest",
            target,
            "--collect-only",
            "-q",
            "-p",
            "no:cacheprovider",
            "-m",
            "os_persistence",
        ],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    collected = {
        line.split("[")[0]
        for line in completed.stdout.splitlines()
        if line.startswith(target + "::")
    }
    assert collected == listed, completed.stdout + completed.stderr
