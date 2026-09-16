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
        assert union == universe, f"{leg} shards do not cover the general universe"


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


def test_the_two_source_only_modules_run_only_on_the_source_contract_lane():
    """Batch 4: the consolidated modules leave every numerical lane.

    Their portability control stays behind on every general lane, so removing
    them costs no platform signal -- that is asserted here rather than left as
    a claim in a comment.
    """
    source_only = set(
        ci_workloads.files_for(invocation_id="source-contract-fp64-linux")
    )
    assert source_only == {
        "tests/test_simulation_candidate_program_certificate.py",
        "tests/test_uniform_process_grid_certificate.py",
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

    `file_weights` sums the four general legs; the program certificate's entry
    there is 37 minutes of Windows and 27 of Linux fp64 added together. Sharding
    against that number would over-provision Linux and under-provision Windows,
    so the layout reads `leg_weights` instead. This test pins the distinction.
    """
    manifest = ci_workloads.load_manifest()
    legs = set(ci_workloads.leg_names())
    assert {"fp64-linux", "fp64-macos", "fp64-windows", "fp32-linux"} <= legs
    program = "tests/test_simulation_candidate_program_certificate.py"
    legs_checked = ("fp64-linux", "fp64-macos", "fp64-windows", "fp32-linux")
    per_leg: dict[str, float] = {}
    for leg in legs_checked:
        seconds = ci_workloads.leg_weights(leg=leg).get(program)
        assert seconds is not None, f"{program} has no recorded weight on {leg}"
        per_leg[leg] = seconds
    assert len(set(per_leg.values())) > 1, "per-leg weights collapsed to one value"
    cross_leg = float(manifest["file_weights"][program]["seconds"])
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
