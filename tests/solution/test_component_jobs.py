"""Independent component jobs: one block-major solve, split across processes.

`lcm.component_jobs` divides the codes of a block-major model's blocked state
into jobs. Each job runs the block-major engine on its own codes — solving each
code through all of its periods and, when the plan simulates, simulating that
code's subjects while its values are on the device — and publishes one
checksummed fragment. The collector checks every fragment against the plan
before it exposes one complete result.

Collected values and panels equal the single-process block-major result byte
for byte: every code runs the same programs on the same subjects with the same
random keys. A missing, failed, duplicated, stale or corrupt fragment is refused,
never collected into a partial result.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import h5py
import jax
import numpy as np
import pytest

from _lcm.solution import block_major
from lcm import InvariantBlockSchedule, Model, load_solution
from lcm.component_jobs import (
    CollectedComponentJobs,
    ComponentJobPlan,
    collect_component_jobs,
    load_component_job_plan,
    plan_component_jobs,
    run_component_job,
)
from lcm.exceptions import ExecutionPlanningError, SolutionIntegrityError
from lcm.result import SimulationResult
from lcm.solver_api import SolutionResult
from tests.simulation import test_type_grouped_simulation as life_cycle
from tests.solution import test_block_major_lifetime as lifetime

_BLOCK_MAJOR = InvariantBlockSchedule.BLOCK_MAJOR
_PERIOD_MAJOR = InvariantBlockSchedule.PERIOD_MAJOR
_SEED = 7
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _workload(*, name: str) -> tuple[Model, dict]:
    """Return a block-major model of a workload and its parameters."""
    return lifetime._workload(name=name, schedule=_BLOCK_MAJOR)


def _life_cycle() -> tuple[Model, dict]:
    return lifetime._life_cycle_model(schedule=_BLOCK_MAJOR), life_cycle._params(
        typed_dead=True
    )


def _fragment(*, directory: Path, job: int) -> Path:
    return directory / "fragments" / f"job-{job:04d}.h5"


def _failure_record(*, directory: Path, job: int) -> Path:
    return directory / "fragments" / f"job-{job:04d}.failed.json"


def _run_all(
    *,
    model: Model,
    params: Mapping,
    directory: Path,
    initial_conditions: Mapping[str, np.ndarray] | None = None,
) -> None:
    """Run every job of the plan in `directory`, one after the other."""
    plan = load_component_job_plan(directory=directory)
    for job in range(len(plan.jobs)):
        run_component_job(
            model=model,
            params=params,
            directory=directory,
            job=job,
            initial_conditions=initial_conditions,
            log_level="off",
        )


def _collect(*, model: Model, params: Mapping, directory: Path) -> CollectedComponentJobs:
    return collect_component_jobs(
        model=model, params=params, directory=directory, log_level="off"
    )


def _solve_only_jobs(
    *, name: str, directory: Path, n_jobs: int = 3
) -> tuple[Model, dict]:
    """Plan and run every job of a solve-only plan of a workload."""
    model, params = _workload(name=name)
    plan_component_jobs(model=model, params=params, directory=directory, n_jobs=n_jobs)
    _run_all(model=model, params=params, directory=directory)
    return model, params


@pytest.fixture(scope="module", name="complete_jobs")
def _fixture_complete_jobs(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, Model, dict, SolutionResult]:
    """A complete three-job directory of the independent-types model, and its reference."""
    directory = tmp_path_factory.mktemp("complete") / "jobs"
    model, params = _solve_only_jobs(name="independent_types", directory=directory)
    reference_model, _ = _workload(name="independent_types")
    reference = reference_model.solve(params=params, log_level="off")
    return directory, model, params, reference


def _copy_jobs(*, source: Path, tmp_path: Path) -> Path:
    """Copy a job directory so a test can damage it alone."""
    target = tmp_path / "jobs"
    shutil.copytree(source, target)
    return target


def test_plan_splits_the_codes_into_contiguous_jobs_in_grid_order(
    tmp_path: Path,
) -> None:
    """`n_jobs` cuts the codes in grid order, larger jobs first."""
    model, params = _workload(name="independent_types")

    plan = plan_component_jobs(
        model=model, params=params, directory=tmp_path / "jobs", n_jobs=2
    )

    assert (plan.state_name, plan.codes, plan.jobs) == (
        "pref_type",
        (0, 1, 2),
        ((0, 1), (2,)),
    )


def test_plan_keeps_an_explicit_assignment(tmp_path: Path) -> None:
    """An explicit assignment is kept job by job, each job's codes in grid order."""
    model, params = _workload(name="independent_types")

    plan = plan_component_jobs(
        model=model,
        params=params,
        directory=tmp_path / "jobs",
        assignment=((2, 0), (1,)),
    )

    assert plan.jobs == ((0, 2), (1,))


def test_plan_round_trips_through_its_file(tmp_path: Path) -> None:
    """The plan a job reads is the plan the launcher wrote."""
    model, params = _workload(name="independent_types")
    plan = plan_component_jobs(
        model=model, params=params, directory=tmp_path / "jobs", n_jobs=3
    )

    assert load_component_job_plan(directory=tmp_path / "jobs") == plan


_INVALID_SPLITS = (
    pytest.param({}, "n_jobs or assignment", id="neither"),
    pytest.param({"n_jobs": 1, "assignment": ((0, 1, 2),)}, "not both", id="both"),
    pytest.param({"n_jobs": 0}, "at least one", id="no_job"),
    pytest.param({"n_jobs": 4}, "at most 3", id="more_jobs_than_codes"),
    pytest.param({"assignment": ((0, 1, 5),)}, "5", id="unknown_code"),
    pytest.param({"assignment": ((0, 1), (1, 2))}, "twice", id="repeated_code"),
    pytest.param({"assignment": ((0,), (1,))}, "missing", id="missing_code"),
    pytest.param({"assignment": ((), (0, 1, 2))}, "empty", id="empty_job"),
)


@pytest.mark.parametrize(("split", "match"), _INVALID_SPLITS)
def test_plan_refuses_an_invalid_split(
    *, tmp_path: Path, split: dict[str, Any], match: str
) -> None:
    """Every code goes to exactly one nonempty job, or nothing is planned."""
    model, params = _workload(name="independent_types")

    with pytest.raises(ExecutionPlanningError, match=match):
        plan_component_jobs(
            model=model, params=params, directory=tmp_path / "jobs", **split
        )
    assert not (tmp_path / "jobs" / "plan.json").exists()


def test_plan_refuses_a_period_major_model(tmp_path: Path) -> None:
    """Jobs run the block-major engine; another schedule is refused with a remedy."""
    model, params = lifetime._workload(
        name="independent_types", schedule=_PERIOD_MAJOR
    )

    with pytest.raises(ExecutionPlanningError, match="BLOCK_MAJOR"):
        plan_component_jobs(
            model=model, params=params, directory=tmp_path / "jobs", n_jobs=3
        )


def test_plan_refuses_a_directory_that_is_not_empty(tmp_path: Path) -> None:
    """A plan never adopts the files of another plan."""
    model, params = _workload(name="independent_types")
    (tmp_path / "jobs").mkdir()
    (tmp_path / "jobs" / "leftover.h5").write_bytes(b"")

    with pytest.raises(ExecutionPlanningError, match="empty"):
        plan_component_jobs(
            model=model, params=params, directory=tmp_path / "jobs", n_jobs=3
        )


def test_plan_refuses_a_seed_without_initial_conditions(tmp_path: Path) -> None:
    """A simulating plan needs both its population and its seed."""
    model, params = _life_cycle()

    with pytest.raises(ExecutionPlanningError, match="initial_conditions"):
        plan_component_jobs(
            model=model,
            params=params,
            directory=tmp_path / "jobs",
            n_jobs=3,
            seed=_SEED,
        )


def test_job_refuses_an_index_outside_the_plan(tmp_path: Path) -> None:
    """A job index names one of the plan's jobs."""
    model, params = _workload(name="independent_types")
    plan_component_jobs(
        model=model, params=params, directory=tmp_path / "jobs", n_jobs=3
    )

    with pytest.raises(ExecutionPlanningError, match="3 jobs"):
        run_component_job(
            model=model,
            params=params,
            directory=tmp_path / "jobs",
            job=3,
            log_level="off",
        )


def test_job_refuses_parameters_other_than_the_plans(tmp_path: Path) -> None:
    """A job whose parameters differ from the plan's publishes nothing."""
    model, params = lifetime._workload(
        name="sector_typed_terminal", schedule=_BLOCK_MAJOR
    )
    plan_component_jobs(
        model=model, params=params, directory=tmp_path / "jobs", n_jobs=3
    )

    with pytest.raises(ExecutionPlanningError, match="params_fingerprint"):
        run_component_job(
            model=model,
            params=lifetime.stage3._sector_params(typed_terminal=True, scale=1.3),
            directory=tmp_path / "jobs",
            job=0,
            log_level="off",
        )
    assert not _fragment(directory=tmp_path / "jobs", job=0).exists()


def test_job_refuses_a_population_other_than_the_plans(tmp_path: Path) -> None:
    """A simulating job simulates the population the plan was made for."""
    model, params = _life_cycle()
    plan_component_jobs(
        model=model,
        params=params,
        directory=tmp_path / "jobs",
        n_jobs=3,
        initial_conditions=life_cycle._initial(),
        seed=_SEED,
    )

    with pytest.raises(ExecutionPlanningError, match="initial_conditions"):
        run_component_job(
            model=model,
            params=params,
            directory=tmp_path / "jobs",
            job=0,
            initial_conditions=life_cycle._initial(codes=(0, 1, 2) * 3),
            log_level="off",
        )


def test_job_refuses_initial_conditions_for_a_solve_only_plan(
    tmp_path: Path,
) -> None:
    """A plan made without a population simulates nothing in its jobs."""
    model, params = _life_cycle()
    plan_component_jobs(
        model=model, params=params, directory=tmp_path / "jobs", n_jobs=3
    )

    with pytest.raises(ExecutionPlanningError, match="solve-only"):
        run_component_job(
            model=model,
            params=params,
            directory=tmp_path / "jobs",
            job=0,
            initial_conditions=life_cycle._initial(),
            log_level="off",
        )


@pytest.mark.parametrize(
    ("workload", "n_jobs"),
    [
        ("independent_types", 3),
        ("sector_typed_terminal", 2),
        ("life_cycle", 3),
    ],
)
def test_collected_values_equal_the_single_process_values_bitwise(
    *, tmp_path: Path, workload: str, n_jobs: int
) -> None:
    """Every collected value is the single-process block-major value, byte for byte."""
    model, params = _solve_only_jobs(
        name=workload, directory=tmp_path / "jobs", n_jobs=n_jobs
    )
    reference, _ = _workload(name=workload)

    lifetime._assert_value_bytes_equal(
        got=_collect(model=model, params=params, directory=tmp_path / "jobs")
        .solution.values,
        want=reference.solve(params=params, log_level="off").values,
    )


def test_collected_values_follow_the_plans_parameters(tmp_path: Path) -> None:
    """The comparator sees the values of other parameters as different."""
    model, params = _solve_only_jobs(
        name="sector_typed_terminal", directory=tmp_path / "jobs"
    )
    changed = lifetime.stage3._sector_params(typed_terminal=True, scale=1.3)
    plan_component_jobs(
        model=model, params=changed, directory=tmp_path / "changed", n_jobs=3
    )
    _run_all(model=model, params=changed, directory=tmp_path / "changed")

    base = _collect(model=model, params=params, directory=tmp_path / "jobs")
    other = _collect(model=model, params=changed, directory=tmp_path / "changed")

    assert lifetime._leaf_bytes(
        other.solution.value(period=0, regime="working")
    ) != lifetime._leaf_bytes(base.solution.value(period=0, regime="working"))


def test_a_collected_solve_only_plan_has_no_simulation(complete_jobs: tuple) -> None:
    """Without a planned population the collector publishes values alone."""
    directory, model, params, _ = complete_jobs

    collected = _collect(model=model, params=params, directory=directory)

    assert collected.simulation is None


def _simulating_jobs(
    *,
    directory: Path,
    codes: tuple[int, ...] = life_cycle._UNBALANCED,
    assignment: tuple[tuple[int, ...], ...] | None = None,
    seed: int = _SEED,
) -> tuple[Model, dict, dict[str, np.ndarray]]:
    """Plan and run every job of a simulating plan of the life cycle."""
    model, params = _life_cycle()
    initial = life_cycle._initial(codes=codes)
    plan_component_jobs(
        model=model,
        params=params,
        directory=directory,
        n_jobs=None if assignment is not None else 3,
        assignment=assignment,
        initial_conditions=initial,
        seed=seed,
    )
    _run_all(
        model=model, params=params, directory=directory, initial_conditions=initial
    )
    return model, params, initial


def _single_process(
    *, params: Mapping, initial: Mapping[str, np.ndarray], seed: int = _SEED
) -> SimulationResult:
    """Solve and simulate the life cycle block-major in this process."""
    reference, _ = _life_cycle()
    return reference.simulate(
        params=params,
        initial_conditions=initial,
        solution=None,
        seed=seed,
        log_level="off",
    )


_ASSIGNMENTS = (
    pytest.param(None, id="one_code_per_job"),
    pytest.param(((0, 2), (1,)), id="uneven"),
)


@pytest.mark.parametrize("codes", lifetime._POPULATIONS)
@pytest.mark.parametrize("assignment", _ASSIGNMENTS)
def test_collected_panel_equals_the_single_process_panel_bitwise(
    *,
    tmp_path: Path,
    codes: tuple[int, ...],
    assignment: tuple[tuple[int, ...], ...] | None,
) -> None:
    """Each job simulates its codes' subjects; together they are the whole panel.

    Subjects keep their original rows and random keys, and a job whose codes no
    subject holds still solves them.
    """
    model, params, initial = _simulating_jobs(
        directory=tmp_path / "jobs", codes=codes, assignment=assignment
    )

    collected = _collect(model=model, params=params, directory=tmp_path / "jobs")

    assert collected.simulation is not None
    life_cycle._assert_panels_identical(
        got=collected.simulation,
        want=_single_process(params=params, initial=initial),
    )


def test_collected_simulation_carries_the_single_process_solution(
    tmp_path: Path,
) -> None:
    """The solution behind a collected panel holds the single-process values."""
    model, params, initial = _simulating_jobs(directory=tmp_path / "jobs")
    want = _single_process(params=params, initial=initial).solution
    assert isinstance(want, SolutionResult)

    collected = _collect(model=model, params=params, directory=tmp_path / "jobs")

    lifetime._assert_value_bytes_equal(
        got=collected.solution.values, want=want.values
    )


def test_the_panel_comparator_sees_a_changed_seed(tmp_path: Path) -> None:
    """Another seed draws another panel, which the comparator rejects."""
    model, params, initial = _simulating_jobs(directory=tmp_path / "jobs")
    collected = _collect(model=model, params=params, directory=tmp_path / "jobs")
    assert collected.simulation is not None

    with pytest.raises(AssertionError):
        life_cycle._assert_panels_identical(
            got=collected.simulation,
            want=_single_process(params=params, initial=initial, seed=_SEED + 1),
        )


def test_a_collected_solution_simulates_like_the_single_process_solution(
    tmp_path: Path,
) -> None:
    """Simulating the collected solution reads one code at a time, like 5B's."""
    model, params = _solve_only_jobs(name="life_cycle", directory=tmp_path / "jobs")
    initial = life_cycle._initial()
    reference, _ = _life_cycle()
    want = reference.simulate(
        params=params,
        initial_conditions=initial,
        solution=reference.solve(params=params, log_level="off"),
        seed=_SEED,
        log_level="off",
    )

    collected = _collect(model=model, params=params, directory=tmp_path / "jobs")

    life_cycle._assert_panels_identical(
        got=model.simulate(
            params=params,
            initial_conditions=initial,
            solution=collected.solution,
            seed=_SEED,
            log_level="off",
        ),
        want=want,
    )


def test_a_collected_solution_saves_and_loads_its_values(
    *, tmp_path: Path, complete_jobs: tuple
) -> None:
    """The archive round trip of a collected solution keeps every value."""
    directory, model, params, reference = complete_jobs
    collected = _collect(model=model, params=params, directory=directory)

    loaded = load_solution(path=collected.solution.save(path=tmp_path / "solution"))

    assert isinstance(loaded, SolutionResult)
    lifetime._assert_value_bytes_equal(
        got=loaded.values, want=reference.values, ordered=False
    )


def _run_job_in_this_process(*, directory: Path, job: int, simulate: bool) -> None:
    """Run one life-cycle job; the entry point of the separate-process test."""
    model, params = _life_cycle()
    run_component_job(
        model=model,
        params=params,
        directory=directory,
        job=job,
        initial_conditions=life_cycle._initial() if simulate else None,
        log_level="off",
    )


def test_a_job_run_in_a_separate_process_collects_to_the_same_result(
    tmp_path: Path,
) -> None:
    """A job holds no state of the process that planned or collects the jobs."""
    directory = tmp_path / "jobs"
    model, params = _life_cycle()
    initial = life_cycle._initial()
    plan_component_jobs(
        model=model,
        params=params,
        directory=directory,
        n_jobs=3,
        initial_conditions=initial,
        seed=_SEED,
    )
    x64 = bool(jax.config.read("jax_enable_x64"))
    script = (
        "import jax\n"
        f"jax.config.update('jax_enable_x64', {x64})\n"
        "from pathlib import Path\n"
        "from tests.solution import test_component_jobs as jobs\n"
        f"jobs._run_job_in_this_process(directory=Path({str(directory)!r}), "
        "job=1, simulate=True)\n"
    )
    completed = subprocess.run(  # noqa: S603
        [sys.executable, "-c", script],
        cwd=_REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(_REPO_ROOT)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    for job in (0, 2):
        run_component_job(
            model=model,
            params=params,
            directory=directory,
            job=job,
            initial_conditions=initial,
            log_level="off",
        )

    collected = _collect(model=model, params=params, directory=directory)

    assert collected.simulation is not None
    life_cycle._assert_panels_identical(
        got=collected.simulation,
        want=_single_process(params=params, initial=initial),
    )


def test_collector_refuses_a_missing_job(
    *, tmp_path: Path, complete_jobs: tuple
) -> None:
    """A job that never published leaves its codes uncovered; nothing is collected."""
    source, model, params, _ = complete_jobs
    directory = _copy_jobs(source=source, tmp_path=tmp_path)
    _fragment(directory=directory, job=1).unlink()

    with pytest.raises(ExecutionPlanningError, match=r"missing.*job 1"):
        _collect(model=model, params=params, directory=directory)


def test_a_failing_job_records_its_failure_and_publishes_nothing(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The job raises, writes a failure record and leaves no fragment."""
    model, params = _workload(name="independent_types")
    directory = tmp_path / "jobs"
    plan_component_jobs(model=model, params=params, directory=directory, n_jobs=3)
    _fail_component(monkeypatch=monkeypatch, code=1)

    with pytest.raises(RuntimeError, match="injected component failure"):
        run_component_job(
            model=model, params=params, directory=directory, job=1, log_level="off"
        )

    assert not _fragment(directory=directory, job=1).exists()
    record = json.loads(_failure_record(directory=directory, job=1).read_text())
    assert (record["job"], record["error_type"]) == (1, "RuntimeError")


def _fail_component(*, monkeypatch: pytest.MonkeyPatch, code: int) -> None:
    """Make solving `code` raise."""
    solve_component = block_major.ComponentSchedule.solve_component

    # keyword-only-exempt: library-callback=pytest.MonkeyPatch.setattr
    def failing(self: Any, *, code: int) -> object:  # noqa: ANN401
        if code == failing_code:
            msg = "injected component failure"
            raise RuntimeError(msg)
        return solve_component(self, code=code)

    failing_code = code
    monkeypatch.setattr(block_major.ComponentSchedule, "solve_component", failing)


def test_collector_refuses_a_failed_job(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A recorded failure is reported as a failure, not as a completed solution."""
    model, params = _workload(name="independent_types")
    directory = tmp_path / "jobs"
    plan_component_jobs(model=model, params=params, directory=directory, n_jobs=3)
    for job in (0, 2):
        run_component_job(
            model=model, params=params, directory=directory, job=job, log_level="off"
        )
    _fail_component(monkeypatch=monkeypatch, code=1)
    with pytest.raises(RuntimeError, match="injected component failure"):
        run_component_job(
            model=model, params=params, directory=directory, job=1, log_level="off"
        )

    with pytest.raises(ExecutionPlanningError, match=r"failed.*job 1"):
        _collect(model=model, params=params, directory=directory)


def test_a_rerun_of_a_failed_job_completes_the_collection(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A successful rerun replaces the failure record with its fragment."""
    model, params = _workload(name="independent_types")
    directory = tmp_path / "jobs"
    plan_component_jobs(model=model, params=params, directory=directory, n_jobs=3)
    with monkeypatch.context() as patch:
        _fail_component(monkeypatch=patch, code=1)
        with pytest.raises(RuntimeError, match="injected component failure"):
            _run_all(model=model, params=params, directory=directory)
    _run_all(model=model, params=params, directory=directory)
    reference, _ = _workload(name="independent_types")

    lifetime._assert_value_bytes_equal(
        got=_collect(model=model, params=params, directory=directory).solution.values,
        want=reference.solve(params=params, log_level="off").values,
    )


def test_collector_refuses_a_fragment_beside_a_failure_record(
    *, tmp_path: Path, complete_jobs: tuple
) -> None:
    """A job that both published and failed was run twice; neither is trusted."""
    source, model, params, _ = complete_jobs
    directory = _copy_jobs(source=source, tmp_path=tmp_path)
    _failure_record(directory=directory, job=2).write_text(
        json.dumps({"job": 2}), encoding="utf-8"
    )

    with pytest.raises(ExecutionPlanningError, match=r"duplicated.*job 2"):
        _collect(model=model, params=params, directory=directory)


def test_collector_refuses_a_fragment_filed_under_another_job(
    *, tmp_path: Path, complete_jobs: tuple
) -> None:
    """A copied fragment covers its codes twice and is refused as a duplicate."""
    source, model, params, _ = complete_jobs
    directory = _copy_jobs(source=source, tmp_path=tmp_path)
    shutil.copyfile(
        _fragment(directory=directory, job=0), _fragment(directory=directory, job=2)
    )

    with pytest.raises(ExecutionPlanningError, match=r"duplicated.*job 0"):
        _collect(model=model, params=params, directory=directory)


def test_collector_refuses_a_fragment_of_another_plan(
    *, tmp_path: Path, complete_jobs: tuple
) -> None:
    """A fragment left by an earlier plan is stale, even with the same parameters."""
    source, model, params, _ = complete_jobs
    directory = _copy_jobs(source=source, tmp_path=tmp_path)
    other = tmp_path / "other"
    plan_component_jobs(model=model, params=params, directory=other, n_jobs=3)
    run_component_job(
        model=model, params=params, directory=other, job=1, log_level="off"
    )
    shutil.copyfile(
        _fragment(directory=other, job=1), _fragment(directory=directory, job=1)
    )

    with pytest.raises(ExecutionPlanningError, match=r"stale.*job 1"):
        _collect(model=model, params=params, directory=directory)


def test_collector_refuses_parameters_other_than_the_plans(
    complete_jobs: tuple,
) -> None:
    """The collector reproduces the plan's identity before it trusts a fragment."""
    directory, model, _, _ = complete_jobs
    params = lifetime.independent_types.get_params()
    params["discount_factor"] = 0.5

    with pytest.raises(ExecutionPlanningError, match="params_fingerprint"):
        _collect(model=model, params=params, directory=directory)


def _rewrite_manifest(*, path: Path, change: Mapping[str, Any]) -> None:
    """Change top-level manifest entries of a fragment and reseal the manifest."""
    with h5py.File(path, "r+") as fragment:
        manifest = json.loads(bytes(fragment["manifest"][()]).decode())
        manifest.update(change)
        payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        del fragment["manifest"]
        dataset = fragment.create_dataset(
            "manifest", data=np.frombuffer(payload, dtype=np.uint8)
        )
        dataset.attrs["sha256"] = hashlib.sha256(payload).hexdigest()


def test_collector_refuses_fragments_from_different_executions(
    *, tmp_path: Path, complete_jobs: tuple
) -> None:
    """Fragments solved on different backends or device counts are not mixed."""
    source, model, params, _ = complete_jobs
    directory = _copy_jobs(source=source, tmp_path=tmp_path)
    path = _fragment(directory=directory, job=1)
    with h5py.File(path, "r") as fragment:
        execution = json.loads(bytes(fragment["manifest"][()]).decode())["execution"]
    _rewrite_manifest(
        path=path, change={"execution": {**execution, "n_devices": 1000}}
    )

    with pytest.raises(ExecutionPlanningError, match="execution"):
        _collect(model=model, params=params, directory=directory)


def test_collector_refuses_a_corrupt_value_block(
    *, tmp_path: Path, complete_jobs: tuple
) -> None:
    """One flipped byte in a value fails its checksum."""
    source, model, params, _ = complete_jobs
    directory = _copy_jobs(source=source, tmp_path=tmp_path)
    with h5py.File(_fragment(directory=directory, job=1), "r+") as fragment:
        name = next(iter(fragment["values"]))
        block = np.array(fragment["values"][name])
        flat = block.reshape(-1).view(np.uint8)
        flat[0] ^= 1
        fragment["values"][name][...] = block

    with pytest.raises(SolutionIntegrityError, match="checksum"):
        _collect(model=model, params=params, directory=directory)


def test_collector_refuses_a_truncated_fragment(
    *, tmp_path: Path, complete_jobs: tuple
) -> None:
    """A fragment copied only in part is unreadable, and refused as such."""
    source, model, params, _ = complete_jobs
    directory = _copy_jobs(source=source, tmp_path=tmp_path)
    path = _fragment(directory=directory, job=1)
    path.write_bytes(path.read_bytes()[: path.stat().st_size // 2])

    with pytest.raises(SolutionIntegrityError, match="job-0001.h5"):
        _collect(model=model, params=params, directory=directory)


def test_an_interrupted_write_is_ignored_and_a_rerun_completes_the_set(
    *, tmp_path: Path, complete_jobs: tuple
) -> None:
    """A temporary file of an interrupted job is never read as a fragment."""
    source, model, params, reference = complete_jobs
    directory = _copy_jobs(source=source, tmp_path=tmp_path)
    fragment = _fragment(directory=directory, job=1)
    shutil.move(fragment, fragment.parent / ".job-0001.h5.interrupted.tmp")
    with pytest.raises(ExecutionPlanningError, match=r"missing.*job 1"):
        _collect(model=model, params=params, directory=directory)

    run_component_job(
        model=model, params=params, directory=directory, job=1, log_level="off"
    )

    lifetime._assert_value_bytes_equal(
        got=_collect(model=model, params=params, directory=directory).solution.values,
        want=reference.values,
    )


def test_plan_reports_its_jobs_and_population(tmp_path: Path) -> None:
    """The returned plan names its state, codes, jobs and whether it simulates."""
    model, params = _life_cycle()

    plan = plan_component_jobs(
        model=model,
        params=params,
        directory=tmp_path / "jobs",
        n_jobs=3,
        initial_conditions=life_cycle._initial(),
        seed=_SEED,
    )

    assert isinstance(plan, ComponentJobPlan)
    assert (plan.jobs, plan.seed, plan.n_subjects) == (
        ((0,), (1,), (2,)),
        _SEED,
        len(life_cycle._UNBALANCED),
    )


def test_the_identity_names_the_precision_and_the_pylcm_build(tmp_path: Path) -> None:
    """The plan pins the precision and the installed pylcm sources it was made with."""
    model, params = _workload(name="independent_types")
    plan_component_jobs(
        model=model, params=params, directory=tmp_path / "jobs", n_jobs=3
    )

    identity = json.loads((tmp_path / "jobs" / "plan.json").read_text())["identity"]

    assert (identity["precision"], len(identity["pylcm_source_sha256"])) == (
        "float64" if jax.config.read("jax_enable_x64") else "float32",
        64,
    )

