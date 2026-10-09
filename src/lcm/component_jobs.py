"""Split one block-major solve into independent component jobs, and collect them.

A block-major model (`ExecutionConfig(invariant_block_widths={...: 1},
invariant_block_schedule=InvariantBlockSchedule.BLOCK_MAJOR)`) solves each code
of its blocked state as an independent problem. A component job runs that
block-major engine on some of the codes, so jobs on separate nodes or processes
exchange nothing while they run:

1. `plan_component_jobs` assigns the codes to jobs and writes the task manifest,
   which pins the model, parameters, precision and pylcm build every job must
   reproduce.
2. `run_component_job` runs one job. It takes the model and parameters the
   user's script builds, solves the job's codes and, when the plan simulates,
   simulates the subjects holding them; then it publishes one fragment.
3. `collect_component_jobs` verifies every fragment and returns the complete
   result: the solution a single block-major solve publishes and, when the plan
   simulates, the panel of the whole population in its original order.

A job is the unit of retry: rerunning one replaces its fragment. A missing,
failed, duplicated, stale or corrupt fragment is refused, never collected into a
partial result.

Workers and the collector use the same log level, JIT/action configuration,
compiler versions/options, installed native build and kind of hardware. These
execution choices are recorded separately from the mathematical solution identity.
Effective ambient JIT, PRNG implementation, seed offset and Threefry partitioning
also match across numeric workers, collection and the single-process reference.
Planning, running and collecting component jobs require `durable_identity=True`.
"""

import contextlib
import functools
import hashlib
import json
import os
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from types import MappingProxyType
from typing import cast

import jax
import jaxlib
import numpy as np
import pandas as pd

import _lcm
import lcm
from _lcm import version as _version
from _lcm.egm.upper_envelope._exact_affine.ffi import _installed_native_directory
from _lcm.engine import PeriodRegimeSimulationData, placed_devices_for_ids
from _lcm.persistence.solution import (
    _require_exact_dict,
    _require_exact_list,
    _require_nonnegative_exact_int,
    _require_positive_exact_int,
    _require_sha256,
)
from _lcm.simulation.chunk_offload import chunk_host_device
from _lcm.simulation.programs import forward_regimes_by_period
from _lcm.simulation.subject_devices import simulation_subject_devices
from _lcm.simulation.subject_groups import group_codes, group_sizes
from _lcm.solution.artifacts import fingerprint_flat_params
from _lcm.solution.block_major import (
    RetainedComponentValues,
    invariant_components,
    selected_components,
)
from _lcm.solution.component_fragments import (
    FORMAT_VERSION,
    FRAGMENT_DIRECTORY,
    PLAN_FILE,
    PLAN_FORMAT,
    Coordinate,
    Fragment,
    FragmentPanel,
    LeafAddress,
    array_checksum,
    canonical_json,
    failure_job,
    failure_name,
    fragment_job,
    fragment_name,
    read_fragment,
    sha256_hex,
    write_fragment,
    write_json_atomically,
)
from _lcm.transition_checks import validate_transitions
from _lcm.typing import FlatParams, InitialConditions
from _lcm.utils.logging import LogLevel, get_logger
from lcm.exceptions import ExecutionPlanningError, SolutionIntegrityError
from lcm.model import _VALUES_RETAINED_ON_THE_HOST, Model
from lcm.result import SimulationResult
from lcm.solver_api import ResultRetention, SolutionResult
from lcm.typing import RegimeName, StateName, UserInitialConditions, UserParams

__all__ = [
    "CollectedComponentJobs",
    "ComponentJobPlan",
    "collect_component_jobs",
    "load_component_job_plan",
    "plan_component_jobs",
    "run_component_job",
]

_SCHEDULE_REMEDY = (
    "Build the model with ExecutionConfig(invariant_block_widths={<state>: 1}, "
    "invariant_block_schedule=InvariantBlockSchedule.BLOCK_MAJOR)."
)


@dataclass(frozen=True, kw_only=True)
class ComponentJobPlan:
    """The task manifest of one set of component jobs."""

    directory: Path
    """The directory holding the plan and the jobs' fragments."""

    plan_id: str
    """Identifier of this plan; fragments of another plan are refused."""

    digest: str
    """SHA-256 of the plan file every fragment records."""

    state_name: StateName
    """The blocked state whose codes are divided."""

    codes: tuple[int, ...]
    """Every code of the state, in grid order."""

    jobs: tuple[tuple[int, ...], ...]
    """The codes of each job, each in grid order."""

    identity: MappingProxyType[str, object]
    """Model, parameter, precision, build and solve-configuration identity."""

    seed: int | None
    """Simulation seed, or `None` for a plan that does not simulate."""

    n_subjects: int | None
    """Number of simulated subjects, or `None` for a plan that does not simulate."""

    initial_conditions_sha256: str | None
    """Digest of the planned population, or `None` for a plan that does not simulate."""

    job_rows_sha256: tuple[str, ...] | None
    """Digest of each job's original rows, or `None` for a solve-only plan."""

    code_counts: tuple[int, ...] | None
    """Subjects holding each code, in grid order, or `None` for a solve-only plan."""

    @property
    def simulates(self) -> bool:
        """Whether the jobs simulate their codes' subjects."""
        return self.seed is not None


@dataclass(frozen=True, kw_only=True, eq=False)
class CollectedComponentJobs:
    """The complete result of a set of component jobs."""

    plan: ComponentJobPlan
    """The plan the jobs ran."""

    solution: SolutionResult
    """Every value, as the single-process block-major solve publishes it."""

    simulation: SimulationResult | None
    """The whole population's panel, or `None` for a plan that does not simulate."""


def plan_component_jobs(
    *,
    model: Model,
    params: UserParams,
    directory: Path,
    n_jobs: int | None = None,
    assignment: tuple[tuple[int, ...], ...] | None = None,
    initial_conditions: UserInitialConditions | pd.DataFrame | None = None,
    seed: int | None = None,
) -> ComponentJobPlan:
    """Divide the blocked state's codes into jobs and write the task manifest.

    Args:
        model: A block-major model with durable identity.
        params: The parameters every job solves with.
        directory: Where the plan and the fragments go; absent or empty.
        n_jobs: Number of jobs, each taking a contiguous run of codes in grid
            order, larger runs first. Give this or `assignment`.
        assignment: The codes of each job. Every code appears in exactly one
            nonempty job.
        initial_conditions: The population every job simulates its codes'
            subjects of, or `None` for jobs that only solve. Give it together
            with `seed`.
        seed: The simulation seed, given together with `initial_conditions`.

    Returns:
        The plan, as every job reads it back.

    Raises:
        ExecutionPlanningError: The model is ephemeral or not block-major, the
            split is invalid, only one of `initial_conditions` and `seed` is given, or
            `directory` is not empty.
        ModelIdentityError: The model's durable identity is invalid.

    """
    _fail_if_ephemeral_model(model=model, operation="plan_component_jobs")
    _fail_if_not_block_major(model=model)
    model._check_identity_runtime()  # noqa: SLF001
    state_name, codes = _get_model_components(model=model)
    jobs = _split_codes(codes=codes, n_jobs=n_jobs, assignment=assignment)
    if (initial_conditions is None) != (seed is None):
        msg = (
            "A simulating plan needs both initial_conditions and seed; pass both "
            "to simulate in the jobs, or neither to solve only."
        )
        raise ExecutionPlanningError(msg)
    if seed is not None and type(seed) is not int:
        msg = "A simulation seed must be an exact integer."
        raise ExecutionPlanningError(msg)
    n_subjects = (
        None
        if initial_conditions is None
        else _n_subjects(initial_conditions=initial_conditions)
    )
    if n_subjects == 0:
        msg = "A simulating component plan requires a nonempty overall population."
        raise ExecutionPlanningError(msg)
    if directory.exists() and any(directory.iterdir()):
        msg = (
            f"The plan directory {directory} is not empty; a plan never adopts the "
            "files of another plan. Pass a new or empty directory."
        )
        raise ExecutionPlanningError(msg)
    if initial_conditions is None:
        flat_params = model._process_params(params)  # noqa: SLF001
        job_rows_sha256 = None
        code_counts = None
    else:
        canonical, flat_params = model._canonical_feasibility_inputs(  # noqa: SLF001
            initial_conditions=initial_conditions, params=params
        )
        job_rows_sha256, code_counts = _planned_rows(
            model=model,
            jobs=jobs,
            codes=codes,
            initial_conditions=canonical,
            n_subjects=cast("int", n_subjects),
        )
    (directory / FRAGMENT_DIRECTORY).mkdir(parents=True, exist_ok=True)
    write_json_atomically(
        path=directory / PLAN_FILE,
        payload={
            "format": PLAN_FORMAT,
            "format_version": FORMAT_VERSION,
            "plan_id": uuid.uuid4().hex,
            "identity": _identity(model=model, flat_params=flat_params),
            "state_name": state_name,
            "codes": list(codes),
            "jobs": [list(job) for job in jobs],
            "simulation": None
            if initial_conditions is None
            else {
                "seed": seed,
                "n_subjects": n_subjects,
                "initial_conditions_sha256": _initial_conditions_sha256(
                    initial_conditions=initial_conditions
                ),
                "job_rows_sha256": list(cast("tuple[str, ...]", job_rows_sha256)),
                "code_counts": list(cast("tuple[int, ...]", code_counts)),
            },
        },
    )
    return load_component_job_plan(directory=directory)


def load_component_job_plan(*, directory: Path) -> ComponentJobPlan:
    """Read the task manifest a launcher wrote into `directory`.

    Raises:
        SolutionIntegrityError: The plan file is missing or not a plan.

    """
    path = directory / PLAN_FILE
    try:
        payload = path.read_bytes()
        plan = _require_exact_dict(
            value=json.loads(payload), label="component job plan"
        )
        format_version = _require_nonnegative_exact_int(
            value=plan["format_version"], label="component plan format version"
        )
        if (plan["format"], format_version) != (PLAN_FORMAT, FORMAT_VERSION):
            msg = f"{path} is not a {PLAN_FORMAT!r} version {FORMAT_VERSION} plan."
            raise SolutionIntegrityError(msg)
        simulation = (
            None
            if plan["simulation"] is None
            else _require_exact_dict(value=plan["simulation"], label="simulation plan")
        )
        seed = None if simulation is None else simulation["seed"]
        if simulation is not None and type(seed) is not int:
            msg = "A simulation seed must be an exact integer."
            raise SolutionIntegrityError(msg)
        codes = tuple(
            _require_nonnegative_exact_int(value=code, label="component plan code")
            for code in _require_exact_list(
                value=plan["codes"], label="component codes"
            )
        )
        jobs = tuple(
            tuple(
                _require_nonnegative_exact_int(value=code, label="component job code")
                for code in _require_exact_list(value=job, label="component job codes")
            )
            for job in _require_exact_list(value=plan["jobs"], label="component jobs")
        )
        _split_codes(codes=codes, n_jobs=None, assignment=jobs)
        job_rows_sha256 = (
            None
            if simulation is None
            else tuple(
                _require_sha256(value=digest, label="component job rows")
                for digest in _require_exact_list(
                    value=simulation["job_rows_sha256"],
                    label="component job row digests",
                )
            )
        )
        if job_rows_sha256 is not None and len(job_rows_sha256) != len(jobs):
            msg = "Job row digests do not match the planned code jobs."
            raise SolutionIntegrityError(msg)
        n_subjects = (
            None
            if simulation is None
            else _require_positive_exact_int(
                value=simulation["n_subjects"], label="simulation population count"
            )
        )
        code_counts = (
            None
            if simulation is None
            else tuple(
                _require_nonnegative_exact_int(value=count, label="code population")
                for count in _require_exact_list(
                    value=simulation["code_counts"], label="code populations"
                )
            )
        )
        if code_counts is not None and (
            len(code_counts) != len(codes) or sum(code_counts) != n_subjects
        ):
            msg = (
                f"Code populations {code_counts!r} do not give one count per code "
                f"of {codes!r} summing to the population of {n_subjects}."
            )
            raise SolutionIntegrityError(msg)
        return ComponentJobPlan(
            directory=directory,
            plan_id=str(plan["plan_id"]),
            digest=sha256_hex(payload),
            state_name=str(plan["state_name"]),
            codes=codes,
            jobs=jobs,
            identity=MappingProxyType(
                dict(cast("Mapping[str, object]", plan["identity"]))
            ),
            seed=cast("int | None", seed),
            n_subjects=n_subjects,
            initial_conditions_sha256=None
            if simulation is None
            else _require_sha256(
                value=simulation["initial_conditions_sha256"],
                label="initial population digest",
            ),
            job_rows_sha256=job_rows_sha256,
            code_counts=code_counts,
        )
    except (OSError, KeyError, TypeError, ValueError, ExecutionPlanningError) as error:
        msg = f"{path} cannot be read as a component job plan: {error}"
        raise SolutionIntegrityError(msg) from error


def run_component_job(
    *,
    model: Model,
    params: UserParams,
    directory: Path,
    job: int,
    log_level: LogLevel,
    initial_conditions: UserInitialConditions | pd.DataFrame | None = None,
    max_compilation_workers: int | None = None,
) -> Path:
    """Run one job of a plan and publish its fragment.

    The job solves each of its codes through every period on the block-major
    engine. When the plan simulates, it simulates the subjects holding its
    codes while their values are on the device, on the chunks and with the
    random keys the whole population's simulation gives them. Whatever the job
    raises, it records the failure, publishes no fragment and re-raises.

    Args:
        model: The durable block-major model the plan was made for.
        params: The parameters the plan was made with.
        directory: The plan directory.
        job: Index of the job in the plan.
        log_level: Verbosity and runtime-validation policy of the solve and
            simulation.
        initial_conditions: The planned population; required exactly when the
            plan simulates.
        max_compilation_workers: Maximum threads for parallel XLA compilation.

    Returns:
        The path of the published fragment.

    Raises:
        ExecutionPlanningError: The model is ephemeral, the job is not in the plan,
            the model or parameters do not reproduce the plan's identity, or the
            population is missing, unexpected or differs from the plan's.

    """
    _fail_if_ephemeral_model(model=model, operation="run_component_job")
    _fail_if_not_block_major(model=model)
    model._check_identity_runtime()  # noqa: SLF001
    plan = load_component_job_plan(directory=directory)
    if type(job) is not int or not 0 <= job < len(plan.jobs):
        msg = (
            f"Job {job} is not a job of the plan in {directory}, which has "
            f"{len(plan.jobs)} jobs numbered from 0."
        )
        raise ExecutionPlanningError(msg)
    flat_params = model._process_params(params)  # noqa: SLF001
    _fail_if_identity_differs(
        model=model,
        plan=plan,
        identity=_identity(model=model, flat_params=flat_params),
        source=f"job {job}",
    )
    _fail_if_population_differs(plan=plan, initial_conditions=initial_conditions)
    fragments = directory / FRAGMENT_DIRECTORY
    codes = plan.jobs[job]
    try:
        if plan.simulates:
            retained, panel = _simulate_job(
                model=model,
                params=params,
                plan=plan,
                codes=codes,
                initial_conditions=initial_conditions,
                log_level=log_level,
                max_compilation_workers=max_compilation_workers,
            )
        else:
            retained = _solve_job(
                model=model,
                flat_params=flat_params,
                codes=codes,
                log_level=log_level,
                max_compilation_workers=max_compilation_workers,
            )
            panel = None
        path = fragments / fragment_name(job=job)
        write_fragment(
            path=path,
            header={
                "plan_sha256": plan.digest,
                "plan_id": plan.plan_id,
                "job": job,
                "codes": list(codes),
                "identity": dict(plan.identity),
                "execution": _execution_record(model=model, log_level=log_level),
            },
            coordinates=retained.coordinates,
            values={code: retained.host_blocks(code=code) for code in codes},
            panel=panel,
        )
    except BaseException as error:
        # The job's own error is what the caller sees, even when the record
        # cannot be written.
        with contextlib.suppress(OSError):
            write_json_atomically(
                path=fragments / failure_name(job=job),
                payload={
                    "plan_sha256": plan.digest,
                    "job": job,
                    "codes": list(codes),
                    "error_type": type(error).__name__,
                    "message": str(error),
                },
            )
        raise
    (fragments / failure_name(job=job)).unlink(missing_ok=True)
    return path


def collect_component_jobs(
    *,
    model: Model,
    params: UserParams,
    directory: Path,
    log_level: LogLevel,
    max_compilation_workers: int | None = None,
) -> CollectedComponentJobs:
    """Verify every job's fragment and return the complete result.

    Each fragment must belong to this plan, reproduce its identity, cover
    exactly its job's codes and pass every checksum; every job must have
    published exactly one fragment and no failure record; every fragment must
    come from one kind of execution. Only then are the values handed to the
    block-major retention a single-process solve builds, and the simulated rows
    restored to the population's order.

    Args:
        model: The durable block-major model the plan was made for.
        params: The parameters the plan was made with.
        directory: The plan directory.
        log_level: Verbosity and runtime-validation policy. Must match the
            workers' log level and recorded execution configuration.
        max_compilation_workers: Maximum threads for parallel XLA compilation.

    Returns:
        The plan, the complete solution and, when the plan simulates, the
        complete simulation.

    Raises:
        ExecutionPlanningError: The model is ephemeral, the model or parameters
            do not reproduce the plan's identity, or a job is missing, failed,
            duplicated, stale or ran on another kind of execution. Every such job
            is named.
        SolutionIntegrityError: A fragment is unreadable, partial or fails a
            checksum.

    """
    _fail_if_ephemeral_model(model=model, operation="collect_component_jobs")
    _fail_if_not_block_major(model=model)
    model._check_identity_runtime()  # noqa: SLF001
    plan = load_component_job_plan(directory=directory)
    flat_params = model._process_params(params)  # noqa: SLF001
    _fail_if_identity_differs(
        model=model,
        plan=plan,
        identity=_identity(model=model, flat_params=flat_params),
        source="the collector",
    )
    fragments = _complete_fragments(
        plan=plan, execution=_execution_record(model=model, log_level=log_level)
    )
    log = get_logger(log_level=log_level)
    validate_transitions(
        regimes=model._regimes,  # noqa: SLF001
        flat_params=flat_params,
        ages=model.ages,
        logger=log,
        process_grid_resolver=None,
    )
    preparation = model._prepare_solution(  # noqa: SLF001
        flat_params=flat_params,
        log=log,
        retention=ResultRetention.VALUES_AND_REPLAY,
        process_grid_resolver=None,
        call_id=None,
    )
    schedule = model._component_schedule(  # noqa: SLF001
        preparation=preparation,
        flat_params=flat_params,
        log=log,
        max_compilation_workers=max_compilation_workers,
        process_grid_resolver=None,
        call_id=None,
    )
    # DeclaredAuthority.values is a coordinate mapping, not a pandas object.
    value_dtypes = {
        coordinate: descriptor.dtype
        for coordinate, descriptor in preparation.declared_authority.values.items()  # noqa: PD011
    }
    _retain_fragments(
        retained=schedule.retained,
        fragments=fragments,
        plan=plan,
        value_dtypes=value_dtypes,
    )
    solution = model._finish_solution(  # noqa: SLF001
        preparation=preparation,
        internal_result=_VALUES_RETAINED_ON_THE_HOST,
        flat_params=flat_params,
        log=log,
        call_id=None,
        component_values=schedule.finish(),
    )
    simulation = (
        _collected_simulation(
            model=model,
            flat_params=flat_params,
            plan=plan,
            fragments=fragments,
            solution=solution,
            value_dtypes=value_dtypes,
        )
        if plan.simulates
        else None
    )
    return CollectedComponentJobs(plan=plan, solution=solution, simulation=simulation)


def _fail_if_ephemeral_model(*, model: Model, operation: str) -> None:
    """Require a reproducible identity before publishing or consuming a campaign."""
    if not model.durable_identity:
        msg = (
            f"{operation} cannot use an ephemeral model. Build the model with "
            "durable_identity=True, or use ordinary local solve()/simulate()."
        )
        raise ExecutionPlanningError(msg)


def _fail_if_not_block_major(*, model: Model) -> None:
    """Refuse a model whose solve is not the block-major schedule."""
    if not model._solves_block_major:  # noqa: SLF001
        msg = (
            "Component jobs run the block-major engine, but the model's "
            "invariant_block_schedule is "
            f"{model._execution.invariant_block_schedule!r}. "  # noqa: SLF001
            f"{_SCHEDULE_REMEDY} BLOCK_MAJOR is required."
        )
        raise ExecutionPlanningError(msg)


def _split_codes(
    *,
    codes: tuple[int, ...],
    n_jobs: int | None,
    assignment: tuple[tuple[int, ...], ...] | None,
) -> tuple[tuple[int, ...], ...]:
    """Return the codes of each job, each job in grid order.

    Raises:
        ExecutionPlanningError: Neither or both of `n_jobs` and `assignment`
            are given, `n_jobs` is out of range, or the assignment names an
            unknown code, repeats or misses one, or holds an empty job.

    """
    if (n_jobs is None) == (assignment is None):
        msg = (
            "Give the jobs as n_jobs or assignment, not both and not neither: "
            f"n_jobs={n_jobs!r}, assignment={assignment!r}."
        )
        raise ExecutionPlanningError(msg)
    if n_jobs is not None:
        if type(n_jobs) is not int or not 1 <= n_jobs <= len(codes):
            msg = (
                f"n_jobs={n_jobs} must be at least one and at most {len(codes)}, "
                "the number of codes, so that every job holds a code."
            )
            raise ExecutionPlanningError(msg)
        return tuple(
            tuple(int(code) for code in part)
            for part in np.array_split(np.asarray(codes), n_jobs)
        )
    jobs = cast("tuple[tuple[int, ...], ...]", assignment)
    if any(type(code) is not int for job in jobs for code in job):
        msg = "Every assignment code must be an exact original integer code."
        raise ExecutionPlanningError(msg)
    failures = []
    unknown = sorted({code for job in jobs for code in job} - set(codes))
    if unknown:
        failures.append(f"codes {unknown!r} are not codes of the state {codes!r}")
    assigned = [code for job in jobs for code in job]
    repeated = sorted({code for code in assigned if assigned.count(code) > 1})
    if repeated:
        failures.append(f"codes {repeated!r} are assigned twice")
    missing = sorted(set(codes) - set(assigned))
    if missing:
        failures.append(f"codes {missing!r} are missing from every job")
    empty = [index for index, job in enumerate(jobs) if not job]
    if empty:
        failures.append(f"jobs {empty!r} are empty")
    if failures:
        msg = (
            "The assignment must give every code to exactly one nonempty job: "
            + "; ".join(failures)
            + "."
        )
        raise ExecutionPlanningError(msg)
    return tuple(tuple(code for code in codes if code in job) for job in jobs)


def _identity(*, model: Model, flat_params: FlatParams) -> dict[str, object]:
    """Return what every job and the collector must reproduce exactly."""
    execution = model._execution  # noqa: SLF001
    return {
        "model_fingerprint": model._model_fingerprint(flat_params=flat_params),  # noqa: SLF001
        "params_fingerprint": model._params_fingerprint(flat_params=flat_params),  # noqa: SLF001
        "flat_params_sha256": fingerprint_flat_params(flat_params),
        "program_fingerprint": model._program_fingerprint(flat_params=flat_params),  # noqa: SLF001
        "precision": "float64" if jax.config.read("jax_enable_x64") else "float32",
        "pylcm_version": _version.__version__,
        "pylcm_source_sha256": _pylcm_source_sha256(),
        "solve_config": {
            "invariant_block_widths": dict(
                sorted(execution.invariant_block_widths.items())
            ),
            "invariant_block_schedule": execution.invariant_block_schedule.value,
            "sharded_states": sorted(execution.sharded_states),
            "axis_widths": dict(sorted(execution.axis_widths.items())),
            "axis_widths_by_regime": {
                regime: dict(sorted(widths.items()))
                for regime, widths in sorted(execution.axis_widths_by_regime.items())
            },
            "covered_axes": sorted(execution.covered_axes),
            "simulation_sharding": execution.simulation_sharding,
            "donate_buffers": execution.donate_buffers,
        },
    }


@functools.cache
def _pylcm_source_sha256() -> str:
    """Digest every Python source of the imported `lcm` and `_lcm` packages."""
    digest = hashlib.sha256()
    for package in (lcm, _lcm):
        root = Path(package.__file__).parent
        for path in sorted(root.rglob("*.py")):
            for part in (
                f"{package.__name__}/{path.relative_to(root).as_posix()}".encode(),
                path.read_bytes(),
            ):
                digest.update(len(part).to_bytes(8, byteorder="big"))
                digest.update(part)
    return digest.hexdigest()


def _get_model_components(*, model: Model) -> tuple[StateName, tuple[int, ...]]:
    """Return the actual model's blocked state and codes in grid order."""
    (state_name,) = model._execution.invariant_block_widths  # noqa: SLF001
    codes = tuple(
        component.code
        for component in invariant_components(
            regimes=model._regimes,  # noqa: SLF001
            state_name=state_name,
        )
    )
    return state_name, codes


def _fail_if_identity_differs(
    *, model: Model, plan: ComponentJobPlan, identity: Mapping[str, object], source: str
) -> None:
    """Refuse a model, parameters or build that do not reproduce the plan's."""
    state_name, codes = _get_model_components(model=model)
    if plan.state_name != state_name or plan.codes != codes:
        msg = (
            f"The component plan in {plan.directory} names state {plan.state_name!r} "
            f"and codes {plan.codes!r}, not the actual model's blocked state "
            f"{state_name!r} and canonical codes {codes!r}."
        )
        raise ExecutionPlanningError(msg)
    differing = sorted(
        key
        for key in set(plan.identity) | set(identity)
        if canonical_json(plan.identity.get(key)) != canonical_json(identity.get(key))
    )
    if differing:
        msg = (
            f"The model and parameters of {source} do not reproduce the plan in "
            f"{plan.directory}: {differing!r} differ. Run every job and the "
            "collector with the model, parameters, precision and pylcm build the "
            "plan was made with, or make a new plan."
        )
        raise ExecutionPlanningError(msg)


def _fail_if_population_differs(
    *,
    plan: ComponentJobPlan,
    initial_conditions: UserInitialConditions | pd.DataFrame | None,
) -> None:
    """Refuse a population the plan does not simulate."""
    if not plan.simulates:
        if initial_conditions is not None:
            msg = (
                "The plan is solve-only, so its jobs simulate nothing; omit "
                "initial_conditions, or make a simulating plan."
            )
            raise ExecutionPlanningError(msg)
        return
    if initial_conditions is None:
        msg = (
            "The plan simulates; pass the initial_conditions it was made with "
            "to every job."
        )
        raise ExecutionPlanningError(msg)
    if (
        _initial_conditions_sha256(initial_conditions=initial_conditions)
        != plan.initial_conditions_sha256
    ):
        msg = (
            "The initial_conditions differ from the population the plan was "
            "made with; pass that population to every job, or make a new plan."
        )
        raise ExecutionPlanningError(msg)


def _n_subjects(*, initial_conditions: UserInitialConditions | pd.DataFrame) -> int:
    """Return the number of subjects of a population."""
    if isinstance(initial_conditions, pd.DataFrame):
        return len(initial_conditions)
    return len(next(iter(initial_conditions.values())))


def _initial_conditions_sha256(
    *, initial_conditions: UserInitialConditions | pd.DataFrame
) -> str:
    """Digest a population: every column's name, dtype, shape and bytes."""
    columns: dict[str, np.ndarray]
    if isinstance(initial_conditions, pd.DataFrame):
        columns = {
            str(name): (
                np.asarray(column.to_numpy())
                if pd.api.types.is_numeric_dtype(column)
                else np.asarray(column.astype(str).to_numpy(), dtype=np.str_)
            )
            for name, column in initial_conditions.items()
        }
    else:
        columns = {
            str(name): np.asarray(jax.device_get(column))
            for name, column in initial_conditions.items()
        }
    digest = hashlib.sha256()
    for name in sorted(columns):
        array = np.ascontiguousarray(columns[name])
        for part in (
            name.encode(),
            array.dtype.str.encode(),
            canonical_json(list(array.shape)),
            array.tobytes(),
        ):
            digest.update(len(part).to_bytes(8, byteorder="big"))
            digest.update(part)
    return digest.hexdigest()


def _planned_rows(
    *,
    model: Model,
    jobs: tuple[tuple[int, ...], ...],
    codes: tuple[int, ...],
    initial_conditions: InitialConditions,
    n_subjects: int,
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    """Bind each job to the canonical population's rows; count each code's rows.

    Returns:
        The digest of each job's original rows, and the number of subjects
        holding each code, in grid order.

    """
    route = next(iter(model._regimes.values())).simulation.programs.grouping  # noqa: SLF001
    if route is None:
        msg = (
            "Simulating component jobs require the model's invariant subject grouping."
        )
        raise ExecutionPlanningError(msg)
    if tuple(route.codes) != codes:
        msg = (
            f"The subject grouping orders the codes {tuple(route.codes)!r}, not "
            f"the grid order {codes!r}."
        )
        raise ExecutionPlanningError(msg)
    population_codes = (
        np.asarray(jax.device_get(initial_conditions[route.state_name]))
        if route.state_name in initial_conditions
        else None
    )
    keys = group_codes(route=route, codes=population_codes, n_real=n_subjects)
    digests = tuple(
        _job_row_checksum(
            job=job,
            codes=job_codes,
            rows=np.flatnonzero(np.isin(keys, job_codes)).astype(np.int64),
        )
        for job, job_codes in enumerate(jobs)
    )
    counts = group_sizes(route=route, codes=population_codes, n_real=n_subjects)
    return digests, counts


def _job_row_checksum(*, job: int, codes: tuple[int, ...], rows: np.ndarray) -> str:
    """Checksum the ordered original rows owned by one planned job."""
    return array_checksum(
        identity={"kind": "component-job-rows", "job": job, "codes": list(codes)},
        array=rows,
    )


def _execution_record(*, model: Model, log_level: LogLevel) -> dict[str, object]:
    """Record the actual compiler, execution mode, native build and device budget."""
    execution = model._execution  # noqa: SLF001
    devices = [device for device in jax.devices() if device.id in execution.device_ids]
    native_manifest = _installed_native_directory() / "native-manifest.json"
    return {
        "backend": jax.default_backend(),
        "device_kinds": sorted({device.device_kind for device in devices}),
        "n_devices": len(execution.device_ids),
        "device_memory_bytes": execution.device_memory_bytes,
        "jax_version": jax.__version__,
        "jaxlib_version": jaxlib.__version__,
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "log_level": log_level,
        "enable_jit": model.enable_jit,
        "jax_disable_jit": bool(jax.config.jax_disable_jit),
        "jax_default_prng_impl": jax.config.jax_default_prng_impl,
        "jax_random_seed_offset": jax.config.jax_random_seed_offset,
        "jax_threefry_partitionable": jax.config.jax_threefry_partitionable,
        "action_partitions": dict(sorted(execution.action_partitions.items())),
        "native_build_fingerprint": (
            json.loads(native_manifest.read_bytes())["fingerprint"]
            if native_manifest.is_file()
            else None
        ),
    }


def _solve_job(
    *,
    model: Model,
    flat_params: FlatParams,
    codes: tuple[int, ...],
    log_level: LogLevel,
    max_compilation_workers: int | None,
) -> RetainedComponentValues:
    """Solve the job's codes, each through every period, and retain them."""
    log = get_logger(log_level=log_level)
    validate_transitions(
        regimes=model._regimes,  # noqa: SLF001
        flat_params=flat_params,
        ages=model.ages,
        logger=log,
        process_grid_resolver=None,
    )
    preparation = model._prepare_solution(  # noqa: SLF001
        flat_params=flat_params,
        log=log,
        retention=ResultRetention.VALUES_AND_REPLAY,
        process_grid_resolver=None,
        call_id=None,
    )
    with selected_components(codes=codes):
        schedule = model._component_schedule(  # noqa: SLF001
            preparation=preparation,
            flat_params=flat_params,
            log=log,
            max_compilation_workers=max_compilation_workers,
            process_grid_resolver=None,
            call_id=None,
        )
    for component in schedule.components:
        blocks = schedule.solve_component(code=component.code)
        schedule.retain_component(code=component.code, blocks=blocks)
    return schedule.retained


def _simulate_job(
    *,
    model: Model,
    params: UserParams,
    plan: ComponentJobPlan,
    codes: tuple[int, ...],
    initial_conditions: UserInitialConditions | pd.DataFrame | None,
    log_level: LogLevel,
    max_compilation_workers: int | None,
) -> tuple[RetainedComponentValues, FragmentPanel]:
    """Solve the job's codes while simulating their subjects; return both."""
    with selected_components(codes=codes) as selection:
        result = model.simulate(
            params=params,
            initial_conditions=cast("UserInitialConditions", initial_conditions),
            solution=None,
            seed=plan.seed,
            log_level=log_level,
            max_compilation_workers=max_compilation_workers,
        )
    rows = result._subject_rows  # noqa: SLF001
    if rows is None:
        msg = "A selected simulation did not report the rows it simulated."
        raise ExecutionPlanningError(msg)
    raw_results = result.raw_results
    leaves = {
        address: leaf
        for regime, periods in raw_results.items()
        for period, data in periods.items()
        for address, leaf in _data_leaves(regime=regime, period=period, data=data)
    }
    devices = _selected_devices(model=model)
    # One transfer for every leaf; a tuple keeps the addresses in their order.
    host_leaves = jax.device_get(tuple(leaves.values()))
    return selection.retained, FragmentPanel(
        n_subjects=cast("int", plan.n_subjects),
        subject_batch_size=int(cast("int", result._subject_batch_size)),  # noqa: SLF001
        rows=np.asarray(rows, dtype=np.int64),
        regimes=tuple(
            (regime, tuple(int(period) for period in periods))
            for regime, periods in raw_results.items()
        ),
        leaves=MappingProxyType(
            {
                address: np.asarray(leaf)
                for address, leaf in zip(leaves, host_leaves, strict=True)
            }
        ),
        layouts=MappingProxyType(
            {
                address: MappingProxyType(_leaf_layout(leaf=leaf, devices=devices))
                for address, leaf in leaves.items()
            }
        ),
    )


def _data_leaves(
    *, regime: RegimeName, period: int, data: PeriodRegimeSimulationData
) -> tuple[tuple[LeafAddress, object], ...]:
    """Return every leaf of one regime-period's raw result with its address."""
    leaves: list[tuple[LeafAddress, object]] = []
    for field in fields(PeriodRegimeSimulationData):
        value = getattr(data, field.name)
        if isinstance(value, Mapping):
            leaves.extend(
                ((regime, int(period), field.name, str(key)), leaf)
                for key, leaf in value.items()
            )
        else:
            leaves.append(((regime, int(period), field.name, None), value))
    return tuple(leaves)


def _complete_fragments(
    *, plan: ComponentJobPlan, execution: Mapping[str, object]
) -> tuple[Fragment, ...]:
    """Return one verified fragment per job, in job order.

    Raises:
        ExecutionPlanningError: A job failed, is duplicated, is stale, is
            missing, or the jobs ran on different kinds of execution.
        SolutionIntegrityError: A fragment fails verification.

    """
    directory = plan.directory / FRAGMENT_DIRECTORY
    names = (
        sorted(path.name for path in directory.iterdir()) if directory.is_dir() else []
    )
    failed = sorted(
        job for name in names if (job := failure_job(name=name)) is not None
    )
    read = [
        (fragment_job(name=name), read_fragment(path=directory / name))
        for name in names
        if fragment_job(name=name) is not None
    ]
    problems: dict[str, list[str]] = {
        "failed": [],
        "duplicated": [],
        "stale": [],
        "missing": [],
    }
    by_job: dict[int, Fragment] = {}
    for filed_job, fragment in read:
        name = fragment.path.name
        if fragment.plan_sha256 != plan.digest or fragment.plan_id != plan.plan_id:
            problems["stale"].append(
                f"{name} belongs to another plan (job {filed_job})"
            )
            continue
        if canonical_json(dict(fragment.identity)) != canonical_json(
            dict(plan.identity)
        ):
            problems["stale"].append(
                f"{name} reproduces another identity (job {filed_job})"
            )
            continue
        if (
            not 0 <= fragment.job < len(plan.jobs)
            or fragment.codes != plan.jobs[fragment.job]
        ):
            problems["stale"].append(
                f"{name} covers codes {fragment.codes!r}, which are not those of "
                f"job {fragment.job}"
            )
            continue
        if fragment.job != filed_job or fragment.job in by_job:
            problems["duplicated"].append(
                f"{name} holds job {fragment.job}, whose codes "
                f"{fragment.codes!r} another fragment covers or should cover"
            )
            continue
        by_job[fragment.job] = fragment
    for job in failed:
        if job in by_job:
            problems["duplicated"].append(
                f"job {job} published a fragment and also recorded a failure; "
                f"rerun job {job} to replace both"
            )
            del by_job[job]
        else:
            problems["failed"].append(
                f"job {job} recorded a failure in {failure_name(job=job)}"
            )
    problems["missing"].extend(
        f"job {job} (codes {codes!r}) published no fragment"
        for job, codes in enumerate(plan.jobs)
        if job not in by_job and job not in failed
    )
    reported = [
        f"{kind}: " + "; ".join(entries)
        for kind, entries in problems.items()
        if entries
    ]
    if reported:
        msg = (
            f"The component jobs in {plan.directory} are incomplete, so no result "
            "is collected. "
            + ". ".join(reported)
            + ". Rerun the named jobs; a rerun replaces its fragment and failure "
            "record."
        )
        raise ExecutionPlanningError(msg)
    fragments = tuple(by_job[job] for job in range(len(plan.jobs)))
    _fail_if_execution_differs(fragments=fragments, execution=execution)
    return fragments


def _fail_if_execution_differs(
    *, fragments: tuple[Fragment, ...], execution: Mapping[str, object]
) -> None:
    """Require all workers to reproduce the collector's admitted execution."""
    executions = {
        canonical_json(dict(fragment.execution)): fragment.job for fragment in fragments
    }
    if len(executions) > 1:
        msg = (
            "The jobs ran on different kinds of execution, whose values need not "
            "agree bit for bit: "
            + "; ".join(
                f"job {fragment.job}: {dict(fragment.execution)!r}"
                for fragment in fragments
            )
            + ". Rerun the jobs on one kind of node."
        )
        raise ExecutionPlanningError(msg)
    if next(iter(executions)) != canonical_json(dict(execution)):
        msg = (
            "The jobs' execution does not reproduce the collector's compiler, "
            "logging, execution mode, native build or device configuration. "
            "Run every job and the collector with the same admitted execution."
        )
        raise ExecutionPlanningError(msg)


def _retain_fragments(
    *,
    retained: RetainedComponentValues,
    fragments: tuple[Fragment, ...],
    plan: ComponentJobPlan,
    value_dtypes: Mapping[Coordinate, str],
) -> None:
    """Hand every code's verified blocks to the retention, in grid order.

    Raises:
        SolutionIntegrityError: A block's dtype is not the model's value dtype.

    """
    # Fragment.values maps original codes to arrays, rather than pandas columns.
    blocks_by_code = {
        code: fragment.values[code]  # noqa: PD011
        for fragment in fragments
        for code in fragment.codes
    }
    for code in plan.codes:
        blocks = blocks_by_code[code]
        mismatched = [
            coordinate
            for coordinate, block in blocks.items()
            if str(block.dtype) != value_dtypes[coordinate]
        ]
        if mismatched:
            msg = (
                f"Code {code} holds values at {mismatched!r} whose dtype is not "
                "the model's value dtype at this precision."
            )
            raise SolutionIntegrityError(msg)
        retained.retain_host(code=code, blocks=blocks)


def _collected_simulation(
    *,
    model: Model,
    flat_params: FlatParams,
    plan: ComponentJobPlan,
    fragments: tuple[Fragment, ...],
    solution: SolutionResult,
    value_dtypes: Mapping[Coordinate, str],
) -> SimulationResult:
    """Restore every job's rows to one panel of the whole population.

    Raises:
        SolutionIntegrityError: A fragment holds no panel, the rows do not
            cover the population exactly once, or the jobs' panels disagree in
            structure or chunk width.

    """
    panels = [fragment.panel for fragment in fragments]
    if any(panel is None for panel in panels):
        msg = "Every fragment of a simulating plan holds a panel; one does not."
        raise SolutionIntegrityError(msg)
    panels = cast("list[FragmentPanel]", panels)
    job_rows_sha256 = cast("tuple[str, ...]", plan.job_rows_sha256)
    for fragment, panel in zip(fragments, panels, strict=True):
        if (
            _job_row_checksum(job=fragment.job, codes=fragment.codes, rows=panel.rows)
            != job_rows_sha256[fragment.job]
        ):
            msg = f"Job {fragment.job} holds rows other than its planned original rows."
            raise SolutionIntegrityError(msg)
        counts = cast("tuple[int, ...]", plan.code_counts)
        planned = sum(counts[plan.codes.index(code)] for code in fragment.codes)
        if len(panel.rows) != planned:
            msg = (
                f"Job {fragment.job} holds {len(panel.rows)} rows, not the "
                f"{planned} subjects the plan counts for codes {fragment.codes!r}."
            )
            raise SolutionIntegrityError(msg)
    n_subjects = cast("int", plan.n_subjects)
    rows = np.concatenate([panel.rows for panel in panels])
    covered = np.zeros(n_subjects, dtype=np.int64)
    np.add.at(covered, rows[(rows >= 0) & (rows < n_subjects)], 1)
    widths = {panel.subject_batch_size for panel in panels}
    if (
        {panel.n_subjects for panel in panels} != {n_subjects}
        or len(widths) != 1
        or len(rows) != n_subjects
        or not np.all(covered == 1)
    ):
        msg = (
            "The jobs' panels do not hold every subject of the population exactly "
            "once on one chunk width."
        )
        raise SolutionIntegrityError(msg)
    populated = [panel for panel in panels if len(panel.rows)]
    _validate_raw_fields(model=model, panels=populated, value_dtypes=value_dtypes)
    structures = {
        (
            panel.regimes,
            tuple(
                (address, leaf.dtype.str, leaf.shape[1:])
                for address, leaf in panel.leaves.items()
            ),
        )
        for panel in populated
    }
    if len(structures) != 1:
        msg = "The jobs' panels disagree in regimes, periods, leaves or dtypes."
        raise SolutionIntegrityError(msg)
    template = populated[0]
    shardings = _reference_shardings(
        model=model,
        plan=plan,
        width=next(iter(widths)),
        populated=populated,
    )
    fields_by_cell: dict[
        tuple[RegimeName, int], dict[str, dict[str | None, jax.Array]]
    ] = {}
    for address, leaf in template.leaves.items():
        regime, period, field_name, key = address
        full = np.empty((n_subjects, *leaf.shape[1:]), dtype=leaf.dtype)
        for panel in populated:
            full[panel.rows] = panel.leaves[address]
        fields_by_cell.setdefault((regime, period), {}).setdefault(field_name, {})[
            key
        ] = jax.device_put(full, shardings[address])
    raw_results = MappingProxyType(
        {
            regime: MappingProxyType(
                {
                    period: _rebuild_data(
                        fields_by_name=fields_by_cell.get((regime, period), {})
                    )
                    for period in periods
                }
            )
            for regime, periods in template.regimes
        }
    )
    regimes = model._regimes  # noqa: SLF001
    result = SimulationResult(
        raw_results=raw_results,
        regimes=regimes,
        flat_params=flat_params,
        period_to_regime_to_V_arr=solution.values,
        ages=model.ages,
        simulation_output_dtypes=model.simulation_output_dtypes,
        subject_batch_size=widths.pop(),
        nested_policy_regimes=frozenset(
            regime_name
            for regime_name, regime in regimes.items()
            if regime.simulation.replay_route.consumer_route == "nnbegm_nested"
        ),
    )
    result._solution = solution  # noqa: SLF001
    result._durable_identity = model.durable_identity  # noqa: SLF001
    return result


def _reference_shardings(
    *,
    model: Model,
    plan: ComponentJobPlan,
    width: int,
    populated: list[FragmentPanel],
) -> dict[LeafAddress, jax.sharding.Sharding]:
    """Return, per raw leaf, where the single-process simulation leaves it.

    The single-process simulation cuts each code's subjects into chunks of
    `width`. Several chunks are each moved to the host assembly device as they
    finish and assembled there; a single chunk stays on the layout it was
    computed on, which the one job holding every subject recorded.

    Raises:
        SolutionIntegrityError: A single chunk's recorded layout is missing, not
            on the selected devices, or does not fit its leaf.

    """
    n_chunks = sum(
        -(-count // width) for count in cast("tuple[int, ...]", plan.code_counts)
    )
    template = populated[0]
    if n_chunks > 1:
        host = chunk_host_device(
            subject_devices=simulation_subject_devices(
                regimes=model._regimes,  # noqa: SLF001
                device_ids=model._execution.device_ids,  # noqa: SLF001
            )
        )
        return {
            address: jax.sharding.SingleDeviceSharding(host)
            for address in template.leaves
        }
    if len(populated) != 1:
        msg = (
            f"{len(populated)} jobs hold subjects of a population the plan "
            "simulates in one chunk."
        )
        raise SolutionIntegrityError(msg)
    devices = _selected_devices(model=model)
    n_subjects = cast("int", plan.n_subjects)
    shardings: dict[LeafAddress, jax.sharding.Sharding] = {}
    for address, leaf in template.leaves.items():
        shape = (n_subjects, *leaf.shape[1:])
        try:
            sharding = _sharding_from_layout(
                layout=template.layouts[address], devices=devices
            )
            sharding.shard_shape(shape)
        except (KeyError, TypeError, ValueError, IndexError) as error:
            msg = (
                f"Raw leaf {address!r} of shape {shape} has no usable recorded "
                f"layout {dict(template.layouts.get(address, {}))!r}: {error}"
            )
            raise SolutionIntegrityError(msg) from error
        shardings[address] = sharding
    return shardings


def _selected_devices(*, model: Model) -> tuple[jax.Device, ...]:
    """Return the model's selected devices, in the order positions refer to."""
    return placed_devices_for_ids(
        submesh_device_ids=(),
        visible_device_ids=model._execution.device_ids,  # noqa: SLF001
    )


def _leaf_layout(*, leaf: object, devices: tuple[jax.Device, ...]) -> dict[str, object]:
    """Describe one raw leaf's layout in positions of the selected devices.

    A layout on devices outside the selection or of an unsupported sharding is
    described as such; collection refuses it where it would be needed.
    """
    if not isinstance(leaf, jax.Array):
        return {"kind": "unsupported", "sharding": type(leaf).__name__}
    sharding = leaf.sharding
    position = {device: index for index, device in enumerate(devices)}
    if not sharding.device_set <= set(devices):
        return {"kind": "outside_selection"}
    if isinstance(sharding, jax.sharding.SingleDeviceSharding):
        (device,) = sharding.device_set
        return {
            "kind": "single_device",
            "device": position[device],
            "memory_kind": sharding.memory_kind,
        }
    if isinstance(sharding, jax.sharding.NamedSharding):
        mesh = sharding.mesh
        spec: list[object] = []
        for entry in sharding.spec:
            if entry is None or isinstance(entry, str):
                spec.append(entry)
            elif isinstance(entry, tuple) and all(
                isinstance(name, str) for name in entry
            ):
                spec.append(list(entry))
            else:
                return {"kind": "unsupported", "sharding": repr(sharding)}
        return {
            "kind": "named",
            "mesh_shape": [int(size) for size in mesh.devices.shape],
            "mesh_devices": [position[device] for device in mesh.devices.flat],
            "axis_names": [str(name) for name in mesh.axis_names],
            "spec": spec,
            "memory_kind": sharding.memory_kind,
        }
    return {"kind": "unsupported", "sharding": type(sharding).__name__}


def _sharding_from_layout(
    *, layout: Mapping[str, object], devices: tuple[jax.Device, ...]
) -> jax.sharding.Sharding:
    """Rebuild a recorded layout on the selected devices.

    Raises:
        ValueError: The layout is not one `_leaf_layout` describes on the
            selected devices.

    """
    memory_kind = layout.get("memory_kind")
    if memory_kind is not None and type(memory_kind) is not str:
        msg = f"Memory kind {memory_kind!r} is not a string."
        raise ValueError(msg)
    if layout.get("kind") == "single_device":
        return jax.sharding.SingleDeviceSharding(
            devices[_device_position(value=layout["device"], devices=devices)],
            memory_kind=memory_kind,
        )
    if layout.get("kind") == "named":
        mesh_devices = [
            devices[_device_position(value=value, devices=devices)]
            for value in _require_exact_list(
                value=layout["mesh_devices"], label="mesh devices"
            )
        ]
        mesh_shape = tuple(
            _require_positive_exact_int(value=size, label="mesh shape")
            for size in _require_exact_list(
                value=layout["mesh_shape"], label="mesh shape"
            )
        )
        mesh = jax.sharding.Mesh(
            np.asarray(mesh_devices, dtype=object).reshape(mesh_shape),
            axis_names=tuple(
                str(name)
                for name in _require_exact_list(
                    value=layout["axis_names"], label="mesh axis names"
                )
            ),
        )
        spec = jax.sharding.PartitionSpec(
            *(
                tuple(entry) if type(entry) is list else entry
                for entry in _require_exact_list(
                    value=layout["spec"], label="partition spec"
                )
            )
        )
        return jax.sharding.NamedSharding(mesh, spec, memory_kind=memory_kind)
    msg = f"Layout {dict(layout)!r} cannot be placed on the selected devices."
    raise ValueError(msg)


def _device_position(*, value: object, devices: tuple[jax.Device, ...]) -> int:
    """Return a recorded device position after checking it names a selected device."""
    position = _require_nonnegative_exact_int(value=value, label="device position")
    if position >= len(devices):
        msg = f"Device position {position} is not one of {len(devices)} devices."
        raise ValueError(msg)
    return position


def _validate_raw_fields(
    *, model: Model, panels: list[FragmentPanel], value_dtypes: Mapping[Coordinate, str]
) -> None:
    """Require model-owned raw cells, names and fixed-field representations."""
    regimes = model._regimes  # noqa: SLF001
    forward = forward_regimes_by_period(regimes=regimes, n_periods=model.n_periods)
    expected_regimes = tuple(
        (name, tuple(period for period, active in enumerate(forward) if name in active))
        for name in regimes
    )
    expected: set[LeafAddress] = set()
    for period, active in enumerate(forward):
        for name, regime in active.items():
            expected.update(
                (name, period, field_name, None)
                for field_name in (
                    "V_arr",
                    "in_regime",
                    "own_stakeholder",
                    "nested_policy_fallback",
                )
            )
            expected.update(
                (name, period, "states", state)
                for state in regime.simulation.state_names
            )
            expected.update(
                (name, period, "actions", action)
                for action in regime.simulation.action_names
            )
    for panel in panels:
        if panel.regimes != expected_regimes or set(panel.leaves) != expected:
            msg = (
                "A fragment's raw regimes, periods or fields do not match the "
                "model's forward simulation outputs."
            )
            raise SolutionIntegrityError(msg)
        for (name, period, field_name, _), leaf in panel.leaves.items():
            regime = regimes[name]
            trailing = (
                (len(regime.stakeholders),)
                if field_name == "V_arr" and regime.stakeholders is not None
                else ()
            )
            dtype = (
                np.dtype(value_dtypes[(period, name)])
                if field_name == "V_arr"
                else np.dtype(np.int32)
                if field_name == "own_stakeholder"
                else np.dtype(np.bool_)
                if field_name in {"in_regime", "nested_policy_fallback"}
                else None
            )
            if leaf.shape != (len(panel.rows), *trailing) or (
                dtype is not None and leaf.dtype != dtype
            ):
                msg = (
                    f"Raw field {(name, period, field_name)!r} has shape "
                    f"{leaf.shape!r} and dtype {leaf.dtype!s}, incompatible with "
                    "the model's subject and stakeholder axes or field dtype."
                )
                raise SolutionIntegrityError(msg)


def _rebuild_data(
    *, fields_by_name: Mapping[str, Mapping[str | None, jax.Array]]
) -> PeriodRegimeSimulationData:
    """Rebuild one regime-period's raw result from its collected leaves.

    An array field has its leaf under the key `None`; a mapping field has one
    leaf per entry, in the entries' order, and none when it is empty.
    """
    values: dict[str, object] = {}
    for field in fields(PeriodRegimeSimulationData):
        keyed = fields_by_name.get(field.name, {})
        values[field.name] = (
            keyed[None]
            if None in keyed
            else MappingProxyType(cast("dict[str, jax.Array]", dict(keyed)))
        )
    return PeriodRegimeSimulationData(**values)  # ty: ignore[invalid-argument-type]
