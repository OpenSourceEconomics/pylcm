"""Stored outside reads when the regimes of an exit are sharded over devices.

A reader's kernel tabulates each destination its aligned outside reads take and
gathers from that table. These witnesses pin what that does on several devices, so
that later work cannot make a multi-device solve with exits worse than reading every
outside value at its point:

- a reader and a destination that both carry a sharded discrete state whose law of
  motion is not the identity run on disjoint device meshes; stored and point reads
  solve and agree with each other and with the one-device solve, the table reads the
  destination's continuation by the transfer the point read uses, it is replicated
  on the reader's devices, charged to the reader's cell and released with its kernel;
- a destination that prunes the reader's sharded model-level state, because it does
  not read it, takes the exit's projection of the states it carries and solves as on
  one device, with stored and with point reads;
- a sharded state that keeps its value on the reader or the destination is co-mapped
  in that regime's ordinary solve, which reads only each device's slice; the exit
  still solves as on one device, with stored and with point reads, because a query
  takes the state as an ordinary axis. Each of the reader's devices computes the
  whole grid of the state in the exit period and holds every next-period value that
  carries it whole, and the reader's published value keeps only each device's own
  slice. A projection that flips the fixed state and a reader that prunes it are
  covered;
- singles, with or without a state of their own, who marry by consent into a couple
  that shards a model-level state, flipping or fixed, and dissolves into them,
  neither single reading the state, solve as on one device;
- a simulation over subjects sharded across devices reproduces the unsharded one.

The witnesses run in child processes, because the placements under test only exist
where several devices do, and a topology is pinned before JAX initializes a backend.
"""

import functools
import gc
import json
import re
import subprocess
import sys
import weakref
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, NotRequired, TypedDict, overload

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

import tests.conftest
from _lcm.execution.output_layout import PlannedCore
from lcm import (
    DiscreteGrid,
    ExecutionConfig,
    Exit,
    ExitTarget,
    InitialNodes,
    Model,
    Periods,
    Regime,
    Unilateral,
    categorical,
    fixed_transition,
)
from lcm.exceptions import RegimeInitializationError
from lcm.solver_api import KernelOutput
from lcm.typing import (
    BoolND,
    DiscreteState,
    FloatND,
    IntND,
    RegimeName,
    ScalarInt,
    UserFunction,
)
from tests.solution.test_outside_value_truncation import marriage_model
from tests.solution.test_stored_outside_values import (
    _assert_agree,
    _keep_wealth,
    _wealth,
    _wealth_grid,
    _work,
)

_REPO_ROOT = Path(__file__).parent.parent

#: Devices the child process runs on: one two-device mesh per sharded regime.
_N_DEVICES = 4

_PARAMS = {"discount_factor": 0.9}

#: A device budget, so the planner admits every cell and charges its arrays.
_BUDGET = 2**30

#: The kind's law toward a regime's terminal successor: the identity, which makes a
#: sharded kind co-mapped, or a flip, which keeps it a device-local axis.
type _Law = Literal["fixed", "flip"]

#: A solve's value arrays as nested lists, by "<period>/<regime>".
type _ValueLists = dict[str, list[list[float]]]


@categorical(ordered=False)
class _RegimeId:
    source: ScalarInt
    destination: ScalarInt
    source_terminal: ScalarInt
    destination_terminal: ScalarInt


@categorical(ordered=False)
class _Kind:
    low: ScalarInt
    high: ScalarInt


def _source_utility(*, wealth: FloatND, kind: DiscreteState) -> FloatND:
    return 1.0 + wealth + 0.6 * kind


def _blind_source_utility(*, wealth: FloatND) -> FloatND:
    return 1.0 + wealth


def _destination_utility(
    *, wealth: FloatND, work: IntND, kind: DiscreteState
) -> FloatND:
    return 2.0 * wealth + work * (0.9 + 0.1 * kind) + (1 - work) * 0.6


def _blind_destination_utility(*, wealth: FloatND, work: IntND) -> FloatND:
    return 2.0 * wealth + work * 0.9 + (1 - work) * 0.6


def _flip(kind: DiscreteState) -> DiscreteState:
    return 1 - kind


def _kind(kind: DiscreteState) -> DiscreteState:
    return kind


def _low_kind() -> DiscreteState:
    return jnp.asarray(_Kind.low)


def _kind_law(*, law: _Law, target: RegimeName) -> dict[RegimeName, UserFunction]:
    return {target: fixed_transition("kind") if law == "fixed" else _flip}


def build_model(
    *,
    devices: tuple[int, ...],
    sharded: tuple[str, ...] = ("kind",),
    reader_law: _Law = "flip",
    destination_law: _Law | None = "flip",
    project_kind: bool = True,
    kind_projection: Literal["keep", "flip"] = "keep",
    reader_reads_kind: bool = True,
    simulation_sharding: Literal["legacy", "subjects"] = "legacy",
) -> Model:
    """Build a source that may leave for a destination in the first of two periods.

    Both carry wealth onto the grid of their terminal successor. The source reads a
    model-level discrete `kind` unless `reader_reads_kind` is False; the destination
    reads it too unless `destination_law` is None. A regime that does not read the
    kind prunes it. The exit projects every state the destination declares, the
    model-level kind included, unless `project_kind` is False: a reader that reads
    the kind keeps it or flips it, one that prunes it lands on the low kind. With
    the kind kept, leaving pays where `1.9 * wealth > 0.64 + 0.05 * kind`: at wealth
    0.5 and 1, never at 0.

    Args:
        devices: Device ids the model is planned for.
        sharded: Model-level states given a device axis.
        reader_law: Law of the source's kind toward its successor.
        destination_law: Law of the destination's kind toward its successor, or
            None for a destination that does not read the kind.
        project_kind: Whether the exit projects the kind as well as wealth.
        kind_projection: Whether the exit keeps the reader's kind or flips it.
        reader_reads_kind: Whether the source reads the kind.
        simulation_sharding: How the simulated subjects are spread over devices.

    Returns:
        The model.

    """
    utility = (
        _blind_destination_utility if destination_law is None else _destination_utility
    )
    destination_kind = (
        {}
        if destination_law is None
        else {"kind": _kind_law(law=destination_law, target="destination_terminal")}
    )
    source_utility = _source_utility if reader_reads_kind else _blind_source_utility
    source_kind = (
        {"kind": _kind_law(law=reader_law, target="source_terminal")}
        if reader_reads_kind
        else {}
    )
    projected_kind = (
        (_kind if kind_projection == "keep" else _flip)
        if reader_reads_kind
        else _low_kind
    )
    return Model(
        regimes={
            "source": Regime(
                states={"wealth": _wealth_grid()},
                functions={"utility": source_utility},
                state_transitions={"wealth": _keep_wealth, **source_kind},
                exit=Exit(
                    targets={
                        "self": ExitTarget(
                            regime="destination",
                            projection={"wealth": _wealth}
                            | ({"kind": projected_kind} if project_kind else {}),
                        )
                    },
                    rule=Unilateral(),
                ),
            ),
            "destination": Regime(
                states={"wealth": _wealth_grid()},
                actions={"work": _work()},
                functions={"utility": utility},
                state_transitions={"wealth": _keep_wealth, **destination_kind},
            ),
            "source_terminal": Regime(
                states={"wealth": _wealth_grid()},
                functions={"utility": source_utility},
            ),
            "destination_terminal": Regime(
                states={"wealth": _wealth_grid()},
                actions={"work": _work()},
                functions={"utility": utility},
            ),
        },
        states={"kind": DiscreteGrid(category_class=_Kind)},
        regime_id_class=_RegimeId,
        n_periods=2,
        edges={
            name: {f"{name}_terminal": Periods(values=(0,))}
            for name in ("source", "destination")
        },
        initial_nodes=InitialNodes(by_period={0: ("source", "destination")}),
        execution_config=ExecutionConfig(
            devices=devices,
            sharded_states=sharded,
            simulation_sharding=simulation_sharding,
            device_memory_bytes=_BUDGET,
        ),
    )


class _Laws(TypedDict):
    """How the reader and the destination treat the kind, as `build_model` takes it."""

    reader_law: _Law
    destination_law: _Law | None
    kind_projection: NotRequired[Literal["keep", "flip"]]
    reader_reads_kind: NotRequired[bool]


_DEVICE_LOCAL: _Laws = {"reader_law": "flip", "destination_law": "flip"}

#: Exits whose reader or destination keeps a sharded kind fixed, so that the kind
#: is co-mapped in that regime's own solve.
_FIXED_CASES: dict[str, _Laws] = {
    "fixed reader, pruning destination": {
        "reader_law": "fixed",
        "destination_law": None,
    },
    "fixed reader, flipping destination": {
        "reader_law": "fixed",
        "destination_law": "flip",
    },
    "flipping reader, fixed destination": {
        "reader_law": "flip",
        "destination_law": "fixed",
    },
    "fixed reader, fixed destination": {
        "reader_law": "fixed",
        "destination_law": "fixed",
    },
    "fixed reader, fixed destination, flipping projection": {
        "reader_law": "fixed",
        "destination_law": "fixed",
        "kind_projection": "flip",
    },
    "pruning reader, fixed destination": {
        "reader_law": "fixed",
        "destination_law": "fixed",
        "reader_reads_kind": False,
    },
}


class TransferRead(TypedDict):
    core: str
    target_regime: RegimeName
    kind: str


class SolveArm(TypedDict):
    values: _ValueLists
    devices: dict[RegimeName, list[int]]
    replicated: dict[RegimeName, bool]
    shard_shapes: dict[RegimeName, list[list[int]]]
    tables: list[str]
    reads: list[TransferRead]
    reader_reservation: int
    table_bytes: int
    table_devices: list[list[int]]
    tables_replicated: list[bool]
    tables_released: list[bool]


class DeviceLocalReport(TypedDict):
    point: SolveArm
    stored: SolveArm
    one_device_point: SolveArm


class FixedCaseReport(TypedDict):
    point: SolveArm
    stored: SolveArm
    one_device_point: SolveArm
    one_device_stored: SolveArm


class PrunedReport(TypedDict):
    point: _ValueLists
    stored: _ValueLists
    one_device_point: _ValueLists
    one_device_stored: _ValueLists


class MarriageReport(TypedDict):
    sharded: _ValueLists
    one_device: _ValueLists


class SimulationReport(TypedDict):
    regimes: list[str]
    subjects_match_unsharded: str
    stored_matches_point: str


def _after[**P, R](
    *, function: Callable[P, R], then: Callable[[R], None]
) -> Callable[P, R]:
    """Wrap `function` so each result is handed to `then` before it is returned."""

    def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        result = function(*args, **kwargs)
        then(result)
        return result

    return wrapped


@dataclass(kw_only=True)
class _TableRecord:
    """Where each table array a solve built lives, and whether it was released."""

    nbytes: list[int] = field(default_factory=list)
    devices: list[list[int]] = field(default_factory=list)
    replicated: list[bool] = field(default_factory=list)
    released: list[bool] = field(default_factory=list)
    pending: list[weakref.ref[jax.Array]] = field(default_factory=list)
    """Arrays built by the kernel call that is running."""

    def built(self, tables: Mapping[str, Mapping[str, FloatND | BoolND]]) -> None:
        for array in (array for table in tables.values() for array in table.values()):
            self.nbytes.append(array.nbytes)
            self.devices.append(sorted(device.id for device in array.devices()))
            self.replicated.append(array.sharding.is_fully_replicated)
            self.pending.append(weakref.ref(array))

    def returned(self, output: KernelOutput) -> None:
        del output
        gc.collect()
        self.released.extend(ref() is None for ref in self.pending)
        self.pending.clear()


def _solve_arm(*, stored: bool, devices: tuple[int, ...], laws: _Laws) -> SolveArm:
    """Solve the model with stored or point reads and record its plan."""
    from _lcm import exit_queries, policy_queries  # noqa: PLC0415
    from _lcm.solution import backward_induction  # noqa: PLC0415

    cores: list[PlannedCore] = []
    reservations: list[Mapping[tuple[RegimeName, int], int]] = []
    tables = _TableRecord()
    kernel_class = policy_queries._PolicyQueryKernel
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(exit_queries, "_READ_STORED_OUTSIDE_VALUES", stored)
        patch.setattr(
            backward_induction,
            "_attach_resolved_output_layout",
            _after(
                function=backward_induction._attach_resolved_output_layout,
                then=cores.append,
            ),
        )
        patch.setattr(
            backward_induction,
            "_internal_reservations_by_cell",
            _after(
                function=backward_induction._internal_reservations_by_cell,
                then=reservations.append,
            ),
        )
        patch.setattr(
            kernel_class,
            "_tables",
            _after(function=kernel_class._tables, then=tables.built),
        )
        patch.setattr(
            kernel_class,
            "__call__",
            _after(function=kernel_class.__call__, then=tables.returned),
        )
        model = build_model(
            devices=devices, sharded=("kind",) if len(devices) > 1 else (), **laws
        )
        kernel = model._regimes["source"].solution.period_kernels[0]
        assert isinstance(kernel, kernel_class)
        solution = model.solve(params=_PARAMS, log_level="off", runtime_checks=False)
    table_programs = [
        program
        for name, program in kernel.programs.items()
        if name.startswith(policy_queries.OUTSIDE_PROGRAM)
    ]
    return {
        "values": {
            f"{period}/{regime}": np.asarray(value).tolist()
            for period, by_regime in solution.values.items()
            for regime, value in by_regime.items()
        },
        "devices": {
            regime: sorted(device.id for device in value.sharding.device_set)
            for regime, value in solution.values[0].items()
        },
        "replicated": {
            regime: value.sharding.is_fully_replicated
            for regime, value in solution.values[0].items()
        },
        "shard_shapes": {
            regime: [list(shard.data.shape) for shard in value.addressable_shards]
            for regime, value in solution.values[0].items()
        },
        "tables": sorted(program.name for program in table_programs),
        "reads": [
            {
                "core": transfer.source.core_key,
                "target_regime": transfer.target.regime,
                "kind": str(transfer.kind),
            }
            for core in cores
            for transfer in core.input_transfer_plan
            if transfer.source.source_regime == "source"
        ],
        "reader_reservation": int(reservations[-1][("source", 0)]),
        "table_bytes": sum(tables.nbytes),
        "table_devices": tables.devices,
        "tables_replicated": tables.replicated,
        "tables_released": tables.released,
    }


def report_device_local() -> DeviceLocalReport:
    """Solve the device-local model with both reads on four devices and on one."""
    devices = tuple(range(_N_DEVICES))
    return {
        "point": _solve_arm(stored=False, devices=devices, laws=_DEVICE_LOCAL),
        "stored": _solve_arm(stored=True, devices=devices, laws=_DEVICE_LOCAL),
        "one_device_point": _solve_arm(stored=False, devices=(0,), laws=_DEVICE_LOCAL),
    }


def report_fixed_case(*, case: str) -> FixedCaseReport:
    """Solve one fixed-kind case with both reads on four devices and on one."""
    laws = _FIXED_CASES[case]
    devices = tuple(range(_N_DEVICES))
    return {
        "point": _solve_arm(stored=False, devices=devices, laws=laws),
        "stored": _solve_arm(stored=True, devices=devices, laws=laws),
        "one_device_point": _solve_arm(stored=False, devices=(0,), laws=laws),
        "one_device_stored": _solve_arm(stored=True, devices=(0,), laws=laws),
    }


def _pruned_destination_values(
    *, stored: bool, devices: tuple[int, ...]
) -> _ValueLists:
    """Solve the model whose destination prunes the kind; its values by key."""
    from _lcm import exit_queries  # noqa: PLC0415

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(exit_queries, "_READ_STORED_OUTSIDE_VALUES", stored)
        solution = build_model(
            devices=devices,
            sharded=("kind",) if len(devices) > 1 else (),
            destination_law=None,
        ).solve(params=_PARAMS, log_level="off", runtime_checks=False)
    return {
        f"{period}/{regime}": np.asarray(value).tolist()
        for period, by_regime in solution.values.items()
        for regime, value in by_regime.items()
    }


def report_pruned_destination() -> PrunedReport:
    """Solve the pruned-destination model with both reads on four devices and on one."""
    devices = tuple(range(_N_DEVICES))
    return {
        "point": _pruned_destination_values(stored=False, devices=devices),
        "stored": _pruned_destination_values(stored=True, devices=devices),
        "one_device_point": _pruned_destination_values(stored=False, devices=(0,)),
        "one_device_stored": _pruned_destination_values(stored=True, devices=(0,)),
    }


def _marriage_values(
    *,
    devices: tuple[int, ...],
    single_states: bool,
    match_law: Literal["flip", "fixed"] = "flip",
    stored: bool = True,
) -> _ValueLists:
    """Solve the marriage whose couple alone reads the model-level match."""
    from _lcm import exit_queries  # noqa: PLC0415

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(exit_queries, "_READ_STORED_OUTSIDE_VALUES", stored)
        solution = marriage_model(
            model_level=True,
            single_states=single_states,
            match_law=match_law,
            execution_config=ExecutionConfig(
                devices=devices,
                sharded_states=("match",) if len(devices) > 1 else (),
                device_memory_bytes=_BUDGET,
            ),
        ).solve(params=_PARAMS, log_level="off", runtime_checks=False)
    return {
        f"{period}/{regime}": np.asarray(value).tolist()
        for period, by_regime in solution.values.items()
        for regime, value in by_regime.items()
    }


def report_marriage(*, single_states: bool) -> MarriageReport:
    """Solve the marriage with the couple's match sharded on four devices and on one."""
    return {
        "sharded": _marriage_values(
            devices=tuple(range(_N_DEVICES)), single_states=single_states
        ),
        "one_device": _marriage_values(devices=(0,), single_states=single_states),
    }


def report_fixed_marriage() -> PrunedReport:
    """Solve the marriage whose couple keeps its sharded match fixed, singles with
    states, with both reads on four devices and on one.
    """
    solve = functools.partial(_marriage_values, single_states=True, match_law="fixed")
    devices = tuple(range(_N_DEVICES))
    return {
        "point": solve(devices=devices, stored=False),
        "stored": solve(devices=devices, stored=True),
        "one_device_point": solve(devices=(0,), stored=False),
        "one_device_stored": solve(devices=(0,), stored=True),
    }


#: Subjects start in the source at every wealth point and both kinds.
_INITIAL_WEALTH = (0.0, 0.5, 1.0, 0.0, 0.5, 1.0)
_INITIAL_KIND = (0, 0, 0, 1, 1, 1)


def _simulate(
    *,
    stored: bool,
    devices: tuple[int, ...],
    subjects: bool,
    sharded: tuple[str, ...] = (),
    laws: _Laws = _DEVICE_LOCAL,
) -> pd.DataFrame:
    """Solve with stored or point reads, then simulate the subjects."""
    from _lcm import exit_queries  # noqa: PLC0415

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(exit_queries, "_READ_STORED_OUTSIDE_VALUES", stored)
        model = build_model(
            devices=devices,
            sharded=sharded,
            simulation_sharding="subjects" if subjects else "legacy",
            **laws,
        )
        solution = model.solve(params=_PARAMS, log_level="off", runtime_checks=False)
    n_subjects = len(_INITIAL_WEALTH)
    return model.simulate(
        params=_PARAMS,
        initial_conditions={
            "period": jnp.zeros(n_subjects, dtype=jnp.int32),
            "regime_id": jnp.full(n_subjects, _RegimeId.source, dtype=jnp.int32),
            "wealth": jnp.asarray(_INITIAL_WEALTH),
            "kind": jnp.asarray(_INITIAL_KIND, dtype=jnp.int32),
        },
        solution=solution,
        log_level="off",
        seed=7,
    ).to_dataframe(use_categorical_codes=True)


def _mismatch(*, check: Callable[[], None]) -> str:
    """The message of a failed frame comparison, or an empty string."""
    try:
        check()
    except AssertionError as mismatch:
        return str(mismatch)
    return ""


def report_subject_sharded_simulation(*, decimal: int) -> SimulationReport:
    """Simulate stored-read solutions over sharded and unsharded subjects."""
    sharded = _simulate(stored=True, devices=tuple(range(_N_DEVICES)), subjects=True)
    unsharded = _simulate(stored=True, devices=(0,), subjects=False)
    point = _simulate(stored=False, devices=(0,), subjects=False)
    tolerance = 10.0**-decimal
    first = sharded.loc[sharded["period"] == 0].sort_values("subject_id")
    return {
        "regimes": [str(name) for name in first["regime_name"]],
        "subjects_match_unsharded": _mismatch(
            check=lambda: pd.testing.assert_frame_equal(
                sharded, unsharded, check_exact=True
            )
        ),
        "stored_matches_point": _mismatch(
            check=lambda: pd.testing.assert_frame_equal(
                sharded, point, rtol=tolerance, atol=tolerance
            )
        ),
    }


def report_fixed_reader_simulation(*, decimal: int) -> dict[str, str]:
    """Simulate a fixed reader's solves sharded on four devices against one device.

    The value is each four-device read's mismatch with point reads on one device.
    """
    laws = _FIXED_CASES["fixed reader, pruning destination"]
    one_device = _simulate(stored=False, devices=(0,), subjects=False, laws=laws)
    tolerance = 10.0**-decimal
    return {
        name: _mismatch(
            check=functools.partial(
                pd.testing.assert_frame_equal,
                _simulate(
                    stored=stored,
                    devices=tuple(range(_N_DEVICES)),
                    subjects=False,
                    sharded=("kind",),
                    laws=laws,
                ),
                one_device,
                rtol=tolerance,
                atol=tolerance,
            )
        )
        for name, stored in (("point", False), ("stored", True))
    }


@overload
def _run_in_child_process(
    *, entry_point: Literal["report_device_local"], arguments: str = ""
) -> DeviceLocalReport: ...


@overload
def _run_in_child_process(
    *, entry_point: Literal["report_fixed_case"], arguments: str = ""
) -> FixedCaseReport: ...


@overload
def _run_in_child_process(
    *, entry_point: Literal["report_subject_sharded_simulation"], arguments: str = ""
) -> SimulationReport: ...


@overload
def _run_in_child_process(
    *, entry_point: Literal["report_fixed_reader_simulation"], arguments: str = ""
) -> dict[str, str]: ...


@overload
def _run_in_child_process(
    *,
    entry_point: Literal["report_pruned_destination", "report_fixed_marriage"],
    arguments: str = "",
) -> PrunedReport: ...


@overload
def _run_in_child_process(
    *, entry_point: Literal["report_marriage"], arguments: str = ""
) -> MarriageReport: ...


def _run_in_child_process(
    *, entry_point: str, arguments: str = ""
) -> (
    DeviceLocalReport
    | FixedCaseReport
    | SimulationReport
    | PrunedReport
    | MarriageReport
    | dict[str, str]
):
    """Run one module-level report function on four CPU devices and return its report.

    The child carries this run's float policy, because a topology pin needs a fresh
    process and a fresh process reads none of pytest's options.

    Args:
        entry_point: Name of a function in this module returning a
            JSON-serializable mapping.
        arguments: Call arguments for the entry point, as source text.

    Returns:
        Dictionary of the report the child process produced.

    """
    code = (
        "import json, sys; import jax; "
        f"jax.config.update('jax_num_cpu_devices', {_N_DEVICES}); "
        "jax.config.update('jax_platform_name', 'cpu'); "
        f"jax.config.update('jax_enable_x64', {tests.conftest.X64_ENABLED!r}); "
        "jax.config.update('jax_default_matmul_precision', 'highest'); "
        "from tests.test_outside_values_sharded_state import "
        f"{entry_point} as entry; "
        f"sys.stdout.write('@@' + json.dumps(entry({arguments})) + '@@')"
    )
    completed = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        cwd=_REPO_ROOT,
    )
    assert completed.returncode == 0, completed.stderr
    _, _, rest = completed.stdout.partition("@@")
    payload, _, _ = rest.partition("@@")
    assert payload, completed.stdout
    return json.loads(payload)


@pytest.fixture(scope="module")
def device_local() -> DeviceLocalReport:
    """Return the device-local solves from one four-device child process."""
    return _run_in_child_process(entry_point="report_device_local")


@functools.cache
def _fixed_case(case: str) -> FixedCaseReport:
    """Return one fixed-kind case's solves from one four-device child process."""
    return _run_in_child_process(
        entry_point="report_fixed_case", arguments=f"case={case!r}"
    )


@pytest.fixture(scope="module")
def pruned_destination() -> PrunedReport:
    """Return the pruned-destination solves from one four-device child process."""
    return _run_in_child_process(entry_point="report_pruned_destination")


@pytest.fixture(scope="module")
def fixed_marriage() -> PrunedReport:
    """Return the fixed-match marriage solves from one four-device child process."""
    return _run_in_child_process(entry_point="report_fixed_marriage")


@pytest.fixture(scope="module")
def subject_sharded() -> SimulationReport:
    """Return the subject-sharded simulation from one four-device child process."""
    return _run_in_child_process(
        entry_point="report_subject_sharded_simulation",
        arguments=f"decimal={tests.conftest.DECIMAL_PRECISION!r}",
    )


def _values(values: _ValueLists) -> dict[str, np.ndarray]:
    dtype = np.float64 if tests.conftest.X64_ENABLED else np.float32
    return {key: np.asarray(value, dtype=dtype) for key, value in values.items()}


def test_the_reader_and_its_destination_run_on_disjoint_meshes(
    device_local: DeviceLocalReport,
) -> None:
    """Each regime shards the kind over its own pair of devices, so every outside
    read crosses from one mesh to the other.
    """
    assert device_local["stored"]["devices"] == {
        "source": [0, 1],
        "destination": [2, 3],
    }


def test_only_stored_reads_tabulate_the_destination(
    device_local: DeviceLocalReport,
) -> None:
    """The sharded reader takes its aligned read of the sharded destination from a
    table; with point reads it builds none.
    """
    assert (device_local["point"]["tables"], device_local["stored"]["tables"]) == (
        [],
        ["_outside:destination"],
    )


def test_sharded_stored_reads_agree_with_sharded_point_reads(
    device_local: DeviceLocalReport,
) -> None:
    """On four devices, stored reads give point reads' values to 16 ULP."""
    _assert_agree(
        got=_values(device_local["stored"]["values"]),
        expected=_values(device_local["point"]["values"]),
    )


def test_sharded_stored_reads_agree_with_the_one_device_solve(
    device_local: DeviceLocalReport,
) -> None:
    """On four devices, stored reads give the values of point reads on one device
    to 16 ULP.
    """
    _assert_agree(
        got=_values(device_local["stored"]["values"]),
        expected=_values(device_local["one_device_point"]["values"]),
    )


def _destination_reads(arm: SolveArm) -> list[tuple[str, str]]:
    return sorted(
        (read["core"], read["kind"])
        for read in arm["reads"]
        if read["target_regime"] == "destination_terminal"
    )


def test_point_reads_take_the_destination_continuation_across_meshes(
    device_local: DeviceLocalReport,
) -> None:
    """The reader's chunk loop reads the destination's next-period value from the
    other mesh by a cross-mesh copy.
    """
    assert _destination_reads(device_local["point"]) == [("_chunks", "cross_mesh_copy")]


def test_the_table_takes_the_destination_continuation_by_the_same_transfer(
    device_local: DeviceLocalReport,
) -> None:
    """With stored reads the table, and not the chunk loop, reads the destination's
    next-period value, by the cross-mesh copy the point read uses.
    """
    assert _destination_reads(device_local["stored"]) == [
        ("_outside:destination", "cross_mesh_copy")
    ]


def test_a_table_is_replicated_on_the_reader_s_devices(
    device_local: DeviceLocalReport,
) -> None:
    """Every array of the table, values and support, lives whole on each device of
    the reader's mesh.
    """
    stored = device_local["stored"]
    assert (stored["table_devices"], stored["tables_replicated"]) == (
        [[0, 1], [0, 1]],
        [True, True],
    )


def test_the_planner_charges_a_table_s_bytes_to_the_reader_s_cell(
    device_local: DeviceLocalReport,
) -> None:
    """The reader's cell reserves the table's values and support beyond what point
    reads reserve there, at the size the table program declares.
    """
    stored, point = device_local["stored"], device_local["point"]
    assert stored["table_bytes"] > 0
    assert (
        stored["reader_reservation"] - point["reader_reservation"]
        == stored["table_bytes"]
    )


def test_a_table_is_released_when_its_kernel_returns(
    device_local: DeviceLocalReport,
) -> None:
    """No array of a table outlives the reader's kernel call that built it."""
    assert device_local["stored"]["tables_released"] == [True, True]


@pytest.mark.parametrize("arm", ["point", "stored", "one_device_stored"])
def test_a_pruned_destination_solves_as_point_reads_on_one_device(
    *,
    arm: Literal["point", "stored", "one_device_stored"],
    pruned_destination: PrunedReport,
) -> None:
    """A destination that does not read the reader's sharded model-level kind prunes
    it, and the exit, which projects the kind too, solves on four devices and with
    stored reads as with point reads on one device, to 16 ULP.
    """
    _assert_agree(
        got=_values(pruned_destination[arm]),
        expected=_values(pruned_destination["one_device_point"]),
    )


@pytest.mark.parametrize("arm", ["point", "stored", "one_device_stored"])
@pytest.mark.parametrize("case", list(_FIXED_CASES))
def test_a_fixed_sharded_kind_solves_as_point_reads_on_one_device(
    *, case: str, arm: Literal["point", "stored", "one_device_stored"]
) -> None:
    """An exit whose reader or destination keeps the sharded kind fixed solves on
    four devices, with stored and with point reads, as with point reads on one
    device, to 16 ULP.
    """
    report = _fixed_case(case)
    _assert_agree(
        got=_values(report[arm]["values"]),
        expected=_values(report["one_device_point"]["values"]),
    )


@pytest.mark.parametrize("arm", ["point", "stored"])
@pytest.mark.parametrize(
    "case",
    [
        name
        for name, laws in _FIXED_CASES.items()
        if laws.get("reader_reads_kind", True)
    ],
)
def test_the_reader_publishes_its_value_split_along_the_kind(
    *, case: str, arm: Literal["point", "stored"]
) -> None:
    """The reader's exit-period value lives on the reader's two devices, each
    holding one kind and every wealth point; the query evaluates the whole grid
    inside the reader's program and no device keeps the other's kind.
    """
    solved = _fixed_case(case)[arm]
    assert (
        solved["devices"]["source"],
        solved["replicated"]["source"],
        solved["shard_shapes"]["source"],
    ) == ([0, 1], False, [[1, 3], [1, 3]])


@pytest.mark.parametrize("case", list(_FIXED_CASES))
def test_a_fixed_case_s_table_is_replicated_on_the_reader_s_devices(
    case: str,
) -> None:
    """With stored reads the reader tabulates the destination once, and every
    array of the table lives whole on each device of the reader's mesh.
    """
    stored = _fixed_case(case)["stored"]
    reader_devices = stored["devices"]["source"]
    assert (
        stored["tables"],
        stored["table_devices"],
        stored["tables_replicated"],
    ) == (["_outside:destination"], [reader_devices] * 2, [True, True])


@pytest.mark.parametrize(
    "case",
    [
        name
        for name, laws in _FIXED_CASES.items()
        if laws["destination_law"] and laws.get("reader_reads_kind", True)
    ],
)
def test_a_destination_carrying_the_kind_is_read_across_meshes(case: str) -> None:
    """A destination that carries the sharded kind beside a reader that carries it
    too runs on devices of its own; the reader's chunk loop with point reads, and
    its table with stored reads, take the destination's next-period value by a
    cross-mesh copy.
    """
    report = _fixed_case(case)
    assert (
        report["point"]["devices"]["destination"],
        _destination_reads(report["point"]),
        _destination_reads(report["stored"]),
    ) == (
        [2, 3],
        [("_chunks", "cross_mesh_copy")],
        [("_outside:destination", "cross_mesh_copy")],
    )


def test_a_pruning_reader_copies_a_fixed_destination_into_its_layout() -> None:
    """A destination that keeps the kind fixed takes the first two devices and a
    reader that prunes the kind runs on the next one; the reader's chunk loop with
    point reads, and its table with stored reads, copy the destination's next-period
    value into the reader's layout.
    """
    report = _fixed_case("pruning reader, fixed destination")
    assert (
        report["point"]["devices"],
        _destination_reads(report["point"]),
        _destination_reads(report["stored"]),
    ) == (
        {"source": [2], "destination": [0, 1]},
        [("_chunks", "copy_to_source_layout")],
        [("_outside:destination", "copy_to_source_layout")],
    )


@pytest.mark.parametrize(
    "single_states",
    [
        pytest.param(True, id="single_states"),
        pytest.param(False, id="stateless_singles"),
    ],
)
def test_a_sharded_couple_state_the_singles_prune_solves_as_on_one_device(
    *, single_states: bool
) -> None:
    """Singles who marry by consent into a couple that shards the model-level match
    and dissolves into them, neither single reading the match, solve on four
    devices as on one, to 16 ULP.
    """
    marriage = _run_in_child_process(
        entry_point="report_marriage", arguments=f"single_states={single_states!r}"
    )
    _assert_agree(
        got=_values(marriage["sharded"]), expected=_values(marriage["one_device"])
    )


@pytest.mark.parametrize("arm", ["point", "stored", "one_device_stored"])
def test_a_fixed_sharded_couple_state_solves_as_point_reads_on_one_device(
    *,
    arm: Literal["point", "stored", "one_device_stored"],
    fixed_marriage: PrunedReport,
) -> None:
    """Singles with a state of their own who marry by consent into a couple that
    keeps its sharded model-level match fixed, and dissolve from it, solve on four
    devices, with stored and with point reads, as with point reads on one device,
    to 16 ULP.
    """
    _assert_agree(
        got=_values(fixed_marriage[arm]),
        expected=_values(fixed_marriage["one_device_point"]),
    )


def test_an_exit_projection_names_every_state_its_destination_declares() -> None:
    """A model-level state is declared in every regime, so an exit's projection
    names it even where the destination prunes it.
    """
    with pytest.raises(
        RegimeInitializationError,
        match=re.escape(
            "Exit of source solve projection must name destination states "
            "['kind', 'wealth']."
        ),
    ):
        build_model(devices=(0,), sharded=(), destination_law=None, project_kind=False)


def test_movers_leave_where_the_destination_pays_more(
    subject_sharded: SimulationReport,
) -> None:
    """Subjects at wealth 0.5 and 1 leave for the destination; at wealth 0 they
    stay, for either kind.
    """
    assert subject_sharded["regimes"] == [
        "source",
        "destination",
        "destination",
        "source",
        "destination",
        "destination",
    ]


def test_a_subject_sharded_simulation_reproduces_the_unsharded_one(
    subject_sharded: SimulationReport,
) -> None:
    """Spreading the subjects of a stored-read solve over four devices leaves every
    simulated value unchanged, bit for bit.
    """
    assert subject_sharded["subjects_match_unsharded"] == ""


def test_a_stored_read_solve_simulates_like_a_point_read_solve(
    subject_sharded: SimulationReport,
) -> None:
    """The simulation of a stored-read solve over sharded subjects matches that of
    a point-read solve on one device, to this precision's decimals.
    """
    assert subject_sharded["stored_matches_point"] == ""


def test_a_fixed_reader_sharded_on_four_devices_simulates_as_on_one() -> None:
    """A reader that keeps its sharded kind fixed, solved on four devices with point
    and with stored reads, simulates as a point-read solve on one device, to this
    precision's decimals.
    """
    assert _run_in_child_process(
        entry_point="report_fixed_reader_simulation",
        arguments=f"decimal={tests.conftest.DECIMAL_PRECISION!r}",
    ) == {"point": "", "stored": ""}
