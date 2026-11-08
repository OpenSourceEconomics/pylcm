"""A solution archive written value by value while a solve runs.

Values reach the file as the solve produces them, with their checksums computed
as they are written; the artifacts and the manifest complete it once the result
is known. The finished file is the archive `SolutionResult.save` writes: it loads
to the same values and metadata. From its creation the file records the solve
and how far it got, so a solve that fails leaves an incomplete file holding that
record, which `load_solution` refuses.
"""

import gc
import json
from pathlib import Path

import h5py
import jax
import numpy as np
import pytest

from _lcm.persistence.solution import DiskValueEntry, StreamedSolutionArchive
from _lcm.typing import JSONValue
from lcm import ExecutionConfig
from lcm.exceptions import IncompatibleSolutionError, SolutionIntegrityError
from lcm.persistence import load_solution
from lcm.solver_api import SolutionResult
from tests.solution.test_period_target_value_reads import _three_regime_model

_PARAMS = {"discount_factor": 0.95}
_RECORD = {"dtype": "float64", "pylcm_version": "test"}


def _host_solution() -> SolutionResult:
    return _three_regime_model(ExecutionConfig(value_storage="host")).solve(
        params=_PARAMS, log_level="off"
    )


def _stream_values(
    *, archive: StreamedSolutionArchive, solution: SolutionResult
) -> dict[tuple[int, str], DiskValueEntry]:
    """Write every value of `solution`, latest period first, as a solve would."""
    return {
        (period, regime_name): archive.write_value(
            period=period,
            regime=regime_name,
            value=np.asarray(value),
            sharding=value.sharding,
        )
        for period in sorted(solution.values, reverse=True)
        for regime_name, value in solution.values[period].items()
    }


def test_a_completed_streamed_archive_loads_like_a_saved_solution(
    tmp_path: Path,
) -> None:
    """Values, value schemas and artifacts equal those of `SolutionResult.save`."""
    solution = _host_solution()
    saved = load_solution(path=solution.save(path=tmp_path / "saved.h5"))
    archive = StreamedSolutionArchive(
        directory=tmp_path, name="solution-1", record=_RECORD
    )
    _stream_values(archive=archive, solution=solution)

    streamed = load_solution(path=archive.complete(solution=solution))

    assert streamed.metadata.value_schemas == saved.metadata.value_schemas
    assert {p: set(v) for p, v in streamed.values.items()} == {
        p: set(v) for p, v in saved.values.items()
    }
    for period, regime_to_value in saved.values.items():
        for regime_name, value in regime_to_value.items():
            np.testing.assert_array_equal(
                np.asarray(streamed.value(period=period, regime=regime_name)),
                np.asarray(value),
                strict=True,
            )
    assert streamed.replay_artifacts.keys() == saved.replay_artifacts.keys()


def test_a_completed_archive_is_named_after_the_solve(tmp_path: Path) -> None:
    solution = _host_solution()
    archive = StreamedSolutionArchive(
        directory=tmp_path, name="solution-7", record=_RECORD
    )
    _stream_values(archive=archive, solution=solution)

    path = archive.complete(solution=solution)

    assert path == tmp_path / "solution-7.h5"
    assert sorted(child.name for child in tmp_path.iterdir()) == ["solution-7.h5"]


def test_a_disk_value_entry_reads_its_value_onto_its_sharding(tmp_path: Path) -> None:
    """Before and after completion, a read is the written value on the device."""
    solution = _host_solution()
    archive = StreamedSolutionArchive(
        directory=tmp_path, name="solution-2", record=_RECORD
    )
    entries = _stream_values(archive=archive, solution=solution)

    def _check() -> None:
        for (period, regime_name), entry in entries.items():
            read = entry.materialize()
            assert read.sharding.memory_kind == "device"
            np.testing.assert_array_equal(
                np.asarray(read),
                np.asarray(solution.value(period=period, regime=regime_name)),
                strict=True,
            )
            np.testing.assert_array_equal(
                entry.host_value(), np.asarray(read), strict=True
            )

    _check()
    archive.complete(solution=solution)
    _check()


def test_an_unfinished_archive_records_how_far_the_solve_got(tmp_path: Path) -> None:
    """The record is in the file as each period lands; unfinished, it never loads."""
    solution = _host_solution()
    archive = StreamedSolutionArchive(
        directory=tmp_path, name="solution-3", record=_RECORD
    )
    assert _record_of(archive) == _RECORD | {"last_completed_period": None}
    archive.write_value(
        period=4,
        regime="c",
        value=np.asarray(solution.value(period=4, regime="c")),
        sharding=solution.value(period=4, regime="c").sharding,
    )
    archive.mark_period_written(period=4)
    assert _record_of(archive) == _RECORD | {"last_completed_period": 4}

    del archive
    gc.collect()

    path = tmp_path / "solution-3.incomplete"
    assert sorted(child.name for child in tmp_path.iterdir()) == [path.name]
    with h5py.File(path, "r") as file:
        assert "manifest" not in file
        assert json.loads(bytes(file["incomplete"][()]).decode()) == _RECORD | {
            "last_completed_period": 4
        }
    with pytest.raises(SolutionIntegrityError):
        load_solution(path=path)


def _record_of(archive: StreamedSolutionArchive) -> dict[str, JSONValue]:
    """Read the record from the archive's file, through its writing handle."""
    dataset = archive._file.read(dataset="incomplete")
    return json.loads(bytes(dataset).decode())


def test_a_written_value_whose_bytes_changed_is_refused(tmp_path: Path) -> None:
    """A read verifies the checksum computed when the value was written."""
    solution = _host_solution()
    archive = StreamedSolutionArchive(
        directory=tmp_path, name="solution-4", record=_RECORD
    )
    entries = _stream_values(archive=archive, solution=solution)
    path = archive.complete(solution=solution)
    entry = entries[(4, "c")]
    with h5py.File(path, "r+") as file:
        dataset = file[entry.dataset]
        dataset[...] = dataset[...] + 1.0

    with pytest.raises(SolutionIntegrityError):
        jax.block_until_ready(entry.materialize())


def _cpu_sharding() -> jax.sharding.Sharding:
    return jax.sharding.SingleDeviceSharding(jax.devices("cpu")[0])


def test_a_weakly_typed_value_is_planned_and_read_weakly_typed(tmp_path: Path) -> None:
    """The planning descriptor and the read keep the weak type the solve gave."""
    archive = StreamedSolutionArchive(
        directory=tmp_path, name="solution-5", record=_RECORD
    )
    entry = archive.write_value(
        period=0,
        regime="a",
        value=np.ones(3, dtype=np.float32),
        sharding=_cpu_sharding(),
        weak_type=True,
    )

    assert entry.device_stand_in().weak_type
    assert entry.materialize().weak_type


@pytest.mark.usefixtures("x64_disabled")
def test_a_read_refuses_to_narrow_a_value_the_solve_wrote_in_x64(
    tmp_path: Path,
) -> None:
    """Without x64, a float64 value read back is refused, not narrowed to float32."""
    archive = StreamedSolutionArchive(
        directory=tmp_path, name="solution-6", record=_RECORD
    )
    entry = archive.write_value(
        period=0,
        regime="a",
        value=np.ones(3, dtype=np.float64),
        sharding=_cpu_sharding(),
    )

    with pytest.raises(IncompatibleSolutionError, match="float64"):
        entry.materialize()
