"""A capture pads the values the solve does not hold on the host, never the device.

A capture records the next-period value of every regime. The solve holds values
only for the regimes solved at the next period, so a capture fills the others
with zeros laid out like their templates. Admission reserves nothing for those
zeros, so no route puts them on the device: the public archive writes host
zeros, and the replay payload carries the templates. A replay keeps every regime
it holds no value for as its template, so it runs with no padding on the device
either, and still reproduces the solved value. The model is the three-regime
grid-search model of `test_period_target_value_reads`; at its last period the
solve holds no value, so every regime is padded.
"""

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Unpack

import jax
import numpy as np
import pytest

from _lcm.execution.output_layout import PlannedCore
from _lcm.persistence.io import _Picklable
from _lcm.solution import (
    backward_induction,
    period_capture,
    period_replay,
    public_period_capture,
)
from _lcm.typing import JSONValue, PytreeValue
from lcm.persistence import PeriodCapture, replay_period
from tests.solution._callback_types import RunPeriodKernelKwargs, RunPeriodKernelResult
from tests.solution.test_period_target_value_reads import _three_regime_model

_TARGET = ("c", 4)
_PARAMS = {"discount_factor": 0.95}
_SOURCE = {"test": "capture-padding"}

# Shape, dtype name and buffer address of one live array.
type _Resident = tuple[tuple[int, ...], str, int]


def _live_multi_axis_arrays() -> frozenset[_Resident]:
    return frozenset(
        (array.shape, array.dtype.name, array.unsafe_buffer_pointer())
        for array in jax.live_arrays()
        if array.ndim >= 2
    )


def _capture_with_environment(
    *, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    regime, period = _TARGET
    monkeypatch.setenv("LCM_CAPTURE_PERIOD", f"{regime}@{period}")
    monkeypatch.setenv("LCM_CAPTURE_DIR", str(tmp_path))
    _three_regime_model().solve(params=_PARAMS, log_level="off")


def _capture_with_request(*, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("LCM_CAPTURE_PERIOD", raising=False)
    _three_regime_model().solve(
        params=_PARAMS,
        log_level="off",
        period_capture=PeriodCapture(
            directory=tmp_path, periods=(_TARGET,), source_identity=_SOURCE
        ),
    )


@pytest.mark.parametrize(
    "capture",
    [_capture_with_environment, _capture_with_request],
    ids=["replay-capture", "public-capture"],
)
def test_a_capture_puts_no_padding_on_the_device(
    *,
    capture: Callable[..., None],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Every value-shaped array live while a capture is written was computed."""
    computed: set[int] = set()
    samples: list[tuple[frozenset[int], frozenset[_Resident]]] = []
    run_period_kernel = backward_induction._run_period_kernel
    save_pkl = period_capture._save_pkl
    write_period_archive = public_period_capture.write_period_archive

    def _sample() -> None:
        samples.append((frozenset(computed), _live_multi_axis_arrays()))

    def _run(**kwargs: Unpack[RunPeriodKernelKwargs]) -> RunPeriodKernelResult:
        output = run_period_kernel(**kwargs)
        assert isinstance(output.value, jax.Array)
        computed.add(output.value.unsafe_buffer_pointer())
        return output

    def _save(*, path: Path, obj: _Picklable) -> None:
        _sample()
        save_pkl(path=path, obj=obj)

    def _write(
        *,
        path: Path,
        metadata: Mapping[str, JSONValue],
        arrays: Mapping[str, np.ndarray],
    ) -> str:
        # The completed reference is written after the period ran, while its
        # own output is live but not yet returned; only the entry is sampled.
        if path.name == "entry.h5":
            _sample()
        return write_period_archive(path=path, metadata=metadata, arrays=arrays)

    monkeypatch.setattr(backward_induction, "_run_period_kernel", _run)
    monkeypatch.setattr(period_capture, "_save_pkl", _save)
    monkeypatch.setattr(public_period_capture, "write_period_archive", _write)
    computed.update(resident[2] for resident in _live_multi_axis_arrays())
    capture(monkeypatch=monkeypatch, tmp_path=tmp_path)
    value_layouts = {
        (array.shape, array.dtype.name)
        for regime_to_value in _three_regime_model()
        .solve(params=_PARAMS, log_level="off")
        .values.values()
        for array in regime_to_value.values()
    }
    assert samples, "no capture was written"
    assert [
        resident
        for computed_so_far, residents in samples
        for resident in residents
        if resident[:2] in value_layouts and resident[2] not in computed_so_far
    ] == []


def _capture(
    *,
    target: tuple[str, int],
    public: bool,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> np.ndarray:
    """Capture one regime-period and return the value the solve computed for it."""
    regime, period = target
    if public:
        monkeypatch.delenv("LCM_CAPTURE_PERIOD", raising=False)
        solution = _three_regime_model().solve(
            params=_PARAMS,
            log_level="off",
            period_capture=PeriodCapture(
                directory=tmp_path, periods=(target,), source_identity=_SOURCE
            ),
        )
    else:
        monkeypatch.setenv("LCM_CAPTURE_PERIOD", f"{regime}@{period}")
        monkeypatch.setenv("LCM_CAPTURE_DIR", str(tmp_path))
        solution = _three_regime_model().solve(params=_PARAMS, log_level="off")
        monkeypatch.delenv("LCM_CAPTURE_PERIOD")
    values = [
        np.asarray(array)
        for regime_to_value in solution.values.values()
        for array in regime_to_value.values()
    ]
    # Padding is the only all-zero value-shaped array a replay could hold.
    assert not any((value == 0).all() for value in values)
    return np.asarray(solution.values[period][regime])


def _replay(*, route: str, directory: Path) -> np.ndarray:
    if route == "logical":
        return np.asarray(replay_period(directory=directory).output.value)
    if route == "layout":
        return np.asarray(
            period_replay.replay_period_on_recorded_layout(
                directory=directory, devices=jax.devices()[:1]
            ).output.value
        )
    return np.asarray(
        _three_regime_model()
        .replay_period(directory=directory, params=_PARAMS, source_identity=_SOURCE)
        .value
    )


@pytest.mark.parametrize("target", [("c", 4), ("a", 2)], ids=["c@4", "a@2"])
@pytest.mark.parametrize("route", ["logical", "layout", "public"])
def test_a_replay_holds_no_padding_on_the_device_and_reproduces_the_value(
    *,
    route: str,
    target: tuple[str, int],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    solved = _capture(
        target=target,
        public=route == "public",
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
    )
    zero_values: list[tuple[int, ...]] = []
    planned_call = PlannedCore.__call__

    def _call(
        self: PlannedCore, *args: PytreeValue, **kwargs: PytreeValue
    ) -> PytreeValue:
        zero_values.extend(
            array.shape
            for array in jax.live_arrays()
            if array.shape == solved.shape and not np.asarray(array).any()
        )
        return planned_call(self, *args, **kwargs)

    monkeypatch.setattr(PlannedCore, "__call__", _call)
    regime, period = target
    replayed = _replay(route=route, directory=tmp_path / f"{regime}@{period}")
    np.testing.assert_array_equal(replayed, solved)
    assert zero_values == []
