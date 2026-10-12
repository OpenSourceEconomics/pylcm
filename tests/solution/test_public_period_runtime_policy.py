"""A public period capture replays under the output and check policy it recorded."""

import logging
from pathlib import Path
from unittest.mock import Mock

import h5py
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.persistence.period import read_period_archive, write_period_archive
from _lcm.solution import public_period_replay
from _lcm.typing import JSONValue
from _lcm.utils.logging import LogLevel, validation_enabled
from lcm import (
    AgeGrid,
    ByAge,
    LinSpacedGrid,
    Model,
    Regime,
    StochasticTransition,
    Transition,
    categorical,
    fixed_transition,
)
from lcm.exceptions import InvalidRegimeTransitionProbabilitiesError
from lcm.execution import ExecutionConfig
from lcm.persistence import PeriodCapture, load_period_capture
from lcm.typing import ContinuousState, FloatND, ScalarInt

pytestmark = pytest.mark.coverage(backends=("cpu",), precisions="both")


@categorical(ordered=False)
class _RegimeId:
    alive: ScalarInt
    done: ScalarInt


def _utility(wealth: ContinuousState) -> FloatND:
    return wealth


def _probability(*, wealth: ContinuousState, probability_mass: float) -> FloatND:
    """Expose one canonical parameter while retaining identical model code."""
    return jnp.asarray(probability_mass) + 0.0 * wealth


def _model() -> Model:
    wealth = LinSpacedGrid(start=1.0, stop=2.0, n_points=2)
    return Model(
        regimes={
            "alive": Regime(
                states={"wealth": wealth},
                functions={"utility": _utility},
                state_transitions={"wealth": fixed_transition("wealth")},
            ),
            "done": Regime(states={"wealth": wealth}, functions={"utility": _utility}),
        },
        ages=AgeGrid(start=0, inclusive_stop=2, step="Y"),
        regime_id_class=_RegimeId,
        initial_nodes={0: "alive"},
        execution_config=ExecutionConfig(devices=(0,)),
        edges={
            "alive": Transition(
                targets={"alive": 0, "done": (0, 1)},
                law=ByAge(
                    cases={
                        0: {
                            "alive": StochasticTransition(func=_probability),
                            "done": StochasticTransition(func=_probability),
                        },
                        1: "done",
                    }
                ),
            )
        },
    )


_SOURCE = {"witness": "public-replay-policy-v1"}
_LEVELS = {
    "off": logging.CRITICAL,
    "warning": logging.WARNING,
    "progress": logging.INFO,
    "debug": logging.DEBUG,
}


def _params(*, probability_mass: float = 0.5) -> dict[str, float]:
    return {"discount_factor": 0.9, "probability_mass": probability_mass}


def _capture(
    *,
    directory: Path,
    probability_mass: float = 0.5,
    log_level: LogLevel = "off",
    runtime_checks: bool = True,
    target: tuple[str, int] = ("done", 1),
) -> Path:
    capture = PeriodCapture(
        directory=directory, periods=(target,), source_identity=_SOURCE
    )
    _model().solve(
        params=_params(probability_mass=probability_mass),
        log_level=log_level,
        runtime_checks=runtime_checks,
        period_capture=capture,
    )
    return directory / f"{target[0]}@{target[1]}"


def _replace_metadata(*, directory: Path, metadata: dict[str, JSONValue]) -> None:
    """Rewrite an entry and correctly rebind its completed reference checksum."""
    _, arrays, _ = read_period_archive(path=directory / "entry.h5")
    digest = write_period_archive(
        path=directory / "entry.h5", metadata=metadata, arrays=arrays
    )
    completed, references, _ = read_period_archive(path=directory / "completed.h5")
    completed["entry_sha256"] = digest
    write_period_archive(
        path=directory / "completed.h5", metadata=completed, arrays=references
    )


@pytest.mark.parametrize(
    ("probability_mass", "target"),
    [
        pytest.param(0.5, ("done", 1), id="valid-mass-finite-terminal"),
        pytest.param(0.4, ("done", 1), id="under-mass-finite-terminal"),
        pytest.param(0.6, ("done", 1), id="over-mass-finite-terminal"),
        pytest.param(-0.1, ("done", 1), id="negative-mass-finite-terminal"),
        pytest.param(0.5, ("alive", 0), id="valid-mass-earlier-entry"),
        pytest.param(0.4, ("alive", 0), id="under-mass-nan-entry"),
        pytest.param(0.6, ("alive", 0), id="over-mass-nan-entry"),
        pytest.param(-0.1, ("alive", 0), id="negative-mass-nan-entry"),
    ],
)
def test_public_replay_accepts_a_completed_unchecked_capture(
    *, tmp_path: Path, probability_mass: float, target: tuple[str, int]
) -> None:
    """An unchecked capture replays byte-for-byte without re-running optional checks."""
    directory = _capture(
        directory=tmp_path,
        probability_mass=probability_mass,
        runtime_checks=False,
        target=target,
    )
    record = load_period_capture(directory=directory)
    assert record.completed
    assert record.reference is not None
    if target == ("done", 1):
        # A different earlier law is invalid; the selected terminal value is exact.
        np.testing.assert_array_equal(record.reference, np.array([1.0, 2.0]))
    elif probability_mass == 0.5:
        assert np.isfinite(record.reference).all()
    else:
        # The library deliberately leaves numerical invalidity visible as NaN.
        assert np.isnan(record.reference).all()
    replay = _model().replay_period(
        directory=directory,
        params=_params(probability_mass=probability_mass),
        source_identity=_SOURCE,
    )
    assert replay.reference_matches is True


def test_checked_capture_still_rejects_the_bad_probability_row(
    *, tmp_path: Path
) -> None:
    """A checked solve refuses an invalid probability law before publishing an entry."""
    with pytest.raises(InvalidRegimeTransitionProbabilitiesError):
        _capture(directory=tmp_path, probability_mass=0.4, runtime_checks=True)
    assert not (tmp_path / "done@1" / "entry.h5").exists()


@pytest.mark.parametrize(
    ("log_level", "runtime_checks"),
    [
        pytest.param(level, checks, id=f"{level}-checks-{checks}")
        for level in ("off", "warning", "progress", "debug")
        for checks in (False, True)
    ],
)
def test_public_replay_dispatch_receives_the_captured_call_policy(
    *,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    log_level: LogLevel,
    runtime_checks: bool,
) -> None:
    """The replayed period runs with the captured output level and check flag."""
    directory = _capture(
        directory=tmp_path, log_level=log_level, runtime_checks=runtime_checks
    )
    run_period = Mock(wraps=public_period_replay._run_period_kernel)
    monkeypatch.setattr(public_period_replay, "_run_period_kernel", run_period)
    replay = _model().replay_period(
        directory=directory, params=_params(), source_identity=_SOURCE
    )
    assert replay.reference_matches is True
    run_period.assert_called_once()
    logger = run_period.call_args.kwargs["logger"]
    assert logger.getEffectiveLevel() == _LEVELS[log_level]
    assert validation_enabled(logger) is runtime_checks


def test_public_replay_refuses_a_fieldless_legacy_policy(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A capture without a recorded policy is refused before inputs are rebuilt."""
    directory = _capture(directory=tmp_path)
    entry, _, _ = read_period_archive(path=directory / "entry.h5")
    # Keep the complete old envelope and deliberately remove new policy fields.
    legacy_names = {
        "identity",
        "regime",
        "period",
        "age",
        "layouts",
        "widths",
        "admission",
        "optimized_hlo",
        "retain_replay",
    }
    legacy = {name: value for name, value in entry.items() if name in legacy_names}
    _replace_metadata(directory=directory, metadata=legacy)
    run_period = Mock(wraps=public_period_replay._run_period_kernel)
    monkeypatch.setattr(public_period_replay, "_run_period_kernel", run_period)
    restore_inputs = Mock(wraps=public_period_replay._restore_public_inputs)
    monkeypatch.setattr(public_period_replay, "_restore_public_inputs", restore_inputs)
    with pytest.raises(ValueError, match=r"policy|recapture"):
        _model().replay_period(
            directory=directory, params=_params(), source_identity=_SOURCE
        )
    restore_inputs.assert_not_called()
    run_period.assert_not_called()


@pytest.mark.parametrize(
    "call_policy",
    [
        None,
        {},
        {"log_level": "off"},
        {"log_level": "verbose", "runtime_checks": False},
        {"log_level": "off", "runtime_checks": 0},
        {"log_level": "off", "runtime_checks": "false"},
        {"log_level": "off", "runtime_checks": False, "unknown": 1},
    ],
    ids=[
        "null",
        "empty",
        "missing-checks",
        "unknown-level",
        "bool-as-int",
        "bool-as-text",
        "unknown-key",
    ],
)
def test_public_replay_refuses_malformed_checksum_consistent_policy(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, call_policy: JSONValue
) -> None:
    """A malformed recorded policy is refused before inputs are rebuilt."""
    directory = _capture(directory=tmp_path, runtime_checks=False)
    entry, _, _ = read_period_archive(path=directory / "entry.h5")
    entry["call_policy"] = call_policy
    _replace_metadata(directory=directory, metadata=entry)
    run_period = Mock(wraps=public_period_replay._run_period_kernel)
    monkeypatch.setattr(public_period_replay, "_run_period_kernel", run_period)
    restore_inputs = Mock(wraps=public_period_replay._restore_public_inputs)
    monkeypatch.setattr(public_period_replay, "_restore_public_inputs", restore_inputs)
    with pytest.raises(
        (TypeError, ValueError), match=r"policy|log_level|runtime_checks"
    ):
        _model().replay_period(
            directory=directory, params=_params(), source_identity=_SOURCE
        )
    restore_inputs.assert_not_called()
    run_period.assert_not_called()


def test_recorded_checked_policy_still_validates_probability_laws(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A recorded checked policy validates probability laws before dispatch."""
    directory = _capture(directory=tmp_path, probability_mass=0.4, runtime_checks=False)
    entry, _, _ = read_period_archive(path=directory / "entry.h5")
    entry["call_policy"] = {"log_level": "off", "runtime_checks": True}
    _replace_metadata(directory=directory, metadata=entry)
    run_period = Mock(wraps=public_period_replay._run_period_kernel)
    monkeypatch.setattr(public_period_replay, "_run_period_kernel", run_period)
    with pytest.raises(InvalidRegimeTransitionProbabilitiesError):
        _model().replay_period(
            directory=directory,
            params=_params(probability_mass=0.4),
            source_identity=_SOURCE,
        )
    run_period.assert_not_called()


def test_unchecked_public_replay_still_requires_identity(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unchecked capture still refuses a different source identity."""
    directory = _capture(directory=tmp_path, probability_mass=0.4, runtime_checks=False)
    run_period = Mock(wraps=public_period_replay._run_period_kernel)
    monkeypatch.setattr(public_period_replay, "_run_period_kernel", run_period)
    with pytest.raises(ValueError, match="identity"):
        _model().replay_period(
            directory=directory,
            params=_params(probability_mass=0.4),
            source_identity={"witness": "different-source"},
        )
    run_period.assert_not_called()


def test_unchecked_public_replay_still_requires_archive_checksum(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unchecked capture still refuses an archive whose checksum does not match."""
    directory = _capture(directory=tmp_path, probability_mass=0.4, runtime_checks=False)
    with h5py.File(directory / "entry.h5", "r+") as archive:
        archive["manifest"].attrs["sha256"] = "0" * 64
    run_period = Mock(wraps=public_period_replay._run_period_kernel)
    monkeypatch.setattr(public_period_replay, "_run_period_kernel", run_period)
    with pytest.raises(ValueError, match="checksum"):
        _model().replay_period(
            directory=directory,
            params=_params(probability_mass=0.4),
            source_identity=_SOURCE,
        )
    run_period.assert_not_called()
