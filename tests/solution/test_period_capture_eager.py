"""A capture taken on the eager route stays writable and refuses a layout replay.

The eager route's cores are not XLA executables and publish no `input_shardings`
or `output_shardings`. A capture records their layout block as absent rather than
failing, so the logical replay stays available, and the layout-faithful replay
refuses by naming the cores whose placement it cannot reinstate.
"""

from pathlib import Path

import jax
import pytest

from _lcm.solution import period_capture, period_replay
from lcm import Model
from tests.test_models import nbegm_ride_along_toy as toy
from tests.test_models.nbegm_common import RegimeId


@pytest.fixture
def eager_capture(*, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Solve the alive/dead toy without JIT and return its capture directory."""
    monkeypatch.setenv("LCM_CAPTURE_PERIOD", "alive@1")
    monkeypatch.setenv("LCM_CAPTURE_DIR", str(tmp_path))
    base = toy.build_model(
        variant="brute", n_periods=4, n_liquid=8, n_consumption=6, n_savings=8
    )
    eager = Model(
        regimes=base.user_regimes,
        ages=base.ages,
        regime_id_class=RegimeId,
        fixed_params=dict(base.fixed_params),
        enable_jit=False,
    )
    eager.solve(params=toy.build_params(), log_level="off")
    return tmp_path / "alive@1"


def test_capture_kernel_inputs_writes_a_payload_on_the_eager_route(
    *,
    eager_capture: Path,
) -> None:
    """An eager solve writes the selected regime-period's capture payload."""
    assert (eager_capture / period_capture._PAYLOAD_NAME).exists()


def test_capture_records_the_eager_core_layout_block_as_absent(
    *,
    eager_capture: Path,
) -> None:
    """Every core of an eager capture records `None` in place of a layout block."""
    payload = period_replay._load_capture_payload(directory=eager_capture)

    assert set(payload[period_capture.LAYOUTS_KEY].cores.values()) == {None}


def test_replay_period_replays_an_eager_capture_logically(
    *,
    eager_capture: Path,
) -> None:
    """The logical replay of an eager capture reports scope `logical`."""
    assert period_replay.replay_period(directory=eager_capture).scope == "logical"


def test_replay_period_on_recorded_layout_names_the_cores_it_cannot_reinstate(
    *,
    eager_capture: Path,
) -> None:
    """A layout replay of an eager capture refuses, naming the core it cannot place."""
    with pytest.raises(ValueError, match=r"no device layout for core\(s\) \['main'\]"):
        period_replay.replay_period_on_recorded_layout(
            directory=eager_capture, devices=jax.devices()
        )
