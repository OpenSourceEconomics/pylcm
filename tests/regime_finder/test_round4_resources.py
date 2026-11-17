"""Resource exhaustion does not promote or materialize unchecked checkpoints."""

import json
from pathlib import Path
import sys
import tempfile
from _lcm_regime_finder.resources import run_bounded


def test_monitoring_without_time_or_memory_limits() -> None:
    with tempfile.TemporaryDirectory() as folder:
        output = Path(folder)
        code = run_bounded(["sleep", "0.1"], output=output, seconds=None, host_gib=None)
        assert (code) == (0)
        receipt = json.loads((output / "resource.json").read_text())
        assert (receipt["budget"]) == ({"seconds": None, "host_gib": None})
        assert (receipt["exhaustion_reason"]) is None
        assert not ((output / "report.json").exists())


def test_exhaustion_preserves_reference_and_zero_bound_without_adoption() -> None:
    with tempfile.TemporaryDirectory() as folder:
        output = Path(folder)
        checkpoint = output / "partial-frontier.json"
        checkpoint.write_text(
            json.dumps(
                {
                    "status": "INCOMPLETE",
                    "frontier": [
                        {"id": "unverified-source-projection", "cost": {"work": 1}}
                    ],
                }
            )
        )
        before = checkpoint.read_bytes()
        code = run_bounded(
            [sys.executable, "-c", "import time; time.sleep(5)"],
            output=output,
            seconds=0.03,
            host_gib=0.5,
        )
        assert (code) == (2)
        result = json.loads((output / "report.json").read_text())
        assert (result["status"]) == ("INCOMPLETE")
        assert (result["partial_frontier"]) is None
        assert (result["recommendation"]) is None
        assert (result["partial_frontier_reference"]["candidate_safety"]) == (
            "unverified"
        )
        assert (result["partial_frontier_reference"]["size_bytes"]) == (len(before))
        assert (result["established_lower_bound"]) == (
            {"work": 0, "storage": 0, "clarity": 0}
        )
        assert (checkpoint.read_bytes()) == (before)


def test_large_checkpoint_is_not_decoded_by_supervisor() -> None:
    # Sparse resource unit probe, not evidence for a production candidate.
    with tempfile.TemporaryDirectory() as folder:
        output = Path(folder)
        checkpoint = output / "partial-frontier.json"
        with checkpoint.open("wb") as stream:
            stream.truncate(128 * 1024**2)
        code = run_bounded(
            [sys.executable, "-c", "import time; time.sleep(5)"],
            output=output,
            seconds=0.03,
            host_gib=0.5,
        )
        assert (code) == (2)
        result = json.loads((output / "report.json").read_text())
        assert (result["partial_frontier_reference"]["size_bytes"]) == (128 * 1024**2)
        assert ((output / "report.json").stat().st_size) < (4096)


def test_early_exhaustion_without_checkpoint_has_no_candidate() -> None:
    with tempfile.TemporaryDirectory() as folder:
        output = Path(folder)
        code = run_bounded(
            [sys.executable, "-c", "import time; time.sleep(5)"],
            output=output,
            seconds=0,
            host_gib=0.5,
        )
        assert (code) == (2)
        result = json.loads((output / "report.json").read_text())
        assert (result["partial_frontier_reference"]) is None
        assert (result["recommendation"]) is None
