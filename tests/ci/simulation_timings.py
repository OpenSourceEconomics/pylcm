"""Preserve ordered simulation timings and the compile requests measured with them."""

import hashlib
import json
import os
import platform
import statistics
from dataclasses import dataclass
from pathlib import Path

import jax

# Prefix a timing row puts in its skip reason when the host was not steady. A
# ratio between two log levels is a statement about pylcm's runtime validation,
# and it carries that meaning only when the machine served both legs
# comparably. When it did not, the row declines to report a verdict and says so
# with this marker, which the report checker reads as a measurement that was
# not taken rather than a row that was lost.
UNSTABLE_HOST_MARKER = "TIMING_UNSTABLE_HOST"

# Largest control-leg spread a batch may show and still be interpreted.
# Calibrated on this branch's CPU timing lanes: across eight runs and 128
# batches the Linux lanes never exceeded 0.086 and the Windows lane never
# exceeded 0.136, while every macOS batch whose ratio disagreed with the other
# platforms (0.867 and 1.719 against true values near 1.02 and 1.32) sat at
# 0.152 or above. The ceiling therefore discards contaminated batches without
# touching a single Linux or Windows observation on record.
HOST_TIME_MAX_RELATIVE_IQR = 0.15


def _relative_iqr(values: tuple[float, ...]) -> float:
    """Return the inclusive interquartile range of `values` over their median."""
    if len(values) < 2:
        return 0.0
    first, _, third = statistics.quantiles(values, n=4, method="inclusive")
    return (third - first) / statistics.median(values)


@dataclass(frozen=True)
class TimingMeasurement:
    """Wall-time observations and JAX requests from one warm timing batch."""

    samples: tuple[tuple[str, float], ...]
    """Log level and elapsed seconds for each call, in measurement order."""

    trace_requests: int
    """JAX trace requests inside the timed batch, excluding warmups."""

    lowering_requests: int
    """JAX lowering requests inside the timed batch, excluding warmups."""

    compile_requests: int
    """JAX backend compilation requests inside the timed batch, excluding warmups."""

    def write_receipt(
        self, *, directory: Path, nodeid: str, witness: str, stub_preflight: bool
    ) -> str:
        """Write one worker's test receipt and return its full failure-message text."""
        worker = os.environ.get("PYTEST_XDIST_WORKER", "main")
        pid = os.getpid()
        precision = 64 if jax.config.jax_enable_x64 else 32
        receipt = json.dumps(
            {
                "nodeid": nodeid,
                "witness": witness,
                "stub_preflight": stub_preflight,
                "worker_id": worker,
                "pid": pid,
                "platform": platform.system(),
                "python": platform.python_version(),
                "precision": precision,
                "backend": jax.default_backend(),
                "samples": self.samples,
                "medians_seconds": {
                    "off": self.off_seconds,
                    "progress": self.progress_seconds,
                },
                "progress_over_off": self.progress_seconds / self.off_seconds,
                "relative_iqr": {
                    "off": self.off_relative_iqr,
                    "progress": self.progress_relative_iqr,
                },
                "host_is_steady": self.host_is_steady,
                "compile_requests": {
                    "trace": self.trace_requests,
                    "lowering": self.lowering_requests,
                    "compile": self.compile_requests,
                },
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        test_id = hashlib.sha256(nodeid.encode("utf-8")).hexdigest()[:16]
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"fp{precision}-{worker}-{pid}-{test_id}.json"
        path.write_text(receipt + "\n", encoding="utf-8")
        return receipt

    @property
    def off_seconds(self) -> float:
        """Median wall time of the validation-free calls."""
        return statistics.median(
            value for level, value in self.samples if level == "off"
        )

    @property
    def progress_seconds(self) -> float:
        """Median wall time of the progress-logging calls."""
        return statistics.median(
            value for level, value in self.samples if level == "progress"
        )

    @property
    def off_relative_iqr(self) -> float:
        """Spread of the validation-free calls, relative to their own median."""
        return _relative_iqr(
            tuple(value for level, value in self.samples if level == "off")
        )

    @property
    def progress_relative_iqr(self) -> float:
        """Spread of the progress-logging calls, relative to their own median."""
        return _relative_iqr(
            tuple(value for level, value in self.samples if level == "progress")
        )

    @property
    def host_is_steady(self) -> bool:
        """Report whether this batch was taken on a machine steady enough to read.

        Only the `off` leg decides. It carries no runtime validation, so any
        spread in it is the machine's, whereas spread in the `progress` leg is
        partly the quantity under test and must not be allowed to excuse a real
        regression.
        """
        return self.off_relative_iqr <= HOST_TIME_MAX_RELATIVE_IQR
