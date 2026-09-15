"""Preserve ordered simulation timings and the compile requests measured with them."""

import hashlib
import json
import os
import platform
import statistics
from dataclasses import dataclass
from pathlib import Path

import jax


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
