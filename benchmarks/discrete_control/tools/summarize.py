"""Compare only complete matched B1 observations; preserve raw spread and triggers."""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from statistics import median

import numpy as np
from value_contract import assert_agrees_to_ulp

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
output = parser.parse_args().output
rows = []
manifest = json.loads(
    (Path(__file__).resolve().parents[1] / "SOURCE-MANIFEST.json").read_text()
)
installations = {
    arm: json.loads((output / f"{arm}-install-ready.json").read_text())
    for arm in ("base", "head")
}
for arm, installation in installations.items():
    assert installation == json.loads((output / f"{arm}-install-post.json").read_text())
    assert installation["head"] == manifest[arm]
    assert (
        installation["lock_sha256"]
        == manifest["source_files"][manifest[arm]]["pixi.lock"]
    )
instruments = {
    arm: {
        path.split("/site-packages/", 1)[1]: digest
        for path, digest in installation["native_and_instrumentation_sha256"].items()
        if "/site-packages/jax/" in path
    }
    for arm, installation in installations.items()
}
assert instruments["base"] == instruments["head"]
assert len(instruments["base"]) == 3
matched_packages = None
for precision in (64, 32):
    arms = {}
    with np.load(output / f"base-fp{precision}-repeat1/values.npz") as reference:
        assert len(reference.files) == 9
        for arm in ("base", "head"):
            cases = (
                ET.parse(output / f"{arm}-semantic-fp{precision}/junit.xml")  # noqa: S314 - own pytest XML
                .getroot()
                .findall(".//testcase")
            )
            assert len(cases) == 1
            assert all(
                case.find(tag) is None
                for case in cases
                for tag in ("failure", "error", "skipped")
            )
            records = []
            for repeat in (1, 2):
                directory = output / f"{arm}-fp{precision}-repeat{repeat}"
                identity = json.loads(
                    (directory / "events.jsonl").read_text().splitlines()[0]
                )
                assert identity["event"] == "identity"
                assert identity["precision"] == precision
                packages = {
                    name: identity["packages"][name]
                    for name in ("jax", "jaxlib", "numpy")
                }
                if matched_packages is None:
                    matched_packages = packages
                assert packages == matched_packages
                result = json.loads((directory / "result.json").read_text())
                assert result["status"] == "completed"
                samples = result["samples"]
                assert [s["label"] for s in samples] == [
                    "cold",
                    "warm1",
                    "warm2",
                    "warm3",
                ]
                backend = "/jax/core/compile/backend_compile_duration"
                assert all(
                    event["count"] > 0
                    for event in samples[0]["compiler_events"].values()
                )
                assert samples[0]["compile_orchestration_inclusive_seconds"]
                assert all(
                    s["compiler_events"][backend]["count"] == 0 for s in samples[1:]
                )
                with np.load(directory / "values.npz") as values:
                    assert set(values.files) == set(reference.files)
                    for key in values.files:
                        assert values[key].shape == reference[key].shape
                        assert (
                            values[key].dtype
                            == reference[key].dtype
                            == np.dtype(f"float{precision}")
                        )
                        assert_agrees_to_ulp(
                            got=values[key], expected=reference[key], n_ulp=8
                        )
                records.append(
                    {
                        "repeat": repeat,
                        "cold_seconds": samples[0]["wall_seconds"],
                        "warm_seconds": [s["wall_seconds"] for s in samples[1:]],
                        "cold_compiler_events": samples[0]["compiler_events"],
                        "cold_orchestration_inclusive_seconds": samples[0][
                            "compile_orchestration_inclusive_seconds"
                        ],
                        "live_peak_bytes_by_device": [
                            m["peak_bytes_in_use"]
                            for m in samples[-1]["allocator_after"]
                        ],
                        "host_peak_rss_kib": samples[-1]["host_process_peak_rss_kib"],
                    }
                )
                assert len(records[-1]["live_peak_bytes_by_device"]) == 4
            arms[arm] = records
    metrics = {}
    for arm, records in arms.items():
        metrics[arm] = {
            "warm": median([s for r in records for s in r["warm_seconds"]]),
            "backend": median(
                [
                    r["cold_compiler_events"][backend]["duration_seconds"]
                    for r in records
                ]
            ),
            "lowering": median(
                [
                    r["cold_compiler_events"][
                        "/jax/core/compile/jaxpr_to_mlir_module_duration"
                    ]["duration_seconds"]
                    for r in records
                ]
            ),
            "orchestration": median(
                [sum(r["cold_orchestration_inclusive_seconds"]) for r in records]
            ),
            "device_peak": median(
                [max(r["live_peak_bytes_by_device"]) for r in records]
            ),
            "host_peak": median([r["host_peak_rss_kib"] for r in records]),
        }
    ratios = {
        key: (
            metrics["head"][key] / metrics["base"][key]
            if metrics["base"][key] > 0
            else None
        )
        for key in metrics["base"]
    }
    triggers = {
        key: (None if value is None else value > (1.05 if key == "warm" else 1.10))
        for key, value in ratios.items()
    }
    rows.append(
        {
            "precision": precision,
            "raw": arms,
            "medians": metrics,
            "head_over_base": ratios,
            "review_trigger_candidates": triggers,
        }
    )
(output / "B1.json").write_text(
    json.dumps(
        {
            "status": "measured_pending_author_disposition",
            "semantic_cases": 4,
            "timing_processes": 8,
            "timed_solves": 32,
            "rows": rows,
            "scope": (
                "Supported discrete solve control only; no simulation, ACA, "
                "main or continuous-eightway acceptance"
            ),
            "limitations": [
                (
                    "Review-trigger candidates require comparison to retained "
                    "spread and explicit disposition; no automatic waiver"
                ),
                (
                    "Live device high-water cumulative per isolated process, "
                    "including retained first solution"
                ),
                (
                    "Host ru_maxrss is process RSS high-water, not a separate "
                    "future-storage ledger"
                ),
                "Compiler duration categories overlap and must not be added",
                (
                    "Eight-ULP published-value contract only; no new "
                    "decision/RNG or continuum accuracy claim"
                ),
            ],
        },
        indent=2,
    )
    + "\n"
)
