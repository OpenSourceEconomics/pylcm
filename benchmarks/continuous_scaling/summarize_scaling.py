"""Reconcile complete fixed-workload scaling observations without merging precisions."""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from statistics import median

import numpy as np

from tests.conftest import assert_agrees_to_ulp

parser = argparse.ArgumentParser()
parser.add_argument("--results", type=Path, required=True)
args = parser.parse_args()
rows = []
xml_counts = []
for precision in (64, 32):
    for devices in (3, 4, 6, 8):
        path = args.results / f"semantic-gpu{devices}-fp{precision}/junit.xml"
        cases = ET.parse(path).getroot().findall(".//testcase")  # noqa: S314 - own pytest XML
        assert len(cases) == 6
        assert all(
            case.find(tag) is None
            for case in cases
            for tag in ("failure", "error", "skipped")
        )
        xml_counts.append(
            {
                "precision": precision,
                "devices": devices,
                "passed": 6,
                "failed": 0,
                "errors": 0,
                "skipped": 0,
            }
        )
    reference = np.load(
        args.results / f"head-continuous-1-fp{precision}-repeat1/values.npz"
    )
    precision_rows = []
    for devices in (1, 3, 4, 6, 8):
        warm, cold, backend, lowering, orchestration, peaks, host, per_repeat = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for repeat in (1, 2):
            directory = (
                args.results / f"head-continuous-{devices}-fp{precision}-repeat{repeat}"
            )
            data = json.loads((directory / "result.json").read_text())
            assert data["status"] == "completed"
            samples = data["samples"]
            assert [s["label"] for s in samples] == ["cold", "warm1", "warm2", "warm3"]
            for sample in samples[1:]:
                assert (
                    sample["compiler_events"][
                        "/jax/core/compile/backend_compile_duration"
                    ]["count"]
                    == 0
                )
            values = np.load(directory / "values.npz")
            assert sorted(values.files) == sorted(reference.files)
            for key in values.files:
                assert_agrees_to_ulp(got=values[key], expected=reference[key], n_ulp=8)
            current = [s["wall_seconds"] for s in samples[1:]]
            warm.extend(current)
            per_repeat.append(median(current))
            cold.append(samples[0]["wall_seconds"])
            backend.append(
                samples[0]["compiler_events"][
                    "/jax/core/compile/backend_compile_duration"
                ]["duration_seconds"]
            )
            lowering.append(
                samples[0]["compiler_events"][
                    "/jax/core/compile/jaxpr_to_mlir_module_duration"
                ]["duration_seconds"]
            )
            orchestration.append(
                sum(samples[0]["compile_orchestration_inclusive_seconds"])
            )
            selected_peaks = [
                m["peak_bytes_in_use"] for m in samples[-1]["allocator_after"]
            ]
            assert len(selected_peaks) == devices
            peaks.append(selected_peaks)
            host.append(samples[-1]["host_process_peak_rss_kib"])
        precision_rows.append(
            {
                "precision": precision,
                "devices": devices,
                "cold_seconds": cold,
                "warm_seconds": warm,
                "median_warm_seconds": median(warm),
                "repeat_warm_medians": per_repeat,
                "warm_min_seconds": min(warm),
                "warm_max_seconds": max(warm),
                "cold_backend_request_seconds": backend,
                "cold_lowering_seconds": lowering,
                "cold_compile_orchestration_inclusive_seconds": orchestration,
                "process_live_peak_bytes_per_device": peaks,
                "host_process_peak_rss_kib": host,
            }
        )
    reference_time = precision_rows[0]["median_warm_seconds"]
    for row in precision_rows:
        row["speedup_vs_one_gpu"] = reference_time / row["median_warm_seconds"]
        row["parallel_efficiency"] = row["speedup_vs_one_gpu"] / row["devices"]
    rows.extend(precision_rows)
summary = {
    "status": "completed",
    "semantic_counts": xml_counts,
    "rows": rows,
    "scope": (
        "Strong scaling of fixed tiny 3x3x24 witness, not ACA "
        "production or larger-grid scaling"
    ),
    "limitations": [
        (
            "Six warm observations from two independent processes per "
            "topology and precision; raw spread retained"
        ),
        (
            "Two shared-node allocations may differ; only within this "
            "allocation same-source comparisons are reported"
        ),
        "GPU peaks cumulative isolated-process live high-water, not per-call peak",
        "Timing does not subtract listener overhead; compiler categories overlap",
        (
            "This scaling study does not waive unfinished G4 "
            "baseline/main comparisons or resource-review triggers"
        ),
    ],
}
(args.results / "scaling-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
lines = [
    "# Fixed-workload GPU scaling",
    "",
    summary["scope"],
    "",
    "| Precision | GPUs | Warm median (s) | Speedup | Efficiency |",
    "| --- | ---: | ---: | ---: | ---: |",
]
lines.extend(
    f"| fp{row['precision']} | {row['devices']} | "
    f"{row['median_warm_seconds']:.6f} | {row['speedup_vs_one_gpu']:.3f} | "
    f"{row['parallel_efficiency']:.3f} |"
    for row in rows
)
lines += ["", *summary["limitations"]]
(args.results / "SCALING-RESULTS.md").write_text("\n".join(lines) + "\n")
print(
    json.dumps(
        {
            "semantic_passes": 48,
            "timing_processes": 20,
            "synchronized_solves": 80,
            "summary": str(args.results / "scaling-summary.json"),
        }
    )
)
