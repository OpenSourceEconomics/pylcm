"""Compare stage5b_lifetime arms: value/panel/raw sha parity, walls, peaks, I/O.

pixi run python compare_stage5b.py DIR_unblocked DIR_period_major DIR_block_major
"""

# ruff: noqa: INP001, T201

import json
import re
import sys
from pathlib import Path

records = [
    json.loads((Path(directory) / "result.json").read_text())
    for directory in sys.argv[1:]
]
arms = {record["arm"]: record for record in records}
expected_arms = {"unblocked", "period_major", "block_major"}
if len(records) != len(expected_arms) or set(arms) != expected_arms:
    raise SystemExit(
        "Expected exactly three distinct arms: unblocked, period_major, block_major"
    )
labels = ("combined_cold", "combined_warm", "solve_warm", "split_simulate_warm")
for arm, record in arms.items():
    calls = record["calls"]
    if len(calls) != len(labels) or {call["label"] for call in calls} != set(labels):
        raise SystemExit(f"{arm}: expected four distinct calls: {labels}")
provenance_fields = (
    "pylcm_sha",
    "lcm_file",
    "jax",
    "x64",
    "devices",
    "model",
    "pylcm_dirty",
    "gpu_exclusivity",
)
reference_provenance = None
for arm, record in arms.items():
    if any(field not in record for field in provenance_fields):
        raise SystemExit(f"{arm}: missing driver provenance")
    if (
        not isinstance(record["pylcm_sha"], str)
        or not re.fullmatch(r"[0-9a-f]{40}", record["pylcm_sha"])
        or not all(
            isinstance(record[field], str) and record[field]
            for field in ("lcm_file", "jax")
        )
        or type(record["x64"]) is not bool
        or not isinstance(record["devices"], list)
        or not record["devices"]
        or not all(isinstance(device, str) and device for device in record["devices"])
    ):
        raise SystemExit(f"{arm}: invalid source or environment provenance")
    exclusivity = record["gpu_exclusivity"]
    if (
        record["pylcm_dirty"] != ""
        or "construction_error" in record
        or not isinstance(exclusivity, dict)
        or exclusivity.get("exclusive") is not True
        or exclusivity.get("foreign_at_start") != {}
        or exclusivity.get("foreign_seen_during_run") != {}
    ):
        raise SystemExit(
            f"{arm}: provenance requires a clean, successful, exclusive arm"
        )
    if not isinstance(record["model"], dict) or not record["model"]:
        raise SystemExit(f"{arm}: missing economic model provenance")
    model = {
        key: value
        for key, value in record["model"].items()
        if key != "execution_config"
    }
    if (
        not {"builder", "grid_config", "initial_conditions", "pref_types"}
        <= model.keys()
        or not all(
            isinstance(model.get(field), str) and model[field]
            for field in ("builder", "grid_config", "initial_conditions")
        )
        or type(model.get("pref_types")) is not int
        or model["pref_types"] <= 0
    ):
        raise SystemExit(f"{arm}: missing economic model provenance")
    provenance = (
        *(
            record[field]
            for field in ("pylcm_sha", "lcm_file", "jax", "x64", "devices")
        ),
        model,
    )
    if reference_provenance is None:
        reference_provenance = provenance
    if provenance != reference_provenance:
        raise SystemExit(
            f"{arm}: source, environment or economic model provenance differs"
        )
value_keys = None
for arm, record in arms.items():
    for call in record["calls"]:
        values = call.get("value_sha256")
        if (
            not isinstance(values, dict)
            or not values
            or not all(isinstance(key, str) and key for key in values)
        ):
            raise SystemExit(
                f"{arm}/{call['label']}: missing nonempty value digest evidence"
            )
        if value_keys is None:
            value_keys = set(values)
        if set(values) != value_keys:
            raise SystemExit(f"{arm}/{call['label']}: incomplete value coverage")
        digests = list(values.values())
        if call["label"] != "solve_warm":
            digests.extend(call.get(key) for key in ("panel_sha256", "raw_sha256"))
        if not all(
            isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest)
            for digest in digests
        ):
            raise SystemExit(
                f"{arm}/{call['label']}: missing or invalid SHA-256 digest"
            )
ok = True
for label in labels:
    calls = {
        arm: next(c for c in rec["calls"] if c["label"] == label)
        for arm, rec in arms.items()
    }
    bm, pm = calls.get("block_major"), calls.get("period_major")
    for key in ("value_sha256", "panel_sha256", "raw_sha256"):
        if key == "value_sha256" or label != "solve_warm":
            same = bm[key] == pm[key]
            ok &= same
            print(label, key, "block_major==period_major", same)
    for arm, c in calls.items():
        peak = max(
            (s or {}).get("peak_bytes_in_use", 0) for s in c["memory_stats_after"]
        )
        print(
            label,
            arm,
            f"wall={c['wall_seconds']:.1f}s peak_device={peak / 2**30:.3f}GiB "
            f"host_maxrss={c['host_maxrss_kib'] / 2**20:.3f}GiB "
            f"retention={c.get('retention_record')}",
        )
sys.exit(0 if ok else 1)
