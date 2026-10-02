"""Compare stage5b_lifetime arms: value/panel/raw sha parity, walls, peaks, I/O.

    python compare_stage5b.py DIR_unblocked DIR_period_major DIR_block_major
"""

import json
import sys
from pathlib import Path

arms = {json.loads((Path(d) / "result.json").read_text())["arm"]: json.loads((Path(d) / "result.json").read_text()) for d in sys.argv[1:]}
ok = True
for label in ("combined_cold", "combined_warm", "solve_warm", "split_simulate_warm"):
    calls = {arm: next(c for c in rec["calls"] if c["label"] == label) for arm, rec in arms.items()}
    bm, pm = calls.get("block_major"), calls.get("period_major")
    for key in ("value_sha256", "panel_sha256", "raw_sha256"):
        if bm and pm and key in bm:
            same = bm[key] == pm[key]
            ok &= same
            print(label, key, "block_major==period_major", same)
    for arm, c in calls.items():
        peak = max((s or {}).get("peak_bytes_in_use", 0) for s in c["memory_stats_after"])
        print(label, arm, f"wall={c['wall_seconds']:.1f}s peak_device={peak/2**30:.3f}GiB host_maxrss={c['host_maxrss_kib']/2**20:.3f}GiB retention={c.get('retention_record')}")
sys.exit(0 if ok else 1)
