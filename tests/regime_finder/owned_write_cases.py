"""Public-entrypoint filesystem ownership and alias mutation cases."""
from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


COMMANDS = ("check-result", "check-model-result")
OUTPUT_MODES = ("default", "explicit", "equals")
ROLES = ("source", "report")
# Literal externally observed writer names. Do not replace these with imports
# from the implementation's proposed ownership registry.
SUPERVISOR_WRITES = ("analysis-execution.log", "resource.json")
STARTUP_REMOVALS = (
    "report.json", "report.json.writing", "report.md",
    "partial-frontier.json", "partial-frontier.json.writing",
)


def cases():
    for command, mode, role, slot in itertools.product(
            COMMANDS, OUTPUT_MODES, ROLES, SUPERVISOR_WRITES):
        yield {"kind": "supervisor_write", "command": command, "output": mode,
               "protected_role": role, "slot": slot, "end_of_options": False}
    for command, mode, role, slot in itertools.product(
            COMMANDS, OUTPUT_MODES, ROLES, STARTUP_REMOVALS):
        # A default destination is derived from the supplied report's parent.
        # A report inside that destination's own run subdirectory cannot have
        # the required fixed-point layout. Source aliases cover default mode;
        # checked-report aliases cover both explicit syntax forms.
        if role == "report" and mode == "default":
            continue
        yield {"kind": "startup_removal", "command": command, "output": mode,
               "protected_role": role, "slot": slot, "end_of_options": False}
    for command, mode, end in itertools.product(COMMANDS, OUTPUT_MODES, (False, True)):
        yield {"kind": "positive", "command": command, "output": mode,
               "protected_role": None, "slot": None, "end_of_options": end}


def run(project, argv):
    completed = subprocess.run(
        [sys.executable, "-m", "_lcm_regime_finder", *argv],
        cwd=project, capture_output=True, text=True, timeout=45,
        env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
    )
    return {"exit_code": completed.returncode, "stdout": completed.stdout,
            "stderr": completed.stderr}


def one_case(project, case, oracle):
    with tempfile.TemporaryDirectory(prefix="rf10-owned-writes-") as name:
        root = Path(name)
        source = root / "source.json"
        source.write_bytes((project / "tests/regime_finder/examples/declarations/tiny-huge-grid.json").read_bytes())
        analysis = root / "analysis"
        preparation = run(project, ["analyze-model", str(source), "--output", str(analysis),
                                    "--worker", "--seconds", "10"])
        if preparation["exit_code"] != 0:
            raise RuntimeError("Unmutated fixture preparation failed: " + json.dumps(preparation))
        report = analysis / "report.json"
        result = (report.parent / "verification" / "check.json" if case["output"] == "default"
                  else root / "checks" / "selected.json")
        protected = [source, report]
        supplied_source, supplied_report = source, report
        if case["kind"] != "positive":
            alias = result.parent / "verification-run" / case["slot"]
            alias.parent.mkdir(parents=True, exist_ok=True)
            target = source if case["protected_role"] == "source" else report
            alias.symlink_to(target)
            # For destructive writer cases the primary argv uses ordinary
            # external paths. A sound implementation may replace an owned
            # output symlink safely and continue; only original inputs need
            # preservation. A startup alias becomes protected below when it
            # is itself a supplied original input pathname.
            if case["kind"] == "startup_removal":
                protected.append(alias)
                if case["protected_role"] == "source":
                    supplied_source = alias
                else:
                    supplied_report = alias
        before = oracle.snapshot(protected)
        options = ["--seconds", "10", "--host-gib", "0.5"]
        if case["output"] == "explicit":
            options += ["--output", str(result)]
        elif case["output"] == "equals":
            options += ["--output=" + str(result)]
        args = [case["command"], *options]
        if case["end_of_options"]:
            args.append("--")
        args += [str(supplied_source), str(supplied_report)]
        observed = run(project, args)
        issues = oracle.assess(exit_code=observed["exit_code"], result_path=result,
                               protected=before, success=case["kind"] == "positive")
        status = None
        if result.is_file():
            try:
                status = json.loads(result.read_text()).get("status")
            except (OSError, ValueError, AttributeError):
                status = "UNREADABLE"
        return {"case": case, "command": [sys.executable, "-m", "_lcm_regime_finder", *args],
                "exit_code": observed["exit_code"], "public_status": status,
                "passed": not issues, "violations": issues,
                "stdout": observed["stdout"], "stderr": observed["stderr"]}
