"""Independent public-entrypoint source-provenance and alias cases."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def execute(project, arguments):
    command = [sys.executable, "-m", *arguments]
    completed = subprocess.run(command, cwd=project, text=True, capture_output=True,
                               timeout=60,
                               env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
    return {"command": command, "exit_code": completed.returncode,
            "stdout": completed.stdout, "stderr": completed.stderr}


def cases():
    for source, command, output, slot, alias in itertools.product(
            ("direct", "symlink_with_mirror"),
            ("check-result", "check-model-result"),
            ("default", "explicit", "equals"),
            ("analysis-execution.log", "resource.json"), ("symlink", "hardlink")):
        yield {"kind": "supervisor_alias", "source": source, "command": command,
               "output": output, "slot": slot, "alias": alias,
               "worker": False, "end_of_options": False}
    for source, command, output, end in itertools.product(
            ("direct", "symlink_with_mirror", "symlink_without_mirror"),
            ("check-result", "check-model-result"),
            ("default", "explicit", "equals"), (False, True)):
        yield {"kind": "positive", "source": source, "command": command,
               "output": output, "slot": None, "alias": None,
               "worker": False, "end_of_options": end}
    for source, command, output, alias in itertools.product(
            ("direct", "symlink_with_mirror"),
            ("check-result", "check-model-result"),
            ("explicit", "equals"), ("symlink", "hardlink")):
        yield {"kind": "worker_temporary_alias", "source": source, "command": command,
               "output": output, "slot": "result.writing", "alias": alias,
               "worker": True, "end_of_options": False}
    for source, command in itertools.product(
            ("direct", "symlink_with_mirror", "symlink_without_mirror"),
            ("check-result", "check-model-result")):
        yield {"kind": "positive", "source": source, "command": command,
               "output": "stdout", "slot": None, "alias": None,
               "worker": True, "end_of_options": False}
    for source, command, output in itertools.product(
            ("direct", "symlink_with_mirror"),
            ("check-result", "check-model-result"), ("explicit", "equals")):
        yield {"kind": "result_is_actual_manifest", "source": source, "command": command,
               "output": output, "slot": "result", "alias": None,
               "worker": False, "end_of_options": False}


def one_case(project, case, oracle, fixture):
    with tempfile.TemporaryDirectory(prefix="rf11-provenance-") as temporary:
        root = Path(temporary)
        real = root / "A"
        supplied = root / "B"
        report_output = root / "C"
        real.mkdir(); supplied.mkdir()
        original = real / "input.json"
        original.write_bytes((fixture / "input.json").read_bytes())
        header = json.loads(original.read_text())["structural_store"]
        relative = Path(header["path"])
        shutil.copytree(fixture / relative, real / relative)
        source = original
        if case["source"] in {"symlink_with_mirror", "symlink_without_mirror"}:
            source = supplied / "source-link.json"
            source.symlink_to(original)
            # This complete independent mirror makes the current admission pass.
            # The actual reader still consumes A, following source-link.json.
            if case["source"] == "symlink_with_mirror":
                shutil.copytree(real / relative, supplied / relative)
        preparation = execute(project, [
            "_lcm_regime_finder.pipeline", "--search-mode", "exact", str(original), "--output", str(report_output),
            "--seconds", "30", "--host-gib", "1", "--worker"])
        report = report_output / "report.json"
        if preparation["exit_code"] != 0 or not report.is_file():
            raise AssertionError("Unchanged pipeline fixture preparation failed: " + json.dumps(preparation))
        if json.loads(report.read_text()).get("status") != "COMPLETE":
            raise AssertionError("Expected a COMPLETE unchanged fixture report")
        dependencies = oracle.fixture_dependencies(source)
        actual_manifest = dependencies["actual_manifest"]
        report_manifest = report_output / relative / "manifest.json"
        if not report_manifest.is_file():
            raise AssertionError("Expected separately published report-side structural manifest")
        if os.path.samefile(actual_manifest, report_manifest):
            raise AssertionError("Witness requires distinct original/report manifest inodes")
        if actual_manifest.read_bytes() != report_manifest.read_bytes():
            # Canonical JSON output may use different whitespace; source identity
            # remains exact, and only semantic manifest equality is needed here.
            if json.loads(actual_manifest.read_text()) != json.loads(report_manifest.read_text()):
                raise AssertionError("Report mirror changed manifest contents")
        result = (report.parent / "verification/check.json" if case["output"] == "default"
                  else root / "D/check.json")
        if case["output"] == "stdout":
            result = None
        if case["kind"] == "result_is_actual_manifest":
            result = actual_manifest
        protected_paths = dependencies["paths"] + [report, report_manifest]
        protected_paths.extend(sorted((real / relative).glob("*.bin")))
        protected_paths.extend(sorted((report_output / relative).glob("*.bin")))
        if case["source"] == "symlink_with_mirror":
            protected_paths.extend(sorted((supplied / relative).iterdir()))
        protected = oracle.snapshot(protected_paths)
        mutation_path = None
        if case["kind"] == "supervisor_alias":
            assert result is not None
            mutation_path = result.parent / "verification-run" / case["slot"]
        elif case["kind"] == "worker_temporary_alias":
            assert result is not None
            mutation_path = result.with_name(result.name + ".writing")
        if mutation_path is not None:
            mutation_path.parent.mkdir(parents=True, exist_ok=True)
            if case["alias"] == "symlink":
                mutation_path.symlink_to(actual_manifest)
            else:
                os.link(actual_manifest, mutation_path)
        preexisting_temporary = (oracle.identity(mutation_path)
                                 if case["kind"] == "worker_temporary_alias" else None)
        options = ["--seconds", "30", "--host-gib", "1"]
        if case["output"] == "explicit":
            options += ["--output", str(result)]
        elif case["output"] == "equals":
            options += ["--output=" + str(result)]
        if case["worker"]:
            options.append("--worker")
        arguments = ["_lcm_regime_finder", case["command"], *options]
        if case["end_of_options"]:
            arguments.append("--")
        arguments += [str(source), str(report)]
        observed = execute(project, arguments)
        failures = oracle.assess(
            protected=protected, exit_code=observed["exit_code"], result_path=result,
            stdout=observed["stdout"] if result is None else None,
            positive=case["kind"] == "positive",
            preexisting_temporary=preexisting_temporary,
            result_is_original=case["kind"] == "result_is_actual_manifest")
        status = None
        try:
            status = json.loads(result.read_text() if result is not None else observed["stdout"]).get("status")
        except (OSError, ValueError, AttributeError):
            pass
        return {
            "case": case, "passed": not failures, "violations": failures,
            "exit_code": observed["exit_code"], "public_status": status,
            "command": observed["command"], "stdout": observed["stdout"],
            "stderr": observed["stderr"], "preparation": preparation,
            "fixture_sha256": hashlib.sha256((fixture / "input.json").read_bytes()).hexdigest(),
            "dependency_context": {
                "supplied_source": str(source), "resolved_source": str(source.resolve()),
                "actual_manifest": str(actual_manifest),
                "admission_lexical_manifest": str(dependencies["lexical_manifest"]),
                "report_manifest": str(report_manifest),
                "distinct_original_report_manifest_inodes": True,
                "mutation_path": str(mutation_path) if mutation_path is not None else None,
                "independent_oracle_basis": dependencies["basis"],
            },
        }
