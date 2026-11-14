"""Qualify a bundled evaluator candidate without changing its admission table.

Run in a fresh Pixi process from the checkout under review. All model construction
is confined to the tiny native unit fixture; supplied cases are immutable metadata.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from types import MappingProxyType
import xml.etree.ElementTree as ET

from _lcm_regime_finder.fibre_source import Wire, _mapping, _rows, _strings, _text
from _lcm_regime_finder.implicit.check_projection import _project
from _lcm_regime_finder.implicit.implicit_native import _produce
from _lcm_regime_finder.implicit.source_versions import (
    MANDATORY_CONTROLS,
    algorithm_sources,
    assert_runtime_source,
    descriptor_digest,
    resolve_source_candidate,
    sha,
)
from tests.regime_finder.test_implicit_version_candidate import native_capture

REPO = Path(__file__).resolve().parents[2]
CONTROL_TARGETS = MappingProxyType(
    {
        "trusted-version-boundary": (
            "tests/regime_finder/test_implicit_source_versions.py",
        ),
        "typed-terminals-and-differential-facts": (
            "tests/regime_finder/test_implicit_core.py",
            "tests/regime_finder/test_implicit_source.py",
        ),
        "support-closure-and-finite-premises": (
            "tests/regime_finder/test_implicit_screen.py",
            "tests/regime_finder/test_implicit_projection.py",
        ),
        "cache-order-and-binding-depth": (
            "tests/regime_finder/test_implicit_source.py::test_pinned_cache_order_boundary_and_exact_streaming_fallback",
        ),
        "native-namespace-and-immutable-capture": (
            "tests/test_graphs.py::test_analysis_preserves_source_only_conditional_draw",
            "tests/test_graphs.py::test_analysis_preserves_conditioned_current_shock",
            "tests/test_graphs.py::test_conditioned_persistent_process_preserves_original_phase_inputs",
            "tests/test_graphs.py::test_target_conditioned_process_preserves_needed_source_history",
            "tests/regime_finder/test_implicit_version_candidate.py::test_genuine_native_capture_replays_without_relabelling",
        ),
        "guarded-candidate-transaction": (
            "tests/regime_finder/test_implicit_version_candidate.py",
        ),
    }
)


def _write(path: Path, value: Wire) -> None:
    """Create a new evidence document; do not replace an existing record."""
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _matches(*, testcase: ET.Element, selector: str) -> bool:
    module, separator, function = selector.partition("::")
    module_name = module.removesuffix(".py").replace("/", ".")
    classname = testcase.get("classname", "")
    if classname != module_name and not classname.startswith(module_name + "."):
        return False
    name = testcase.get("name", "")
    return not separator or name == function or name.startswith(function + "[")


def qualify_source_version(
    *, source_version: str, cases_file: Path, evidence_directory: Path
) -> Wire:
    """Run fixed controls and independent full replay; never promote the candidate."""
    if set(CONTROL_TARGETS) != set(MANDATORY_CONTROLS):
        raise ValueError("Mandatory qualification control inventory changed")
    version = resolve_source_candidate(source_version=source_version)
    package = REPO / "src/_lcm_regime_finder"
    identity = assert_runtime_source(package=package, version=version)
    environment = os.environ.get("PIXI_ENVIRONMENT_NAME")
    if (
        not environment
        or Path(sys.prefix).resolve() != (REPO / ".pixi/envs" / environment).resolve()
    ):
        raise ValueError(
            "Qualification requires this checkout's physical Pixi environment"
        )
    cases_sha = sha(cases_file)
    payload = _mapping(json.loads(cases_file.read_text()))
    if (
        set(payload) != {"schema", "cases"}
        or payload["schema"] != "implicit-qualification-cases/v1"
    ):
        raise ValueError("Cases select immutable metadata, not qualification controls")
    cases = _rows(payload["cases"])
    evidence_directory.mkdir(parents=True, exist_ok=False)
    targets = set(target for group in CONTROL_TARGETS.values() for target in group)
    targets = {
        target
        for target in targets
        if "::" not in target or target.partition("::")[0] not in targets
    }
    xml = evidence_directory / "controls.xml"
    command = [
        "pixi",
        "run",
        "--as-is",
        "-e",
        environment,
        "pytest",
        *sorted(targets),
        "-v",
        f"--junitxml={xml}",
    ]
    manifest = {name: list(group) for name, group in CONTROL_TARGETS.items()}
    control_paths = set((REPO / "tests").rglob("*.py"))
    control_paths.update(
        REPO / name
        for name in ("conftest.py", "pyproject.toml", "pixi.lock")
        if (REPO / name).is_file()
    )
    control_sources = {
        str(path.relative_to(REPO)): sha(path) for path in sorted(control_paths)
    }
    _write(
        evidence_directory / "INVOCATION.json",
        {
            "command": command,
            "cwd": str(REPO),
            "prefix": sys.prefix,
            "control_manifest": manifest,
            "control_sources": control_sources,
            "cases_sha256": cases_sha,
            "semantic_descriptor_sha256": descriptor_digest(version),
        },
    )
    with (evidence_directory / "controls.log").open("xb") as stream:
        execution = subprocess.run(
            command, cwd=REPO, stdout=stream, stderr=subprocess.STDOUT, check=False
        )
    rows = tuple(ET.parse(xml).getroot().iter("testcase")) if xml.is_file() else ()
    controls = {}
    for name, group in CONTROL_TARGETS.items():
        covered = tuple(
            row
            for row in rows
            if any(_matches(testcase=row, selector=target) for target in group)
        )
        complete = (
            execution.returncode == 0
            and all(
                any(_matches(testcase=row, selector=target) for row in covered)
                for target in group
            )
            and not any(
                row.find(tag) is not None
                for row in covered
                for tag in ("failure", "error", "skipped")
            )
        )
        controls[name] = {
            "status": "PASS" if complete else "INCOMPLETE",
            "testcases": len(covered),
            "junit_sha256": sha(xml) if xml.is_file() else None,
        }
    results = []
    if all(value["status"] == "PASS" for value in controls.values()):
        native = evidence_directory / "native-source.json"
        native_axes = evidence_directory / "native-axes.json"
        _write(native, native_capture())
        _write(native_axes, {"alive": ["history"], "final": ["history"]})
        selected = (
            {
                "name": "native-current",
                "snapshot": str(native),
                "axes_file": str(native_axes),
            },
            *cases,
        )
        names = [_text(case["name"]) for case in selected]
        if len(set(names)) != len(names) or any(
            not name or Path(name).name != name or name in {".", ".."} for name in names
        ):
            raise ValueError(
                "Qualification input names must be distinct directory names"
            )
        for case in selected:
            name = _text(case["name"])
            stage = evidence_directory / name
            stage.mkdir()
            snapshot, axes = (
                Path(_text(case["snapshot"])),
                Path(_text(case["axes_file"])),
            )
            snapshot_sha, axes_sha = sha(snapshot), sha(axes)
            artifact = stage / "facts.json"
            _write(
                artifact, _produce(snapshot=snapshot, axes_file=axes, version=version)
            )
            removals = case.get("removed", [])
            if not isinstance(removals, (list, tuple)):
                raise ValueError("Supplied removals must be a sequence")
            removed_rows = tuple(_strings(value) for value in removals)
            if any(len(pair) != 2 for pair in removed_rows) or len(
                set(removed_rows)
            ) != len(removed_rows):
                raise ValueError(
                    "Supplied removals must be distinct regime/status pairs"
                )
            result = _project(
                snapshot=snapshot,
                axes_file=axes,
                artifact=artifact,
                removed=frozenset((pair[0], pair[1]) for pair in removed_rows),
                version=version,
            )
            _write(stage / "replay.json", result)
            if sha(snapshot) != snapshot_sha or sha(axes) != axes_sha:
                raise ValueError("Qualification input bytes changed")
            results.append(
                {
                    "status": "QUALIFICATION_REPLAY_COMPLETE",
                    "name": name,
                    "snapshot_sha256": snapshot_sha,
                    "axes_sha256": axes_sha,
                    "all_point_counts": _mapping(result["fact_verification"])[
                        "all_point_counts"
                    ],
                    "source_obligations": result["source_obligations"],
                    "finite_conditions_pass": _mapping(result["screen"])[
                        "conditions_pass"
                    ],
                    "fact_packet_sha256": sha(artifact),
                    "replay_sha256": sha(stage / "replay.json"),
                }
            )
    if sha(cases_file) != cases_sha or any(
        sha(REPO / name) != digest for name, digest in control_sources.items()
    ):
        raise ValueError("Qualification cases or control implementation changed")
    identity.check()
    receipt = {
        "schema": "implicit-source-qualification/v1",
        "status": "QUALIFICATION_COMPLETE"
        if results and all(value["status"] == "PASS" for value in controls.values())
        else "QUALIFICATION_INCOMPLETE",
        "semantic_descriptor_sha256": descriptor_digest(version),
        "source_files": dict(version.source_files),
        "algorithm_sources": algorithm_sources(),
        "control_manifest": manifest,
        "control_sources": control_sources,
        "control_results": controls,
        "input_results": results,
        "qualification": "Selected evaluator's point program only; every source qualification remains. No automatic admission, native adoption, maximum removal, or global optimality.",
    }
    _write(evidence_directory / "qualification.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-version", required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    result = qualify_source_version(
        source_version=args.source_version,
        cases_file=args.cases,
        evidence_directory=args.evidence,
    )
    raise SystemExit(0 if result["status"] == "QUALIFICATION_COMPLETE" else 1)


if __name__ == "__main__":
    main()
