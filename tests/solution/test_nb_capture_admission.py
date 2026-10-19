"""Plain pytest administrative controls; no numerical or project imports."""

import asyncio
import copy
import functools
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import NotRequired, TypedDict

import pytest

from tests.solution._nb_scientific_capture_admission import (
    admit_capture,
    bind_source_files,
    drain_owned_callbacks,
    encode_object_strings,
    source_identity,
    verify_capture_files,
)

type _Json = bool | int | float | str | list[_Json] | dict[str, _Json] | None

type _ArrayRecord = dict[str, str | list[int]]


class _SavedRecord(TypedDict):
    file: str
    arrays: dict[str, _ArrayRecord]
    sha256: NotRequired[str]
    original_query_leaf_names: NotRequired[list[str]]


def row_arrays() -> dict[str, _ArrayRecord]:
    """Describe this fixture's actual 120/150/two-interval row contract."""
    shapes = {
        "continuation_V": [150],
        "continuation_M": [150],
        "liquid": [120],
        "savings_nodes": [150],
        "discount": [],
        "coh_slopes": [2],
        "coh_intercepts": [2],
        "breakpoints": [1],
        "published_V": [120],
        "published_M": [120],
        "published_C": [120],
        "query__savings": [120],
        "query__node_kind": [120],
        "query__link_valid": [119],
        "query__owner_id": [120],
        "query__side_id": [120],
        "query__floor_used": [],
        "query__corner_continuation": [2],
        "query__corner_kind": [2],
        "query__corner_interval_id": [2],
        "query__corner_side_id": [2],
    }
    integer = {
        "query__node_kind",
        "query__owner_id",
        "query__side_id",
        "query__corner_kind",
        "query__corner_interval_id",
        "query__corner_side_id",
    }
    return {
        name: {
            "shape": shape,
            "dtype": "int32"
            if name in integer
            else "bool"
            if name == "query__link_valid"
            else "float32",
        }
        for name, shape in shapes.items()
    }


def valid_manifest():
    """Finite metadata admission fixture, not a native numerical computation."""
    arrays = row_arrays()
    component_fields = {
        "value-coordinates": ["p1__saving__income", "p1__saving__liquid"],
        "actions": ["consumption"],
        "params": ["discount_factor", "tax_rate", "tax_exemption"],
        "income-primitives": [
            "income_codes",
            "transition_rows",
            "income_amounts",
            "declared_floor",
        ],
        "initial": ["age", "regime_id", "liquid", "income"],
        "simulation": ["period", "subject_id", "regime_name", "consumption"],
        "period0": ["period", "subject_id", "regime_name", "consumption"],
    }
    saved: list[_SavedRecord] = [
        {
            "file": f"{solver}-{suffix}.npz",
            "arrays": {
                name: {
                    "shape": [2500] if suffix == "actions" else [],
                    "dtype": "float32",
                }
                for name in names
            },
        }
        for solver in ("nbegm", "grid")
        for suffix, names in component_fields.items()
    ]
    saved += [
        {
            "file": "actual-period0-actions.npz",
            "arrays": {
                solver: {"shape": [60], "dtype": "float32"}
                for solver in ("nbegm", "grid")
            },
        }
    ]
    saved += [
        {
            "file": f"{solver}-public-values.npz",
            "arrays": {
                name: {"shape": shape, "dtype": "float32"}
                for name, shape in (
                    ("p0__saving", [2, 120]),
                    ("p1__saving", [2, 120]),
                    ("p2__done", [120]),
                )
            },
        }
        for solver in ("nbegm", "grid")
    ]
    rows = [f"nbegm-row-{i:04d}.npz" for i in range(4)]
    saved += [
        {
            "file": name,
            "arrays": copy.deepcopy(arrays),
            "original_query_leaf_names": sorted(
                key for key in arrays if key.startswith("query__")
            ),
        }
        for name in rows
    ]
    saved += [
        {
            "file": "original-policy-comparison.npz",
            "arrays": {
                key: {"shape": [], "dtype": "float32"}
                for key in (
                    "absolute_difference",
                    "violates_original_budget",
                    "original_budget",
                )
            },
        }
    ]
    return {
        "precision": 32,
        "source_map_sha256": "a" * 64,
        "source_files": {"fixture.py": "b" * 64},
        "original_assertion_retained": False,
        "original_comparison_role": "retained_diagnostic",
        "original_budget": 0.03,
        "panels": {solver: list(range(60)) for solver in ("nbegm", "grid")},
        "row_matches": [[1, 0], [1, 1], [0, 0], [0, 1]],
        "rows": rows,
        "saved": saved,
    }


def test_complete_typed_metadata_is_admitted():
    assert "metadata admitted" in admit_capture(
        manifest=valid_manifest(), precision=32, source_sha="a" * 64
    )


@pytest.mark.parametrize("field", list(row_arrays()))
def test_every_original_row_input_and_corner_field_is_required(field):
    manifest = valid_manifest()
    row = next(
        entry for entry in manifest["saved"] if entry["file"] == manifest["rows"][0]
    )
    del row["arrays"][field]
    with pytest.raises(ValueError, match=r"Missing or mismatched"):
        admit_capture(manifest=manifest, precision=32, source_sha="a" * 64)


@pytest.mark.parametrize(
    "mutation",
    ["whole_corner", "empty_W", "float_status", "duplicate_row", "extra_query_leaf"],
)
def test_incomplete_or_mistyped_rows_are_refused(mutation):
    manifest = valid_manifest()
    row = next(
        entry for entry in manifest["saved"] if entry["file"] == manifest["rows"][0]
    )
    if mutation == "whole_corner":
        row["arrays"] = {
            key: value
            for key, value in row["arrays"].items()
            if not key.startswith("query__corner_")
        }
    elif mutation == "empty_W":
        row["arrays"]["continuation_V"]["shape"] = [0]
    elif mutation == "float_status":
        row["arrays"]["query__node_kind"]["dtype"] = "float32"
    elif mutation == "duplicate_row":
        manifest["rows"][0] = manifest["rows"][1]
    else:
        row["original_query_leaf_names"].append("query__additional_original_leaf")
    with pytest.raises(
        ValueError,
        match=r"Missing or mismatched|Source, precision|Original query leaf inventory",
    ):
        admit_capture(manifest=manifest, precision=32, source_sha="a" * 64)


def test_only_strings_are_encoded_losslessly():
    values = ["saving", "done", "", "a\x00", "\u2603", '"\\']
    encoded, dtype = encode_object_strings(values)
    assert [json.loads(item) for item in encoded] == values
    assert dtype == f"<U{max(map(len, encoded))}"
    for invalid in ([None], [1], [object()]):
        with pytest.raises(TypeError):
            encode_object_strings(invalid)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("source_map_sha256", "c" * 64),
        ("precision", 64),
        ("panels", {"nbegm": list(range(59)), "grid": list(range(60))}),
        ("row_matches", [[1, 0]] * 4),
        ("original_budget", 0.05),
        ("original_assertion_retained", True),
        ("source_files", {}),
    ],
)
def test_source_precision_panel_and_budget_mutations_are_refused(*, key, value):
    manifest = valid_manifest()
    manifest[key] = value
    with pytest.raises(ValueError, match=r"Source, precision"):
        admit_capture(manifest=manifest, precision=32, source_sha="a" * 64)


@pytest.mark.parametrize(
    "suffix",
    [
        "actions",
        "value-coordinates",
        "income-primitives",
        "initial",
        "simulation",
        "period0",
        "params",
    ],
)
def test_complete_capture_components_are_required(suffix):
    manifest = valid_manifest()
    manifest["saved"] = [
        entry for entry in manifest["saved"] if entry["file"] != f"grid-{suffix}.npz"
    ]
    with pytest.raises(ValueError, match=r"Missing or mismatched|Missing complete"):
        admit_capture(manifest=manifest, precision=32, source_sha="a" * 64)


def make_npy(*, descriptor, shape, payload):
    header = repr({"descr": descriptor, "fortran_order": False, "shape": shape}) + "\n"
    return (
        b"\x93NUMPY\x01\x00"
        + struct.pack("<H", len(header))
        + header.encode()
        + payload
    )


@pytest.mark.parametrize(
    ("payload", "descriptor", "dtype"),
    [
        (struct.pack("<f", 1.0), "<f4", "float32"),
        ("123".encode("utf-32-le"), "<U3", "<U3"),
    ],
)
def test_object_origin_requires_actual_unicode_json_string_payload(
    *, tmp_path, payload, descriptor, dtype
):
    path = tmp_path / "column.npz"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "x.npy", make_npy(descriptor=descriptor, shape=(1,), payload=payload)
        )
    manifest: dict[str, list[_SavedRecord]] = {
        "saved": [
            {
                "file": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "arrays": {
                    "x": {
                        "shape": [1],
                        "dtype": dtype,
                        "original_dtype": "object",
                        "encoding": "json-string-ascii-in-fixed-unicode",
                    }
                },
            }
        ]
    }
    with pytest.raises(ValueError, match=r"Object-origin"):
        verify_capture_files(manifest=manifest, root=tmp_path)


def test_actual_unicode_json_payload_and_byte_mutations(tmp_path):
    encoded, dtype = encode_object_strings(["saving\x00\u2603"])
    payload = encoded[0].encode("utf-32-le")
    path = tmp_path / "column.npz"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "x.npy", make_npy(descriptor=dtype, shape=(1,), payload=payload)
        )
    manifest: dict[str, list[_SavedRecord]] = {
        "saved": [
            {
                "file": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "arrays": {
                    "x": {
                        "shape": [1],
                        "dtype": dtype,
                        "original_dtype": "object",
                        "encoding": "json-string-ascii-in-fixed-unicode",
                    }
                },
            }
        ]
    }
    assert "NPZ byte inventory admitted" in verify_capture_files(
        manifest=manifest, root=tmp_path
    )
    for field, value in (("shape", [2]), ("dtype", "float64")):
        mutated = copy.deepcopy(manifest)
        mutated["saved"][0]["arrays"]["x"][field] = value
        with pytest.raises(ValueError, match=r"NPY shape|NPY dtype"):
            verify_capture_files(manifest=mutated, root=tmp_path)
    manifest["saved"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="Capture NPZ checksum mismatch"):
        verify_capture_files(manifest=manifest, root=tmp_path)


def test_provenance_uses_real_source_of_decorated_original():
    def original():
        return "actual"

    def wrapper():
        return "wrapper"

    wrapper.__code__ = wrapper.__code__.replace(co_filename="<beartype-pseudo>")
    wrapped = functools.wraps(original)(wrapper)
    path, digest = source_identity(obj=wrapped, module=sys.modules[__name__])
    assert Path(path) == Path(__file__).resolve()
    assert len(digest) == 64
    assert wrapped() == "wrapper"


def test_source_file_identity_must_match_expected_immutable_map(tmp_path):
    path = tmp_path / "module.py"
    path.write_text("x = 1\n")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    bind_source_files(
        files={str(path): digest},
        expected_map={"module.py": {"sha256": digest}},
        source_root=tmp_path,
    )
    with pytest.raises(ValueError, match="Observed module differs"):
        bind_source_files(
            files={str(path): digest},
            expected_map={"module.py": {"sha256": "0" * 64}},
            source_root=tmp_path,
        )


def test_failure_drain_preserves_primary_error():
    def fail_drain():
        raise RuntimeError("drain failure")

    with (
        pytest.raises(ValueError, match="primary") as caught,
        drain_owned_callbacks(fail_drain),
    ):
        raise ValueError("primary")
    assert "drain failure" in caught.value.__notes__[0]


async def _run_administration_probe(*, argv: tuple[str, ...], directory: Path) -> str:
    """Run the exact Pixi argument vector and capture its isolated child's output."""
    process = await asyncio.create_subprocess_exec(
        *argv,
        cwd=directory,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=60)
    except BaseException:
        if process.returncode is None:
            process.kill()
            await process.wait()
        raise
    if process.returncode:
        raise subprocess.CalledProcessError(
            process.returncode, argv, output=stdout.decode(), stderr=stderr.decode()
        )
    return stdout.decode()


def test_administration_did_not_import_numerical_or_project_modules() -> None:
    """Admission imports no numerical modules in a fresh Pixi child."""
    root = Path(__file__).resolve().parents[2]
    manifest = Path(os.environ["PIXI_PROJECT_MANIFEST"]).resolve()
    environment = os.environ["PIXI_ENVIRONMENT_NAME"]
    lock = manifest.parent / "pixi.lock"
    assert manifest.is_file()
    assert lock.is_file()
    probe = """
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, sys.argv[1])
from tests.solution import _nb_scientific_capture_admission as admission
path = Path(admission.__file__).resolve()
manifest = Path(sys.argv[2]).resolve()
print(json.dumps({
    "path": str(path),
    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    "forbidden_loaded": sorted(
        {"numpy", "jax", "lcm", "_lcm"}.intersection(sys.modules)
    ),
    "pixi_manifest": str(manifest),
    "pixi_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    "pixi_lock_sha256": hashlib.sha256(
        (manifest.parent / "pixi.lock").read_bytes()
    ).hexdigest(),
    "runtime": str(Path(sys.executable).resolve()),
    "prefix": str(Path(sys.prefix).resolve()),
}))
"""
    pixi = shutil.which("pixi")
    assert pixi is not None
    stdout = asyncio.run(
        _run_administration_probe(
            argv=(
                pixi,
                "run",
                "--as-is",
                "--manifest-path",
                str(manifest),
                "--environment",
                environment,
                "python",
                "-I",
                "-B",
                "-c",
                probe,
                str(root),
                str(manifest),
            ),
            directory=root,
        )
    )
    path = root / "tests/solution/_nb_scientific_capture_admission.py"
    assert json.loads(stdout) == {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "forbidden_loaded": [],
        "pixi_manifest": str(manifest),
        "pixi_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "pixi_lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest(),
        "runtime": str(Path(sys.executable).resolve()),
        "prefix": str(Path(sys.prefix).resolve()),
    }


def _limited_manifest(*, source: Path) -> dict[str, _Json]:
    """Describe limited observed modules without inventing a whole-source map."""
    manifest = valid_manifest()
    manifest["provenance_scope"] = "observed_module_sources"
    manifest["source_map_externally_admitted"] = False
    manifest["source_map_sha256"] = None
    manifest["source_files"] = {
        str(source): hashlib.sha256(source.read_bytes()).hexdigest()
    }
    return manifest


def test_ordinary_module_sources_need_no_external_whole_map(tmp_path: Path) -> None:
    """Real in-root observed modules admit limited provenance without an audit map."""
    source = tmp_path / "module.py"
    source.write_text("fixture_constant = 1\n")
    manifest = _limited_manifest(source=source)
    bind_source_files(
        files=manifest["source_files"], expected_map=None, source_root=tmp_path
    )
    assert manifest["source_map_externally_admitted"] is False
    assert manifest["source_map_sha256"] is None
    assert "metadata admitted" in admit_capture(
        manifest=manifest, precision=32, source_sha=None
    )


def test_declared_audit_scope_requires_its_external_map() -> None:
    """A declared audit lane cannot use missing-map equality as authentication."""
    manifest = valid_manifest()
    manifest["provenance_scope"] = "externally_admitted_whole_source"
    manifest["source_map_externally_admitted"] = True
    manifest["source_map_sha256"] = None
    with pytest.raises(ValueError, match=r"source|Source|provenance|audit"):
        admit_capture(manifest=manifest, precision=32, source_sha=None)


def test_ordinary_observed_sources_refuse_foreign_roots(tmp_path: Path) -> None:
    """Limited provenance still refuses an observed module outside its true root."""
    source = tmp_path / "foreign.py"
    source.write_text("fixture_constant = 1\n")
    root = tmp_path / "project"
    root.mkdir()
    files = {str(source): hashlib.sha256(source.read_bytes()).hexdigest()}
    with pytest.raises(ValueError, match=r"relative path|subpath|source|root"):
        bind_source_files(files=files, expected_map=None, source_root=root)


def test_unknown_provenance_scope_is_refused() -> None:
    """An unrecognized provenance label cannot inherit full-map authentication."""
    manifest = valid_manifest()
    manifest["provenance_scope"] = "inferred_whole_source"
    manifest["source_map_externally_admitted"] = True
    with pytest.raises(ValueError, match=r"source|Source|provenance|scope"):
        admit_capture(manifest=manifest, precision=32, source_sha="a" * 64)


def test_limited_module_scope_cannot_claim_external_map_authentication(
    tmp_path: Path,
) -> None:
    """Observed module hashes cannot be promoted to an externally admitted map."""
    source = tmp_path / "module.py"
    source.write_text("fixture_constant = 1\n")
    manifest = _limited_manifest(source=source)
    manifest["source_map_externally_admitted"] = True
    with pytest.raises(ValueError, match=r"source|Source|provenance|scope"):
        admit_capture(manifest=manifest, precision=32, source_sha=None)


def _retired_manifest() -> dict[str, _Json]:
    """Use the existing finite admission fixture with explicit retired-gate wording."""
    manifest = valid_manifest()
    manifest["original_assertion_retained"] = False
    manifest["original_comparison_role"] = "retained_diagnostic"
    return manifest


def test_retained_diagnostic_metadata_is_admitted() -> None:
    """Complete observations remain admissible when their old gate is diagnostic."""
    assert "metadata admitted" in admit_capture(
        manifest=_retired_manifest(), precision=32, source_sha="a" * 64
    )


def test_retired_comparison_cannot_claim_the_assertion_remains_active() -> None:
    """Diagnostic metadata cannot claim the retired policy assertion is retained."""
    manifest = _retired_manifest()
    manifest["original_assertion_retained"] = True
    with pytest.raises(ValueError, match="original contract"):
        admit_capture(manifest=manifest, precision=32, source_sha="a" * 64)


@pytest.mark.parametrize("role", ["primary_gate", "passing_assertion"])
def test_comparison_metadata_cannot_claim_primary_acceptance(role: str) -> None:
    """A retained distance comparison cannot become a primary or passing assertion."""
    manifest = _retired_manifest()
    manifest["original_comparison_role"] = role
    with pytest.raises(ValueError, match="original contract"):
        admit_capture(manifest=manifest, precision=32, source_sha="a" * 64)
