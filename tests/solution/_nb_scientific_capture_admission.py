"""Standard-library observational metadata admission and lossless text transport."""

import argparse
import ast
import hashlib
import inspect
import json
import math
import struct
import sys
import zipfile
from contextlib import contextmanager, suppress
from pathlib import Path


def encode_object_strings(values):
    """Encode only string object leaves without coercing arbitrary objects."""
    if any(not isinstance(value, str) for value in values):
        raise TypeError("Object columns must contain strings exclusively")
    encoded = [json.dumps(value, ensure_ascii=True) for value in values]
    return encoded, f"<U{max(map(len, encoded), default=1)}"


def admit_capture(*, manifest, precision, source_sha):
    """Reject incomplete original-fixture observation metadata."""
    scope = manifest.get("provenance_scope", "externally_admitted_whole_source")
    if scope == "externally_admitted_whole_source":
        if (
            not isinstance(source_sha, str)
            or len(source_sha) != 64
            or any(character not in "0123456789abcdef" for character in source_sha)
            or manifest.get("source_map_externally_admitted", True) is not True
        ):
            raise ValueError("Audit source provenance requires its external source map")
    elif scope == "observed_module_sources":
        if (
            source_sha is not None
            or manifest.get("source_map_sha256") is not None
            or manifest.get("source_map_externally_admitted") is not False
        ):
            raise ValueError("Limited source scope cannot claim external map admission")
    else:
        raise ValueError("Unknown source provenance scope")
    checks = (
        manifest.get("precision") == precision,
        manifest.get("source_map_sha256") == source_sha,
        bool(manifest.get("source_files")),
        manifest.get("original_assertion_retained") is False,
        manifest.get("original_comparison_role") == "retained_diagnostic",
        manifest.get("original_budget") == 3 * ((25.0 - 0.01) / 2499),
        manifest.get("panels")
        == {solver: list(range(60)) for solver in ("nbegm", "grid")},
        sorted(manifest.get("row_matches", [])) == [[0, 0], [0, 1], [1, 0], [1, 1]],
        len(manifest.get("rows", [])) == len(set(manifest.get("rows", []))) == 4,
    )
    if not all(checks):
        raise ValueError(
            "Source, precision, complete panel, row identity or original contract "
            "missing"
        )
    files = {entry["file"]: entry["arrays"] for entry in manifest["saved"]}
    dtype = f"float{precision}"
    for solver in ("nbegm", "grid"):
        _require_array(
            files=files,
            filename="actual-period0-actions.npz",
            name=solver,
            shape=[60],
            dtype=dtype,
        )
        _require_array(
            files=files,
            filename=f"{solver}-actions.npz",
            name="consumption",
            shape=[2500],
            dtype=dtype,
        )
        _require_components(files=files, solver=solver)
        for name, shape in (
            ("p0__saving", [2, 120]),
            ("p1__saving", [2, 120]),
            ("p2__done", [120]),
        ):
            _require_array(
                files=files,
                filename=f"{solver}-public-values.npz",
                name=name,
                shape=shape,
                dtype=dtype,
            )
    entries = {entry["file"]: entry for entry in manifest["saved"]}
    for name in manifest["rows"]:
        _require_original_row(files=files, entry=entries[name], dtype=dtype)
    return (
        "metadata admitted; NPZ bytes and native provenance require "
        "independent verification"
    )


def admit_processed_params(*, manifest, precision):
    """Require both operations' actual model-owned processed numeric inventories."""
    calls = manifest.get("processed_parameter_calls", [])
    expected = {
        (solver, operation)
        for solver in ("nbegm", "grid")
        for operation in ("solve", "simulate")
    }
    owners = [(call.get("solver"), call.get("operation")) for call in calls]
    if len(owners) != 4 or set(owners) != expected:
        raise ValueError(
            "Both model-owned solve/simulate parameter results are required"
        )
    entries = {entry["file"]: entry["arrays"] for entry in manifest["saved"]}
    models, values = {}, {}
    for call in calls:
        solver, operation = call["solver"], call["operation"]
        filename = f"{solver}-{operation}-processed-params.npz"
        if (
            call.get("file") != filename
            or call.get("method") != "lcm.model.Model._canonicalize_params"
            or call.get("returned_original_mapping") is not True
            or not isinstance(call.get("returned_mapping_id"), int)
        ):
            raise ValueError("Processed result lacks actual method/return provenance")
        identity = call.get("model_object_id")
        if (
            not isinstance(identity, int)
            or models.setdefault(solver, identity) != identity
        ):
            raise ValueError(
                "Processed solve/simulate results belong to different models"
            )
        _admit_processed_leaves(
            call=call, arrays=entries.get(filename, {}), precision=precision
        )
        values[solver, operation] = {
            tuple(leaf["path"]): (
                leaf["native_dtype"],
                leaf["native_shape"],
                leaf["bit_pattern_hex"],
            )
            for leaf in call["leaves"]
        }
    for solver in ("nbegm", "grid"):
        if values[solver, "solve"] != values[solver, "simulate"]:
            raise ValueError(
                "Actual same-model solve/simulate canonical parameter values differ"
            )
    if models["nbegm"] == models["grid"]:
        raise ValueError("Both methods cannot claim the same actual model object")
    return (
        "Processed metadata admitted; actual method forwarding/native parity "
        "remains separate"
    )


def _admit_processed_leaves(*, call, arrays, precision):
    leaves = call.get("leaves", [])
    names = [leaf["array"] for leaf in leaves]
    paths = [tuple(leaf["path"]) for leaf in leaves]
    if (
        not leaves
        or len(names) != len(set(names))
        or len(paths) != len(set(paths))
        or set(names) != set(arrays)
    ):
        raise ValueError(
            "Processed qualified leaf inventory is incomplete or ambiguous"
        )
    for leaf in leaves:
        metadata = arrays[leaf["array"]]
        if (
            not leaf["path"]
            or any(not isinstance(name, str) for name in leaf["path"])
            or not isinstance(leaf.get("object_id"), int)
            or not leaf.get("python_type")
            or leaf.get("native_dtype") != f"float{precision}"
            or metadata.get("dtype") != leaf["native_dtype"]
            or metadata.get("shape") != leaf.get("native_shape")
            or metadata.get("encoding") != "identity"
            or metadata.get("original_dtype") != leaf["native_dtype"]
        ):
            raise ValueError(
                "Processed original-fixture leaf lacks observed native dtype/shape"
            )
        payload = bytes.fromhex(leaf.get("bit_pattern_hex", ""))
        if len(payload) != math.prod(leaf["native_shape"]) * (precision // 8):
            raise ValueError("Processed native bit-pattern inventory is incomplete")


def _require_original_row(*, files, entry, dtype):
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
        "query__floor_used": [],
        "query__corner_continuation": [2],
    }
    integers = {
        "query__node_kind": [120],
        "query__owner_id": [120],
        "query__side_id": [120],
        "query__corner_kind": [2],
        "query__corner_interval_id": [2],
        "query__corner_side_id": [2],
    }
    for name, shape in shapes.items():
        _require_array(
            files=files, filename=entry["file"], name=name, shape=shape, dtype=dtype
        )
    for name, shape in integers.items():
        _require_array(
            files=files, filename=entry["file"], name=name, shape=shape, dtype="int32"
        )
    _require_array(
        files=files,
        filename=entry["file"],
        name="query__link_valid",
        shape=[119],
        dtype="bool",
    )
    present = sorted(name for name in entry["arrays"] if name.startswith("query__"))
    if present != sorted(entry.get("original_query_leaf_names", [])):
        raise ValueError("Original query leaf inventory is incomplete or contradictory")


def source_identity(*, obj, module):
    """Hash a real expected module, unwrapping only for provenance identity."""
    original = inspect.unwrap(obj)
    if inspect.getmodule(original) is not module:
        raise ValueError("Original callable differs from its expected source module")
    path = Path(module.__file__).resolve()
    if not path.is_file() or path.suffix != ".py":
        raise ValueError("Expected module has no real Python source file")
    return str(path), hashlib.sha256(path.read_bytes()).hexdigest()


def bind_source_files(*, files, expected_map, source_root):
    """Bind real in-root sources and authenticate a supplied external map."""
    root = Path(source_root).resolve()
    for filename, digest in files.items():
        path = Path(filename).resolve()
        relative = path.relative_to(root).as_posix()
        if (
            not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest() != digest
        ):
            raise ValueError(
                f"Observed source bytes differ from their digest: {relative}"
            )
        if (
            expected_map is not None
            and expected_map.get(relative, {}).get("sha256") != digest
        ):
            raise ValueError(
                f"Observed module differs from expected source map: {relative}"
            )


@contextmanager
def drain_owned_callbacks(drain):
    """Drain owned callbacks on both exits while preserving the primary exception."""
    try:
        yield
    except BaseException as primary:
        with suppress(BaseException):
            try:
                drain()
            except BaseException as secondary:
                primary.add_note(f"Owned callback drain also failed: {secondary!r}")
                raise
        raise
    else:
        drain()


def _require_components(*, files, solver):
    required = {
        "value-coordinates": {"p1__saving__income", "p1__saving__liquid"},
        "params": {"discount_factor", "tax_rate", "tax_exemption"},
        "income-primitives": {
            "income_codes",
            "transition_rows",
            "income_amounts",
            "declared_floor",
        },
        "initial": {"age", "regime_id", "liquid", "income"},
        "simulation": {"period", "subject_id", "regime_name", "consumption"},
        "period0": {"period", "subject_id", "regime_name", "consumption"},
    }
    for suffix, fields in required.items():
        if not fields.issubset(files.get(f"{solver}-{suffix}.npz", {})):
            raise ValueError(f"Missing complete {solver}:{suffix} capture")
    fields = {"absolute_difference", "violates_original_budget", "original_budget"}
    if not fields.issubset(files.get("original-policy-comparison.npz", {})):
        raise ValueError("Original policy comparison diagnostic missing")


def _require_array(*, files, filename, name, shape, dtype):
    record = files.get(filename, {}).get(name, {})
    if record.get("shape") != shape or record.get("dtype") != dtype:
        raise ValueError(f"Missing or mismatched {filename}:{name}")


def verify_capture_files(*, manifest, root):
    """Verify complete NPZ inventory, hashes, CRC and NPY header shape without NumPy."""
    root = Path(root)
    names = [entry["file"] for entry in manifest["saved"]]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate capture filename")
    processed_bits = {
        (call["file"], leaf["array"]): bytes.fromhex(leaf["bit_pattern_hex"])
        for call in manifest.get("processed_parameter_calls", [])
        for leaf in call["leaves"]
    }
    for entry in manifest["saved"]:
        name = entry["file"]
        if Path(name).name != name:
            raise ValueError("Capture filename escapes its root")
        path = root / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError("Capture NPZ checksum mismatch")
        with zipfile.ZipFile(path) as archive:
            if archive.testzip() is not None:
                raise ValueError("Capture NPZ CRC mismatch")
            if sorted(archive.namelist()) != sorted(
                key + ".npy" for key in entry["arrays"]
            ):
                raise ValueError("Capture NPZ member inventory mismatch")
            for key, record in entry["arrays"].items():
                payload = _verify_npy(data=archive.read(key + ".npy"), record=record)
                if (name, key) in processed_bits and payload != processed_bits[
                    name, key
                ]:
                    raise ValueError(
                        "Processed native leaf bit pattern differs from NPZ bytes"
                    )
    return (
        "NPZ byte inventory admitted; native source and numerical assertions "
        "remain separate"
    )


def _verify_npy(*, data, record):
    if data[:6] != b"\x93NUMPY" or data[6:8] not in (b"\x01\x00", b"\x02\x00"):
        raise ValueError("Unsupported NPY header")
    width = 2 if data[6] == 1 else 4
    size = struct.unpack("<H" if width == 2 else "<I", data[8 : 8 + width])[0]
    header = ast.literal_eval(data[8 + width : 8 + width + size].decode("latin1"))
    if (
        not isinstance(header["descr"], str)
        or list(header["shape"]) != record["shape"]
        or "O" in header["descr"]
    ):
        raise ValueError("NPY shape mismatch or forbidden object payload")
    descriptor = header["descr"].lstrip("<>=|")
    mapping = {
        "f4": "float32",
        "f8": "float64",
        "i4": "int32",
        "i8": "int64",
        "u4": "uint32",
        "u8": "uint64",
        "b1": "bool",
    }
    dtype = mapping.get(descriptor, "<" + descriptor)
    if dtype != record["dtype"]:
        raise ValueError("NPY dtype mismatch")
    if descriptor not in mapping and not (
        descriptor.startswith("U") and descriptor[1:].isdigit()
    ):
        raise ValueError("Unsupported observational NPY dtype")
    itemsize = int(descriptor[1:]) * (4 if descriptor.startswith("U") else 1)
    if len(data) - (8 + width + size) != math.prod(header["shape"]) * itemsize:
        raise ValueError("NPY payload byte length mismatch")
    if (
        record.get("original_dtype") == "object"
        and record.get("encoding") != "json-string-ascii-in-fixed-unicode"
    ):
        raise ValueError("Object-origin column lacks explicit lossless string encoding")
    if record.get("original_dtype") == "object":
        _verify_string_payload(payload=data[8 + width + size :], descriptor=descriptor)
    return data[8 + width + size :]


def _verify_string_payload(*, payload, descriptor):
    if not descriptor.startswith("U"):
        raise ValueError(
            "Object-origin string transport requires fixed Unicode storage"
        )
    width = int(descriptor[1:]) * 4
    if width == 0:
        raise ValueError("Object-origin JSON string storage has zero width")
    for start in range(0, len(payload), width):
        text = payload[start : start + width].decode("utf-32-le").rstrip("\x00")
        if not text.isascii() or not isinstance(json.loads(text), str):
            raise ValueError(
                "Object-origin Unicode payload is not an ASCII JSON string"
            )


def main():
    """Bind one completed observation to verifier-supplied precision and source map."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--source-map", type=Path, required=True)
    parser.add_argument("--precision", type=int, choices=(32, 64), required=True)
    args = parser.parse_args()
    manifest = json.loads((args.capture / "CAPTURE.json").read_text())
    source_sha = hashlib.sha256(args.source_map.read_bytes()).hexdigest()
    sys.stdout.write(
        admit_capture(
            manifest=manifest, precision=args.precision, source_sha=source_sha
        )
        + "\n"
    )
    sys.stdout.write(
        admit_processed_params(manifest=manifest, precision=args.precision) + "\n"
    )
    sys.stdout.write(verify_capture_files(manifest=manifest, root=args.capture) + "\n")


if __name__ == "__main__":
    main()
