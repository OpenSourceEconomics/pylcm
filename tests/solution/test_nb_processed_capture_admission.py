"""Synthetic metadata controls do not establish native model capture provenance."""

import copy
import hashlib
import zipfile

import pytest

from tests.solution._nb_scientific_capture_admission import (
    admit_processed_params,
    verify_capture_files,
)
from tests.solution.test_nb_capture_admission import make_npy


def _manifest():
    """Describe four synthetic operation records with one declared scalar leaf."""
    saved, calls = [], []
    for solver, identity in (("nbegm", 10), ("grid", 20)):
        for operation in ("solve", "simulate"):
            filename = f"{solver}-{operation}-processed-params.npz"
            saved.append(
                {
                    "file": filename,
                    "arrays": {
                        "leaf0000": {
                            "shape": [],
                            "dtype": "float32",
                            "original_dtype": "float32",
                            "encoding": "identity",
                        }
                    },
                }
            )
            calls.append(
                {
                    "solver": solver,
                    "operation": operation,
                    "file": filename,
                    "method": "lcm.model.Model._process_params",
                    "model_object_id": identity,
                    "returned_mapping_id": identity + 1,
                    "returned_original_mapping": True,
                    "leaves": [
                        {
                            "array": "leaf0000",
                            "path": ["saving", "synthetic__coefficient"],
                            "object_id": identity + 2,
                            "python_type": "synthetic.NativeArray",
                            "native_dtype": "float32",
                            "native_shape": [],
                            "bit_pattern_hex": "00000000",
                        }
                    ],
                }
            )
    return {"saved": saved, "processed_parameter_calls": calls}


def test_four_synthetic_processed_records_are_metadata_only():
    """Complete metadata has a qualified admission message, not native acceptance."""
    assert "native parity remains separate" in admit_processed_params(
        manifest=_manifest(), precision=32
    )


@pytest.mark.parametrize(
    "fault",
    [
        "missing-grid",
        "duplicate-owner",
        "wrong-method",
        "replaced-return",
        "nominal-float64",
        "missing-leaf",
        "model-changed",
        "bad-bits",
        "values-changed",
    ],
)
def test_incomplete_or_substituted_processed_metadata_is_refused(*, fault):
    """Every original model/operation requires its own complete native declaration."""
    manifest = copy.deepcopy(_manifest())
    calls = manifest["processed_parameter_calls"]
    first = calls[0]
    if fault == "missing-grid":
        calls.pop()
    elif fault == "duplicate-owner":
        calls[-1] = copy.deepcopy(first)
    elif fault == "wrong-method":
        first["method"] = "supplied-user-params"
    elif fault == "replaced-return":
        first["returned_original_mapping"] = False
    elif fault == "nominal-float64":
        first["leaves"][0]["native_dtype"] = "float64"
    elif fault == "missing-leaf":
        first["leaves"] = []
    elif fault == "model-changed":
        calls[1]["model_object_id"] = 99
    elif fault == "bad-bits":
        first["leaves"][0]["bit_pattern_hex"] = "00"
    elif fault == "values-changed":
        calls[1]["leaves"][0]["bit_pattern_hex"] = "0000803f"
    with pytest.raises(
        ValueError, match=r"Both model-owned|Processed |Actual same-model"
    ):
        admit_processed_params(manifest=manifest, precision=32)


def test_processed_metadata_bits_are_bound_to_actual_npz_payload(tmp_path):
    """A correct checksum cannot hide metadata substitution of one native bit."""
    path = tmp_path / "processed.npz"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "leaf0000.npy", make_npy(descriptor="<f4", shape=(), payload=b"\x00" * 4)
        )
    manifest = {
        "saved": [
            {
                "file": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "arrays": {"leaf0000": {"shape": [], "dtype": "float32"}},
            }
        ],
        "processed_parameter_calls": [
            {
                "file": path.name,
                "leaves": [
                    {
                        "array": "leaf0000",
                        "bit_pattern_hex": "00000000",
                    }
                ],
            }
        ],
    }
    assert "NPZ byte inventory admitted" in verify_capture_files(
        manifest=manifest, root=tmp_path
    )
    manifest["processed_parameter_calls"][0]["leaves"][0]["bit_pattern_hex"] = (
        "01000000"
    )
    with pytest.raises(ValueError, match="bit pattern differs"):
        verify_capture_files(manifest=manifest, root=tmp_path)
