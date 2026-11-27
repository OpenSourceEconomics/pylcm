"""Synthetic metadata controls do not establish native model capture provenance."""

import copy
import hashlib
import zipfile
from pathlib import Path
from typing import cast

import pytest

from tests.solution._nb_scientific_capture_admission import (
    CaptureManifest,
    ProcessedCall,
    ProcessedLeaf,
    SavedRecord,
    admit_processed_params,
    verify_capture_files,
)
from tests.solution.test_nb_capture_admission import make_npy


def _manifest(*, fault: str | None = None) -> CaptureManifest:
    """Describe four synthetic operation records with one declared scalar leaf.

    `fault` names one incomplete or substituted declaration to build in; the
    first record is the nbegm solve, the second the nbegm simulate.
    """
    saved: list[SavedRecord] = []
    calls: list[ProcessedCall] = []
    for solver, identity in (("nbegm", 10), ("grid", 20)):
        for operation in ("solve", "simulate"):
            index = len(calls)
            first, second = index == 0, index == 1
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
            bits = "00000000"
            if first and fault == "bad-bits":
                bits = "00"
            elif second and fault == "values-changed":
                bits = "0000803f"
            leaves: list[ProcessedLeaf] = [
                {
                    "array": "leaf0000",
                    "path": ["saving", "synthetic__coefficient"],
                    "object_id": identity + 2,
                    "python_type": "synthetic.NativeArray",
                    "native_dtype": (
                        "float64" if first and fault == "nominal-float64" else "float32"
                    ),
                    "native_shape": [],
                    "bit_pattern_hex": bits,
                }
            ]
            calls.append(
                {
                    "solver": solver,
                    "operation": operation,
                    "file": filename,
                    "method": (
                        "supplied-user-params"
                        if first and fault == "wrong-method"
                        else "lcm.model.Model._canonicalize_params"
                    ),
                    "model_object_id": (
                        99 if second and fault == "model-changed" else identity
                    ),
                    "returned_mapping_id": identity + 1,
                    "returned_original_mapping": not (
                        first and fault == "replaced-return"
                    ),
                    "leaves": [] if first and fault == "missing-leaf" else leaves,
                }
            )
    if fault == "missing-grid":
        calls.pop()
    elif fault == "duplicate-owner":
        calls[-1] = copy.deepcopy(calls[0])
    return {"saved": saved, "processed_parameter_calls": calls}


def test_four_synthetic_processed_records_are_metadata_only() -> None:
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
def test_incomplete_or_substituted_processed_metadata_is_refused(*, fault: str) -> None:
    """Every original model/operation requires its own complete native declaration."""
    with pytest.raises(
        ValueError, match=r"Both model-owned|Processed |Actual same-model"
    ):
        admit_processed_params(manifest=_manifest(fault=fault), precision=32)


def _payload_manifest(*, path: Path, bits: str) -> CaptureManifest:
    """Declare one saved scalar leaf of `path` with the bit pattern `bits`.

    Only the fields the file check reads are present, so the manifest is cast.
    """
    return cast(
        "CaptureManifest",
        {
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
                    "leaves": [{"array": "leaf0000", "bit_pattern_hex": bits}],
                }
            ],
        },
    )


def test_processed_metadata_bits_are_bound_to_actual_npz_payload(
    tmp_path: Path,
) -> None:
    """A correct checksum cannot hide metadata substitution of one native bit."""
    path = tmp_path / "processed.npz"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "leaf0000.npy", make_npy(descriptor="<f4", shape=(), payload=b"\x00" * 4)
        )
    assert "NPZ byte inventory admitted" in verify_capture_files(
        manifest=_payload_manifest(path=path, bits="00000000"), root=tmp_path
    )
    with pytest.raises(ValueError, match="bit pattern differs"):
        verify_capture_files(
            manifest=_payload_manifest(path=path, bits="01000000"), root=tmp_path
        )
