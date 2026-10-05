"""Atomically store non-executable period metadata and native numerical leaves."""

import contextlib
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

import h5py
import numpy as np

from _lcm.persistence.solution import (
    _json_object_without_duplicate_keys,
    _lazy_entry,
    _PreparedPayload,
    _read_and_verify_leaves,
    _require_local_dataset,
    _validate_archive_structure,
    _write_payload_entry,
)


def write_period_archive(
    *, path: Path, metadata: dict[str, Any], arrays: Mapping[str, np.ndarray]
) -> str:
    """Publish one complete archive atomically and return its manifest identity."""
    descriptor, filename = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(descriptor)
    temporary = Path(filename)
    try:
        with h5py.File(temporary, "w") as archive:
            payloads = archive.create_group("payloads")
            entries = {}
            for index, (name, array) in enumerate(arrays.items()):
                entries[name] = _write_payload_entry(
                    payloads=payloads,
                    address=f"{index:08d}",
                    prepared=_PreparedPayload(
                        identity=MappingProxyType({"name": name}),
                        payload_kind="array",
                        leaf_paths=((),),
                        leaves=(np.asarray(array),),
                    ),
                )
            encoded = json.dumps(
                {"schema": 1, "metadata": metadata, "arrays": entries},
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
            digest = hashlib.sha256(encoded).hexdigest()
            manifest = archive.create_dataset(
                "manifest", data=np.frombuffer(encoded, dtype=np.uint8)
            )
            manifest.attrs["sha256"] = digest
        with temporary.open("r+b") as stream:
            os.fsync(stream.fileno())
        temporary.replace(path)
        directory_flag = getattr(os, "O_DIRECTORY", None)
        if directory_flag is not None:
            with contextlib.suppress(OSError):
                directory_fd = os.open(path.parent, os.O_RDONLY | directory_flag)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
        return digest
    finally:
        temporary.unlink(missing_ok=True)


def read_period_archive(
    *, path: Path
) -> tuple[dict[str, Any], dict[str, np.ndarray], str]:
    """Verify metadata and native payload checksums before returning host arrays."""
    with h5py.File(path, "r") as archive:
        dataset = _require_local_dataset(
            parent=archive,
            name="manifest",
            label="period manifest",
            allowed_attributes=frozenset({"sha256"}),
        )
        if dataset.ndim != 1 or dataset.dtype != np.dtype("uint8"):
            raise ValueError("Invalid period manifest representation.")
        encoded = dataset[()].tobytes()
        digest = hashlib.sha256(encoded).hexdigest()
        if dataset.attrs.get("sha256") != digest:
            raise ValueError("Period manifest checksum differs.")
    manifest = json.loads(
        encoded, object_pairs_hook=_json_object_without_duplicate_keys
    )
    if (
        type(manifest) is not dict
        or set(manifest) != {"schema", "metadata", "arrays"}
        or type(manifest["schema"]) is not int
        or manifest["schema"] != 1
        or type(manifest["metadata"]) is not dict
        or type(manifest["arrays"]) is not dict
    ):
        raise ValueError("Unsupported period archive schema.")
    entries = {
        name: _lazy_entry(path=path, entry=entry, label=name, identity={"name": name})
        for name, entry in manifest["arrays"].items()
    }
    _validate_archive_structure(path=path, entries=tuple(entries.values()))
    arrays = {
        name: _read_and_verify_leaves(
            path=path,
            label=name,
            address=entry.address,
            identity=entry.identity,
            leaves=entry.leaves,
        )[0]
        for name, entry in entries.items()
    }
    return manifest["metadata"], arrays, digest
