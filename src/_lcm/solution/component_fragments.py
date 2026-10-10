"""On-disk task manifests and fragments of independent component jobs.

A component job plan lives in one caller-owned directory:

- `plan.json`: the task manifest, written once.
- `fragments/job-NNNN.h5`: what job `NNNN` published, one checksummed HDF5
  file per job.
- `fragments/job-NNNN.failed.json`: the record of a job that raised.

Every file is written next to its final name and renamed onto it only once it
is complete and flushed, so a reader sees a whole file or none; an interrupted
writer leaves a hidden `.tmp` file that no reader opens.

A fragment holds, per code of its job, the exact host bytes of every value
block, and optionally the rows of its subjects' simulated panel. Each array is
stored with a SHA-256 over its logical address, shape, dtype and bytes, and
the JSON manifest with a SHA-256 of its own bytes. Reading a fragment verifies
all of them before any array is returned. No model, callable or Python object
is serialized.
"""

import contextlib
import hashlib
import json
import os
import re
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final, cast

import h5py
import numpy as np

from _lcm.persistence.solution import (
    _array_checksum,
    _exact_shape,
    _require_exact_dict,
    _require_exact_list,
    _require_nonempty_exact_str,
    _require_nonnegative_exact_int,
    _require_positive_exact_int,
)
from _lcm.typing import JSONValue, RegimeName
from lcm.exceptions import SolutionIntegrityError

PLAN_FILE: Final = "plan.json"
FRAGMENT_DIRECTORY: Final = "fragments"
PLAN_FORMAT: Final = "pylcm-component-job-plan"
FRAGMENT_FORMAT: Final = "pylcm-component-fragment"
FORMAT_VERSION: Final = 2

_MANIFEST: Final = "manifest"
_VALUES: Final = "values"
_PANEL: Final = "panel"
_FRAGMENT_NAME: Final = re.compile(r"job-(\d{4,})\.h5")
_FAILURE_NAME: Final = re.compile(r"job-(\d{4,})\.failed\.json")
_SHA256_HEX_LENGTH: Final = 64

type Coordinate = tuple[int, RegimeName]
type LeafAddress = tuple[RegimeName, int, str, str | None]


@dataclass(frozen=True, kw_only=True, eq=False)
class FragmentPanel:
    """The simulated rows one job holds."""

    n_subjects: int
    """Number of real subjects in the whole population."""

    subject_batch_size: int
    """Rows of every chunk the job dispatched."""

    rows: np.ndarray
    """Original rows the job simulated, ascending."""

    regimes: tuple[tuple[RegimeName, tuple[int, ...]], ...]
    """Each regime of the raw results and its periods, in their order."""

    leaves: MappingProxyType[LeafAddress, np.ndarray]
    """Per `(regime, period, field, key)`, the rows of one raw-result leaf.

    `key` names an entry of a mapping field (`actions`, `states`) and is `None`
    for an array field. Leaves are in raw-result order.
    """

    layouts: MappingProxyType[LeafAddress, MappingProxyType[str, JSONValue]]
    """Per leaf, the JSON description of the device layout the job produced it on.

    Devices are positions in the model's selected devices, never device ids.
    """


@dataclass(frozen=True, kw_only=True, eq=False)
class Fragment:
    """One verified fragment."""

    path: Path
    """Where the fragment was read from."""

    job: int
    """The job that published it."""

    codes: tuple[int, ...]
    """The codes it covers, in grid order."""

    plan_sha256: str
    """Digest of the plan file the job ran."""

    plan_id: str
    """Identifier of the plan the job ran."""

    identity: MappingProxyType[str, JSONValue]
    """The model, parameter and build identity the job reproduced."""

    execution: MappingProxyType[str, JSONValue]
    """Backend, devices and budget the job ran on."""

    coordinates: tuple[Coordinate, ...]
    """Every solved `(period, regime)`, in the order the solve published them."""

    values: MappingProxyType[int, MappingProxyType[Coordinate, np.ndarray]]
    """Per code, its value block at each coordinate, in `coordinates` order."""

    panel: FragmentPanel | None
    """The job's simulated rows, or `None` when the plan does not simulate."""


def fragment_name(*, job: int) -> str:
    """Return the file name of a job's fragment."""
    return f"job-{job:04d}.h5"


def failure_name(*, job: int) -> str:
    """Return the file name of a job's failure record."""
    return f"job-{job:04d}.failed.json"


def fragment_job(*, name: str) -> int | None:
    """Return the job a fragment file name belongs to, or `None` for another name."""
    match = _FRAGMENT_NAME.fullmatch(name)
    return None if match is None else int(match.group(1))


def failure_job(*, name: str) -> int | None:
    """Return the job a failure-record file name belongs to, or `None`."""
    match = _FAILURE_NAME.fullmatch(name)
    return None if match is None else int(match.group(1))


def canonical_json(payload: JSONValue) -> bytes:
    """Encode JSON deterministically: sorted keys, no whitespace, no NaN."""
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def sha256_hex(payload: bytes) -> str:
    """Return the SHA-256 of `payload` as hex."""
    return hashlib.sha256(payload).hexdigest()


def write_atomically(*, path: Path, write: Callable[[Path], None]) -> None:
    """Write a file next to `path`, flush it, then rename it onto `path`.

    `write` fills the temporary file it is given. A reader of `path` sees the
    previous file or the complete new one; a failure leaves neither a partial
    `path` nor the temporary file.

    The file keeps the permissions `tempfile.mkstemp` gives the temporary file:
    readable and writable by its owner only (`0600`).
    """
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    temporary: Path | None = Path(temporary_name)
    try:
        write(temporary)
        # Opened for writing without truncation: the flush needs a writable
        # handle, since Windows refuses to flush a read-only one.
        with temporary.open("r+b") as handle:
            os.fsync(handle.fileno())
        temporary.replace(path)
        temporary = None
        _fsync_directory(directory=path.parent)
    finally:
        if temporary is not None:
            with contextlib.suppress(OSError):
                temporary.unlink()


def write_json_atomically(*, path: Path, payload: JSONValue) -> None:
    """Write `payload` as canonical JSON to `path` through an atomic rename."""
    encoded = canonical_json(payload)
    write_atomically(path=path, write=_BytesWriter(payload=encoded))


@dataclass(frozen=True, kw_only=True)
class _BytesWriter:
    """Write one encoded manifest without a per-call function closure."""

    payload: bytes

    def __call__(self, path: Path) -> None:
        """Write the encoded manifest to its temporary path."""
        path.write_bytes(self.payload)


def array_checksum(*, identity: Mapping[str, JSONValue], array: np.ndarray) -> str:
    """Hash one array together with its logical address and representation."""
    return _array_checksum(identity=dict(identity), array=np.ascontiguousarray(array))


@dataclass(frozen=True, kw_only=True)
class _FragmentWriter:
    """Write one fragment without retaining its arrays in a function closure."""

    header: Mapping[str, JSONValue]
    coordinates: tuple[Coordinate, ...]
    values: Mapping[int, Mapping[Coordinate, np.ndarray]]
    panel: FragmentPanel | None

    def __call__(self, path: Path) -> None:
        """Write the complete fragment to its temporary path."""
        _write_fragment_file(
            path=path,
            header=self.header,
            coordinates=self.coordinates,
            values=self.values,
            panel=self.panel,
        )


def write_fragment(
    *,
    path: Path,
    header: Mapping[str, JSONValue],
    coordinates: tuple[Coordinate, ...],
    values: Mapping[int, Mapping[Coordinate, np.ndarray]],
    panel: FragmentPanel | None,
) -> None:
    """Publish one job's fragment at `path` through an atomic rename.

    Args:
        path: The fragment's final path.
        header: The manifest entries that describe the job: `plan_sha256`,
            `plan_id`, `job`, `codes`, `identity` and `execution`.
        coordinates: Every solved coordinate in publication order.
        values: Per code, its host block at every coordinate.
        panel: The job's simulated rows, or `None`.

    """
    write_atomically(
        path=path,
        write=_FragmentWriter(
            header=header,
            coordinates=coordinates,
            values=values,
            panel=panel,
        ),
    )


def read_fragment(*, path: Path) -> Fragment:
    """Read and verify one fragment.

    Raises:
        SolutionIntegrityError: The file is not a readable HDF5 file, or its
            manifest, its structure or any array fails verification. The
            message names the file.

    """
    try:
        with h5py.File(path, "r") as file:
            return _read_fragment_file(path=path, file=file)
    except SolutionIntegrityError:
        raise
    except (OSError, KeyError, ValueError, TypeError) as error:
        msg = f"Fragment {path.name} cannot be read as a complete fragment: {error}"
        raise SolutionIntegrityError(msg) from error


def _write_fragment_file(
    *,
    path: Path,
    header: Mapping[str, JSONValue],
    coordinates: tuple[Coordinate, ...],
    values: Mapping[int, Mapping[Coordinate, np.ndarray]],
    panel: FragmentPanel | None,
) -> None:
    """Write the complete fragment to a path that is not yet public."""
    manifest: dict[str, JSONValue] = {
        **dict(header),
        "format": FRAGMENT_FORMAT,
        "format_version": FORMAT_VERSION,
        "coordinates": [[int(period), str(regime)] for period, regime in coordinates],
        "values": [],
        "panel": None,
    }
    with h5py.File(path, "w") as file:
        value_group = file.create_group(_VALUES)
        value_entries = cast("list[JSONValue]", manifest["values"])
        for code, blocks in values.items():
            for (period, regime), block in blocks.items():
                address = f"{len(value_entries):06d}"
                identity = {
                    "kind": "value",
                    "code": int(code),
                    "period": int(period),
                    "regime": str(regime),
                }
                value_entries.append(
                    _write_leaf(
                        group=value_group,
                        address=address,
                        identity=identity,
                        array=block,
                    )
                )
        if panel is not None:
            panel_group = file.create_group(_PANEL)
            manifest["panel"] = {
                "n_subjects": panel.n_subjects,
                "subject_batch_size": panel.subject_batch_size,
                "regimes": [
                    [regime, list(periods)] for regime, periods in panel.regimes
                ],
                "rows": _write_leaf(
                    group=panel_group,
                    address="rows",
                    identity={"kind": "rows"},
                    array=panel.rows,
                ),
                "leaves": [
                    {
                        **_write_leaf(
                            group=panel_group,
                            address=f"{index:06d}",
                            identity={
                                "kind": "panel",
                                "regime": regime,
                                "period": period,
                                "field": field_name,
                                "key": key,
                            },
                            array=array,
                        ),
                        "layout": dict(
                            panel.layouts[(regime, period, field_name, key)]
                        ),
                    }
                    for index, ((regime, period, field_name, key), array) in enumerate(
                        panel.leaves.items()
                    )
                ],
            }
        payload = canonical_json(manifest)
        dataset = file.create_dataset(
            _MANIFEST, data=np.frombuffer(payload, dtype=np.uint8)
        )
        dataset.attrs["sha256"] = sha256_hex(payload)
        file.flush()


def _write_leaf(
    *,
    group: h5py.Group,
    address: str,
    identity: Mapping[str, JSONValue],
    array: np.ndarray,
) -> dict[str, JSONValue]:
    """Write one array and return its manifest entry."""
    contiguous = np.ascontiguousarray(array)
    group.create_dataset(address, data=contiguous)
    return {
        "dataset": f"{group.name.lstrip('/')}/{address}",
        "identity": dict(identity),
        "shape": list(contiguous.shape),
        "dtype": contiguous.dtype.str,
        "sha256": array_checksum(identity=identity, array=contiguous),
    }


def _read_fragment_file(*, path: Path, file: h5py.File) -> Fragment:
    """Verify one open fragment and return its contents."""
    manifest = _read_manifest(path=path, file=file)
    listed: set[str] = set()
    coordinates = tuple(
        (
            _require_nonnegative_exact_int(value=period, label="fragment period"),
            _require_nonempty_exact_str(value=regime, label="fragment regime"),
        )
        for period, regime in (
            _require_exact_list(value=coordinate, label="fragment coordinate")
            for coordinate in _require_exact_list(
                value=manifest["coordinates"], label="fragment coordinates"
            )
        )
    )
    codes = tuple(
        _require_nonnegative_exact_int(value=code, label="fragment code")
        for code in _require_exact_list(value=manifest["codes"], label="fragment codes")
    )
    blocks: dict[int, dict[Coordinate, np.ndarray]] = {code: {} for code in codes}
    for entry in cast("list[dict[str, JSONValue]]", manifest["values"]):
        identity = _require_exact_dict(
            value=entry["identity"], label=f"fragment {path.name} value address"
        )
        if (
            set(identity) != {"kind", "code", "period", "regime"}
            or identity["kind"] != "value"
        ):
            msg = f"Fragment {path.name} has an invalid value address {identity!r}."
            raise SolutionIntegrityError(msg)
        code = _require_nonnegative_exact_int(
            value=identity["code"], label=f"fragment {path.name} value code"
        )
        if code not in blocks:
            msg = (
                f"Fragment {path.name} holds a value of code {code} it does not cover."
            )
            raise SolutionIntegrityError(msg)
        coordinate = (
            _require_nonnegative_exact_int(
                value=identity["period"], label=f"fragment {path.name} value period"
            ),
            _require_nonempty_exact_str(
                value=identity["regime"], label=f"fragment {path.name} value regime"
            ),
        )
        if coordinate in blocks[code]:
            msg = (
                f"Fragment {path.name} publishes duplicate value address "
                f"{(code, *coordinate)!r}."
            )
            raise SolutionIntegrityError(msg)
        blocks[code][coordinate] = _read_leaf(
            path=path, file=file, entry=entry, listed=listed
        )
    for code, code_blocks in blocks.items():
        if tuple(code_blocks) != coordinates:
            msg = (
                f"Fragment {path.name} lists the values of code {code} at "
                f"{tuple(code_blocks)!r}, not at its coordinates {coordinates!r}."
            )
            raise SolutionIntegrityError(msg)
    panel_entry = manifest["panel"]
    panel = (
        None
        if panel_entry is None
        else _read_panel(
            path=path,
            file=file,
            entry=cast("dict[str, JSONValue]", panel_entry),
            listed=listed,
        )
    )
    stored = _dataset_names(file=file) - {_MANIFEST}
    if stored != listed:
        msg = (
            f"Fragment {path.name} stores datasets its manifest does not list "
            f"({sorted(stored - listed)!r}) or lists datasets it does not store "
            f"({sorted(listed - stored)!r})."
        )
        raise SolutionIntegrityError(msg)
    return Fragment(
        path=path,
        job=_require_nonnegative_exact_int(value=manifest["job"], label="fragment job"),
        codes=codes,
        plan_sha256=str(manifest["plan_sha256"]),
        plan_id=str(manifest["plan_id"]),
        identity=MappingProxyType(cast("dict[str, JSONValue]", manifest["identity"])),
        execution=MappingProxyType(cast("dict[str, JSONValue]", manifest["execution"])),
        coordinates=coordinates,
        values=MappingProxyType(
            {
                code: MappingProxyType(code_blocks)
                for code, code_blocks in blocks.items()
            }
        ),
        panel=panel,
    )


_MANIFEST_KEYS: Final = frozenset(
    {
        "format",
        "format_version",
        "plan_sha256",
        "plan_id",
        "job",
        "codes",
        "identity",
        "execution",
        "coordinates",
        "values",
        "panel",
    }
)


def _read_manifest(*, path: Path, file: h5py.File) -> dict[str, JSONValue]:
    """Verify and parse a fragment's manifest."""
    dataset = file[_MANIFEST]
    if not isinstance(dataset, h5py.Dataset):
        msg = f"Fragment {path.name} has no manifest dataset."
        raise SolutionIntegrityError(msg)
    payload = bytes(np.asarray(dataset[()], dtype=np.uint8))
    expected = dataset.attrs.get("sha256")
    if not isinstance(expected, str) or sha256_hex(payload) != expected:
        msg = f"Fragment {path.name} fails its manifest checksum."
        raise SolutionIntegrityError(msg)
    manifest = json.loads(payload)
    if not isinstance(manifest, dict) or set(manifest) != _MANIFEST_KEYS:
        msg = f"Fragment {path.name} does not have the fragment manifest schema."
        raise SolutionIntegrityError(msg)
    format_version = _require_nonnegative_exact_int(
        value=manifest["format_version"], label=f"fragment {path.name} format version"
    )
    if (manifest["format"], format_version) != (
        FRAGMENT_FORMAT,
        FORMAT_VERSION,
    ):
        msg = (
            f"Fragment {path.name} has format {manifest['format']!r} version "
            f"{manifest['format_version']!r}; this pylcm reads {FRAGMENT_FORMAT!r} "
            f"version {FORMAT_VERSION}."
        )
        raise SolutionIntegrityError(msg)
    return manifest


def _read_leaf(
    *, path: Path, file: h5py.File, entry: Mapping[str, JSONValue], listed: set[str]
) -> np.ndarray:
    """Read one listed array and verify its shape, dtype and checksum."""
    name = str(entry["dataset"])
    if name in listed:
        msg = f"Fragment {path.name} references dataset {name!r} twice."
        raise SolutionIntegrityError(msg)
    listed.add(name)
    dataset = file[name]
    if not isinstance(dataset, h5py.Dataset):
        msg = f"Fragment {path.name} lists {name!r}, which is not a dataset."
        raise SolutionIntegrityError(msg)
    array = np.asarray(dataset[()])
    shape = _exact_shape(
        value=entry["shape"], label=f"fragment {path.name} leaf {name!r}"
    )
    checksum = entry["sha256"]
    if (
        array.shape != shape
        or array.dtype.str != entry["dtype"]
        or not isinstance(checksum, str)
        or len(checksum) != _SHA256_HEX_LENGTH
        or array_checksum(
            identity=cast("Mapping[str, JSONValue]", entry["identity"]), array=array
        )
        != checksum
    ):
        msg = (
            f"Fragment {path.name} fails the checksum of {name!r} "
            f"({entry['identity']!r})."
        )
        raise SolutionIntegrityError(msg)
    return array


def _read_panel(
    *, path: Path, file: h5py.File, entry: Mapping[str, JSONValue], listed: set[str]
) -> FragmentPanel:
    """Read and verify a fragment's simulated rows."""
    rows = _read_leaf(
        path=path,
        file=file,
        entry=cast("dict[str, JSONValue]", entry["rows"]),
        listed=listed,
    )
    if rows.ndim != 1 or rows.dtype != np.dtype(np.int64):
        msg = f"Fragment {path.name} requires one-dimensional int64 original rows."
        raise SolutionIntegrityError(msg)
    leaves: dict[LeafAddress, np.ndarray] = {}
    layouts: dict[LeafAddress, MappingProxyType[str, JSONValue]] = {}
    for leaf in cast("list[dict[str, JSONValue]]", entry["leaves"]):
        identity = _require_exact_dict(value=leaf["identity"], label="raw leaf address")
        if (
            set(identity) != {"kind", "regime", "period", "field", "key"}
            or identity["kind"] != "panel"
        ):
            msg = f"Fragment {path.name} has an invalid raw leaf address {identity!r}."
            raise SolutionIntegrityError(msg)
        key = identity["key"]
        address = (
            _require_nonempty_exact_str(value=identity["regime"], label="raw regime"),
            _require_nonnegative_exact_int(
                value=identity["period"], label="raw period"
            ),
            _require_nonempty_exact_str(value=identity["field"], label="raw field"),
            None
            if key is None
            else _require_nonempty_exact_str(value=key, label="raw key"),
        )
        if address in leaves:
            msg = f"Fragment {path.name} publishes duplicate panel address {address!r}."
            raise SolutionIntegrityError(msg)
        leaves[address] = _read_leaf(path=path, file=file, entry=leaf, listed=listed)
        layouts[address] = MappingProxyType(
            _require_exact_dict(value=leaf["layout"], label="raw leaf layout")
        )
    return FragmentPanel(
        n_subjects=_require_nonnegative_exact_int(
            value=entry["n_subjects"], label="raw population count"
        ),
        subject_batch_size=_require_positive_exact_int(
            value=entry["subject_batch_size"], label="raw chunk width"
        ),
        rows=rows,
        regimes=tuple(
            (
                _require_nonempty_exact_str(value=regime, label="raw regime"),
                tuple(
                    _require_nonnegative_exact_int(value=period, label="raw period")
                    for period in _require_exact_list(
                        value=periods, label="raw periods"
                    )
                ),
            )
            for regime, periods in cast("list[list[JSONValue]]", entry["regimes"])
        ),
        leaves=MappingProxyType(leaves),
        layouts=MappingProxyType(layouts),
    )


def _dataset_names(*, file: h5py.File) -> set[str]:
    """Return the path of every dataset in the file."""
    collector = _DatasetNames(names=set())
    file.visititems(collector)
    return collector.names


@dataclass(frozen=True, kw_only=True)
class _DatasetNames:
    """Collect dataset addresses during one HDF5 traversal."""

    names: set[str]

    # keyword-only-exempt: library-callback=h5py.Group.visititems
    def __call__(
        self, name: str, item: h5py.Group | h5py.Dataset | h5py.Datatype
    ) -> None:
        """Record the addresses of numerical datasets."""
        if isinstance(item, h5py.Dataset):
            self.names.add(name)


def _fsync_directory(*, directory: Path) -> None:
    """Flush a directory entry where the platform supports it."""
    flag = getattr(os, "O_DIRECTORY", None)
    if flag is None:
        return
    with contextlib.suppress(OSError):
        descriptor = os.open(directory, os.O_RDONLY | flag)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
