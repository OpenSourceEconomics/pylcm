"""File-store trust boundaries and non-materializing structural validation."""

import pytest
from pathlib import Path
import tempfile

from _lcm_regime_finder.structural_transport import Writer, reseal
from _lcm_regime_finder.structural_check import Reader
from _lcm_regime_finder.structural_files import (
    FilePayloads,
    load_store,
    verify_store_again,
)


def test_packed_rows_validate_without_expanding_python_tuple_objects() -> None:
    source = [tuple([i, i + 1]) for i in range(9000)]
    bundle = Writer().finish(source, {})
    reader = Reader(bundle)
    result = reader.validate_graph()
    assert (result["row_objects"]) == (9000)
    assert (reader.row_cache) == ({})


def test_file_store_matches_memory_store_and_final_change_rejected() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        writer = Writer()
        writer.payloads = FilePayloads(tmp, writable=True)
        bundle = writer.finish([tuple([2, 4]), bytearray(b"original")], {})
        reader = load_store(bundle["manifest"], tmp)
        assert (reader.decode()) == ([(2, 4), bytearray(b"original")])
        verify_store_again(reader)
        key = next(iter(writer.payloads))
        writer.payloads.path(key).write_bytes(b"corruption")
        with pytest.raises(ValueError):
            verify_store_again(reader)


def test_paths_and_chunk_overwrite_are_rejected() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = FilePayloads(tmp, writable=True)
        for name in ("../p0", "/p0", "p0/elsewhere", "p-1"):
            with pytest.raises(ValueError):
                store[name] = b"bad"
        store["p0"] = b"first"
        with pytest.raises(ValueError):
            store["p0"] = b"second"


def test_chunk_symlink_is_not_followed() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        Path(tmp, "external").write_bytes(b"secret")
        Path(tmp, "p0.bin").symlink_to(Path(tmp, "external"))
        with pytest.raises(ValueError):
            FilePayloads(tmp, ["p0"])["p0"]


def test_rehashed_duplicate_set_row_does_not_validate() -> None:
    bundle = Writer().finish({tuple([0]), tuple([1])}, {})
    key = bundle["manifest"]["records"][0][2]
    bundle["payloads"][key] = b"\0" * 16
    reseal(bundle=bundle)
    with pytest.raises(ValueError):
        Reader(bundle).validate_graph()


def test_large_ordered_scalar_payload_does_not_expand_header() -> None:
    values = [str(i) for i in range(9000)]
    bundle = Writer().finish(values, {})
    assert (len(bundle["manifest"]["records"][0])) == (3)
    reader = Reader(bundle)
    reader.validate_graph()
    assert (reader.decode()) == (values)
