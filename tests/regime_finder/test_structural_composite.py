"""Generated support joins, not an economic reachability oracle."""

import pytest
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
import tempfile
import importlib.util

type StateName = str
type ActionName = str

from _lcm_regime_finder.structural_store import CaptureStore
from _lcm_regime_finder.structural_composite import check_composite


@dataclass(frozen=True)
class ReachableFixture:
    state_names: tuple[StateName, ...]
    by_node: MappingProxyType[tuple[int, str], frozenset[tuple[int, ...]]]
    union: tuple[tuple[int, ...], ...]
    carriers: frozenset[str]


def _capture(path, *, bad_union=False, bad_table=False, bad_encoder=False):
    import numpy as np

    union = ((0, 0), (1, 2))
    reach = ReachableFixture(
        ("a", "b"),
        MappingProxyType(
            {
                (0, "work"): frozenset({union[0]}),
                (1, "work"): frozenset(union),
                (2, "work"): frozenset(),
            }
        ),
        union[:1] if bad_union else union,
        frozenset({"work"}),
    )
    table = np.asarray(union, dtype="<i4")
    encoder = np.full((2, 3), -(2**31), dtype="<i4")
    encoder[0, 0], encoder[1, 2] = 0, 1
    if bad_table:
        table[1, 1] = 1
    if bad_encoder:
        encoder[1, 1], encoder[1, 2] = 1, -(2**31)
    store = CaptureStore(path, check=lambda: None, source_context={"generated": True})
    metadata = {
        "fields": {
            "members": ["a", "b"],
            "carriers": ["work"],
            "reachable": {"structural_backing": store.reference(reach)},
            "table": store.reference(table),
            "encoding": store.reference(encoder),
        }
    }
    store.finish()
    return store, metadata


@pytest.mark.skipif(
    not (importlib.util.find_spec("numpy")), reason="actual NumPy required"
)
def test_actual_stored_rows_nodes_table_and_encoder_agree() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        owner, metadata = _capture(Path(tmp) / "input.json")
        result = check_composite(metadata, missing_code=-(2**31))
        assert (result["status"]) == ("PASS")
        assert (result["nodes"]) == (3)
        assert (owner.reader.row_cache) == ({})
        assert (result["economic_reachability"]) == ("UNRESOLVED")


@pytest.mark.skipif(
    not (importlib.util.find_spec("numpy")), reason="actual NumPy required"
)
def test_internally_consistent_capture_does_not_hide_invalid_joins(
    *, subtests: pytest.Subtests
) -> None:
    for defect in ("bad_union", "bad_table", "bad_encoder"):
        with subtests.test(defect=defect), tempfile.TemporaryDirectory() as tmp:
            owner, metadata = _capture(Path(tmp) / "input.json", **{defect: True})
            with pytest.raises(ValueError):
                check_composite(metadata, missing_code=-(2**31))
