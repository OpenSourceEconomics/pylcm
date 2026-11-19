"""The independent layout check binds each block's exported payload and the lane listing.

A witness block exports its periods, its states with their grids, its actions, its
excluded actions and its fibres; the layout result also lists every dense regime's
lanes. Users read these fields, so the layout check derives them from the kept fibres
with its own code and rejects a report that says otherwise, even when every price in it
is genuine. The order in which a block lists its fibres carries no meaning and is free.
"""

from collections.abc import Callable
from copy import deepcopy
from functools import cache
from typing import Literal

import pytest

from _lcm_regime_finder import find_regimes, verify_certificate
from _lcm_regime_finder.facts import FactsBackendName, Record
from _lcm_regime_finder.model_snapshot import read_model
from lcm import Model
from lcm.typing import ActionName, StateName
from tests.collective_fixtures import make_two_stakeholder_model
from tests.solution import test_renegotiation as renegotiation

BACKENDS: tuple[FactsBackendName, ...] = ("dense", "metadata")

type Label = Literal["two_stakeholder", "renegotiate_2p"]
type Table = Literal["exact", "curve"]

_MODELS: dict[Label, Callable[[], Model]] = {
    "two_stakeholder": lambda: make_two_stakeholder_model()[0],
    "renegotiate_2p": lambda: renegotiation._model(
        values=renegotiation._VALUES, n_periods=2
    ),
}

# The block field each alteration changes, as the layout check names it.
_FIELD = {
    "drop_state": "states",
    "shrink_state": "states",
    "wrong_codes": "states",
    "drop_action": "actions",
    "periods": "periods",
    "excluded": "excluded_actions",
    "fibres": "fibres",
    "duplicate_fibre": "fibres",
}


@cache
def _captured(label: Label) -> Record:
    return read_model(_MODELS[label]())


@cache
def _report(label: Label, backend: FactsBackendName) -> Record:
    return find_regimes(_captured(label), facts_backend=backend)


def _verify(label: Label, report: Record) -> Record:
    return verify_certificate(_captured(label), report)


def _blocks(report: Record) -> list[Record]:
    return [
        block
        for table in ("exact", "curve")
        for point in report["layouts"][table]
        for group in point["layout"]
        for block in group["blocks"]
    ]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("label", ["two_stakeholder", "renegotiate_2p"])
def test_genuine_block_payloads_pass(label: Label, backend: FactsBackendName) -> None:
    """The finder's own blocks and lanes pass; the source verdict is the fixture's own."""
    verified = _verify(label, _report(label, backend))
    expected = "PASS" if label == "two_stakeholder" else "UNRESOLVED"
    assert (verified["layout_check"]["status"], verified["status"]) == (
        "PASS",
        expected,
    )


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("table", ["exact", "curve"])
@pytest.mark.parametrize(
    ("label", "state", "action"),
    [("two_stakeholder", "wage", "work"), ("renegotiate_2p", "mu", "choice")],
)
@pytest.mark.parametrize("mutation", list(_FIELD))
def test_block_with_altered_payload_fails_the_layout_check(
    *,
    backend: FactsBackendName,
    table: Table,
    label: Label,
    state: StateName,
    action: ActionName,
    mutation: str,
) -> None:
    """Each exported block field is checked, while the block's prices stay genuine."""
    original = _report(label, backend)
    genuine = _verify(label, original)
    altered = deepcopy(original)
    block = next(
        block
        for point in altered["layouts"][table]
        for group in point["layout"]
        for block in group["blocks"]
        if state in block["states"]
    )
    prices = block["evaluations"], block["storage"]
    match mutation:
        case "drop_state":
            del block["states"][state]
        case "shrink_state":
            assert block["states"][state] > 1
            block["states"][state] = 1
        case "wrong_codes":
            # As many codes as the grid has values, none of them the grid's own.
            block["states"][state] = list(range(100, 100 + block["states"][state]))
        case "drop_action":
            del block["actions"][action]
        case "periods":
            assert block["periods"]
            block["periods"] = []
        case "excluded":
            assert action not in block["excluded_actions"]
            block["excluded_actions"][action] = [0]
        case "fibres":
            assert block["fibres"]
            block["fibres"] = []
        case "duplicate_fibre":
            block["fibres"].append(deepcopy(block["fibres"][0]))
    verified = _verify(label, altered)
    assert (
        (block["evaluations"], block["storage"]) == prices,
        genuine["layout_check"]["status"],
        verified["source_semantics"]["status"],
        verified["layout_check"]["status"],
        verified["status"],
        any(
            f" {_FIELD[mutation]} differ from the independently" in error
            for error in verified["layout_check"]["errors"]
        ),
    ) == (True, "PASS", genuine["source_semantics"]["status"], "FAIL", "FAIL", True)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("mutation", ["renegotiated_status", "dropped_lane"])
def test_altered_lane_listing_fails_the_layout_check(
    backend: FactsBackendName, mutation: str
) -> None:
    """The lane listing can neither split the renegotiated state nor omit a lane."""
    label = "renegotiate_2p"
    altered = deepcopy(_report(label, backend))
    lanes = next(
        lanes
        for lanes in altered["layouts"]["lanes"]
        if any(lane["regime"] == "couple" for lane in lanes)
    )
    couple = next(lane for lane in lanes if lane["regime"] == "couple")
    match mutation:
        case "renegotiated_status":
            assert "mu" not in couple["statuses"]
            couple["statuses"]["mu"] = 0
        case "dropped_lane":
            lanes.remove(couple)
    verified = _verify(label, altered)
    assert (
        verified["layout_check"]["status"],
        "reported lanes differ from the independently derived lanes"
        in verified["layout_check"]["errors"],
    ) == ("FAIL", True)


@pytest.mark.parametrize("backend", BACKENDS)
def test_reordered_fibre_listing_passes(backend: FactsBackendName) -> None:
    """The check binds which fibres a block holds, not the order it lists them in."""
    # The two-stakeholder blocks hold one fibre each, so only this fixture reorders.
    label = "renegotiate_2p"
    original = _report(label, backend)
    altered = deepcopy(original)
    for block in _blocks(altered):
        block["fibres"].reverse()
    verified = _verify(label, altered)
    assert (
        altered != original,
        verified["layout_check"]["status"],
        verified["status"],
    ) == (True, "PASS", _verify(label, original)["status"])
