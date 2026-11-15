"""Source-bound fact replay and supplied-projection transaction controls."""

import json
from pathlib import Path
import sys

import pytest

from _lcm_regime_finder.fibre_source import FibreSource, _mapping
from _lcm_regime_finder.implicit import check_projection as transaction
from _lcm_regime_finder.implicit.compact_screen import PREMISES, Package, ScreenResult
from _lcm_regime_finder.implicit.implicit_core import unpack
from _lcm_regime_finder.implicit.implicit_native import compile_table
from _lcm_regime_finder.records import JSONValue
from tests.regime_finder.test_fibre_source import _fields, _original
from tests.regime_finder.test_model_adapter import function
from tests.regime_finder.test_implicit_version_candidate import installed_candidate


def inputs(tmp_path: Path, *, exclude_zero: bool = False) -> tuple[Path, Path, Path]:
    snapshot, axes, artifact = (
        tmp_path / name for name in ("model.json", "axes.json", "facts.json")
    )
    original = _original()
    if exclude_zero:
        _fields(original)["constraints"] = {
            "allowed": function(
                "def allowed(choice):\n    return choice != 0\n", ["choice"]
            )
        }
    snapshot.write_text(json.dumps(original))
    axes.write_text(json.dumps({"original": ["x", "history"]}))
    packet = compile_table(
        snapshot=snapshot,
        axes_file=axes,
        source_version=installed_candidate().version_id,
    )
    assert packet["status"] == "PRODUCED_UNVERIFIED"
    artifact.write_text(json.dumps(packet))
    return snapshot, axes, artifact


@pytest.mark.parametrize("removed", [frozenset(), frozenset({("original", "history")})])
def test_original_fact_replay_and_supplied_projection(
    tmp_path: Path,
    removed: frozenset[tuple[str, str]],
) -> None:
    snapshot, axes, artifact = inputs(tmp_path)
    result = transaction.check_projection(
        snapshot=snapshot,
        axes_file=axes,
        artifact=artifact,
        removed=removed,
        source_version=installed_candidate().version_id,
    )
    assert result["status"] == "FINITE_PREMISES_PROVED"
    receipt = _mapping(result["fact_verification"])
    assert receipt["status"] == "COMPLETE_FACT_EQUALITY"
    source = FibreSource(_original())
    counts = _mapping(receipt["all_point_counts"])
    assert set(counts) == set(source.get_cell_ids())
    assert all(count == 4 for count in counts.values())
    assert result["source_obligations"] == list(source.source_obligations)
    premises = _mapping(_mapping(result["screen"])["premises"])
    assert set(premises) == PREMISES
    assert all(_mapping(premise)["status"] == "proved" for premise in premises.values())
    assert result["forced_actions"] == []
    assert result["forced_action_savings"] == "not measured"


def test_forged_fact_packet_cannot_get_a_projection_receipt(tmp_path: Path) -> None:
    snapshot, axes, artifact = inputs(tmp_path)
    packet = json.loads(artifact.read_text())
    roots = packet["cells"][0]["diagram"]["roots"]
    assert "direct" in roots
    roots.pop("direct")
    artifact.write_text(json.dumps(packet))
    with pytest.raises(ValueError, match="Missing or invented fact roots"):
        transaction.check_projection(
            snapshot=snapshot,
            axes_file=axes,
            artifact=artifact,
            removed=frozenset(),
            source_version=installed_candidate().version_id,
        )


def test_changed_input_after_replay_cannot_get_a_projection_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot, axes, artifact = inputs(tmp_path)
    screen = transaction.check_screen

    def change_input(
        package: Package, removed: frozenset[tuple[str, str]]
    ) -> ScreenResult:
        result = screen(package, removed)
        axes.write_text("{}")
        return result

    monkeypatch.setattr(transaction, "check_screen", change_input)
    with pytest.raises(ValueError, match="Source/certificate transaction changed"):
        transaction.check_projection(
            snapshot=snapshot,
            axes_file=axes,
            artifact=artifact,
            removed=frozenset(),
            source_version=installed_candidate().version_id,
        )


def test_box_expansion_is_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt = tmp_path / "receipt.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["check_projection", "model", "axes", "facts", "removed", str(receipt)],
    )
    with pytest.raises(SystemExit) as error:
        transaction.main()
    assert error.value.code == 2
    assert not receipt.exists()


@pytest.mark.parametrize(
    "terminal",
    [
        ["Support", [["frozenset", [["bool", False]]], ["bool", True]]],
        [
            "ReadCertificate",
            [["frozenset", [["int", 1]]], ["tuple", []], ["tuple", []]],
        ],
    ],
)
def test_internal_terminal_types_reject_nominal_aliases(terminal: JSONValue) -> None:
    with pytest.raises(ValueError, match="Unknown terminal tag"):
        unpack(terminal)


def test_duplicate_nominal_alias_cannot_get_fact_equality(tmp_path: Path) -> None:
    snapshot, axes, artifact = inputs(tmp_path, exclude_zero=True)
    packet = json.loads(artifact.read_text())
    diagram = packet["cells"][0]["diagram"]
    leaf = diagram["nodes"][diagram["roots"]["exclude:choice"]]["leaf"]
    assert leaf == ["frozenset", [["int", 0]]]
    leaf[1].append(["bool", False])
    artifact.write_text(json.dumps(packet))
    with pytest.raises(ValueError, match="Duplicate set terminal member"):
        transaction.check_projection(
            snapshot=snapshot,
            axes_file=axes,
            artifact=artifact,
            removed=frozenset(),
            source_version=installed_candidate().version_id,
        )


@pytest.mark.parametrize(
    "members",
    [
        [["int", 0], ["bool", False]],
        [["bool", False], ["int", 0]],
        [["int", 0], ["int", 0]],
    ],
)
@pytest.mark.parametrize("tag", ["set", "frozenset"])
def test_independent_decoder_rejects_duplicate_and_alias_members(
    members: list[JSONValue],
    tag: str,
) -> None:
    from _lcm_regime_finder.implicit.verify_native import decode

    with pytest.raises(ValueError, match="Duplicate set terminal member"):
        decode([tag, members])
