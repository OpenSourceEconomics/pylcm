"""The anchor check names a mutation whose literal no longer resolves once."""

from pathlib import Path
from shutil import copyfile

import pytest

from tests.candidate_certificate import check_mutation_anchors


def _tree_with_anchored_sources(destination: Path) -> None:
    for relative, _old in check_mutation_anchors.collect_anchors().values():
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        copyfile(check_mutation_anchors.ROOT / relative, target)


def test_check_mutation_anchors_reports_a_duplicated_anchor(
    *,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    """Duplicating one anchored snippet is reported by the mutation's name."""
    _tree_with_anchored_sources(tmp_path)
    name = "solution_runtime:fixed_owner_inventory_omitted"
    relative, old = check_mutation_anchors.collect_anchors()[name]
    path = tmp_path / relative
    path.write_text(path.read_text(encoding="utf-8") + "\n" + old, encoding="utf-8")
    monkeypatch.setattr(check_mutation_anchors, "ROOT", tmp_path)
    monkeypatch.setattr(
        check_mutation_anchors.direct_flow,
        "direct_flow_mutation_specs",
        lambda **_kwargs: {},
    )
    monkeypatch.setattr(
        check_mutation_anchors.direct_flow,
        "supplemental_direct_flow_mutation_specs",
        lambda **_kwargs: {},
    )

    status = check_mutation_anchors.main()

    assert (
        status,
        f"{name}: {relative} holds the anchor 2 times" in capsys.readouterr().err,
    ) == (1, True)


def test_check_mutation_anchors_passes_on_the_live_tree():
    """Every registered anchor occurs exactly once in the checkout."""
    assert check_mutation_anchors.main() == 0
