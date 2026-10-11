"""Normal support mutations are rejected after independent byte resealing."""

import hashlib
from pathlib import Path
from shutil import copyfile

import pytest

from tests.candidate_certificate import direct_flow
from tests.candidate_certificate.generate_sources import sha256_file


def test_normal_controls_preserve_all_historical_populations() -> None:
    root = Path(__file__).parents[1]
    population = direct_flow.normal_process_mutation_specs(repo_root=root)
    assert len(population) == 17
    assert (
        hashlib.sha256(("\n".join(sorted(population)) + "\n").encode()).hexdigest()
        == direct_flow.EXPECTED_NORMAL_PROCESS_MUTATION_NAMES_SHA256
    )
    historical: set[str] = set()
    for generator in (
        direct_flow.direct_flow_mutation_specs,
        direct_flow.supplemental_direct_flow_mutation_specs,
        direct_flow.uniform_process_mutation_specs,
        direct_flow.action_grid_mutation_specs,
        direct_flow.feasibility_mutation_specs,
        direct_flow.grouped_mapper_mutation_specs,
        direct_flow.grouped_guard_mutation_specs,
        direct_flow.allocation_reservation_mutation_specs,
    ):
        names = set(generator(repo_root=root))
        assert historical.isdisjoint(names)
        assert set(population).isdisjoint(names)
        historical.update(names)
    added_feasibility_controls = {
        "feasibility_summary:published_value_changed",
        "feasibility_summary:empty_count_reversed",
    }
    assert added_feasibility_controls <= historical
    assert len(historical) == 558
    assert len(historical - added_feasibility_controls) == 556
    assert "src/_lcm/solution/feasibility.py" in direct_flow._CERTIFIED_CORRIDOR_SOURCES
    assert len(direct_flow._CERTIFIED_CORRIDOR_SOURCES) == 118
    original_corridor_sources = tuple(
        relative
        for relative in direct_flow._CERTIFIED_CORRIDOR_SOURCES
        if relative != "src/_lcm/solution/feasibility.py"
    )
    assert len(original_corridor_sources) == 117
    # Exact ordered source roster from 0a39539fd628bb60dcae2d3cca601816fe4832f4.
    assert (
        hashlib.sha256(
            ("\n".join(original_corridor_sources) + "\n").encode()
        ).hexdigest()
        == "176867afe4246804efd4231a7a7a8439c6852d2bd099f6f537d3597473d70c85"
    )


@pytest.mark.parametrize("mutation", tuple(direct_flow._NORMAL_PROCESS_MUTATIONS))
def test_normal_mutation_is_rejected_after_independent_byte_reseal(
    *, mutation: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = Path(__file__).parents[1]
    clean = direct_flow.verify_direct_candidate_flow(repo_root=root)
    assert clean["ok"], clean["errors"]
    sources = clean["certified_corridor_sources"]
    for relative in sources:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        copyfile(root / relative, destination)
    spec = direct_flow.normal_process_mutation_specs(repo_root=root)[mutation]
    (tmp_path / spec["path"]).write_text(spec["source"], encoding="utf-8")
    monkeypatch.setattr(
        direct_flow,
        "_SOURCE_SEALS",
        {relative: sha256_file(tmp_path / relative) for relative in sources},
    )
    result = direct_flow.verify_direct_candidate_flow(repo_root=tmp_path)
    assert not result["ok"]
    assert result["offending_paths"] == [spec["path"]], result["errors"]
