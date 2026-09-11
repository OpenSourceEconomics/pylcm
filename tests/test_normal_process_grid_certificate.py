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
    assert len(historical) == 552
    assert len(direct_flow._CERTIFIED_CORRIDOR_SOURCES) == 116


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
