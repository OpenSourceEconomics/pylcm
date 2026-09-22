"""Prefix-reuse eligibility controls survive independent source byte resealing."""

import hashlib
from pathlib import Path
from shutil import copyfile

import pytest

from tests.candidate_certificate import direct_flow
from tests.candidate_certificate.generate_sources import sha256_file


def test_grouped_guard_population_is_independent() -> None:
    """Guard controls preserve all five existing mutation populations."""
    root = Path(__file__).parents[1]
    populations = (
        direct_flow.direct_flow_mutation_specs(repo_root=root),
        direct_flow.supplemental_direct_flow_mutation_specs(repo_root=root),
        direct_flow.uniform_process_mutation_specs(repo_root=root),
        direct_flow.action_grid_mutation_specs(repo_root=root),
        direct_flow.grouped_mapper_mutation_specs(repo_root=root),
    )
    assert tuple(map(len, populations)) == (406, 50, 37, 10, 18)
    guard = direct_flow.grouped_guard_mutation_specs(repo_root=root)
    assert len(guard) == 6
    digest = hashlib.sha256(("\n".join(sorted(guard)) + "\n").encode()).hexdigest()
    assert digest == "e866f1cbeda39679ae30891885e1101904d48a5a571add28552fdacc7dd2f5ca"
    previous = set().union(*(set(population) for population in populations))
    assert not set(guard) & previous
    assert {spec["path"] for spec in guard.values()} == {direct_flow.DISPATCHERS_SOURCE}


@pytest.mark.parametrize("mutation", tuple(direct_flow._GROUPED_GUARD_MUTATIONS))
def test_grouped_guard_mutation_fails_after_byte_reseal(
    *, mutation: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The semantic mapper contract rejects altered routing and fallback indexing."""
    root = Path(__file__).parents[1]
    clean = direct_flow.verify_direct_candidate_flow(repo_root=root)
    assert clean["ok"], clean["errors"]
    sources = clean["certified_corridor_sources"]
    for relative in sources:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        copyfile(root / relative, destination)
    spec = direct_flow.grouped_guard_mutation_specs(repo_root=root)[mutation]
    (tmp_path / spec["path"]).write_text(spec["source"], encoding="utf-8")
    monkeypatch.setattr(
        direct_flow,
        "_SOURCE_SEALS",
        {relative: sha256_file(tmp_path / relative) for relative in sources},
    )
    result = direct_flow.verify_direct_candidate_flow(repo_root=tmp_path)
    assert not result["ok"]
    assert result["offending_paths"] == [spec["path"]], result["errors"]
    assert any("grouped mapper" in error for error in result["errors"])
