"""The action-partitioned GridSearch route is a certified corridor with controls."""

import hashlib
from pathlib import Path
from shutil import copyfile

import pytest

from tests.candidate_certificate import direct_flow
from tests.candidate_certificate.generate_sources import sha256_file


def test_action_partition_controls_are_complete_and_disjoint() -> None:
    """The partitioned route has its own complete, separately named population."""
    root = Path(__file__).parents[1]
    population = direct_flow.action_partition_mutation_specs(repo_root=root)
    assert len(population) == 29
    assert (
        hashlib.sha256(("\n".join(sorted(population)) + "\n").encode()).hexdigest()
        == "cf858b9b0b747da0137c39fab58d7005555f2eda7aeecf471d43b4d3ad5c2ba3"
    )
    assert len(population) == direct_flow.EXPECTED_ACTION_PARTITION_MUTATION_COUNT
    for generator in (
        direct_flow.direct_flow_mutation_specs,
        direct_flow.supplemental_direct_flow_mutation_specs,
        direct_flow.uniform_process_mutation_specs,
        direct_flow.action_grid_mutation_specs,
        direct_flow.feasibility_mutation_specs,
        direct_flow.grouped_mapper_mutation_specs,
        direct_flow.grouped_guard_mutation_specs,
        direct_flow.allocation_reservation_mutation_specs,
        direct_flow.normal_process_mutation_specs,
    ):
        assert set(population).isdisjoint(generator(repo_root=root))


def test_action_partition_expected_name_digest_matches_population() -> None:
    """The self-test's pinned name digest is the digest of the live population."""
    population = direct_flow.action_partition_mutation_specs(
        repo_root=Path(__file__).parents[1]
    )
    assert (
        hashlib.sha256(("\n".join(sorted(population)) + "\n").encode()).hexdigest()
        == direct_flow.EXPECTED_ACTION_PARTITION_MUTATION_NAMES_SHA256
    )


def test_direct_flow_names_the_action_partitioned_route() -> None:
    """The certificate states the partitioned route it proves."""
    result = direct_flow.verify_direct_candidate_flow(
        repo_root=Path(__file__).parents[1]
    )
    assert result["ok"], result["errors"]
    assert "singleton_action_partitioned_solve" in result["routes"]


@pytest.mark.parametrize("mutation", tuple(direct_flow._ACTION_PARTITION_MUTATIONS))
def test_action_partition_mutation_is_rejected_after_independent_byte_reseal(
    *, mutation: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A partitioned-route defect fails its own corridor after byte resealing."""
    root = Path(__file__).parents[1]
    clean = direct_flow.verify_direct_candidate_flow(repo_root=root)
    assert clean["ok"], clean["errors"]
    sources = clean["certified_corridor_sources"]
    for relative in sources:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        copyfile(root / relative, destination)
    spec = direct_flow.action_partition_mutation_specs(repo_root=root)[mutation]
    (tmp_path / spec["path"]).write_text(spec["source"], encoding="utf-8")
    monkeypatch.setattr(
        direct_flow,
        "_SOURCE_SEALS",
        {relative: sha256_file(tmp_path / relative) for relative in sources},
    )
    result = direct_flow.verify_direct_candidate_flow(repo_root=tmp_path)
    assert not result["ok"]
    assert result["offending_paths"] == [spec["path"]], result["errors"]
    assert any("action-partitioned route" in error for error in result["errors"]), (
        result["errors"]
    )
