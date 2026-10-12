"""Keep Uniform admission controls independent of candidate mutation populations."""

import hashlib
from pathlib import Path
from shutil import copyfile

import pytest

from tests.candidate_certificate import direct_flow
from tests.candidate_certificate.generate_sources import sha256_file

type RepositoryRelativePath = str


@pytest.fixture(scope="module")
def clean_corridor_sources() -> tuple[RepositoryRelativePath, ...]:
    """Check the frozen checkout once, independently of all resealed mutants."""
    clean = direct_flow.verify_direct_candidate_flow(
        repo_root=Path(__file__).parents[1]
    )
    assert clean["ok"], clean["errors"]
    return tuple(clean["certified_corridor_sources"])


@pytest.fixture(scope="module")
def uniform_mutations() -> dict[str, dict[str, str]]:
    """Generate the complete population once; do not sample mutation names."""
    return direct_flow.uniform_process_mutation_specs(
        repo_root=Path(__file__).parents[1]
    )


def test_uniform_controls_preserve_both_historical_mutation_populations() -> None:
    """Uniform admission has its own disjoint, complete source-control family."""
    root = Path(__file__).parents[1]
    registered = direct_flow.direct_flow_mutation_specs(repo_root=root)
    supplemental = direct_flow.supplemental_direct_flow_mutation_specs(repo_root=root)
    uniform = direct_flow.uniform_process_mutation_specs(repo_root=root)

    historical = {
        name: spec
        for name, spec in registered.items()
        if not name.startswith("simulation_adapter:initial_physical_")
    }
    assert len(historical) == 408
    assert (
        _name_digest(historical)
        == "d27d7bdcb349eea62f44ad1a559abc676746e025ded98e52b42770564c6807b2"
    )
    assert len(registered) == 420
    assert _name_digest(registered) == (
        "2ce1ac92a6e3cefa1c4c6b70ff0eab3139be5a3aa3ab7a2efa6d143b09be8808"
    )
    assert len(supplemental) == 54
    assert _name_digest(supplemental) == (
        "1d95a163810d9ffecb7bc2f2324065c0a3d6168d9dd01002cd00ff30c5802da5"
    )
    assert len(uniform) == 37
    assert _name_digest(uniform) == (
        "63206e56f35b0b5d4581354c33f127416231b365384a78f2adee314ca93566b5"
    )
    assert not set(uniform) & (set(registered) | set(supplemental))
    assert set(direct_flow._UNIFORM_PROCESS_SOURCES) <= {
        spec["path"] for spec in uniform.values()
    }
    assert {
        spec["path"] for spec in (registered | supplemental | uniform).values()
    } == set(direct_flow._CERTIFIED_CORRIDOR_SOURCES) - set(
        direct_flow._ACTION_GRID_SOURCES
    )


@pytest.mark.parametrize("mutation", tuple(direct_flow._UNIFORM_PROCESS_MUTATIONS))
def test_uniform_admission_mutation_is_rejected_after_independent_byte_reseal(
    *,
    mutation: str,
    clean_corridor_sources: tuple[RepositoryRelativePath, ...],
    uniform_mutations: dict[str, dict[str, str]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A producer defect names its own semantic corridor after byte resealing."""
    root = Path(__file__).parents[1]
    sources = clean_corridor_sources
    for relative in sources:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        copyfile(root / relative, destination)
    spec = uniform_mutations[mutation]
    (tmp_path / spec["path"]).write_text(spec["source"], encoding="utf-8")
    monkeypatch.setattr(
        direct_flow,
        "_SOURCE_SEALS",
        {relative: sha256_file(tmp_path / relative) for relative in sources},
    )

    result = direct_flow.verify_direct_candidate_flow(repo_root=tmp_path)

    assert not result["ok"]
    assert result["offending_paths"] == [spec["path"]], result["errors"]
    assert any("uniform process" in error for error in result["errors"])


def _name_digest(population: dict[str, dict[str, str]]) -> str:
    """Hash the exact sorted name population, independently of generator constants."""
    return hashlib.sha256(("\n".join(sorted(population)) + "\n").encode()).hexdigest()
