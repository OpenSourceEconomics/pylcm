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
    assert len(historical) == 409
    assert (
        _name_digest(historical)
        == "3c5fdaf1c4a0eb22ad92ac592d4ca84cef0c97a4b94bc525a7f8592d886880a7"
    )
    assert len(registered) == 421
    assert _name_digest(registered) == (
        "f58754206b785f05a7dff36c690946024effc931c888c72383fc0bd98be060cd"
    )
    assert len(supplemental) == 96
    assert _name_digest(supplemental) == (
        "bfb20467575ab892db7be74b859bac0288c8840f7a1aeb36eb13dee129ed3490"
    )
    assert len(uniform) == 37
    assert _name_digest(uniform) == (
        "63206e56f35b0b5d4581354c33f127416231b365384a78f2adee314ca93566b5"
    )
    assert not set(uniform) & (set(registered) | set(supplemental))
    assert set(direct_flow._UNIFORM_PROCESS_SOURCES) <= {
        spec["path"] for spec in uniform.values()
    }
    nested = {
        name: spec
        for generator in (
            direct_flow.s03_scalar_mutation_specs,
            direct_flow.reviewed_prefix_mutation_specs,
            direct_flow.decoupled_bound_mutation_specs,
            direct_flow.associated_saving_mutation_specs,
            direct_flow.retained_prefix_mutation_specs,
            direct_flow.outer_helper_mutation_specs,
            direct_flow.final_nested_mutation_specs,
            direct_flow.child_native_mutation_specs,
            direct_flow.additional_helper_mutation_specs,
        )
        for name, spec in generator(repo_root=root).items()
    }
    assert {
        spec["path"] for spec in (registered | supplemental | uniform | nested).values()
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
