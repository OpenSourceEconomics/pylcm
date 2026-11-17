"""Keep Cartesian preflight admission controls independent of historic populations."""

import hashlib
from collections.abc import Mapping
from pathlib import Path
from shutil import copyfile
from types import MappingProxyType

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
def action_grid_mutations() -> Mapping[str, Mapping[str, str]]:
    """Generate the complete population once, read-only; do not sample names."""
    population = direct_flow.action_grid_mutation_specs(
        repo_root=Path(__file__).parents[1]
    )
    return MappingProxyType(
        {name: MappingProxyType(dict(spec)) for name, spec in population.items()}
    )


def test_action_grid_controls_preserve_historical_mutation_populations() -> None:
    """Preflight products have their own disjoint, complete admission controls."""
    root = Path(__file__).parents[1]
    registered = direct_flow.direct_flow_mutation_specs(repo_root=root)
    supplemental = direct_flow.supplemental_direct_flow_mutation_specs(repo_root=root)
    uniform = direct_flow.uniform_process_mutation_specs(repo_root=root)
    action = direct_flow.action_grid_mutation_specs(repo_root=root)
    historical = {
        name: spec
        for name, spec in registered.items()
        # `public_shell:` controls arrived when the public grid and process
        # classes moved to `lcm`; the earlier population stays pinned without them.
        if not name.startswith(
            ("simulation_adapter:initial_physical_", "public_shell:")
        )
    }
    # The dissolution-flag removal deleted streaming_collective:published_dissolution_
    # inverted and native_graph:collective_output_role_dropped with their targets;
    # the fold-machinery removal deleted ten streaming fold controls and two
    # supplemental s02_status:fold_weight_* controls with theirs; the dead-code
    # removal deleted the three streaming_ev1 semantic-key controls with
    # GridSearchEV1ActionReduction.
    assert len(historical) == 395
    assert (
        _name_digest(historical)
        == "2f1f0e0a6ce13bc16159eb8f154cb230e9912ccae149cef245c1b531c8ec03e9"
    )
    assert len(registered) == 412
    assert (
        _name_digest(registered)
        == "39b2e6f7ea82047c1b09c454bbe353fb2cc1a89d81a588b5bd8a91f3ee02bb76"
    )
    assert len(supplemental) == 94
    assert (
        _name_digest(supplemental)
        == "f8f427f6dbbee3f52840b8e77e9fa36344a5131be9e1ed24e874523b09cf1673"
    )
    assert len(uniform) == 37
    assert (
        _name_digest(uniform)
        == "63206e56f35b0b5d4581354c33f127416231b365384a78f2adee314ca93566b5"
    )
    assert len(action) == 10
    assert (
        _name_digest(action)
        == "7e488a937dac3679ca20bb050837c968cd491e53f431ddea0ffab388a2d19d50"
    )
    assert not set(action) & (set(registered) | set(supplemental) | set(uniform))
    assert {
        spec["path"]
        for spec in (
            registered | supplemental | uniform | action | _nested_corridor_controls()
        ).values()
    } == set(direct_flow._CERTIFIED_CORRIDOR_SOURCES)


def _nested_corridor_controls() -> dict[str, dict[str, str]]:
    """Return the families that own the nested and scalar-query corridor sources."""
    root = Path(__file__).parents[1]
    return {
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
            direct_flow.current_iid_mutation_specs,
        )
        for name, spec in generator(repo_root=root).items()
    }


@pytest.mark.parametrize("mutation", tuple(direct_flow._ACTION_GRID_MUTATIONS))
def test_action_grid_mutation_is_rejected_after_independent_byte_reseal(
    *,
    mutation: str,
    clean_corridor_sources: tuple[RepositoryRelativePath, ...],
    action_grid_mutations: Mapping[str, Mapping[str, str]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Cartesian producer defect identifies its corridor after byte resealing."""
    root = Path(__file__).parents[1]
    sources = clean_corridor_sources
    for relative in sources:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        copyfile(root / relative, destination)
    spec = action_grid_mutations[mutation]
    (tmp_path / spec["path"]).write_text(spec["source"], encoding="utf-8")
    monkeypatch.setattr(
        direct_flow,
        "_SOURCE_SEALS",
        {relative: sha256_file(tmp_path / relative) for relative in sources},
    )
    result = direct_flow.verify_direct_candidate_flow(repo_root=tmp_path)
    assert not result["ok"]
    assert result["offending_paths"] == [spec["path"]], result["errors"]
    assert any("action grid admission" in error for error in result["errors"])


def _name_digest(population: dict[str, dict[str, str]]) -> str:
    """Hash the exact sorted population independently of generator constants."""
    return hashlib.sha256(("\n".join(sorted(population)) + "\n").encode()).hexdigest()
