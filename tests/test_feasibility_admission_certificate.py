"""Independent byte-resealed controls for feasibility counts and admission."""

import ast
import hashlib
from pathlib import Path
from shutil import copyfile

import pytest

from tests.candidate_certificate import direct_flow
from tests.candidate_certificate.generate_sources import sha256_file

_EXACT_COUNT_CONTROL_ERRORS = {
    "feasibility:exact_count_dtype_narrowed": (
        "native feasibility summary: exact callable corridor "
        "'get_feasibility_count_dtype' changed"
    ),
    "feasibility:count_capacity_refusal_omitted": (
        "native feasibility summary: exact callable corridor "
        "'_fail_if_feasibility_count_exceeds_capacity' changed"
    ),
}


def test_feasibility_wrapper_preserves_the_admitted_producer_handoff() -> None:
    """Moved admission controls still protect the actual feasibility caller."""
    root = Path(__file__).parents[1]
    tree = ast.parse((root / direct_flow.INITIAL_CONDITIONS_SOURCE).read_text())
    assert any(
        isinstance(node, ast.ImportFrom)
        and node.module == "_lcm.simulation.memory"
        and any(
            alias.name == "run_simulation_producer" and alias.asname is None
            for alias in node.names
        )
        for node in tree.body
    )
    wrapper = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_run_profiled_feasibility"
    )
    assert ast.get_docstring(wrapper) is not None
    expected = ast.parse(
        "return run_simulation_producer("
        "memory=memory, function=function, arguments=arguments)"
    ).body
    assert [ast.dump(node) for node in wrapper.body[1:]] == [
        ast.dump(node) for node in expected
    ]
    assert (
        "run_simulation_producer"
        in direct_flow._SIMULATION_ADAPTER_CONTRACTS[
            direct_flow.SIMULATION_MEMORY_SOURCE
        ][1]
    )


def test_feasibility_controls_are_complete_and_disjoint() -> None:
    root = Path(__file__).parents[1]
    population = direct_flow.feasibility_mutation_specs(repo_root=root)
    assert len(population) == 13
    assert _EXACT_COUNT_CONTROL_ERRORS.keys() <= population.keys()
    assert len(population.keys() - _EXACT_COUNT_CONTROL_ERRORS.keys()) == 11
    assert (
        hashlib.sha256(("\n".join(sorted(population)) + "\n").encode()).hexdigest()
        == direct_flow.EXPECTED_FEASIBILITY_MUTATION_NAMES_SHA256
    )
    for generator in (
        direct_flow.direct_flow_mutation_specs,
        direct_flow.supplemental_direct_flow_mutation_specs,
        direct_flow.uniform_process_mutation_specs,
        direct_flow.action_grid_mutation_specs,
    ):
        assert set(population).isdisjoint(generator(repo_root=root))


@pytest.mark.parametrize("mutation", tuple(direct_flow._FEASIBILITY_MUTATIONS))
def test_feasibility_mutation_is_rejected_after_independent_byte_reseal(
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
    spec = direct_flow.feasibility_mutation_specs(repo_root=root)[mutation]
    (tmp_path / spec["path"]).write_text(spec["source"], encoding="utf-8")
    monkeypatch.setattr(
        direct_flow,
        "_SOURCE_SEALS",
        {relative: sha256_file(tmp_path / relative) for relative in sources},
    )
    result = direct_flow.verify_direct_candidate_flow(repo_root=tmp_path)
    assert not result["ok"]
    assert result["offending_paths"] == [spec["path"]], result["errors"]
    expected_error = _EXACT_COUNT_CONTROL_ERRORS.get(
        mutation, "simulation owned-input corridor"
    )
    assert any(expected_error in error for error in result["errors"])
