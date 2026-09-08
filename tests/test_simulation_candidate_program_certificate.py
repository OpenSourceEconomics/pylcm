"""The simulation certificate follows the published program into dispatch.

Refreshing byte seals cannot admit a changed candidate transport: each seeded
defect must also fail the independently pinned program corridor.
"""

from pathlib import Path
from shutil import copyfile

import pytest

from tests.candidate_certificate import direct_flow
from tests.candidate_certificate.direct_flow import (
    direct_flow_mutation_specs,
    verify_direct_candidate_flow,
)
from tests.candidate_certificate.generate_sources import sha256_file

_PROGRAM_MUTATIONS = (
    "dense_reducer_replaced",
    "q_and_f_replaced",
    "action_coordinates_reversed",
    "action_order_changed",
    "hard_max_reduction_replaced",
    "streamed_width_ignored",
    "streamed_q_and_f_filtered",
    "streamed_index_shifted",
    "streamed_value_filtered",
    "subject_tiles_reversed",
    "argument_mapping_filtered",
    "decision_mapping_dropped",
    "streamed_guard_bypassed",
    "program_snapshot_filtered",
    "resolved_body_bypassed",
    "lowered_body_replaced",
    "dispatch_family_replaced",
    "dispatch_arguments_filtered",
    "body_cache_identity_dropped",
    "compiler_options_dropped",
    "duplicate_future_replaced",
    "resolved_widths_ignored",
    "prewarm_program_replaced",
    "prewarm_failure_hidden",
)


@pytest.fixture(scope="module")
def program_mutations() -> dict[str, dict[str, str]]:
    """Build the same seeded defects the full certificate control executes."""
    return direct_flow_mutation_specs(repo_root=Path(__file__).parents[1])


@pytest.mark.parametrize(
    "source",
    [
        "src/_lcm/simulation/programs.py",
        "src/_lcm/simulation/program_types.py",
        "src/_lcm/simulation/runtime.py",
    ],
)
def test_live_simulation_program_sources_are_certified(source: str):
    """Declaration, argument binding, resolution, and dispatch are live obligations."""
    result = verify_direct_candidate_flow(repo_root=Path(__file__).parents[1])

    assert source in result["certified_corridor_sources"]


@pytest.mark.parametrize(
    "mutation",
    [f"simulation_program:{name}" for name in _PROGRAM_MUTATIONS]
    + [
        "caller_simulate:action_names_slice",
        "caller_simulate:wrong_discrete_axis_count",
        "caller_simulate:taste_flag_disabled",
        "caller_simulate:live_taste_flag_rebinding",
        "caller_simulate:published_empty_mapping",
        "caller_simulate:attribute_simulation_phase",
        "aot_compile:argmax_index_shift",
        "aot_model:compiled_regime_filter",
        "simulation_index_consumer:next_candidate",
    ]
    + list(direct_flow._SIMULATION_ADAPTER_MUTATIONS),
)
def test_program_mutation_is_rejected_after_byte_seals_are_refreshed(
    *,
    mutation: str,
    program_mutations: dict[str, dict[str, str]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Each transport defect fails its own semantic corridor after byte resealing."""
    root = Path(__file__).parents[1]
    sources = verify_direct_candidate_flow(repo_root=root)["certified_corridor_sources"]
    for relative in sources:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        copyfile(root / relative, destination)
    spec = program_mutations[mutation]
    (tmp_path / spec["path"]).write_text(spec["source"], encoding="utf-8")
    monkeypatch.setattr(
        direct_flow,
        "_SOURCE_SEALS",
        {relative: sha256_file(tmp_path / relative) for relative in sources},
    )

    result = verify_direct_candidate_flow(repo_root=tmp_path)

    assert result["offending_paths"] == [spec["path"]], result["errors"]
