"""A donating solve admits the ordinary executable's real complete total too."""

import json
from collections.abc import Callable
from typing import Any

import jax
import numpy as np
import pytest

from _lcm.execution.workspace_planning import compiler_peak_bytes
from _lcm.solution import backward_induction
from lcm import ExecutionConfig
from lcm.exceptions import ExecutionPlanningError
from lcm.solver_api import ResultRetention
from tests.test_models import nbegm_ride_along_toy


def test_budget_admits_both_real_variants_before_any_donating_dispatch(
    *,
    monkeypatch: pytest.MonkeyPatch,
    record_testsuite_property: Callable[[str, object], None],
) -> None:
    records: dict[
        tuple[int, tuple[tuple[str, int], ...]], list[tuple[Any, int, int]]
    ] = {}
    resident = backward_induction._candidate_resident_bytes
    compile_all = backward_induction._compile_all_functions
    completed = []

    def observe_residency(**kwargs: Any) -> int:
        actual = resident(**kwargs)
        executable = kwargs["compiled"]
        widths = kwargs["program"].tile_widths
        assert isinstance(executable, jax.stages.Compiled)
        peak = compiler_peak_bytes(compiled=executable, widths=widths)
        records.setdefault((id(kwargs["inventory"]), tuple(widths.items())), []).append(
            (executable, peak, actual)
        )
        return actual

    def observe_compilation(**kwargs: Any) -> Any:
        result = compile_all(**kwargs)
        completed.append(result)
        return result

    monkeypatch.setattr(
        backward_induction, "_candidate_resident_bytes", observe_residency
    )
    monkeypatch.setattr(
        backward_induction, "_compile_all_functions", observe_compilation
    )
    model = nbegm_ride_along_toy.build_model(
        variant="nbegm",
        n_liquid=8,
        n_savings=10,
        n_consumption=12,
        execution_config=ExecutionConfig(
            device_memory_bytes=10**8, axis_widths={"cell": 2}
        ),
    )
    result = model.solve(
        params=nbegm_ride_along_toy.build_params(),
        retention=ResultRetention.VALUES,
        log_level="off",
    )
    assert np.isfinite(np.asarray(result.values[0]["alive"])).all()
    (programs,) = completed
    assert programs.donation_fallbacks
    donors = {
        id(programs.executables[triple[:2]][triple[2]].compiled)
        for triple in programs.donation_fallbacks
    }
    pairs = [variants for variants in records.values() if len(variants) == 2]
    assert pairs
    assert all(
        sum(id(executable) in donors for executable, _, _ in pair) == 1
        for pair in pairs
    )
    donor_ceiling = max(
        peak + external
        for variants in records.values()
        for executable, peak, external in variants
        if len(variants) == 1 or id(executable) in donors
    )
    paired_ceiling = max(
        peak + external
        for variants in records.values()
        for _, peak, external in variants
    )
    assert paired_ceiling > donor_ceiling, (
        donor_ceiling,
        paired_ceiling,
        [
            [
                (id(executable) in donors, peak, external)
                for executable, peak, external in pair
            ]
            for pair in pairs
        ],
    )
    record_testsuite_property("donor_ceiling_bytes", donor_ceiling)
    record_testsuite_property("paired_ceiling_bytes", paired_ceiling)
    record_testsuite_property(
        "variant_profiles",
        json.dumps(
            [
                [
                    {
                        "donating": id(executable) in donors,
                        "raw_peak": peak,
                        "resident": external,
                    }
                    for executable, peak, external in pair
                ]
                for pair in pairs
            ]
        ),
    )
    for triple, fallback in programs.donation_fallbacks.items():
        donating = programs.executables[triple[:2]][triple[2]]
        assert fallback.tile_widths == donating.tile_widths
        assert fallback.compiled is not donating.compiled
        assert fallback.donated_arguments == ()
    records.clear()

    def forbid_dispatch(**_kwargs: object) -> object:
        raise AssertionError("An inadmissible ordinary alternative reached execution.")

    monkeypatch.setattr(backward_induction, "_run_period_kernel", forbid_dispatch)
    tight = nbegm_ride_along_toy.build_model(
        variant="nbegm",
        n_liquid=8,
        n_savings=10,
        n_consumption=12,
        execution_config=ExecutionConfig(
            device_memory_bytes=donor_ceiling, axis_widths={"cell": 2}
        ),
    )
    with pytest.raises(
        ExecutionPlanningError, match=r"workspace-width candidate|workspace|budget"
    ):
        tight.solve(
            params=nbegm_ride_along_toy.build_params(),
            retention=ResultRetention.VALUES,
            log_level="off",
        )
    assert records, (
        "The refusal must inspect real candidates, not only fixed input bytes."
    )
