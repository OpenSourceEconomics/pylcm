"""Actual physical ownership chooses an admitted ordinary NB-EGM executable."""

from typing import Any

import jax
import numpy as np
import pytest

from _lcm.execution.scheduler import buffer_identity
from _lcm.solution import backward_induction
from _lcm.solution.solve_inputs import locate_artifact
from lcm.solver_api import ResultRetention
from tests.test_models import nbegm_ride_along_toy


@pytest.mark.parametrize("conflict", ["shared", "unproduced"])
def test_runtime_ownership_falls_back_without_compiling_or_consuming_the_source(
    *, conflict: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = nbegm_ride_along_toy.build_model(
        variant="nbegm", n_liquid=8, n_savings=10, n_consumption=12
    )
    select = backward_induction._select_runtime_donation_cores
    held: list[tuple[Any, np.ndarray]] = []
    observed: list[tuple[int, tuple[str, ...]]] = []

    def forbid_compilation(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("Runtime donation fallback compiled a new executable.")

    def with_physical_conflict(**kwargs: Any) -> Any:
        unit = kwargs["unit"]
        programs = kwargs["compiled_programs"]
        registry = kwargs["registry"]
        touched = []
        for name in unit.programs:
            triple = (unit.regime, unit.period, name)
            for decision in programs.donations.get(triple, ()):
                if not decision.donated:
                    continue
                for artifact in decision.artifacts:
                    array = locate_artifact(inputs=kwargs["inputs"], artifact=artifact)
                    assert isinstance(array, jax.Array)
                    assert array.nbytes > 0
                    assert not array.is_deleted()
                    held.append((array, np.asarray(array).copy()))
                    if conflict == "shared":
                        registry.register(array=array, artifact=("external", triple))
                        assert ("external", triple) in registry.artifacts_sharing(
                            array=array
                        )
                        assert buffer_identity(array=array)
                    else:
                        registry.declare_not_produced(tree=(array,))
                        assert registry.is_not_produced(array=array)
                touched.append(triple)
        monkeypatch.setattr(jax.stages.Lowered, "compile", forbid_compilation)
        cores, donations = select(**kwargs)
        for triple in touched:
            core = cores[triple[2]]
            assert core is programs.donation_fallbacks[triple]
            assert core.compiled is programs.donation_fallbacks[triple].compiled
            assert (
                core.tile_widths
                == programs.executables[triple[:2]][triple[2]].tile_widths
            )
            assert donations[triple] == ()
            observed.append((unit.period, core.donated_arguments))
        return cores, donations

    monkeypatch.setattr(
        backward_induction, "_select_runtime_donation_cores", with_physical_conflict
    )
    result = model.solve(
        params=nbegm_ride_along_toy.build_params(),
        retention=ResultRetention.VALUES,
        log_level="off",
    )
    assert observed
    assert all(not arguments for _, arguments in observed)
    assert np.isfinite(np.asarray(result.values[0]["alive"])).all()
    assert held
    for original, expected in held:
        assert not original.is_deleted()
        np.testing.assert_array_equal(original, expected)
