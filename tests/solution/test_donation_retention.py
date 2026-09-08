"""Retention, eager execution and the complete live read inventory bound donation."""

from typing import Any

import jax
import numpy as np
import pytest

from _lcm.egm.carry import EGMCarry
from _lcm.execution.core_program import CoreBuildContext
from _lcm.solution import backward_induction
from _lcm.solution.continuation_arguments import (
    MARGINAL_ARGUMENT,
    MarginalLeafArguments,
)
from lcm import Model
from lcm.solver_api import EGM_CONTINUATION, ResultRetention
from tests.test_models import nbegm_ride_along_toy
from tests.test_models.nbegm_common import RegimeId


@pytest.mark.parametrize("mode", ["default", "retained", "eager"])
def test_retention_and_eager_execution_protect_all_output_owners(
    *, mode: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = nbegm_ride_along_toy.build_model(
        variant="nbegm", n_liquid=8, n_savings=10, n_consumption=12
    )
    if mode == "eager":
        model = Model(
            regimes=model.user_regimes,
            ages=model.ages,
            regime_id_class=RegimeId,
            fixed_params=dict(model.fixed_params),
            enable_jit=False,
        )
    retention = (
        ResultRetention.ALL_PERSISTABLE_ARTIFACTS
        if mode == "retained"
        else ResultRetention.VALUES_AND_REPLAY
    )
    run = backward_induction._run_period_kernel
    nominations: list[tuple[str, ...]] = []
    outputs: list[tuple[Any, np.ndarray]] = []
    declarations: list[int] = []

    def observe(**kwargs: Any) -> Any:
        if kwargs["regime_name"] == "alive":
            nominations.extend(
                core.donated_arguments for core in kwargs["compiled_cores"].values()
            )
            declarations.extend(_assert_complete_live_reads(kwargs=kwargs))
        result = run(**kwargs)
        if mode != "default":
            outputs.extend(
                (leaf, np.asarray(leaf).copy())
                for leaf in jax.tree.leaves(result.continuations)
            )
        return result

    monkeypatch.setattr(backward_induction, "_run_period_kernel", observe)
    result = model.solve(
        params=nbegm_ride_along_toy.build_params(), retention=retention, log_level="off"
    )
    assert declarations
    assert min(declarations) >= 4
    assert bool(any(nominations)) is (mode == "default")
    assert np.isfinite(np.asarray(result.values[0]["alive"])).all()
    if mode != "default":
        assert outputs
        for original, expected in outputs:
            assert not original.is_deleted()
            np.testing.assert_array_equal(original, expected)
    if mode == "retained":
        assert any(ref.key == EGM_CONTINUATION for ref in result.retained_continuations)


def _assert_complete_live_reads(*, kwargs: dict[str, Any]) -> list[int]:
    counts = []
    programs = (
        kwargs["regime"].solution.period_kernels[kwargs["period"]].core_programs()
    )
    for program in programs.values():
        if not isinstance(program.argument_builder, MarginalLeafArguments):
            continue
        carry = kwargs["next_regime_to_continuation"]["alive"]
        assert isinstance(carry, EGMCarry)
        arguments = program.argument_builder(
            CoreBuildContext(
                state_action_space=kwargs["state_action_space"],
                next_regime_to_V_arr=kwargs["next_regime_to_V_arr"],
                next_regime_to_continuation=kwargs["next_regime_to_continuation"],
                flat_params=kwargs["flat_params"],
                period=kwargs["period"],
                ages=kwargs["ages"],
            )
        )
        declared = program.requirements.value_reads
        assert {read.target.leaf_path for read in declared} == set(carry.leaves())
        assert len(declared) == len(carry.leaves())
        for read in declared:
            actual: Any = arguments[read.source.argument or read.source.channel.value]
            for segment in read.source.path:
                actual = actual[segment]
            assert actual is carry.leaves()[read.target.leaf_path]
            assert read.source.core_key == program.name
        assert arguments[MARGINAL_ARGUMENT] is carry.marginal_utility
        assert (
            sum(leaf is carry.marginal_utility for leaf in jax.tree.leaves(arguments))
            == 1
        )
        residual: Any = arguments["next_regime_to_continuation"]
        assert "marginal_utility" not in residual["alive"]
        counts.append(len(declared))
    return counts
