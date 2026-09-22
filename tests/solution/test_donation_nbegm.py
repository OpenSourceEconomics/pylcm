"""A real values-only NB-EGM dispatch can donate one owned marginal leaf."""

import json
from collections.abc import Callable, Mapping
from typing import Any, cast

import jax
import numpy as np
import pytest
from beartype.roar import BeartypeCallHintParamViolation

from _lcm.egm.carry import EGMCarry
from _lcm.execution.output_layout import PlannedCore
from _lcm.solution import backward_induction
from lcm import ExecutionConfig
from lcm.solver_api import ResultRetention
from tests.test_models import nbegm_ride_along_toy, nbegm_ride_discrete_toy


def test_values_only_nbegm_dispatch_donates_the_marginal_leaf(
    *,
    monkeypatch: pytest.MonkeyPatch,
    record_testsuite_property: Callable[[str, object], None],
) -> None:
    model = nbegm_ride_along_toy.build_model(
        variant="nbegm", n_liquid=8, n_savings=10, n_consumption=12
    )
    original = backward_induction._run_period_kernel
    observations: list[tuple[int, str, tuple[str, ...], int]] = []
    retire = backward_induction._retire_donated_inputs
    retired: list[tuple[int, tuple[str, ...], bool]] = []

    def observe_retirement(**kwargs: Any) -> Any:
        before = [
            (entry.artifact.period, entry.artifact.leaf_path, entry.array.is_deleted())
            for entry in kwargs["donated_inputs"]
        ]
        result = retire(**kwargs)
        assert all(entry.array.is_deleted() for entry in kwargs["donated_inputs"])
        retired.extend(before)
        return result

    def observe(**kwargs: Any) -> Any:
        cores = cast("Mapping[str, PlannedCore]", kwargs["compiled_cores"])
        if kwargs["regime_name"] == "alive":
            carry = kwargs["next_regime_to_continuation"]["alive"]
            assert isinstance(carry, EGMCarry)
            assert isinstance(carry.marginal_utility, jax.Array)
            assert not carry.marginal_utility.is_deleted()
            for name, core in cores.items():
                assert isinstance(core, PlannedCore)
                observations.append(
                    (
                        kwargs["period"],
                        name,
                        core.donated_arguments,
                        carry.marginal_utility.nbytes,
                    )
                )
        return original(**kwargs)

    monkeypatch.setattr(backward_induction, "_run_period_kernel", observe)
    monkeypatch.setattr(
        backward_induction, "_retire_donated_inputs", observe_retirement
    )
    result = model.solve(
        params=nbegm_ride_along_toy.build_params(),
        retention=ResultRetention.VALUES,
        log_level="off",
    )
    assert len(observations) >= 2
    assert {name for _, name, _, _ in observations} == {"main"}
    assert all(nbytes > 0 for _, _, _, nbytes in observations)
    assert np.isfinite(np.asarray(result.values[0]["alive"])).all()
    assert any(
        arguments == ("__lcm_continuation_marginal__",)
        for _, _, arguments, _ in observations
    ), observations
    assert retired
    assert all(path == ("marginal_utility",) for _, path, _ in retired)
    record_testsuite_property("marginal_dispatches", json.dumps(observations))
    record_testsuite_property("backend_consumed_before_retirement", json.dumps(retired))


@pytest.mark.parametrize("route", ["smooth", "discrete", "jump-discrete"])
def test_donation_off_preserves_values_and_the_model_fingerprint(
    *,
    route: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results = []
    seen: list[tuple[bool, tuple[str, ...]]] = []
    original = backward_induction._run_period_kernel
    enabled = True
    carries: dict[bool, dict[int, tuple[np.ndarray, ...]]] = {True: {}, False: {}}

    def observe(**kwargs: Any) -> Any:
        if kwargs["regime_name"] == "alive":
            seen.extend(
                (enabled, core.donated_arguments)
                for core in kwargs["compiled_cores"].values()
            )
        result = original(**kwargs)
        if kwargs["regime_name"] == "alive":
            carries[enabled][kwargs["period"]] = tuple(
                np.asarray(leaf).copy()
                for leaf in jax.tree.leaves(result.continuations)
            )
        return result

    monkeypatch.setattr(backward_induction, "_run_period_kernel", observe)
    for enabled in (True, False):
        if route == "smooth":
            model = nbegm_ride_along_toy.build_model(
                variant="nbegm",
                n_liquid=8,
                n_savings=10,
                n_consumption=12,
                execution_config=ExecutionConfig(donate_buffers=enabled),
            )
        else:
            model = nbegm_ride_discrete_toy.build_model(
                variant="nbegm",
                n_liquid=8,
                n_savings=10,
                n_consumption=12,
                execution_config=ExecutionConfig(donate_buffers=enabled),
                jump_schedule=route == "jump-discrete",
            )
        params = (
            nbegm_ride_along_toy.build_params()
            if route == "smooth"
            else nbegm_ride_discrete_toy.build_params(
                jump_schedule=route == "jump-discrete"
            )
        )
        results.append(
            model.solve(
                params=params,
                retention=ResultRetention.VALUES,
                log_level="off",
            )
        )
    assert any(flag and args for flag, args in seen)
    assert all(not args for flag, args in seen if not flag)
    left, right = results
    assert left.metadata.model_fingerprint == right.metadata.model_fingerprint
    assert tuple(left.values) == tuple(right.values)
    for period, regimes in left.values.items():
        assert tuple(regimes) == tuple(right.values[period])
        for regime, value in regimes.items():
            np.testing.assert_array_equal(value, right.values[period][regime])
    assert carries[True].keys() == carries[False].keys()
    for period, actual in carries[True].items():
        expected = carries[False][period]
        assert len(actual) == len(expected)
        for actual_leaf, expected_leaf in zip(actual, expected, strict=True):
            np.testing.assert_array_equal(actual_leaf, expected_leaf)


@pytest.mark.parametrize("value", [0, 1, None, "yes", np.bool_(1)])
def test_donation_switch_requires_an_exact_bool(value: object) -> None:
    with pytest.raises(
        (TypeError, BeartypeCallHintParamViolation), match=r"donate_buffers.*bool"
    ):
        ExecutionConfig(donate_buffers=value)  # ty: ignore[invalid-argument-type]
