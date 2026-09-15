"""All real forward core families profile their actual output schemas abstractly."""

import importlib
from types import MappingProxyType
from typing import Any

import jax
import jax._src.core
import jax.numpy as jnp
import pytest

from _lcm.simulation.forward_program_profiles import profile_forward_programs
from _lcm.simulation.runtime import SimulationRuntime
from tests.simulation.test_abstract_simulation_profiles import (
    _ConcreteAllocationError,
    _forbid_allocation,
)
from tests.simulation.test_budget_lifecycle import (
    _LifecycleRegimeId,
    _stateful_target_model,
)
from tests.test_models.taste_shocks_toy import ToyRegimeId, get_model, get_params


def _schema(tree: object) -> object:
    return jax.tree.map(lambda leaf: (tuple(leaf.shape), str(leaf.dtype)), tree)


def _key_descriptor(*, impl: str) -> jax.ShapeDtypeStruct:
    key = jax.random.key(0, impl=impl)
    return jax.ShapeDtypeStruct(key.shape, key.dtype, sharding=key.sharding)


def test_all_core_families_profile_actual_output_schemas_without_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _stateful_target_model()
    params = {"alive": {"koopmans_aggregator": {"discount_factor": 0.0}}}
    initial = {
        "wealth": jnp.asarray([1.0, 2.0, 3.0]),
        "age": jnp.zeros(3),
        "regime_id": jnp.full(3, _LifecycleRegimeId.alive, dtype=jnp.int32),
    }
    solution = model.solve(params=params, log_level="off")
    observed = {}
    runtimes = []
    addresses = {
        id(program): (period, name, family)
        for name, regime in model._regimes.items()
        for family in ("decision", "transition", "route")
        for period, program in getattr(regime.simulation.programs, family).items()
    }
    original = SimulationRuntime.dispatch

    def observe(self: SimulationRuntime, **call: Any) -> object:
        result = original(self, **call)
        if not runtimes:
            runtimes.append(self)
        observed[addresses[id(call["program"])]] = _schema(result)
        return result

    with monkeypatch.context() as capture:
        capture.setattr(SimulationRuntime, "dispatch", observe)
        result = model.simulate(
            params=params,
            initial_conditions=initial,
            solution=solution,
            seed=17,
            log_level="off",
        )
    assert len(observed) == 5  # Two decisions at0, one at1; state and regime laws.
    spaces = MappingProxyType(
        {
            name: regime.solution.state_action_space(
                regime_params=result.flat_params[name]
            )
            for name, regime in result._regimes.items()
        }
    )
    profiles_module = importlib.import_module(
        "_lcm.simulation.forward_program_profiles"
    )
    ordinary_key = _key_descriptor(impl=jax.config.jax_default_prng_impl)
    with monkeypatch.context() as guard:
        guard.setattr(jax, "device_put", _forbid_allocation)
        guard.setattr(jax._src.core.EvalTrace, "process_primitive", _forbid_allocation)
        with pytest.raises(_ConcreteAllocationError):
            jnp.zeros(3)
        profiles = profiles_module.profile_forward_programs(
            runtime=runtimes[0],
            regimes=result._regimes,
            flat_params=result.flat_params,
            base_spaces=spaces,
            values=result.period_to_regime_to_V_arr,
            ages=model.ages,
            n_subjects=3,
            widths={"subject": 3},
            ordinary_key=ordinary_key,
            taste_key=None,
        )
    assert set(profiles) == set(observed)
    for key, profile in profiles.items():
        assert _schema(profile.executable.out_info) == observed[key]
        assert all(
            isinstance(leaf, jax.ShapeDtypeStruct)
            for leaf in jax.tree.leaves(profile.arguments)
        )
        if key[2] == "decision":
            assert profile.action_decoder is not None
            assert _schema(profile.action_decoder.executable.out_info) == _schema(
                result.raw_results[key[1]][key[0]].actions
            )


def test_independent_taste_profile_keeps_its_actual_key_dtype_under_rbg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    captured = []
    original = SimulationRuntime.dispatch

    def observe(self: SimulationRuntime, **call: Any) -> object:
        result = original(self, **call)
        if "taste_shock_key" in call["arguments"]:
            captured.append((self, call["arguments"]["taste_shock_key"].dtype))
        return result

    with jax.default_prng_impl("rbg"):
        model = get_model()
        with monkeypatch.context() as observer:
            observer.setattr(SimulationRuntime, "dispatch", observe)
            result = model.simulate(
                params=get_params(scale=0.2),
                initial_conditions={
                    "wealth": jnp.asarray([2.0, 3.0]),
                    "age": jnp.full(2, 40.0),
                    "regime_id": jnp.full(2, ToyRegimeId.alive, dtype=jnp.int32),
                },
                seed=17,
                taste_shock_seed=23,
                log_level="off",
            )
        assert captured
        assert str(captured[0][1]) == "key<fry>"
        spaces = MappingProxyType(
            {
                name: regime.solution.state_action_space(
                    regime_params=result.flat_params[name]
                )
                for name, regime in model._regimes.items()
            }
        )
        profiles = profile_forward_programs(
            runtime=captured[0][0],
            regimes=model._regimes,
            flat_params=result.flat_params,
            base_spaces=spaces,
            values=result.period_to_regime_to_V_arr,
            ages=model.ages,
            n_subjects=2,
            widths={"subject": 2},
            ordinary_key=_key_descriptor(impl="rbg"),
            taste_key=_key_descriptor(impl="threefry2x32"),
        )
    key_argument = profiles[(0, "alive", "decision")].arguments["taste_shock_key"]
    assert isinstance(key_argument, jax.ShapeDtypeStruct)
    assert key_argument.dtype == captured[0][1]
