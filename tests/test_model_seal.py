"""A model's identity is sealed when it is built.

Every user callable, and every global or closure binding it reads, is captured
at `Model(...)`. The structure fingerprint is computed there, once, and a solve
or simulation refuses to run when one of those bindings has since been rebound
— the cached identity would otherwise describe code the model no longer runs.
"""

import jax.numpy as jnp
import pytest

from lcm import AgeGrid, LinSpacedGrid, Model, categorical
from lcm.exceptions import ModelSealError
from lcm.regime import Regime
from lcm.typing import FloatND, ScalarInt

_UTILITY_SCALE = jnp.float64(1.0)


@categorical(ordered=False)
class _RegimeId:
    working: ScalarInt
    dead: ScalarInt


def _utility(consumption: FloatND) -> FloatND:
    return _UTILITY_SCALE * jnp.log(consumption)


def _next_wealth(*, wealth: FloatND, consumption: FloatND) -> FloatND:
    return wealth - consumption + 1.0


def _feasible(*, wealth: FloatND, consumption: FloatND) -> FloatND:
    return consumption <= wealth


def _next_regime(age: float) -> ScalarInt:
    return jnp.where(age >= 18, _RegimeId.dead, _RegimeId.working)


def _build_model(*, enable_jit: bool) -> Model:
    working = Regime(
        transition=_next_regime,
        states={"wealth": LinSpacedGrid(start=1, stop=3, n_points=3)},
        state_transitions={"wealth": _next_wealth},
        actions={"consumption": LinSpacedGrid(start=0.5, stop=2.5, n_points=3)},
        functions={"utility": _utility},
        constraints={"feasible": _feasible},
        active=lambda age: age < 19,
    )
    dead = Regime(transition=None, functions={"utility": lambda: 0.0})
    return Model(
        regimes={"working": working, "dead": dead},
        ages=AgeGrid(start=18, stop=20, step="Y"),
        regime_id_class=_RegimeId,
        enable_jit=enable_jit,
    )


_PARAMS = {"discount_factor": 0.95}


@pytest.fixture
def restore_scale() -> object:
    yield None
    globals()["_UTILITY_SCALE"] = jnp.float64(1.0)


@pytest.mark.parametrize("enable_jit", [False, True])
def test_rebinding_a_referenced_global_after_build_is_refused(
    *, enable_jit: bool, restore_scale: object
) -> None:
    del restore_scale
    model = _build_model(enable_jit=enable_jit)
    model.solve(params=_PARAMS, log_level="off")

    globals()["_UTILITY_SCALE"] = jnp.float64(7.0)

    with pytest.raises(ModelSealError, match="_UTILITY_SCALE"):
        model.solve(params=_PARAMS, log_level="off")


def test_simulate_checks_the_seal_too(restore_scale: object) -> None:
    del restore_scale
    model = _build_model(enable_jit=False)
    solution = model.solve(params=_PARAMS, log_level="off")
    globals()["_UTILITY_SCALE"] = jnp.float64(7.0)

    with pytest.raises(ModelSealError, match="_UTILITY_SCALE"):
        model.simulate(
            params=_PARAMS,
            initial_conditions={
                "wealth": jnp.asarray([2.0]),
                "regime_id": jnp.asarray([_RegimeId.working], dtype=jnp.int32),
            },
            solution=solution,
            log_level="off",
        )


def test_rebinding_back_to_the_captured_object_lifts_the_refusal(
    restore_scale: object,
) -> None:
    del restore_scale
    original = _UTILITY_SCALE
    model = _build_model(enable_jit=False)
    globals()["_UTILITY_SCALE"] = jnp.float64(7.0)
    globals()["_UTILITY_SCALE"] = original

    solution = model.solve(params=_PARAMS, log_level="off")

    assert solution.metadata.model_fingerprint


def test_structure_fingerprint_is_fixed_at_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Solving never walks the user callables again."""
    import lcm.model as model_module  # noqa: PLC0415

    model = _build_model(enable_jit=False)

    def _must_not_walk(**_kwargs: object) -> str:
        raise AssertionError("structure fingerprint recomputed after build")

    monkeypatch.setattr(model_module, "fingerprint_model_structure", _must_not_walk)
    first = model.solve(params=_PARAMS, log_level="off")
    second = model.solve(params={"discount_factor": 0.9}, log_level="off")

    assert first.metadata.model_fingerprint != second.metadata.model_fingerprint
