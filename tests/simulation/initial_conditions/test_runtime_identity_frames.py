"""Seedless partial frames need this invocation's exact identity correlation."""

from collections.abc import Mapping
from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.shock_overrides import supplied_group_values
from _lcm.simulation import initial_conditions as initial_checks
from lcm import AgeGrid, InitialNodes, LinSpacedGrid, Model, Regime, categorical
from lcm.exceptions import (
    InvalidSimulationInputError,
    ShockDeclarationError,
    UnsupportedOperationError,
)
from lcm.shocks import BaseInnovations, MonteCarlo, Normal, PhysicalValues, ShockGroup
from lcm.typing import BoolND, FloatND, IntND, ScalarInt


@pytest.mark.coverage(backends=("cpu",), precisions="both")
@pytest.mark.parametrize("base", [False, True])
def test_one_compiled_partial_producer_rechecks_identity_and_law(*, base: bool) -> None:
    """The same executable retains only supplied outputs and fresh law status."""
    group = ShockGroup(
        shocks={"eps": Normal(mu=None, sigma=None), "other": Normal(mu=0, sigma=1)},
        correlation=None,
        nodes=MonteCarlo(2, seed=7),
    )

    # JAX lower/compiled calls below bind the positional operand tree.
    # keyword-only-exempt: library-callback=jax.jit
    def produce(
        corr: FloatND, location: FloatND, scale: FloatND, support: BoolND
    ) -> tuple[Mapping[str, FloatND | IntND], BoolND]:
        supplied = (
            BaseInnovations(values=jnp.asarray(3.0))
            if base
            else PhysicalValues(values=jnp.asarray(3.0))
        )
        return supplied_group_values(
            group=group,
            supplied={"eps": supplied},
            params={"corr": corr, "eps__mu": location, "eps__sigma": scale},
            inputs={},
            admitted_correlation_support=support,
        )

    identity = jnp.eye(2)
    support = jnp.asarray([False, False])
    executable = (
        jax.jit(produce)
        .lower(identity, jnp.asarray(1.0), jnp.asarray(2.0), support)
        .compile()
    )
    for corr, location, scale, expected_valid in (
        (identity, 1.0, 2.0, True),
        (jnp.asarray([[1.0, 0.25], [0.25, 1.0]]), 1.0, 2.0, False),
        (jnp.asarray([[1.0, jnp.nan], [jnp.nan, 1.0]]), 1.0, 2.0, False),
        (identity, 1.0, -1.0, False),
        (identity, jnp.nan, 2.0, False),
        (identity, 3.0, 0.0, True),
        (identity, 1.0, 0.0, base),
        (identity, 2.0, 4.0, True),
    ):
        physical, valid = executable(
            corr, jnp.asarray(location), jnp.asarray(scale), support
        )
        assert set(physical) == {"eps"}
        assert bool(valid) == expected_valid
        if expected_valid:
            np.testing.assert_array_equal(
                physical["eps"], location + 3 * scale if base else 3.0
            )
    # Exact identity is not permission to discard admitted factor/support status.
    _, bad_support = executable(
        identity, jnp.asarray(1.0), jnp.asarray(2.0), jnp.asarray([True, False])
    )
    assert not bool(bad_support)
    with pytest.raises(ShockDeclarationError, match="row and column"):
        jax.jit(produce)(jnp.ones((1, 1)), jnp.asarray(1.0), jnp.asarray(2.0), support)


@pytest.mark.coverage(backends=("cpu",), precisions="both")
def test_complete_base_frame_still_uses_the_correlated_transform() -> None:
    """Runtime identity support must not reinterpret a complete nonidentity frame."""
    group = ShockGroup(
        shocks={"eps": Normal(mu=2, sigma=3), "other": Normal(mu=0, sigma=1)},
        correlation=None,
        nodes=MonteCarlo(2, seed=7),
    )
    physical, valid = supplied_group_values(
        group=group,
        supplied={
            "eps": BaseInnovations(values=jnp.asarray(3.0)),
            "other": BaseInnovations(values=jnp.asarray(4.0)),
        },
        params={"corr": jnp.asarray([[1.0, 0.25], [0.25, 1.0]])},
        inputs={},
        admitted_correlation_support=jnp.asarray([False, False]),
    )
    assert bool(valid)
    assert set(physical) == {"eps", "other"}
    np.testing.assert_array_equal(physical["eps"], 11.0)
    np.testing.assert_allclose(
        physical["other"], 0.75 + 4.0 * np.sqrt(1.0 - 0.25**2), rtol=1e-6
    )


@categorical(ordered=False)
class _Entry:
    alive: ScalarInt


def _location(*, age: FloatND, period: ScalarInt, shift: FloatND) -> FloatND:
    return age + period + shift


def _utility(*, consumption: FloatND) -> FloatND:
    return consumption


def _affordable(*, wealth: FloatND, consumption: FloatND, eps: FloatND) -> BoolND:
    return consumption <= wealth + eps


def _affordable_with_other(
    *, wealth: FloatND, consumption: FloatND, eps: FloatND, other: FloatND
) -> BoolND:
    return consumption <= wealth + eps + other


def _model(*, needs_other: bool = False) -> Model:
    return Model(
        regimes={
            "alive": Regime(
                states={"wealth": LinSpacedGrid(start=1, stop=2, n_points=2)},
                actions={"consumption": LinSpacedGrid(start=0, stop=1, n_points=2)},
                functions={"utility": _utility},
                constraints={
                    "affordable": _affordable_with_other if needs_other else _affordable
                },
                shocks=("entry",),
            )
        },
        shocks={
            "entry": ShockGroup(
                shocks={
                    "eps": Normal(mu=_location, sigma=None),
                    "other": Normal(mu=0, sigma=1),
                },
                correlation=None,
                nodes=MonteCarlo(2, seed=7),
            )
        },
        ages=AgeGrid(exact_values=(Fraction(41, 2), Fraction(43, 2))),
        edges={},
        initial_nodes=InitialNodes(by_age={20.5: "alive", 21.5: "alive"}),
        regime_id_class=_Entry,
    )


def _forbidden_feasibility(
    *args: object,  # noqa: PAN001 - Rejects every callback operand without inspecting it.
    **kwargs: object,  # noqa: PAN001 - Rejects every callback operand without inspecting it.
) -> None:
    del args, kwargs
    raise AssertionError("Invalid partial frame reached the feasibility callback.")


@pytest.mark.coverage(backends=("cpu",), precisions="both")
@pytest.mark.parametrize("base", [False, True])
def test_public_seedless_partial_frames_keep_original_rows_clock_and_fresh_params(
    *, base: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Actual public helper and its original frame producer need no missing draw."""
    model = _model()
    ages = jnp.asarray([20.5, 21.5, 21.5, 20.5])
    periods = np.asarray([0, 1, 1, 0], dtype=np.int32)
    physical = jnp.asarray([0.5, -1.5, 1.0, -2.0])
    initial = {
        "age": ages,
        "wealth": jnp.ones(4),
        "regime_id": jnp.full(4, model.regime_names_to_ids["alive"], dtype=jnp.int32),
        "eps": BaseInnovations(values=physical - ages - jnp.asarray(periods))
        if base
        else PhysicalValues(values=physical),
    }
    for shift in (0.0, 2.0):
        params = {"entry": {"corr": jnp.eye(2), "eps__sigma": 1.0, "shift": shift}}
        expected = (
            np.ones(4, dtype=bool) if base and shift else [True, False, True, False]
        )
        np.testing.assert_array_equal(
            model.initial_conditions_feasibility(
                initial_conditions=initial, params=params
            ),
            expected,
        )
        canonical, _, supplied = model._canonical_feasibility_inputs(
            initial_conditions=initial, params=params
        )
        assert supplied is not None
        direct = supplied.physical_inputs(
            regime=model._regimes["alive"],
            regime_name="alive",
            initial_states=canonical,
            indices=[3, 1, 0],
            periods=periods,
            ages=model._time,
            needed=frozenset({"eps"}),
            memory=None,
        )
        assert set(direct) == {"eps"}
        np.testing.assert_array_equal(
            direct["eps"], (physical + (shift if base else 0.0))[jnp.asarray([3, 1, 0])]
        )
        with pytest.raises(UnsupportedOperationError, match="other"):
            supplied.physical_inputs(
                regime=model._regimes["alive"],
                regime_name="alive",
                initial_states=canonical,
                indices=[3, 1, 0],
                periods=periods,
                ages=model._time,
                needed=frozenset({"eps", "other"}),
                memory=None,
            )
    model.validate_initial_conditions(
        initial_conditions={
            **initial,
            "eps": BaseInnovations(values=jnp.zeros(4))
            if base
            else PhysicalValues(values=jnp.ones(4)),
        },
        params={"entry": {"corr": jnp.eye(2), "eps__sigma": 1.0, "shift": 0.0}},
    )
    with monkeypatch.context() as guarded:
        guarded.setattr(
            initial_checks, "_batched_feasibility_check", _forbidden_feasibility
        )
        for corr in (
            jnp.asarray([[1.0, 0.25], [0.25, 1.0]]),
            jnp.asarray([[1.0, jnp.nan], [jnp.nan, 1.0]]),
        ):
            with pytest.raises(ShockDeclarationError):
                model.initial_conditions_feasibility(
                    initial_conditions=initial,
                    params={"entry": {"corr": corr, "eps__sigma": 1.0, "shift": 0.0}},
                )
        with pytest.raises(InvalidSimulationInputError, match="shape"):
            model.validate_initial_conditions(
                initial_conditions={
                    **initial,
                    "eps": BaseInnovations(values=jnp.ones((4, 2)))
                    if base
                    else PhysicalValues(values=jnp.ones((4, 2))),
                },
                params={"entry": {"corr": jnp.eye(2), "eps__sigma": 1.0, "shift": 0.0}},
            )
        required = _model(needs_other=True)
        with pytest.raises(UnsupportedOperationError, match="other"):
            required.initial_conditions_feasibility(
                initial_conditions=initial,
                params={"entry": {"corr": jnp.eye(2), "eps__sigma": 1.0, "shift": 0.0}},
            )
    # The prepared model is reusable after the failed nonidentity invocation.
    np.testing.assert_array_equal(
        model.initial_conditions_feasibility(
            initial_conditions=initial,
            params={"entry": {"corr": jnp.eye(2), "eps__sigma": 1.0, "shift": 0.0}},
        ),
        [True, False, True, False],
    )
