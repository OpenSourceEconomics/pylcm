"""Build-time phase capability for NBEGM-family declarations.

NNBEGM can replay only its solve-time keeper-plus-outer-grid candidate set. A
phase-varying declaration therefore must fail during ``Model`` construction;
falling back to generic simulation-grid maximization changes candidate support.
"""

from collections.abc import Callable, Mapping
from typing import cast

import jax.numpy as jnp
import pytest

from _lcm.engine import NNBEGMPolicyRead
from _lcm.regime_building.phases import normalize_regime_phases
from _lcm.solution.nnbegm import _BoundNNBEGM
from lcm import LinearAggregator, Phased
from lcm.exceptions import ModelInitializationError
from lcm.solvers import NBEGM
from lcm.typing import ContinuousAction, ContinuousState, FloatND, ScalarInt
from tests.test_models import n_nbegm_toy
from tests.test_models.nbegm_common import (
    make_alive_dead_model,
    next_liquid_from_savings,
    savings,
)


def _utility_clone(consumption: ContinuousAction) -> FloatND:
    """Semantically equal to the toy utility, but a distinct Python object."""
    return n_nbegm_toy.utility(consumption)


def _durable_transition_clone(
    new_illiquid: ContinuousState,
) -> ContinuousState:
    """Semantically equal to the toy durable law, but a distinct object."""
    return new_illiquid


def _next_regime_clone(*, age: int, final_age_alive: float) -> ScalarInt:
    """Semantically equal to the toy regime law, but a distinct object."""
    return jnp.where(
        age >= final_age_alive,
        n_nbegm_toy.RegimeId.dead,
        n_nbegm_toy.RegimeId.alive,
    )


def _resources(liquid: ContinuousState) -> FloatND:
    """Identity cash-on-hand node for the bare NBEGM construction witness."""
    return liquid


def _structural_phase_variations(model) -> tuple[str, ...]:
    """Independent oracle over normalized public declarations.

    This reads the existing phase slices and compares corresponding objects by
    ``is``. It does not call the production replay-capability classifier and it
    does not infer capability from ``NNBEGMPolicyRead``.
    """
    regime = model.user_regimes["alive"]
    spec = normalize_regime_phases(regime)
    solve = spec.solution
    simulate = spec.simulation
    varied: list[str] = []

    for slot_name, solve_mapping, simulate_mapping in (
        ("functions", solve.functions, simulate.functions),
        ("states", solve.grid_states, simulate.grid_states),
        ("state_transitions", solve.state_transitions, simulate.state_transitions),
    ):
        names = set(solve_mapping) | set(simulate_mapping)
        varied.extend(
            f"{slot_name}[{name!r}]"
            for name in sorted(names)
            if (
                name not in solve_mapping
                or name not in simulate_mapping
                or solve_mapping[name] is not simulate_mapping[name]
            )
        )

    if solve.regime_transition is not simulate.regime_transition:
        varied.append("transition")
    if solve.koopmans_aggregator is not simulate.koopmans_aggregator:
        varied.append("koopmans_aggregator")
    return tuple(varied)


def _assert_nnbegm_replay(model) -> None:
    assert _structural_phase_variations(model) == ()
    assert isinstance(
        model._regimes["alive"].simulation.egm_policy_read,
        NNBEGMPolicyRead,
    )


def _build_nnbegm(**kwargs):
    return n_nbegm_toy.build_model(variant="n_nbegm", n_periods=2, **kwargs)


def test_nbegm_accepts_an_identical_object_phased_utility() -> None:
    """Bare NBEGM validators consume the solve-resolved function pool."""
    phased_utility = Phased(
        solve=n_nbegm_toy.utility,
        simulate=n_nbegm_toy.utility,
    )
    alive_functions = cast(
        "Mapping[str, Callable[..., object]]",
        {
            "utility": phased_utility,
            "resources": _resources,
            "savings": savings,
        },
    )
    model = make_alive_dead_model(
        n_periods=2,
        n_liquid=5,
        liquid_max=10.0,
        n_consumption=8,
        alive_functions=alive_functions,
        liquid_law=next_liquid_from_savings,
        alive_solver=NBEGM(
            savings_grid=n_nbegm_toy.SAVINGS_GRID,
        ),
        constraints={},
        liquid_post_decision="savings",
    )
    assert model.user_regimes["alive"].functions["utility"] is phased_utility


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {
            "utility_function": Phased(
                solve=n_nbegm_toy.utility,
                simulate=n_nbegm_toy.utility,
            )
        },
    ],
    ids=["bare", "identical-utility"],
)
def test_nnbegm_phase_invariant_controls_retain_replay(kwargs) -> None:
    """Bare and identity-invariant utility declarations retain exact replay."""
    _assert_nnbegm_replay(_build_nnbegm(**kwargs))


def test_nnbegm_identical_object_phased_aggregator_retains_replay() -> None:
    """The singleton aggregator slot participates in identity classification."""
    aggregator = LinearAggregator()
    model = _build_nnbegm(
        koopmans_aggregator=Phased(solve=aggregator, simulate=aggregator)
    )
    _assert_nnbegm_replay(model)


def test_nnbegm_rejects_variation_before_period_kernels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The capability boundary runs before any period kernel is constructed."""

    def must_not_build(*_args, **_kwargs):
        raise AssertionError("NNBEGM period kernels were reached")

    monkeypatch.setattr(_BoundNNBEGM, "build_period_kernels", must_not_build)
    with pytest.raises(
        ModelInitializationError,
        match=r"NNBEGM.*(?:phase variation|replay capability)",
    ):
        _build_nnbegm(
            utility_function=Phased(
                solve=n_nbegm_toy.utility,
                simulate=_utility_clone,
            )
        )


@pytest.mark.parametrize(
    ("kwargs", "oracle_path"),
    [
        (
            {
                "utility_function": Phased(
                    solve=n_nbegm_toy.utility,
                    simulate=_utility_clone,
                )
            },
            "functions['utility']",
        ),
        (
            {
                "durable_law": Phased(
                    solve=n_nbegm_toy.durable_transition,
                    simulate=_durable_transition_clone,
                )
            },
            "state_transitions['illiquid']",
        ),
        (
            {
                "regime_transition": Phased(
                    solve=n_nbegm_toy.next_regime,
                    simulate=_next_regime_clone,
                )
            },
            "transition",
        ),
        (
            {
                "koopmans_aggregator": Phased(
                    solve=LinearAggregator(),
                    simulate=LinearAggregator(),
                )
            },
            "koopmans_aggregator",
        ),
        (
            {"carried_state": True},
            "states['permanent_income']",
        ),
    ],
    ids=[
        "utility",
        "state-transition",
        "regime-transition",
        "aggregator",
        "carried-state",
    ],
)
def test_nnbegm_rejects_every_genuine_phase_variation(
    *,
    kwargs,
    oracle_path: str,
) -> None:
    """Distinct objects fail closed instead of selecting over a foreign grid."""
    # The oracle model is built under grid search so production NNBEGM capability
    # cannot affect its classification.
    oracle_model = n_nbegm_toy.build_model(
        variant="brute",
        n_periods=2,
        **kwargs,
    )
    assert oracle_path in _structural_phase_variations(oracle_model)

    with pytest.raises(
        ModelInitializationError,
        match=r"NNBEGM.*(?:phase variation|replay capability)",
    ):
        _build_nnbegm(**kwargs)
