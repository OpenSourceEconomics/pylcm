"""A simulate-phase joint lottery is validated on the simulation grids.

A carried state (`Phased(solve=callable, simulate=Grid)`) has no solve grid
axis, so a simulate-phase probability function that reads one cannot be
evaluated on the solution state-action space. Validating it there leaves an
invalid probability row unexamined and simulation then samples from it, so the
preflight reads the simulate phase on the simulation grids instead — and where
it still cannot evaluate a lottery, `log_level="debug"` refuses to run it rather
than continuing on an unvalidated law.
"""

from types import MappingProxyType

import jax.numpy as jnp
import pytest

from _lcm.engine import StateActionSpace
from _lcm.transition_checks import _evaluate_joint_weights
from _lcm.utils.logging import get_logger
from lcm import AgeGrid, JointTransition, LinSpacedGrid, Model, Phased, categorical
from lcm.exceptions import InvalidStateTransitionProbabilitiesError, PyLCMError
from lcm.regime import Regime
from lcm.typing import FloatND, ScalarInt

_SUPPORT = jnp.asarray([1.0, 2.0])


@categorical(ordered=False)
class RegimeId:
    working: ScalarInt
    dead: ScalarInt


def _next_regime(age: float) -> ScalarInt:
    return jnp.where(age < 61, RegimeId.working, RegimeId.dead)


def _impute_experience(wealth: float) -> float:
    """Solve-phase experience: imputed, so it never becomes a grid axis."""
    return wealth * 0.1


def _evolve_experience(experience: float) -> float:
    return experience + 1.0


def _utility(consumption: float) -> FloatND:
    return jnp.log(consumption)


def _consumption_leq_wealth(*, consumption: float, wealth: float) -> bool:
    return consumption <= wealth


def _valid_probabilities() -> FloatND:
    """Solve-phase law: an ordinary probability vector over the two nodes."""
    return jnp.asarray([0.5, 0.5])


def _invalid_probabilities_from_carried_state(experience: float) -> FloatND:
    """Simulate-phase law reading the carried state, and not a distribution."""
    return jnp.asarray([1.2, -0.2]) + 0.0 * experience


def _next_wealth_from_match(match: dict[str, FloatND]) -> FloatND:
    return match["value"]


def _kernel(probabilities) -> JointTransition:
    return JointTransition(
        support_size=2,
        support={"value": _SUPPORT},
        probabilities=probabilities,
        outputs={"wealth": _next_wealth_from_match},
    )


def _valid_probabilities_from_carried_state(experience: float) -> FloatND:
    """A distribution that also reads the carried state, for the control."""
    return jnp.asarray([0.5, 0.5]) + 0.0 * experience


def _params() -> dict[str, dict[str, dict[str, float]]]:
    return {
        "working": {"koopmans_aggregator": {"discount_factor": 0.95}},
        "dead": {},
    }


def _build_model(*, simulate_probabilities) -> Model:
    working = Regime(
        transition=_next_regime,
        active=lambda age: age < 64,
        states={
            "wealth": LinSpacedGrid(start=1.0, stop=10.0, n_points=3),
            "experience": Phased(
                solve=_impute_experience,
                simulate=LinSpacedGrid(start=0.0, stop=4.0, n_points=3),
            ),
        },
        state_transitions={"experience": _evolve_experience},
        actions={"consumption": LinSpacedGrid(start=1.0, stop=5.0, n_points=3)},
        constraints={"feasible_consumption": _consumption_leq_wealth},
        functions={"utility": _utility},
        joint_transitions={
            "working": {
                "match": Phased(
                    solve=_kernel(_valid_probabilities),
                    simulate=_kernel(simulate_probabilities),
                )
            }
        },
    )
    dead = Regime(transition=None, functions={"utility": lambda: jnp.asarray(0.0)})
    return Model(
        regimes={"working": working, "dead": dead},
        ages=AgeGrid(start=60, stop=64, step="2Y"),
        regime_id_class=RegimeId,
    )


def test_simulate_rejects_a_joint_law_that_is_not_a_distribution() -> None:
    """Debug simulation refuses a simulate-phase row of `[1.2, -0.2]`.

    The row is a function of a carried state, so it is only visible on the
    simulation grids. Leaving it unchecked lets simulation sample the first
    support node from weights that are neither non-negative nor normalized.
    """
    model = _build_model(
        simulate_probabilities=_invalid_probabilities_from_carried_state
    )

    initial_conditions = {
        "age": jnp.array([60.0, 60.0]),
        "wealth": jnp.array([5.0, 8.0]),
        "experience": jnp.array([1.0, 2.0]),
        "regime_id": jnp.array([RegimeId.working, RegimeId.working]),
    }

    with pytest.raises(PyLCMError) as excinfo:
        model.simulate(
            params=_params(),
            initial_conditions=initial_conditions,
            log_level="debug",
        )

    message = str(excinfo.value)
    assert "match" in message
    assert "simulate phase" in message
    assert "working" in message


def test_a_carried_state_dependent_simulate_law_that_is_a_distribution_runs() -> None:
    """The same shape of model runs when its simulate-phase row sums to one.

    Without this control the rejection above could come from reading a carried
    state at all rather than from the row `[1.2, -0.2]`, and the check would be
    refusing a supported declaration.
    """
    model = _build_model(simulate_probabilities=_valid_probabilities_from_carried_state)

    result = model.simulate(
        params=_params(),
        initial_conditions={
            "age": jnp.array([60.0, 60.0]),
            "wealth": jnp.array([5.0, 8.0]),
            "experience": jnp.array([1.0, 2.0]),
            "regime_id": jnp.array([RegimeId.working, RegimeId.working]),
        },
        log_level="debug",
    )

    assert result.n_subjects == 2


def test_an_unresolvable_joint_argument_is_refused_in_debug() -> None:
    """A weight law whose argument no grid or parameter supplies stops the run.

    Continuing past it would sample a transition-local lottery from a law that
    was never examined, which reads exactly like a validated one.
    """

    def _weights(unknown_quantity: float) -> dict[str, FloatND]:
        return {"match": jnp.asarray([0.5, 0.5]) + 0.0 * unknown_quantity}

    logger = get_logger(log_level="debug")
    with pytest.raises(InvalidStateTransitionProbabilitiesError) as excinfo:
        _evaluate_joint_weights(
            func=_weights,
            state_action_space=StateActionSpace(
                states=MappingProxyType({}),
                discrete_actions=MappingProxyType({}),
                continuous_actions=MappingProxyType({}),
                state_and_discrete_action_names=(),
            ),
            extra_grids=MappingProxyType({}),
            regime_params=MappingProxyType({}),
            period=jnp.int32(0),
            age=jnp.int32(60),
            regime_name="working",
            phase_name="simulate",
            logger=logger,
        )

    message = str(excinfo.value)
    assert "unknown_quantity" in message
    assert "working" in message
