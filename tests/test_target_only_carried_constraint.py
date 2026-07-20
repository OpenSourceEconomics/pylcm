"""A within-period read of `next_<target-only-state>` is REJECTED at construction.

A *target-only* state is declared in a regime's `state_transitions` but not in its
own `states`: the source regime produces it and hands it to a reachable target that
carries it. Its `next_<state>` is therefore a HANDOVER into the target's state
space, not a within-period node of the source — the canonical solve slice routes it
under `<target>__next_<state>`, leaving no unqualified `next_<state>` producer in the
source's own feasibility/utility DAG.

An earlier revision of these tests asserted the opposite (that such a model builds
and solves, with an impossible bound proving the constraint live). That was
confounded: the read resolved only because a parameter-discovery gap misclassified
`next_pension_wealth` as a user parameter, which the test then filled with a finite
number — reproducing both a finite and a `-inf` value WITHOUT the transition ever
being wired in. With parameter discovery fixed, the read has no solve-phase producer
and the model must be rejected early and clearly rather than crashing deep in the
solve build. These tests pin the rejection, for a constraint read and a utility read,
and under a phase-varying regime transition.
"""

import jax.numpy as jnp
import pytest

from lcm import AgeGrid, LinSpacedGrid, Model, Phased, categorical
from lcm.exceptions import ModelInitializationError
from lcm.regime import Regime as UserRegime
from lcm.typing import FloatND, ScalarInt, UserFunction


@categorical(ordered=False)
class _RegimeId:
    working: ScalarInt
    retired: ScalarInt
    dead: ScalarInt


def _to_retired(age: float) -> ScalarInt:  # noqa: ARG001
    return _RegimeId.retired


def _from_retired(age: float) -> ScalarInt:
    return jnp.where(age < 64, _RegimeId.retired, _RegimeId.dead)


def _to_dead(age: float) -> ScalarInt:  # noqa: ARG001
    return _RegimeId.dead


def _impute_pension_wealth(aime: float) -> float:
    return aime * 0.1


def _evolve_pension_wealth(pension_wealth: float) -> float:
    return pension_wealth * 1.03


def _next_wealth(wealth: float, consumption: float) -> float:
    return wealth - consumption


def _next_aime(aime: float) -> float:
    return aime


def _utility(consumption: float) -> FloatND:
    return jnp.log(consumption)


def _bequest_utility(consumption: float, next_pension_wealth: float) -> FloatND:
    """Utility that reads the next value of the target-only state."""
    return jnp.log(consumption) + jnp.log(next_pension_wealth + 1.0)


def _retired_utility(pension_wealth: float) -> FloatND:
    return jnp.log(pension_wealth + 1.0)


_DEAD = UserRegime(transition=None, functions={"utility": lambda: 0.0})


def _retired() -> UserRegime:
    return UserRegime(
        transition=_from_retired,
        active=lambda age: 62 <= age < 64,
        states={
            "pension_wealth": Phased(
                solve=_impute_pension_wealth,
                simulate=LinSpacedGrid(start=0.0, stop=20.0, n_points=4),
            ),
        },
        state_transitions={"pension_wealth": _evolve_pension_wealth},
        functions={"utility": _retired_utility},
    )


def _model(*, working: UserRegime) -> Model:
    return Model(
        regimes={"working": working, "retired": _retired(), "dead": _DEAD},
        ages=AgeGrid(start=60, stop=64, step="2Y"),
        regime_id_class=_RegimeId,
    )


def _working(
    *, transition: UserFunction | Phased, reads_in_constraint: bool
) -> UserRegime:
    """`working` produces target-only `pension_wealth`, read within-period.

    When `reads_in_constraint` is True a feasibility constraint reads
    `next_pension_wealth`; otherwise `utility` reads it (a bequest term).
    """

    def _feasible(
        consumption: float, wealth: float, next_pension_wealth: float
    ) -> bool:
        return (consumption <= wealth) & (next_pension_wealth >= 0.0)

    def _feasible_plain(consumption: float, wealth: float) -> bool:
        return consumption <= wealth

    return UserRegime(
        transition=transition,
        active=lambda age: age < 62,
        states={
            "wealth": LinSpacedGrid(start=1.0, stop=100.0, n_points=8),
            "aime": LinSpacedGrid(start=1.0, stop=50.0, n_points=4),
        },
        state_transitions={
            "wealth": _next_wealth,
            "aime": _next_aime,
            # Target-only: produced from working's own state (aime), handed to
            # retired; working does not grid pension_wealth in its `states`.
            "pension_wealth": _impute_pension_wealth,
        },
        actions={"consumption": LinSpacedGrid(start=1.0, stop=10.0, n_points=5)},
        constraints={"feasible": _feasible if reads_in_constraint else _feasible_plain},
        functions={"utility": _utility if reads_in_constraint else _bequest_utility},
    )


def test_target_only_next_read_in_constraint_is_rejected() -> None:
    """A constraint reading `next_<target-only-state>` is rejected at construction."""
    with pytest.raises(ModelInitializationError, match="target-only state"):
        _model(working=_working(transition=_to_retired, reads_in_constraint=True))


def test_target_only_next_read_in_utility_is_rejected() -> None:
    """A utility reading `next_<target-only-state>` is rejected at construction."""
    with pytest.raises(ModelInitializationError, match="target-only state"):
        _model(working=_working(transition=_to_retired, reads_in_constraint=False))


def test_clean_target_only_handover_without_read_builds() -> None:
    """A target-only state produced but NOT read within-period is accepted.

    The rejection is scoped to a within-period *read* of `next_<target-only>`; a plain
    handover (working produces `pension_wealth` for `retired`, nothing in working reads
    its next value) is the normal, valid use and must still build.
    """

    def _feasible_plain(consumption: float, wealth: float) -> bool:
        return consumption <= wealth

    working = UserRegime(
        transition=_to_retired,
        active=lambda age: age < 62,
        states={
            "wealth": LinSpacedGrid(start=1.0, stop=100.0, n_points=8),
            "aime": LinSpacedGrid(start=1.0, stop=50.0, n_points=4),
        },
        state_transitions={
            "wealth": _next_wealth,
            "aime": _next_aime,
            "pension_wealth": _impute_pension_wealth,
        },
        actions={"consumption": LinSpacedGrid(start=1.0, stop=10.0, n_points=5)},
        constraints={"feasible": _feasible_plain},
        functions={"utility": _utility},
    )
    # Construction must not raise.
    _model(working=working)


def test_target_only_next_read_rejected_under_phase_varying_transition() -> None:
    """The rejection also fires when the carrier is reachable only in simulate.

    `Phased(solve=to_dead, simulate=to_retired)` makes `retired` (the carrier)
    reachable only in the simulate phase; the read is invalid regardless.
    """
    with pytest.raises(ModelInitializationError, match="target-only state"):
        _model(
            working=_working(
                transition=Phased(solve=_to_dead, simulate=_to_retired),
                reads_in_constraint=True,
            )
        )
