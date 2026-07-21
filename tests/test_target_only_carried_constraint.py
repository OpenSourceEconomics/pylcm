"""A within-period read of `next_<state>` needs a producer in that phase's flow.

A *target-only* state is declared in a regime's `state_transitions` but not in its
own `states`: the source regime produces it and hands it to a reachable target that
carries it. Its `next_<state>` is a HANDOVER into the target's state space.

Whether a source utility or constraint may read the unqualified `next_<state>`
within-period is NOT a syntactic property of the source — it depends on whether the
phase's canonical flow supplies a producer. The engine's flow merge
(`_get_deterministic_transitions`) produces an unqualified `next_<state>` whenever a
reachable target carries the state in that phase; the within-period read then resolves
against it. So:

- if a reachable target grids the state ordinarily, the read is VALID and the model
  builds and solves (`test_ordinary_carrier_target_only_read_builds`);
- if NO reachable target grids it in the relevant phase — a carrier that only imputes
  the state in the solve phase, so it has no solve grid — there is no producer and the
  read is rejected early and clearly, per phase, by
  `Q_and_F._fail_if_unproduced_next_state_is_read` (raised at build as a `ValueError`),
  rather than crashing deep in the solve build with a cryptic missing-argument error.

An earlier revision rejected EVERY within-period read of a target-only `next_<state>`
syntactically (and only inspected the solve phase). That over-rejected the valid
ordinary-carrier case and missed simulate-only reads; the producer-aware, per-phase
guard replaces it.
"""

from typing import cast

import jax.numpy as jnp
import pytest

from lcm import AgeGrid, LinSpacedGrid, Model, Phased, categorical
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
    # retired is active only at 62 (see `_*_retired` below); its next regime is dead.
    return jnp.where(age < 62, _RegimeId.retired, _RegimeId.dead)


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


def _carried_retired() -> UserRegime:
    """`retired` carries `pension_wealth`: imputed in solve, gridded in simulate.

    In the solve phase it has NO grid for the state, so no reachable target grids
    `pension_wealth` in solve -> no solve-phase producer for `next_pension_wealth`.
    """
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


def _ordinary_retired() -> UserRegime:
    """`retired` grids `pension_wealth` ORDINARILY in both phases (a real producer)."""
    return UserRegime(
        transition=_from_retired,
        active=lambda age: 62 <= age < 64,
        states={"pension_wealth": LinSpacedGrid(start=0.0, stop=20.0, n_points=4)},
        state_transitions={"pension_wealth": _evolve_pension_wealth},
        functions={"utility": _retired_utility},
    )


def _model(*, working: UserRegime, retired: UserRegime) -> Model:
    return Model(
        regimes={"working": working, "retired": retired, "dead": _DEAD},
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


def test_ordinary_carrier_target_only_read_builds() -> None:
    """A target-only read builds when a reachable target grids the state (a producer).

    `retired` grids `pension_wealth` ordinarily, so the canonical flow supplies an
    unqualified `next_pension_wealth`; `working`'s within-period read resolves against
    it. (Round-8 F2: the previous syntactic validator over-rejected this valid model.)
    """
    model = _model(
        working=_working(transition=_to_retired, reads_in_constraint=False),
        retired=_ordinary_retired(),
    )
    # Building and a finite solve both succeed.
    params = cast("dict", model.get_params_template())
    for regime_params in params.values():
        if "H" in regime_params and "discount_factor" in regime_params["H"]:
            regime_params["H"]["discount_factor"] = 0.95
    solution = model.solve(params=params, log_level="debug")
    working_V = [
        regime_to_V["working"]
        for regime_to_V in solution.values()
        if "working" in regime_to_V
    ]
    assert working_V
    assert all(jnp.all(jnp.isfinite(V)) for V in working_V)


def test_target_only_next_read_in_constraint_is_rejected_without_producer() -> None:
    """A constraint read is rejected with no reachable solve-phase producer."""
    with pytest.raises(ValueError, match="no producer"):
        _model(
            working=_working(transition=_to_retired, reads_in_constraint=True),
            retired=_carried_retired(),
        )


def test_target_only_next_read_in_utility_is_rejected_without_producer() -> None:
    """A utility read is rejected when no reachable target grids the state in solve."""
    with pytest.raises(ValueError, match="no producer"):
        _model(
            working=_working(transition=_to_retired, reads_in_constraint=False),
            retired=_carried_retired(),
        )


def test_target_only_next_read_rejected_under_phase_varying_transition() -> None:
    """The rejection still fires under a phase-varying transition with no producer.

    `Phased(solve=to_dead, simulate=to_retired)` keeps `retired` (the carrier) off the
    solve phase's declared jump; the carrier only imputes `pension_wealth` in solve
    either way, so no solve producer exists and the read is rejected.
    """
    with pytest.raises(ValueError, match="no producer"):
        _model(
            working=_working(
                transition=Phased(solve=_to_dead, simulate=_to_retired),
                reads_in_constraint=True,
            ),
            retired=_carried_retired(),
        )


def test_clean_target_only_handover_without_read_builds() -> None:
    """A target-only state produced but NOT read within-period always builds.

    A plain handover (working produces `pension_wealth` for `retired`, nothing in
    working reads its next value) is the normal, valid use.
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
    # Construction must not raise, with either carrier form.
    _model(working=working, retired=_carried_retired())
    _model(working=working, retired=_ordinary_retired())
