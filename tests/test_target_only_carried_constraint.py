"""A source constraint may read `next_<state>` for a TARGET-ONLY state that a
reachable target carries -- this is a VALID model, not a solve-topology failure.

Round-5 rejects a constraint reading `next_<carried>` for a state the *reading*
regime carries (imputed in solve -> its transition omitted -> no producer). A
later review argued the symmetric case -- a state that is target-only in the
source (declared in `state_transitions`, absent from the source's `states`) and
carried in a reachable target -- evades that rejection and then fails at solve
with an unsupplied `next_<state>`.

It does not. A well-formed target-only law depends on the *source's* own states,
so the source regime both produces and reads `next_<state>` in its solve slice;
the target's carrying only omits the target's own imputation, never the source's
producer. The model therefore builds and solves, and the constraint is truly
enforced. These tests pin that: construction is accepted, and an impossible bound
makes the source value entirely infeasible (proving the constraint is live, not
silently dropped), while a feasible bound leaves it finite.
"""

from typing import cast

import jax.numpy as jnp
import numpy as np

from lcm import AgeGrid, LinSpacedGrid, Model, Phased, categorical
from lcm.regime import Regime as UserRegime
from lcm.typing import FloatND, ScalarInt, UserParams


@categorical(ordered=False)
class _RegimeId:
    working: ScalarInt
    retired: ScalarInt
    dead: ScalarInt


def _to_retired(age: float) -> ScalarInt:  # noqa: ARG001
    return _RegimeId.retired


def _from_retired(age: float) -> ScalarInt:
    return jnp.where(age < 64, _RegimeId.retired, _RegimeId.dead)


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


def _retired_utility(pension_wealth: float) -> FloatND:
    return jnp.log(pension_wealth + 1.0)


_DEAD = UserRegime(transition=None, functions={"utility": lambda: 0.0})


def _build(*, floor: float) -> Model:
    """Working reads `next_pension_wealth`, a target-only state carried in retired.

    `floor` is the lower bound the source constraint imposes on the produced
    `next_pension_wealth`. pension_wealth = aime * 0.1 with aime in [1, 50], so a
    floor of 0.0 is always feasible and a floor of 100.0 is never feasible.
    """

    def _feasible(
        consumption: float, wealth: float, next_pension_wealth: float
    ) -> bool:
        return (consumption <= wealth) & (next_pension_wealth >= floor)

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
            # Target-only: produced from working's own state (aime), handed to
            # retired; working does not grid pension_wealth in its `states`.
            "pension_wealth": _impute_pension_wealth,
        },
        actions={"consumption": LinSpacedGrid(start=1.0, stop=10.0, n_points=5)},
        constraints={"feasible": _feasible},
        functions={"utility": _utility},
    )
    retired = UserRegime(
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
    return Model(
        regimes={"working": working, "retired": retired, "dead": _DEAD},
        ages=AgeGrid(start=60, stop=64, step="2Y"),
        regime_id_class=_RegimeId,
    )


def _fill(tree: object) -> object:
    if isinstance(tree, dict):
        return {k: _fill(v) for k, v in tree.items()}
    if isinstance(tree, str):
        return 0.95
    return tree


def _working_value_at_start(model: Model) -> np.ndarray:
    params = cast("UserParams", _fill(model.get_params_template()))
    solution = model.solve(params=params, log_level="warning")
    return np.asarray(solution[0]["working"])


def test_target_only_carried_constraint_is_accepted_and_solves() -> None:
    """Construction accepts the model and solve produces a finite source value."""
    value = _working_value_at_start(_build(floor=0.0))
    assert np.isfinite(value).any()


def test_target_only_carried_constraint_is_actually_enforced() -> None:
    """An impossible bound makes the whole source regime infeasible.

    If the `next_pension_wealth`-reading constraint were silently dropped, solve
    would return finite values here; instead every source state is -inf, proving
    `next_pension_wealth` is genuinely produced and the constraint enforced.
    """
    value = _working_value_at_start(_build(floor=100.0))
    assert np.all(value == -np.inf)


def _to_dead(age: float) -> ScalarInt:  # noqa: ARG001
    return _RegimeId.dead


def _build_phase_varying(*, floor: float) -> Model:
    """As `_build`, but working's regime transition is PHASE-VARYING.

    `Phased(solve=to_dead, simulate=to_retired)` makes the carrier `retired`
    reachable only in SIMULATE, while solve goes to the stateless terminal `dead`.
    An earlier review argued this is where a target-only carried state must fail at
    solve for want of a producer. It does not: working's own target-only transition
    produces `next_pension_wealth` from `aime` in working's solve slice regardless of
    which regime it targets, so the source constraint resolves in both phases.
    """

    def _feasible(
        consumption: float, wealth: float, next_pension_wealth: float
    ) -> bool:
        return (consumption <= wealth) & (next_pension_wealth >= floor)

    working = UserRegime(
        transition=Phased(solve=_to_dead, simulate=_to_retired),
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
        constraints={"feasible": _feasible},
        functions={"utility": _utility},
    )
    retired = UserRegime(
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
    return Model(
        regimes={"working": working, "retired": retired, "dead": _DEAD},
        ages=AgeGrid(start=60, stop=64, step="2Y"),
        regime_id_class=_RegimeId,
    )


def test_phase_varying_transition_target_only_carried_is_accepted_and_solves() -> None:
    """Carrier reachable only in simulate: still builds and solves finite."""
    value = _working_value_at_start(_build_phase_varying(floor=0.0))
    assert np.isfinite(value).any()


def test_phase_varying_transition_target_only_carried_is_enforced() -> None:
    """The producer exists in solve even when the carrier is simulate-only.

    An impossible bound makes the source regime entirely infeasible, proving
    `next_pension_wealth` is produced in the solve flow and the constraint enforced --
    the phase-varying reachability does not remove the source's own producer.
    """
    value = _working_value_at_start(_build_phase_varying(floor=100.0))
    assert np.all(value == -np.inf)
