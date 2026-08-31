"""A target-only state is handed over by its source and valued by its carrier.

A *target-only* state is declared in a regime's `state_transitions` but not in its
own `states`: the source regime produces it and hands it to a reachable target that
carries it. Its `next_<state>` is a HANDOVER into the target's state space.

`next_<state>` is reserved vocabulary a transition produces, so nothing this period
— not `utility`, not a constraint — reads one; the rejection of such a read is
covered by `tests/regime_building/test_next_prefix_is_reserved.py`. What is left,
and what this module covers, is the handover itself: the source produces the state
from its own states, and whoever carries it values it at *its* current value.

A bequest is the canonical case and shows why no within-period read is wanted. The
bequest is not part of this period's flow utility — it is asserted **upon death**,
in the terminal `dead` regime, over the wealth that regime carries. So `working`
values consumption now, `dead` values `pension_wealth` then, and the two never need
to meet inside one period.
"""

from typing import cast

import jax.numpy as jnp

from lcm import AgeGrid, LinSpacedGrid, Model, Phased, categorical
from lcm.regime import Regime as UserRegime
from lcm.typing import FloatND, ScalarInt, UserFunction

BEQUEST_SCALE = 0.8


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


def _impute_pension_wealth(aime: float) -> float:
    return aime * 0.1


def _evolve_pension_wealth(pension_wealth: float) -> float:
    return pension_wealth * 1.03


def _next_wealth(*, wealth: float, consumption: float) -> float:
    return wealth - consumption


def _next_aime(aime: float) -> float:
    return aime


def _utility(consumption: float) -> FloatND:
    return jnp.log(consumption)


def _retired_utility(pension_wealth: float) -> FloatND:
    return jnp.log(pension_wealth + 1.0)


def _bequest_utility(pension_wealth: float) -> FloatND:
    """Bequest, asserted upon death over the wealth `dead` carries.

    Reads `pension_wealth` — the terminal regime's OWN state — not a `next_<state>`
    of some earlier period. The handover put the value here; valuing it is this
    regime's job.
    """
    return BEQUEST_SCALE * jnp.log(pension_wealth + 1.0)


def _dead(*, values_bequest: bool) -> UserRegime:
    """Terminal regime, optionally carrying `pension_wealth` to value a bequest."""
    if not values_bequest:
        return UserRegime(transition=None, functions={"utility": lambda: 0.0})
    return UserRegime(
        transition=None,
        states={"pension_wealth": LinSpacedGrid(start=0.0, stop=20.0, n_points=4)},
        functions={"utility": _bequest_utility},
    )


def _carried_retired() -> UserRegime:
    """`retired` carries `pension_wealth`: imputed in solve, gridded in simulate."""
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
    """`retired` grids `pension_wealth` ORDINARILY in both phases."""
    return UserRegime(
        transition=_from_retired,
        active=lambda age: 62 <= age < 64,
        states={"pension_wealth": LinSpacedGrid(start=0.0, stop=20.0, n_points=4)},
        state_transitions={"pension_wealth": _evolve_pension_wealth},
        functions={"utility": _retired_utility},
    )


def _model(
    *, working: UserRegime, retired: UserRegime, values_bequest: bool = False
) -> Model:
    return Model(
        regimes={
            "working": working,
            "retired": retired,
            "dead": _dead(values_bequest=values_bequest),
        },
        ages=AgeGrid(start=60, stop=64, step="2Y"),
        regime_id_class=_RegimeId,
    )


def _working(*, transition: UserFunction | Phased) -> UserRegime:
    """`working` produces target-only `pension_wealth` and hands it over.

    Its own flow utility is consumption now; it does not read the handed-over
    value, because that value belongs to the period the target is in.
    """

    def _feasible_plain(*, consumption: float, wealth: float) -> bool:
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
        constraints={"feasible": _feasible_plain},
        functions={"utility": _utility},
    )


def _solve_working_V(model: Model) -> list[FloatND]:
    params = cast("dict", model.get_params_template())
    for regime_params in params.values():
        aggregator = regime_params.get("koopmans_aggregator")
        if aggregator is not None and "discount_factor" in aggregator:
            aggregator["discount_factor"] = 0.95
    solution = model.solve(params=params, log_level="debug")
    return [
        regime_to_V["working"]
        for regime_to_V in solution.values()
        if "working" in regime_to_V
    ]


def test_target_only_handover_builds_and_solves() -> None:
    """The plain handover — produced by the source, carried by the target."""
    working_V = _solve_working_V(
        _model(
            working=_working(transition=_to_retired),
            retired=_ordinary_retired(),
        )
    )
    assert working_V
    assert all(jnp.all(jnp.isfinite(V)) for V in working_V)


def test_terminal_bequest_over_a_handed_over_state_solves() -> None:
    """The bequest is valued upon death, on the state `dead` carries.

    `working` values consumption in `t` as usual; the bequest enters only through
    the terminal regime's own `utility`, so no period mixes this period's flow with
    a next period's value.
    """
    working_V = _solve_working_V(
        _model(
            working=_working(transition=_to_retired),
            retired=_ordinary_retired(),
            values_bequest=True,
        )
    )
    assert working_V
    assert all(jnp.all(jnp.isfinite(V)) for V in working_V)


def test_bequest_raises_the_value_of_working() -> None:
    """Valuing the bequest is not a no-op: it strictly raises `working`'s value.

    Without this the two tests above would both pass on a model that silently
    dropped the terminal regime's utility.
    """
    without = _solve_working_V(
        _model(working=_working(transition=_to_retired), retired=_ordinary_retired())
    )
    with_bequest = _solve_working_V(
        _model(
            working=_working(transition=_to_retired),
            retired=_ordinary_retired(),
            values_bequest=True,
        )
    )
    assert len(without) == len(with_bequest)
    assert any(bool(jnp.any(b > a)) for a, b in zip(without, with_bequest, strict=True))
    assert all(
        bool(jnp.all(b >= a - 1e-8)) for a, b in zip(without, with_bequest, strict=True)
    )


def test_handover_builds_under_a_phase_varying_carrier() -> None:
    """A carrier that only imputes the state in solve still takes the handover."""
    _model(working=_working(transition=_to_retired), retired=_carried_retired())


def test_handover_builds_under_a_phase_varying_transition() -> None:
    """`Phased(solve=..., simulate=...)` on the source's regime transition."""
    _model(
        working=_working(transition=Phased(solve=_to_retired, simulate=_to_retired)),
        retired=_carried_retired(),
    )
