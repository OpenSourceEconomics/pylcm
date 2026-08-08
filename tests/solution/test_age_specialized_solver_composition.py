"""`AgeSpecializedGrid` composes with every solver, not only `GridSearch`.

An age-varying grid is a property of a *state*, so it has to be usable whatever
solver consumes that state. The EGM solvers read a state grid in two ways, and only
one of them is age-dependent:

- shape traits — the grid's class, `n_points`, dtype and `batch_size`. These are
  invariant across ages by the `AgeSpecializedGrid` contract, so the representative
  grid answers them exactly.
- node values — these genuinely differ per age, so anything numerical must read the
  period's own grid.

The tests below pin both halves: an age-invariant specialized grid must reproduce the
plain solve bit-for-bit (the machinery collapses cleanly), and an age-varying one must
be solved on each period's own nodes.
"""

import dataclasses

import numpy as np
import pytest

from _lcm.regime_building import processing
from lcm import AgeGrid, AgeSpecializedGrid, LinSpacedGrid, Model
from lcm.transition import AgeSpecializedFunction
from tests.test_models import (
    n_nbegm_toy,
    nbegm_continuous_ride_along_toy,
    nbegm_medicaid_toy,
    negm_kinked_toy,
)
from tests.test_models.dcegm_paper_twin import (
    DCEGM_SOLVER,
    MIN_AGE,
    N_PERIODS,
    WEALTH_GRID,
    TwinRegimeId,
    _retirement,
    _working_life,
    done_from_working,
    done_retired,
    get_params,
)

_NEGM_PARAMS = {"discount_factor": 0.95, "alive": {}}


def _dcegm_twin_with_wealth_grid(wealth_grid) -> Model:
    """The DC-EGM twin, with its `wealth` state declared on `wealth_grid`."""
    solver = dataclasses.replace(DCEGM_SOLVER)
    return Model(
        regimes={
            "working_life": _working_life("dcegm")
            .replace(solver=solver)
            .replace(states={"wealth": wealth_grid}),
            "retirement": _retirement("dcegm")
            .replace(solver=solver)
            .replace(states={"wealth": wealth_grid}),
            "done_from_working": done_from_working.replace(
                states={"wealth": wealth_grid}
            ),
            "done_retired": done_retired.replace(states={"wealth": wealth_grid}),
        },
        ages=AgeGrid(start=MIN_AGE, stop=MIN_AGE + N_PERIODS - 1, step="Y"),
        regime_id_class=TwinRegimeId,
    )


def test_dcegm_accepts_an_age_specialized_euler_state():
    """A DC-EGM regime may declare its Euler state on an `AgeSpecializedGrid`.

    The grid is the state's, not the solver's, so which solver consumes the state
    cannot decide whether an age-varying grid is allowed.
    """
    model = _dcegm_twin_with_wealth_grid(
        AgeSpecializedGrid(build=lambda _age: WEALTH_GRID, signature=lambda _age: 0)
    )
    assert "wealth" in model.user_regimes["working_life"].states


def test_age_invariant_specialized_grid_reproduces_the_plain_dcegm_solve():
    """An age-invariant `AgeSpecializedGrid` equals the plain-grid DC-EGM solve.

    With the same grid at every age the per-period machinery has nothing to vary, so
    it must collapse to exactly the plain solve — equality, not agreement to a
    tolerance, since any difference here is a defect rather than rounding.
    """
    params = get_params()
    plain = _dcegm_twin_with_wealth_grid(WEALTH_GRID).solve(
        params=params, log_level="debug"
    )
    specialized = _dcegm_twin_with_wealth_grid(
        AgeSpecializedGrid(build=lambda _age: WEALTH_GRID, signature=lambda _age: 0)
    ).solve(params=params, log_level="debug")

    for period, regime_to_V in plain.items():
        for regime_name, V_arr in regime_to_V.items():
            expected = np.asarray(V_arr)
            got = np.asarray(specialized[period][regime_name])
            np.testing.assert_array_equal(
                np.isneginf(got),
                np.isneginf(expected),
                err_msg=f"period {period}, regime {regime_name}: -inf pattern differs",
            )
            finite = np.isfinite(expected)
            np.testing.assert_array_equal(
                got[finite],
                expected[finite],
                err_msg=f"period {period}, regime {regime_name}: values differ",
            )


@pytest.mark.parametrize("shift_per_year", [0.0, 0.25])
def test_dcegm_solves_an_age_varying_euler_state_finitely(shift_per_year):
    """A DC-EGM regime with a moving Euler grid solves to a finite value function.

    Every node of every age's grid stays a reachable wealth level, so nothing is
    infeasible; a period solved on some other age's nodes would read the continuation
    at coordinates that period's grid never covers.
    """

    def shifted(age):
        shift = shift_per_year * (age - MIN_AGE)
        nodes = np.asarray(WEALTH_GRID.to_jax())
        return dataclasses.replace(
            WEALTH_GRID, points=tuple(float(node) + shift for node in nodes)
        )

    grid = AgeSpecializedGrid(
        build=shifted, signature=lambda age: shift_per_year * (age - MIN_AGE)
    )
    solution = _dcegm_twin_with_wealth_grid(grid).solve(
        params=get_params(), log_level="debug"
    )

    offending = [
        (period, regime_name)
        for period, regime_to_V in solution.items()
        for regime_name, V_arr in regime_to_V.items()
        if np.isnan(np.asarray(V_arr)).any()
    ]
    assert offending == [], f"NaN in the value function at {offending}"


def _negm_toy_with_illiquid_grid(illiquid_grid) -> Model:
    """The kinked NEGM toy, with its durable `illiquid` state on `illiquid_grid`."""
    final_age_alive = 20 + (negm_kinked_toy.N_PERIODS - 2) * 5
    alive = negm_kinked_toy.build_alive_regime()
    return Model(
        regimes={
            "alive": alive.replace(
                states={**dict(alive.states), "illiquid": illiquid_grid}
            ),
            "dead": negm_kinked_toy.build_dead_regime(),
        },
        regime_id_class=negm_kinked_toy.RegimeId,
        ages=AgeGrid(
            start=20, stop=20 + (negm_kinked_toy.N_PERIODS - 1) * 5, step="5Y"
        ),
        fixed_params={"final_age_alive": final_age_alive},
    )


def test_negm_reads_the_durable_grid_of_each_period():
    """NEGM's durable nodes come from the period being solved, not one fixed age.

    Backward induction is causal: a period's value function can depend only on that
    period's own grid and on later ones. So two durable schedules that agree from the
    second age onward must produce the *same* value function from the second period
    onward, however much they disagree at the first age.

    NEGM lifts the durable nodes into the credited-cost shift of cash on hand, a read
    entirely separate from the durable state axis. Capturing those nodes once, at the
    first active age, breaks the causal identity above: the first age's disagreement
    then leaks into every later period.
    """
    base = negm_kinked_toy.ILLIQUID_GRID
    shifted = dataclasses.replace(base, start=base.start + 4.0, stop=base.stop + 4.0)
    # Age 20 is the first active age, hence the representative one.
    first_age_only_differs = AgeSpecializedGrid(
        build=lambda age: base if age == 20 else shifted,
        signature=lambda age: 0.0 if age == 20 else 4.0,
    )
    shifted_throughout = AgeSpecializedGrid(
        build=lambda _age: shifted, signature=lambda _age: 4.0
    )

    from_first_age = _negm_toy_with_illiquid_grid(first_age_only_differs).solve(
        params=_NEGM_PARAMS, log_level="debug"
    )
    throughout = _negm_toy_with_illiquid_grid(shifted_throughout).solve(
        params=_NEGM_PARAMS, log_level="debug"
    )

    for period in sorted(from_first_age):
        if period == 0 or "alive" not in from_first_age[period]:
            continue
        np.testing.assert_array_equal(
            np.asarray(from_first_age[period]["alive"]),
            np.asarray(throughout[period]["alive"]),
            err_msg=(
                f"period {period} moved when only the first age's durable grid "
                "changed — the durable nodes are captured once instead of read "
                "per period"
            ),
        )


def _nbegm_toy_with_liquid_grid(liquid_grid) -> Model:
    """The Medicaid case-piece toy, with its `liquid` state on `liquid_grid`."""
    return nbegm_medicaid_toy.build_model(
        variant="nbegm", n_periods=4, liquid_grid=liquid_grid
    )


def test_nbegm_reads_the_liquid_grid_of_each_period():
    """NBEGM's liquid nodes come from the period being solved, not one fixed age.

    Backward induction is causal: a period's value function can depend only on that
    period's own grid and on later ones. So two liquid schedules that agree from the
    second age onward must produce the *same* value function from the second period
    onward, however much they disagree at the first age.

    The liquid nodes are the endogenous-grid solve's merge target and the coordinates
    at which the next period's value is read, so pinning them to one age moves every
    period's answer.
    """
    base = LinSpacedGrid(start=0.1, stop=20.0, n_points=40)
    shifted = dataclasses.replace(base, start=base.start + 4.0, stop=base.stop + 4.0)
    # Age 0 is the first active age, hence the representative one.
    first_age_only_differs = AgeSpecializedGrid(
        build=lambda age: base if age == 0 else shifted,
        signature=lambda age: 0.0 if age == 0 else 4.0,
    )
    shifted_throughout = AgeSpecializedGrid(
        build=lambda _age: shifted, signature=lambda _age: 4.0
    )

    params = nbegm_medicaid_toy.build_params()
    from_first_age = _nbegm_toy_with_liquid_grid(first_age_only_differs).solve(
        params=params, log_level="debug"
    )
    throughout = _nbegm_toy_with_liquid_grid(shifted_throughout).solve(
        params=params, log_level="debug"
    )

    for period in sorted(from_first_age):
        if period == 0:
            continue
        for regime_name, V_arr in from_first_age[period].items():
            np.testing.assert_array_equal(
                np.asarray(V_arr),
                np.asarray(throughout[period][regime_name]),
                err_msg=(
                    f"period {period}, regime {regime_name} moved when only the "
                    "first age's liquid grid changed — the liquid nodes are "
                    "captured once instead of read per period"
                ),
            )


def test_nbegm_reads_its_ride_along_continuation_on_the_next_period_s_grid():
    """NBEGM interpolates `V_{t+1}` over a co-state on that co-state's `t+1` nodes.

    A ride-along co-state is passive: the period-`t` kernel reads its continuation by
    interpolating the target's value function over the co-state axis. With an
    age-varying co-state grid those nodes move, so the read belongs to period `t + 1`.

    Backward induction is causal, so two co-state schedules agreeing from the second
    age onward must give the same value function from the second period onward. A
    kernel that interpolates on the representative age's nodes instead lets the first
    age's disagreement reach every earlier period.
    """
    base = LinSpacedGrid(start=0.5, stop=6.0, n_points=6)
    shifted = dataclasses.replace(base, start=base.start + 1.0, stop=base.stop + 1.0)
    # Age 0 is the first active age, hence the representative one.
    first_age_only_differs = AgeSpecializedGrid(
        build=lambda age: base if age == 0 else shifted,
        signature=lambda age: 0.0 if age == 0 else 1.0,
    )
    shifted_throughout = AgeSpecializedGrid(
        build=lambda _age: shifted, signature=lambda _age: 1.0
    )

    def solve(wage_grid):
        return nbegm_continuous_ride_along_toy.build_model(
            variant="nbegm", n_periods=4, n_liquid=60, wage_grid=wage_grid
        ).solve(
            params=nbegm_continuous_ride_along_toy.build_params(), log_level="debug"
        )

    from_first_age = solve(first_age_only_differs)
    throughout = solve(shifted_throughout)

    for period in sorted(from_first_age):
        if period == 0:
            continue
        for regime_name, V_arr in from_first_age[period].items():
            np.testing.assert_array_equal(
                np.asarray(V_arr),
                np.asarray(throughout[period][regime_name]),
                err_msg=(
                    f"period {period}, regime {regime_name} moved when only the "
                    "first age's co-state grid changed — the continuation is read "
                    "on the representative age's nodes instead of period + 1's"
                ),
            )


def test_nnbegm_reads_its_durable_continuation_on_the_next_period_s_grid():
    """NNBEGM's keeper rides the durable along, and reads it at `period + 1`.

    The nested solver holds the durable stock fixed and hands the inner NB-EGM a
    passive co-state, so an age-varying durable grid moves the nodes the inner
    kernel interpolates the continuation on — the same read the plain ride-along
    NB-EGM does, reached through the nesting.

    Two durable schedules agreeing from the second age onward must therefore give
    the same value function from the second period onward.
    """
    base = n_nbegm_toy.ILLIQUID_GRID
    shifted = dataclasses.replace(base, start=base.start + 2.0, stop=base.stop + 2.0)
    # Age 20 is the first active age, hence the representative one.
    first_age_only_differs = AgeSpecializedGrid(
        build=lambda age: base if age == 20 else shifted,
        signature=lambda age: 0.0 if age == 20 else 2.0,
    )
    shifted_throughout = AgeSpecializedGrid(
        build=lambda _age: shifted, signature=lambda _age: 2.0
    )

    def solve(illiquid_grid):
        return n_nbegm_toy.build_model(
            variant="n_nbegm", n_periods=4, illiquid_grid=illiquid_grid
        ).solve(params={"discount_factor": 0.95}, log_level="debug")

    from_first_age = solve(first_age_only_differs)
    throughout = solve(shifted_throughout)

    for period in sorted(from_first_age):
        if period == 0:
            continue
        for regime_name, V_arr in from_first_age[period].items():
            np.testing.assert_array_equal(
                np.asarray(V_arr),
                np.asarray(throughout[period][regime_name]),
                err_msg=(
                    f"period {period}, regime {regime_name} moved when only the "
                    "first age's durable grid changed — the nested continuation "
                    "is read on the representative age's nodes"
                ),
            )


def test_no_age_marker_reaches_a_solver_build_context(monkeypatch):
    """Solvers never see an `AgeSpecializedGrid` or `AgeSpecializedFunction`.

    Age normalization is the single boundary that turns the markers into concrete
    objects, so everything downstream — every solver's build context — carries only
    resolved grids and functions. A solver that met a raw marker would have to
    type-check for it, and the ones that do not would silently drop the state.
    """
    seen = []
    original = processing.SolverBuildContext

    def spy(**kwargs):
        context = original(**kwargs)
        seen.append(context)
        return context

    monkeypatch.setattr(processing, "SolverBuildContext", spy)
    _dcegm_twin_with_wealth_grid(
        AgeSpecializedGrid(
            build=lambda age: dataclasses.replace(
                WEALTH_GRID,
                points=tuple(
                    float(node) + 0.25 * (age - MIN_AGE)
                    for node in np.asarray(WEALTH_GRID.to_jax())
                ),
            ),
            signature=lambda age: age - MIN_AGE,
        )
    )

    assert seen, "no solver build context was constructed"
    markers = (AgeSpecializedGrid, AgeSpecializedFunction)
    offending = [
        (context.regime_name, where, name)
        for context in seen
        for where, values in (
            ("grids", dict(context.grids)),
            ("functions", dict(context.functions)),
            *(
                (f"user_regimes[{regime_name}].{slot}", dict(getattr(regime, slot)))
                for regime_name, regime in context.user_regimes.items()
                for slot in ("states", "actions", "functions", "state_transitions")
            ),
        )
        for name, value in values.items()
        if isinstance(value, markers)
    ]
    assert offending == [], f"raw age markers reached a solver: {offending}"
