"""DC-EGM regimes honour `AgeSpecializedFunction` per period.

An age-specialized function is a *different function* at each age, so a period's
kernel has to be built from that period's concrete function. Periods may share one
compiled program only when they share the user's declared signature — which is
exactly the rule the brute-force path already follows.

The oracle is the last working age. Its value depends on its own utility and on a
continuation that no age specialization touches, so it must reproduce *exactly* a
plain solve whose utility is the concrete function that age resolves to. Resolving
the specialization at the wrong age moves that value by a finite amount, which makes
equality the right instrument rather than a tolerance.

Cross-solver parity is deliberately not used here: the brute-force solver picks
consumption off a grid while DC-EGM inverts the Euler equation exactly, so the two
disagree by the grid's resolution wherever CRRA curvature is steep — a gap unrelated
to age specialization, and one no tolerance separates from the defect.
"""

import dataclasses
from collections.abc import Callable
from typing import Literal

import jax.numpy as jnp
import numpy as np
import pytest

from lcm import AgeGrid, AgeSpecializedFunction, Model
from lcm.typing import BoolND, ContinuousAction, FloatND
from tests.conftest import DECIMAL_PRECISION
from tests.test_models.dcegm_paper_twin import (
    DCEGM_SOLVER,
    MIN_AGE,
    N_PERIODS,
    TwinRegimeId,
    _retirement,
    _working_life,
    crra,
    done_from_working,
    done_retired,
    get_params,
)


def _make_scaled_utility(age: float):
    """The working-life utility, with the disutility of work drifting by age."""
    extra_work_disutility = 0.05 * (age - MIN_AGE)

    def utility_working(
        consumption: ContinuousAction, is_working: BoolND, rho: float, delta: float
    ) -> FloatND:
        return crra(consumption, rho) - jnp.where(
            is_working, delta + extra_work_disutility, 0.0
        )

    return utility_working


def _twin(
    *,
    solver_kind: Literal["brute_force", "dcegm"],
    age_specialized: bool,
    utility: Callable | None = None,
) -> Model:
    """The DC-EGM twin, optionally with an age-drifting or pinned utility."""
    working = _working_life(solver_kind)
    retirement = _retirement(solver_kind)
    if solver_kind == "dcegm":
        working = working.replace(solver=dataclasses.replace(DCEGM_SOLVER))
        retirement = retirement.replace(solver=dataclasses.replace(DCEGM_SOLVER))
    if age_specialized:
        utility = AgeSpecializedFunction(
            build=_make_scaled_utility,
            signature=lambda age: age - MIN_AGE,
        )
    if utility is not None:
        working = working.replace(
            functions={**dict(working.functions), "utility": utility}
        )
    return Model(
        regimes={
            "working_life": working,
            "retirement": retirement,
            "done_from_working": done_from_working,
            "done_retired": done_retired,
        },
        ages=AgeGrid(start=MIN_AGE, stop=MIN_AGE + N_PERIODS - 1, step="Y"),
        regime_id_class=TwinRegimeId,
    )


def test_the_last_working_age_uses_that_ages_own_utility():
    """The last working age's value equals a plain solve pinned to that age's utility.

    The last working period's value depends on exactly two things: its own utility
    and its continuation, which no age specialization touches. So the age-specialized
    solve must reproduce, *exactly*, a plain solve whose utility is the concrete
    function `build(age)` returns at that age. Resolving the specialization at some
    other age moves this value by a finite amount, so equality is the right
    instrument — no tolerance can hide the defect without also hiding real error.
    """
    params = get_params()
    specialized = _twin(solver_kind="dcegm", age_specialized=True).solve(
        params=params, log_level="debug"
    )
    last_working = max(
        period for period, regimes in specialized.items() if "working_life" in regimes
    )
    pinned = _twin(
        solver_kind="dcegm",
        age_specialized=False,
        utility=_make_scaled_utility(MIN_AGE + last_working),
    ).solve(params=params, log_level="debug")

    expected = np.asarray(pinned[last_working]["working_life"])
    got = np.asarray(specialized[last_working]["working_life"])
    np.testing.assert_array_equal(np.isneginf(got), np.isneginf(expected))
    finite = np.isfinite(expected)
    np.testing.assert_array_almost_equal(
        got[finite], expected[finite], decimal=DECIMAL_PRECISION
    )


def test_age_specialized_utility_actually_moves_the_dcegm_solution():
    """The drifting utility changes the DC-EGM value function.

    Without this, the agreement test above would still pass if *both* solvers
    ignored the age specialization in the same way.
    """
    params = get_params()
    drifting = _twin(solver_kind="dcegm", age_specialized=True).solve(
        params=params, log_level="debug"
    )
    flat = _twin(solver_kind="dcegm", age_specialized=False).solve(
        params=params, log_level="debug"
    )

    moved = [
        period
        for period in drifting
        if "working_life" in drifting[period]
        and not np.allclose(
            np.asarray(drifting[period]["working_life"]),
            np.asarray(flat[period]["working_life"]),
            equal_nan=True,
        )
    ]
    assert moved, "the age-specialized utility left every period unchanged"


def _compiled_cores(model: Model, regime_name: str) -> list[int]:
    """Identity of each period's compiled core — the object periods actually share."""
    kernels = model._regimes[regime_name].solution.period_kernels
    return [
        id(core)
        for _, kernel in sorted(kernels.items())
        for core in kernel.cores().values()
    ]


@pytest.mark.parametrize("solver_kind", ["brute_force", "dcegm"])
def test_periods_with_distinct_signatures_do_not_share_a_compiled_core(solver_kind):
    """Ages whose declared signatures differ never share one compiled program.

    Sharing is what makes an age-specialized model affordable, so it is keyed on the
    user's signature. Sharing across *distinct* signatures would silently apply one
    age's closure at every age.
    """
    cores = _compiled_cores(
        _twin(solver_kind=solver_kind, age_specialized=True), "working_life"
    )
    assert len(set(cores)) == len(cores)


@pytest.mark.parametrize("solver_kind", ["brute_force", "dcegm"])
def test_an_age_invariant_regime_still_shares_programs_across_periods(solver_kind):
    """Without age specialization, periods keep sharing compiled programs.

    The companion to the test above: keying on the signature must not degrade into
    one program per period whenever the machinery is present. The count is not one —
    the last working period continues into a different regime than the earlier ones,
    which is its own group — so the claim is that sharing happens at all, and that
    dropping the specialization strictly reduces the number of programs.
    """
    shared = _compiled_cores(
        _twin(solver_kind=solver_kind, age_specialized=False), "working_life"
    )
    specialized = _compiled_cores(
        _twin(solver_kind=solver_kind, age_specialized=True), "working_life"
    )
    assert len(set(shared)) < len(shared)
    assert len(set(shared)) < len(set(specialized))
