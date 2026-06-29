"""BQSEGM agreement with the brute-force oracle on the Medicaid one-asset toy.

The case-piece EGM solve must reproduce the dense-grid `GridSearch` value function
where both are exact: across the asset region away from the boundary, and through
the boundary jump itself. BQSEGM additionally rejects a model whose smooth piece
hides branching.
"""

from collections.abc import Mapping

import jax.numpy as jnp
import numpy as np
import pytest

import lcm
from lcm import AgeGrid, LinSpacedGrid, MarkovTransition, Model
from lcm.case_piece import BoundaryKind, EqualityOwner
from lcm.exceptions import BQSEGMCaseError
from lcm.regime import Regime
from lcm.solvers import BQSEGM, GridSearch
from tests.test_models import bqsegm_medicaid_toy as toy

_LIQUID = np.linspace(0.1, 30.0, 120)
_INTERIOR = (_LIQUID > 2.0) & (_LIQUID < 22.0)
_CONSTRAINED = (_LIQUID > 0.3) & (_LIQUID < 3.0)


def _solve(
    variant: str, params: dict, *, n_consumption: int = 120
) -> Mapping[int, Mapping[str, object]]:
    """Solve the Medicaid toy on the shared comparison grids."""
    model = toy.build_model(
        variant=variant,
        n_liquid=120,
        liquid_max=30.0,
        n_savings=150,
        savings_max=22.0,
        n_consumption=n_consumption,
    )
    return model.solve(params=params, log_level="off")


def _last_alive_period(solution: Mapping[int, Mapping[str, object]]) -> int:
    return max(period for period in solution if "alive" in solution[period])


def test_bqsegm_matches_brute_through_a_recurring_jump_every_age():
    """The case-piece solve equals brute at every working age, jump and all.

    The Medicaid jump recurs in every period's continuation, so each period both
    carries a within-period jump and reads a jumped continuation. BQSEGM resolves
    the within-period jump and the boundary-targeting corner (save exactly to the
    limit for the higher eligible continuation), so agreement holds across the
    whole asset interior, every period.
    """
    params = toy.build_params()
    bqsegm = _solve("bqsegm", params)
    brute = _solve("brute", params, n_consumption=1500)
    for period in brute:
        if "alive" not in brute[period] or "alive" not in bqsegm[period]:
            continue
        np.testing.assert_allclose(
            np.asarray(bqsegm[period]["alive"])[_INTERIOR],
            np.asarray(brute[period]["alive"])[_INTERIOR],
            atol=2e-2,
            rtol=5e-3,
            err_msg=f"period={period}",
        )


def test_bqsegm_matches_brute_in_the_constrained_low_asset_region():
    """Where the borrowing constraint binds, the case-piece value tracks brute.

    At low liquid wealth the agent consumes all cash-on-hand and saves nothing.
    The zero-savings corner is an envelope candidate over the whole grid, so the
    merged value matches the dense oracle in the constrained region too — the
    region the interior agreement slice deliberately excludes.
    """
    params = toy.build_params()
    bqsegm = _solve("bqsegm", params)
    brute = _solve("brute", params, n_consumption=1500)
    for period in brute:
        if "alive" not in brute[period] or "alive" not in bqsegm[period]:
            continue
        np.testing.assert_allclose(
            np.asarray(bqsegm[period]["alive"])[_CONSTRAINED],
            np.asarray(brute[period]["alive"])[_CONSTRAINED],
            atol=2e-2,
            rtol=5e-3,
            err_msg=f"period={period}",
        )


def test_bqsegm_matches_brute_multiperiod_without_a_value_jump():
    """With equal subsidies (no jump) the case-piece solve equals brute every age.

    This isolates the multi-period EGM propagation through the masking-and-envelope
    merge from the value-jump continuation: the two cases coincide, so the merged
    solution must track brute across the whole horizon.
    """
    params = toy.build_params(subsidy_high=0.5, subsidy_low=0.5)
    bqsegm = _solve("bqsegm", params)
    brute = _solve("brute", params, n_consumption=1500)
    for period in brute:
        if "alive" not in brute[period] or "alive" not in bqsegm[period]:
            continue
        np.testing.assert_allclose(
            np.asarray(bqsegm[period]["alive"])[_INTERIOR],
            np.asarray(brute[period]["alive"])[_INTERIOR],
            atol=2e-2,
            rtol=5e-3,
            err_msg=f"period={period}",
        )


def test_bqsegm_reproduces_the_medicaid_value_drop_at_the_boundary():
    """Value drops as assets cross the Medicaid limit, matching the brute oracle.

    Just below the limit the agent receives the larger subsidy, so the value is
    higher than just above; the otherwise side owns the exact boundary.
    """
    params = toy.build_params()
    bqsegm = _solve("bqsegm", params)
    brute = _solve("brute", params, n_consumption=1500)
    period = _last_alive_period(brute)
    below = np.argmin(np.abs(_LIQUID - 7.5))
    above = np.argmin(np.abs(_LIQUID - 8.5))
    bqsegm_alive = np.asarray(bqsegm[period]["alive"])
    brute_alive = np.asarray(brute[period]["alive"])
    bqsegm_drop = float(bqsegm_alive[below] - bqsegm_alive[above])
    brute_drop = float(brute_alive[below] - brute_alive[above])
    assert bqsegm_drop > 0.0
    np.testing.assert_allclose(bqsegm_drop, brute_drop, atol=2e-2)


def test_bqsegm_rejects_a_piece_with_a_hidden_where():
    """A smooth piece hiding `jnp.where` fails the smoothness gate at model build."""

    @lcm.case_boundary(
        lcm.boundary("liquid", "limit", equality="otherwise", kind="jump")
    )
    def predicate(liquid, limit):
        return liquid < limit

    @lcm.piece("subsidy", when=predicate)
    def subsidy_when(subsidy_high):
        return jnp.where(subsidy_high > 0.0, subsidy_high, 0.0)

    @lcm.piece("subsidy", otherwise=predicate)
    def subsidy_otherwise(subsidy_low):
        return jnp.asarray(subsidy_low)

    grid = LinSpacedGrid(start=0.1, stop=20.0, n_points=40)
    alive = Regime(
        actions={"consumption": LinSpacedGrid(start=0.1, stop=20.0, n_points=40)},
        states={"liquid": grid},
        state_transitions={
            "liquid": {"alive": toy.next_liquid, "dead": toy.next_liquid}
        },
        constraints={"feasible": toy.feasible},
        transition={
            "alive": MarkovTransition(toy.prob_stay_alive),
            "dead": MarkovTransition(toy.prob_die),
        },
        functions={
            "utility": toy.utility,
            "predicate": predicate,
            "subsidy_when": subsidy_when,
            "subsidy_otherwise": subsidy_otherwise,
            "coh": toy.coh,
        },
        active=lambda age: age < 1.0,
        solver=BQSEGM(savings_grid=LinSpacedGrid(start=0.0, stop=20.0, n_points=40)),
    )
    dead = Regime(
        transition=None,
        states={"liquid": grid},
        functions={"utility": toy.bequest},
        active=lambda age: age >= 1.0,
        solver=GridSearch(),
    )
    with pytest.raises(BQSEGMCaseError, match="smoothness gate"):
        Model(
            regimes={"alive": alive, "dead": dead},
            ages=AgeGrid(start=0, stop=1, step="Y"),
            regime_id_class=toy.RegimeId,
        )


def _build_bqsegm_with_boundary(
    *, equality: EqualityOwner, kind: BoundaryKind, variable: str = "liquid"
) -> Model:
    """Assemble a one-period BQSEGM toy whose boundary carries given metadata."""

    @lcm.case_boundary(lcm.boundary(variable, "limit", equality=equality, kind=kind))
    def predicate(liquid, limit):
        return liquid < limit

    @lcm.piece("subsidy", when=predicate)
    def subsidy_when(subsidy_high):
        return jnp.asarray(subsidy_high)

    @lcm.piece("subsidy", otherwise=predicate)
    def subsidy_otherwise(subsidy_low):
        return jnp.asarray(subsidy_low)

    grid = LinSpacedGrid(start=0.1, stop=20.0, n_points=40)
    alive = Regime(
        actions={"consumption": LinSpacedGrid(start=0.1, stop=20.0, n_points=40)},
        states={"liquid": grid},
        state_transitions={
            "liquid": {"alive": toy.next_liquid, "dead": toy.next_liquid}
        },
        constraints={"feasible": toy.feasible},
        transition={
            "alive": MarkovTransition(toy.prob_stay_alive),
            "dead": MarkovTransition(toy.prob_die),
        },
        functions={
            "utility": toy.utility,
            "predicate": predicate,
            "subsidy_when": subsidy_when,
            "subsidy_otherwise": subsidy_otherwise,
            "coh": toy.coh,
        },
        active=lambda age: age < 1.0,
        solver=BQSEGM(savings_grid=LinSpacedGrid(start=0.0, stop=20.0, n_points=40)),
    )
    dead = Regime(
        transition=None,
        states={"liquid": grid},
        functions={"utility": toy.bequest},
        active=lambda age: age >= 1.0,
        solver=GridSearch(),
    )
    return Model(
        regimes={"alive": alive, "dead": dead},
        ages=AgeGrid(start=0, stop=1, step="Y"),
        regime_id_class=toy.RegimeId,
    )


def test_bqsegm_rejects_a_when_owned_boundary():
    """v1 supports only `equality='otherwise'`; a `when`-owned boundary is rejected."""
    with pytest.raises(BQSEGMCaseError, match="v1"):
        _build_bqsegm_with_boundary(equality="when", kind="jump")


def test_bqsegm_rejects_a_non_jump_boundary_kind():
    """v1 supports only `kind='jump'`; a continuous-kink boundary is rejected."""
    with pytest.raises(BQSEGMCaseError, match="v1"):
        _build_bqsegm_with_boundary(equality="otherwise", kind="continuous_kink")


def test_bqsegm_accepts_the_supported_otherwise_jump_boundary():
    """The supported `otherwise`/`jump` liquid boundary builds without error."""
    _build_bqsegm_with_boundary(equality="otherwise", kind="jump")


def test_bqsegm_rejects_a_state_dependent_subsidy_piece():
    """A subsidy piece reading the liquid state is rejected — v1 pieces are pure.

    The one-asset core evaluates each piece from the flat params alone, so a piece
    that depends on a state or action cannot be the additive cash-on-hand shift v1
    routes. It is rejected at build rather than failing obscurely at solve.
    """

    @lcm.case_boundary(
        lcm.boundary("liquid", "limit", equality="otherwise", kind="jump")
    )
    def predicate(liquid, limit):
        return liquid < limit

    @lcm.piece("subsidy", when=predicate)
    def subsidy_when(liquid, subsidy_high):
        return subsidy_high * jnp.ones_like(liquid)

    @lcm.piece("subsidy", otherwise=predicate)
    def subsidy_otherwise(subsidy_low):
        return jnp.asarray(subsidy_low)

    grid = LinSpacedGrid(start=0.1, stop=20.0, n_points=40)
    alive = Regime(
        actions={"consumption": LinSpacedGrid(start=0.1, stop=20.0, n_points=40)},
        states={"liquid": grid},
        state_transitions={
            "liquid": {"alive": toy.next_liquid, "dead": toy.next_liquid}
        },
        constraints={"feasible": toy.feasible},
        transition={
            "alive": MarkovTransition(toy.prob_stay_alive),
            "dead": MarkovTransition(toy.prob_die),
        },
        functions={
            "utility": toy.utility,
            "predicate": predicate,
            "subsidy_when": subsidy_when,
            "subsidy_otherwise": subsidy_otherwise,
            "coh": toy.coh,
        },
        active=lambda age: age < 1.0,
        solver=BQSEGM(savings_grid=LinSpacedGrid(start=0.0, stop=20.0, n_points=40)),
    )
    dead = Regime(
        transition=None,
        states={"liquid": grid},
        functions={"utility": toy.bequest},
        active=lambda age: age >= 1.0,
        solver=GridSearch(),
    )
    with pytest.raises(BQSEGMCaseError, match="v1"):
        Model(
            regimes={"alive": alive, "dead": dead},
            ages=AgeGrid(start=0, stop=1, step="Y"),
            regime_id_class=toy.RegimeId,
        )


def test_bqsegm_rejects_a_piece_whose_helper_hides_a_where():
    """A piece with a clean AST but a `jnp.where` in a called helper is rejected.

    The AST gate sees only an innocuous helper call, so a piece can smuggle a
    discontinuity past it. The JAXPR gate traces through the call and rejects the
    hidden `select_n`, failing the model build.
    """

    def hidden_subsidy_helper(subsidy_high):
        return jnp.where(subsidy_high > 0.0, subsidy_high, 0.0)

    @lcm.case_boundary(
        lcm.boundary("liquid", "limit", equality="otherwise", kind="jump")
    )
    def predicate(liquid, limit):
        return liquid < limit

    @lcm.piece("subsidy", when=predicate)
    def subsidy_when(subsidy_high):
        return hidden_subsidy_helper(subsidy_high)

    @lcm.piece("subsidy", otherwise=predicate)
    def subsidy_otherwise():
        return jnp.asarray(0.0)

    grid = LinSpacedGrid(start=0.1, stop=20.0, n_points=40)
    alive = Regime(
        actions={"consumption": LinSpacedGrid(start=0.1, stop=20.0, n_points=40)},
        states={"liquid": grid},
        state_transitions={
            "liquid": {"alive": toy.next_liquid, "dead": toy.next_liquid}
        },
        constraints={"feasible": toy.feasible},
        transition={
            "alive": MarkovTransition(toy.prob_stay_alive),
            "dead": MarkovTransition(toy.prob_die),
        },
        functions={
            "utility": toy.utility,
            "predicate": predicate,
            "subsidy_when": subsidy_when,
            "subsidy_otherwise": subsidy_otherwise,
            "coh": toy.coh,
        },
        active=lambda age: age < 1.0,
        solver=BQSEGM(savings_grid=LinSpacedGrid(start=0.0, stop=20.0, n_points=40)),
    )
    dead = Regime(
        transition=None,
        states={"liquid": grid},
        functions={"utility": toy.bequest},
        active=lambda age: age >= 1.0,
        solver=GridSearch(),
    )
    with pytest.raises(BQSEGMCaseError, match="smoothness gate"):
        Model(
            regimes={"alive": alive, "dead": dead},
            ages=AgeGrid(start=0, stop=1, step="Y"),
            regime_id_class=toy.RegimeId,
        )
