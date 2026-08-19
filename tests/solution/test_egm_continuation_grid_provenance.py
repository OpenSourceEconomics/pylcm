"""EGM kernels read `V_{t+1}` on the target regime's period-`t+1` grid.

A period-`t` EGM kernel interpolates its continuation off a grid. Which grid is
correct is fixed by where the continuation lives, not by where the kernel
publishes its own result: it is the **target** regime's grid, at period
**`t+1`**. Two axes make those differ from the source's own current grid:

- **cross-regime** — the target is a different regime whose state is discretized
  on a different grid (the retired regime's liquid support need not match the
  working regime's, and the terminal regime's need not match either);
- **temporal** — the state's grid is age-specialized, so period `t+1`'s nodes
  differ from period `t`'s within one regime.

The oracle throughout is the same model solved by dense `GridSearch`, compared on
the covered interior. Grid search never interpolates a continuation, so it cannot
share the defect.

Every configuration here is reachable from the public API: the DS pension model
takes per-regime grid overrides, all defaulting to one shared grid, so the
negative control is exactly today's model.

A third axis makes the target's array differ from this regime's without moving
any node: the **order** the two regimes' states resolve in, which decides what
the value array's axes mean. W10 covers it, in
`tests/solution/test_egm_continuation_axis_order.py` — a pension-first target
reaching a liquid-first kernel is a layout question rather than a grid one, and
it is asserted against an exact affine oracle rather than against brute force.
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import yaml

from lcm import (
    AgeGrid,
    AgeSpecializedGrid,
    LinSpacedGrid,
    MarkovTransition,
    Model,
    categorical,
)
from lcm.regime import ConsumptionSavingsRegime, LiquidMargin, Regime
from lcm.solvers import EGM, GridSearch
from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    FloatND,
    ScalarInt,
)
from tests.test_models.deterministic.ds_pension import get_model, get_params

_N_PERIODS = 5
_RETIREMENT_PERIOD = 3

_SAVINGS_GRID = LinSpacedGrid(start=0.0, stop=20.0, n_points=40)

# The interior the two solvers are comparable on: the off-grid top-pension
# boundary layer thickens backward, and the steep low-liquid rows are where the
# dense brute is least reliable.
_WORKING_INTERIOR = {2: np.s_[3:, :9], 1: np.s_[3:, :8], 0: np.s_[3:, :7]}
# The lowest liquid nodes are borrowing-constrained, where an exact EGM
# inversion and a discrete consumption sweep are not comparable at all.
_RETIRED_INTERIOR = np.s_[3:]
# When the two periods' grids differ, the candidate cloud inverted from the
# post-decision grids covers one pension column fewer, so the same off-grid
# top-pension layer is one column thicker than in the matched-grid case.
_MOVING_INTERIOR = {2: np.s_[3:, :7], 1: np.s_[3:, :7], 0: np.s_[3:, :6]}
# The renamed 1-D model's borrowing constraint binds over its lowest wealth
# nodes, where the exact EGM solution and a discrete consumption sweep are not
# comparable. Above them the two agree to well under a percent.
_RENAMED_UNCONSTRAINED = np.s_[5:]

# The declared sentinels, read from the contract rather than restated here: a
# bound and the gate enforcing it are then the same artifact, so neither can
# drift from the other.
_CONTRACT = yaml.safe_load(
    (Path(__file__).parent / "provenance_contract.yaml").read_text(encoding="utf-8")
)
_RETIRED_SENTINEL = _CONTRACT["workloads"]["retired_egm"]["sentinel"]


def _solve_pair(**grid_overrides):
    """Solve one grid configuration with the EGM solvers and with dense brute."""
    solvers = grid_overrides.pop("solvers")
    egm = get_model(n_periods=_N_PERIODS, solvers=solvers, **grid_overrides).solve(
        params=get_params(), log_level="debug"
    )
    brute = get_model(n_periods=_N_PERIODS, n_consumption=200, **grid_overrides).solve(
        params=get_params(), log_level="debug"
    )
    return egm, brute


def _assert_within_sentinel(rel, sentinel):
    """Both declared statistics of the regret sample are inside the sentinel.

    Both are asserted, never one. A median is insensitive to a minority of badly
    wrong nodes, and an uncovered node is a large regret that only the maximum
    sees, so the two together are what detect a mask that stops covering its
    interior. Neither is an accuracy claim; see the contract's own preamble.
    """
    assert np.median(rel) < sentinel["median_value_regret"]
    assert np.max(rel) < sentinel["max_value_regret"]


def _assert_retired_matches_brute(egm, brute, period=_RETIREMENT_PERIOD):
    """The retired value agrees with brute on the unconstrained liquid interior."""
    egm_v = np.asarray(egm[period]["retired"])[_RETIRED_INTERIOR]
    brute_v = np.asarray(brute[period]["retired"])[_RETIRED_INTERIOR]
    assert np.isfinite(egm_v).all()
    rel = np.abs(egm_v - brute_v) / np.abs(brute_v)
    _assert_within_sentinel(rel, _RETIRED_SENTINEL)


def _moving_liquid_grid(*, start, stop_at_age, n_points):
    """A liquid grid whose ceiling moves with age, shape held fixed.

    The ceiling stays inside the solver's post-decision `a_grid`, so the
    envelope covers every node and the comparison measures which grid the
    continuation is read on rather than how uncovered holes are filled.
    """
    return AgeSpecializedGrid(
        build=lambda age: LinSpacedGrid(
            start=start, stop=stop_at_age(age), n_points=n_points
        ),
        signature=lambda age: float(stop_at_age(age)),
    )


def _renamed_one_asset_model(*, solver, n_consumption=14):
    """A 1-D consumption--saving model whose state is `wealth`, not `liquid`.

    Structurally identical to the DS pension retired sub-problem, but the
    state's name differs from the EGM kernel's internal role vocabulary.
    Which name the modeller picks is their business; the kernel's private
    liquid/pension roles must be resolved from the solver's declaration, not
    matched against the user's spelling.
    """

    @categorical(ordered=False)
    class RenamedRegimeId:
        alive: ScalarInt
        gone: ScalarInt

    def utility(consumption: ContinuousAction, crra: float) -> FloatND:
        return consumption ** (1.0 - crra) / (1.0 - crra)

    def bequest(wealth: ContinuousState, crra: float) -> FloatND:
        return wealth ** (1.0 - crra) / (1.0 - crra)

    def resources(wealth: ContinuousState) -> FloatND:
        return wealth

    def savings(wealth: ContinuousState, consumption: ContinuousAction) -> FloatND:
        return wealth - consumption

    def next_wealth(
        savings: FloatND,
        return_liquid: float,
        retirement_income: float,
    ) -> ContinuousState:
        return (1.0 + return_liquid) * savings + retirement_income

    def feasible(wealth: ContinuousState, consumption: ContinuousAction) -> BoolND:
        return consumption <= wealth

    def prob_survive(age: int, last_age: float) -> FloatND:
        return jnp.where(age + 1 < last_age, 1.0, 0.0)

    def prob_gone(age: int, last_age: float) -> FloatND:
        return jnp.where(age + 1 >= last_age, 1.0, 0.0)

    wealth_grid = LinSpacedGrid(start=0.1, stop=20.0, n_points=12)
    ages = AgeGrid(start=0, stop=3, step="Y")
    alive = (ConsumptionSavingsRegime if isinstance(solver, EGM) else Regime)(
        actions={
            "consumption": LinSpacedGrid(start=0.1, stop=20.0, n_points=n_consumption)
        },
        states={"wealth": wealth_grid},
        state_transitions={"wealth": {"alive": next_wealth, "gone": next_wealth}},
        constraints={} if isinstance(solver, EGM) else {"feasible": feasible},
        transition={
            "alive": MarkovTransition(prob_survive),
            "gone": MarkovTransition(prob_gone),
        },
        functions={"utility": utility, "resources": resources, "savings": savings},
        active=lambda age: age < 3,
        solver=solver,
        **(
            {
                "liquid": LiquidMargin(
                    state="wealth",
                    action="consumption",
                    resources="resources",
                    post_decision_state="savings",
                )
            }
            if isinstance(solver, EGM)
            else {}
        ),
    )
    gone = Regime(
        transition=None,
        states={"wealth": wealth_grid},
        functions={"utility": bequest},
        active=lambda age: age >= 3,
        solver=GridSearch(),
    )
    return Model(
        regimes={"alive": alive, "gone": gone},
        ages=ages,
        regime_id_class=RenamedRegimeId,
    )


def _renamed_one_asset_params():
    """Params for the renamed 1-D model, in the user's own vocabulary."""
    law = {"return_liquid": 0.02, "retirement_income": 0.5}
    return {
        "alive": {
            "utility": {"crra": 2.0},
            "koopmans_aggregator": {"discount_factor": 0.98},
            "alive": {"next_wealth": law, "next_regime": {"last_age": 3.0}},
            "gone": {"next_wealth": law, "next_regime": {"last_age": 3.0}},
        },
        "gone": {"utility": {"crra": 2.0}},
    }


def test_w5_egm_does_not_require_the_state_to_be_named_liquid():
    """W5 (D3, one asset). The kernel's private role vocabulary stays private.

    A modeller naming the single continuous state `wealth`, with a law of motion
    must get the same answer as the dense brute. Nothing about the EGM
    kernel's internal liquid role may surface as a requirement on the state's
    name.
    """
    egm = _renamed_one_asset_model(solver=EGM(savings_grid=_SAVINGS_GRID)).solve(
        params=_renamed_one_asset_params(), log_level="debug"
    )
    brute = _renamed_one_asset_model(solver=GridSearch(), n_consumption=200).solve(
        params=_renamed_one_asset_params(), log_level="debug"
    )
    for period in (0, 1, 2):
        egm_v = np.asarray(egm[period]["alive"])[_RENAMED_UNCONSTRAINED]
        brute_v = np.asarray(brute[period]["alive"])[_RENAMED_UNCONSTRAINED]
        rel = np.abs(egm_v - brute_v) / np.abs(brute_v)
        _assert_within_sentinel(rel, _RETIRED_SENTINEL)
