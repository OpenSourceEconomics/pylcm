"""EGM kernels do not require the continuous state to carry their role name.

The 1-D EGM kernel calls its continuous state `liquid` internally. W5 solves a
one-asset model whose state is named `wealth` and compares it with the same model
solved by dense `GridSearch` on the unconstrained interior. Grid search never
interpolates a continuation, so it cannot share a defect in how EGM reads one.

The order in which two regimes' states resolve is covered by W10, in
`tests/solution/test_egm_continuation_axis_order.py`.
"""

from functools import partial
from pathlib import Path

import numpy as np
import yaml

from lcm import (
    AgeGrid,
    LinSpacedGrid,
    Model,
    categorical,
)
from lcm.consumption_savings_regime import ConsumptionSavingsRegime, LiquidMargin
from lcm.regime import Regime
from lcm.solvers import EGM, GridSearch
from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    FloatND,
    ScalarInt,
)

_N_PERIODS = 5

_SAVINGS_GRID = LinSpacedGrid(start=0.0, stop=20.0, n_points=40)

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
_ONE_ASSET_SENTINEL = _CONTRACT["workloads"]["one_asset_egm"]["sentinel"]


def _assert_within_sentinel(*, rel, sentinel):
    """Both declared statistics of the regret sample are inside the sentinel.

    Both are asserted, never one. A median is insensitive to a minority of badly
    wrong nodes, and an uncovered node is a large regret that only the maximum
    sees, so the two together are what detect a mask that stops covering its
    interior. Neither is an accuracy claim; see the contract's own preamble.
    """
    assert np.median(rel) < sentinel["median_value_regret"]
    assert np.max(rel) < sentinel["max_value_regret"]


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

    def utility(*, consumption: ContinuousAction, crra: float) -> FloatND:
        return consumption ** (1.0 - crra) / (1.0 - crra)

    def bequest(*, wealth: ContinuousState, crra: float) -> FloatND:
        return wealth ** (1.0 - crra) / (1.0 - crra)

    def resources(wealth: ContinuousState) -> FloatND:
        return wealth

    def savings(*, wealth: ContinuousState, consumption: ContinuousAction) -> FloatND:
        return wealth - consumption

    def next_wealth(
        *,
        savings: FloatND,
        return_liquid: float,
        retirement_income: float,
    ) -> ContinuousState:
        return (1.0 + return_liquid) * savings + retirement_income

    def feasible(*, wealth: ContinuousState, consumption: ContinuousAction) -> BoolND:
        return consumption <= wealth

    wealth_grid = LinSpacedGrid(start=0.1, stop=20.0, n_points=12)
    ages = AgeGrid(start=0, inclusive_stop=3, step="Y")
    alive = (
        partial(
            ConsumptionSavingsRegime,
            liquid=LiquidMargin(
                state="wealth",
                action="consumption",
                resources="resources",
                post_decision_state="savings",
            ),
        )
        if isinstance(solver, EGM)
        else Regime
    )(
        actions={
            "consumption": LinSpacedGrid(start=0.1, stop=20.0, n_points=n_consumption)
        },
        states={"wealth": wealth_grid},
        state_transitions={"wealth": {"alive": next_wealth, "gone": next_wealth}},
        constraints={} if isinstance(solver, EGM) else {"feasible": feasible},
        functions={
            "utility": utility,
            "savings": savings,
            **({"resources": resources} if isinstance(solver, EGM) else {}),
        },
        solver=solver,
    )
    gone = Regime(
        states={"wealth": wealth_grid},
        functions={"utility": bequest},
        solver=GridSearch(),
    )
    return Model(
        regimes={"alive": alive, "gone": gone},
        ages=ages,
        edges={"alive": {"alive": (0, 1), "gone": 2}},
        regime_id_class=RenamedRegimeId,
        initial_nodes={ages.exact_values[0]: "alive"},
    )


def _renamed_one_asset_params():
    """Params for the renamed 1-D model, in the user's own vocabulary."""
    law = {"return_liquid": 0.02, "retirement_income": 0.5}
    return {
        "alive": {
            "utility": {"crra": 2.0},
            "koopmans_aggregator": {"discount_factor": 0.98},
            "alive": {"next_wealth": law},
            "gone": {"next_wealth": law},
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
    egm = (
        _renamed_one_asset_model(solver=EGM(savings_grid=_SAVINGS_GRID))
        .solve(params=_renamed_one_asset_params(), log_level="debug")
        .values
    )
    brute = (
        _renamed_one_asset_model(solver=GridSearch(), n_consumption=200)
        .solve(params=_renamed_one_asset_params(), log_level="debug")
        .values
    )
    for period in (0, 1, 2):
        egm_v = np.asarray(egm[period]["alive"])[_RENAMED_UNCONSTRAINED]
        brute_v = np.asarray(brute[period]["alive"])[_RENAMED_UNCONSTRAINED]
        rel = np.abs(egm_v - brute_v) / np.abs(brute_v)
        _assert_within_sentinel(rel=rel, sentinel=_ONE_ASSET_SENTINEL)
