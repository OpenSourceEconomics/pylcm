"""A budgeted simulation routes through a gate into a fallback with unsorted states.

The fallback regime declares its states as `hc` before `assets`, which is not
their sorted order, and the source regime offers a current-period exit in the
period its gate routes. The budgeted simulation profiles the gate route against
the fallback's declared states after the exit has run, so it must run and
reproduce the unbudgeted panel.
"""

import jax.numpy as jnp
import pandas as pd

from lcm import (
    AgeGrid,
    ByAge,
    ExecutionConfig,
    Gate,
    LinSpacedGrid,
    MemberRoute,
    Model,
    ProjectedRegimeValue,
    Regime,
    StochasticTransition,
    Transition,
    categorical,
    fixed_transition,
)
from lcm.exits import Exit, ExitTarget, Unilateral
from lcm.typing import BoolND, ContinuousState, FloatND, ScalarInt


@categorical(ordered=False)
class RegimeId:
    main: ScalarInt
    end: ScalarInt
    alone: ScalarInt
    quit: ScalarInt


_GRID = LinSpacedGrid(start=1.0, stop=2.0, n_points=3)


def _utility(*, hc: ContinuousState, assets: ContinuousState) -> FloatND:
    return hc + assets


def _quit_utility() -> FloatND:
    return jnp.asarray(-100.0)


def _probability() -> FloatND:
    return jnp.asarray(1.0)


def _gate(*, assets: ContinuousState) -> BoolND:
    return assets > 1.25


def _project_hc(*, hc: ContinuousState) -> ContinuousState:
    return hc


def _project_assets(*, assets: ContinuousState) -> ContinuousState:
    return 0.5 * assets + 0.5


def _regime(*, terminal: bool) -> Regime:
    return Regime(
        states={"hc": _GRID, "assets": _GRID},
        state_transitions=(
            {}
            if terminal
            else {"hc": fixed_transition("hc"), "assets": fixed_transition("assets")}
        ),
        functions={"utility": _utility},
        exit=(
            None
            if terminal
            else Exit(
                targets={"self": ExitTarget(regime="quit", projection={})},
                rule=Unilateral(),
            )
        ),
    )


def _model(*, budgeted: bool) -> Model:
    return Model(
        ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
        regimes={
            "main": _regime(terminal=False),
            "end": _regime(terminal=True),
            "alone": _regime(terminal=True),
            "quit": Regime(functions={"utility": _quit_utility}),
        },
        regime_id_class=RegimeId,
        initial_nodes={0: "main"},
        enable_jit=True,
        execution_config=(
            ExecutionConfig(
                devices=(0,), axis_widths={"subject": 2}, device_memory_bytes=2**30
            )
            if budgeted
            else ExecutionConfig()
        ),
        edges={
            "main": Transition(
                targets={"end": 0, "alone": 0},
                law=ByAge(cases={0: {"end": StochasticTransition(func=_probability)}}),
                gates={
                    "end": Gate(
                        predicate=_gate,
                        routes={
                            "only": MemberRoute(
                                target_member=None,
                                fallback=ProjectedRegimeValue(
                                    regime="alone",
                                    projection={
                                        "hc": _project_hc,
                                        "assets": _project_assets,
                                    },
                                ),
                            )
                        },
                    )
                },
            ),
        },
    )


def _panel(*, budgeted: bool) -> pd.DataFrame:
    model = _model(budgeted=budgeted)
    params = {"discount_factor": 0.5}
    return model.simulate(
        params=params,
        solution=model.solve(params=params, log_level="off"),
        initial_conditions={
            "hc": jnp.asarray([1.0, 1.5, 2.0, 1.5]),
            "assets": jnp.asarray([1.0, 1.0, 2.0, 1.5]),
            "age": jnp.zeros(4),
            "regime_id": jnp.full(4, model.regime_names_to_ids["main"]),
        },
        seed=7,
        log_level="off",
    ).to_dataframe()


def test_budgeted_gate_route_into_unsorted_fallback_matches_eager_panel() -> None:
    """The budgeted panel equals the unbudgeted one, rows turned away included."""
    eager = _panel(budgeted=False)
    assert set(eager.loc[eager["period"] == 1, "regime_name"]) == {"end", "alone"}
    pd.testing.assert_frame_equal(_panel(budgeted=True), eager)
