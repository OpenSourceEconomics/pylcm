"""A fixed width pin narrowed by a width ceiling simulates like the narrow pin."""

from typing import Any

import jax.numpy as jnp
import pandas as pd
import pytest

from _lcm.simulation import runtime
from lcm import AgeGrid, DiscreteGrid, ExecutionConfig, LinSpacedGrid, Model
from lcm.solvers import GridSearch
from tests.test_models.deterministic.regression import (
    START_AGE,
    LaborSupply,
    RegimeId,
    dead,
    get_params,
    working_life,
)


def _toy_model(*, execution: ExecutionConfig) -> Model:
    final_age_alive = START_AGE + 1
    return Model(
        regimes={
            "working_life": working_life.replace(
                active=lambda age: age <= final_age_alive,
                states={"wealth": LinSpacedGrid(start=1, stop=3, n_points=8)},
                actions={
                    "labor_supply": DiscreteGrid(category_class=LaborSupply),
                    "consumption": LinSpacedGrid(start=1, stop=3, n_points=3),
                },
                solver=GridSearch(),
            ),
            "dead": dead,
        },
        ages=AgeGrid(start=START_AGE, stop=final_age_alive + 1, step="Y"),
        regime_id_class=RegimeId,
        execution_config=execution,
    )


def _run(
    *, monkeypatch: pytest.MonkeyPatch, execution: ExecutionConfig
) -> tuple[pd.DataFrame, set[int]]:
    widths: list[dict[str, int]] = []
    original = runtime.plan_workspace

    def record(**kwargs: Any) -> Any:
        plan = original(**kwargs)
        widths.append(dict(plan.widths))
        return plan

    with monkeypatch.context() as context:
        context.setattr(runtime, "plan_workspace", record)
        model = _toy_model(execution=execution)
        params = get_params(n_periods=3)
        solution = model.solve(params=params, log_level="off")
        panel = model.simulate(
            params=params,
            initial_conditions={
                "wealth": jnp.linspace(1.0, 3.0, 8),
                "age": jnp.full(8, float(START_AGE)),
                "regime_id": jnp.full(8, RegimeId.working_life, dtype=jnp.int32),
            },
            solution=solution,
            seed=7,
            log_level="off",
        ).to_dataframe()
    return panel, {w["subject"] for w in widths if "subject" in w}


@pytest.fixture(params=[None, 2**30], ids=["unbudgeted", "budgeted"])
def arms(
    *, request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> tuple[tuple[pd.DataFrame, set[int]], tuple[pd.DataFrame, set[int]]]:
    budget = request.param
    control = _run(
        monkeypatch=monkeypatch,
        execution=ExecutionConfig(
            axis_widths={"subject": 2}, device_memory_bytes=budget
        ),
    )
    ceiled = _run(
        monkeypatch=monkeypatch,
        execution=ExecutionConfig(
            axis_widths={"subject": 8},
            axis_width_ceilings={"subject": 2},
            device_memory_bytes=budget,
        ),
    )
    return control, ceiled


def test_simulate_ceiled_pin_plans_the_ceiling_width(arms: Any) -> None:
    """A pin of 8 under a ceiling of 2 plans subject width 2, as a pin of 2 does."""
    (_, control_widths), (_, ceiled_widths) = arms
    assert (ceiled_widths, control_widths) == ({2}, {2})


def test_simulate_ceiled_pin_matches_explicit_narrow_pin_panel(arms: Any) -> None:
    """The simulated panel is identical to the one from an explicit pin of 2."""
    (control, _), (ceiled, _) = arms
    pd.testing.assert_frame_equal(ceiled, control, check_exact=True)
