"""NB-EGM save-to-cliff targets read the child period's own grids and functions.

The source decides at age 0; its child is next period at age 1. Two public
declarations make the child's cliff geometry differ from the source's:

- an age-specialized `wage` grid whose nodes at the child's age differ from the
  source's, with the same number of points, so the child's blended wage rows
  are not the rows the source grid would bracket;
- an age-specialized `gross_income` function closing over its age, so the
  child's cliff sits where the child's own income crosses the threshold.

Both the production cliff targets and the scalar oracle's must bracket the
savings preimages of the child's cliffs. The full period kernel must also agree
with the scalar oracle.
"""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import lcm
from _lcm.execution.core_program import (
    CoreBuildContext,
    core_program_graph,
    materialize_core_program,
)
from _lcm.solution.nbegm import _cliff_savings_targets as production_targets
from lcm import AgeSpecializedFunction, AgeSpecializedGrid, DiscreteGrid, LinSpacedGrid
from lcm.typing import ContinuousState, FloatND
from tests.solution._nbegm_direct_oracle import (
    _cliff_savings_targets as oracle_targets,
)
from tests.solution._nbegm_direct_oracle import (
    child_period_kwargs,
    ride_along_kernel,
)
from tests.solution.test_nbegm_direct_oracle import (
    _assert_kernel_agrees_with_oracle as assert_kernel_agrees_with_oracle,
)
from tests.test_models import nbegm_continuous_ride_along_toy
from tests.test_models.nbegm_common import (
    make_alive_dead_model,
    resolve_solver,
    savings,
    utility,
)
from tests.test_models.nbegm_indexed_threshold_toy import (
    ConsumerKind,
    resources,
    subsidy,
)


def identity_liquid(*, savings: FloatND) -> ContinuousState:
    """Liquid wealth next period equals savings."""
    return savings


def zero_bequest(*, liquid: ContinuousState) -> FloatND:
    """The terminal regime values remaining wealth at zero."""
    return jnp.zeros_like(liquid)


def _wage_grid_model(*, child_top: float) -> tuple[Any, dict[str, Any]]:
    """Wage nodes `(0, 2, 4)` at age 0 and `(0, top / 2, top)` from age 1 on."""

    def build_grid(age: float) -> LinSpacedGrid:
        return LinSpacedGrid(
            start=0.0, stop=4.0 if age < 1.0 else child_top, n_points=3
        )

    model = nbegm_continuous_ride_along_toy.build_model(
        variant="nbegm",
        n_periods=3,
        n_liquid=17,
        n_consumption=17,
        n_savings=8,
        wage_grid=AgeSpecializedGrid(
            build=build_grid, signature=lambda age: 4.0 if age < 1.0 else child_top
        ),
    )
    params = nbegm_continuous_ride_along_toy.build_params(
        crra=1.0,
        return_liquid=0.0,
        income=1.0,
        subsidy_high=0.5,
        final_age_alive=2.0,
    )
    return model, params


def _income_closure_model(*, increment: float) -> tuple[Any, dict[str, Any]]:
    """Income `liquid + base_income + increment * age`, with age closed over."""

    def build_income(age: float) -> Any:
        def gross_income(*, liquid: ContinuousState, base_income: float) -> FloatND:
            return liquid + base_income + increment * age

        return gross_income

    model = make_alive_dead_model(
        n_periods=3,
        n_liquid=17,
        liquid_max=30.0,
        n_consumption=17,
        liquid_grid=LinSpacedGrid(start=0.0, stop=30.0, n_points=17),
        alive_functions={
            "utility": utility,
            "gross_income": AgeSpecializedFunction(
                build=build_income, signature=lambda age: increment * age
            ),
            "subsidy": subsidy,
            "resources": resources,
            "savings": savings,
        },
        liquid_law=identity_liquid,
        alive_solver=resolve_solver(
            variant="nbegm",
            savings_grid=LinSpacedGrid(start=0.0, stop=28.0, n_points=8),
        ),
        constraints={},
        extra_states={"kind": DiscreteGrid(category_class=ConsumerKind)},
        extra_state_transitions={"kind": {"alive": lcm.fixed_transition("kind")}},
        dead_functions={"utility": zero_bequest},
    )
    params = {
        "alive": {
            "utility": {"crra": 1.0},
            "koopmans_aggregator": {"discount_factor": 0.95},
            "gross_income": {"base_income": 2.0},
            "subsidy": {
                "subsidy_low": 0.0,
                "subsidy_high": 5.0,
                "fpl_cliff": jnp.asarray([11.0, 11.0]),
            },
            "final_age_alive": 2.0,
        }
    }
    return model, params


# Each case: the model kind, its parameter, and the savings preimages of the
# child's cliffs.
# - Wage grids: source wage 4 moves to 0.9 * 4 = 3.6; the child blends the two
#   child nodes bracketing 3.6; each node's cliff is `liquid = 15 - wage` and
#   `liquid' = s + 1`, so each centre is `14 - wage_node`.
# - Income closures: income `liquid + 2 + increment` at child age 1 crosses 11
#   at `liquid = 9 - increment`, and `liquid' = s`.
_CASES = {
    "grid-unchanged": ("grid", 4.0, (10.0, 12.0)),
    "grid-top-6": ("grid", 6.0, (8.0, 11.0)),
    "grid-top-8": ("grid", 8.0, (10.0, 14.0)),
    "income-age-invariant": ("income", 0.0, (9.0,)),
    "income-rising-with-age": ("income", 3.0, (6.0,)),
}


@pytest.fixture(scope="module", params=tuple(_CASES), ids=tuple(_CASES))
def seam(request: pytest.FixtureRequest) -> dict[str, Any]:
    kind, parameter, centres = _CASES[request.param]
    model, params = (
        _wage_grid_model(child_top=parameter)
        if kind == "grid"
        else _income_closure_model(increment=parameter)
    )
    kernel, context = ride_along_kernel(model=model, params=params, period=0)
    replay = core_program_graph(kernel=kernel)["replay"]
    materialized = materialize_core_program(
        program=replay, context=CoreBuildContext(**context)
    )
    kwargs = dict(materialized.arguments)
    kwargs.update(getattr(replay.function, "keywords", None) or {})
    dtype = jnp.asarray(kwargs[kernel.statics.liquid_name]).dtype
    cell = (
        {"wage": jnp.asarray(4.0, dtype=dtype)}
        if kind == "grid"
        else {"kind": jnp.asarray(0, dtype=jnp.int32)}
    )
    return {
        "kernel": kernel,
        "context": context,
        "kwargs": kwargs,
        "cell": cell,
        "dtype": dtype,
        "centres": centres,
    }


def _combo_pool(*, seam: dict[str, Any]) -> dict[str, Any]:
    statics = seam["kernel"].statics
    params = {
        key: value
        for key, value in seam["kwargs"].items()
        if key not in statics.state_names and key != "next_regime_to_continuation"
    }
    return {**params, **seam["cell"]}


def _live_rows(raw: Any) -> np.ndarray:
    rows = np.asarray(raw).reshape(-1, 2)
    return rows[np.isfinite(rows).all(axis=1)]


def _brackets_exactly(
    *, rows: np.ndarray, centres: tuple[float, ...], dtype: Any
) -> bool:
    """Whether every centre is tightly bracketed and every pair brackets a centre.

    The centres are separated interior cliffs, so the width bound measures only
    the one-sided displacement of the targets, a few dozen units of roundoff.
    """
    eps = float(np.finfo(np.dtype(dtype)).eps)
    for centre in centres:
        widths = [upper - lower for lower, upper in rows if lower < centre < upper]
        if not widths or min(widths) > 64 * eps * max(1.0, abs(centre)):
            return False
    return all(
        any(lower < centre < upper for centre in centres) for lower, upper in rows
    )


@pytest.mark.parametrize("jit", [False, True], ids=["eager", "jit"])
def test_production_cliff_targets_bracket_the_child_periods_cliffs(
    *, seam: dict[str, Any], jit: bool
) -> None:
    """The kernel's save-to-cliff targets straddle exactly the child's centres."""
    kernel = seam["kernel"]

    def evaluate() -> FloatND:
        return production_targets(
            continuation_plan=kernel.continuation_plan,
            regime_name="alive",
            statics=kernel.statics,
            child_carry=seam["context"]["next_regime_to_continuation"]["alive"],
            combo_pool=_combo_pool(seam=seam),
            savings_grid=jnp.asarray(kernel.savings_grid),
            dtype=seam["dtype"],
        )

    rows = _live_rows(jax.jit(evaluate)() if jit else evaluate())
    assert _brackets_exactly(rows=rows, centres=seam["centres"], dtype=seam["dtype"]), (
        rows,
        seam["centres"],
    )


def test_oracle_cliff_targets_bracket_the_child_periods_cliffs(
    *, seam: dict[str, Any]
) -> None:
    """The scalar oracle's cliff targets straddle exactly the child's centres."""
    kernel, kwargs = seam["kernel"], seam["kwargs"]
    raw = oracle_targets(
        plan=kernel.continuation_plan,
        regime_name="alive",
        combo_pool=_combo_pool(seam=seam),
        statics=kernel.statics,
        child_kwargs=child_period_kwargs(kwargs=kwargs, context=seam["context"]),
        liquid_grid=np.asarray(kwargs[kernel.statics.liquid_name], dtype=np.float64),
        savings_grid=np.asarray(kernel.savings_grid, dtype=np.float64),
        dtype=seam["dtype"],
    )
    rows = _live_rows(raw)
    assert _brackets_exactly(rows=rows, centres=seam["centres"], dtype=seam["dtype"]), (
        rows,
        seam["centres"],
    )


def test_period_kernel_agrees_with_the_oracle(*, seam: dict[str, Any]) -> None:
    """Value, carry and consumption match the oracle reading the child period."""
    assert_kernel_agrees_with_oracle(kernel=seam["kernel"], context=seam["context"])
