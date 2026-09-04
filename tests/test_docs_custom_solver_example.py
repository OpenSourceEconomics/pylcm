"""The solver written out in the custom-solver reference builds and solves.

`docs/reference/custom_solvers.md` presents a minimal out-of-tree solver as the
shape every custom solver starts from. That page earns the claim only while the
example still imports from the public surface, still builds period kernels the
engine accepts, and still publishes the value it says it publishes, so the fence
is executed here instead of read.
"""

import re
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

from lcm import AgeGrid, LinSpacedGrid, Model, Regime, categorical
from lcm.typing import Float1D, ScalarFloat, ScalarInt

_PAGE = Path(__file__).parents[1] / "docs" / "reference" / "custom_solvers.md"
_FENCE = re.compile(r"```python\n(.*?)```", re.DOTALL)

_N_PERIODS = 3
_WEALTH = LinSpacedGrid(start=1.0, stop=5.0, n_points=5)


@categorical(ordered=False)
class RegimeId:
    """The two regimes the documented solver is exercised over."""

    alive: ScalarInt
    dead: ScalarInt


def _documented_solver_source() -> str:
    """Return the single Python fence on the custom-solver page."""
    fences = _FENCE.findall(_PAGE.read_text(encoding="utf-8"))
    assert len(fences) == 1, (
        f"expected exactly one Python fence in {_PAGE.name}, found {len(fences)}; "
        "the extraction below would otherwise run whichever block came first"
    )
    return fences[0]


@pytest.fixture(scope="module")
def documented() -> dict[str, Any]:
    """Execute the documented example and return the names it defines."""
    namespace: dict[str, Any] = {}
    exec(compile(_documented_solver_source(), str(_PAGE), "exec"), namespace)  # noqa: S102
    return namespace


def _utility(*, wealth: Float1D) -> Float1D:
    return wealth


def _next_wealth(*, wealth: Float1D) -> Float1D:
    return wealth


def _next_regime(age: ScalarFloat) -> ScalarInt:
    """Leave for the terminal regime at the last living age."""
    return jnp.where(age >= _N_PERIODS - 2, RegimeId.dead, RegimeId.alive)


def _terminal_utility(*, wealth: Float1D) -> Float1D:
    return 0.0 * wealth


def test_the_documented_page_defines_the_solver_it_describes(
    documented: dict[str, Any],
) -> None:
    """The example defines `WealthSolver` and the kernel it dispatches."""
    assert {"WealthSolver", "WealthKernel", "wealth_value"} <= set(documented)


def test_the_documented_solver_publishes_the_wealth_grid_as_its_value(
    documented: dict[str, Any],
) -> None:
    """Every alive period's value equals the regime's own wealth grid."""
    model = Model(
        regimes={
            "alive": Regime(
                transition=_next_regime,
                active=lambda age: age < _N_PERIODS - 1,
                states={"wealth": _WEALTH},
                state_transitions={"wealth": _next_wealth},
                functions={"utility": _utility},
                solver=documented["WealthSolver"](),
            ),
            "dead": Regime(
                transition=None,
                states={"wealth": _WEALTH},
                functions={"utility": _terminal_utility},
            ),
        },
        ages=AgeGrid(start=0, stop=_N_PERIODS - 1, step="Y"),
        regime_id_class=RegimeId,
    )
    solution = model.solve(params={"discount_factor": 1.0}, log_level="debug")

    expected = np.asarray(_WEALTH.to_jax())
    for period in range(_N_PERIODS - 1):
        np.testing.assert_array_equal(
            np.asarray(solution.values[period]["alive"]), expected
        )
