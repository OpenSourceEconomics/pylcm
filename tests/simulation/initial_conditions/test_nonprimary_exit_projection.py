"""A named source member keeps its own represented destination projection."""

import jax.numpy as jnp
import numpy as np

from lcm import AgeGrid, Exit, ExitTarget, Model
from lcm.solver_api import ResultRetention
from lcm.typing import FloatND
from tests.regime_building.test_collective_regime_simulate import (
    _DISSOLUTION_PARAMS,
    DissolutionRegimeId,
    _identity_wage,
    _make_dissolution_regimes,
    _ParticipationRule,
)


def _male_wage(wage: FloatND) -> FloatND:
    return 0.0 * wage + 3.0


def test_public_current_exit_keeps_the_nonprimary_members_own_projection() -> None:
    """The male row lands at wage three while the female row keeps wage two."""
    regimes = _make_dissolution_regimes()
    regimes["married_ir"] = regimes["married_ir"].replace(
        exits={
            "dissolve": Exit(
                targets={
                    "f": ExitTarget(
                        regime="single_f", projection={"wage": _identity_wage}
                    ),
                    "m": ExitTarget(regime="single_m", projection={"wage": _male_wage}),
                },
                rule=_ParticipationRule(),
            )
        }
    )
    model = Model(
        regimes=regimes,
        ages=AgeGrid(start=0, inclusive_stop=3, step="Y"),
        regime_id_class=DissolutionRegimeId,
        initial_nodes={1: "married_ir"},
        edges={
            "married": {"married_ir": 0},
            "married_ir": {"married_terminal": 1},
            "single_f": {"single_f_terminal": 1},
            "single_m": {"single_m_terminal": 1},
        },
    )
    solution = model.solve(
        params=_DISSOLUTION_PARAMS,
        retention=ResultRetention.VALUES,
        log_level="off",
    )
    result = model.simulate(
        params=_DISSOLUTION_PARAMS,
        initial_conditions={
            "wage": jnp.asarray([2.0, 2.0]),
            "age": jnp.asarray([1.0, 1.0]),
            "regime_id": jnp.full(
                2, model.regime_names_to_ids["married_ir"], dtype=jnp.int32
            ),
            "own_stakeholder": jnp.asarray(
                [model.stakeholder_names_to_ids[member] for member in ("f", "m")],
                dtype=jnp.int32,
            ),
        },
        solution=solution,
        log_level="off",
        seed=0,
    ).to_dataframe()
    acting = result.loc[result["period"] == 1].sort_values("subject_id")
    np.testing.assert_array_equal(acting["subject_id"], [0, 1])
    np.testing.assert_array_equal(
        acting["regime_name"].to_numpy(), ["single_f", "single_m"]
    )
    np.testing.assert_array_equal(acting["wage"], [2.0, 3.0])
    # The actual male utility is 1 * work + 0 * wage, with zero terminal value.
    # The wife's threshold slack is not subtracted from her outside payment.
    np.testing.assert_array_equal(acting["value"], [5.5, 1.0])
