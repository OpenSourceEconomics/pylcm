"""Continuation templates describe shapes and placements without owning device memory.

A gated collective model carries one value template per regime and one `Wbar`
template per gated edge. They size the planner's inventory and stand in for a
released input; neither needs a device buffer, so a budgeted solve must neither
allocate them nor charge them as fixed resident bytes.
"""

from types import MappingProxyType

import jax
import pytest

from _lcm.solution import backward_induction
from _lcm.solution.contract import ContinuationPayload
from _lcm.typing import NextValue, RegimeName
from lcm import ExecutionConfig, Model
from lcm.ages import AgeGrid
from tests.regime_building.test_gated_edges_collective_solve import (
    EKLRegimeId,
    _make_full_topology_regimes,
    _with_full_topology_laws,
)

type _Templates = tuple[
    MappingProxyType[RegimeName, NextValue],
    MappingProxyType[RegimeName, ContinuationPayload],
    MappingProxyType[tuple[RegimeName, RegimeName], NextValue],
]


def _solve_budgeted_gated_model(*, monkeypatch: pytest.MonkeyPatch) -> list[_Templates]:
    recorded: list[_Templates] = []
    original = backward_induction._build_continuation_templates
    monkeypatch.setattr(
        backward_induction,
        "_build_continuation_templates",
        lambda **kwargs: recorded.append(original(**kwargs)) or recorded[-1],
    )
    ages = AgeGrid(start=0, inclusive_stop=3, step="Y")
    model = Model(
        regimes=_make_full_topology_regimes(),
        edges=_with_full_topology_laws(
            {
                "single_f": {"married": 0, "single_f_p1": 0},
                "single_m": {"married": 0, "single_m_p1": 0},
                "single_f_p1": {"single_f_terminal": 1},
                "single_m_p1": {"single_m_terminal": (0, 1)},
                "married": {
                    "married_terminal": 1,
                    "single_f_terminal": 1,
                    "single_m_terminal": 1,
                },
            }
        ),
        ages=ages,
        regime_id_class=EKLRegimeId,
        initial_nodes={ages.exact_values[0]: "single_f"},
        execution_config=ExecutionConfig(device_memory_bytes=2**30),
    )
    model.solve(
        params={"discount_factor": 0.95, "delta_f": 0.5, "delta_m": 0.2},
        log_level="off",
    )
    return recorded


@pytest.mark.parametrize(
    ("position", "label"), [(0, "regime value"), (2, "gated edge")]
)
def test_budgeted_gated_solve_holds_no_concrete_template(
    *, position: int, label: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every regime-value and gated-edge template is abstract, never a device array."""
    recorded = _solve_budgeted_gated_model(monkeypatch=monkeypatch)
    leaves = [
        leaf for templates in recorded for leaf in jax.tree.leaves(templates[position])
    ]
    assert leaves, f"the model built no {label} template"
    assert not any(isinstance(leaf, jax.Array) for leaf in leaves)
