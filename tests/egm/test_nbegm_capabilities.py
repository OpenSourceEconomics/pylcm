"""What the case-piece solvers declare they can do with a constraint.

The NBEGM kernels invert the Euler equation on a savings grid and evaluate no
user constraint at any point, so their declaration is the strongest one the
language allows: no name is readable where a constraint would be called. Every
constraint therefore reaches a disposition of `Reject` unless a proof claims it
first, and the one proof they carry is the borrowing limit their own savings
grid already enforces.

The proof keys on the comparison the declaration stands for, not on the type of
object the user happened to construct, so a bound written out by hand is proved
exactly as the convenience constructor's is.
"""

from types import MappingProxyType

import jax.numpy as jnp
import pytest

import lcm
from _lcm.constraints.capabilities import compile_constraints
from _lcm.constraints.dispositions import (
    ConstraintContext,
    ProvedByConstruction,
    Reject,
)
from _lcm.constraints.processed import normalize_constraints
from _lcm.egm.nbegm_capabilities import case_piece_capabilities
from lcm import LinSpacedGrid, LiquidMargin, post_decision_lower_bound
from lcm.solvers import NBEGM, NNBEGM
from lcm.typing import BoolND, ContinuousAction, ContinuousState

_SAVINGS_GRID = LinSpacedGrid(start=0.0, stop=20.0, n_points=10)
_MARGIN = LiquidMargin(
    state="liquid",
    action="consumption",
    resources="resources",
    post_decision_state="savings",
)


def _context() -> ConstraintContext:
    """A minimal context for a one-asset case-piece regime."""
    return ConstraintContext(
        regime_name="alive",
        phase="solve",
        grids=MappingProxyType({"liquid": _SAVINGS_GRID}),
        function_names=frozenset({"resources", "savings", "utility"}),
        param_names=frozenset({"crra"}),
    )


def _dispositions(constraints):
    """Compile a constraint pool under the case-piece declaration."""
    return compile_constraints(
        constraints=normalize_constraints(constraints=constraints),
        capabilities=case_piece_capabilities(
            savings_grid=_SAVINGS_GRID, post_decision_function="savings"
        ),
        context=_context(),
    )


def rationing(consumption: ContinuousAction, liquid: ContinuousState) -> BoolND:
    """A feasibility predicate no savings-grid node locates."""
    return jnp.square(consumption) + jnp.square(liquid) <= 400.0


def test_a_bound_the_savings_grid_enforces_is_proved_by_construction():
    """The borrowing limit the grid already imposes needs no evaluation."""
    got = _dispositions(
        {"borrowing_limit": post_decision_lower_bound(margin=_MARGIN, lower=0.0)}
    )

    assert isinstance(got["borrowing_limit"], ProvedByConstruction)


def test_the_proof_names_the_savings_grid_as_what_enforces_the_bound():
    """The discharged constraint carries a reason a diagnostic can quote."""
    got = _dispositions(
        {"borrowing_limit": post_decision_lower_bound(margin=_MARGIN, lower=0.0)}
    )

    assert "savings grid" in got["borrowing_limit"].proof.reason


def test_a_hand_written_bound_is_proved_exactly_as_the_constructor_is():
    """`ref("savings") >= 0.0` means what `post_decision_lower_bound` means.

    The proof keys on the comparison, so the two spellings cannot be admitted
    and refused respectively — which is what happens the moment a proof keys on
    a marker attribute only one of them carries.
    """
    got = _dispositions({"borrowing_limit": lcm.ref("savings") >= 0.0})

    assert isinstance(got["borrowing_limit"], ProvedByConstruction)


def test_a_bound_the_grid_contradicts_is_not_proved():
    """A limit the grid does not impose is refused rather than quietly proved."""
    got = _dispositions({"borrowing_limit": lcm.ref("savings") >= 5.0})

    assert isinstance(got["borrowing_limit"], Reject)


def test_a_general_predicate_is_rejected():
    """No name is readable where the kernel would call a constraint."""
    got = _dispositions({"rationing": rationing})

    assert isinstance(got["rationing"], Reject)


def test_the_refusal_names_the_constraint_the_kernel_cannot_evaluate():
    """The diagnostic identifies which declaration was refused."""
    got = _dispositions({"rationing": rationing})

    assert "rationing" in got["rationing"].reason


@pytest.mark.parametrize("solver_name", ["NBEGM", "NNBEGM"])
def test_the_case_piece_solvers_declare_an_empty_allow_list_not_the_default(
    solver_name,
):
    """Neither solver runs on the inherited "no restriction" default.

    An absent declaration and a permissive one produce the same disposition for
    any constraint that happens to be evaluable, so asserting the dispositions
    alone would not tell the two apart. `None` would mean the kernel reads every
    name where it evaluates a constraint, which is the opposite of true for it.
    """
    solver = (
        NBEGM(savings_grid=_SAVINGS_GRID)
        if solver_name == "NBEGM"
        else NNBEGM(inner=NBEGM(savings_grid=_SAVINGS_GRID), outer_grid=_SAVINGS_GRID)
    )

    assert solver.constraint_capabilities.pre_inner_available_names == frozenset()
