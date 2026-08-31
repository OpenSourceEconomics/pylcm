"""One answer to "may this gated edge fold at this period".

Backward induction and forward simulation both fold a gated edge's ``Wbar`` on
one period's solved value arrays, and both decide the same three-way question:
fold, keep the value already held, or reject the model.

The distinction that carries the behaviour is whether the source regime is
there to READ the folded ``Wbar`` one period earlier. Where it is not, a
reference regime that was never solved is a boundary no-op; where it is, the
same absence is a misconfigured edge that would feed the source a stale
later-period value. The last test pins that qualifier on a live model — a
repeating self-loop edge whose fallback is inactive in the target's earliest
period — because without it the rule collapses into the naive "every reference
must always be solved", which rejects a model that is legal and solves.
"""

from types import MappingProxyType
from typing import cast

import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal as aaae

from _lcm.regime_building.gated_edges import (
    ResolvedGatedEdge,
    ResolvedStakeholderRoute,
    edge_may_fold_at_period,
    source_reads_folded_wbar,
)
from _lcm.regime_building.Q_and_F import ResolvedProjectedRegimeValue
from _lcm.solution.backward_induction import solve
from _lcm.typing import ConstraintFunction
from _lcm.utils.logging import get_logger
from lcm.ages import AgeGrid
from lcm.exceptions import ModelInitializationError
from lcm.typing import BoolND, FloatND
from tests.conftest import DECIMAL_PRECISION
from tests.regime_building.test_collective_regime_simulate import (
    _make_repeating_self_loop_regimes,
    _solve_and_process,
)

_BETA = 0.95


def _gate(V_target: FloatND) -> BoolND:
    return V_target > 0.0


def _edge(
    *, target: str, fallback: str, gate_ref: str | None = None
) -> ResolvedGatedEdge:
    """A minimal resolved edge whose fold reads `fallback` (and maybe `gate_ref`)."""
    gate_refs = (
        {}
        if gate_ref is None
        else {
            "V_ref": ResolvedProjectedRegimeValue(
                regime=gate_ref, projection={}, stakeholder_index=None
            )
        }
    )
    references = tuple(
        name
        for name in dict.fromkeys((fallback, *((gate_ref,) if gate_ref else ())))
        if name != target
    )
    return ResolvedGatedEdge(
        target=target,
        # The availability question never calls the gate; a plain predicate
        # stands in for the user's, which the user-facing `GatedEdge` widens
        # to the protocol on the modeller's behalf.
        gate=cast("ConstraintFunction", _gate),
        gate_refs=MappingProxyType(gate_refs),
        legs=(
            ResolvedStakeholderRoute(
                source_stakeholder=None,
                target_component_index=None,
                fallback=ResolvedProjectedRegimeValue(
                    regime=fallback, projection={}, stakeholder_index=None
                ),
            ),
        ),
        reference_regimes=references,
    )


def test_source_reads_folded_wbar_when_the_source_is_active_one_period_earlier():
    """A folded ``Wbar`` is read exactly when the source is active at `t - 1`."""
    assert source_reads_folded_wbar(source_active_periods=(0, 1), fold_period=1) is True


def test_source_does_not_read_a_wbar_folded_at_the_targets_earliest_period():
    """A self-loop target's earliest active period has no source one period earlier."""
    assert (
        source_reads_folded_wbar(source_active_periods=(0, 1), fold_period=0) is False
    )


def test_edge_does_not_fold_when_its_target_is_unsolved():
    """An edge whose target was not solved this period keeps the value already held."""
    assert (
        edge_may_fold_at_period(
            edge=_edge(target="src", fallback="src_fallback"),
            source_name="src",
            fold_period=2,
            solved_regimes={"src_fallback"},
            source_reads_wbar=True,
        )
        is False
    )


def test_edge_folds_when_its_target_and_every_reference_are_solved():
    """An edge folds once the target and every reference regime are solved."""
    assert (
        edge_may_fold_at_period(
            edge=_edge(target="src", fallback="src_fallback"),
            source_name="src",
            fold_period=1,
            solved_regimes={"src", "src_fallback"},
            source_reads_wbar=True,
        )
        is True
    )


def test_edge_does_not_fold_when_an_unread_periods_reference_is_unsolved():
    """An unsolved reference is a boundary no-op where no source reads the ``Wbar``."""
    assert (
        edge_may_fold_at_period(
            edge=_edge(target="src", fallback="src_fallback"),
            source_name="src",
            fold_period=0,
            solved_regimes={"src"},
            source_reads_wbar=False,
        )
        is False
    )


def test_edge_with_an_unsolved_reference_at_a_read_period_is_rejected():
    """An unsolved reference whose ``Wbar`` a source reads rejects the model."""
    with pytest.raises(ModelInitializationError, match="src_fallback"):
        edge_may_fold_at_period(
            edge=_edge(target="src", fallback="src_fallback"),
            source_name="src",
            fold_period=1,
            solved_regimes={"src"},
            source_reads_wbar=True,
        )


def test_rejection_names_every_unsolved_reference_regime():
    """The rejection names all unsolved references, not just the first."""
    with pytest.raises(ModelInitializationError, match=r"'fb'.*'ref'|'ref'.*'fb'"):
        edge_may_fold_at_period(
            edge=_edge(target="tgt", fallback="fb", gate_ref="ref"),
            source_name="src",
            fold_period=1,
            solved_regimes={"tgt"},
            source_reads_wbar=True,
        )


def test_repeating_self_loop_solves_with_its_fallback_inactive_in_the_unread_period():
    """A self-loop edge solves although its fallback is absent where no source reads.

    `src` is active in periods 0 and 1 and gates an edge back to itself, whose
    fallback `src_fallback` is active only from period 1 on. Period 0's fold has
    the target (`src`) solved and the fallback not — yet no `src` exists at
    period -1 to read it, so the edge keeps the value it already holds and the
    model solves. Requiring the fallback in every target-active period would
    reject this model instead.

    Hand computation, `beta = 0.95`: period 2 has no `src`; period 1's
    continuation is `src_exit` alone, `V_1(wage) = wage + beta * 0.5 * wage =
    1.475 * wage`; period 0's gate (`V_target > 2`) is closed at wage=1
    (`V_1(1) = 1.475`) and open at wage=2 (`V_1(2) = 2.95`), giving
    `V_0(1) = 1 + beta * 0.1 * 1 = 1.095` off the fallback and
    `V_0(2) = 2 + beta * 2.95 = 4.8025` off the target.
    """
    ages = AgeGrid(start=0, stop=3, step="Y")
    regimes_dict = _make_repeating_self_loop_regimes()
    regimes, _regime_names_to_ids = _solve_and_process(
        regimes_dict=regimes_dict, ages=ages, regime_names=list(regimes_dict)
    )
    # The activity windows that make period 0 the unread one.
    assert regimes["src"].active_periods == (0, 1)
    assert regimes["src_fallback"].active_periods == (1, 2, 3)
    edge = regimes["src"].gated_edges["src"]
    assert edge.reference_regimes == ("src_fallback",)
    assert (
        source_reads_folded_wbar(
            source_active_periods=regimes["src"].active_periods, fold_period=0
        )
        is False
    )

    flat_params = MappingProxyType(
        {
            "src": MappingProxyType(
                {"koopmans_aggregator__discount_factor": jnp.asarray(_BETA)}
            ),
            "src_exit": MappingProxyType({}),
            "src_fallback": MappingProxyType({}),
        }
    )
    solution = solve(
        flat_params=flat_params,
        ages=ages,
        regimes=regimes,
        logger=get_logger(log_level="debug"),
        enable_jit=False,
    ).value_functions

    aaae(np.asarray(solution[0]["src"]), [1.095, 4.8025], decimal=DECIMAL_PRECISION)
