"""Lowering a case-piece model into per-case smooth DAG variants.

The lowering reads structured predicates and piece metadata from a regime's
function pool, checks that every split output is fully covered, and produces one
producer-swap map per (predicate, side) case — the specialized smooth DAG NBEGM
runs EGM on.
"""

from types import MappingProxyType, MethodType
from typing import cast

import pytest

from _lcm.egm.nbegm import PieceSet, collect_nbegm_metadata
from _lcm.solution.nbegm import (
    _NBEGMCaseIntervalEvaluators,
    build_nbegm_interval_recipe,
)
from _lcm.typing import EconFunctionsMapping
from lcm import ref
from lcm.case_piece import case_boundary, piece
from lcm.exceptions import NBEGMCaseError

medicaid_eligible = case_boundary(
    condition=ref("assets") < ref("medicaid_asset_limit"),
    kind="jump",
)


@piece(output="oop", when=medicaid_eligible)
def oop_medicaid(medical_expense):
    return 0.1 * medical_expense


@piece(output="oop", otherwise=medicaid_eligible)
def oop_private(medical_expense):
    return 0.9 * medical_expense


def utility(*, consumption, oop):
    return consumption - oop


def _covered_pool():
    return {
        "medicaid_eligible": medicaid_eligible,
        "oop_medicaid": oop_medicaid,
        "oop_private": oop_private,
        "utility": utility,
    }


def test_collect_records_the_piece_set_and_its_boundary():
    """A fully-covered split yields one `PieceSet` and registers its boundary."""
    registry = collect_nbegm_metadata(functions=_covered_pool())
    assert registry.piece_sets == (
        PieceSet(
            output="oop",
            predicate_name="medicaid_eligible",
            when_func="oop_medicaid",
            otherwise_func="oop_private",
        ),
    )
    assert set(registry.boundaries) == {"medicaid_eligible"}


def test_collect_rejects_a_split_missing_its_otherwise_side():
    """A `when` piece without its `otherwise` counterpart fails coverage."""
    pool = {
        "medicaid_eligible": medicaid_eligible,
        "oop_medicaid": oop_medicaid,
        "utility": utility,
    }
    with pytest.raises(NBEGMCaseError, match="otherwise"):
        collect_nbegm_metadata(functions=pool)


def test_collect_rejects_a_piece_referencing_an_undeclared_boundary():
    """A piece whose predicate is not a `case_boundary` cannot be lowered."""

    undeclared_boundary = case_boundary(condition=ref("assets") < 0.0, kind="jump")

    @piece(output="oop", when=undeclared_boundary)
    def oop_a(medical_expense):
        return medical_expense

    @piece(output="oop", otherwise=undeclared_boundary)
    def oop_b(medical_expense):
        return medical_expense

    pool = {"oop_a": oop_a, "oop_b": oop_b}
    with pytest.raises(NBEGMCaseError, match="not present"):
        collect_nbegm_metadata(functions=pool)


def test_a_literal_case_threshold_is_bound_once_when_the_recipe_is_built():
    """Resolving a literal boundary returns the one JAX scalar bound at build.

    The compiled body that resolves the threshold performs no conversion: every
    call reads back the same array the recipe bound in the working float format.
    """
    literal_boundary = case_boundary(condition=ref("assets") < 5.0, kind="jump")

    @piece(output="oop", when=literal_boundary)
    def oop_low(medical_expense):
        return 0.1 * medical_expense

    @piece(output="oop", otherwise=literal_boundary)
    def oop_high(medical_expense):
        return 0.9 * medical_expense

    pool = MappingProxyType(
        {"eligible": literal_boundary, "oop_low": oop_low, "oop_high": oop_high}
    )
    recipe = build_nbegm_interval_recipe(
        declarations=pool,
        functions=cast("EconFunctionsMapping", pool),
        liquid_name="assets",
        state_names=("assets",),
        action_names=(),
    )
    evaluators = cast(
        "_NBEGMCaseIntervalEvaluators",
        cast("MethodType", recipe.evaluate_preimages).__self__,
    )
    resolve = evaluators.threshold
    assert resolve({}) is resolve({})
