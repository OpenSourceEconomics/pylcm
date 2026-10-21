"""The finite query blends each kind over the stencil, then chooses at the query.

Every case here builds a synthetic two-row bank on the canonical finite nested
fixture, reads it through the installed `DCEGMQuery.read_choices`, and compares
the result with the scalar reference in `_finite_mixed_corner_oracle`. The
rows carry constant values, so the only thing under test is the operator: how
the ordinary row maxima and the held corner continuations blend under signed
and zero weights, which alternative wins at the actual query, and when the
read declares itself uncovered.
"""

import math
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.egm.dcegm_query import DCEGMQuery, FiniteNEGMQueryResult
from _lcm.egm.query_data import EGMQueryData, QueryNodeKind
from _lcm.egm.upper_envelope.segment_envelope import (
    OwnedEnvelope,
    refine_envelope_exact,
)
from _lcm.typing import EconFunctionKwargs
from lcm.typing import FloatND
from tests.conftest import DECIMAL_PRECISION, EXACT_KERNEL_SKIP_REASON, X64_ENABLED
from tests.egm._finite_germ_oracle import _emitted_row
from tests.egm._finite_germ_query_fixture import make_canonical_query
from tests.egm._finite_mixed_corner_oracle import (
    COORDINATE,
    PASSIVE_NODES,
    MixedCornerCase,
    OrdinaryRow,
    Reference,
    _blend_corner,
    _blend_ordinary,
    _eligible,
    corner_value,
    identities,
    nodewise_reference,
    reference,
)

pytestmark = pytest.mark.requires_exact_affine_kernel(reason=EXACT_KERNEL_SKIP_REASON)

type _Bank = tuple[FloatND, FloatND, FloatND, EGMQueryData]
type _Corners = tuple[tuple[float | None, ...], ...]

_LEGAL_SAVINGS = 0.125
_ILLEGAL_SAVINGS = -1.0
_MISSING = -100.0
_VALUE_POOL = (-0.75, -0.4, -0.1, 0.0, 0.2, 0.45, 0.8)
_MARGINAL_POOL = (1.0, 5.0, 17.0)
_STOCK_POOL = (0.0, 0.25, 0.5, 0.3, 0.75, 1.0, -0.5, 1.5)
_ZERO_CORNERS: _Corners = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))


def _bank(*, query: DCEGMQuery, case: MixedCornerCase) -> _Bank:
    """Emit each row through the native envelope with the case's constant tuple."""
    capacity = query.model.authority.row_capacity
    grids, values, marginals, records = [], [], [], []
    for row in range(2):
        for outer in range(3):
            ordinary = case.ordinary[row][outer]
            economic = ordinary is not None and math.isinf(ordinary.value)
            if ordinary is None:
                emitted = _emitted_row(capacity=capacity, empty=True)
            else:
                # An economic row is emitted with a placeholder value and then
                # relabelled, as the producer's known economic rows are.
                emitted = refine_envelope_exact(
                    endog_grid=jnp.asarray([1.0, 1.5]),
                    policy=jnp.asarray([1.0, 1.5]) - ordinary.savings,
                    value=jnp.full((2,), 0.0 if economic else ordinary.value),
                    savings=jnp.full((2,), ordinary.savings),
                    n_refined=capacity,
                )
                assert isinstance(emitted, OwnedEnvelope)
                assert int(emitted.n_kept) == 2
                np.testing.assert_array_equal(emitted.grid[:2], [1.0, 1.5])
                np.testing.assert_array_equal(emitted.savings[:2], ordinary.savings)
                assert bool(emitted.link_valid[0])
            known = jnp.arange(capacity) < emitted.n_kept
            held = case.corner[row][outer]
            grids.append(emitted.grid)
            values.append(jnp.where(known & economic, -jnp.inf, emitted.value))
            marginals.append(
                jnp.where(
                    known,
                    0.0 if ordinary is None or economic else ordinary.marginal,
                    jnp.nan,
                )
            )
            records.append(
                EGMQueryData(
                    savings=emitted.savings,
                    node_kind=jnp.where(
                        known,
                        QueryNodeKind.ECONOMIC_MINUS_INF
                        if economic
                        else QueryNodeKind.FINITE,
                        QueryNodeKind.PAD,
                    ).astype(jnp.int32),
                    link_valid=emitted.link_valid,
                    owner_id=emitted.owner_id,
                    side_id=emitted.side_id,
                    floor_used=jnp.asarray(0.0),
                    corner_continuation=jnp.asarray(
                        [_MISSING if held is None else held]
                    ),
                    corner_kind=jnp.asarray(
                        [
                            QueryNodeKind.PAD
                            if held is None
                            else QueryNodeKind.ECONOMIC_MINUS_INF
                            if math.isinf(held)
                            else QueryNodeKind.FINITE
                        ],
                        dtype=jnp.int32,
                    ),
                    corner_interval_id=jnp.asarray([0], dtype=jnp.int32),
                    corner_side_id=jnp.asarray([0], dtype=jnp.int32),
                )
            )

    def stack(leaves: list[FloatND]) -> FloatND:
        return jnp.stack(leaves).reshape(2, 3, *leaves[0].shape)

    data = jax.tree.map(lambda *leaves: stack(list(leaves)), *records)
    data = replace(data, floor_used=jnp.asarray([node / 4.0 for node in PASSIVE_NODES]))
    return stack(grids), stack(values), stack(marginals), data


def _read(
    *,
    query: DCEGMQuery,
    pool: EconFunctionKwargs,
    bank: _Bank,
    stock: float,
    jit: bool,
    counts: bool,
) -> tuple[FiniteNEGMQueryResult, tuple[int, int] | None]:
    grid, value, marginal, data = bank

    def read(
        *,
        grid: FloatND,
        value: FloatND,
        marginal: FloatND,
        data: EGMQueryData,
        pool: EconFunctionKwargs,
        stock: FloatND,
    ):
        if counts:
            return query.read_choices(
                grid=grid,
                value=value,
                marginal=marginal,
                data=data,
                row_queries=jnp.full((2,), COORDINATE),
                row_gradients=jnp.ones((2,)),
                passive_values=(stock,),
                passive_grids=(jnp.asarray(PASSIVE_NODES),),
                action_names=(),
                action_values=(),
                pool=pool,
                euler_gradient=jnp.asarray(1.0),
                collect_continuation_read_counts=True,
            )
        return query.read_choices(
            grid=grid,
            value=value,
            marginal=marginal,
            data=data,
            row_queries=jnp.full((2,), COORDINATE),
            row_gradients=jnp.ones((2,)),
            passive_values=(stock,),
            passive_grids=(jnp.asarray(PASSIVE_NODES),),
            action_names=(),
            action_values=(),
            pool=pool,
            euler_gradient=jnp.asarray(1.0),
            collect_continuation_read_counts=False,
        )

    passive = jnp.asarray(stock)
    pool = {**pool, "stock": passive}
    result = (jax.jit(read) if jit else read)(
        grid=grid, value=value, marginal=marginal, data=data, pool=pool, stock=passive
    )
    if counts:
        result, summary = result
        pairs = (int(summary.below_first_live), int(summary.eligible))
    else:
        pairs = None
    assert isinstance(result, FiniteNEGMQueryResult)
    assert not bool(jnp.any(result.bound_status.nonfinite_floor))
    assert not bool(jnp.any(result.bound_status.invalid_shifted_grid))
    return result, pairs


def _assert_matches(
    *, result: FiniteNEGMQueryResult, expected: Reference, pairs: tuple[int, int] | None
) -> None:
    tolerance = 10.0 ** (-DECIMAL_PRECISION)
    np.testing.assert_array_equal(result.coverage_missing, [expected.coverage_missing])
    if pairs is not None:
        assert pairs == (0, expected.eligible_pairs)
    if expected.coverage_missing:
        return
    if math.isinf(expected.value):
        np.testing.assert_array_equal(result.value, [-jnp.inf])
        np.testing.assert_array_equal(result.marginal, [0.0])
        return
    np.testing.assert_allclose(
        result.value, [expected.value], rtol=tolerance, atol=tolerance
    )
    np.testing.assert_allclose(
        result.marginal, [expected.marginal], rtol=tolerance, atol=tolerance
    )


def _run(*, case: MixedCornerCase, jit: bool = False, counts: bool = False) -> None:
    query, pool = make_canonical_query()
    pool = {**pool, "liquid": jnp.asarray(case.liquid)}
    np.testing.assert_array_equal(query.read_discount_factor(pool), 1.0)
    result, pairs = _read(
        query=query,
        pool=pool,
        bank=_bank(query=query, case=case),
        stock=case.stock,
        jit=jit,
        counts=counts,
    )
    _assert_matches(result=result, expected=reference(case=case), pairs=pairs)


def _ordinary(
    *, value: float, marginal: float = 17.0, savings: float = _LEGAL_SAVINGS
) -> OrdinaryRow:
    return OrdinaryRow(value=value, marginal=marginal, savings=savings)


def _tie_corner_to_ordinary(
    *, case: MixedCornerCase, outer: int
) -> MixedCornerCase | None:
    """Hold `W` on both rows so the corner of `outer` ties the blended ordinary.

    The construction `target - u` only lands on `target` up to rounding, so the
    held value is nudged by a few representable steps and the candidate is kept
    only when the blended corner value equals the ordinary channel exactly;
    `None` when no step hits.
    """
    ordinary = _blend_ordinary(case=case)
    if not ordinary.supported or not math.isfinite(ordinary.value):
        return None
    if not _eligible(case=case, outer=outer):
        return None
    identity = identities(stock=case.stock)[outer]
    held = ordinary.value - corner_value(case=case, identity=identity, held=0.0)
    candidates = [held]
    for _ in range(4):
        candidates.append(math.nextafter(candidates[-1], math.inf))
    candidate = held
    for _ in range(4):
        candidate = math.nextafter(candidate, -math.inf)
        candidates.append(candidate)
    for candidate in candidates:
        tied = _with_corner(case=case, row=0, outer=outer, held=candidate)
        tied = _with_corner(case=tied, row=1, outer=outer, held=candidate)
        corner = _blend_corner(case=tied, outer=outer)
        if corner.supported and corner.value == ordinary.value:
            return tied
    return None


def _with_corner(
    *, case: MixedCornerCase, row: int, outer: int, held: float | None
) -> MixedCornerCase:
    corners = [list(rows) for rows in case.corner]
    corners[row][outer] = held
    return replace(case, corner=tuple(tuple(rows) for rows in corners))


def test_blended_ordinary_channel_competes_with_the_best_actual_corner() -> None:
    """The ordinary row maxima blend first; the best corner then competes.

    Choosing on each row and blending the chosen tuples would credit one row's
    corner and the other row's ordinary tuple, a pair no single action attains,
    so this case separates the blended order from the nodewise order.
    """
    case = MixedCornerCase(
        stock=0.5,
        liquid=0.0,
        ordinary=(
            (
                _ordinary(value=-0.5, marginal=1.0),
                _ordinary(value=-0.4, marginal=2.0),
                _ordinary(value=-0.75, marginal=3.0),
            ),
            (
                _ordinary(value=0.8, marginal=5.0),
                _ordinary(value=0.2, marginal=6.0),
                _ordinary(value=0.1, marginal=7.0),
            ),
        ),
        corner=_ZERO_CORNERS,
    )
    expected = reference(case=case)
    best_corner = max(
        corner_value(case=case, identity=identity, held=0.0)
        for identity in identities(stock=0.5)
    )
    assert -0.4 < best_corner < 0.8
    nodewise = 0.5 * (best_corner + 0.8)
    assert not math.isclose(expected.value, nodewise, abs_tol=1e-3)
    np.testing.assert_allclose(expected.value, max(0.5 * (-0.4 + 0.8), best_corner))
    _run(case=case)


@pytest.mark.parametrize("jit", [False, True], ids=["eager", "jit"])
@pytest.mark.parametrize("counts", [False, True], ids=["unchecked", "checked"])
def test_mixed_rows_agree_under_jit_and_counts(*, jit: bool, counts: bool) -> None:
    """Compiled and checked reads return the same mixed pair and raw counts."""
    case = MixedCornerCase(
        stock=0.25,
        liquid=0.0,
        ordinary=(
            (_ordinary(value=-0.5, marginal=1.0), _ordinary(value=-0.6), None),
            (_ordinary(value=0.8, marginal=5.0), None, _ordinary(value=0.2)),
        ),
        corner=((0.0, -math.inf, 0.0), (0.0, 0.0, 0.0)),
    )
    assert not reference(case=case).coverage_missing
    _run(case=case, jit=jit, counts=counts)


def test_exact_mixed_tie_is_owned_by_the_ordinary_channel() -> None:
    """A corner tying the blended ordinary channel leaves the ordinary marginal."""
    case = MixedCornerCase(
        stock=0.5,
        liquid=0.0,
        ordinary=(
            (None, _ordinary(value=0.3, marginal=17.0), _ordinary(value=-0.5)),
            (_ordinary(value=0.1, marginal=2.0), _ordinary(value=0.0), None),
        ),
        corner=_ZERO_CORNERS,
    )
    tied = _tie_corner_to_ordinary(case=case, outer=0)
    assert tied is not None
    expected = reference(case=tied)
    assert not expected.coverage_missing
    np.testing.assert_allclose(expected.value, 0.5 * (0.3 + 0.1))
    assert expected.marginal == 0.5 * (17.0 + 2.0)
    _run(case=tied)


@pytest.mark.parametrize(
    ("stock", "weights"),
    [(1.5, (-0.5, 1.5)), (-0.5, (1.5, -0.5)), (1.0, (0.0, 1.0)), (0.0, (1.0, 0.0))],
    ids=["negative-first", "negative-second", "zero-first", "zero-second"],
)
def test_signed_and_zero_weights_blend_the_selected_tuples(
    *, stock: float, weights: tuple[float, float]
) -> None:
    """Extrapolation weights blend each kind's rows; zero weights are neutral."""
    # Saving 0.5 clears the floor `stock / 4` at every stock of the family.
    case = MixedCornerCase(
        stock=stock,
        liquid=0.5,
        ordinary=(
            (
                _ordinary(value=-0.5, marginal=1.0, savings=0.5),
                _ordinary(value=-0.75, savings=0.5),
                None,
            ),
            (
                _ordinary(value=0.8, marginal=5.0, savings=0.5),
                _ordinary(value=0.2, savings=0.5),
                None,
            ),
        ),
        corner=_ZERO_CORNERS,
    )
    expected = reference(case=case)
    assert not expected.coverage_missing
    best_corner = max(
        corner_value(case=case, identity=identity, held=0.0)
        for identity in identities(stock=stock)
    )
    channel = sum(w * v for w, v in zip(weights, (-0.5, 0.8), strict=True))
    np.testing.assert_allclose(expected.value, max(channel, best_corner))
    _run(case=case)


@pytest.mark.parametrize(
    "stock", [0.5, 1.5], ids=["positive-weights", "negative-weight"]
)
def test_economic_corner_alone_leaves_a_row_uncovered(*, stock: float) -> None:
    """An economic corner does not represent the row's missing interior candidates."""
    case = MixedCornerCase(
        stock=stock,
        liquid=0.0,
        ordinary=((None, None, None), (_ordinary(value=0.8), None, None)),
        corner=((-math.inf, -math.inf, -math.inf), (0.0, 0.0, 0.0)),
    )
    assert reference(case=case).coverage_missing
    _run(case=case)


@pytest.mark.parametrize(
    "stock", [0.5, 1.5], ids=["positive-weights", "negative-weight"]
)
def test_known_economic_row_follows_the_signed_convention(*, stock: float) -> None:
    """A known economic row is -inf at a positive weight, uncovered at a negative."""
    case = MixedCornerCase(
        stock=stock,
        liquid=0.0,
        ordinary=(
            (
                _ordinary(value=-math.inf, savings=0.5),
                _ordinary(value=-math.inf, savings=0.5),
                _ordinary(value=-math.inf, savings=0.5),
            ),
            (
                _ordinary(value=0.1, savings=0.5),
                _ordinary(value=0.0, savings=0.5),
                _ordinary(value=-0.1, savings=0.5),
            ),
        ),
        corner=((-math.inf, -math.inf, -math.inf), (0.0, 0.0, 0.0)),
    )
    expected = reference(case=case)
    assert expected.coverage_missing == (stock == 1.5)
    if stock == 0.5:
        assert math.isinf(expected.value)
    _run(case=case)


def test_all_corner_and_all_ordinary_rows_are_controls() -> None:
    """When only one kind is live, the blended and nodewise orders agree."""
    all_corner = MixedCornerCase(
        stock=0.5,
        liquid=0.0,
        ordinary=(
            (
                OrdinaryRow(value=-0.5, marginal=1.0, savings=_ILLEGAL_SAVINGS),
                None,
                None,
            ),
            (
                OrdinaryRow(value=0.8, marginal=5.0, savings=_ILLEGAL_SAVINGS),
                None,
                None,
            ),
        ),
        corner=_ZERO_CORNERS,
    )
    all_ordinary = replace(
        all_corner,
        ordinary=(
            (_ordinary(value=0.9, marginal=1.0), _ordinary(value=0.3), None),
            (_ordinary(value=0.8, marginal=5.0), _ordinary(value=0.2), None),
        ),
    )
    for case in (all_corner, all_ordinary):
        _run(case=case)


def _generate(*, seed: int) -> MixedCornerCase:
    """Draw one case from the class with distinct ordinary values per row."""
    rng = np.random.default_rng(seed=seed)
    stock = float(rng.choice(_STOCK_POOL))
    liquid = float(rng.choice((0.0, 0.5)))
    legal_savings = _LEGAL_SAVINGS if stock <= 0.5 else 0.5
    ordinary: list[tuple[OrdinaryRow | None, ...]] = []
    corner: list[tuple[float | None, ...]] = []
    for _ in range(2):
        values = rng.choice(_VALUE_POOL, size=3, replace=False)
        rows: list[OrdinaryRow | None] = []
        for value in values:
            kind = rng.choice(
                ("legal", "illegal", "empty", "economic"), p=(0.45, 0.15, 0.25, 0.15)
            )
            if kind == "empty":
                rows.append(None)
            else:
                rows.append(
                    OrdinaryRow(
                        value=-math.inf if kind == "economic" else float(value),
                        marginal=0.0
                        if kind == "economic"
                        else float(rng.choice(_MARGINAL_POOL)),
                        savings=_ILLEGAL_SAVINGS
                        if kind == "illegal"
                        else legal_savings,
                    )
                )
        ordinary.append(tuple(rows))
        held: list[float | None] = []
        for _outer in range(3):
            kind = rng.choice(("finite", "economic", "missing"), p=(0.6, 0.2, 0.2))
            held.append(
                float(rng.uniform(-1.0, 1.0))
                if kind == "finite"
                else -math.inf
                if kind == "economic"
                else None
            )
        corner.append(tuple(held))
    case = MixedCornerCase(
        stock=stock, liquid=liquid, ordinary=tuple(ordinary), corner=tuple(corner)
    )
    # A tie is exact only in the arithmetic that constructed it; at float32 the
    # float64 construction would not land, so only fp64 draws tie cases.
    if X64_ENABLED and rng.uniform() < 0.3:
        tied = _tie_corner_to_ordinary(case=case, outer=int(rng.integers(3)))
        if tied is not None:
            case = tied
    return case


@pytest.mark.parametrize("seed", range(120))
def test_generated_class_agrees_with_the_scalar_reference(*, seed: int) -> None:
    """Deterministic draws over kinds, support, weights and ties match the oracle."""
    _run(case=_generate(seed=seed), jit=seed % 4 == 0, counts=seed % 3 == 0)


def test_generated_class_covers_every_mode() -> None:
    """The draws exercise covered and uncovered reads, off-grid and on-node."""
    cases = [_generate(seed=seed) for seed in range(120)]
    expectations = [reference(case=case) for case in cases]
    finite = [
        expectation
        for expectation in expectations
        if not expectation.coverage_missing and math.isfinite(expectation.value)
    ]
    assert len(finite) >= 30
    assert any(expectation.coverage_missing for expectation in expectations)
    assert any(case.stock < 0.0 or case.stock > 1.0 for case in cases)
    assert any(case.stock in (0.0, 1.0) for case in cases)


@pytest.mark.parametrize("seed", range(120))
def test_nodewise_order_never_lies_below_the_blended_read(*, seed: int) -> None:
    """Choosing on every row and blending overstates the blended read (Jensen)."""
    case = _generate(seed=seed)
    expected = reference(case=case)
    nodewise = nodewise_reference(case=case)
    if nodewise is None or expected.coverage_missing or math.isinf(expected.value):
        pytest.skip("Jensen is stated for covered finite reads at interior weights.")
    assert nodewise >= expected.value - 10.0 ** (-DECIMAL_PRECISION)


def test_generated_class_has_a_strict_jensen_gap() -> None:
    """Some draw has a corner on one row and an ordinary tuple winning the other."""
    gaps = []
    for seed in range(120):
        case = _generate(seed=seed)
        expected = reference(case=case)
        nodewise = nodewise_reference(case=case)
        if nodewise is None or expected.coverage_missing or math.isinf(expected.value):
            continue
        gaps.append(nodewise - expected.value)
    assert max(gaps) > 1e-3
