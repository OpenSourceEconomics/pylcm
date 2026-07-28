"""Exact rational reference for the MSS segment-envelope contract.

An independent oracle for the exact upper envelope of a set of affine value
segments and for the certified sign of an affine difference. It uses
`fractions.Fraction` throughout, shares no control flow with the production MSS
backend, and is the acceptance reference the optimized `refine_envelope` is
tested against (open-cell ownership, node-aligned crossings, translation
invariance, collinear tie intervals).

The module carries two independent representations:

- rational `Branch`/`Segment` for the geometry (exact envelope, breakpoints,
  interval owners, pairwise crossings); and
- `FloatSegment` for classifying IEEE-float inputs by the exact sign of their
  rational margin (the reference for the certified-sign primitive).
"""

from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations, pairwise

F = Fraction


@dataclass(frozen=True)
class Branch:
    """A strictly x-monotone polyline run with aligned value and policy nodes."""

    name: str
    x: tuple[Fraction, ...]
    v: tuple[Fraction, ...]
    policy: tuple[Fraction, ...]

    def __post_init__(self) -> None:
        if not (len(self.x) == len(self.v) == len(self.policy)):
            raise ValueError("branch arrays must align")
        if len(self.x) < 2:
            raise ValueError("a branch must contain at least one nonzero-width link")
        if any(b <= a for a, b in pairwise(self.x)):
            raise ValueError("branch x nodes must be strictly increasing")


@dataclass(frozen=True)
class Segment:
    """One affine link of a branch, carrying its value and policy endpoints."""

    branch: str
    x0: Fraction
    x1: Fraction
    v0: Fraction
    v1: Fraction
    p0: Fraction
    p1: Fraction

    def value(self, q: Fraction) -> Fraction:
        """Return the exact value at `q`, which must lie within the link."""
        if not self.x0 <= q <= self.x1:
            raise ValueError("query outside segment")
        t = (q - self.x0) / (self.x1 - self.x0)
        return self.v0 + t * (self.v1 - self.v0)

    def policy(self, q: Fraction) -> Fraction:
        """Return the exact policy at `q`, which must lie within the link."""
        if not self.x0 <= q <= self.x1:
            raise ValueError("query outside segment")
        t = (q - self.x0) / (self.x1 - self.x0)
        return self.p0 + t * (self.p1 - self.p0)


def segments(branch: Branch) -> tuple[Segment, ...]:
    """Return the consecutive affine links of `branch`."""
    return tuple(
        Segment(branch.name, x0, x1, v0, v1, p0, p1)
        for x0, x1, v0, v1, p0, p1 in zip(
            branch.x[:-1],
            branch.x[1:],
            branch.v[:-1],
            branch.v[1:],
            branch.policy[:-1],
            branch.policy[1:],
            strict=True,
        )
    )


def exact_intersection(a: Segment, b: Segment) -> Fraction | None:
    """Return the exact crossing abscissa of two links inside their overlap.

    Returns `None` when the links do not overlap or are collinear/parallel
    (collinear overlap is a separate tie-interval degeneracy, not a crossing).
    """
    lo = max(a.x0, b.x0)
    hi = min(a.x1, b.x1)
    if lo > hi:
        return None
    slope_a = (a.v1 - a.v0) / (a.x1 - a.x0)
    slope_b = (b.v1 - b.v0) / (b.x1 - b.x0)
    intercept_a = a.v0 - slope_a * a.x0
    intercept_b = b.v0 - slope_b * b.x0
    if slope_a == slope_b:
        return None
    q = (intercept_b - intercept_a) / (slope_a - slope_b)
    return q if lo <= q <= hi else None


def brute_pair_crossings(a: Branch, b: Branch) -> set[Fraction]:
    """Return every crossing abscissa over all link pairs of two branches."""
    out: set[Fraction] = set()
    for seg_a in segments(a):
        for seg_b in segments(b):
            q = exact_intersection(seg_a, seg_b)
            if q is not None:
                out.add(q)
    return out


def merge_pair_crossings(a: Branch, b: Branch) -> set[Fraction]:
    """Return the crossings of two x-monotone branches via a merged sweep.

    Merges the two node sequences over their common support and tests the one
    link pair active on each cell, plus certified endpoint coincidences. This is
    the consecutive-in-merge enumeration whose completeness the mutation suite
    checks against `brute_pair_crossings`.
    """
    lo = max(a.x[0], b.x[0])
    hi = min(a.x[-1], b.x[-1])
    if lo > hi:
        return set()
    cuts = sorted({x for x in a.x + b.x if lo <= x <= hi} | {lo, hi})
    seg_a = segments(a)
    seg_b = segments(b)
    out: set[Fraction] = set()
    for left, right in pairwise(cuts):
        if right <= left:
            continue
        mid = (left + right) / 2
        active_a = next((s for s in seg_a if s.x0 <= mid <= s.x1), None)
        active_b = next((s for s in seg_b if s.x0 <= mid <= s.x1), None)
        if active_a is None or active_b is None:
            continue
        q = exact_intersection(active_a, active_b)
        if q is not None and left <= q <= right:
            out.add(q)
    for q in cuts:
        active_a = next((s for s in seg_a if s.x0 <= q <= s.x1), None)
        active_b = next((s for s in seg_b if s.x0 <= q <= s.x1), None)
        if (
            active_a is not None
            and active_b is not None
            and active_a.value(q) == active_b.value(q)
        ):
            out.add(q)
    return out


def exact_envelope(
    branches: tuple[Branch, ...], q: Fraction
) -> tuple[Fraction, tuple[str, ...]]:
    """Return the exact envelope value and the sorted owner names at `q`."""
    candidates: list[tuple[Fraction, str]] = [
        (seg.value(q), branch.name)
        for branch in branches
        for seg in segments(branch)
        if seg.x0 <= q <= seg.x1
    ]
    if not candidates:
        raise ValueError("query outside union support")
    best = max(value for value, _ in candidates)
    owners = tuple(sorted({name for value, name in candidates if value == best}))
    return best, owners


def all_breakpoints(branches: tuple[Branch, ...]) -> set[Fraction]:
    """Return every node abscissa and every pairwise run crossing."""
    points = {x for branch in branches for x in branch.x}
    for a, b in combinations(branches, 2):
        points |= merge_pair_crossings(a, b)
    return points


def interval_owners(
    branches: tuple[Branch, ...], points: set[Fraction]
) -> list[tuple[Fraction, Fraction, tuple[str, ...]]]:
    """Return the exact owner of every open cell between consecutive breakpoints.

    Evaluating one interior representative (the cell midpoint) resolves open-cell
    ownership that vertex-only argmax misses.
    """
    ordered = sorted(points)
    out: list[tuple[Fraction, Fraction, tuple[str, ...]]] = []
    for left, right in pairwise(ordered):
        if right <= left:
            continue
        mid = (left + right) / 2
        try:
            _, owners = exact_envelope(branches, mid)
        except ValueError:
            continue
        out.append((left, right, owners))
    return out


def f1_branches() -> tuple[Branch, ...]:
    """Return the round-15 F1 three-branch witness (A=1+R/10, B=.84+R/3, C=.2+R)."""
    return (
        Branch("A", (F(0), F(1)), (F(1), F(11, 10)), (F(10), F(10))),
        Branch("B", (F(0), F(1)), (F(21, 25), F(88, 75)), (F(3), F(3))),
        Branch("C", (F(3, 5), F(6, 5)), (F(4, 5), F(7, 5)), (F(1), F(1))),
    )


def to_fraction(x: float) -> Fraction:
    """Return the exact rational value of an IEEE float."""
    return Fraction.from_float(float(x))


@dataclass(frozen=True)
class FloatSegment:
    """An affine link given by IEEE-float endpoints, read exactly as rationals."""

    x0: float
    x1: float
    v0: float
    v1: float
    policy: float

    def exact(self) -> tuple[Fraction, Fraction, Fraction, Fraction]:
        """Return the exact rational endpoints of this float link."""
        return (
            to_fraction(self.x0),
            to_fraction(self.x1),
            to_fraction(self.v0),
            to_fraction(self.v1),
        )


def exact_value(seg: FloatSegment, x: float | Fraction) -> Fraction:
    """Return the exact value of a float link at `x`."""
    x0, x1, v0, v1 = seg.exact()
    q = x if isinstance(x, Fraction) else to_fraction(x)
    if not x0 <= q <= x1:
        raise ValueError("query outside segment")
    return (v0 * (x1 - q) + v1 * (q - x0)) / (x1 - x0)


def exact_margin(a: FloatSegment, b: FloatSegment, x: float | Fraction) -> Fraction:
    """Return the exact value difference `a - b` at `x`."""
    return exact_value(a, x) - exact_value(b, x)


def sign(x: Fraction) -> int:
    """Return the sign of a rational as -1, 0, or +1."""
    return (x > 0) - (x < 0)


def exact_crossing(a: FloatSegment, b: FloatSegment) -> Fraction | None:
    """Return the exact crossing of two float links inside their overlap, else None."""
    lo = max(to_fraction(a.x0), to_fraction(b.x0))
    hi = min(to_fraction(a.x1), to_fraction(b.x1))
    if lo > hi:
        return None
    margin_lo = exact_margin(a, b, lo)
    margin_hi = exact_margin(a, b, hi)
    if margin_lo == margin_hi:
        return lo if margin_lo == 0 and lo == hi else None
    root = lo - margin_lo * (hi - lo) / (margin_hi - margin_lo)
    return root if lo <= root <= hi else None


def classify_cell(
    a: FloatSegment, b: FloatSegment, left: float, right: float
) -> tuple[str, Fraction | None]:
    """Classify a cell by the certified endpoint signs of the value difference.

    Returns one of `"none"`, `"left_endpoint"`, `"right_endpoint"`, `"interior"`,
    `"tie_interval"` with the exact crossing abscissa where applicable.
    """
    left_q, right_q = to_fraction(left), to_fraction(right)
    margin_left, margin_right = exact_margin(a, b, left_q), exact_margin(a, b, right_q)
    if margin_left == 0 and margin_right == 0:
        return "tie_interval", None
    if margin_left == 0:
        return "left_endpoint", left_q
    if margin_right == 0:
        return "right_endpoint", right_q
    if sign(margin_left) != sign(margin_right):
        root = left_q - margin_left * (right_q - left_q) / (margin_right - margin_left)
        return "interior", root
    return "none", None


def has_collinear_overlap(a: Branch, b: Branch) -> bool:
    """Return whether any two links coincide on a positive-width overlap."""
    for seg_a in segments(a):
        for seg_b in segments(b):
            lo, hi = max(seg_a.x0, seg_b.x0), min(seg_a.x1, seg_b.x1)
            if lo > hi:
                continue
            slope_a = (seg_a.v1 - seg_a.v0) / (seg_a.x1 - seg_a.x0)
            slope_b = (seg_b.v1 - seg_b.v0) / (seg_b.x1 - seg_b.x0)
            if slope_a == slope_b and seg_a.value(lo) == seg_b.value(lo):
                return True
    return False
