#ifndef PYLCM_EXACT_CELL_HULL_CORE_H_
#define PYLCM_EXACT_CELL_HULL_CORE_H_

#include "exact_affine_core.h"

#ifdef __CUDACC__
#define PYLCM_HULL_HD __host__ __device__
#define PYLCM_HULL_INLINE __forceinline__
#else
#define PYLCM_HULL_HD
#define PYLCM_HULL_INLINE inline
#endif

namespace pylcm_exact_affine {

// The hull reuses the exact affine determinant but also rounds one rational
// crossing. Four spare limbs leave room for the at-most-precision-bit shift
// needed by that final rounding without changing the fixed-width contract.
template <typename T>
struct HullTraits {
  static constexpr int kLimbs = FloatTraits<T>::kCompareLimbs + 4;
};

template <typename T, int N>
PYLCM_HULL_HD PYLCM_HULL_INLINE bool RoundRatioGeneral(
    const BigUInt<N>& numerator, bool negative, int numerator_base,
    const BigUInt<N>& denominator, int denominator_base, T* result) {
  using Traits = FloatTraits<T>;
  using UInt = typename Traits::UInt;
  if (IsZero(denominator)) return false;
  if (IsZero(numerator)) {
    *result = T{0};
    return true;
  }

  const int power = numerator_base - denominator_base;
  int exponent = FloorLog2Ratio(numerator, denominator) + power;
  const int min_subnormal = Traits::kMinSubnormalExponent;
  UInt bits = negative ? Traits::kSignMask : UInt{0};

  if (exponent >= Traits::kEmin) {
    const int binary_shift = power - exponent + (Traits::kPrecision - 1);
    uint64_t significand = 0;
    if (binary_shift >= 0) {
      BigUInt<N> scaled_numerator;
      if (!ShiftLeft(numerator, binary_shift, &scaled_numerator) ||
          !DivideRoundedSmall(scaled_numerator, denominator, &significand)) {
        return false;
      }
    } else {
      BigUInt<N> scaled_denominator;
      if (!ShiftLeft(denominator, -binary_shift, &scaled_denominator) ||
          !DivideRoundedSmall(numerator, scaled_denominator, &significand)) {
        return false;
      }
    }
    const uint64_t overflow = uint64_t{1} << Traits::kPrecision;
    if (significand == overflow) {
      significand >>= 1;
      ++exponent;
    }
    if (exponent > Traits::kEmax) return false;
    const uint64_t hidden = uint64_t{1} << (Traits::kPrecision - 1);
    if (significand < hidden || significand >= overflow) return false;
    const UInt exponent_field = static_cast<UInt>(exponent + Traits::kBias);
    bits |= exponent_field << Traits::kFractionBits;
    bits |= static_cast<UInt>(significand - hidden);
  } else {
    const int binary_shift = power - min_subnormal;
    uint64_t fraction = 0;
    if (binary_shift >= 0) {
      BigUInt<N> scaled_numerator;
      if (!ShiftLeft(numerator, binary_shift, &scaled_numerator) ||
          !DivideRoundedSmall(scaled_numerator, denominator, &fraction)) {
        return false;
      }
    } else {
      BigUInt<N> scaled_denominator;
      if (!ShiftLeft(denominator, -binary_shift, &scaled_denominator) ||
          !DivideRoundedSmall(numerator, scaled_denominator, &fraction)) {
        return false;
      }
    }
    const uint64_t min_normal = uint64_t{1} << (Traits::kPrecision - 1);
    if (fraction > min_normal) return false;
    if (fraction == min_normal) {
      bits |= UInt{1} << Traits::kFractionBits;
    } else {
      bits |= static_cast<UInt>(fraction);
    }
  }
  *result = BitCast<T>(bits);
  return true;
}

template <typename T>
PYLCM_HULL_HD PYLCM_HULL_INLINE bool IsFinite(T value) {
  return Decode(value).finite;
}

template <typename T>
PYLCM_HULL_HD PYLCM_HULL_INLINE T NextUp(T value) {
  using Traits = FloatTraits<T>;
  using UInt = typename Traits::UInt;
  UInt bits = BitCast<UInt>(value);
  const UInt magnitude = bits & ~Traits::kSignMask;
  if (magnitude == 0) return BitCast<T>(UInt{1});
  if ((bits & Traits::kSignMask) == 0) {
    ++bits;
  } else {
    --bits;
  }
  return BitCast<T>(bits);
}

template <typename T>
PYLCM_HULL_HD PYLCM_HULL_INLINE T NextDown(T value) {
  using Traits = FloatTraits<T>;
  using UInt = typename Traits::UInt;
  UInt bits = BitCast<UInt>(value);
  const UInt magnitude = bits & ~Traits::kSignMask;
  if (magnitude == 0) return BitCast<T>(Traits::kSignMask | UInt{1});
  if ((bits & Traits::kSignMask) == 0) {
    --bits;
  } else {
    ++bits;
  }
  return BitCast<T>(bits);
}

template <typename T, int N>
PYLCM_HULL_HD PYLCM_HULL_INLINE bool BuildDifferenceCoefficients(
    T ax0, T ax1, T av0, T av1, T bx0, T bx1, T bv0, T bv1,
    BigUInt<N>* constant, bool* constant_negative,
    BigUInt<N>* rate, bool* rate_negative) {
  using Traits = FloatTraits<T>;
  const FloatParts p[8] = {Decode(ax0), Decode(ax1), Decode(av0), Decode(av1),
                           Decode(bx0), Decode(bx1), Decode(bv0), Decode(bv1)};
  for (int i = 0; i < 8; ++i) {
    if (!p[i].finite) return false;
  }
  if (!ExactGreater(ax1, ax0) || !ExactGreater(bx1, bx0)) return false;

  BigUInt<N> c_pos, c_neg;
  Clear(&c_pos);
  Clear(&c_neg);
  constexpr int constant_base = 3 * Traits::kMinSubnormalExponent;
  bool ok = true;
  // (av0*ax1 - av1*ax0) * (bx1-bx0)
  ok &= AccumulateProduct3(&c_pos, &c_neg, p[2], p[1], p[5], 1, constant_base);
  ok &= AccumulateProduct3(&c_pos, &c_neg, p[2], p[1], p[4], -1, constant_base);
  ok &= AccumulateProduct3(&c_pos, &c_neg, p[3], p[0], p[5], -1, constant_base);
  ok &= AccumulateProduct3(&c_pos, &c_neg, p[3], p[0], p[4], 1, constant_base);
  // -(bv0*bx1 - bv1*bx0) * (ax1-ax0)
  ok &= AccumulateProduct3(&c_pos, &c_neg, p[6], p[5], p[1], -1, constant_base);
  ok &= AccumulateProduct3(&c_pos, &c_neg, p[6], p[5], p[0], 1, constant_base);
  ok &= AccumulateProduct3(&c_pos, &c_neg, p[7], p[4], p[1], 1, constant_base);
  ok &= AccumulateProduct3(&c_pos, &c_neg, p[7], p[4], p[0], -1, constant_base);
  ResolveSigned(c_pos, c_neg, constant, constant_negative);

  BigUInt<N> r_pos, r_neg;
  Clear(&r_pos);
  Clear(&r_neg);
  constexpr int rate_base = 2 * Traits::kMinSubnormalExponent;
  // (av1-av0)*(bx1-bx0) - (bv1-bv0)*(ax1-ax0)
  ok &= AccumulateProduct2(&r_pos, &r_neg, p[3], p[5], 1, rate_base);
  ok &= AccumulateProduct2(&r_pos, &r_neg, p[3], p[4], -1, rate_base);
  ok &= AccumulateProduct2(&r_pos, &r_neg, p[2], p[5], -1, rate_base);
  ok &= AccumulateProduct2(&r_pos, &r_neg, p[2], p[4], 1, rate_base);
  ok &= AccumulateProduct2(&r_pos, &r_neg, p[7], p[1], -1, rate_base);
  ok &= AccumulateProduct2(&r_pos, &r_neg, p[7], p[0], 1, rate_base);
  ok &= AccumulateProduct2(&r_pos, &r_neg, p[6], p[1], 1, rate_base);
  ok &= AccumulateProduct2(&r_pos, &r_neg, p[6], p[0], -1, rate_base);
  ResolveSigned(r_pos, r_neg, rate, rate_negative);
  return ok;
}

template <typename T>
PYLCM_HULL_HD PYLCM_HULL_INLINE int32_t AffineSlopeCompare(
    T ax0, T ax1, T av0, T av1, T bx0, T bx1, T bv0, T bv1) {
  constexpr int N = HullTraits<T>::kLimbs;
  BigUInt<N> constant, rate;
  bool constant_negative = false;
  bool rate_negative = false;
  if (!BuildDifferenceCoefficients(ax0, ax1, av0, av1, bx0, bx1, bv0, bv1,
                                   &constant, &constant_negative, &rate,
                                   &rate_negative)) {
    return kUnresolved;
  }
  if (IsZero(rate)) return 0;
  return rate_negative ? -1 : 1;
}

template <typename T>
PYLCM_HULL_HD PYLCM_HULL_INLINE bool ExactHandoverState(
    T ax0, T ax1, T av0, T av1, T bx0, T bx1, T bv0, T bv1,
    T left, T right, T* state) {
  using Traits = FloatTraits<T>;
  constexpr int N = HullTraits<T>::kLimbs;
  if (!IsFinite(left) || !IsFinite(right) || !ExactGreater(right, left)) {
    return false;
  }

  BigUInt<N> constant, rate;
  bool constant_negative = false;
  bool rate_negative = false;
  if (!BuildDifferenceCoefficients(ax0, ax1, av0, av1, bx0, bx1, bv0, bv1,
                                   &constant, &constant_negative, &rate,
                                   &rate_negative)) {
    return false;
  }
  if (IsZero(rate) || !rate_negative) return false;

  constexpr int constant_base = 3 * Traits::kMinSubnormalExponent;
  constexpr int rate_base = 2 * Traits::kMinSubnormalExponent;
  T nearest;
  if (!RoundRatioGeneral<T>(constant, constant_negative, constant_base, rate,
                            rate_base, &nearest) ||
      !IsFinite(nearest)) {
    return false;
  }

  int32_t at_nearest = CertifiedAffineCompare(
      ax0, ax1, av0, av1, bx0, bx1, bv0, bv1, nearest);
  if (at_nearest == kUnresolved) return false;

  T candidate = nearest;
  if (at_nearest > 0) {
    candidate = NextUp(nearest);
  } else {
    const T predecessor = NextDown(nearest);
    if (IsFinite(predecessor)) {
      const int32_t at_predecessor = CertifiedAffineCompare(
          ax0, ax1, av0, av1, bx0, bx1, bv0, bv1, predecessor);
      if (at_predecessor == kUnresolved) return false;
      if (at_predecessor <= 0) candidate = predecessor;
    }
  }

  if (ExactGreater(left, candidate)) candidate = left;
  if (ExactGreater(candidate, right)) candidate = right;
  const int32_t at_candidate = CertifiedAffineCompare(
      ax0, ax1, av0, av1, bx0, bx1, bv0, bv1, candidate);
  if (at_candidate == kUnresolved || at_candidate > 0) {
    return false;
  }
  if (ExactGreater(candidate, left)) {
    const T predecessor = NextDown(candidate);
    if (!ExactGreater(left, predecessor)) {
      const int32_t at_predecessor = CertifiedAffineCompare(
          ax0, ax1, av0, av1, bx0, bx1, bv0, bv1, predecessor);
      if (at_predecessor == kUnresolved || at_predecessor <= 0) return false;
    }
  }
  *state = candidate;
  return true;
}

template <typename T>
struct LineView {
  T x0;
  T x1;
  T v0;
  T v1;
  int32_t run;
};

template <typename T>
PYLCM_HULL_HD PYLCM_HULL_INLINE bool LoadLine(
    int32_t run, const int32_t* low, const int32_t* high,
    const T* endog_grid, const T* value, int32_t n_candidates,
    LineView<T>* line) {
  const int32_t lo = low[run];
  const int32_t hi = high[run];
  if (lo < 0 || hi < 0 || lo >= n_candidates || hi >= n_candidates) {
    return false;
  }
  line->x0 = endog_grid[lo];
  line->x1 = endog_grid[hi];
  line->v0 = value[lo];
  line->v1 = value[hi];
  line->run = run;
  return IsFinite(line->x0) && IsFinite(line->x1) &&
         IsFinite(line->v0) && IsFinite(line->v1) &&
         ExactGreater(line->x1, line->x0);
}

template <typename T>
PYLCM_HULL_HD PYLCM_HULL_INLINE int32_t CompareLinesAt(const LineView<T>& a,
                                             const LineView<T>& b,
                                             T query) {
  return CertifiedAffineCompare(a.x0, a.x1, a.v0, a.v1,
                                b.x0, b.x1, b.v0, b.v1, query);
}

template <typename T>
PYLCM_HULL_HD PYLCM_HULL_INLINE int32_t CompareLineSlopes(const LineView<T>& a,
                                                const LineView<T>& b) {
  return AffineSlopeCompare(a.x0, a.x1, a.v0, a.v1,
                            b.x0, b.x1, b.v0, b.v1);
}

template <typename T>
PYLCM_HULL_HD PYLCM_HULL_INLINE bool BetterAt(const LineView<T>& candidate,
                                    const LineView<T>& incumbent,
                                    T query, bool* better) {
  const int32_t order = CompareLinesAt(candidate, incumbent, query);
  if (order == kUnresolved) return false;
  if (order != 0) {
    *better = order > 0;
    return true;
  }
  const int32_t slope = CompareLineSlopes(candidate, incumbent);
  if (slope == kUnresolved) return false;
  if (slope != 0) {
    *better = slope > 0;
    return true;
  }
  *better = candidate.run < incumbent.run;
  return true;
}

template <typename T>
PYLCM_HULL_HD PYLCM_HULL_INLINE bool ExactCellHull(
    T left, T right, const int32_t* live, const int32_t* low,
    const int32_t* high, int32_t max_runs, const T* endog_grid,
    const T* value, int32_t n_candidates, T* bounds, int32_t* owners) {
  bounds[0] = left;
  for (int32_t i = 0; i < max_runs; ++i) {
    owners[i] = 0;
    bounds[i + 1] = right;
  }

  int32_t first_live = -1;
  for (int32_t run = 0; run < max_runs; ++run) {
    if (live[run] != 0) {
      first_live = run;
      break;
    }
  }
  if (first_live < 0) return true;  // Padding cell.
  if (!IsFinite(left) || !IsFinite(right) || !ExactGreater(right, left)) {
    return false;
  }

  LineView<T> owner;
  if (!LoadLine(first_live, low, high, endog_grid, value, n_candidates,
                &owner)) {
    return false;
  }
  for (int32_t run = first_live + 1; run < max_runs; ++run) {
    if (live[run] == 0) continue;
    LineView<T> candidate;
    if (!LoadLine(run, low, high, endog_grid, value, n_candidates,
                  &candidate)) {
      return false;
    }
    bool better = false;
    if (!BetterAt(candidate, owner, left, &better)) return false;
    if (better) owner = candidate;
  }
  owners[0] = owner.run;
  T owned_from = left;
  int32_t pieces = 1;

  for (; pieces < max_runs; ++pieces) {
    bool found = false;
    T earliest = right;
    for (int32_t run = 0; run < max_runs; ++run) {
      if (live[run] == 0 || run == owner.run) continue;
      LineView<T> rival;
      if (!LoadLine(run, low, high, endog_grid, value, n_candidates,
                    &rival)) {
        return false;
      }
      const int32_t at_right = CompareLinesAt(owner, rival, right);
      if (at_right == kUnresolved) return false;
      if (at_right >= 0) continue;
      T event;
      if (!ExactHandoverState(owner.x0, owner.x1, owner.v0, owner.v1,
                              rival.x0, rival.x1, rival.v0, rival.v1,
                              owned_from, right, &event)) {
        return false;
      }
      if (!found || ExactGreater(earliest, event)) {
        earliest = event;
        found = true;
      }
    }
    if (!found) break;

    LineView<T> successor;
    bool have_successor = false;
    for (int32_t run = 0; run < max_runs; ++run) {
      if (live[run] == 0) continue;
      LineView<T> candidate;
      if (!LoadLine(run, low, high, endog_grid, value, n_candidates,
                    &candidate)) {
        return false;
      }
      if (!have_successor) {
        successor = candidate;
        have_successor = true;
        continue;
      }
      bool better = false;
      if (!BetterAt(candidate, successor, earliest, &better)) return false;
      if (better) successor = candidate;
    }
    if (!have_successor || successor.run == owner.run) return false;
    const int32_t slope_order = CompareLineSlopes(successor, owner);
    if (slope_order == kUnresolved || slope_order <= 0) return false;

    bounds[pieces] = earliest;
    owners[pieces] = successor.run;
    owner = successor;
    owned_from = earliest;
  }

  for (int32_t i = pieces; i < max_runs; ++i) {
    bounds[i] = right;
    owners[i] = owner.run;
  }
  bounds[max_runs] = right;
  return true;
}

}  // namespace pylcm_exact_affine

#undef PYLCM_HULL_HD
#undef PYLCM_HULL_INLINE

#endif  // PYLCM_EXACT_CELL_HULL_CORE_H_
