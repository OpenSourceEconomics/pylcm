#ifndef PYLCM_EXACT_HERMITE_CRITICAL_CORE_H_
#define PYLCM_EXACT_HERMITE_CRITICAL_CORE_H_

#include <new>

#include "exact_affine_core.h"

#ifdef __CUDACC__
#define PYLCM_CRITICAL_HD __host__ __device__
#define PYLCM_CRITICAL_INLINE __forceinline__
#else
#define PYLCM_CRITICAL_HD
#define PYLCM_CRITICAL_INLINE inline
#endif

namespace pylcm_exact_affine {

template <typename T>
struct CriticalTraits;

template <>
struct CriticalTraits<float> {
  static constexpr int kLimbs = 35;
};

template <>
struct CriticalTraits<double> {
  static constexpr int kLimbs = 263;
};

// Four finite operands and coefficient weight <=435 need
// 4*(emax+1-emin_sub)+9 bits: 1117 for float, 8401 for double.
PYLCM_CRITICAL_HD PYLCM_CRITICAL_INLINE void Product4(
    uint64_t a, uint64_t b, uint64_t c, uint64_t d, uint32_t words[8]) {
  uint32_t first[6]{}, digits[12]{}, dd[4]{}, result[16]{};
  Product3(a, b, c, first);
  for (int i = 0; i < 6; ++i) {
    digits[2 * i] = first[i] & 0xffffu;
    digits[2 * i + 1] = first[i] >> 16;
  }
  SplitBase16(d, dd);
  MulBase16(digits, 12, dd, 4, result, 16);
  for (int i = 0; i < 8; ++i) {
    words[i] = result[2 * i] | (result[2 * i + 1] << 16);
  }
}

template <int N>
PYLCM_CRITICAL_HD PYLCM_CRITICAL_INLINE bool AccumulateCriticalProduct(
    BigUInt<N>* positive, BigUInt<N>* negative,
    const FloatParts& a, const FloatParts& b,
    const FloatParts& c, const FloatParts& d,
    int coefficient, int base) {
  if (coefficient < -12 || coefficient > 12) return false;
  if (a.significand == 0 || b.significand == 0 ||
      c.significand == 0 || d.significand == 0) return true;
  uint32_t words[8]{};
  Product4(a.significand, b.significand, c.significand, d.significand,
           words);
  const bool sign = a.negative ^ b.negative ^ c.negative ^ d.negative ^
                    (coefficient < 0);
  BigUInt<N>* target = sign ? negative : positive;
  const int count = coefficient < 0 ? -coefficient : coefficient;
  const int shift = a.exponent + b.exponent + c.exponent + d.exponent - base;
  for (int i = 0; i < count; ++i) {
    if (!AddShiftedWords(target, words, 8, shift)) return false;
  }
  return true;
}

// p = (x0,h,v0,v1,a,b,L,A,lower,upper); R=h^3*((m-L)*H'(m)-A).
template <int N>
PYLCM_CRITICAL_HD PYLCM_CRITICAL_INLINE bool AccumulateCriticalR(
    const FloatParts* p, const FloatParts& m, int scale, int base,
    BigUInt<N>* positive, BigUInt<N>* negative) {
  const FloatParts y[2] = {m, p[6]};
  const FloatParts z[2] = {m, p[0]};
  const int sign[2] = {1, -1};
  const int k0[3] = {3, 2, 4}, c0[3] = {1, -1, 1};
  const int k1[2] = {5, 4}, c1[2] = {2, -4};
  const int k2[2] = {4, 5}, c2[2] = {3, -3};
  bool ok = true;
  for (int i = 0; i < 2; ++i) {
    for (int k = 0; k < 3; ++k) {
      ok &= AccumulateCriticalProduct(
          positive, negative, y[i], p[k0[k]], p[1], p[1],
          scale * sign[i] * c0[k], base);
    }
    for (int j = 0; j < 2; ++j) {
      for (int k = 0; k < 2; ++k) {
        ok &= AccumulateCriticalProduct(
            positive, negative, y[i], p[k1[k]], z[j], p[1],
            scale * sign[i] * sign[j] * c1[k], base);
      }
      for (int j2 = 0; j2 < 2; ++j2) {
        for (int k = 0; k < 2; ++k) {
          ok &= AccumulateCriticalProduct(
              positive, negative, y[i], p[k2[k]], z[j], z[j2],
              scale * sign[i] * sign[j] * sign[j2] * c2[k], base);
        }
      }
    }
  }
  ok &= AccumulateCriticalProduct(
      positive, negative, p[7], p[1], p[1], p[1], -scale, base);
  return ok;
}

// Accumulate (upper-lower)*R'(m), without rounded differences.
template <int N>
PYLCM_CRITICAL_HD PYLCM_CRITICAL_INLINE bool AccumulateCriticalDerivativeWidth(
    const FloatParts* p, const FloatParts& m, int scale, int base,
    BigUInt<N>* positive, BigUInt<N>* negative) {
  const FloatParts d[2] = {p[9], p[8]};
  const FloatParts y[2] = {m, p[6]}, z[2] = {m, p[0]};
  const int sign[2] = {1, -1};
  const int k0[3] = {3, 2, 4}, c0[3] = {1, -1, 1};
  const int k1[2] = {5, 4}, c1[2] = {2, -4};
  const int k2[2] = {4, 5}, c2[2] = {3, -3};
  bool ok = true;
  for (int i = 0; i < 2; ++i) {
    for (int k = 0; k < 3; ++k) {
      ok &= AccumulateCriticalProduct(
          positive, negative, d[i], p[k0[k]], p[1], p[1],
          scale * sign[i] * c0[k], base);
    }
    for (int j = 0; j < 2; ++j) {
      for (int k = 0; k < 2; ++k) {
        const int coefficient = scale * sign[i] * sign[j] * c1[k];
        ok &= AccumulateCriticalProduct(
            positive, negative, d[i], p[k1[k]], z[j], p[1],
            coefficient, base);
        ok &= AccumulateCriticalProduct(
            positive, negative, d[i], p[k1[k]], y[j], p[1],
            coefficient, base);
      }
      for (int j2 = 0; j2 < 2; ++j2) {
        for (int k = 0; k < 2; ++k) {
          const int coefficient =
              scale * sign[i] * sign[j] * sign[j2] * c2[k];
          ok &= AccumulateCriticalProduct(
              positive, negative, d[i], p[k2[k]], z[j], z[j2],
              coefficient, base);
          ok &= AccumulateCriticalProduct(
              positive, negative, d[i], p[k2[k]], y[j], z[j2],
              2 * coefficient, base);
        }
      }
    }
  }
  return ok;
}

template <typename T>
PYLCM_CRITICAL_HD PYLCM_CRITICAL_INLINE int32_t HermiteCriticalBernstein(
    T x0, T h, T v0, T v1, T a, T b, T lower_limit, T log_scale,
    T lower, T upper, uint32_t* workspace, int32_t signs[4]) {
  constexpr int N = CriticalTraits<T>::kLimbs;
  constexpr int base = 4 * FloatTraits<T>::kMinSubnormalExponent;
  static_assert(sizeof(BigUInt<N>) == N * sizeof(uint32_t));
  auto* positive = ::new (static_cast<void*>(workspace)) BigUInt<N>;
  auto* negative = ::new (static_cast<void*>(workspace + N)) BigUInt<N>;
  Clear(positive);
  Clear(negative);
  for (int i = 0; i < 4; ++i) signs[i] = 0;
  const FloatParts p[10] = {
      Decode(x0), Decode(h), Decode(v0), Decode(v1), Decode(a),
      Decode(b), Decode(lower_limit), Decode(log_scale),
      Decode(lower), Decode(upper)};
  for (const auto& value : p) {
    if (!value.finite) return kUnresolved;
  }
  // Decoded integers keep positivity independent of floating-point flush modes.
  if (p[1].negative || p[1].significand == 0 ||
      p[7].negative || p[7].significand == 0 ||
      !ExactGreater(upper, lower) || ExactGreater(x0, lower)) {
    return kUnresolved;
  }
  // upper-x0<=h checks the unclamped normalized piece without float addition.
  bool ok = AccumulateSingle(positive, negative, p[9], 1, base);
  ok &= AccumulateSingle(positive, negative, p[0], -1, base);
  ok &= AccumulateSingle(positive, negative, p[1], -1, base);
  if (!ok || Compare(*positive, *negative) > 0) return kUnresolved;
  for (int i = 0; i < 4; ++i) {
    Clear(positive);
    Clear(negative);
    const FloatParts& m = i < 2 ? p[8] : p[9];
    ok = AccumulateCriticalR(p, m, 3, base, positive, negative);
    if (i == 1 || i == 2) {
      ok &= AccumulateCriticalDerivativeWidth(
          p, m, i == 1 ? 1 : -1, base, positive, negative);
    }
    if (!ok) {
      for (int j = 0; j < 4; ++j) signs[j] = 0;
      return kUnresolved;
    }
    signs[i] = static_cast<int32_t>(Compare(*positive, *negative));
  }
  return 0;
}

}  // namespace pylcm_exact_affine

#undef PYLCM_CRITICAL_HD
#undef PYLCM_CRITICAL_INLINE

#endif  // PYLCM_EXACT_HERMITE_CRITICAL_CORE_H_
