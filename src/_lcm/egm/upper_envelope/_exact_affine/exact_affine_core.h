#ifndef PYLCM_EXACT_AFFINE_CORE_H_
#define PYLCM_EXACT_AFFINE_CORE_H_

#include <cstddef>
#include <cstdint>

#ifdef _MSC_VER
#include <intrin.h>
#endif

#ifdef __CUDACC__
#define PYLCM_HD __host__ __device__
#define PYLCM_INLINE __forceinline__
#else
#define PYLCM_HD
#define PYLCM_INLINE inline
#endif

namespace pylcm_exact_affine {

constexpr int32_t kUnresolved = 2;

template <typename T>
struct FloatTraits;

template <>
struct FloatTraits<float> {
  using UInt = uint32_t;
  static constexpr int kFractionBits = 23;
  static constexpr int kPrecision = 24;
  static constexpr int kBias = 127;
  static constexpr int kEmin = -126;
  static constexpr int kEmax = 127;
  static constexpr int kMinSubnormalExponent = -149;
  static constexpr UInt kExponentMask = 0xffu;
  static constexpr UInt kFractionMask = 0x7fffffu;
  static constexpr UInt kSignMask = 0x80000000u;
  static constexpr int kCompareLimbs = 28;
  static constexpr int kReadLimbs = 18;
};

template <>
struct FloatTraits<double> {
  using UInt = uint64_t;
  static constexpr int kFractionBits = 52;
  static constexpr int kPrecision = 53;
  static constexpr int kBias = 1023;
  static constexpr int kEmin = -1022;
  static constexpr int kEmax = 1023;
  static constexpr int kMinSubnormalExponent = -1074;
  static constexpr UInt kExponentMask = 0x7ffull;
  static constexpr UInt kFractionMask = 0x000fffffffffffffull;
  static constexpr UInt kSignMask = 0x8000000000000000ull;
  static constexpr int kCompareLimbs = 198;
  static constexpr int kReadLimbs = 132;
};

template <typename To, typename From>
PYLCM_HD PYLCM_INLINE To BitCast(From value) {
  static_assert(sizeof(To) == sizeof(From));
  To output{};
  const unsigned char* source =
      reinterpret_cast<const unsigned char*>(&value);
  unsigned char* destination = reinterpret_cast<unsigned char*>(&output);
  for (size_t i = 0; i < sizeof(To); ++i) destination[i] = source[i];
  return output;
}

template <int N>
struct BigUInt {
  uint32_t limb[N];
};

template <int N>
PYLCM_HD PYLCM_INLINE void Clear(BigUInt<N>* x) {
  for (int i = 0; i < N; ++i) x->limb[i] = 0;
}

template <int N>
PYLCM_HD PYLCM_INLINE void Copy(const BigUInt<N>& src, BigUInt<N>* dst) {
  for (int i = 0; i < N; ++i) dst->limb[i] = src.limb[i];
}

template <int N>
PYLCM_HD PYLCM_INLINE bool IsZero(const BigUInt<N>& x) {
  uint32_t any = 0;
  for (int i = 0; i < N; ++i) any |= x.limb[i];
  return any == 0;
}

template <int N>
PYLCM_HD PYLCM_INLINE int Compare(const BigUInt<N>& a,
                                  const BigUInt<N>& b) {
  for (int i = N - 1; i >= 0; --i) {
    if (a.limb[i] > b.limb[i]) return 1;
    if (a.limb[i] < b.limb[i]) return -1;
  }
  return 0;
}

template <int N>
PYLCM_HD PYLCM_INLINE int BitLength(const BigUInt<N>& x) {
  for (int i = N - 1; i >= 0; --i) {
    uint32_t word = x.limb[i];
    if (word != 0) {
      // One count-leading-zeros per toolchain. `std::countl_zero` would say
      // this once, but it is C++20 and the CUDA translation unit is built at
      // C++17, so the host pass of a `.cu` would not compile.
#ifdef __CUDA_ARCH__
      return 32 * i + (32 - __clz(word));
#elif defined(_MSC_VER)
      // `_BitScanReverse` reports the index of the highest set bit, so the bit
      // length is one past it. `word` is non-zero here, which is what the
      // intrinsic requires.
      unsigned long highest_set_bit;
      _BitScanReverse(&highest_set_bit, word);
      return 32 * i + static_cast<int>(highest_set_bit) + 1;
#else
      return 32 * i + (32 - __builtin_clz(word));
#endif
    }
  }
  return 0;
}

template <int N>
PYLCM_HD PYLCM_INLINE uint32_t ShiftedLimb(const BigUInt<N>& x,
                                           int bit_shift,
                                           int output_limb) {
  const int word_shift = bit_shift >> 5;
  const int intra = bit_shift & 31;
  const int source = output_limb - word_shift;
  uint64_t value = 0;
  if (source >= 0 && source < N) {
    value |= static_cast<uint64_t>(x.limb[source]) << intra;
  }
  if (intra != 0 && source - 1 >= 0 && source - 1 < N) {
    value |= static_cast<uint64_t>(x.limb[source - 1]) >> (32 - intra);
  }
  return static_cast<uint32_t>(value);
}

template <int N>
PYLCM_HD PYLCM_INLINE int CompareShifted(const BigUInt<N>& a,
                                         const BigUInt<N>& b,
                                         int b_shift) {
  const int a_bits = BitLength(a);
  const int b_bits = BitLength(b);
  if (b_bits == 0) return a_bits == 0 ? 0 : 1;
  const int shifted_b_bits = b_bits + b_shift;
  if (a_bits > shifted_b_bits) return 1;
  if (a_bits < shifted_b_bits) return -1;
  for (int i = N - 1; i >= 0; --i) {
    const uint32_t right = ShiftedLimb(b, b_shift, i);
    if (a.limb[i] > right) return 1;
    if (a.limb[i] < right) return -1;
  }
  return 0;
}

template <int N>
PYLCM_HD PYLCM_INLINE bool AddShiftedWords(BigUInt<N>* dst,
                                           const uint32_t* words,
                                           int word_count,
                                           int bit_shift) {
  if (bit_shift < 0) return false;
  const int word_shift = bit_shift >> 5;
  const int intra = bit_shift & 31;
  uint64_t carry = 0;
  int out_index = word_shift;
  for (int i = 0; i < word_count; ++i, ++out_index) {
    if (out_index >= N) {
      if (carry != 0) return false;
      for (int remaining = i; remaining < word_count; ++remaining) {
        if (words[remaining] != 0) return false;
      }
      return true;
    }
    const uint64_t shifted =
        (static_cast<uint64_t>(words[i]) << intra) + carry;
    const uint64_t sum = static_cast<uint64_t>(dst->limb[out_index]) +
                         static_cast<uint32_t>(shifted);
    dst->limb[out_index] = static_cast<uint32_t>(sum);
    carry = (shifted >> 32) + (sum >> 32);
  }
  while (carry != 0) {
    if (out_index >= N) return false;
    const uint64_t sum = static_cast<uint64_t>(dst->limb[out_index]) +
                         static_cast<uint32_t>(carry);
    dst->limb[out_index] = static_cast<uint32_t>(sum);
    carry = (carry >> 32) + (sum >> 32);
    ++out_index;
  }
  return true;
}

template <int N>
PYLCM_HD PYLCM_INLINE bool AddShiftedU64(BigUInt<N>* dst, uint64_t value,
                                         int bit_shift) {
  const uint32_t words[2] = {static_cast<uint32_t>(value),
                             static_cast<uint32_t>(value >> 32)};
  return AddShiftedWords(dst, words, 2, bit_shift);
}

template <int N>
PYLCM_HD PYLCM_INLINE void Subtract(const BigUInt<N>& a,
                                    const BigUInt<N>& b,
                                    BigUInt<N>* out) {
  // Requires a >= b.
  uint64_t borrow = 0;
  for (int i = 0; i < N; ++i) {
    const uint64_t left = a.limb[i];
    const uint64_t right = static_cast<uint64_t>(b.limb[i]) + borrow;
    if (left >= right) {
      out->limb[i] = static_cast<uint32_t>(left - right);
      borrow = 0;
    } else {
      out->limb[i] = static_cast<uint32_t>((uint64_t{1} << 32) + left - right);
      borrow = 1;
    }
  }
}

template <int N>
PYLCM_HD PYLCM_INLINE void SubtractShifted(BigUInt<N>* a,
                                           const BigUInt<N>& b,
                                           int bit_shift) {
  // Requires *a >= b << bit_shift.
  uint64_t borrow = 0;
  for (int i = 0; i < N; ++i) {
    const uint64_t left = a->limb[i];
    const uint64_t right =
        static_cast<uint64_t>(ShiftedLimb(b, bit_shift, i)) + borrow;
    if (left >= right) {
      a->limb[i] = static_cast<uint32_t>(left - right);
      borrow = 0;
    } else {
      a->limb[i] =
          static_cast<uint32_t>((uint64_t{1} << 32) + left - right);
      borrow = 1;
    }
  }
}

template <int N>
PYLCM_HD PYLCM_INLINE bool ShiftLeft(const BigUInt<N>& x, int bit_shift,
                                     BigUInt<N>* out) {
  Clear(out);
  return AddShiftedWords(out, x.limb, N, bit_shift);
}

template <int N>
PYLCM_HD PYLCM_INLINE bool ShiftLeftOne(BigUInt<N>* x) {
  uint32_t carry = 0;
  for (int i = 0; i < N; ++i) {
    const uint32_t next = x->limb[i] >> 31;
    x->limb[i] = (x->limb[i] << 1) | carry;
    carry = next;
  }
  return carry == 0;
}

struct FloatParts {
  uint64_t significand;
  int exponent;
  bool negative;
  bool finite;
};

template <typename T>
PYLCM_HD PYLCM_INLINE FloatParts Decode(T value) {
  using Traits = FloatTraits<T>;
  using UInt = typename Traits::UInt;
  const UInt bits = BitCast<UInt>(value);
  const bool negative = (bits & Traits::kSignMask) != 0;
  const UInt exponent_field =
      (bits >> Traits::kFractionBits) & Traits::kExponentMask;
  const UInt fraction = bits & Traits::kFractionMask;
  if (exponent_field == Traits::kExponentMask) {
    return {0, 0, negative, false};
  }
  if (exponent_field == 0) {
    return {static_cast<uint64_t>(fraction),
            Traits::kMinSubnormalExponent, negative, true};
  }
  return {(uint64_t{1} << Traits::kFractionBits) |
              static_cast<uint64_t>(fraction),
          static_cast<int>(exponent_field) - Traits::kBias -
              Traits::kFractionBits,
          negative, true};
}

template <typename T>
PYLCM_HD PYLCM_INLINE bool ExactGreater(T left, T right) {
  using Traits = FloatTraits<T>;
  using UInt = typename Traits::UInt;
  const UInt left_bits = BitCast<UInt>(left);
  const UInt right_bits = BitCast<UInt>(right);
  const UInt left_mag = left_bits & ~Traits::kSignMask;
  const UInt right_mag = right_bits & ~Traits::kSignMask;
  if (left_mag == 0 && right_mag == 0) return false;
  const bool left_neg = (left_bits & Traits::kSignMask) != 0;
  const bool right_neg = (right_bits & Traits::kSignMask) != 0;
  if (left_neg != right_neg) return !left_neg;
  if (!left_neg) return left_mag > right_mag;
  return left_mag < right_mag;
}

template <typename T>
PYLCM_HD PYLCM_INLINE bool ExactEqual(T left, T right) {
  return !ExactGreater(left, right) && !ExactGreater(right, left);
}

template <typename T>
struct QueryLine {
  T x0;
  T x1;
  T v0;
  T v1;
  T upper;
};

template <typename T>
PYLCM_HD PYLCM_INLINE bool CanonicalQueryLine(
    T left_x, T right_x, T left_value, T right_value, QueryLine<T>* line) {
  const FloatParts px0 = Decode(left_x);
  const FloatParts px1 = Decode(right_x);
  const FloatParts pv0 = Decode(left_value);
  const FloatParts pv1 = Decode(right_value);
  if (!px0.finite || !px1.finite || !pv0.finite || !pv1.finite) {
    return false;
  }
  const bool descending = ExactGreater(left_x, right_x);
  const T lower = descending ? right_x : left_x;
  const T upper = descending ? left_x : right_x;
  const T at_lower = descending ? right_value : left_value;
  const T at_upper = descending ? left_value : right_value;
  line->upper = upper;
  if (ExactEqual(lower, upper)) {
    // A zero-width link is a self-bracket. Its value is the stored left
    // endpoint and it has no right extension or slope; representing it as a
    // flat unit-width line makes both exact comparisons say precisely that.
    line->x0 = T{0};
    line->x1 = T{1};
    line->v0 = left_value;
    line->v1 = left_value;
  } else {
    line->x0 = lower;
    line->x1 = upper;
    line->v0 = at_lower;
    line->v1 = at_upper;
  }
  return true;
}

PYLCM_HD PYLCM_INLINE void MulBase16(const uint32_t* a, int na,
                                     const uint32_t* b, int nb,
                                     uint32_t* out, int nout) {
  uint64_t accum[16]{};
  for (int i = 0; i < na; ++i) {
    for (int j = 0; j < nb; ++j) {
      accum[i + j] += static_cast<uint64_t>(a[i]) * b[j];
    }
  }
  uint64_t carry = 0;
  for (int k = 0; k < nout; ++k) {
    const uint64_t value = accum[k] + carry;
    out[k] = static_cast<uint32_t>(value & 0xffffu);
    carry = value >> 16;
  }
}

PYLCM_HD PYLCM_INLINE void SplitBase16(uint64_t value, uint32_t out[4]) {
  out[0] = static_cast<uint32_t>(value & 0xffffu);
  out[1] = static_cast<uint32_t>((value >> 16) & 0xffffu);
  out[2] = static_cast<uint32_t>((value >> 32) & 0xffffu);
  out[3] = static_cast<uint32_t>((value >> 48) & 0xffffu);
}

PYLCM_HD PYLCM_INLINE void Product2(uint64_t a, uint64_t b,
                                    uint32_t words[4]) {
  uint32_t ad[4], bd[4], digits[8]{};
  SplitBase16(a, ad);
  SplitBase16(b, bd);
  MulBase16(ad, 4, bd, 4, digits, 8);
  for (int i = 0; i < 4; ++i) {
    words[i] = digits[2 * i] | (digits[2 * i + 1] << 16);
  }
}

PYLCM_HD PYLCM_INLINE void Product3(uint64_t a, uint64_t b, uint64_t c,
                                    uint32_t words[6]) {
  uint32_t ad[4], bd[4], cd[4], first[8]{}, digits[12]{};
  SplitBase16(a, ad);
  SplitBase16(b, bd);
  SplitBase16(c, cd);
  MulBase16(ad, 4, bd, 4, first, 8);
  MulBase16(first, 8, cd, 4, digits, 12);
  for (int i = 0; i < 6; ++i) {
    words[i] = digits[2 * i] | (digits[2 * i + 1] << 16);
  }
}

template <int N>
PYLCM_HD PYLCM_INLINE bool AccumulateSingle(BigUInt<N>* positive,
                                            BigUInt<N>* negative,
                                            const FloatParts& x,
                                            int coefficient,
                                            int base_exponent) {
  if (x.significand == 0) return true;
  const bool is_negative = x.negative ^ (coefficient < 0);
  BigUInt<N>* target = is_negative ? negative : positive;
  return AddShiftedU64(target, x.significand, x.exponent - base_exponent);
}

template <int N>
PYLCM_HD PYLCM_INLINE bool AccumulateProduct2(BigUInt<N>* positive,
                                              BigUInt<N>* negative,
                                              const FloatParts& a,
                                              const FloatParts& b,
                                              int coefficient,
                                              int base_exponent) {
  if (a.significand == 0 || b.significand == 0) return true;
  uint32_t words[4]{};
  Product2(a.significand, b.significand, words);
  const bool is_negative = a.negative ^ b.negative ^ (coefficient < 0);
  BigUInt<N>* target = is_negative ? negative : positive;
  return AddShiftedWords(target, words, 4,
                         a.exponent + b.exponent - base_exponent);
}

template <int N>
PYLCM_HD PYLCM_INLINE bool AccumulateProduct3(BigUInt<N>* positive,
                                              BigUInt<N>* negative,
                                              const FloatParts& a,
                                              const FloatParts& b,
                                              const FloatParts& c,
                                              int coefficient,
                                              int base_exponent) {
  if (a.significand == 0 || b.significand == 0 || c.significand == 0) {
    return true;
  }
  uint32_t words[6]{};
  Product3(a.significand, b.significand, c.significand, words);
  const bool is_negative =
      a.negative ^ b.negative ^ c.negative ^ (coefficient < 0);
  BigUInt<N>* target = is_negative ? negative : positive;
  return AddShiftedWords(target, words, 6,
                         a.exponent + b.exponent + c.exponent - base_exponent);
}

template <int N>
PYLCM_HD PYLCM_INLINE bool ResolveSigned(const BigUInt<N>& positive,
                                         const BigUInt<N>& negative,
                                         BigUInt<N>* magnitude,
                                         bool* is_negative) {
  const int order = Compare(positive, negative);
  if (order >= 0) {
    Subtract(positive, negative, magnitude);
    *is_negative = false;
  } else {
    Subtract(negative, positive, magnitude);
    *is_negative = true;
  }
  return true;
}

template <int N>
PYLCM_HD PYLCM_INLINE int FloorLog2Ratio(const BigUInt<N>& a,
                                         const BigUInt<N>& b) {
  const int a_bits = BitLength(a);
  const int b_bits = BitLength(b);
  int exponent = a_bits - b_bits;
  if (exponent >= 0) {
    if (CompareShifted(a, b, exponent) < 0) --exponent;
  } else {
    if (CompareShifted(b, a, -exponent) > 0) --exponent;
  }
  return exponent;
}

template <int N>
PYLCM_HD PYLCM_INLINE bool DivideRoundedSmall(const BigUInt<N>& numerator,
                                              const BigUInt<N>& denominator,
                                              uint64_t* quotient) {
  BigUInt<N> remainder;
  Copy(numerator, &remainder);
  const int top = BitLength(numerator) - BitLength(denominator);
  if (top > 63) return false;
  uint64_t q = 0;
  for (int bit = top; bit >= 0; --bit) {
    if (CompareShifted(remainder, denominator, bit) >= 0) {
      SubtractShifted(&remainder, denominator, bit);
      q |= uint64_t{1} << bit;
    }
  }
  BigUInt<N> twice;
  Copy(remainder, &twice);
  if (!ShiftLeftOne(&twice)) return false;
  const int half_cmp = Compare(twice, denominator);
  if (half_cmp > 0 || (half_cmp == 0 && (q & 1u) != 0)) ++q;
  *quotient = q;
  return true;
}

template <typename T, int N>
PYLCM_HD PYLCM_INLINE bool RoundRatio(const BigUInt<N>& numerator,
                                      bool negative,
                                      int numerator_base,
                                      const BigUInt<N>& denominator,
                                      int denominator_base,
                                      T* result) {
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
    const int binary_shift =
        power - exponent + (Traits::kPrecision - 1);
    if (binary_shift > 0) return false;
    BigUInt<N> scaled_denominator;
    if (!ShiftLeft(denominator, -binary_shift, &scaled_denominator)) {
      return false;
    }
    uint64_t significand = 0;
    if (!DivideRoundedSmall(numerator, scaled_denominator, &significand)) {
      return false;
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
    if (binary_shift > 0) return false;
    BigUInt<N> scaled_denominator;
    if (!ShiftLeft(denominator, -binary_shift, &scaled_denominator)) {
      return false;
    }
    uint64_t fraction = 0;
    if (!DivideRoundedSmall(numerator, scaled_denominator, &fraction)) {
      return false;
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
PYLCM_HD PYLCM_INLINE int32_t CertifiedAffineCompare(
    T ax0, T ax1, T av0, T av1, T bx0, T bx1, T bv0, T bv1, T query) {
  using Traits = FloatTraits<T>;
  constexpr int N = Traits::kCompareLimbs;
  const FloatParts p[9] = {Decode(ax0), Decode(ax1), Decode(av0),
                           Decode(av1), Decode(bx0), Decode(bx1),
                           Decode(bv0), Decode(bv1), Decode(query)};
  for (int i = 0; i < 9; ++i) {
    if (!p[i].finite) return kUnresolved;
  }
  if (!ExactGreater(ax1, ax0) || !ExactGreater(bx1, bx0)) {
    return kUnresolved;
  }

  // N_A = av0*ax1 - av0*q + av1*q - av1*ax0.
  // N_B = bv0*bx1 - bv0*q + bv1*q - bv1*bx0.
  // D = N_A*(bx1-bx0) - N_B*(ax1-ax0).
  struct PairTerm {
    int coefficient;
    int first;
    int second;
  };
  const PairTerm a_num[4] = {{1, 2, 1}, {-1, 2, 8},
                             {1, 3, 8}, {-1, 3, 0}};
  const PairTerm b_num[4] = {{1, 6, 5}, {-1, 6, 8},
                             {1, 7, 8}, {-1, 7, 4}};
  const int a_width_index[2] = {1, 0};
  const int b_width_index[2] = {5, 4};
  const int width_coefficient[2] = {1, -1};

  BigUInt<N> positive, negative;
  Clear(&positive);
  Clear(&negative);
  constexpr int base = 3 * Traits::kMinSubnormalExponent;
  bool ok = true;
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 2; ++j) {
      ok &= AccumulateProduct3(
          &positive, &negative, p[a_num[i].first], p[a_num[i].second],
          p[b_width_index[j]], a_num[i].coefficient * width_coefficient[j],
          base);
    }
  }
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 2; ++j) {
      ok &= AccumulateProduct3(
          &positive, &negative, p[b_num[i].first], p[b_num[i].second],
          p[a_width_index[j]], -b_num[i].coefficient * width_coefficient[j],
          base);
    }
  }
  if (!ok) return kUnresolved;
  return static_cast<int32_t>(Compare(positive, negative));
}

template <typename T>
PYLCM_HD PYLCM_INLINE int32_t CertifiedSlopeCompare(
    T ax0, T ax1, T av0, T av1, T bx0, T bx1, T bv0, T bv1) {
  using Traits = FloatTraits<T>;
  constexpr int N = Traits::kReadLimbs;
  const FloatParts p[8] = {Decode(ax0), Decode(ax1), Decode(av0), Decode(av1),
                           Decode(bx0), Decode(bx1), Decode(bv0), Decode(bv1)};
  for (int i = 0; i < 8; ++i) {
    if (!p[i].finite) return kUnresolved;
  }
  if (!ExactGreater(ax1, ax0) || !ExactGreater(bx1, bx0)) {
    return kUnresolved;
  }

  // (av1-av0)/(ax1-ax0) - (bv1-bv0)/(bx1-bx0), with both
  // positive widths cross-multiplied before any division.
  struct PairTerm {
    int coefficient;
    int first;
    int second;
  };
  const PairTerm terms[8] = {
      {1, 3, 5}, {-1, 3, 4}, {-1, 2, 5}, {1, 2, 4},
      {-1, 7, 1}, {1, 7, 0}, {1, 6, 1}, {-1, 6, 0}};
  BigUInt<N> positive, negative;
  Clear(&positive);
  Clear(&negative);
  constexpr int base = 2 * Traits::kMinSubnormalExponent;
  bool ok = true;
  for (const PairTerm& term : terms) {
    ok &= AccumulateProduct2(&positive, &negative, p[term.first],
                             p[term.second], term.coefficient, base);
  }
  if (!ok) return kUnresolved;
  return static_cast<int32_t>(Compare(positive, negative));
}

template <typename T>
PYLCM_HD PYLCM_INLINE bool ExactQueryWinner(
    const T* left_grid, const T* right_grid, const T* left_value,
    const T* right_value, const int32_t* live,
    const int32_t* stable_index, int32_t n_segment, T query,
    int32_t* winner) {
  if (n_segment <= 0 || !Decode(query).finite) return false;

  int32_t held_index = -1;
  QueryLine<T> held{};
  for (int32_t index = 0; index < n_segment; ++index) {
    if (live[index] == 0) continue;

    // Establish support from the abscissae before reading channel values. A
    // non-finite value on a link that does not bracket this query is irrelevant;
    // a non-finite abscissa cannot establish that and therefore fails loud.
    if (!Decode(left_grid[index]).finite || !Decode(right_grid[index]).finite) {
      return false;
    }
    const T lower = ExactGreater(left_grid[index], right_grid[index])
                        ? right_grid[index]
                        : left_grid[index];
    const T upper = ExactGreater(left_grid[index], right_grid[index])
                        ? left_grid[index]
                        : right_grid[index];
    const bool brackets = !ExactGreater(lower, query) &&
                          !ExactGreater(query, upper);
    if (!brackets) continue;

    QueryLine<T> candidate;
    if (!CanonicalQueryLine(left_grid[index], right_grid[index],
                            left_value[index], right_value[index],
                            &candidate)) {
      return false;
    }

    if (held_index < 0) {
      held_index = index;
      held = candidate;
      continue;
    }

    const int32_t value_order = CertifiedAffineCompare(
        candidate.x0, candidate.x1, candidate.v0, candidate.v1, held.x0,
        held.x1, held.v0, held.v1, query);
    if (value_order == kUnresolved) return false;
    bool replace = value_order > 0;
    if (value_order == 0) {
      const bool candidate_right = ExactGreater(candidate.upper, query);
      const bool held_right = ExactGreater(held.upper, query);
      if (candidate_right != held_right) {
        replace = candidate_right;
      } else {
        const int32_t slope_order = CertifiedSlopeCompare(
            candidate.x0, candidate.x1, candidate.v0, candidate.v1, held.x0,
            held.x1, held.v0, held.v1);
        if (slope_order == kUnresolved) return false;
        replace = slope_order > 0;
        if (slope_order == 0) {
          // Operand position is not an identity: a blocked reduction may
          // reintroduce the standing winner anywhere in a later block.
          replace = stable_index[index] < stable_index[held_index];
        }
      }
    }
    if (replace) {
      held_index = index;
      held = candidate;
    }
  }

  if (held_index < 0) return false;
  *winner = held_index;
  return true;
}

template <typename T>
PYLCM_HD PYLCM_INLINE bool ExactAffineRead(T x0, T x1, T v0, T v1,
                                           T query, T* result) {
  using Traits = FloatTraits<T>;
  constexpr int N = Traits::kReadLimbs;
  const FloatParts px0 = Decode(x0);
  const FloatParts px1 = Decode(x1);
  const FloatParts pv0 = Decode(v0);
  const FloatParts pv1 = Decode(v1);
  const FloatParts pq = Decode(query);
  if (!px0.finite || !px1.finite || !pv0.finite || !pv1.finite ||
      !pq.finite || !ExactGreater(x1, x0)) {
    return false;
  }

  BigUInt<N> num_pos, num_neg;
  Clear(&num_pos);
  Clear(&num_neg);
  constexpr int numerator_base = 2 * Traits::kMinSubnormalExponent;
  bool ok = true;
  ok &= AccumulateProduct2(&num_pos, &num_neg, pv0, px1, 1,
                           numerator_base);
  ok &= AccumulateProduct2(&num_pos, &num_neg, pv0, pq, -1,
                           numerator_base);
  ok &= AccumulateProduct2(&num_pos, &num_neg, pv1, pq, 1,
                           numerator_base);
  ok &= AccumulateProduct2(&num_pos, &num_neg, pv1, px0, -1,
                           numerator_base);
  BigUInt<N> numerator;
  bool numerator_negative = false;
  ResolveSigned(num_pos, num_neg, &numerator, &numerator_negative);

  BigUInt<N> den_pos, den_neg;
  Clear(&den_pos);
  Clear(&den_neg);
  constexpr int denominator_base = Traits::kMinSubnormalExponent;
  ok &= AccumulateSingle(&den_pos, &den_neg, px1, 1, denominator_base);
  ok &= AccumulateSingle(&den_pos, &den_neg, px0, -1, denominator_base);
  BigUInt<N> denominator;
  bool denominator_negative = false;
  ResolveSigned(den_pos, den_neg, &denominator, &denominator_negative);
  if (!ok || denominator_negative || IsZero(denominator)) return false;
  return RoundRatio<T>(numerator, numerator_negative, numerator_base,
                       denominator, denominator_base, result);
}

template <typename T>
PYLCM_HD PYLCM_INLINE T QuietNaN() {
  using Traits = FloatTraits<T>;
  using UInt = typename Traits::UInt;
  const UInt bits = (Traits::kExponentMask << Traits::kFractionBits) |
                    (UInt{1} << (Traits::kFractionBits - 1));
  return BitCast<T>(bits);
}

}  // namespace pylcm_exact_affine

#undef PYLCM_HD
#undef PYLCM_INLINE

#endif  // PYLCM_EXACT_AFFINE_CORE_H_
