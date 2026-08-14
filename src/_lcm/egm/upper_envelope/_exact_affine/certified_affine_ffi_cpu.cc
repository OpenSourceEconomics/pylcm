#include <cstdint>

#include "exact_affine_core.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;
namespace core = pylcm_exact_affine;

namespace {

template <typename T, ffi::DataType DType>
ffi::Error CompareImpl(ffi::Buffer<DType> ax0, ffi::Buffer<DType> ax1,
                       ffi::Buffer<DType> av0, ffi::Buffer<DType> av1,
                       ffi::Buffer<DType> bx0, ffi::Buffer<DType> bx1,
                       ffi::Buffer<DType> bv0, ffi::Buffer<DType> bv1,
                       ffi::Buffer<DType> query,
                       ffi::ResultBuffer<ffi::S32> output) {
  const size_t n = ax0.element_count();
  const size_t sizes[] = {ax1.element_count(), av0.element_count(),
                           av1.element_count(), bx0.element_count(),
                           bx1.element_count(), bv0.element_count(),
                           bv1.element_count(), query.element_count(),
                           (*output).element_count()};
  for (size_t size : sizes) {
    if (size != n) {
      return ffi::Error::InvalidArgument(
          "all certified-affine comparator buffers must match");
    }
  }
  const T* p_ax0 = ax0.typed_data();
  const T* p_ax1 = ax1.typed_data();
  const T* p_av0 = av0.typed_data();
  const T* p_av1 = av1.typed_data();
  const T* p_bx0 = bx0.typed_data();
  const T* p_bx1 = bx1.typed_data();
  const T* p_bv0 = bv0.typed_data();
  const T* p_bv1 = bv1.typed_data();
  const T* p_query = query.typed_data();
  int32_t* out = (*output).typed_data();
  for (size_t i = 0; i < n; ++i) {
    out[i] = core::CertifiedAffineCompare(
        p_ax0[i], p_ax1[i], p_av0[i], p_av1[i], p_bx0[i], p_bx1[i],
        p_bv0[i], p_bv1[i], p_query[i]);
  }
  return ffi::Error::Success();
}

template <typename T, ffi::DataType DType>
ffi::Error ReadImpl(ffi::Buffer<DType> x0, ffi::Buffer<DType> x1,
                    ffi::Buffer<DType> v0, ffi::Buffer<DType> v1,
                    ffi::Buffer<DType> query,
                    ffi::ResultBuffer<DType> output,
                    ffi::ResultBuffer<ffi::S32> status) {
  const size_t n = x0.element_count();
  const size_t sizes[] = {x1.element_count(), v0.element_count(),
                           v1.element_count(), query.element_count(),
                           (*output).element_count(),
                           (*status).element_count()};
  for (size_t size : sizes) {
    if (size != n) {
      return ffi::Error::InvalidArgument(
          "all exact affine-read buffers must match");
    }
  }
  const T* p_x0 = x0.typed_data();
  const T* p_x1 = x1.typed_data();
  const T* p_v0 = v0.typed_data();
  const T* p_v1 = v1.typed_data();
  const T* p_query = query.typed_data();
  T* out = (*output).typed_data();
  int32_t* flags = (*status).typed_data();
  for (size_t i = 0; i < n; ++i) {
    T value;
    if (core::ExactAffineRead(p_x0[i], p_x1[i], p_v0[i], p_v1[i],
                              p_query[i], &value)) {
      out[i] = value;
      flags[i] = 0;
    } else {
      out[i] = core::QuietNaN<T>();
      flags[i] = core::kUnresolved;
    }
  }
  return ffi::Error::Success();
}

ffi::Error CompareF32Impl(ffi::Buffer<ffi::F32> ax0,
                          ffi::Buffer<ffi::F32> ax1,
                          ffi::Buffer<ffi::F32> av0,
                          ffi::Buffer<ffi::F32> av1,
                          ffi::Buffer<ffi::F32> bx0,
                          ffi::Buffer<ffi::F32> bx1,
                          ffi::Buffer<ffi::F32> bv0,
                          ffi::Buffer<ffi::F32> bv1,
                          ffi::Buffer<ffi::F32> query,
                          ffi::ResultBuffer<ffi::S32> output) {
  return CompareImpl<float, ffi::F32>(ax0, ax1, av0, av1, bx0, bx1, bv0,
                                       bv1, query, output);
}

ffi::Error CompareF64Impl(ffi::Buffer<ffi::F64> ax0,
                          ffi::Buffer<ffi::F64> ax1,
                          ffi::Buffer<ffi::F64> av0,
                          ffi::Buffer<ffi::F64> av1,
                          ffi::Buffer<ffi::F64> bx0,
                          ffi::Buffer<ffi::F64> bx1,
                          ffi::Buffer<ffi::F64> bv0,
                          ffi::Buffer<ffi::F64> bv1,
                          ffi::Buffer<ffi::F64> query,
                          ffi::ResultBuffer<ffi::S32> output) {
  return CompareImpl<double, ffi::F64>(ax0, ax1, av0, av1, bx0, bx1, bv0,
                                        bv1, query, output);
}

ffi::Error ReadF32Impl(ffi::Buffer<ffi::F32> x0,
                       ffi::Buffer<ffi::F32> x1,
                       ffi::Buffer<ffi::F32> v0,
                       ffi::Buffer<ffi::F32> v1,
                       ffi::Buffer<ffi::F32> query,
                       ffi::ResultBuffer<ffi::F32> output,
                       ffi::ResultBuffer<ffi::S32> status) {
  return ReadImpl<float, ffi::F32>(x0, x1, v0, v1, query, output, status);
}

ffi::Error ReadF64Impl(ffi::Buffer<ffi::F64> x0,
                       ffi::Buffer<ffi::F64> x1,
                       ffi::Buffer<ffi::F64> v0,
                       ffi::Buffer<ffi::F64> v1,
                       ffi::Buffer<ffi::F64> query,
                       ffi::ResultBuffer<ffi::F64> output,
                       ffi::ResultBuffer<ffi::S32> status) {
  return ReadImpl<double, ffi::F64>(x0, x1, v0, v1, query, output, status);
}

}  // namespace

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    CertifiedAffineCompareF32, CompareF32Impl,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::S32>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    CertifiedAffineCompareF64, CompareF64Impl,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Ret<ffi::Buffer<ffi::S32>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    ExactAffineReadF32, ReadF32Impl,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::S32>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    ExactAffineReadF64, ReadF64Impl,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Ret<ffi::Buffer<ffi::F64>>()
        .Ret<ffi::Buffer<ffi::S32>>());
