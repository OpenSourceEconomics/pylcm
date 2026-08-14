#include <climits>
#include <cstdint>
#include <string>

#include <cuda_runtime.h>
#include "exact_affine_core.h"
#include "exact_cell_hull_core.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;
namespace core = pylcm_exact_affine;

namespace {

template <typename T>
__global__ void CompareKernel(int64_t n, const T* ax0, const T* ax1,
                              const T* av0, const T* av1, const T* bx0,
                              const T* bx1, const T* bv0, const T* bv1,
                              const T* query, int32_t* output) {
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n; i += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    output[i] = core::CertifiedAffineCompare(
        ax0[i], ax1[i], av0[i], av1[i], bx0[i], bx1[i], bv0[i], bv1[i],
        query[i]);
  }
}

template <typename T>
__global__ void ReadKernel(int64_t n, const T* x0, const T* x1, const T* v0,
                           const T* v1, const T* query, T* output,
                           int32_t* status) {
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n; i += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    T value;
    if (core::ExactAffineRead(x0[i], x1[i], v0[i], v1[i], query[i],
                              &value)) {
      output[i] = value;
      status[i] = 0;
    } else {
      output[i] = core::QuietNaN<T>();
      status[i] = core::kUnresolved;
    }
  }
}


template <typename T>
__global__ void HandoverKernel(
    int64_t n, const T* ax0, const T* ax1, const T* av0, const T* av1,
    const T* bx0, const T* bx1, const T* bv0, const T* bv1,
    const T* left, const T* right, T* output, int32_t* status) {
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n; i += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    T state;
    if (core::ExactHandoverState(ax0[i], ax1[i], av0[i], av1[i], bx0[i],
                                 bx1[i], bv0[i], bv1[i], left[i], right[i],
                                 &state)) {
      output[i] = state;
      status[i] = 0;
    } else {
      output[i] = core::QuietNaN<T>();
      status[i] = core::kUnresolved;
    }
  }
}

template <typename T>
__global__ void CellHullKernel(
    int64_t batch, int32_t max_runs, int32_t n_candidates,
    const T* left, const T* right, const int32_t* live,
    const int32_t* low, const int32_t* high, const T* endog_grid,
    const T* value, T* bounds, int32_t* owners, int32_t* status) {
  for (int64_t cell =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       cell < batch;
       cell += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const bool ok = core::ExactCellHull(
        left[cell], right[cell], live + cell * max_runs,
        low + cell * max_runs, high + cell * max_runs, max_runs,
        endog_grid + cell * n_candidates, value + cell * n_candidates,
        n_candidates, bounds + cell * (max_runs + 1),
        owners + cell * max_runs);
    status[cell] = ok ? 0 : core::kUnresolved;
  }
}

inline ffi::Error LaunchError(const char* operation) {
  const cudaError_t error = cudaGetLastError();
  if (error == cudaSuccess) return ffi::Error::Success();
  return ffi::Error::Internal(std::string(operation) + ": " +
                              cudaGetErrorString(error));
}

template <typename T, ffi::DataType DType>
ffi::Error CompareImpl(cudaStream_t stream, ffi::Buffer<DType> ax0,
                       ffi::Buffer<DType> ax1, ffi::Buffer<DType> av0,
                       ffi::Buffer<DType> av1, ffi::Buffer<DType> bx0,
                       ffi::Buffer<DType> bx1, ffi::Buffer<DType> bv0,
                       ffi::Buffer<DType> bv1, ffi::Buffer<DType> query,
                       ffi::ResultBuffer<ffi::S32> output) {
  const int64_t n = static_cast<int64_t>(ax0.element_count());
  const size_t sizes[] = {ax1.element_count(), av0.element_count(),
                          av1.element_count(), bx0.element_count(),
                          bx1.element_count(), bv0.element_count(),
                          bv1.element_count(), query.element_count(),
                          (*output).element_count()};
  for (size_t size : sizes) {
    if (size != static_cast<size_t>(n)) {
      return ffi::Error::InvalidArgument(
          "all certified-affine comparator buffers must match");
    }
  }
  if (n == 0) return ffi::Error::Success();
  constexpr int threads = 128;
  const int blocks = static_cast<int>((n + threads - 1) / threads);
  CompareKernel<<<blocks, threads, 0, stream>>>(
      n, ax0.typed_data(), ax1.typed_data(), av0.typed_data(), av1.typed_data(),
      bx0.typed_data(), bx1.typed_data(), bv0.typed_data(), bv1.typed_data(),
      query.typed_data(), (*output).typed_data());
  return LaunchError("certified affine compare launch failed");
}

template <typename T, ffi::DataType DType>
ffi::Error ReadImpl(cudaStream_t stream, ffi::Buffer<DType> x0,
                    ffi::Buffer<DType> x1, ffi::Buffer<DType> v0,
                    ffi::Buffer<DType> v1, ffi::Buffer<DType> query,
                    ffi::ResultBuffer<DType> output,
                    ffi::ResultBuffer<ffi::S32> status) {
  const int64_t n = static_cast<int64_t>(x0.element_count());
  const size_t sizes[] = {x1.element_count(), v0.element_count(),
                          v1.element_count(), query.element_count(),
                          (*output).element_count(),
                          (*status).element_count()};
  for (size_t size : sizes) {
    if (size != static_cast<size_t>(n)) {
      return ffi::Error::InvalidArgument(
          "all exact affine-read buffers must match");
    }
  }
  if (n == 0) return ffi::Error::Success();
  constexpr int threads = 128;
  const int blocks = static_cast<int>((n + threads - 1) / threads);
  ReadKernel<<<blocks, threads, 0, stream>>>(
      n, x0.typed_data(), x1.typed_data(), v0.typed_data(), v1.typed_data(),
      query.typed_data(), (*output).typed_data(), (*status).typed_data());
  return LaunchError("exact affine read launch failed");
}


template <typename T, ffi::DataType DType>
ffi::Error HandoverImpl(
    cudaStream_t stream, ffi::Buffer<DType> ax0, ffi::Buffer<DType> ax1,
    ffi::Buffer<DType> av0, ffi::Buffer<DType> av1,
    ffi::Buffer<DType> bx0, ffi::Buffer<DType> bx1,
    ffi::Buffer<DType> bv0, ffi::Buffer<DType> bv1,
    ffi::Buffer<DType> left, ffi::Buffer<DType> right,
    ffi::ResultBuffer<DType> output,
    ffi::ResultBuffer<ffi::S32> status) {
  const int64_t n = static_cast<int64_t>(ax0.element_count());
  const size_t sizes[] = {ax1.element_count(), av0.element_count(),
                          av1.element_count(), bx0.element_count(),
                          bx1.element_count(), bv0.element_count(),
                          bv1.element_count(), left.element_count(),
                          right.element_count(), (*output).element_count(),
                          (*status).element_count()};
  for (size_t size : sizes) {
    if (size != static_cast<size_t>(n)) {
      return ffi::Error::InvalidArgument(
          "all exact handover buffers must match");
    }
  }
  if (n == 0) return ffi::Error::Success();
  constexpr int threads = 64;
  const int blocks = static_cast<int>((n + threads - 1) / threads);
  HandoverKernel<<<blocks, threads, 0, stream>>>(
      n, ax0.typed_data(), ax1.typed_data(), av0.typed_data(), av1.typed_data(),
      bx0.typed_data(), bx1.typed_data(), bv0.typed_data(), bv1.typed_data(),
      left.typed_data(), right.typed_data(), (*output).typed_data(),
      (*status).typed_data());
  return LaunchError("exact affine handover launch failed");
}

template <typename T, ffi::DataType DType>
ffi::Error CellHullImpl(
    cudaStream_t stream, ffi::Buffer<DType> left,
    ffi::Buffer<DType> right, ffi::Buffer<ffi::S32> live,
    ffi::Buffer<ffi::S32> low, ffi::Buffer<ffi::S32> high,
    ffi::Buffer<DType> endog_grid, ffi::Buffer<DType> value,
    ffi::ResultBuffer<DType> bounds,
    ffi::ResultBuffer<ffi::S32> owners,
    ffi::ResultBuffer<ffi::S32> status) {
  const size_t batch_size = left.element_count();
  if (right.element_count() != batch_size ||
      (*status).element_count() != batch_size) {
    return ffi::Error::InvalidArgument(
        "cell-hull scalar buffers must share a batch size");
  }
  if (batch_size == 0) return ffi::Error::Success();
  if (live.element_count() % batch_size != 0 ||
      low.element_count() != live.element_count() ||
      high.element_count() != live.element_count() ||
      (*owners).element_count() != live.element_count()) {
    return ffi::Error::InvalidArgument(
        "cell-hull run buffers must share batch*max_runs elements");
  }
  const size_t max_runs_size = live.element_count() / batch_size;
  if (max_runs_size == 0 ||
      (*bounds).element_count() != batch_size * (max_runs_size + 1)) {
    return ffi::Error::InvalidArgument(
        "cell-hull bounds must have max_runs+1 entries per cell");
  }
  if (endog_grid.element_count() % batch_size != 0 ||
      value.element_count() != endog_grid.element_count()) {
    return ffi::Error::InvalidArgument(
        "cell-hull candidate buffers must share batch*n_candidates elements");
  }
  const size_t n_candidates_size = endog_grid.element_count() / batch_size;
  if (n_candidates_size == 0 || max_runs_size > INT32_MAX ||
      n_candidates_size > INT32_MAX) {
    return ffi::Error::InvalidArgument("cell-hull dimensions are out of range");
  }
  const int64_t batch = static_cast<int64_t>(batch_size);
  const int32_t max_runs = static_cast<int32_t>(max_runs_size);
  const int32_t n_candidates = static_cast<int32_t>(n_candidates_size);
  constexpr int threads = 32;
  const int blocks = static_cast<int>((batch + threads - 1) / threads);
  CellHullKernel<<<blocks, threads, 0, stream>>>(
      batch, max_runs, n_candidates, left.typed_data(), right.typed_data(),
      live.typed_data(), low.typed_data(), high.typed_data(),
      endog_grid.typed_data(), value.typed_data(), (*bounds).typed_data(),
      (*owners).typed_data(), (*status).typed_data());
  return LaunchError("exact cell hull launch failed");
}

ffi::Error CompareF32Impl(cudaStream_t stream, ffi::Buffer<ffi::F32> ax0,
                          ffi::Buffer<ffi::F32> ax1,
                          ffi::Buffer<ffi::F32> av0,
                          ffi::Buffer<ffi::F32> av1,
                          ffi::Buffer<ffi::F32> bx0,
                          ffi::Buffer<ffi::F32> bx1,
                          ffi::Buffer<ffi::F32> bv0,
                          ffi::Buffer<ffi::F32> bv1,
                          ffi::Buffer<ffi::F32> query,
                          ffi::ResultBuffer<ffi::S32> output) {
  return CompareImpl<float, ffi::F32>(stream, ax0, ax1, av0, av1, bx0, bx1,
                                       bv0, bv1, query, output);
}

ffi::Error CompareF64Impl(cudaStream_t stream, ffi::Buffer<ffi::F64> ax0,
                          ffi::Buffer<ffi::F64> ax1,
                          ffi::Buffer<ffi::F64> av0,
                          ffi::Buffer<ffi::F64> av1,
                          ffi::Buffer<ffi::F64> bx0,
                          ffi::Buffer<ffi::F64> bx1,
                          ffi::Buffer<ffi::F64> bv0,
                          ffi::Buffer<ffi::F64> bv1,
                          ffi::Buffer<ffi::F64> query,
                          ffi::ResultBuffer<ffi::S32> output) {
  return CompareImpl<double, ffi::F64>(stream, ax0, ax1, av0, av1, bx0, bx1,
                                        bv0, bv1, query, output);
}

ffi::Error ReadF32Impl(cudaStream_t stream, ffi::Buffer<ffi::F32> x0,
                       ffi::Buffer<ffi::F32> x1,
                       ffi::Buffer<ffi::F32> v0,
                       ffi::Buffer<ffi::F32> v1,
                       ffi::Buffer<ffi::F32> query,
                       ffi::ResultBuffer<ffi::F32> output,
                       ffi::ResultBuffer<ffi::S32> status) {
  return ReadImpl<float, ffi::F32>(stream, x0, x1, v0, v1, query, output,
                                    status);
}

ffi::Error ReadF64Impl(cudaStream_t stream, ffi::Buffer<ffi::F64> x0,
                       ffi::Buffer<ffi::F64> x1,
                       ffi::Buffer<ffi::F64> v0,
                       ffi::Buffer<ffi::F64> v1,
                       ffi::Buffer<ffi::F64> query,
                       ffi::ResultBuffer<ffi::F64> output,
                       ffi::ResultBuffer<ffi::S32> status) {
  return ReadImpl<double, ffi::F64>(stream, x0, x1, v0, v1, query, output,
                                     status);
}


ffi::Error HandoverF32Impl(
    cudaStream_t stream, ffi::Buffer<ffi::F32> ax0,
    ffi::Buffer<ffi::F32> ax1, ffi::Buffer<ffi::F32> av0,
    ffi::Buffer<ffi::F32> av1, ffi::Buffer<ffi::F32> bx0,
    ffi::Buffer<ffi::F32> bx1, ffi::Buffer<ffi::F32> bv0,
    ffi::Buffer<ffi::F32> bv1, ffi::Buffer<ffi::F32> left,
    ffi::Buffer<ffi::F32> right, ffi::ResultBuffer<ffi::F32> output,
    ffi::ResultBuffer<ffi::S32> status) {
  return HandoverImpl<float, ffi::F32>(stream, ax0, ax1, av0, av1, bx0, bx1,
                                       bv0, bv1, left, right, output, status);
}

ffi::Error HandoverF64Impl(
    cudaStream_t stream, ffi::Buffer<ffi::F64> ax0,
    ffi::Buffer<ffi::F64> ax1, ffi::Buffer<ffi::F64> av0,
    ffi::Buffer<ffi::F64> av1, ffi::Buffer<ffi::F64> bx0,
    ffi::Buffer<ffi::F64> bx1, ffi::Buffer<ffi::F64> bv0,
    ffi::Buffer<ffi::F64> bv1, ffi::Buffer<ffi::F64> left,
    ffi::Buffer<ffi::F64> right, ffi::ResultBuffer<ffi::F64> output,
    ffi::ResultBuffer<ffi::S32> status) {
  return HandoverImpl<double, ffi::F64>(stream, ax0, ax1, av0, av1, bx0, bx1,
                                        bv0, bv1, left, right, output, status);
}

ffi::Error CellHullF32Impl(
    cudaStream_t stream, ffi::Buffer<ffi::F32> left,
    ffi::Buffer<ffi::F32> right, ffi::Buffer<ffi::S32> live,
    ffi::Buffer<ffi::S32> low, ffi::Buffer<ffi::S32> high,
    ffi::Buffer<ffi::F32> endog_grid, ffi::Buffer<ffi::F32> value,
    ffi::ResultBuffer<ffi::F32> bounds,
    ffi::ResultBuffer<ffi::S32> owners,
    ffi::ResultBuffer<ffi::S32> status) {
  return CellHullImpl<float, ffi::F32>(stream, left, right, live, low, high,
                                       endog_grid, value, bounds, owners,
                                       status);
}

ffi::Error CellHullF64Impl(
    cudaStream_t stream, ffi::Buffer<ffi::F64> left,
    ffi::Buffer<ffi::F64> right, ffi::Buffer<ffi::S32> live,
    ffi::Buffer<ffi::S32> low, ffi::Buffer<ffi::S32> high,
    ffi::Buffer<ffi::F64> endog_grid, ffi::Buffer<ffi::F64> value,
    ffi::ResultBuffer<ffi::F64> bounds,
    ffi::ResultBuffer<ffi::S32> owners,
    ffi::ResultBuffer<ffi::S32> status) {
  return CellHullImpl<double, ffi::F64>(stream, left, right, live, low, high,
                                        endog_grid, value, bounds, owners,
                                        status);
}

}  // namespace

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    CertifiedAffineCompareF32, CompareF32Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
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
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
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
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
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
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Ret<ffi::Buffer<ffi::F64>>()
        .Ret<ffi::Buffer<ffi::S32>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    ExactAffineHandoverF32, HandoverF32Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::S32>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    ExactAffineHandoverF64, HandoverF64Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Ret<ffi::Buffer<ffi::F64>>()
        .Ret<ffi::Buffer<ffi::S32>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    ExactCellHullF32, CellHullF32Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::S32>>()
        .Ret<ffi::Buffer<ffi::S32>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    ExactCellHullF64, CellHullF64Impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Ret<ffi::Buffer<ffi::F64>>()
        .Ret<ffi::Buffer<ffi::S32>>()
        .Ret<ffi::Buffer<ffi::S32>>());
