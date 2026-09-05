#include <algorithm>
#include <climits>
#include <cstdint>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

#include "exact_affine_core.h"
#include "exact_cell_hull_core.h"
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

template <typename T, ffi::DataType DType>
ffi::Error QueryWinnerImpl(ffi::Buffer<DType> left_grid,
                           ffi::Buffer<DType> right_grid,
                           ffi::Buffer<DType> left_value,
                           ffi::Buffer<DType> right_value,
                           ffi::Buffer<ffi::S32> live,
                           ffi::Buffer<ffi::S32> stable_index,
                           ffi::Buffer<DType> query,
                           ffi::ResultBuffer<ffi::S32> winner,
                           ffi::ResultBuffer<ffi::S32> status) {
  const size_t n_segment = left_grid.element_count();
  if (n_segment == 0 || n_segment > INT32_MAX ||
      right_grid.element_count() != n_segment ||
      left_value.element_count() != n_segment ||
      right_value.element_count() != n_segment ||
      live.element_count() != n_segment ||
      stable_index.element_count() != n_segment) {
    return ffi::Error::InvalidArgument(
        "exact-query segment buffers must be nonempty and match");
  }
  const size_t n_query = query.element_count();
  if ((*winner).element_count() != n_query ||
      (*status).element_count() != n_query) {
    return ffi::Error::InvalidArgument(
        "exact-query outputs must match the query buffer");
  }
  const T* p_left_grid = left_grid.typed_data();
  const T* p_right_grid = right_grid.typed_data();
  const T* p_left_value = left_value.typed_data();
  const T* p_right_value = right_value.typed_data();
  const int32_t* p_live = live.typed_data();
  const int32_t* p_stable_index = stable_index.typed_data();
  const T* p_query = query.typed_data();
  int32_t* p_winner = (*winner).typed_data();
  int32_t* p_status = (*status).typed_data();
  const int32_t count = static_cast<int32_t>(n_segment);
  for (size_t i = 0; i < n_query; ++i) {
    int32_t selected = 0;
    const bool ok = core::ExactQueryWinner(
        p_left_grid, p_right_grid, p_left_value, p_right_value, p_live,
        p_stable_index, count, p_query[i], &selected);
    p_winner[i] = selected;
    p_status[i] = ok ? 0 : core::kUnresolved;
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


ffi::Error QueryWinnerF32Impl(
    ffi::Buffer<ffi::F32> left_grid, ffi::Buffer<ffi::F32> right_grid,
    ffi::Buffer<ffi::F32> left_value, ffi::Buffer<ffi::F32> right_value,
    ffi::Buffer<ffi::S32> live,
    ffi::Buffer<ffi::S32> stable_index, ffi::Buffer<ffi::F32> query,
    ffi::ResultBuffer<ffi::S32> winner,
    ffi::ResultBuffer<ffi::S32> status) {
  return QueryWinnerImpl<float, ffi::F32>(
      left_grid, right_grid, left_value, right_value, live, stable_index,
      query, winner, status);
}

ffi::Error QueryWinnerF64Impl(
    ffi::Buffer<ffi::F64> left_grid, ffi::Buffer<ffi::F64> right_grid,
    ffi::Buffer<ffi::F64> left_value, ffi::Buffer<ffi::F64> right_value,
    ffi::Buffer<ffi::S32> live,
    ffi::Buffer<ffi::S32> stable_index, ffi::Buffer<ffi::F64> query,
    ffi::ResultBuffer<ffi::S32> winner,
    ffi::ResultBuffer<ffi::S32> status) {
  return QueryWinnerImpl<double, ffi::F64>(
      left_grid, right_grid, left_value, right_value, live, stable_index,
      query, winner, status);
}


template <typename T, ffi::DataType DType>
ffi::Error HandoverImpl(ffi::Buffer<DType> ax0, ffi::Buffer<DType> ax1,
                        ffi::Buffer<DType> av0, ffi::Buffer<DType> av1,
                        ffi::Buffer<DType> bx0, ffi::Buffer<DType> bx1,
                        ffi::Buffer<DType> bv0, ffi::Buffer<DType> bv1,
                        ffi::Buffer<DType> left, ffi::Buffer<DType> right,
                        ffi::ResultBuffer<DType> output,
                        ffi::ResultBuffer<ffi::S32> status) {
  const size_t n = ax0.element_count();
  const size_t sizes[] = {ax1.element_count(), av0.element_count(),
                          av1.element_count(), bx0.element_count(),
                          bx1.element_count(), bv0.element_count(),
                          bv1.element_count(), left.element_count(),
                          right.element_count(), (*output).element_count(),
                          (*status).element_count()};
  for (size_t size : sizes) {
    if (size != n) {
      return ffi::Error::InvalidArgument(
          "all exact handover buffers must match");
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
  const T* p_left = left.typed_data();
  const T* p_right = right.typed_data();
  T* out = (*output).typed_data();
  int32_t* flags = (*status).typed_data();
  for (size_t i = 0; i < n; ++i) {
    T state;
    if (core::ExactHandoverState(
            p_ax0[i], p_ax1[i], p_av0[i], p_av1[i], p_bx0[i], p_bx1[i],
            p_bv0[i], p_bv1[i], p_left[i], p_right[i], &state)) {
      out[i] = state;
      flags[i] = 0;
    } else {
      out[i] = core::QuietNaN<T>();
      flags[i] = core::kUnresolved;
    }
  }
  return ffi::Error::Success();
}

template <typename T, ffi::DataType DType>
ffi::Error CellHullImpl(
                        ffi::Buffer<DType> left,
                        ffi::Buffer<DType> right,
                        ffi::Buffer<ffi::S32> live,
                        ffi::Buffer<ffi::S32> low,
                        ffi::Buffer<ffi::S32> high,
                        ffi::Buffer<DType> endog_grid,
                        ffi::Buffer<DType> value,
                        ffi::ResultBuffer<DType> bounds,
                        ffi::ResultBuffer<ffi::S32> owners,
                        ffi::ResultBuffer<ffi::S32> status) {
  const size_t batch = left.element_count();
  if (right.element_count() != batch || (*status).element_count() != batch) {
    return ffi::Error::InvalidArgument(
        "cell-hull scalar buffers must share a batch size");
  }
  if (batch == 0) return ffi::Error::Success();
  if (live.element_count() % batch != 0 ||
      low.element_count() != live.element_count() ||
      high.element_count() != live.element_count() ||
      (*owners).element_count() != live.element_count()) {
    return ffi::Error::InvalidArgument(
        "cell-hull run buffers must share batch*max_runs elements");
  }
  const size_t max_runs_size = live.element_count() / batch;
  if (max_runs_size == 0 ||
      (*bounds).element_count() != batch * (max_runs_size + 1)) {
    return ffi::Error::InvalidArgument(
        "cell-hull bounds must have max_runs+1 entries per cell");
  }
  if (endog_grid.element_count() % batch != 0 ||
      value.element_count() != endog_grid.element_count()) {
    return ffi::Error::InvalidArgument(
        "cell-hull candidate buffers must share batch*n_candidates elements");
  }
  const size_t n_candidates_size = endog_grid.element_count() / batch;
  if (n_candidates_size == 0 || max_runs_size > INT32_MAX ||
      n_candidates_size > INT32_MAX) {
    return ffi::Error::InvalidArgument("cell-hull dimensions are out of range");
  }
  const int32_t max_runs = static_cast<int32_t>(max_runs_size);
  const int32_t n_candidates = static_cast<int32_t>(n_candidates_size);

  const T* p_left = left.typed_data();
  const T* p_right = right.typed_data();
  const int32_t* p_live = live.typed_data();
  const int32_t* p_low = low.typed_data();
  const int32_t* p_high = high.typed_data();
  const T* p_grid = endog_grid.typed_data();
  const T* p_value = value.typed_data();
  T* p_bounds = (*bounds).typed_data();
  int32_t* p_owners = (*owners).typed_data();
  int32_t* p_status = (*status).typed_data();

  const unsigned reported_threads = std::thread::hardware_concurrency();
  const size_t worker_count = std::min(
      batch, static_cast<size_t>(std::max(1u, reported_threads)));
  auto resolve_worker = [&](size_t worker) {
    for (size_t cell = worker; cell < batch; cell += worker_count) {
      const bool ok = core::ExactCellHull(
          p_left[cell], p_right[cell], p_live + cell * max_runs,
          p_low + cell * max_runs, p_high + cell * max_runs, max_runs,
          p_grid + cell * n_candidates, p_value + cell * n_candidates,
          n_candidates, p_bounds + cell * (max_runs + 1),
          p_owners + cell * max_runs);
      p_status[cell] = ok ? 0 : core::kUnresolved;
    }
  };
  std::vector<std::thread> workers;
  workers.reserve(worker_count - 1);
  try {
    for (size_t worker = 1; worker < worker_count; ++worker) {
      workers.emplace_back(resolve_worker, worker);
    }
  } catch (const std::system_error& error) {
    for (std::thread& worker : workers) worker.join();
    return ffi::Error::Internal(
        std::string("failed to start an exact-cell worker: ") + error.what());
  }
  resolve_worker(0);
  for (std::thread& worker : workers) worker.join();
  return ffi::Error::Success();
}

ffi::Error HandoverF32Impl(
    ffi::Buffer<ffi::F32> ax0, ffi::Buffer<ffi::F32> ax1,
    ffi::Buffer<ffi::F32> av0, ffi::Buffer<ffi::F32> av1,
    ffi::Buffer<ffi::F32> bx0, ffi::Buffer<ffi::F32> bx1,
    ffi::Buffer<ffi::F32> bv0, ffi::Buffer<ffi::F32> bv1,
    ffi::Buffer<ffi::F32> left, ffi::Buffer<ffi::F32> right,
    ffi::ResultBuffer<ffi::F32> output,
    ffi::ResultBuffer<ffi::S32> status) {
  return HandoverImpl<float, ffi::F32>(ax0, ax1, av0, av1, bx0, bx1, bv0,
                                       bv1, left, right, output, status);
}

ffi::Error HandoverF64Impl(
    ffi::Buffer<ffi::F64> ax0, ffi::Buffer<ffi::F64> ax1,
    ffi::Buffer<ffi::F64> av0, ffi::Buffer<ffi::F64> av1,
    ffi::Buffer<ffi::F64> bx0, ffi::Buffer<ffi::F64> bx1,
    ffi::Buffer<ffi::F64> bv0, ffi::Buffer<ffi::F64> bv1,
    ffi::Buffer<ffi::F64> left, ffi::Buffer<ffi::F64> right,
    ffi::ResultBuffer<ffi::F64> output,
    ffi::ResultBuffer<ffi::S32> status) {
  return HandoverImpl<double, ffi::F64>(ax0, ax1, av0, av1, bx0, bx1, bv0,
                                        bv1, left, right, output, status);
}

ffi::Error CellHullF32Impl(
    ffi::Buffer<ffi::F32> left, ffi::Buffer<ffi::F32> right,
    ffi::Buffer<ffi::S32> live, ffi::Buffer<ffi::S32> low,
    ffi::Buffer<ffi::S32> high, ffi::Buffer<ffi::F32> endog_grid,
    ffi::Buffer<ffi::F32> value, ffi::ResultBuffer<ffi::F32> bounds,
    ffi::ResultBuffer<ffi::S32> owners,
    ffi::ResultBuffer<ffi::S32> status) {
  return CellHullImpl<float, ffi::F32>(left, right, live, low, high,
                                       endog_grid, value, bounds, owners,
                                       status);
}

ffi::Error CellHullF64Impl(
    ffi::Buffer<ffi::F64> left, ffi::Buffer<ffi::F64> right,
    ffi::Buffer<ffi::S32> live, ffi::Buffer<ffi::S32> low,
    ffi::Buffer<ffi::S32> high, ffi::Buffer<ffi::F64> endog_grid,
    ffi::Buffer<ffi::F64> value, ffi::ResultBuffer<ffi::F64> bounds,
    ffi::ResultBuffer<ffi::S32> owners,
    ffi::ResultBuffer<ffi::S32> status) {
  return CellHullImpl<double, ffi::F64>(left, right, live, low, high,
                                        endog_grid, value, bounds, owners,
                                        status);
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


template <typename T, ffi::DataType DType>
ffi::Error QueryWinnerBatchedImpl(ffi::Buffer<DType> left_grid,
                                  ffi::Buffer<DType> right_grid,
                                  ffi::Buffer<DType> left_value,
                                  ffi::Buffer<DType> right_value,
                                  ffi::Buffer<ffi::S32> live,
                                  ffi::Buffer<ffi::S32> stable_index,
                                  ffi::Buffer<DType> query,
                                  ffi::ResultBuffer<ffi::S32> winner,
                                  ffi::ResultBuffer<ffi::S32> status) {
  const size_t segment_rank = left_grid.dimensions().size();
  const size_t query_rank = query.dimensions().size();
  if (segment_rank < 2 || query_rank != segment_rank) {
    return ffi::Error::InvalidArgument(
        "batched exact-query operands must share a rank of at least two");
  }
  const int64_t n_segment = left_grid.dimensions()[segment_rank - 1];
  const int64_t n_query = query.dimensions()[query_rank - 1];
  if (n_segment == 0 || n_segment > INT32_MAX) {
    return ffi::Error::InvalidArgument(
        "batched exact-query segment axis must be nonempty");
  }
  const int64_t n_row =
      static_cast<int64_t>(left_grid.element_count()) / n_segment;
  const size_t segment_elements =
      static_cast<size_t>(n_row) * static_cast<size_t>(n_segment);
  if (query.element_count() !=
          static_cast<size_t>(n_row) * static_cast<size_t>(n_query) ||
      right_grid.element_count() != segment_elements ||
      left_value.element_count() != segment_elements ||
      right_value.element_count() != segment_elements ||
      live.element_count() != segment_elements ||
      stable_index.element_count() != segment_elements) {
    return ffi::Error::InvalidArgument(
        "batched exact-query segment buffers must be nonempty and share the "
        "query's row count");
  }
  const size_t query_elements =
      static_cast<size_t>(n_row) * static_cast<size_t>(n_query);
  if ((*winner).element_count() != query_elements ||
      (*status).element_count() != query_elements) {
    return ffi::Error::InvalidArgument(
        "batched exact-query outputs must match the query buffer");
  }
  const T* p_left_grid = left_grid.typed_data();
  const T* p_right_grid = right_grid.typed_data();
  const T* p_left_value = left_value.typed_data();
  const T* p_right_value = right_value.typed_data();
  const int32_t* p_live = live.typed_data();
  const int32_t* p_stable_index = stable_index.typed_data();
  const T* p_query = query.typed_data();
  int32_t* p_winner = (*winner).typed_data();
  int32_t* p_status = (*status).typed_data();
  const int32_t count = static_cast<int32_t>(n_segment);
  for (int64_t row = 0; row < n_row; ++row) {
    const int64_t segment_base = row * n_segment;
    const int64_t query_base = row * n_query;
    for (int64_t column = 0; column < n_query; ++column) {
      int32_t selected = 0;
      const bool ok = core::ExactQueryWinner(
          p_left_grid + segment_base, p_right_grid + segment_base,
          p_left_value + segment_base, p_right_value + segment_base,
          p_live + segment_base, p_stable_index + segment_base, count,
          p_query[query_base + column], &selected);
      p_winner[query_base + column] = selected;
      p_status[query_base + column] = ok ? 0 : core::kUnresolved;
    }
  }
  return ffi::Error::Success();
}

ffi::Error QueryWinnerBatchedF32Impl(
    ffi::Buffer<ffi::F32> left_grid, ffi::Buffer<ffi::F32> right_grid,
    ffi::Buffer<ffi::F32> left_value, ffi::Buffer<ffi::F32> right_value,
    ffi::Buffer<ffi::S32> live,
    ffi::Buffer<ffi::S32> stable_index, ffi::Buffer<ffi::F32> query,
    ffi::ResultBuffer<ffi::S32> winner, ffi::ResultBuffer<ffi::S32> status) {
  return QueryWinnerBatchedImpl<float, ffi::F32>(
      left_grid, right_grid, left_value, right_value, live, stable_index,
      query, winner, status);
}

ffi::Error QueryWinnerBatchedF64Impl(
    ffi::Buffer<ffi::F64> left_grid, ffi::Buffer<ffi::F64> right_grid,
    ffi::Buffer<ffi::F64> left_value, ffi::Buffer<ffi::F64> right_value,
    ffi::Buffer<ffi::S32> live,
    ffi::Buffer<ffi::S32> stable_index, ffi::Buffer<ffi::F64> query,
    ffi::ResultBuffer<ffi::S32> winner, ffi::ResultBuffer<ffi::S32> status) {
  return QueryWinnerBatchedImpl<double, ffi::F64>(
      left_grid, right_grid, left_value, right_value, live, stable_index,
      query, winner, status);
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    ExactQueryWinnerF32, QueryWinnerF32Impl,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::S32>>()
        .Ret<ffi::Buffer<ffi::S32>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    ExactQueryWinnerF64, QueryWinnerF64Impl,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Ret<ffi::Buffer<ffi::S32>>()
        .Ret<ffi::Buffer<ffi::S32>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    ExactQueryWinnerBatchedF32, QueryWinnerBatchedF32Impl,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::S32>>()
        .Ret<ffi::Buffer<ffi::S32>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    ExactQueryWinnerBatchedF64, QueryWinnerBatchedF64Impl,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Ret<ffi::Buffer<ffi::S32>>()
        .Ret<ffi::Buffer<ffi::S32>>());


XLA_FFI_DEFINE_HANDLER_SYMBOL(
    ExactAffineHandoverF32, HandoverF32Impl,
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
        .Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::S32>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    ExactAffineHandoverF64, HandoverF64Impl,
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
        .Arg<ffi::Buffer<ffi::F64>>()
        .Ret<ffi::Buffer<ffi::F64>>()
        .Ret<ffi::Buffer<ffi::S32>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    ExactCellHullF32, CellHullF32Impl,
    ffi::Ffi::Bind()
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
