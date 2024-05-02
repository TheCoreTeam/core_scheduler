#include "fc.h"

#include <spdlog/spdlog.h>

#include "logger.h"
#include "matmul.h"

namespace dllm {
void FcNoBias::forward(cublasHandle_t handle, const Tensor3D &y,
                       const Tensor3D &x, const Tensor2D &w,
                       cublasComputeType_t computeType) {
  // y: Batch x Sequence x Feature -> (Batch * Sequence) x Feature
  if (x.layout.stride<0>() != x.layout.shape<1>() * x.layout.stride<1>()) {
    SPDLOG_LOGGER_CRITICAL(&logger(), "Input data is not contiguous");
  }
  if (y.layout.stride<0>() != y.layout.shape<1>() * y.layout.stride<1>()) {
    SPDLOG_LOGGER_CRITICAL(&logger(), "Input data is not contiguous");
  }
  Tensor2D yView{
      y.data,
      cute::make_layout(
          cute::make_layout(
              cute::size(cute::take<0, decltype(y.layout)::rank - 1>(y.layout)),
              cute::stride<decltype(x.layout)::rank - 2>(y.layout)),
          cute::layout<decltype(y.layout)::rank - 1>(y.layout)),
      y.dtype, y.deviceType};
  // x: Batch x Sequence x Feature -> (Batch * Sequence) x Feature
  Tensor2D xView{
      x.data,
      cute::make_layout(
          cute::make_layout(
              cute::size(cute::take<0, decltype(x.layout)::rank - 1>(x.layout)),
              cute::stride<decltype(x.layout)::rank - 2>(x.layout)),
          cute::layout<decltype(x.layout)::rank - 1>(x.layout)),
      x.dtype, x.deviceType};
  y.waitFutureIfValid();
  x.waitFutureIfValid();
  w.waitFutureIfValid();
  RowMajorNTMatmulNoBias(handle, xView, w, yView, computeType);
}

void FcNoBias::backwardW(cublasHandle_t handle, const Tensor2D &dw,
                         const Tensor3D &dy, const Tensor3D &x,
                         cublasComputeType_t computeType) {
  // dx, x: M * K
  // dy: M * N
  // dw = dy^T @ x
  if (x.layout.stride<0>() != x.layout.shape<1>() * x.layout.stride<1>()) {
    SPDLOG_LOGGER_CRITICAL(&logger(), "Input data is not contiguous");
  }
  if (dy.layout.stride<0>() != dy.layout.shape<1>() * dy.layout.stride<1>()) {
    SPDLOG_LOGGER_CRITICAL(&logger(), "Input data is not contiguous");
  }
  Tensor2D dyView{
      dy.data,
      cute::make_layout(
          cute::make_layout(
              cute::size(
                  cute::take<0, decltype(dy.layout)::rank - 1>(dy.layout)),
              cute::stride<decltype(dy.layout)::rank - 2>(dy.layout)),
          cute::layout<decltype(dy.layout)::rank - 1>(dy.layout)),
      dy.dtype, dy.deviceType};
  // x: Batch x Sequence x Feature -> (Batch * Sequence) x Feature
  Tensor2D xView{
      x.data,
      cute::make_layout(
          cute::make_layout(
              cute::size(cute::take<0, decltype(x.layout)::rank - 1>(x.layout)),
              cute::stride<decltype(x.layout)::rank - 2>(x.layout)),
          cute::layout<decltype(x.layout)::rank - 1>(x.layout)),
      x.dtype, x.deviceType};
  dw.waitFutureIfValid();
  dy.waitFutureIfValid();
  x.waitFutureIfValid();
  RowMajorTNMatmulNoBias(handle, dyView, xView, dw, computeType);
}

void FcNoBias::backwardX(cublasHandle_t handle, const Tensor3D &dx,
                         const Tensor3D &dy, const Tensor2D &w,
                         cublasComputeType_t computeType) {
  // dw, w: N * K
  // dy: M * N
  // dx = dy @ w
  if (dx.layout.stride<0>() != dx.layout.shape<1>() * dx.layout.stride<1>()) {
    SPDLOG_LOGGER_CRITICAL(&logger(), "Input data is not contiguous");
  }
  if (dy.layout.stride<0>() != dy.layout.shape<1>() * dy.layout.stride<1>()) {
    SPDLOG_LOGGER_CRITICAL(&logger(), "Input data is not contiguous");
  }
  Tensor2D dyView{
      dy.data,
      cute::make_layout(
          cute::make_layout(
              cute::size(
                  cute::take<0, decltype(dy.layout)::rank - 1>(dy.layout)),
              cute::stride<decltype(dy.layout)::rank - 2>(dy.layout)),
          cute::layout<decltype(dy.layout)::rank - 1>(dy.layout)),
      dy.dtype, dy.deviceType};
  // x: Batch x Sequence x Feature -> (Batch * Sequence) x Feature
  Tensor2D dxView{
      dx.data,
      cute::make_layout(
          cute::make_layout(
              cute::size(
                  cute::take<0, decltype(dx.layout)::rank - 1>(dx.layout)),
              cute::stride<decltype(dx.layout)::rank - 2>(dx.layout)),
          cute::layout<decltype(dx.layout)::rank - 1>(dx.layout)),
      dx.dtype, dx.deviceType};
  dx.waitFutureIfValid();
  dy.waitFutureIfValid();
  w.waitFutureIfValid();
  RowMajorNNMatmulNoBias(handle, dyView, w, dxView, computeType);
}
}  // namespace dllm
