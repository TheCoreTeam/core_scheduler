#include "fc.h"
#include "logger.h"
#include "matmul.h"
#include <spdlog/spdlog.h>

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
              cute::stride<1>(y.layout)),
          cute::layout<2>(y.layout)),
      y.dtype, y.deviceType};
  // x: Batch x Sequence x Feature -> (Batch * Sequence) x Feature
  Tensor2D xView{
      x.data,
      cute::make_layout(
          cute::make_layout(
              cute::size(cute::take<0, decltype(x.layout)::rank - 1>(x.layout)),
              cute::stride<1>(x.layout)),
          cute::layout<2>(x.layout)),
      x.dtype, x.deviceType};
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
              cute::size(cute::take<0, decltype(dy.layout)::rank>(dy.layout)),
              cute::stride<1>(dy.layout)),
          cute::layout<2>(dy.layout)),
      dy.dtype, dy.deviceType};
  // x: Batch x Sequence x Feature -> (Batch * Sequence) x Feature
  Tensor2D xView{
      x.data,
      cute::make_layout(
          cute::make_layout(
              cute::size(cute::take<0, decltype(x.layout)::rank>(x.layout)),
              cute::stride<1>(x.layout)),
          cute::layout<2>(x.layout)),
      x.dtype, x.deviceType};
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
              cute::size(cute::take<0, decltype(dy.layout)::rank>(dy.layout)),
              cute::stride<1>(dy.layout)),
          cute::layout<2>(dy.layout)),
      dy.dtype, dy.deviceType};
  // x: Batch x Sequence x Feature -> (Batch * Sequence) x Feature
  Tensor2D dxView{
      dx.data,
      cute::make_layout(
          cute::make_layout(
              cute::size(cute::take<0, decltype(dx.layout)::rank>(dx.layout)),
              cute::stride<1>(dx.layout)),
          cute::layout<2>(dx.layout)),
      dx.dtype, dx.deviceType};
  RowMajorNNMatmulNoBias(handle, dyView, w, dxView, computeType);
}
} // namespace dllm
