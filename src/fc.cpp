#include "fc.h"

#include <spdlog/spdlog.h>

#include "logger.h"
#include "matmul.h"

namespace dllm {
Task FcNoBias::forward(const std::shared_ptr<Tensor3D> &y,
                       const std::shared_ptr<const Tensor3D> &x,
                       const std::shared_ptr<const Tensor2D> &w,
                       const cublasComputeType_t computeType) {
  // y: Batch x Sequence x Feature -> (Batch * Sequence) x Feature
  if (x->layout.stride<0>() != x->layout.shape<1>() * x->layout.stride<1>()) {
    SPDLOG_LOGGER_CRITICAL(&logger(), "Input data is not contiguous");
  }
  if (y->layout.stride<0>() != y->layout.shape<1>() * y->layout.stride<1>()) {
    SPDLOG_LOGGER_CRITICAL(&logger(), "Input data is not contiguous");
  }
  return Task{[=](const Context *context) {
    Tensor2D yView{
        y->data(),
        cute::make_layout(
            cute::make_layout(
                cute::size(
                    cute::take<0, decltype(y->layout)::rank - 1>(y->layout)),
                cute::stride<decltype(x->layout)::rank - 2>(y->layout)),
            cute::layout<decltype(y->layout)::rank - 1>(y->layout)),
        y->dtype, y->deviceType};
    // x: Batch x Sequence x Feature -> (Batch * Sequence) x Feature
    const Tensor2D xView{
        x->data(),
        cute::make_layout(
            cute::make_layout(
                cute::size(
                    cute::take<0, decltype(x->layout)::rank - 1>(x->layout)),
                cute::stride<decltype(x->layout)::rank - 2>(x->layout)),
            cute::layout<decltype(x->layout)::rank - 1>(x->layout)),
        x->dtype, x->deviceType};
    x->waitFutureIfValid();
    w->waitFutureIfValid();
    RowMajorNTMatmulNoBias(context->cublasHandle, xView, *w, yView,
                           computeType);
    CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
  }};
}

Task FcNoBias::backwardW(const std::shared_ptr<Tensor2D> &dw,
                         const std::shared_ptr<const Tensor3D> &dy,
                         const std::shared_ptr<const Tensor3D> &x,
                         cublasComputeType_t computeType) {
  // dx, x: M * K
  // dy: M * N
  // dw = dy^T @ x
  if (x->layout.stride<0>() != x->layout.shape<1>() * x->layout.stride<1>()) {
    SPDLOG_LOGGER_CRITICAL(&logger(), "Input data is not contiguous");
  }
  if (dy->layout.stride<0>() !=
      dy->layout.shape<1>() * dy->layout.stride<1>()) {
    SPDLOG_LOGGER_CRITICAL(&logger(), "Input data is not contiguous");
  }

  return Task{[=](const Context *context) {
    const Tensor2D dyView{
        dy->data(),
        cute::make_layout(
            cute::make_layout(
                cute::size(
                    cute::take<0, decltype(dy->layout)::rank - 1>(dy->layout)),
                cute::stride<decltype(dy->layout)::rank - 2>(dy->layout)),
            cute::layout<decltype(dy->layout)::rank - 1>(dy->layout)),
        dy->dtype, dy->deviceType};
    // x: Batch x Sequence x Feature -> (Batch * Sequence) x Feature
    const Tensor2D xView{
        x->data(),
        cute::make_layout(
            cute::make_layout(
                cute::size(
                    cute::take<0, decltype(x->layout)::rank - 1>(x->layout)),
                cute::stride<decltype(x->layout)::rank - 2>(x->layout)),
            cute::layout<decltype(x->layout)::rank - 1>(x->layout)),
        x->dtype, x->deviceType};
    dy->waitFutureIfValid();
    x->waitFutureIfValid();
    RowMajorTNMatmulNoBias(context->cublasHandle, dyView, xView, *dw,
                           computeType);
    CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
  }};
}

Task FcNoBias::backwardX(const std::shared_ptr<Tensor3D> &dx,
                         const std::shared_ptr<const Tensor3D> &dy,
                         const std::shared_ptr<const Tensor2D> &w,
                         cublasComputeType_t computeType) {
  // dw, w: N * K
  // dy: M * N
  // dx = dy @ w
  if (dx->layout.stride<0>() !=
      dx->layout.shape<1>() * dx->layout.stride<1>()) {
    SPDLOG_LOGGER_CRITICAL(&logger(), "Input data is not contiguous");
  }
  if (dy->layout.stride<0>() !=
      dy->layout.shape<1>() * dy->layout.stride<1>()) {
    SPDLOG_LOGGER_CRITICAL(&logger(), "Input data is not contiguous");
  }
  return Task{[=](const Context *context) {
    const Tensor2D dyView{
        dy->data(),
        cute::make_layout(
            cute::make_layout(
                cute::size(
                    cute::take<0, decltype(dy->layout)::rank - 1>(dy->layout)),
                cute::stride<decltype(dy->layout)::rank - 2>(dy->layout)),
            cute::layout<decltype(dy->layout)::rank - 1>(dy->layout)),
        dy->dtype, dy->deviceType};
    // x: Batch x Sequence x Feature -> (Batch * Sequence) x Feature
    Tensor2D dxView{
        dx->data(),
        cute::make_layout(
            cute::make_layout(
                cute::size(
                    cute::take<0, decltype(dx->layout)::rank - 1>(dx->layout)),
                cute::stride<decltype(dx->layout)::rank - 2>(dx->layout)),
            cute::layout<decltype(dx->layout)::rank - 1>(dx->layout)),
        dx->dtype, dx->deviceType};
    dy->waitFutureIfValid();
    w->waitFutureIfValid();
    RowMajorNNMatmulNoBias(context->cublasHandle, dyView, *w, dxView,
                           computeType);
    CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
  }};
}
}  // namespace dllm
