#include "optimizer/sgd.h"

#include "logger.h"
#include "util.h"

namespace dllm::optimizer::Sgd {
void stepKernel(cudaStream_t stream, Tensor1D &w, const Tensor1D &dw,
                double lr);

TaskCompute step(const std::shared_ptr<Tensor1D> &w,
                 const std::shared_ptr<const Tensor1D> &dw, const double lr) {
  if (w->layout.shape<0>() != dw->layout.shape<0>()) {
    SPDLOG_LOGGER_CRITICAL(&logger(), "Input data dim not same");
  }
  auto task = TaskCompute{[=, futureW = *w->future, future = *dw->future](
                              const ContextCompute *context) {
    util::waitFutureIfValid(futureW);
    util::waitFutureIfValid(future);
    stepKernel(context->cudaStream, *w, *dw, lr);
    CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
  }};
  *w->future = task.get_future();
  *dw->future = *w->future;
  return task;
}
}  // namespace dllm::optimizer::Sgd
