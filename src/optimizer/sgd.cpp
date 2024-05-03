#include "optimizer/sgd.h"

#include "logger.h"
#include "util.h"

namespace dllm::optimizer::Sgd {
void stepKernel(cudaStream_t stream, Tensor1D &w, const Tensor1D &dw,
                double lr);

Task step(const std::shared_ptr<Tensor1D> &w,
          const std::shared_ptr<const Tensor1D> &dw, const double lr) {
  if (w->layout.shape<0>() != dw->layout.shape<0>()) {
    SPDLOG_LOGGER_CRITICAL(&logger(), "Input data dim not same");
  }
  return Task{[=, future = dw->future](const Context *context) {
    util::waitFutureIfValid(future);
    stepKernel(context->cudaStream, *w, *dw, lr);
    CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
  }};
}
}  // namespace dllm::optimizer::Sgd
