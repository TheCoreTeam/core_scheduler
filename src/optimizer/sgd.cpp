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
  auto task = TaskCompute{
      [w = w, dw = dw, lr = lr, dFuture = *w->future,
       dwFuture = dw->future->wFuture](const ContextCompute *context) mutable {
        util::FutureGuard dRGuard{dFuture.rFuture};
        util::FutureGuard dWGuard{dFuture.wFuture};
        util::FutureGuard dwGuard{dwFuture};
        stepKernel(context->cudaStream, *w, *dw, lr);
        CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
        w.reset();
        dw.reset();
      }};
  const TaskFuture future = task.get_future();
  w->future->wFuture = future;
  dw->future->rFuture = future;
  return task;
}
}  // namespace dllm::optimizer::Sgd
