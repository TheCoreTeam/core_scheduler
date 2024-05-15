#include "compute/nll.h"

#include "logger.h"
#include "util.h"

namespace dllm::compute::NLL {
void nllForwardKernel(cudaStream_t stream, Tensor1D &loss,
                      const Tensor2D &input, const Tensor2D &target);

TaskCompute forward(const std::shared_ptr<Tensor1D> &loss,
                    const std::shared_ptr<const Tensor2D> &input,
                    const std::shared_ptr<const Tensor2D> &target) {
  if (cute::shape<0>(loss->layout) != cute::shape<0>(input->layout) ||
      cute::shape<0>(loss->layout) != cute::shape<0>(target->layout)) {
    SPDLOG_LOGGER_CRITICAL(&logger(), "Input data dim not same");
  }
  if (loss->dtype != input->dtype) {
    SPDLOG_LOGGER_CRITICAL(&logger(), "Data type mismatches");
  }
  auto task = TaskCompute{
      [loss = loss, input = input, target = target, lossFuture = *loss->future,
       inputFuture = *input->future,
       targetFuture = *target->future](const ContextCompute *context) mutable {
        util::FutureGuard lossGuard{lossFuture};
        util::FutureGuard inputGuard{inputFuture};
        util::FutureGuard targetGuard{targetFuture};
        nllForwardKernel(context->cudaStream, *loss, *input, *target);
        CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
        loss.reset();
        input.reset();
        target.reset();
      }};
  const auto &future = *loss->future = task.get_future();
  *input->future = future;
  *target->future = future;
  return task;
}
}  // namespace dllm::compute::NLL
