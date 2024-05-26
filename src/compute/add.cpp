#include "compute/add.h"

#include <torch/torch.h>

#include "internal_utils.h"
#include "logger.h"

namespace dllm::compute::Add {

TaskCompute forward(const std::shared_ptr<Tensor>& output,
                    const std::shared_ptr<const Tensor>& A,
                    const std::shared_ptr<const Tensor>& B) {
  auto task = TaskCompute{
      [output = output, A = A, B = B, AFuture = A->future().wFuture,
       BFuture = B->future().wFuture,
       outputFuture = output->future()](const ContextCompute* context) mutable {
        {
          util::FutureGuard aGuard{AFuture};
          util::FutureGuard bGuard{BFuture};
          util::FutureGuard outputGuard{outputFuture};
          output->tensor() = A->tensor() + B->tensor();
        }
        CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
        A.reset();
        B.reset();
        output.reset();
      }};
  const TaskFuture future = task.get_future();
  A->future().rFuture = future;
  B->future().rFuture = future;
  output->future().wFuture = future;
  return task;
}
}  // namespace dllm::compute::Add
