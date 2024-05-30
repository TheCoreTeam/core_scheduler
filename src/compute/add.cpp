#include "compute/add.h"

#include <torch/torch.h>

#include "internal_utils.h"
#include "logger.h"
#include "nvtx_helper.h"
#include "tensor_friend.h"

namespace dllm::compute::Add {

TaskCompute forward(const std::shared_ptr<Tensor>& output,
                    const std::shared_ptr<const ReadOnlyTensor>& A,
                    const std::shared_ptr<const ReadOnlyTensor>& B) {
  DLLM_ASSERT_TRUE(A->sizes() == B->sizes(),
                   "We do not supprot implicit broadcast add now!");
  auto task =
      TaskCompute{[output = output, A = A, B = B, AFuture = A->future(),
                   BFuture = B->future(), outputFuture = output->future()](
                      const ContextCompute* context) mutable {
        DLLM_NVTX_RANGE_FN("dllm::compute::Add::forward");
        {
          util::FutureGuard aGuard{AFuture};
          util::FutureGuard bGuard{BFuture};
          util::FutureGuard outputGuard{outputFuture};
          DLLM_EXTRACT_TENSOR(output) =
              TensorFriend::extract_tensor(A) + TensorFriend::extract_tensor(B);
        }
        CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
        A.reset();
        B.reset();
        output.reset();
      }};
  const TaskFuture future = task.get_future();
  A->resetFuture(future);
  B->resetFuture(future);
  output->resetFuture(future);
  output->sizes() = A->sizes();
  return task;
}
}  // namespace dllm::compute::Add
