#include "compute/add.h"

#include <torch/torch.h>

#include "internal_utils.h"
#include "logger.h"
#include "nvtx_helper.h"
#include "tensor_impl.h"
#include "threading/scheduler_impl.h"
#include "threading/task_compute.h"

namespace dllm::compute::Add {
void forward(const Scheduler& scheduler, Tensor& output,
             const ReadOnlyTensor& A, const ReadOnlyTensor& B) {
  DLLM_ASSERT_TRUE(A.sizes() == B.sizes(),
                   "We do not supprot implicit broadcast add now!");
  auto task = TaskCompute{
      [output = output, A = A, B = B, AFuture = utils::future(A),
       BFuture = utils::future(B), outputFuture = utils::future(output)](
          const ContextCompute* context) mutable {
        DLLM_NVTX_RANGE_FN("dllm::compute::Add::forward");
        {
          utils::FutureGuard aGuard{AFuture};
          utils::FutureGuard bGuard{BFuture};
          utils::FutureGuard outputGuard{outputFuture};
          output.impl()->tensor() = A.impl()->tensor() + B.impl()->tensor();
        }
        CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
        A.reset();
        B.reset();
        output.reset();
      }};
  const TaskFuture future = task.get_future();
  utils::resetFuture(A, future);
  utils::resetFuture(B, future);
  utils::resetFuture(output, future);
  output.sizes() = A.sizes();
  scheduler.impl()->submit(std::move(task));
}
}  // namespace dllm::compute::Add
