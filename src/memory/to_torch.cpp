#include "memory/to_torch.h"

#include <ATen/core/Formatting.h>

#include "internal_utils.h"
#include "logger.h"
#include "nvtx_helper.h"
#include "tensor_friend.h"
#include "threading/scheduler_impl.h"
#include "threading/task_cudart.h"

namespace dllm::memory {
void toTorch(const Scheduler &scheduler, at::Tensor &dst,
             const std::shared_ptr<const ReadOnlyTensor> &src) {
  auto task = TaskCudart{[&dst = dst, src = src, srcFuture = src->future()](
                             const ContextCudart *context) mutable {
    DLLM_NVTX_RANGE_FN("dllm::memory::toTorch");
    {
      util::FutureGuard guard{srcFuture};
      if (dst.defined()) {
        dst.copy_(DLLM_EXTRACT_TENSOR(src).clone().detach_());
      } else {
        dst = DLLM_EXTRACT_TENSOR(src).clone().detach_();
      }
      src.reset();
    }
    CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
  }};

  src->resetFuture(task.get_future());
  scheduler.impl()->submit(std::move(task));
}
}  // namespace dllm::memory
