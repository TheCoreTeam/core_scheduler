#include "memory/to_torch.h"

#include <ATen/core/Formatting.h>

#include "internal_utils.h"
#include "logger.h"
#include "nvtx_helper.h"
#include "tensor_friend.h"

namespace dllm::memory {
TaskCudart toTorch(at::Tensor &dst,
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
  return task;
}
}  // namespace dllm::memory
