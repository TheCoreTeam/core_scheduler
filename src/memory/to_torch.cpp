#include "memory/to_torch.h"

#include <ATen/core/Formatting.h>

#include "internal_utils.h"
#include "logger.h"
#include "nvtx_helper.h"
#include "tensor_impl.h"
#include "threading/scheduler_impl.h"
#include "threading/task_cudart.h"

namespace dllm::memory {
void toTorch(const Scheduler &scheduler, at::Tensor &dst,
             const ReadOnlyTensor &src) {
  auto task =
      TaskCudart{[&dst = dst, src = src, srcFuture = utils::future(src)](
                     const ContextCudart *context) mutable {
        DLLM_NVTX_RANGE_FN("dllm::memory::toTorch");
        {
          utils::FutureGuard guard{srcFuture};
          if (dst.defined()) {
            dst.copy_(src.impl()->tensor().clone().detach_());
          } else {
            dst = src.impl()->tensor().clone().detach_();
          }
          src.reset();
        }
        CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
      }};

  utils::resetFuture(src, task.get_future());
  scheduler.impl()->submit(std::move(task));
}
}  // namespace dllm::memory
