#include "tensor.h"
#include "threading/task_cudart.h"

namespace dllm::memory {
TaskCudart memcpyFromHost(const std::shared_ptr<Tensor1D>& dst,
                          const void* src);

TaskCudart memcpyToHost(void* dst, const std::shared_ptr<const Tensor1D>& src);
}  // namespace dllm::memory