#include "memory/to_torch.h"

#include <c10/cuda/CUDAStream.h>

#include "tensor_impl.h"
#include "threading/scheduler_impl.h"
#include "threading/task_impl.h"

namespace dllm::memory {
void toTorch(const Scheduler &scheduler, at::Tensor &dst,
             const ReadOnlyTensor &src) {
  struct Impl : Task::Impl {
    explicit Impl(std::vector<Tensor> output /* tensor */,
                  std::vector<ReadOnlyTensor> input /* input */)
        : Task::Impl{std::move(output), std::move(input), compute} {}
    void operator()() const override {
      c10::cuda::getCurrentCUDAStream().synchronize();
      auto &dst = output()[0].impl()->tensor();
      auto &src = input()[0].impl()->tensor();
      if (dst.defined()) {
        dst.copy_(src.clone().detach_());
      } else {
        dst = src.clone().detach_();
      }
      c10::cuda::getCurrentCUDAStream().synchronize();
    }
    [[nodiscard]] const char *name() const override {
      return "dllm::memory::toTorch";
    }
  };
  Tensor dst_;
  scheduler.impl()->submit(Task{std::make_shared<Impl>(Impl{{dst_}, {src}})});
  dst = dst_.impl()->tensor();
}
}  // namespace dllm::memory
