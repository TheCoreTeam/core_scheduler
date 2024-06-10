#include "compute/add.h"

#include <ATen/ops/add.h>

#include "logger.h"
#include "tensor_impl.h"
#include "threading/scheduler_impl.h"
#include "threading/task_impl.h"

namespace dllm::compute::Add {
void forward(const Scheduler& scheduler, Tensor& output,
             const ReadOnlyTensor& A, const ReadOnlyTensor& B) {
  struct Impl : Task::Impl {
    explicit Impl(std::vector<Tensor> output /* tensor */,
                  std::vector<ReadOnlyTensor> input /* A, B */)
        : Task::Impl{std::move(output), std::move(input), compute} {}
    void operator()() const override {
      output()[0].impl()->tensor() =
          at::add(input()[0].impl()->tensor(), input()[1].impl()->tensor());
    }
    [[nodiscard]] const char* name() const override {
      return "dllm::compute::Utils::Add";
    }
  };

  DLLM_ASSERT_TRUE(A.sizes() == B.sizes(),
                   "We do not supprot implicit broadcast add now!");

  output.sizes() = A.sizes();
  scheduler.impl()->submit(
      Task{std::make_shared<Impl>(Impl{{output}, {A, B}})});
}
}  // namespace dllm::compute::Add
