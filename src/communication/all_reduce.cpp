#include "communication/all_reduce.h"

#include <torch/csrc/autograd/generated/variable_factories.h>

#include "communication/communication_impl.h"
#include "logger.h"
#include "tensor_impl.h"
#include "threading/scheduler_impl.h"
#include "threading/task_impl.h"

namespace dllm::communication {
namespace {
constexpr auto toC10dRedOp(const Operation operation) {
  switch (operation) {
    case SUM:
      return c10d::ReduceOp::SUM;
    default:
      DLLM_ASSERT_TRUE(false, "unsupported operation for NCCL all reduce");
  }
}
}  // namespace

void AllReduce::runInplace(const Scheduler &scheduler, const Comm &comm,
                           const std::vector<Tensor> &tensors,
                           const Operation operation) {
  struct Impl : Task::Impl {
    const Comm comm;
    const Operation operation;

    explicit Impl(std::vector<Tensor> output /* tensors */,
                  std::vector<ReadOnlyTensor> input /* tensors */, Comm comm,
                  const Operation operation)
        : Task::Impl{std::move(output), std::move(input), nccl},
          comm{std::move(comm)},
          operation{operation} {}
    void operator()() const override {
      std::vector<at::Tensor> v;
      v.reserve(output().size());
      for (const auto &t : output()) {
        DLLM_ASSERT_TRUE(t.impl()->tensor().device().type() == at::kCUDA,
                         "NCCL backend only support CUDA GPUs");
        v.push_back(t.impl()->tensor());
      }
      comm.impl()
          ->backend()
          ->allreduce(
              v, c10d::AllreduceOptions{.reduceOp = toC10dRedOp(operation)})
          ->wait();
    }
    [[nodiscard]] const char *name() const override {
      return "dllm::communication::AllReduce::runInplace";
    }
  };

  std::vector<ReadOnlyTensor> input;
  input.reserve(tensors.size());
  for (auto &t : tensors) {
    input.push_back(t);
  }

  scheduler.impl()->submit(
      Task{std::make_shared<Impl>(Impl{tensors, input, comm, operation})});
}
}  // namespace dllm::communication
