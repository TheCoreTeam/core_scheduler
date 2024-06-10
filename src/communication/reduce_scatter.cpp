#include "communication/reduce_scatter.h"

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
void ReduceScatter::run(
    const Scheduler &scheduler, const Comm &comm,
    const std::vector<Tensor> &tensorReceive,
    const std::vector<std::vector<ReadOnlyTensor>> &tensorSend,
    const Operation operation) {
  struct Impl : Task::Impl {
    const std::vector<std::vector<ReadOnlyTensor>> actualSend;
    const Comm comm;
    const Operation operation;

    explicit Impl(std::vector<Tensor> output /* tensors */,
                  std::vector<ReadOnlyTensor> input /* tensors */,
                  const std::vector<std::vector<ReadOnlyTensor>> &actualSend,
                  Comm comm, const Operation operation)
        : Task::Impl{std::move(output), std::move(input), nccl},
          actualSend{actualSend},
          comm{std::move(comm)},
          operation{operation} {}
    void operator()() const override {
      std::vector<std::vector<at::Tensor>> vSend{};
      vSend.reserve(actualSend.size());
      for (auto &i : actualSend) {
        std::vector<at::Tensor> v;
        v.reserve(i.size());
        for (const auto &t : i) {
          DLLM_ASSERT_TRUE(t.impl()->tensor().device().type() == at::kCUDA,
                           "NCCL backend only support CUDA GPUs");
          v.push_back(t.impl()->tensor());
        }
        vSend.push_back(std::move(v));
      }
      std::vector<at::Tensor> vReceive;
      vReceive.reserve(output().size());
      for (const auto &t : output()) {
        DLLM_ASSERT_TRUE(t.impl()->tensor().device().type() == at::kCUDA,
                         "NCCL backend only support CUDA GPUs");
        vReceive.push_back(t.impl()->tensor());
      }
      comm.impl()
          ->backend()
          ->reduce_scatter(
              vReceive, vSend,
              c10d::ReduceScatterOptions{.reduceOp = toC10dRedOp(operation)})
          ->wait();
    }
    [[nodiscard]] const char *name() const override {
      return "dllm::communication::ReduceScatter::run";
    }
  };

  std::vector<ReadOnlyTensor> input;
  input.reserve(tensorSend.size() * tensorSend[0].size());
  for (auto &v : tensorSend) {
    for (auto &t : v) {
      input.push_back(t);
    }
  }

  scheduler.impl()->submit(Task{std::make_shared<Impl>(
      Impl{tensorReceive, input, tensorSend, comm, operation})});
}
}  // namespace dllm::communication
