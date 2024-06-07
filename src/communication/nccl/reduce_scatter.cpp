#include "communication/reduce_scatter.h"

#include <nccl.h>
#include <torch/csrc/autograd/generated/variable_factories.h>

#include "internal_utils.h"
#include "logger.h"
#include "nvtx_helper.h"
#include "tensor_friend.h"
#include "threading/scheduler_impl.h"

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
void ReduceScatter<NCCL>::run(
    const Scheduler &scheduler,
    const std::vector<std::shared_ptr<Tensor>> &tensorReceive,
    const std::vector<std::vector<std::shared_ptr<const ReadOnlyTensor>>>
        &tensorSend,
    Operation operation) {
  std::vector<TensorFuture> futureReceive;
  std::vector<TaskFuture> futureSend;
  futureReceive.reserve(tensorReceive.size());
  DLLM_ASSERT_TRUE(!tensorSend.empty(), "Sending buffer is empty");
  futureSend.reserve(tensorSend.size() * tensorSend[0].size());
  for (const auto &t : tensorReceive) {
    futureReceive.push_back(t->future());
  }
  for (const auto &vt : tensorSend) {
    for (const auto &t : vt) {
      futureSend.push_back(t->future());
    }
  }
  auto task = TaskNccl{[operation = operation, tensorSend = tensorSend,
                        tensorReceive = tensorReceive,
                        futureReceive = std::move(futureReceive),
                        futureSend = std::move(futureSend)](
                           const ContextNccl *context) mutable {
    DLLM_NVTX_RANGE_FN("dllm::communication::ReduceScatter<NCCL>::run");
    {
      for (auto &f : futureReceive) {
        util::FutureGuard{f};
      }
      for (auto &f : futureSend) {
        util::FutureGuard{f};
      }
      std::vector<std::vector<at::Tensor>> vSend{};
      vSend.reserve(tensorSend.size());
      for (auto &i : tensorSend) {
        std::vector<at::Tensor> v;
        v.reserve(i.size());
        for (const auto &t : i) {
          DLLM_ASSERT_TRUE(DLLM_EXTRACT_TENSOR(t).device().type() == at::kCUDA,
                           "NCCL backend only support CUDA GPUs");
          v.push_back(DLLM_EXTRACT_TENSOR(t));
        }
        vSend.push_back(std::move(v));
      }
      std::vector<at::Tensor> vReceive;
      vReceive.reserve(tensorReceive.size());
      for (const auto &t : tensorReceive) {
        DLLM_ASSERT_TRUE(DLLM_EXTRACT_TENSOR(t).device().type() == at::kCUDA,
                         "NCCL backend only support CUDA GPUs");
        vReceive.push_back(DLLM_EXTRACT_TENSOR(t));
      }
      context->backend
          ->reduce_scatter(
              vReceive, vSend,
              c10d::ReduceScatterOptions{.reduceOp = toC10dRedOp(operation)})
          ->wait();
      CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
    }
    tensorReceive.clear();
    tensorSend.clear();
  }};
  const TaskFuture future = task.get_future();
  for (const auto &vt : tensorSend) {
    for (const auto &t : vt) {
      t->resetFuture(future);
    }
  }
  for (const auto &t : tensorReceive) {
    t->resetFuture(future);
  }
  scheduler.impl()->submit(std::move(task));
}
}  // namespace dllm::communication
