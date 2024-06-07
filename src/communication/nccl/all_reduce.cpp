#include "communication/all_reduce.h"

#include <nccl.h>
#include <torch/csrc/autograd/generated/variable_factories.h>

#include "internal_utils.h"
#include "logger.h"
#include "nvtx_helper.h"
#include "tensor_impl.h"
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

void AllReduce<NCCL>::runInplace(const Scheduler &scheduler,
                                 const std::vector<Tensor> &tensors,
                                 Operation operation) {
  std::vector<decltype(utils::future(tensors[0]))> futures;
  futures.reserve(tensors.size());
  for (const auto &t : tensors) {
    futures.push_back(utils::future(t));
  }
  auto task = TaskNccl{
      [tensors = tensors, operation = operation,
       future = std::move(futures)](const ContextNccl *context) mutable {
        DLLM_NVTX_RANGE_FN("dllm::communication::AllReduce<NCCL>::runInplace");
        {
          for (auto f : future) {
            utils::FutureGuard{f};
          }
          std::vector<at::Tensor> v;
          v.reserve(tensors.size());
          for (const auto &t : tensors) {
            DLLM_ASSERT_TRUE(t.impl()->tensor().device().type() == at::kCUDA,
                             "NCCL backend only support CUDA GPUs");
            v.push_back(t.impl()->tensor());
          }
          context->backend
              ->allreduce(
                  v, c10d::AllreduceOptions{.reduceOp = toC10dRedOp(operation)})
              ->wait();
          CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
        }
        tensors.clear();
      }};
  const TaskFuture future = task.get_future();
  for (const auto &t : tensors) {
    utils::resetFuture(t, future);
  }
  scheduler.impl()->submit(std::move(task));
}
}  // namespace dllm::communication
