#include "communication/nccl/all_gather.h"

#include <nccl.h>
#include <torch/csrc/autograd/generated/variable_factories.h>

#include "internal_utils.h"
#include "logger.h"
#include "tensor_friend.h"

namespace dllm::communication {
namespace {
ncclDataType_t toNcclDataType(const at::ScalarType dtype) {
  switch (dtype) {
    case at::kDouble:
      return ncclFloat64;
    case at::kFloat:
      return ncclFloat32;
    case at::kHalf:
      return ncclFloat16;
    case at::kBFloat16:
      return ncclBfloat16;
    default:
      return static_cast<ncclDataType_t>(0);
  }
}
}  // namespace

TaskNccl AllGather<NCCL>::run(
    const std::shared_ptr<Tensor> &tensorReceive,
    const std::shared_ptr<const ReadOnlyTensor> &tensorSend,
    const int64_t receiveCount) {
  if (DLLM_EXTRACT_TENSOR(tensorReceive).defined() &&
      !DLLM_EXTRACT_TENSOR(tensorReceive).is_contiguous()) {
    SPDLOG_LOGGER_CRITICAL(&logger(), "receive tensor not contiguout");
  }
  auto task = TaskNccl{
      [receiveCount = receiveCount, tensorSend = tensorSend,
       tensorReceive = tensorReceive, futureReceive = tensorReceive->future(),
       futureSend = tensorSend->future()](const ContextNccl *context) mutable {
        {
          const auto count = tensorSend->numel();
          util::FutureGuard guardReceive{futureReceive};
          util::FutureGuard guardSend{futureSend};
          if (!DLLM_EXTRACT_TENSOR(tensorReceive).defined()) {
            DLLM_EXTRACT_TENSOR(tensorReceive) = torch::empty(
                {receiveCount}, DLLM_EXTRACT_TENSOR(tensorSend).options());
          }
          const auto tensorSendContiguout =
              DLLM_EXTRACT_TENSOR(tensorSend).contiguous();
          CHECK_NCCL(ncclAllGather(
              DLLM_EXTRACT_TENSOR(tensorSend).data_ptr(),
              DLLM_EXTRACT_TENSOR(tensorReceive).data_ptr(), count,
              toNcclDataType(DLLM_EXTRACT_TENSOR(tensorSend).scalar_type()),
              context->ncclComm, context->cudaStream));
          CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
        }
        tensorReceive.reset();
        tensorSend.reset();
      }};
  const TaskFuture future = task.get_future();
  tensorSend->resetFuture(future);
  tensorReceive->resetFuture(future);
  tensorReceive->sizes() = IntArray{receiveCount};
  return task;
}

TaskNccl AllGather<NCCL>::runInplace(const std::shared_ptr<Tensor> &tensor) {
  if (DLLM_EXTRACT_TENSOR(tensor).defined() &&
      !DLLM_EXTRACT_TENSOR(tensor).is_contiguous()) {
    SPDLOG_LOGGER_CRITICAL(&logger(), "tensor not contiguout");
  }
  auto task = TaskNccl{[tensor = tensor, future = tensor->future()](
                           const ContextNccl *context) mutable {
    {
      const auto count = tensor->numel();
      util::FutureGuard guard{future};
      CHECK_NCCL(ncclAllGather(
          DLLM_EXTRACT_TENSOR(tensor).data_ptr(),
          DLLM_EXTRACT_TENSOR(tensor).data_ptr(), count / context->commSize,
          toNcclDataType(DLLM_EXTRACT_TENSOR(tensor).scalar_type()),
          context->ncclComm, context->cudaStream));
    }
    CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
    tensor.reset();
  }};
  const TaskFuture future = task.get_future();
  tensor->resetFuture(future);
  return task;
}
}  // namespace dllm::communication
