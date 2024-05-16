#include "communication/nccl/all_gather.h"

#include <nccl.h>

#include "dtensor_nccl.h"
#include "logger.h"
#include "util.h"

namespace dllm::communication {
TaskNccl AllGather<NCCL>::run(const std::shared_ptr<const Tensor1D> &tensorSend,
                              const std::shared_ptr<Tensor1D> &tensorReceive,
                              const std::size_t sendCount) {
  if (sendCount > cute::size(tensorSend->layout) ||
      sendCount > cute::size(tensorReceive->layout)) {
    SPDLOG_LOGGER_CRITICAL(&logger(), "sendSize larger than buffer size");
  }
  if (tensorSend->deviceType != CUDA || tensorReceive->deviceType != CUDA) {
    SPDLOG_LOGGER_CRITICAL(&logger(), "NCCL backend only supports CUDA tensor");
  }
  if (tensorSend->dtype != tensorReceive->dtype) {
    SPDLOG_LOGGER_CRITICAL(&logger(),
                           "sendbuff's dtype is different from the recvbuff's");
  }
  auto task = TaskNccl{
      [tensorSend = tensorSend, tensorReceive = tensorReceive,
       sendCount = sendCount, futureReceive = *tensorReceive->future,
       futureSend = *tensorSend->future](const ContextNccl *context) mutable {
        {
          util::FutureGuard guardReceive{futureReceive};
          util::FutureGuard guardSend{futureSend};
          CHECK_NCCL(ncclAllGather(tensorSend->data(), tensorReceive->data(),
                                   sendCount,
                                   util::toNcclDataType(tensorSend->dtype),
                                   context->ncclComm, context->cudaStream));
        }
        CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
        tensorReceive.reset();
        tensorSend.reset();
      }};
  const auto &future = *tensorSend->future = task.get_future();
  *tensorReceive->future = future;
  return task;
}

TaskNccl AllGather<NCCL>::runInplace(const std::shared_ptr<Tensor1D> &tensor,
                                     const std::size_t sendCount) {
  if (sendCount > cute::size(tensor->layout)) {
    SPDLOG_LOGGER_CRITICAL(&logger(), "sendSize larger than buffer size");
  }
  if (tensor->deviceType != CUDA) {
    SPDLOG_LOGGER_CRITICAL(&logger(), "NCCL backend only supports CUDA tensor");
  }
  auto task =
      TaskNccl{[tensor = tensor, sendCount = sendCount,
                future = *tensor->future](const ContextNccl *context) mutable {
        // Be careful: possible deadlock
        {
          util::FutureGuard guard{future};
          CHECK_NCCL(ncclAllGather(tensor->data(), tensor->data(), sendCount,
                                   util::toNcclDataType(tensor->dtype),
                                   context->ncclComm, context->cudaStream));
        }
        CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
        tensor.reset();
      }};
  *tensor->future = task.get_future();
  return task;
}
}  // namespace dllm::communication
