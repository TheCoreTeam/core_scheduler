#include "communication/nccl/all_gather.h"

#include <nccl.h>

#include "dtensor_nccl.h"
#include "logger.h"
#include "util.h"

namespace dllm::communication {
TaskNccl AllGather<NCCL>::run(const std::shared_ptr<const Tensor1D> &tensorSend,
                              const std::shared_ptr<Tensor1D> &tensorReceive) {
  if (tensorSend->deviceType != CUDA || tensorReceive->deviceType != CUDA) {
    SPDLOG_LOGGER_CRITICAL(&logger(), "NCCL backend only supports CUDA tensor");
  }
  if (tensorSend->dtype != tensorReceive->dtype) {
    SPDLOG_LOGGER_CRITICAL(&logger(),
                           "sendbuff's dtype is different from the recvbuff's");
  }
  auto task = TaskNccl{[tensorSend = tensorSend, tensorReceive = tensorReceive,
                        futureReceive = *tensorReceive->future,
                        futureSend = tensorSend->future->rFuture](
                           const ContextNccl *context) mutable {
    {
      util::FutureGuard guardSend{futureSend};
      util::FutureGuard guardRReceive{futureReceive.rFuture};
      util::FutureGuard guardWReceive{futureReceive.wFuture};
      CHECK_NCCL(ncclAllGather(tensorSend->data(), tensorReceive->data(),
                               cute::size(tensorSend->layout),
                               util::toNcclDataType(tensorSend->dtype),
                               context->ncclComm, context->cudaStream));
    }
    CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
    tensorReceive.reset();
    tensorSend.reset();
  }};
  const TaskFuture future = task.get_future();
  tensorSend->future->rFuture = future;
  tensorReceive->future->wFuture = future;
  return task;
}

TaskNccl AllGather<NCCL>::runInplace(const std::shared_ptr<Tensor1D> &tensor) {
  if (tensor->deviceType != CUDA) {
    SPDLOG_LOGGER_CRITICAL(&logger(), "NCCL backend only supports CUDA tensor");
  }
  auto task = TaskNccl{[tensor = tensor, future = *tensor->future](
                           const ContextNccl *context) mutable {
    // Be careful: possible deadlock
    {
      util::FutureGuard rGuard{future.rFuture};
      util::FutureGuard wGuard{future.wFuture};
      CHECK_NCCL(ncclAllGather(tensor->data(), tensor->data(),
                               cute::size(tensor->layout) / context->commSize,
                               util::toNcclDataType(tensor->dtype),
                               context->ncclComm, context->cudaStream));
    }
    CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
    tensor.reset();
  }};
  const TaskFuture future = task.get_future();
  tensor->future->rFuture = future;
  tensor->future->wFuture = future;
  return task;
}
}  // namespace dllm::communication
