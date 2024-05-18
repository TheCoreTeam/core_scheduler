#include "communication/nccl/all_reduce.h"

#include <nccl.h>

#include "dtensor_nccl.h"
#include "logger.h"
#include "util.h"

namespace dllm::communication {
TaskNccl AllReduce<NCCL>::run(const std::shared_ptr<const Tensor1D> &tensorSend,
                              const std::shared_ptr<Tensor1D> &tensorReceive,
                              const Operation operation) {
  if (cute::size(tensorSend->layout) != cute::size(tensorReceive->layout)) {
    SPDLOG_LOGGER_CRITICAL(&logger(),
                           "sendbuff is not the same as the recvbuff");
  }
  if (tensorSend->deviceType != CUDA || tensorReceive->deviceType != CUDA) {
    SPDLOG_LOGGER_CRITICAL(&logger(), "NCCL backend only supports CUDA tensor");
  }
  if (tensorSend->dtype != tensorReceive->dtype) {
    SPDLOG_LOGGER_CRITICAL(&logger(),
                           "sendbuff's dtype is different from the recvbuff's");
  }
  auto task =
      TaskNccl{[tensorSend = tensorSend, tensorReceive = tensorReceive,
                operation = operation, futureReceive = *tensorReceive->future,
                futureSend = tensorSend->future->rFuture](
                   const ContextNccl *context) mutable {
        {
          util::FutureGuard guardSend{futureSend};
          util::FutureGuard guardRReceive{futureReceive.rFuture};
          util::FutureGuard guardWReceive{futureReceive.wFuture};
          CHECK_NCCL(ncclAllReduce(tensorSend->data(), tensorReceive->data(),
                                   cute::size(tensorSend->layout),
                                   util::toNcclDataType(tensorSend->dtype),
                                   util::toNcclRedOp(operation),
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

TaskNccl AllReduce<NCCL>::runInplace(const std::shared_ptr<Tensor1D> &tensor,
                                     const Operation operation) {
  if (tensor->deviceType != CUDA) {
    SPDLOG_LOGGER_CRITICAL(&logger(), "NCCL backend only supports CUDA tensor");
  }
  auto task =
      TaskNccl{[tensor = tensor, operation = operation,
                future = *tensor->future](const ContextNccl *context) mutable {
        // Be careful: possible deadlock
        {
          util::FutureGuard rGuard{future.rFuture};
          util::FutureGuard wGuard{future.wFuture};
          CHECK_NCCL(ncclAllReduce(
              tensor->data(), tensor->data(), cute::size(tensor->layout),
              util::toNcclDataType(tensor->dtype), util::toNcclRedOp(operation),
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
