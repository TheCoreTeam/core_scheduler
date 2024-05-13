#include <nccl.h>

#include "communication/nccl/all_reduce.h"
#include "dtensor_nccl.h"
#include "logger.h"
#include "util.h"

namespace dllm::communication {
TaskNccl AllReduce<NCCL>::run(const std::shared_ptr<const Tensor1D> &tensorSend,
                              const std::shared_ptr<Tensor1D> &tensorReceive,
                              Operation operation) {
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
  auto task = TaskNccl{[=, futureReceive = *tensorReceive->future,
                        futureSend =
                            *tensorSend->future](const ContextNccl *context) {
    util::waitFutureIfValid(futureReceive);
    util::waitFutureIfValid(futureSend);
    CHECK_NCCL(ncclAllReduce(
        tensorSend->data(), tensorReceive->data(),
        cute::size(tensorSend->layout), util::toNcclDataType(tensorSend->dtype),
        util::toNcclRedOp(operation), context->ncclComm, context->cudaStream));
    CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
  }};
  const auto &future = *tensorSend->future = task.get_future();
  *tensorReceive->future = future;
  return task;
}

TaskNccl AllReduce<NCCL>::runInplace(const std::shared_ptr<Tensor1D> &tensor,
                                     const Operation operation) {
  if (tensor->deviceType != CUDA) {
    SPDLOG_LOGGER_CRITICAL(&logger(), "NCCL backend only supports CUDA tensor");
  }
  auto task =
      TaskNccl{[=, future = *tensor->future](const ContextNccl *context) {
        // Be careful: possible deadlock
        util::waitFutureIfValid(future);
        CHECK_NCCL(ncclAllReduce(
            tensor->data(), tensor->data(), cute::size(tensor->layout),
            util::toNcclDataType(tensor->dtype), util::toNcclRedOp(operation),
            context->ncclComm, context->cudaStream));
        CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
      }};
  *tensor->future = task.get_future();
  return task;
}
}  // namespace dllm::communication
