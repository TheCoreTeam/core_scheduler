#include <nccl.h>

#include "communication/nccl/all_reduce.h"
#include "dtensor_nccl.h"
#include "logger.h"
#include "util.h"

namespace dllm::communication {
TaskNccl AllReduce<NCCL>::run(
    const std::shared_ptr<const DTensor1D<NCCL>> &tensorSend,
    const std::shared_ptr<DTensor1D<NCCL>> &tensorReceive,
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
  return TaskNccl{
      [=, futureSend = tensorSend->future](const ContextNccl *context) {
        util::waitFutureIfValid(futureSend);
        CHECK_NCCL(ncclAllReduce(
            tensorSend->data(), tensorReceive->data(),
            cute::size(tensorSend->layout), toNcclDataType(tensorSend->dtype),
            toNcclRedOp(operation), context->ncclComm, context->cudaStream));
        CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
      }};
}

TaskNccl AllReduce<NCCL>::runInplace(
    const std::shared_ptr<DTensor1D<NCCL>> &tensor, Operation operation) {
  if (tensor->deviceType != CUDA) {
    SPDLOG_LOGGER_CRITICAL(&logger(), "NCCL backend only supports CUDA tensor");
  }
  return TaskNccl{[=, future = tensor->future](const ContextNccl *context) {
    // Be careful: possible deadlock
    util::waitFutureIfValid(future);
    CHECK_NCCL(ncclAllReduce(
        tensor->data(), tensor->data(), cute::size(tensor->layout),
        toNcclDataType(tensor->dtype), toNcclRedOp(operation),
        context->ncclComm, context->cudaStream));
    CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
  }};
}
}  // namespace dllm::communication
