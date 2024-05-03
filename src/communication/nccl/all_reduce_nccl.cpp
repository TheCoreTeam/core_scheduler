#include <nccl.h>

#include "communication/all_reduce.h"
#include "dtensor_nccl.h"
#include "logger.h"
#include "util.h"

namespace dllm::communication {
template <>
Task AllReduce<NCCL>::run(
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
  return Task{[=, futureSend = tensorSend->future](const Context *context) {
    util::waitFutureIfValid(futureSend);
    CHECK_NCCL(ncclAllReduce(
        tensorSend->data(), tensorReceive->data(),
        cute::size(tensorSend->layout), toNcclDataType(tensorSend->dtype),
        toNcclRedOp(operation), tensorSend->comm, context->cudaStream));
    CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
  }};
}

template <>
Task AllReduce<NCCL>::runInplace(const std::shared_ptr<DTensor1D<NCCL>> &tensor,
                                 Operation operation) {
  if (tensor->deviceType != CUDA) {
    SPDLOG_LOGGER_CRITICAL(&logger(), "NCCL backend only supports CUDA tensor");
  }
  return Task{[=, future = tensor->future](const Context *context) {
    // Be careful: possible deadlock
    util::waitFutureIfValid(future);
    CHECK_NCCL(ncclAllReduce(
        tensor->data(), tensor->data(), cute::size(tensor->layout),
        toNcclDataType(tensor->dtype), toNcclRedOp(operation), tensor->comm,
        context->cudaStream));
    CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
  }};
}
}  // namespace dllm::communication
