#include "communication/all_to_all.h"

#include <ATen/Dispatch.h>
#include <mpi.h>
#include <nccl.h>
#include <torch/csrc/autograd/generated/variable_factories.h>

#include <limits>

#include "internal_utils.h"
#include "logger.h"
#include "nvtx_helper.h"
#include "tensor_friend.h"

namespace dllm::communication {
TaskNccl AllToAll<NCCL>::run(
    const std::shared_ptr<Tensor> &tensorReceive,
    const std::shared_ptr<const ReadOnlyTensor> &tensorSend,
    const int64_t commSize) {
  auto task = TaskNccl{[tensorSend = tensorSend, tensorReceive = tensorReceive,
                        futureReceive = tensorReceive->future(),
                        futureSend = tensorSend->future()](
                           const ContextNccl *context) mutable {
    DLLM_NVTX_RANGE_FN("dllm::communication::AllToAll<NCCL>::run");
    {
      util::FutureGuard sendGuard{futureSend};
      util::FutureGuard receiveGuard{futureReceive};
      int64_t byteScaleSend, byteScaleReceive;
      AT_DISPATCH_FLOATING_TYPES_AND2(
          at::ScalarType::Half, at::ScalarType::BFloat16,
          DLLM_EXTRACT_TENSOR(tensorSend).scalar_type(), "Find size in byte",
          [&] { byteScaleSend = sizeof(scalar_t); });
      const int64_t byteSend =
          DLLM_EXTRACT_TENSOR(tensorSend).numel() * byteScaleSend;
      if (!DLLM_EXTRACT_TENSOR(tensorReceive).defined()) {
        DLLM_EXTRACT_TENSOR(tensorReceive) = torch::empty(
            tensorSend->sizes(), DLLM_EXTRACT_TENSOR(tensorSend).options());
      }
      AT_DISPATCH_FLOATING_TYPES_AND2(
          at::ScalarType::Half, at::ScalarType::BFloat16,
          DLLM_EXTRACT_TENSOR(tensorReceive).scalar_type(), "Find size in byte",
          [&] { byteScaleReceive = sizeof(scalar_t); });
      const int64_t byteReceive =
          DLLM_EXTRACT_TENSOR(tensorReceive).numel() * byteScaleReceive;
      const auto tensorSendContiguout =
          DLLM_EXTRACT_TENSOR(tensorSend).contiguous();
      if (!DLLM_EXTRACT_TENSOR(tensorReceive).is_contiguous()) {
        DLLM_EXTRACT_TENSOR(tensorReceive) =
            DLLM_EXTRACT_TENSOR(tensorReceive).contiguous();
      }
      DLLM_ASSERT_TRUE(
          DLLM_EXTRACT_TENSOR(tensorReceive).device().type() == at::kCUDA,
          "NCCL backend only support CUDA GPUs");
      DLLM_ASSERT_TRUE(
          DLLM_EXTRACT_TENSOR(tensorSend).device().type() == at::kCUDA,
          "NCCL backend only support CUDA GPUs");
      CHECK_NCCL(ncclGroupStart());
      for (int r = 0; r < context->commSize; r++) {
        CHECK_NCCL(ncclSend(tensorSendContiguout.data_ptr(), byteSend, ncclInt8,
                            r, context->ncclComm, context->cudaStream));
        CHECK_NCCL(ncclRecv(DLLM_EXTRACT_TENSOR(tensorReceive).data_ptr(),
                            byteReceive, ncclInt8, r, context->ncclComm,
                            context->cudaStream));
      }
      CHECK_NCCL(ncclGroupEnd());
    }
    tensorSend.reset();
    tensorReceive.reset();
  }};
  const TaskFuture future = task.get_future();
  tensorSend->resetFuture(future);
  tensorReceive->resetFuture(future);
  tensorReceive->sizes() = [&] {
    auto sizes = tensorSend->sizes();
    sizes[0] *= commSize;
    return sizes;
  }();
  return task;
}

TaskNccl AllToAll<NCCL>::runInplace(const std::shared_ptr<Tensor> &tensor) {
  auto task = TaskNccl{[tensor = tensor, future = tensor->future()](
                           const ContextNccl *context) mutable {
    DLLM_NVTX_RANGE_FN("dllm::communication::AllToAll<NCCL>::runInplace");
    util::FutureGuard guard{future};
    int64_t byteScale;
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        DLLM_EXTRACT_TENSOR(tensor).scalar_type(), "Find size in byte",
        [&] { byteScale = sizeof(scalar_t); });
    const int64_t byte =
        DLLM_EXTRACT_TENSOR(tensor).numel() * byteScale / context->commSize;
    if (!DLLM_EXTRACT_TENSOR(tensor).is_contiguous()) {
      DLLM_EXTRACT_TENSOR(tensor) = DLLM_EXTRACT_TENSOR(tensor).contiguous();
    }
    DLLM_ASSERT_TRUE(DLLM_EXTRACT_TENSOR(tensor).device().type() == at::kCUDA,
                     "NCCL backend only support CUDA GPUs");
    CHECK_NCCL(ncclGroupStart());
    const auto base_ptr =
        static_cast<std::byte *>(DLLM_EXTRACT_TENSOR(tensor).data_ptr());
    for (int r = 0; r < context->commSize; r++) {
      if (r != context->ncclRank) {
        const auto *send_ptr = base_ptr + context->ncclRank * byte;
        auto *recv_ptr = base_ptr + r * byte;
        CHECK_NCCL(ncclSend(send_ptr, byte, ncclInt8, r, context->ncclComm,
                            context->cudaStream));
        CHECK_NCCL(ncclRecv(recv_ptr, byte, ncclInt8, r, context->ncclComm,
                            context->cudaStream));
      }
    }
    CHECK_NCCL(ncclGroupEnd());
    tensor.reset();
  }};
  tensor->resetFuture(task.get_future());
  return task;
}
}  // namespace dllm::communication
