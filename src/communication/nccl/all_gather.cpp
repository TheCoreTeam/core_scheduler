#include "communication/all_gather.h"

#include <nccl.h>
#include <torch/csrc/autograd/generated/variable_factories.h>

#include "internal_utils.h"
#include "logger.h"
#include "nvtx_helper.h"
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
      DLLM_ASSERT_TRUE(false, "unsupported data type for NCCL all gather");
  }
}
}  // namespace

TaskNccl AllGather<NCCL>::run(
    const std::shared_ptr<Tensor> &tensorReceive,
    const std::shared_ptr<const ReadOnlyTensor> &tensorSend,
    const int64_t receiveCount) {
  DLLM_ASSERT_TRUE(!DLLM_EXTRACT_TENSOR(tensorReceive).defined() ||
                       DLLM_EXTRACT_TENSOR(tensorReceive).is_contiguous(),
                   "receive tensor not contiguout");
  auto task = TaskNccl{
      [receiveCount = receiveCount, tensorSend = tensorSend,
       tensorReceive = tensorReceive, futureReceive = tensorReceive->future(),
       futureSend = tensorSend->future()](const ContextNccl *context) mutable {
        DLLM_NVTX_RANGE_FN("dllm::communication::AllGather<NCCL>::run");
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
          CHECK_NCCL(ncclAllGather(
              tensorSendContiguout.data_ptr(),
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
  auto task = TaskNccl{[tensor = tensor, future = tensor->future()](
                           const ContextNccl *context) mutable {
    DLLM_NVTX_RANGE_FN("dllm::communication::AllGather<NCCL>::runInplace");
    {
      const auto count = tensor->numel();
      util::FutureGuard guard{future};
      if (!DLLM_EXTRACT_TENSOR(tensor).is_contiguous()) {
        DLLM_EXTRACT_TENSOR(tensor) = DLLM_EXTRACT_TENSOR(tensor).contiguous();
      }
      DLLM_ASSERT_TRUE(count % context->commSize == 0,
                       "transfer volume {} is not dividable by commSize {}",
                       count, context->commSize);
      DLLM_ASSERT_TRUE(DLLM_EXTRACT_TENSOR(tensor).device().type() == at::kCUDA,
                       "NCCL backend only support CUDA GPUs");
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
