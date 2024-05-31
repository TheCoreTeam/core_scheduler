#include "communication/reduce_scatter.h"

#include <nccl.h>
#include <torch/csrc/autograd/generated/variable_factories.h>

#include "internal_utils.h"
#include "logger.h"
#include "nvtx_helper.h"
#include "tensor_friend.h"

namespace dllm::communication {
namespace {
constexpr ncclRedOp_t toNcclRedOp(const Operation operation) {
  switch (operation) {
    case SUM:
      return ncclSum;
    default:
      DLLM_ASSERT_TRUE(false, "unsupported operation for NCCL all reduce");
  }
}

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

TaskNccl ReduceScatter<NCCL>::run(
    const std::shared_ptr<Tensor> &tensorReceive,
    const std::shared_ptr<const ReadOnlyTensor> &tensorSend,
    const int64_t receiveCount, Operation operation) {
  DLLM_ASSERT_TRUE(!DLLM_EXTRACT_TENSOR(tensorReceive).defined() ||
                       DLLM_EXTRACT_TENSOR(tensorReceive).is_contiguous(),
                   "receive tensor not contiguout");
  auto task = TaskNccl{[operation = operation, receiveCount = receiveCount,
                        tensorSend = tensorSend, tensorReceive = tensorReceive,
                        futureReceive = tensorReceive->future(),
                        futureSend = tensorSend->future()](
                           const ContextNccl *context) mutable {
    DLLM_NVTX_RANGE_FN("dllm::communication::ReduceScatter<NCCL>::run");
    {
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
      CHECK_NCCL(ncclReduceScatter(
          tensorSendContiguout.data_ptr(),
          DLLM_EXTRACT_TENSOR(tensorReceive).data_ptr(), tensorReceive->numel(),
          toNcclDataType(DLLM_EXTRACT_TENSOR(tensorSend).scalar_type()),
          toNcclRedOp(operation), context->ncclComm, context->cudaStream));
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
}  // namespace dllm::communication
