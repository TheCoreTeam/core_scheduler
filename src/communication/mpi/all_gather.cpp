#include "communication/all_gather.h"

#include <mpi.h>
#include <tensor_friend.h>
#include <torch/csrc/autograd/generated/variable_factories.h>

#include "internal_utils.h"
#include "logger.h"
#include "nvtx_helper.h"

namespace dllm::communication {
TaskMpi AllGather<MPI>::run(
    const std::shared_ptr<Tensor> &tensorReceive,
    const std::shared_ptr<const ReadOnlyTensor> &tensorSend,
    const int64_t receiveCount) {
  auto task = TaskMpi{[tensorSend = tensorSend, tensorReceive = tensorReceive,
                       futureReceive = tensorReceive->future(),
                       futureSend = tensorSend->future()](
                          const ContextMpi *context) mutable {
    DLLM_NVTX_RANGE_FN("dllm::communication::AllGather<MPI>::run");
    utils::FutureGuard guardReceive{futureReceive};
    utils::FutureGuard guardSend{futureSend};
    if (!DLLM_EXTRACT_TENSOR(tensorReceive).defined()) {
      DLLM_EXTRACT_TENSOR(tensorReceive) = torch::empty(
          {tensorReceive.numel()}, DLLM_EXTRACT_TENSOR(tensorSend).options());
    }
    int64_t byteScaleSend;
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        DLLM_EXTRACT_TENSOR(tensorSend).scalar_type(), "Find size in byte",
        [&] { byteScaleSend = sizeof(scalar_t); });
    const int64_t byteSend =
        DLLM_EXTRACT_TENSOR(tensorSend).numel() * byteScaleSend;
    DLLM_ASSERT_TRUE(byteSend <= std::numeric_limits<int>::max(),
                     "Do not support very large message");
    int64_t byteScaleReceive;
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        DLLM_EXTRACT_TENSOR(tensorReceive).scalar_type(), "Find size in byte",
        [&] { byteScaleReceive = sizeof(scalar_t); });
    const int64_t byteReceive =
        DLLM_EXTRACT_TENSOR(tensorReceive).numel() * byteScaleReceive;
    DLLM_ASSERT_TRUE(byteReceive <= std::numeric_limits<int>::max(),
                     "Do not support very large message");
    const auto tensorSendContiguous =
        DLLM_EXTRACT_TENSOR(tensorSend).contiguous();
    if (!DLLM_EXTRACT_TENSOR(tensorReceive).is_contiguous()) {
      DLLM_EXTRACT_TENSOR(tensorReceive) =
          DLLM_EXTRACT_TENSOR(tensorReceive).contiguous();
    }
    CHECK_MPI(MPI_Allgather(tensorSendContiguous.data_ptr(), byteSend, MPI_BYTE,
                            DLLM_EXTRACT_TENSOR(tensorReceive).data_ptr(),
                            byteReceive, MPI_BYTE, context->mpiComm));
    tensorSend.reset();
    tensorReceive.reset();
  }};
  const TaskFuture future = task.get_future();
  tensorSend->resetFuture(future);
  tensorReceive->resetFuture(future);
  tensorReceive->sizes() = IntArray{receiveCount};
  return task;
}

TaskMpi AllGather<MPI>::runInplace(const std::shared_ptr<Tensor> &tensor) {
  auto task = TaskMpi{[tensor = tensor, future = utils::future(tensor)](
                          const ContextMpi *context) mutable {
    DLLM_NVTX_RANGE_FN("dllm::communication::AllGather<MPI>::runInplace");
    utils::FutureGuard guard{future};
    int64_t byteScale;
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        tensor.impl()->tensor().scalar_type(), "Find size in byte",
        [&] { byteScale = sizeof(scalar_t); });
    const int64_t byte = tensor.numel() * byteScale;
    DLLM_ASSERT_TRUE(byte <= std::numeric_limits<int>::max(),
                     "Do not support very large message");
    DLLM_ASSERT_TRUE(byte % context->commSize == 0,
                     "transfer volume {} is not dividable by commSize {}", byte,
                     context->commSize);
    if (!tensor.impl()->tensor().is_contiguous()) {
      tensor.impl()->tensor() = tensor.impl()->tensor().contiguous();
    }
    DLLM_WARN_TRUE(tensor.impl()->tensor().is_cpu(),
                   "MPI non CPU version is very slow");
    CHECK_MPI(MPI_Allgather(
        MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, tensor.impl()->tensor().data_ptr(),
        byte / context->commSize, MPI_BYTE, context->mpiComm));
    tensor.reset();
  }};
  tensor->resetFuture(task.get_future());
  return task;
}
}  // namespace dllm::communication
