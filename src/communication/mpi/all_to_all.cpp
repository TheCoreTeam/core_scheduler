#include "communication/all_to_all.h"

#include <ATen/Dispatch.h>
#include <mpi.h>
#include <torch/csrc/autograd/generated/variable_factories.h>

#include <limits>

#include "internal_utils.h"
#include "logger.h"
#include "nvtx_helper.h"
#include "tensor_friend.h"

namespace dllm::communication {
TaskMpi AllToAll<MPI>::run(
    const std::shared_ptr<Tensor> &tensorReceive,
    const std::shared_ptr<const ReadOnlyTensor> &tensorSend,
    const int64_t commSize) {
  auto task = TaskMpi{[tensorSend = tensorSend, tensorReceive = tensorReceive,
                       futureReceive = tensorReceive->future(),
                       futureSend = tensorSend->future()](
                          const ContextMpi *context) mutable {
    DLLM_NVTX_RANGE_FN("dllm::communication::AllToAll<MPI>::run");
    {
      util::FutureGuard sendGuard{futureSend};
      util::FutureGuard receiveGuard{futureReceive};
      int64_t byteScaleSend, byteScaleReceive;
      AT_DISPATCH_FLOATING_TYPES_AND2(
          at::ScalarType::Half, at::ScalarType::BFloat16,
          DLLM_EXTRACT_TENSOR(tensorSend).scalar_type(), "Find size in byte",
          [&] { byteScaleSend = sizeof(scalar_t); });
      const int64_t countSend =
          DLLM_EXTRACT_TENSOR(tensorSend).numel() * byteScaleSend;
      DLLM_ASSERT_TRUE(countSend <= std::numeric_limits<int>::max(),
                       "Do not support very large message");
      if (!DLLM_EXTRACT_TENSOR(tensorReceive).defined()) {
        DLLM_EXTRACT_TENSOR(tensorReceive) = torch::empty(
            tensorSend->sizes(), DLLM_EXTRACT_TENSOR(tensorSend).options());
      }
      AT_DISPATCH_FLOATING_TYPES_AND2(
          at::ScalarType::Half, at::ScalarType::BFloat16,
          DLLM_EXTRACT_TENSOR(tensorReceive).scalar_type(), "Find size in byte",
          [&] { byteScaleReceive = sizeof(scalar_t); });
      const int64_t countReceive =
          DLLM_EXTRACT_TENSOR(tensorReceive).numel() * byteScaleReceive;
      DLLM_ASSERT_TRUE(countSend <= std::numeric_limits<int>::max(),
                       "Do not support very large message");

      const auto tensorSendContiguout =
          DLLM_EXTRACT_TENSOR(tensorSend).contiguous();
      if (!DLLM_EXTRACT_TENSOR(tensorReceive).is_contiguous()) {
        DLLM_EXTRACT_TENSOR(tensorReceive) =
            DLLM_EXTRACT_TENSOR(tensorReceive).contiguous();
      }
      CHECK_MPI(MPI_Alltoall(tensorSendContiguout.data_ptr(), countSend,
                             MPI_BYTE,
                             DLLM_EXTRACT_TENSOR(tensorReceive).data_ptr(),
                             countReceive, MPI_BYTE, context->mpiComm));
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

TaskMpi AllToAll<MPI>::runInplace(const std::shared_ptr<Tensor> &tensor) {
  auto task = TaskMpi{[tensor = tensor, future = tensor->future()](
                          const ContextMpi *context) mutable {
    DLLM_NVTX_RANGE_FN("dllm::communication::AllToAll<MPI>::runInplace");
    util::FutureGuard guard{future};
    int64_t byteScale;
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        DLLM_EXTRACT_TENSOR(tensor).scalar_type(), "Find size in byte",
        [&] { byteScale = sizeof(scalar_t); });
    const int64_t count =
        DLLM_EXTRACT_TENSOR(tensor).numel() * byteScale / context->commSize;
    DLLM_ASSERT_TRUE(count <= std::numeric_limits<int>::max(),
                     "Do not support very large message");
    if (!DLLM_EXTRACT_TENSOR(tensor).is_contiguous()) {
      DLLM_EXTRACT_TENSOR(tensor) = DLLM_EXTRACT_TENSOR(tensor).contiguous();
    }
    DLLM_WARN_TRUE(DLLM_EXTRACT_TENSOR(tensor).is_cpu(),
                   "MPI non CPU version is very slow");
    CHECK_MPI(MPI_Alltoall(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                           DLLM_EXTRACT_TENSOR(tensor).data_ptr(), count,
                           MPI_BYTE, context->mpiComm));
    tensor.reset();
  }};
  tensor->resetFuture(task.get_future());
  return task;
}
}  // namespace dllm::communication
