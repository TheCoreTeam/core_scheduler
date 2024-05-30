#include "communication/all_reduce.h"

#include <mpi.h>
#include <tensor_friend.h>
#include <torch/csrc/autograd/generated/variable_factories.h>

#include "internal_utils.h"
#include "logger.h"
#include "nvtx_helper.h"

namespace dllm::communication {
namespace {
constexpr auto toMPIOp(const Operation operation) {
  switch (operation) {
    case SUM:
      return MPI_SUM;
    default:
      DLLM_ASSERT_TRUE(false, "unsupported operation for MPI all reduce");
  }
}

auto toMPIDataType(const at::ScalarType dtype) {
  switch (dtype) {
    case at::kDouble:
      return MPI_DOUBLE;
    case at::kFloat:
      return MPI_FLOAT;
    default:
      DLLM_ASSERT_TRUE(false, "unsupported data type for MPI all reduce");
  }
}
}  // namespace

TaskMpi AllReduce<MPI>::run(
    const std::shared_ptr<Tensor> &tensorReceive,
    const std::shared_ptr<const ReadOnlyTensor> &tensorSend,
    const Operation operation) {
  DLLM_ASSERT_TRUE(tensorReceive->sizes() == tensorSend->sizes(),
                   "sendbuff is not the same as the recvbuff");
  auto task = TaskMpi{
      [tensorSend = tensorSend, tensorReceive = tensorReceive,
       operation = operation, futureReceive = tensorReceive->future(),
       futureSend = tensorSend->future()](const ContextMpi *context) mutable {
        DLLM_NVTX_RANGE_FN("dllm::communication::AllReduce<MPI>::run");
        {
          util::FutureGuard guardSend{futureSend};
          util::FutureGuard guardReceive{futureReceive};
          if (!DLLM_EXTRACT_TENSOR(tensorReceive).defined()) {
            DLLM_EXTRACT_TENSOR(tensorReceive) =
                torch::empty_like(DLLM_EXTRACT_TENSOR(tensorSend));
          }
          DLLM_ASSERT_TRUE(DLLM_EXTRACT_TENSOR(tensorSend).scalar_type() ==
                               DLLM_EXTRACT_TENSOR(tensorReceive).scalar_type(),
                           "All Reduce datatype mismatches");
          const int64_t count = DLLM_EXTRACT_TENSOR(tensorSend).numel();
          DLLM_ASSERT_TRUE(count <= std::numeric_limits<int>::max(),
                           "Do not support very large message");
          if (!DLLM_EXTRACT_TENSOR(tensorReceive).is_contiguous()) {
            DLLM_EXTRACT_TENSOR(tensorReceive) =
                DLLM_EXTRACT_TENSOR(tensorReceive).contiguous();
          }
          const auto tensorSendContiguout =
              DLLM_EXTRACT_TENSOR(tensorSend).contiguous();
          CHECK_MPI(MPI_Allreduce(
              tensorSendContiguout.data_ptr(),
              DLLM_EXTRACT_TENSOR(tensorReceive).data_ptr(), count,
              toMPIDataType(DLLM_EXTRACT_TENSOR(tensorSend).scalar_type()),
              toMPIOp(operation), context->mpiComm));
        }
        tensorSend.reset();
        tensorReceive.reset();
      }};
  const TaskFuture future = task.get_future();
  tensorSend->resetFuture(future);
  tensorReceive->resetFuture(future);
  tensorReceive->sizes() = tensorSend->sizes();
  return task;
}

TaskMpi AllReduce<MPI>::runInplace(const std::shared_ptr<Tensor> &tensor,
                                   Operation operation) {
  auto task =
      TaskMpi{[tensor = tensor, operation = operation,
               future = tensor->future()](const ContextMpi *context) mutable {
        DLLM_NVTX_RANGE_FN("dllm::communication::AllReduce<MPI>::runInplace");
        util::FutureGuard guard{future};
        const int64_t count = DLLM_EXTRACT_TENSOR(tensor).numel();
        DLLM_ASSERT_TRUE(count <= std::numeric_limits<int>::max(),
                         "Do not support very large message");
        if (!DLLM_EXTRACT_TENSOR(tensor).is_contiguous()) {
          DLLM_EXTRACT_TENSOR(tensor) =
              DLLM_EXTRACT_TENSOR(tensor).contiguous();
        }
        DLLM_WARN_TRUE(DLLM_EXTRACT_TENSOR(tensor).is_cpu(),
                       "MPI non CPU version is very slow");
        CHECK_MPI(MPI_Allreduce(
            MPI_IN_PLACE, DLLM_EXTRACT_TENSOR(tensor).data_ptr(), count,
            toMPIDataType(DLLM_EXTRACT_TENSOR(tensor).scalar_type()),
            toMPIOp(operation), context->mpiComm));
        tensor.reset();
      }};
  tensor->resetFuture(task.get_future());
  return task;
}
}  // namespace dllm::communication