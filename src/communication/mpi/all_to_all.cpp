#include "communication/all_to_all.h"

#include <ATen/Dispatch.h>
#include <mpi.h>

#include <limits>

#include "internal_utils.h"
#include "logger.h"
#include "tensor_friend.h"

namespace dllm::communication {
// TaskMpi AllToAll::run(const std::shared_ptr<Tensor> &tensorReceive,
//                       const std::shared_ptr<const ReadOnlyTensor>
//                       &tensorSend) {
//   auto task = TaskMpi{[tensorSend = tensorSend, tensorReceive =
//   tensorReceive,
//                        futureReceive = *tensorReceive->future,
//                        futureSend = tensorSend->future->wFuture](
//                           const ContextMpi *context) mutable {
//     const MPI_Datatype sendtype = [&]() {
//       switch (tensorSend->dtype) {
//         case R_64F:
//           return MPI_DOUBLE;
//         case R_32F:
//           return MPI_FLOAT;
//         default:
//           SPDLOG_LOGGER_CRITICAL(&logger(),
//                                  "Not supported MPI all-reduce datatype");
//           return reinterpret_cast<MPI_Datatype>(0);
//       }
//     }();
//     const MPI_Datatype recvtype = [&]() {
//       switch (tensorReceive->dtype) {
//         case R_64F:
//           return MPI_DOUBLE;
//         case R_32F:
//           return MPI_FLOAT;
//         default:
//           SPDLOG_LOGGER_CRITICAL(&logger(),
//                                  "Not supported MPI all-reduce datatype");
//           return reinterpret_cast<MPI_Datatype>(0);
//       }
//     }();
//     util::FutureGuard guardSend{futureSend};
//     util::FutureGuard guardRReceive{futureReceive.rFuture};
//     util::FutureGuard guardWReceive{futureReceive.wFuture};
//     CHECK_MPI(MPI_Alltoall_c(
//         tensorSend->data(), cute::size(tensorSend->layout) /
//         context->commSize, sendtype, tensorReceive->data(),
//         cute::size(tensorReceive->layout) / context->commSize, recvtype,
//         context->mpiComm));
//     tensorSend.reset();
//     tensorReceive.reset();
//   }};
//   const TaskFuture future = task.get_future();
//   tensorSend->future->rFuture = future;
//   tensorReceive->future->wFuture = future;
//   return task;
// }

TaskMpi AllToAll::runInplace(const std::shared_ptr<Tensor> &tensor) {
  auto task = TaskMpi{[tensor = tensor, future = tensor->future()](
                          const ContextMpi *context) mutable {
    // Be careful: possible deadlock
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
    CHECK_MPI(MPI_Alltoall(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                           DLLM_EXTRACT_TENSOR(tensor).data_ptr(), count,
                           MPI_BYTE, context->mpiComm));
    tensor.reset();
  }};
  tensor->resetFuture(task.get_future());
  return task;
}
}  // namespace dllm::communication