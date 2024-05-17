#include "communication/mpi/all_to_all.h"

#include <mpi.h>

#include "dtensor_mpi.h"
#include "logger.h"
#include "util.h"

namespace dllm::communication {
TaskMpi AllToAll<MPI>::run(const std::shared_ptr<const Tensor1D> &tensorSend,
                           const std::shared_ptr<Tensor1D> &tensorReceive) {
  if (tensorSend->dtype != tensorReceive->dtype) {
    SPDLOG_LOGGER_CRITICAL(&logger(),
                           "sendbuff's dtype is different from the recvbuff's");
  }

  auto task = TaskMpi{[tensorSend = tensorSend, tensorReceive = tensorReceive,
                       futureReceive = *tensorReceive->future,
                       futureSend = tensorSend->future->wFuture](
                          const ContextMpi *context) mutable {
    const MPI_Datatype sendtype = [&]() {
      switch (tensorSend->dtype) {
        case R_64F:
          return MPI_DOUBLE;
        case R_32F:
          return MPI_FLOAT;
        default:
          SPDLOG_LOGGER_CRITICAL(&logger(),
                                 "Not supported MPI all-reduce datatype");
          return reinterpret_cast<MPI_Datatype>(0);
      }
    }();
    const MPI_Datatype recvtype = [&]() {
      switch (tensorReceive->dtype) {
        case R_64F:
          return MPI_DOUBLE;
        case R_32F:
          return MPI_FLOAT;
        default:
          SPDLOG_LOGGER_CRITICAL(&logger(),
                                 "Not supported MPI all-reduce datatype");
          return reinterpret_cast<MPI_Datatype>(0);
      }
    }();
    util::FutureGuard guardSend{futureSend};
    util::FutureGuard guardRReceive{futureReceive.rFuture};
    util::FutureGuard guardWReceive{futureReceive.wFuture};
    CHECK_MPI(MPI_Alltoall_c(
        tensorSend->data(), cute::size(tensorSend->layout) / context->commSize,
        sendtype, tensorReceive->data(),
        cute::size(tensorReceive->layout) / context->commSize, recvtype,
        context->mpiComm));
    tensorSend.reset();
    tensorReceive.reset();
  }};
  const TaskFuture future = task.get_future();
  tensorSend->future->rFuture = future;
  tensorReceive->future->wFuture = future;
  return task;
}

TaskMpi AllToAll<MPI>::runInplace(const std::shared_ptr<Tensor1D> &tensor) {
  auto task = TaskMpi{[tensor = tensor, future = *tensor->future](
                          const ContextMpi *context) mutable {
    // Be careful: possible deadlock
    const MPI_Datatype datatype = [&]() {
      switch (tensor->dtype) {
        case R_64F:
          return MPI_DOUBLE;
        case R_32F:
          return MPI_FLOAT;
        default:
          SPDLOG_LOGGER_CRITICAL(&logger(),
                                 "Not supported MPI all-reduce datatype");
          return reinterpret_cast<MPI_Datatype>(0);
      }
    }();
    util::FutureGuard rGuard{future.rFuture};
    util::FutureGuard wGuard{future.wFuture};
    CHECK_MPI(MPI_Alltoall_c(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, tensor->data(),
                             cute::size(tensor->layout) / context->commSize,
                             datatype, context->mpiComm));
    tensor.reset();
  }};
  const TaskFuture future = task.get_future();
  tensor->future->rFuture = future;
  tensor->future->wFuture = future;
  return task;
}
}  // namespace dllm::communication