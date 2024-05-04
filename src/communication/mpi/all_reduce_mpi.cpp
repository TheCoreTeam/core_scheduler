#include <mpi.h>

#include "communication/mpi/all_reduce.h"
#include "dtensor_mpi.h"
#include "logger.h"
#include "util.h"

namespace dllm::communication {
TaskMpi AllReduce<MPI>::run(
    const std::shared_ptr<const DTensor1D<MPI>> &tensorSend,
    const std::shared_ptr<DTensor1D<MPI>> &tensorReceive, Operation operation) {
  if (cute::size(tensorSend->layout) != cute::size(tensorReceive->layout)) {
    SPDLOG_LOGGER_CRITICAL(&logger(),
                           "sendbuff's size is different from the recvbuff's");
  }
  if (tensorSend->dtype != tensorReceive->dtype) {
    SPDLOG_LOGGER_CRITICAL(&logger(),
                           "sendbuff's dtype is different from the recvbuff's");
  }

  MPI_Datatype datatype = [&]() {
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

  return TaskMpi{[=, future = tensorSend->future](const ContextMpi *context) {
    util::waitFutureIfValid(future);
    CHECK_MPI(MPI_Allreduce(tensorSend->data(), tensorReceive->data(),
                            cute::size(tensorSend->layout), datatype,
                            toMpiOp(operation), context->mpiComm));
  }};
}

TaskMpi AllReduce<MPI>::runInplace(
    const std::shared_ptr<DTensor1D<MPI>> &tensor, Operation operation) {
  MPI_Datatype datatype = [&]() {
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

  return TaskMpi{[=, future = tensor->future](const ContextMpi *context) {
    // Be careful: possible deadlock
    util::waitFutureIfValid(future);
    CHECK_MPI(MPI_Allreduce(MPI_IN_PLACE, tensor->data(),
                            cute::size(tensor->layout), datatype,
                            toMpiOp(operation), context->mpiComm));
  }};
}
}  // namespace dllm::communication