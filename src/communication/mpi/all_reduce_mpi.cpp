#include <mpi.h>

#include "communication/all_reduce.h"
#include "dtensor_mpi.h"
#include "logger.h"

namespace dllm::communication {
template <>
Task AllReduce<MPI>::run(
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
        return 0;
    }
  }();

  return Task{[=](const Context *context) {
    tensorSend->waitFutureIfValid();
    CHECK_MPI(MPI_Allreduce(tensorSend->data(), tensorReceive->data(),
                            cute::size(tensorSend->layout), datatype,
                            toMpiOp(operation), tensorSend->comm));
  }};
}

template <>
Task AllReduce<MPI>::runInplace(const std::shared_ptr<DTensor1D<MPI>> &tensor,
                                Operation operation) {
  MPI_Datatype datatype = [&]() {
    switch (tensor->dtype) {
      case R_64F:
        return MPI_DOUBLE;
      case R_32F:
        return MPI_FLOAT;
      default:
        SPDLOG_LOGGER_CRITICAL(&logger(),
                               "Not supported MPI all-reduce datatype");
        return 0;
    }
  }();

  return Task{[=](const Context *context) {
    // Be careful: possible deadlock
    tensor->waitFutureIfValid();
    CHECK_MPI(MPI_Allreduce(MPI_IN_PLACE, tensor->data(),
                            cute::size(tensor->layout), datatype,
                            toMpiOp(operation), tensor->comm));
  }};
}
}  // namespace dllm::communication