#include "communication/mpi/all_reduce.h"

#include <mpi.h>

#include "dtensor_mpi.h"
#include "logger.h"
#include "util.h"

namespace dllm::communication {
TaskMpi AllReduce<MPI>::run(const std::shared_ptr<const Tensor1D> &tensorSend,
                            const std::shared_ptr<Tensor1D> &tensorReceive,
                            const Operation operation) {
  if (cute::size(tensorSend->layout) != cute::size(tensorReceive->layout)) {
    SPDLOG_LOGGER_CRITICAL(&logger(),
                           "sendbuff's size is different from the recvbuff's");
  }
  if (tensorSend->dtype != tensorReceive->dtype) {
    SPDLOG_LOGGER_CRITICAL(&logger(),
                           "sendbuff's dtype is different from the recvbuff's");
  }

  auto task = TaskMpi{
      [tensorSend = tensorSend, tensorReceive = tensorReceive,
       operation = operation, futureReceive = *tensorReceive->future,
       futureSend = *tensorSend->future](const ContextMpi *context) mutable {
        const MPI_Datatype datatype = [&]() {
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
        util::FutureGuard{futureSend};
        util::FutureGuard{futureReceive};
        CHECK_MPI(MPI_Allreduce_c(tensorSend->data(), tensorReceive->data(),
                                  cute::size(tensorSend->layout), datatype,
                                  util::toMpiOp(operation), context->mpiComm));
        tensorSend.reset();
        tensorReceive.reset();
      }};
  const auto &future = *tensorReceive->future = task.get_future();
  *tensorSend->future = future;
  return task;
}

TaskMpi AllReduce<MPI>::runInplace(const std::shared_ptr<Tensor1D> &tensor,
                                   Operation operation) {
  auto task =
      TaskMpi{[tensor = tensor, operation = operation,
               future = *tensor->future](const ContextMpi *context) mutable {
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
        // Be careful: possible deadlock
        util::FutureGuard{future};
        CHECK_MPI(MPI_Allreduce_c(MPI_IN_PLACE, tensor->data(),
                                  cute::size(tensor->layout), datatype,
                                  util::toMpiOp(operation), context->mpiComm));
        tensor.reset();
      }};
  *tensor->future = task.get_future();
  return task;
}
}  // namespace dllm::communication