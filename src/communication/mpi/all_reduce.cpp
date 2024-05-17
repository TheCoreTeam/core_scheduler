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

  auto task =
      TaskMpi{[tensorSend = tensorSend, tensorReceive = tensorReceive,
               operation = operation, futureReceive = *tensorReceive->future,
               futureSend = tensorSend->future->rFuture](
                  const ContextMpi *context) mutable {
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
        util::FutureGuard guardSend{futureSend};
        util::FutureGuard guardRReceive{futureReceive.rFuture};
        util::FutureGuard guardWReceive{futureReceive.wFuture};
        CHECK_MPI(MPI_Allreduce_c(tensorSend->data(), tensorReceive->data(),
                                  cute::size(tensorSend->layout), datatype,
                                  util::toMpiOp(operation), context->mpiComm));
        tensorSend.reset();
        tensorReceive.reset();
      }};
  const TaskFuture future = task.get_future();
  tensorSend->future->rFuture = future;
  tensorReceive->future->wFuture = future;
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
        util::FutureGuard rGuard{future.rFuture};
        util::FutureGuard wGuard{future.wFuture};
        CHECK_MPI(MPI_Allreduce_c(MPI_IN_PLACE, tensor->data(),
                                  cute::size(tensor->layout), datatype,
                                  util::toMpiOp(operation), context->mpiComm));
        tensor.reset();
      }};
  const TaskFuture future = task.get_future();
  tensor->future->rFuture = future;
  tensor->future->wFuture = future;
  return task;
}
}  // namespace dllm::communication