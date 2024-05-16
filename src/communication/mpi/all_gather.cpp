#include "communication/mpi/all_gather.h"

#include <mpi.h>

#include "dtensor_mpi.h"
#include "logger.h"
#include "util.h"

namespace dllm::communication {
TaskMpi AllGather<MPI>::run(const std::shared_ptr<const Tensor1D> &tensorSend,
                            const std::shared_ptr<Tensor1D> &tensorReceive,
                            const std::size_t sendCount) {
  if (sendCount > cute::size(tensorSend->layout) ||
      sendCount > cute::size(tensorReceive->layout)) {
    SPDLOG_LOGGER_CRITICAL(&logger(), "sendSize larger than buffer size");
  }
  if (tensorSend->dtype != tensorReceive->dtype) {
    SPDLOG_LOGGER_CRITICAL(&logger(),
                           "sendbuff's dtype is different from the recvbuff's");
  }

  auto task = TaskMpi{
      [tensorSend = tensorSend, tensorReceive = tensorReceive,
       sendCount = sendCount, futureReceive = *tensorReceive->future,
       futureSend = *tensorSend->future](const ContextMpi *context) mutable {
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
        util::FutureGuard guardReceive{futureReceive};
        CHECK_MPI(MPI_Allgather_c(tensorSend->data(), sendCount, sendtype,
                                  tensorReceive->data(), sendCount, recvtype,
                                  context->mpiComm));
        tensorSend.reset();
        tensorReceive.reset();
      }};
  const auto &future = *tensorReceive->future = task.get_future();
  *tensorSend->future = future;
  return task;
}

TaskMpi AllGather<MPI>::runInplace(const std::shared_ptr<Tensor1D> &tensor,
                                   const std::size_t sendCount) {
  if (sendCount > cute::size(tensor->layout)) {
    SPDLOG_LOGGER_CRITICAL(&logger(), "sendSize larger than buffer size");
  }
  auto task =
      TaskMpi{[tensor = tensor, sendCount = sendCount,
               future = *tensor->future](const ContextMpi *context) mutable {
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
        util::FutureGuard guard{future};
        CHECK_MPI(MPI_Allgather_c(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                                  tensor->data(), sendCount, datatype,
                                  context->mpiComm));
        tensor.reset();
      }};
  *tensor->future = task.get_future();
  return task;
}
}  // namespace dllm::communication