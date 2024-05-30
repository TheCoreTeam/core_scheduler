#pragma once
#include "communication.h"
#include "tensor.h"
#include "threading/task_mpi.h"
#include "threading/task_nccl.h"

namespace dllm::communication {
template <Backend backend>
struct AllGather;

template <>
struct AllGather<NCCL> {
  static TaskNccl run(const std::shared_ptr<Tensor> &tensorReceive,
                      const std::shared_ptr<const ReadOnlyTensor> &tensorSend,
                      int64_t receiveCount);

  static TaskNccl runInplace(const std::shared_ptr<Tensor> &tensor);
};

template <>
struct AllGather<MPI> {
  static TaskMpi run(const std::shared_ptr<Tensor> &tensorReceive,
                     const std::shared_ptr<const ReadOnlyTensor> &tensorSend,
                     int64_t receiveCount);

  static TaskMpi runInplace(const std::shared_ptr<Tensor> &tensor);
};
}  // namespace dllm::communication
