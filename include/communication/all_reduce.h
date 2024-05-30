#pragma once
#include "communication.h"
#include "tensor.h"
#include "threading/task_mpi.h"
#include "threading/task_nccl.h"

namespace dllm::communication {
template <Backend backend>
struct AllReduce;

template <>
struct AllReduce<NCCL> {
  static TaskNccl run(const std::shared_ptr<Tensor> &tensorReceive,
                      const std::shared_ptr<const ReadOnlyTensor> &tensorSend,
                      Operation operation);

  static TaskNccl runInplace(const std::shared_ptr<Tensor> &tensor,
                             Operation operation);
};

template <>
struct AllReduce<MPI> {
  static TaskMpi run(const std::shared_ptr<Tensor> &tensorReceive,
                     const std::shared_ptr<const ReadOnlyTensor> &tensorSend,
                     Operation operation);

  static TaskMpi runInplace(const std::shared_ptr<Tensor> &tensor,
                            Operation operation);
};
}  // namespace dllm::communication
