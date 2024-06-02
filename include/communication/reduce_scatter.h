#pragma once
#include "communication.h"
#include "tensor.h"
#include "threading/task_mpi.h"
#include "threading/task_nccl.h"

namespace dllm::communication {
template <Backend backend>
struct ReduceScatter;

template <>
struct ReduceScatter<NCCL> {
  static TaskNccl run(
      const std::shared_ptr<Tensor> &tensorReceive,
      const std::vector<std::shared_ptr<const ReadOnlyTensor>> &tensorSend,
      Operation operation);

  static TaskNccl run(
      const std::vector<std::shared_ptr<Tensor>> &tensorReceive,
      const std::vector<std::vector<std::shared_ptr<const ReadOnlyTensor>>>
          &tensorSend,
      Operation operation);
};

// template <>
// struct ReduceScatter<MPI> {
//   static TaskMpi run(const std::shared_ptr<Tensor> &tensorReceive,
//                      const std::shared_ptr<const ReadOnlyTensor> &tensorSend,
//                      int64_t receiveCount, Operation operation);
// };
}  // namespace dllm::communication
