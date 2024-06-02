#include "communication/all_to_all.h"

#include <mpi.h>
#include <nccl.h>
#include <torch/csrc/autograd/generated/variable_factories.h>

#include <limits>

#include "internal_utils.h"
#include "logger.h"
#include "nvtx_helper.h"
#include "tensor_friend.h"

namespace dllm::communication {
TaskNccl AllToAll<NCCL>::run(
    const std::vector<std::shared_ptr<Tensor>> &tensorReceive,
    const std::vector<std::shared_ptr<const ReadOnlyTensor>> &tensorSend) {
  std::vector<TensorFuture> futureReceive;
  futureReceive.reserve(tensorReceive.size());
  for (const auto &t : tensorReceive) {
    futureReceive.push_back(t->future());
  }
  std::vector<TaskFuture> futureSend;
  futureSend.reserve(futureSend.size());
  for (const auto &t : tensorSend) {
    futureSend.push_back(t->future());
  }
  auto task = TaskNccl{[tensorSend = tensorSend, tensorReceive = tensorReceive,
                        futureReceive = std::move(futureReceive),
                        futureSend = std::move(futureSend)](
                           const ContextNccl *context) mutable {
    DLLM_NVTX_RANGE_FN("dllm::communication::AllToAll<NCCL>::run");
    {
      for (auto &f : futureReceive) {
        util::FutureGuard sendGuard{f};
      }
      for (auto &f : futureSend) {
        util::FutureGuard sendGuard{f};
      }
      std::vector<at::Tensor> vReceive;
      vReceive.reserve(tensorReceive.size());
      for (const auto &t : tensorReceive) {
        DLLM_ASSERT_TRUE(DLLM_EXTRACT_TENSOR(t).device().type() == at::kCUDA,
                         "NCCL backend only support CUDA GPUs");
        vReceive.push_back(DLLM_EXTRACT_TENSOR(t));
      }

      std::vector<at::Tensor> vSend;
      vSend.reserve(tensorSend.size());
      for (const auto &t : tensorSend) {
        DLLM_ASSERT_TRUE(DLLM_EXTRACT_TENSOR(t).device().type() == at::kCUDA,
                         "NCCL backend only support CUDA GPUs");
        vSend.push_back(DLLM_EXTRACT_TENSOR(t));
      }

      context->backend->alltoall(vReceive, vSend)->wait();
      CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
    }
    tensorSend.clear();
    tensorReceive.clear();
  }};
  const TaskFuture future = task.get_future();
  for (const auto &t : tensorSend) {
    t->resetFuture(future);
  }
  for (const auto &t : tensorReceive) {
    t->resetFuture(future);
  }
  return task;
}
}  // namespace dllm::communication
