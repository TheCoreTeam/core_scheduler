#include "communication/all_to_all.h"

#include <mpi.h>
#include <nccl.h>
#include <torch/csrc/autograd/generated/variable_factories.h>

#include <limits>

#include "internal_utils.h"
#include "logger.h"
#include "nvtx_helper.h"
#include "tensor_impl.h"
#include "threading/scheduler_impl.h"

namespace dllm::communication {
void AllToAll<NCCL>::run(const Scheduler &scheduler,
                         const std::vector<Tensor> &tensorReceive,
                         const std::vector<ReadOnlyTensor> &tensorSend) {
  std::vector<decltype(utils::future(tensorReceive[0]))> futureReceive;
  futureReceive.reserve(tensorReceive.size());
  for (const auto &t : tensorReceive) {
    futureReceive.push_back(utils::future(t));
  }
  std::vector<TaskFuture> futureSend;
  futureSend.reserve(futureSend.size());
  for (const auto &t : tensorSend) {
    futureSend.push_back(utils::future(t));
  }
  auto task = TaskNccl{
      [tensorSend = tensorSend, tensorReceive = tensorReceive,
       futureReceive = std::move(futureReceive),
       futureSend = std::move(futureSend)](const ContextNccl *context) mutable {
        DLLM_NVTX_RANGE_FN("dllm::communication::AllToAll<NCCL>::run");
        {
          for (auto &f : futureReceive) {
            utils::FutureGuard sendGuard{f};
          }
          for (auto &f : futureSend) {
            utils::FutureGuard sendGuard{f};
          }
          std::vector<at::Tensor> vReceive;
          vReceive.reserve(tensorReceive.size());
          for (const auto &t : tensorReceive) {
            DLLM_ASSERT_TRUE(t.impl()->tensor().device().type() == at::kCUDA,
                             "NCCL backend only support CUDA GPUs");
            vReceive.push_back(t.impl()->tensor());
          }

          std::vector<at::Tensor> vSend;
          vSend.reserve(tensorSend.size());
          for (const auto &t : tensorSend) {
            DLLM_ASSERT_TRUE(t.impl()->tensor().device().type() == at::kCUDA,
                             "NCCL backend only support CUDA GPUs");
            vSend.push_back(t.impl()->tensor());
          }

          context->backend->alltoall(vReceive, vSend)->wait();
          CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
        }
        tensorSend.clear();
        tensorReceive.clear();
      }};
  const TaskFuture future = task.get_future();
  for (const auto &t : tensorSend) {
    utils::resetFuture(t, future);
  }
  for (const auto &t : tensorReceive) {
    utils::resetFuture(t, future);
  }
  scheduler.impl()->submit(std::move(task));
}
}  // namespace dllm::communication
