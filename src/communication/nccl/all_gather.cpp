#include "communication/all_gather.h"

#include <nccl.h>
#include <torch/csrc/autograd/generated/variable_factories.h>

#include "internal_utils.h"
#include "logger.h"
#include "nvtx_helper.h"
#include "tensor_impl.h"
#include "threading/scheduler_impl.h"

namespace dllm::communication {
void AllGather<NCCL>::run(const Scheduler &scheduler,
                          const std::vector<std::vector<Tensor>> &tensorReceive,
                          const std::vector<ReadOnlyTensor> &tensorSend) {
  std::vector<decltype(utils::future(tensorSend[0]))> futureSend;
  futureSend.reserve(tensorSend.size());
  for (const auto &t : tensorSend) {
    futureSend.push_back(utils::future(t));
  }
  std::vector<decltype(utils::future(tensorReceive[0][0]))> futureReceive;
  DLLM_ASSERT_TRUE(!tensorReceive.empty(), "Receive buffer is empty");
  futureReceive.reserve(tensorReceive.size() * tensorReceive[0].size());
  futureReceive.reserve(tensorReceive.size());
  for (const auto &vt : tensorReceive) {
    for (const auto &t : vt) {
      futureReceive.push_back(utils::future(t));
    }
  }
  auto task = TaskNccl{
      [tensorSend = tensorSend, tensorReceive = tensorReceive,
       futureReceive = std::move(futureReceive),
       futureSend = std::move(futureSend)](const ContextNccl *context) mutable {
        DLLM_NVTX_RANGE_FN("dllm::communication::AllGather<NCCL>::run");
        {
          for (auto &f : futureReceive) {
            utils::FutureGuard guardReceive{f};
          }
          for (auto &f : futureSend) {
            utils::FutureGuard guardReceive{f};
          }
          std::vector<at::Tensor> vSend;
          vSend.reserve(tensorSend.size());
          for (const auto &t : tensorSend) {
            DLLM_ASSERT_TRUE(t.impl()->tensor().device().type() == at::kCUDA,
                             "NCCL backend only support CUDA GPUs");
            vSend.push_back(t.impl()->tensor());
          }
          std::vector<std::vector<at::Tensor>> vReceive;
          vReceive.reserve(tensorReceive.size());
          for (const auto &vt : tensorReceive) {
            std::vector<at::Tensor> v;
            v.reserve(vt.size());
            for (const auto &t : vt) {
              DLLM_ASSERT_TRUE(t.impl()->tensor().device().type() == at::kCUDA,
                               "NCCL backend only support CUDA GPUs");
              v.push_back(t.impl()->tensor());
            }
            vReceive.push_back(std::move(v));
          }
          context->backend->allgather(vReceive, vSend)->wait();
          CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
        }
        tensorSend.clear();
        tensorReceive.clear();
      }};
  const TaskFuture future = task.get_future();
  for (const auto &t : tensorSend) {
    utils::resetFuture(t, future);
  }
  for (const auto &vt : tensorReceive) {
    for (const auto &t : vt) {
      utils::resetFuture(t, future);
    }
  }
  scheduler.impl()->submit(std::move(task));
}
}  // namespace dllm::communication
