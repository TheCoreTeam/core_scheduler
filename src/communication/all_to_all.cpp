#include "communication/all_to_all.h"

#include <torch/csrc/autograd/generated/variable_factories.h>

#include <limits>

#include "communication_impl.h"
#include "internal_utils.h"
#include "logger.h"
#include "nvtx_helper.h"
#include "tensor_impl.h"
#include "threading/scheduler_impl.h"
#include "threading/task_impl.h"

namespace dllm::communication {
void AllToAll::run(const Scheduler &scheduler, const Comm &comm,
                   const std::vector<Tensor> &tensorReceive,
                   const std::vector<ReadOnlyTensor> &tensorSend) {
  struct Impl : Task::Impl {
    const Comm comm;

    explicit Impl(std::vector<Tensor> output /* tensors */,
                  std::vector<ReadOnlyTensor> input /* tensors */, Comm comm)
        : Task::Impl{std::move(output), std::move(input), nccl},
          comm{std::move(comm)} {}
    void operator()() const override {
      std::vector<at::Tensor> vReceive;
      vReceive.reserve(output().size());
      for (const auto &t : output()) {
        DLLM_ASSERT_TRUE(t.impl()->tensor().device().type() == at::kCUDA,
                         "NCCL backend only support CUDA GPUs");
        vReceive.push_back(t.impl()->tensor());
      }

      std::vector<at::Tensor> vSend;
      vSend.reserve(input().size());
      for (const auto &t : input()) {
        DLLM_ASSERT_TRUE(t.impl()->tensor().device().type() == at::kCUDA,
                         "NCCL backend only support CUDA GPUs");
        vSend.push_back(t.impl()->tensor());
      }

      comm.impl()->backend()->alltoall(vReceive, vSend)->wait();
    }
    [[nodiscard]] const char *name() const override {
      return "dllm::communication::AllToAll::run";
    }
  };

  scheduler.impl()->submit(
      Task{std::make_shared<Impl>(Impl{tensorReceive, tensorSend, comm})});
}
}  // namespace dllm::communication
