#include "communication/all_gather.h"

#include <torch/csrc/autograd/generated/variable_factories.h>

#include "communication_impl.h"
#include "logger.h"
#include "tensor_impl.h"
#include "threading/scheduler_impl.h"
#include "threading/task_impl.h"

namespace dllm::communication {
void AllGather::run(const Scheduler &scheduler, const Comm &comm,
                    const std::vector<std::vector<Tensor>> &tensorReceive,
                    const std::vector<ReadOnlyTensor> &tensorSend) {
  struct Impl : Task::Impl {
    std::vector<std::vector<Tensor>> actualReceive;
    const Comm comm;

    explicit Impl(std::vector<Tensor> output /* tensors */,
                  std::vector<ReadOnlyTensor> input /* tensors */,
                  std::vector<std::vector<Tensor>> actualReceive, Comm comm)
        : Task::Impl{std::move(output), std::move(input), nccl},
          actualReceive{std::move(actualReceive)},
          comm{std::move(comm)} {}
    void operator()() const override {
      std::vector<at::Tensor> vSend;
      vSend.reserve(input().size());
      for (const auto &t : input()) {
        DLLM_ASSERT_TRUE(t.impl()->tensor().device().type() == at::kCUDA,
                         "NCCL backend only support CUDA GPUs");
        vSend.push_back(t.impl()->tensor());
      }
      std::vector<std::vector<at::Tensor>> vReceive;
      vReceive.reserve(actualReceive.size());
      for (const auto &vt : actualReceive) {
        std::vector<at::Tensor> v;
        v.reserve(vt.size());
        for (const auto &t : vt) {
          DLLM_ASSERT_TRUE(t.impl()->tensor().device().type() == at::kCUDA,
                           "NCCL backend only support CUDA GPUs");
          v.push_back(t.impl()->tensor());
        }
        vReceive.push_back(std::move(v));
      }
      comm.impl()->backend()->allgather(vReceive, vSend)->wait();
    }
    [[nodiscard]] const char *name() const override {
      return "dllm::communication::AllGather::run";
    }
  };

  std::vector<Tensor> output;
  output.reserve(tensorReceive.size() * tensorReceive[0].size());
  for (auto &v : tensorReceive) {
    for (auto &t : v) {
      output.push_back(t);
    }
  }

  scheduler.impl()->submit(Task{
      std::make_shared<Impl>(Impl{output, tensorSend, tensorReceive, comm})});
}
}  // namespace dllm::communication
