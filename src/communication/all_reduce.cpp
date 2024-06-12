#include "communication/all_reduce.h"

#include <torch/csrc/autograd/generated/variable_factories.h>

#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>

#include "communication/communication_impl.h"
#include "logger.h"
#include "tensor_impl.h"
#include "threading/scheduler_impl.h"
#include "threading/task_impl.h"

namespace dllm::communication {
namespace {
constexpr auto toC10dRedOp(const Operation operation) {
  switch (operation) {
    case SUM:
      return c10d::ReduceOp::SUM;
    default:
      DLLM_ASSERT_TRUE(false, "unsupported operation for NCCL all reduce");
  }
}
}  // namespace

void AllReduce::runInplace(const Scheduler &scheduler, const Comm &comm,
                           const std::vector<Tensor> &tensors,
                           const Operation operation) {
  if (tensors.size() == 1) {
    struct Impl : Task::Impl {
      const Comm comm;
      const Operation operation;

      explicit Impl(std::vector<Tensor> output /* tensors */,
                    std::vector<ReadOnlyTensor> input /* tensors */, Comm comm,
                    const Operation operation)
          : Task::Impl{std::move(output), std::move(input), nccl},
            comm{std::move(comm)},
            operation{operation} {}
      void operator()() const override {
        std::vector<at::Tensor> v;
        v.reserve(output().size());
        for (const auto &t : output()) {
          DLLM_ASSERT_TRUE(t.impl()->tensor().device().type() == at::kCUDA,
                           "NCCL backend only support CUDA GPUs");
          v.push_back(t.impl()->tensor());
        }
        const auto work = comm.impl()->backend()->allreduce(
            v, c10d::AllreduceOptions{.reduceOp = toC10dRedOp(operation)});
        const auto castedWork =
            dynamic_cast<c10d::ProcessGroupNCCL::WorkNCCL *>(&*work);
        if (castedWork != nullptr) {
          castedWork->synchronizeStream();
        }
      }
      [[nodiscard]] const char *name() const override {
        return "dllm::communication::AllReduce::runInplace";
      }
    };

    scheduler.impl()->submit(Task{
        std::make_shared<Impl>(Impl{tensors, {tensors[0]}, comm, operation})});
  } else {
    struct Impl : Task::Impl {
      const Comm comm;
      const Operation operation;

      explicit Impl(std::vector<Tensor> output /* tensors */,
                    std::vector<ReadOnlyTensor> input /* tensors */, Comm comm,
                    const Operation operation)
          : Task::Impl{std::move(output), std::move(input), nccl},
            comm{std::move(comm)},
            operation{operation} {}
      void operator()() const override {
        std::vector<at::Tensor> v;
        v.reserve(output().size());
        std::vector<int64_t> sizes;
        sizes.reserve(output().size());
        for (const auto &t : output()) {
          DLLM_ASSERT_TRUE(t.impl()->tensor().device().type() == at::kCUDA,
                           "NCCL backend only support CUDA GPUs");
          auto &nakeTensor = t.impl()->tensor();
          if (!nakeTensor.is_contiguous()) {
            intermediate().push_back(nakeTensor);
            nakeTensor = nakeTensor.contiguous();
          }
          v.push_back(nakeTensor.view(-1));
          sizes.push_back(nakeTensor.numel());
        }
        std::vector vReduce{at::cat(v)};
        const auto work = comm.impl()->backend()->allreduce(
            {vReduce},
            c10d::AllreduceOptions{.reduceOp = toC10dRedOp(operation)});
        const auto castedWork =
            dynamic_cast<c10d::ProcessGroupNCCL::WorkNCCL *>(&*work);
        if (castedWork != nullptr) {
          castedWork->synchronizeStream();
        }
        intermediate().push_back(vReduce[0]);
        int64_t start = 0;
        for (std::size_t i = 0; i + 1 < sizes.size(); ++i) {
          v[i].copy_(vReduce[0].narrow(0, start, sizes[i]));
          start += sizes[i];
        }
      }
      [[nodiscard]] const char *name() const override {
        return "dllm::communication::AllReduce::runInplace";
      }
    };

    std::vector<ReadOnlyTensor> input;
    input.reserve(tensors.size());
    for (auto &t : tensors) {
      input.push_back(t);
    }

    scheduler.impl()->submit(
        Task{std::make_shared<Impl>(Impl{tensors, input, comm, operation})});
  }
}
}  // namespace dllm::communication
