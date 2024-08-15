/*
 * Copyright (c) 2024 The Core team
 *
 * Licensed under the Apache License, Version 2.0;
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an 'AS IS' BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "communication/all_reduce.h"

#include <torch/csrc/autograd/generated/variable_factories.h>

#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>

#include "communication/communication_impl.h"
#include "logger.h"
#include "tensor_impl.h"
#include "threading/scheduler_impl.h"
#include "threading/task_impl.h"

namespace cs::communication {
namespace {
constexpr auto toC10dRedOp(const Operation operation) {
  switch (operation) {
    case kSUM:
      return c10d::ReduceOp::SUM;
    case kAVG:
      return c10d::ReduceOp::AVG;
    default:
      CS_ASSERT_TRUE(false, "unsupported operation for NCCL all reduce");
  }
}
}  // namespace

struct AllReduceBucketImpl final : Bucket::Impl {
  AllReduceBucketImpl(const int64_t byteThreshold, const Operation operation)
      : byteThreshold{byteThreshold}, operation{operation} {}

  void apply(const Scheduler &scheduler, const Comm &comm) override;

  const int64_t byteThreshold{};
  const Operation operation{};
  std::vector<Tensor> buffer{};
  int64_t currentByte = 0;
};

void AllReduceBucketImpl::apply(const Scheduler &scheduler, const Comm &comm) {
  AllReduce::run_inplace(scheduler, comm, buffer, operation);
  buffer.clear();
  currentByte = 0;
}

AllReduceBucket::AllReduceBucket(const int64_t byteThreshold,
                                 const Operation operation) {
  impl_ = std::make_shared<AllReduceBucketImpl>(byteThreshold, operation);
}

void AllReduceBucket::push_back(const Scheduler &scheduler, const Comm &comm,
                                Tensor tensor) const {
  const auto impl = std::dynamic_pointer_cast<AllReduceBucketImpl>(impl_);
  impl->currentByte += tensor.impl()->tensor().nbytes();
  impl->buffer.push_back(std::move(tensor));
  if (impl->currentByte >= impl->byteThreshold) {
    AllReduce::run_inplace(scheduler, comm, impl->buffer, impl->operation);
    impl->buffer.clear();
    impl->currentByte = 0;
  }
}

void AllReduceBucket::sync(const Scheduler &scheduler, const Comm &comm) const {
  const auto impl = std::dynamic_pointer_cast<AllReduceBucketImpl>(impl_);
  AllReduce::run_inplace(scheduler, comm, impl->buffer, impl->operation);
  impl->buffer.clear();
  impl->currentByte = 0;
}

void AllReduce::run_inplace(const Scheduler &scheduler, const Comm &comm,
                            const std::vector<Tensor> &tensors,
                            const Operation operation) {
  if (tensors.size() == 1) {
    struct Impl : Task::Impl {
      const Comm comm;
      const Operation operation;

      explicit Impl(std::vector<Tensor> output /* tensors */,
                    std::vector<ReadOnlyTensor> input /* tensors */, Comm comm,
                    const Operation operation)
          : Task::Impl{std::move(output), std::move(input), kComm, kNccl},
            comm{std::move(comm)},
            operation{operation} {}
      void operator()() const override {
        std::vector<at::Tensor> v;
        v.reserve(output().size());
        for (const auto &t : output()) {
          CS_ASSERT_TRUE(t.impl()->tensor().device().type() == at::kCUDA,
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
        return "cs::communication::AllReduce::run_inplace";
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
          : Task::Impl{std::move(output), std::move(input), kComm, kNccl},
            comm{std::move(comm)},
            operation{operation} {}
      void operator()() const override {
        std::vector<at::Tensor> v;
        v.reserve(output().size());
        std::vector<int64_t> sizes;
        sizes.reserve(output().size());
        for (const auto &t : output()) {
          CS_ASSERT_TRUE(t.impl()->tensor().device().type() == at::kCUDA,
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
        return "cs::communication::AllReduce::run_inplace";
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
}  // namespace cs::communication
