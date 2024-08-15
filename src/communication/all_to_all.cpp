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

#include "communication/all_to_all.h"

#include <torch/csrc/autograd/generated/variable_factories.h>

#include <limits>

#include "communication/communication_impl.h"
#include "logger.h"
#include "tensor_impl.h"
#include "threading/scheduler_impl.h"
#include "threading/task_impl.h"

namespace cs::communication {
void AllToAll::run(const Scheduler &scheduler, const Comm &comm,
                   const std::vector<Tensor> &tensorReceive,
                   const std::vector<ReadOnlyTensor> &tensorSend) {
  struct Impl : Task::Impl {
    const Comm comm;

    explicit Impl(std::vector<Tensor> output /* tensors */,
                  std::vector<ReadOnlyTensor> input /* tensors */, Comm comm)
        : Task::Impl{std::move(output), std::move(input), kComm, kNccl},
          comm{std::move(comm)} {}
    void operator()() const override {
      std::vector<at::Tensor> vReceive;
      vReceive.reserve(output().size());
      for (const auto &t : output()) {
        CS_ASSERT_TRUE(t.impl()->tensor().device().type() == at::kCUDA,
                       "NCCL backend only support CUDA GPUs");
        vReceive.push_back(t.impl()->tensor());
      }

      std::vector<at::Tensor> vSend;
      vSend.reserve(input().size());
      for (const auto &t : input()) {
        CS_ASSERT_TRUE(t.impl()->tensor().device().type() == at::kCUDA,
                       "NCCL backend only support CUDA GPUs");
        vSend.push_back(t.impl()->tensor());
      }

      comm.impl()->backend()->alltoall(vReceive, vSend)->wait();
    }
    [[nodiscard]] const char *name() const override {
      return "cs::communication::AllToAll::run";
    }
  };

  scheduler.impl()->submit(
      Task{std::make_shared<Impl>(Impl{tensorReceive, tensorSend, comm})});
}
}  // namespace cs::communication
