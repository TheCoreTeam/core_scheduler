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

#include "memory/to_torch.h"

#include <c10/cuda/CUDAStream.h>

#include "tensor_impl.h"
#include "threading/scheduler_impl.h"
#include "threading/task_impl.h"

namespace cs::memory {
at::Tensor to_torch(const Scheduler &scheduler, const ReadOnlyTensor &src) {
  struct Impl : Task::Impl {
    explicit Impl(std::vector<Tensor> output /* tensor */,
                  std::vector<ReadOnlyTensor> input /* input */)
        : Task::Impl{std::move(output), std::move(input), kMain, kCompute} {}
    void operator()() const override {
      c10::cuda::getCurrentCUDAStream().synchronize();
      auto &dst = output()[0].impl()->tensor();
      auto &src = input()[0].impl()->tensor();
      if (dst.defined()) {
        dst.copy_(src.clone().detach_());
      } else {
        dst = src.clone().detach_();
      }
      c10::cuda::getCurrentCUDAStream().synchronize();
    }
    [[nodiscard]] const char *name() const override {
      return "cs::memory::to_torch";
    }
  };
  Tensor dst_;
  scheduler.impl()->submit(Task{std::make_shared<Impl>(Impl{{dst_}, {src}})});
  auto dst = dst_.impl()->tensor();
  dst_.wait();
  return dst;
}
}  // namespace cs::memory
