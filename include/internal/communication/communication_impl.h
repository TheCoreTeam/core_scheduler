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

#pragma once
#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>

#include "communication/communication.h"

namespace dllm::communication {
struct Bucket::Impl {
  virtual ~Impl() = default;

  virtual void apply(const Scheduler &scheduler, const Comm &comm) = 0;
};

struct Comm::Impl {
  Impl(MPI_Comm group, c10::intrusive_ptr<c10d::Store> store,
       c10::intrusive_ptr<c10d::Backend> backend);

  const c10::intrusive_ptr<c10d::Store> &store() const;

  const c10::intrusive_ptr<c10d::Backend> &backend() const;

 private:
  MPI_Comm group_{};
  const c10::intrusive_ptr<c10d::Store> store_{};
  const c10::intrusive_ptr<c10d::Backend> backend_{};
};
}  // namespace dllm::communication