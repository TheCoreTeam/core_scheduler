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

#include "compute/init.h"

#include <torch/nn/init.h>

#include "tensor_impl.h"
#include "threading/scheduler_impl.h"
#include "threading/task_impl.h"

namespace cs::compute::Init {
void kaiming_normal_(const Scheduler &scheduler, const Tensor &tensor,
                     const double a, const torch::nn::init::FanModeType &m,
                     const torch::nn::init::NonlinearityType &nonlineari) {
  struct Impl : Task::Impl {
    const double a;
    const torch::nn::init::FanModeType m;
    const torch::nn::init::NonlinearityType nonlineari;

    explicit Impl(const Tensor &tensor /* tensor */, const double a,
                  const torch::nn::init::FanModeType &m,
                  const torch::nn::init::NonlinearityType &nonlineari)
        : Task::Impl{{tensor}, {tensor}, kMain, kCompute},
          a{a},
          m{m},
          nonlineari{nonlineari} {}
    void operator()() const override {
      torch::nn::init::kaiming_normal_(output()[0].impl()->tensor(), a, m,
                                       nonlineari);
    }
    [[nodiscard]] const char *name() const override {
      return "cs::compute::Init::kaiming_normal_";
    }
  };

  scheduler.impl()->submit(
      Task{std::make_shared<Impl>(Impl{tensor, a, m, nonlineari})});
}

void kaiming_uniform_(const Scheduler &scheduler, const Tensor &tensor,
                      const double a, const torch::nn::init::FanModeType &m,
                      const torch::nn::init::NonlinearityType &nonlineari) {
  struct Impl : Task::Impl {
    const double a;
    const torch::nn::init::FanModeType m;
    const torch::nn::init::NonlinearityType nonlineari;

    explicit Impl(const Tensor &tensor /* tensor */, const double a,
                  const torch::nn::init::FanModeType &m,
                  const torch::nn::init::NonlinearityType &nonlineari)
        : Task::Impl{{tensor}, {tensor}, kMain, kCompute},
          a{a},
          m{m},
          nonlineari{nonlineari} {}
    void operator()() const override {
      torch::nn::init::kaiming_uniform_(output()[0].impl()->tensor(), a, m,
                                        nonlineari);
    }
    [[nodiscard]] const char *name() const override {
      return "cs::compute::Init::kaiming_uniform_";
    }
  };

  scheduler.impl()->submit(
      Task{std::make_shared<Impl>(Impl{tensor, a, m, nonlineari})});
}
}  // namespace cs::compute::Init
