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

#include "optimizer/adamw.h"

#include <c10/cuda/CUDAStream.h>
#include <torch/csrc/autograd/generated/variable_factories.h>

#include "logger.h"
#include "module/module.h"
#include "tensor_impl.h"
#include "threading/scheduler.h"
#include "threading/scheduler_impl.h"
#include "threading/task_impl.h"

namespace cs::optimizer {
void stepKernel(cudaStream_t stream, const AdamW::State::Options &options,
                const Tensor &w, const Tensor &m, const Tensor &v,
                const ReadOnlyTensor &dw);

void stepKernelAmsgrad(cudaStream_t stream,
                       const AdamW::State::Options &options, const Tensor &w,
                       const Tensor &m, const Tensor &v, const Tensor &vMax,
                       const ReadOnlyTensor &dw);

AdamW::State::State(const Tensors &tensors, const Options &options)
    : tensors{tensors}, options{options} {}

void AdamW::State::set_lr(const double lr) const { options.lr = lr; }

double AdamW::State::get_lr() const { return options.lr; }

std::shared_ptr<AdamW::State> AdamW::init(const Scheduler &scheduler,
                                          const ReadOnlyTensor &parameter,
                                          const Options &options) {
  Tensor m;
  Tensor v;
  if (options.amsgrad()) {
    struct Impl : Task::Impl {
      explicit Impl(std::vector<Tensor> output /* m, v, vMax */,
                    std::vector<ReadOnlyTensor> input /* parameter */)
          : Task::Impl{std::move(output), std::move(input), kAssist, kCompute} {
      }
      void operator()() const override {
        output()[0].impl()->tensor() =
            at::zeros_like(input()[0].impl()->tensor());
        output()[1].impl()->tensor() =
            at::zeros_like(input()[0].impl()->tensor());
        output()[2].impl()->tensor() =
            at::zeros_like(input()[0].impl()->tensor());
      }
      [[nodiscard]] const char *name() const override {
        return "cs::optimizer::AdamW::init";
      }
    };

    Tensor vMax;

    scheduler.impl()->submit(
        Task{std::make_shared<Impl>(Impl{{m, v, vMax}, {parameter}})});

    return std::make_shared<State>(
        State::Tensors{m, v, vMax},
        State::Options{options.lr(), options.beta1(), options.beta2(),
                       options.eps(), options.weight_decay(), options.amsgrad(),
                       options.t()});
  } else {
    struct Impl : Task::Impl {
      explicit Impl(std::vector<Tensor> output /* m, v */,
                    std::vector<ReadOnlyTensor> input /* parameter */)
          : Task::Impl{std::move(output), std::move(input), kAssist, kCompute} {
      }
      void operator()() const override {
        output()[0].impl()->tensor() =
            at::zeros_like(input()[0].impl()->tensor());
        output()[1].impl()->tensor() =
            at::zeros_like(input()[0].impl()->tensor());
      }
      [[nodiscard]] const char *name() const override {
        return "cs::optimizer::AdamW::init";
      }
    };

    scheduler.impl()->submit(
        Task{std::make_shared<Impl>(Impl{{m, v}, {parameter}})});

    return std::make_shared<State>(
        State::Tensors{m, v},
        State::Options{options.lr(), options.beta1(), options.beta2(),
                       options.eps(), options.weight_decay(), options.amsgrad(),
                       options.t()});
  }
}

void AdamW::step(const Scheduler &scheduler,
                 const std::shared_ptr<module::OptimizerState> &state_,
                 const Tensor &w, const ReadOnlyTensor &dw) {
  const auto state = std::dynamic_pointer_cast<State>(state_);
  state->options.t++;
  if (state->options.amsgrad) {
    struct Impl : Task::Impl {
      State::Options options;

      explicit Impl(std::vector<Tensor> output /* w, m, v, vMax */,
                    std::vector<ReadOnlyTensor> input /* w, m, v, vMax, dw */,
                    const State::Options &options)
          : Task::Impl{std::move(output), std::move(input), kAssist, kCompute},
            options{options} {}
      void operator()() const override {
        const auto stream = c10::cuda::getCurrentCUDAStream();
        if (input()[4].impl()->tensor().defined()) {
          stepKernelAmsgrad(stream.stream(), options, output()[0], output()[1],
                            output()[2], output()[3], input()[4]);
        } else {
          CS_WARN_TRUE(false, "got non-defined gradient, skip the update");
        }
      }
      [[nodiscard]] const char *name() const override {
        return "cs::optimizer::AdamW::step";
      }
    };

    const auto &m = state->tensors.m;
    const auto &v = state->tensors.v;
    const auto &vMax = state->tensors.v_max;
    scheduler.impl()->submit(Task{std::make_shared<Impl>(
        Impl{{w, m, v, vMax}, {w, m, v, vMax, dw}, state->options})});
  } else {
    struct Impl : Task::Impl {
      State::Options options;

      explicit Impl(std::vector<Tensor> output /* w, m, v */,
                    std::vector<ReadOnlyTensor> input /* w, m, v, dw */,
                    const State::Options &options)
          : Task::Impl{std::move(output), std::move(input), kAssist, kCompute},
            options{options} {}
      void operator()() const override {
        const auto stream = c10::cuda::getCurrentCUDAStream();
        if (input()[3].impl()->tensor().defined()) {
          stepKernel(stream.stream(), options, output()[0], output()[1],
                     output()[2], input()[3]);
        } else {
          CS_WARN_TRUE(false, "got non-defined gradient, skip the update");
        }
      }
      [[nodiscard]] const char *name() const override {
        return "cs::optimizer::AdamW::step";
      }
    };

    const auto &m = state->tensors.m;
    const auto &v = state->tensors.v;
    scheduler.impl()->submit(Task{std::make_shared<Impl>(
        Impl{{w, m, v}, {w, m, v, dw}, state->options})});
  }
}
}  // namespace cs::optimizer
