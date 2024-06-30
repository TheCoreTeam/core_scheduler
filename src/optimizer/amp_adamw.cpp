/*
 * Copyright (c) 2024 The Core Team
 *
 * Licensed under the Apache License, Version 2.0;
 * You may not use this file except in compliance with the License.
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

#include "optimizer/amp_adamw.h"

#include <c10/cuda/CUDAStream.h>

#include "logger.h"
#include "module/amp_state.h"
#include "tensor_impl.h"
#include "threading/scheduler_impl.h"
#include "threading/task_impl.h"

namespace cs::optimizer {
void ampStepKernel(cudaStream_t stream, const AdamW::State::Options& options,
                   const Tensor& w, const Tensor& wFp32, const Tensor& m,
                   const Tensor& v, const ReadOnlyTensor& dw);

void ampStepKernelAmsgrad(cudaStream_t stream,
                          const AdamW::State::Options& options, const Tensor& w,
                          const Tensor& wFp32, const Tensor& m, const Tensor& v,
                          const Tensor& vMax, const ReadOnlyTensor& dw);

void AmpAdamW::init(const Scheduler& scheduler, const module::Module& module,
                    const Options& options) {
  for (auto& kvState : module.named_states()) {
    for (auto& kvIncrement : kvState.value()->increments()) {
      auto state = init(scheduler, kvIncrement->parameter, options);
      kvIncrement->optimizerState = state;
    }
  }
}

void AmpAdamW::step(const Scheduler& scheduler, const module::Module& module) {
  auto states = module.named_states();
  for (auto& kvState : states) {
    auto increments = kvState.value()->increments();
    auto ampState =
        std::dynamic_pointer_cast<module::AmpState>(kvState.value());
    CS_ASSERT_TRUE(ampState != nullptr, "The module is not an AMP module");
    auto parametersHighPrecision = ampState->parametersHighPrecision();
    for (auto& kvIncrement : increments) {
      CS_ASSERT_TRUE(parametersHighPrecision.contains(kvIncrement.key()),
                     "Internal error, key is not found, mostly because your "
                     "parametersHighPrecision implementation is not corrent");
      step(scheduler,
           std::dynamic_pointer_cast<State>(kvIncrement->optimizerState),
           kvIncrement->parameter, parametersHighPrecision[kvIncrement.key()],
           kvIncrement->gradient);
      kvIncrement->gradient = Tensor{};
    }
  }
}

std::shared_ptr<AmpAdamW::State> AmpAdamW::init(const Scheduler& scheduler,
                                                const ReadOnlyTensor& parameter,
                                                const Options& options) {
  Tensor m;
  Tensor v;
  if (options.amsgrad()) {
    struct Impl : Task::Impl {
      explicit Impl(std::vector<Tensor> output /* m, v, vMax */,
                    std::vector<ReadOnlyTensor> input /* parameter */)
          : Task::Impl{std::move(output), std::move(input), compute} {}
      void operator()() const override {
        output()[0].impl()->tensor() =
            at::zeros(input()[0].impl()->tensor().sizes(),
                      input()[0].impl()->tensor().options().dtype(at::kFloat));
        output()[1].impl()->tensor() =
            at::zeros(input()[0].impl()->tensor().sizes(),
                      input()[0].impl()->tensor().options().dtype(at::kFloat));
        output()[2].impl()->tensor() =
            at::zeros(input()[0].impl()->tensor().sizes(),
                      input()[0].impl()->tensor().options().dtype(at::kFloat));
      }
      [[nodiscard]] const char* name() const override {
        return "cs::optimizer::AmpAdamW::init";
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
          : Task::Impl{std::move(output), std::move(input), compute} {}
      void operator()() const override {
        output()[0].impl()->tensor() =
            at::zeros(input()[0].impl()->tensor().sizes(),
                      input()[0].impl()->tensor().options().dtype(at::kFloat));
        output()[1].impl()->tensor() =
            at::zeros(input()[0].impl()->tensor().sizes(),
                      input()[0].impl()->tensor().options().dtype(at::kFloat));
      }
      [[nodiscard]] const char* name() const override {
        return "cs::optimizer::AmpAdamW::init";
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

void AmpAdamW::step(const Scheduler& scheduler,
                    const std::shared_ptr<State>& state, Tensor& w,
                    Tensor& wFp32, const ReadOnlyTensor& dw) {
  state->options.t++;
  if (state->options.amsgrad) {
    struct Impl : Task::Impl {
      State::Options options;

      explicit Impl(
          std::vector<Tensor> output /* w, wFp32, m, v, vMax */,
          std::vector<ReadOnlyTensor> input /* w, wFp32, m, v, vMax, dw */,
          const State::Options& options)
          : Task::Impl{std::move(output), std::move(input), compute},
            options{options} {}
      void operator()() const override {
        const auto stream = c10::cuda::getCurrentCUDAStream();
        if (input()[4].impl()->tensor().defined()) {
          ampStepKernelAmsgrad(stream.stream(), options, output()[0],
                               output()[1], output()[2], output()[3],
                               output()[4], input()[5]);
        } else {
          CS_WARN_TRUE(false, "got non-defined gradient, skip the update");
        }
      }
      [[nodiscard]] const char* name() const override {
        return "cs::optimizer::AmpAdamW::step";
      }
    };

    const auto& m = state->tensors.m;
    const auto& v = state->tensors.v;
    const auto& vMax = state->tensors.vMax;
    scheduler.impl()->submit(Task{std::make_shared<Impl>(Impl{
        {w, wFp32, m, v, vMax}, {w, wFp32, m, v, vMax, dw}, state->options})});
  } else {
    struct Impl : Task::Impl {
      State::Options options;

      explicit Impl(std::vector<Tensor> output /* w, wFp32, m, v */,
                    std::vector<ReadOnlyTensor> input /* w, wFp32, m, v, dw */,
                    const State::Options& options)
          : Task::Impl{std::move(output), std::move(input), compute},
            options{options} {}
      void operator()() const override {
        const auto stream = c10::cuda::getCurrentCUDAStream();
        if (input()[3].impl()->tensor().defined()) {
          ampStepKernel(stream.stream(), options, output()[0], output()[1],
                        output()[2], output()[3], input()[4]);
        } else {
          CS_WARN_TRUE(false, "got non-defined gradient, skip the update");
        }
      }
      [[nodiscard]] const char* name() const override {
        return "cs::optimizer::AmpAdamW::step";
      }
    };

    const auto& m = state->tensors.m;
    const auto& v = state->tensors.v;
    scheduler.impl()->submit(Task{std::make_shared<Impl>(
        Impl{{w, wFp32, m, v}, {w, wFp32, m, v, dw}, state->options})});
  }
}
}  // namespace cs::optimizer
