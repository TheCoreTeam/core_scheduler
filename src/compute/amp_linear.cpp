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

// Header Order Protection
// ReSharper disable once CppUnusedIncludeDirective
#include <c10/util/Exception.h>
// Header Order Protection

#include <torch/nn/functional/linear.h>
#include <torch/nn/init.h>

#include "compute/amp_linear.h"
#include "logger.h"
#include "tensor_impl.h"
#include "threading/scheduler_impl.h"
#include "threading/task_impl.h"

namespace cs::compute {
AmpLinear::State::State(const Forward& forward,
                        const ForwardHighPrecision& forwardHighPrecision,
                        const Backward& backward, const Args& args)
    : Linear::State{forward, backward, args},
      forward_high_precision{forwardHighPrecision} {}

OrderedDict<std::string, Tensor> AmpLinear::State::parameters_high_precision()
    const {
  OrderedDict<std::string, Tensor> dict;
  dict.insert("weight", forward_high_precision.weight);
  if (args.bias) {
    dict.insert("bias", forward_high_precision.bias);
  }
  return dict;
}

OrderedDict<std::string, Tensor> AmpLinear::State::parameters() const {
  return parameters_high_precision();
}

std::shared_ptr<AmpLinear::State> AmpLinear::init(const Scheduler& scheduler,
                                                  const Options& options) {
  TensorOptions tensorOptions{};
  if (options.device().has_value()) {
    tensorOptions = tensorOptions.device(options.device().value());
  }
  if (options.dtype().has_value()) {
    tensorOptions = tensorOptions.dtype(options.dtype().value());
  }

  if (options.bias()) {
    struct Impl : Task::Impl {
      const TensorOptions options;
      const int64_t in_futures;
      const int64_t out_futures;

      explicit Impl(
          std::vector<Tensor> output /* weight, bias, weightFp32, biasFp32 */,
          const TensorOptions options, const int64_t in_futures,
          const int64_t out_futures)
          : Task::Impl{std::move(output), {}, kCompute},
            options{options},
            in_futures{in_futures},
            out_futures{out_futures} {}
      void operator()() const override {
        const auto weight =
            torch::empty({out_futures, in_futures}, options.dtype(at::kFloat));
        output()[2].impl()->tensor() = weight;
        const auto bias =
            torch::empty({out_futures}, options.dtype(at::kFloat));
        output()[3].impl()->tensor() = bias;
        torch::nn::init::kaiming_uniform_(output()[2].impl()->tensor(),
                                          std::sqrt(5));
        auto [fan_in, fan_out] = torch::nn::init::_calculate_fan_in_and_fan_out(
            output()[2].impl()->tensor());
        const auto bound = 1 / std::sqrt(fan_in);
        torch::nn::init::uniform_(output()[3].impl()->tensor(), -bound, bound);

        output()[0].impl()->tensor() = output()[2].impl()->tensor().to(options);
        output()[1].impl()->tensor() = output()[3].impl()->tensor().to(options);
      }
      [[nodiscard]] const char* name() const override {
        return "cs::compute::AmpLinear::init";
      }
    };

    Tensor weight;
    Tensor bias;
    Tensor weightFp32;
    Tensor biasFp32;
    scheduler.impl()->submit(
        Task{std::make_shared<Impl>(Impl{{weight, bias, weightFp32, biasFp32},
                                         tensorOptions,
                                         options.in_futures(),
                                         options.out_futures()})});
    return std::make_shared<State>(
        State::Forward{std::move(weight), std::move(bias)},
        State::ForwardHighPrecision{std::move(weightFp32), std::move(biasFp32)},
        State::Backward{}, State::Args{options.bias()});
  } else {
    struct Impl : Task::Impl {
      const TensorOptions options;
      const int64_t in_futures;
      const int64_t out_futures;

      explicit Impl(std::vector<Tensor> output /* weight, weightFp32 */,
                    const TensorOptions options, const int64_t in_futures,
                    const int64_t out_futures)
          : Task::Impl{std::move(output), {}, kCompute},
            options{options},
            in_futures{in_futures},
            out_futures{out_futures} {}
      void operator()() const override {
        const auto weight =
            torch::empty({out_futures, in_futures}, options.dtype(at::kFloat));
        output()[1].impl()->tensor() = weight;
        torch::nn::init::kaiming_uniform_(output()[1].impl()->tensor(),
                                          std::sqrt(5));
        output()[0].impl()->tensor() = output()[1].impl()->tensor().to(options);
      }
      [[nodiscard]] const char* name() const override {
        return "cs::compute::AmpLinear::init";
      }
    };

    Tensor weight;
    Tensor weightFp32;
    scheduler.impl()->submit(
        Task{std::make_shared<Impl>(Impl{{weight, weightFp32},
                                         tensorOptions,
                                         options.in_futures(),
                                         options.out_futures()})});
    return std::make_shared<State>(
        State::Forward{std::move(weight)},
        State::ForwardHighPrecision{std::move(weightFp32)}, State::Backward{},
        State::Args{options.bias()});
  }
}
}  // namespace cs::compute
