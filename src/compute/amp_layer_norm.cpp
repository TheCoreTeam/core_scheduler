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

#include <ATen/ops/ones.h>
#include <ATen/ops/zeros.h>

#include "compute/amp_layer_norm.h"
#include "logger.h"
#include "tensor_impl.h"
#include "threading/scheduler_impl.h"
#include "threading/task_impl.h"

namespace cs::compute {

AmpLayerNorm::State::State(const Forward& forward,
                           const ForwardHighPrecision& forwardHighPrecision,
                           const Backward& backward, const Args& args)
    : LayerNorm::State{forward, backward, args},
      forwardHighPrecision{forwardHighPrecision} {}

OrderedDict<std::string, Tensor> AmpLayerNorm::State::parametersHighPrecision()
    const {
  OrderedDict<std::string, Tensor> dict;
  dict.insert("weight", forwardHighPrecision.weight);
  if (args.bias) {
    dict.insert("bias", forwardHighPrecision.bias);
  }
  return dict;
}

OrderedDict<std::string, Tensor> AmpLayerNorm::State::parameters() const {
  return parametersHighPrecision();
}

std::shared_ptr<AmpLayerNorm::State> AmpLayerNorm::init(
    const Scheduler& scheduler, const Options& options) {
  CS_ASSERT_TRUE(options.elementwise_affine() == true,
                 "elementwise_affine must be enabled now");
  at::TensorOptions tensorOptions{};
  if (options.device().has_value()) {
    tensorOptions = tensorOptions.device(options.device());
  }
  if (options.dtype().has_value()) {
    tensorOptions = tensorOptions.dtype(options.dtype());
  }
  Tensor weight;
  Tensor weightFp32;
  if (options.bias()) {
    struct Impl : Task::Impl {
      const Options options;
      const TensorOptions tensorOptions;

      explicit Impl(std::vector<Tensor> output /* weight, bias */,
                    const Options& options, const TensorOptions tensorOptions)
          : Task::Impl{std::move(output), {}, compute},
            options{options},
            tensorOptions{tensorOptions} {}
      void operator()() const override {
        output()[2].impl()->tensor() = at::ones(
            options.normalized_shape(), tensorOptions.dtype(at::kFloat));
        output()[3].impl()->tensor() = at::zeros(
            options.normalized_shape(), tensorOptions.dtype(at::kFloat));
        output()[0].impl()->tensor() =
            output()[2].impl()->tensor().to(tensorOptions);
        output()[1].impl()->tensor() =
            output()[3].impl()->tensor().to(tensorOptions);
      }
      [[nodiscard]] const char* name() const override {
        return "cs::compute::AmpLayerNorm::init";
      }
    };

    Tensor bias;
    Tensor biasFp32;
    auto task = Task{std::make_shared<Impl>(
        Impl{{weight, bias, weightFp32, biasFp32}, options, tensorOptions})};
    scheduler.impl()->submit(std::move(task));
    return std::make_shared<State>(
        State::Forward{std::move(weight), std::move(bias)},
        State::ForwardHighPrecision{std::move(weightFp32), std::move(biasFp32)},
        State::Backward{},
        State::Args{options.normalized_shape(), options.eps(),
                    options.elementwise_affine(), options.bias()});
  } else {
    struct Impl : Task::Impl {
      const Options options;
      const TensorOptions tensorOptions;

      explicit Impl(std::vector<Tensor> output /* weight */,
                    const Options& options, const TensorOptions tensorOptions)
          : Task::Impl{std::move(output), {}, compute},
            options{options},
            tensorOptions{tensorOptions} {}
      void operator()() const override {
        output()[1].impl()->tensor() = at::ones(
            options.normalized_shape(), tensorOptions.dtype(at::kFloat));
        output()[0].impl()->tensor() =
            output()[1].impl()->tensor().to(tensorOptions);
      }
      [[nodiscard]] const char* name() const override {
        return "cs::compute::AmpLayerNorm::init";
      }
    };

    auto task = Task{std::make_shared<Impl>(
        Impl{{weight, weightFp32}, options, tensorOptions})};
    scheduler.impl()->submit(std::move(task));
    return std::make_shared<State>(
        State::Forward{std::move(weight)},
        State::ForwardHighPrecision{std::move(weightFp32)}, State::Backward{},
        State::Args{options.normalized_shape(), options.eps(),
                    options.elementwise_affine(), options.bias()});
  }
}
}  // namespace cs::compute
