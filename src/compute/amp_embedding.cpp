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

#include <ATen/ops/normal.h>

#include "compute/amp_embedding.h"
#include "tensor_impl.h"
#include "threading/scheduler_impl.h"
#include "threading/task_impl.h"

namespace cs::compute {
AmpEmbedding::State::State(const Forward& forward,
                           const ForwardHighPrecision& forwardHighPrecision,
                           const Backward& backward, const Args& args)
    : Embedding::State{forward, backward, args},
      forwardHighPrecision{forwardHighPrecision} {}

OrderedDict<std::string, Tensor> AmpEmbedding::State::parametersHighPrecision()
    const {
  OrderedDict<std::string, Tensor> dict;
  dict.insert("weight", forwardHighPrecision.weight);
  return dict;
}

OrderedDict<std::string, Tensor> AmpEmbedding::State::parameters() const {
  return parametersHighPrecision();
}

std::shared_ptr<AmpEmbedding::State> AmpEmbedding::init(
    const Scheduler& scheduler, const Options& options) {
  struct Impl : Task::Impl {
    const Options options;

    explicit Impl(std::vector<Tensor> output /* weight */,
                  const Options& options)
        : Task::Impl{std::move(output), {}, compute}, options{options} {}
    void operator()() const override {
      const auto weight =
          at::normal(0, 1, {options.num_embeddings(), options.embedding_dim()},
                     {}, at::kFloat, {}, options.device(), {});
      // if (max_norm != c10::nullopt) {
      //   input_ = input_.contiguous();
      //   _no_grad_embedding_renorm_(weight, input_, *max_norm, norm_type);
      // }
      output()[0].impl()->tensor() = options.dtype().has_value()
                                         ? weight.to(options.dtype().value())
                                         : weight.to(at::get_default_dtype());
      output()[1].impl()->tensor() = weight;
    }
    [[nodiscard]] const char* name() const override {
      return "cs::compute::AmpEmbedding::init";
    }
  };

  int64_t padding_idx = -1;
  if (options.padding_idx() != c10::nullopt) {
    if (*options.padding_idx() > 0) {
      TORCH_CHECK(*options.padding_idx() < options.num_embeddings(),
                  "Padding_idx must be within num_embeddings");
    } else if (*options.padding_idx() < 0) {
      TORCH_CHECK(*options.padding_idx() >= -options.num_embeddings(),
                  "Padding_idx must be within num_embedding");
      padding_idx = options.num_embeddings() + *options.padding_idx();
    }
  }

  TORCH_CHECK(options.max_norm() == c10::nullopt)

  Tensor weight;
  Tensor weightHighPrecision;
  scheduler.impl()->submit(Task{
      std::make_shared<Impl>(Impl{{weight, weightHighPrecision}, options})});

  return std::make_shared<State>(
      State::Forward{std::move(weight)},
      State::ForwardHighPrecision{std::move(weightHighPrecision)},
      State::Backward{},
      State::Args{options.num_embeddings(), padding_idx, options.max_norm(),
                  options.norm_type(), options.scale_grad_by_freq(),
                  options.sparse()});
}
}  // namespace cs::compute
