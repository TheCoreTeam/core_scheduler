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

#include <torch/nn/functional/embedding.h>

#include "compute/embedding.h"
#include "logger.h"
#include "tensor_impl.h"
#include "threading/scheduler_impl.h"
#include "threading/task_impl.h"

namespace cs::compute {
Embedding::State::State(const Forward &forward, const Backward &backward,
                        const Args &args)
    : forward{forward}, backward{backward}, args{args} {}

OrderedDict<std::string, Tensor> Embedding::State::parameters() const {
  OrderedDict<std::string, Tensor> dict;
  dict.insert("weight", forward.weight);
  return dict;
}

OrderedDict<std::string, Tensor> Embedding::State::gradients() const {
  OrderedDict<std::string, Tensor> dict;
  dict.insert("weight", forward.grad_weight);
  return dict;
}

OrderedDict<std::string, module::State::Increment>
Embedding::State::increments() const {
  OrderedDict<std::string, Increment> dict;
  dict.insert("weight", {forward.weight, forward.grad_weight});
  return dict;
}

void Embedding::State::zero_grad() { forward.grad_weight = {}; }

std::shared_ptr<Embedding::State> Embedding::init(const Scheduler &scheduler,
                                                  const Options &options) {
  struct Impl : Task::Impl {
    const Options options;

    explicit Impl(std::vector<Tensor> output /* weight */,
                  const Options &options)
        : Task::Impl{std::move(output), {}, kMain, kCompute},
          options{options} {}
    void operator()() const override {
      const auto weight_ =
          at::normal(0, 1, {options.num_embeddings(), options.embedding_dim()},
                     {}, options.dtype(), {}, options.device(), {});
      // if (max_norm != c10::nullopt) {
      //   input_ = input_.contiguous();
      //   _no_grad_embedding_renorm_(weight, input_, *max_norm, norm_type);
      // }
      output()[0].impl()->tensor() = weight_;
    }
    [[nodiscard]] const char *name() const override {
      return "cs::compute::Embedding::init";
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
  scheduler.impl()->submit(
      Task{std::make_shared<Impl>(Impl{{weight}, options})});

  return std::make_shared<State>(
      State::Forward{std::move(weight)}, State::Backward{},
      State::Args{options.num_embeddings(), padding_idx, options.max_norm(),
                  options.norm_type(), options.scale_grad_by_freq(),
                  options.sparse()});
}

Tensor Embedding::forward(const Scheduler &scheduler,
                          const std::shared_ptr<State> &state,
                          const ReadOnlyTensor &indices) {
  struct Impl : Task::Impl {
    State::Args args;

    explicit Impl(std::vector<Tensor> output /* output */,
                  std::vector<ReadOnlyTensor> input /* weight, indices */,
                  const State::Args &args)
        : Task::Impl{std::move(output), std::move(input), kMain, kCompute},
          args{args} {}
    void operator()() const override {
      output()[0].impl()->tensor() = torch::embedding(
          input()[0].impl()->tensor(),
          input()[1].impl()->tensor().view(
              {-1, input()[1].impl()->tensor().size(-1)}),
          args.padding_idx, args.scale_grad_by_freq, args.sparse);
    }
    [[nodiscard]] const char *name() const override {
      return "cs::compute::Embedding::forward";
    }
  };

  Tensor output{};
  state->backward.indices = indices;
  scheduler.impl()->submit(Task{std::make_shared<Impl>(
      Impl{{output}, {state->forward.weight, indices}, state->args})});
  return output;
}

void Embedding::backward(const Scheduler &scheduler,
                         const std::shared_ptr<State> &state,
                         const ReadOnlyTensor &grad_output) {
  struct Impl : Task::Impl {
    State::Args args;

    explicit Impl(std::vector<Tensor> output /* grad_weight */,
                  std::vector<ReadOnlyTensor> input /* grad_output, indices */,
                  const State::Args &args)
        : Task::Impl{std::move(output), std::move(input), kAssist, kCompute},
          args{args} {}
    void operator()() const override {
      if (output()[0].impl()->tensor().defined()) {
        output()[0].impl()->tensor() += at::embedding_backward(
            input()[0].impl()->tensor(), input()[1].impl()->tensor(),
            args.num_weights, args.padding_idx, args.scale_grad_by_freq,
            args.sparse);
      } else /* accumulate grad */ {
        output()[0].impl()->tensor() = at::embedding_backward(
            input()[0].impl()->tensor(), input()[1].impl()->tensor(),
            args.num_weights, args.padding_idx, args.scale_grad_by_freq,
            args.sparse);
      }
    }
    [[nodiscard]] const char *name() const override {
      return "cs::compute::Embedding::backward";
    }
  };

  scheduler.impl()->submit(
      Task{std::make_shared<Impl>(Impl{{state->forward.grad_weight},
                                       {grad_output, state->backward.indices},
                                       state->args})});
  // decrease counter
  state->backward.indices.reset();
}
}  // namespace cs::compute
