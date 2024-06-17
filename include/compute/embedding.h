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
#include "arg.h"
#include "module/state.h"
#include "tensor.h"
#include "threading/scheduler.h"

namespace dllm::compute {
struct Embedding {
  struct State final : module::State {
    struct Forward {
      Tensor weight;
      Tensor grad_weight{};
      std::shared_ptr<module::OptimizerState> optimizer_weight = nullptr;
    } forward;
    struct Backward {
      ReadOnlyTensor indices;
    } backward;
    struct Args {
      int64_t num_weights;
      int64_t padding_idx;
      c10::optional<double> max_norm;
      double norm_type;
      bool scale_grad_by_freq;
      bool sparse;
    } args;

    State(const Forward &forward, const Backward &backward, const Args &args)
        : forward{forward}, backward{backward}, args{args} {}

    [[nodiscard]] OrderedDict<std::string, Tensor> parameters() const override;

    [[nodiscard]] OrderedDict<std::string, Increment> increments() override;
  };

  struct Options {
    Options(const int64_t num_embeddings, const int64_t embedding_dim)
        : num_embeddings_{num_embeddings}, embedding_dim_{embedding_dim} {}
    DLLM_ARG(int64_t, num_embeddings);
    DLLM_ARG(int64_t, embedding_dim);
    DLLM_ARG(c10::optional<int64_t>, padding_idx) = {};
    DLLM_ARG(c10::optional<double>, max_norm) = {};
    DLLM_ARG(double, norm_type) = 2.;
    DLLM_ARG(bool, scale_grad_by_freq) = false;
    DLLM_ARG(bool, sparse) = false;
    DLLM_ARG(c10::optional<at::Device>, device) = {};
    DLLM_ARG(c10::optional<at::ScalarType>, dtype) = {};
  };

  static std::shared_ptr<State> init(const Scheduler &scheduler,
                                     const Options &options);

  static Tensor forward(const Scheduler &scheduler,
                        const std::shared_ptr<State> &state,
                        const ReadOnlyTensor &indices);

  static void backward(const Scheduler &scheduler,
                       const std::shared_ptr<State> &state,
                       const ReadOnlyTensor &grad_output);
};

}  // namespace dllm::compute
