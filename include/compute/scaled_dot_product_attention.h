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
#include "tensor.h"
#include "threading/scheduler.h"

namespace cs::compute {
struct CS_API ScaledDotProductFlashAttention {
  struct State {
    struct Forward {
    } forward;
    struct Backward {
      ReadOnlyTensor query{};
      ReadOnlyTensor key{};
      ReadOnlyTensor value{};
      ReadOnlyTensor out{};
      ReadOnlyTensor logsumexp{};
      ReadOnlyTensor cum_seq_q{};
      ReadOnlyTensor cum_seq_k{};
      struct Max {
        int64_t max_q;
        int64_t max_k;
      };
      std::shared_ptr<Max> max = std::make_shared<Max>();
      ReadOnlyTensor philox_seed{};
      ReadOnlyTensor philox_offset{};
    } backward;
    struct Args {
      const double dropout_p = 0;
      const bool is_causal = false;
      const bool return_debug_mask = false /* This must be false! */;
      const c10::optional<double> scale = c10::nullopt;
    } args;
  };

  struct Options {
    Options() {}
    CS_ARG(double, dropout_p) = 0;
    CS_ARG(bool, is_causal) = false;
    CS_ARG(bool, return_debug_mask) = false;
    CS_ARG(c10::optional<double>, scale) = {};
  };

  static std::shared_ptr<State> init(const Scheduler &scheduler,
                                     const Options &options = {});

  static Tensor forward(const Scheduler &scheduler,
                        const std::shared_ptr<State> &state,
                        const ReadOnlyTensor &query, const ReadOnlyTensor &key,
                        const ReadOnlyTensor &value);

  // grad_query, grad_key, grad_value
  static std::array<Tensor, 3> backward(const Scheduler &scheduler,
                                        const std::shared_ptr<State> &state,
                                        const ReadOnlyTensor &grad_out);
};

struct CS_API ScaledDotProductCuDnn {
  struct State {
    struct Forward {
    } forward;
    struct Backward {
      ReadOnlyTensor query{};
      ReadOnlyTensor key{};
      ReadOnlyTensor value{};
      ReadOnlyTensor out{};
      ReadOnlyTensor stats{};
    } backward;
    struct Args {
      const double dropout_p = 0;
      const bool is_causal = false;
      const bool return_debug_mask = false /* This must be false! */;
      const c10::optional<double> scale = c10::nullopt;
      struct RNG {
        int64_t seed;
        int64_t offset;
      };
      std::shared_ptr<RNG> rng;
    } args;
  };

  struct Options {
    Options() {}
    CS_ARG(double, dropout_p) = 0;
    CS_ARG(bool, is_causal) = false;
    CS_ARG(bool, return_debug_mask) = false;
    CS_ARG(c10::optional<double>, scale) = {};
  };

  static std::shared_ptr<State> init(const Scheduler &scheduler,
                                     const Options &options = {});

  static Tensor forward(const Scheduler &scheduler,
                        const std::shared_ptr<State> &state,
                        const ReadOnlyTensor &query, const ReadOnlyTensor &key,
                        const ReadOnlyTensor &value);

  // grad_query, grad_key, grad_value
  static std::array<Tensor, 3> backward(const Scheduler &scheduler,
                                        const std::shared_ptr<State> &state,
                                        const ReadOnlyTensor &grad_out);
};
}  // namespace cs::compute
