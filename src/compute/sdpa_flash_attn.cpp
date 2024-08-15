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

#include <ATen/ops/_scaled_dot_product_flash_attention.h>
#include <ATen/ops/_scaled_dot_product_flash_attention_backward.h>

#include "compute/scaled_dot_product_attention.h"
#include "logger.h"
#include "tensor_impl.h"
#include "threading/scheduler_impl.h"
#include "threading/task_impl.h"

namespace cs::compute {
std::shared_ptr<ScaledDotProductFlashAttention::State>
ScaledDotProductFlashAttention::init(const Scheduler &scheduler,
                                     const Options &options) {
  return std::make_shared<State>(
      State::Forward{}, State::Backward{},
      State::Args{options.dropout_p(), options.is_causal(),
                  options.return_debug_mask(), options.scale()});
}

Tensor ScaledDotProductFlashAttention::forward(
    const Scheduler &scheduler, const std::shared_ptr<State> &state,
    const ReadOnlyTensor &query, const ReadOnlyTensor &key,
    const ReadOnlyTensor &value) {
  struct Impl : Task::Impl {
    const State::Args args;
    const std::shared_ptr<State::Backward::Max> max;

    explicit Impl(
        std::vector<Tensor> output /* output, logsumexp, cum_seq_q, cum_seq_k,
                                      philox_seed, philox_offset */
        ,
        std::vector<ReadOnlyTensor> input /* query, key, value */,
        const State::Args &args, std::shared_ptr<State::Backward::Max> max)
        : Task::Impl{std::move(output), std::move(input), kMain, kCompute},
          args{args},
          max{std::move(max)} {}
    void operator()() const override {
      auto [output_, logsumexp_, cum_seq_q_, cum_seq_k_, max_q_, max_k_,
            philox_seed_, philox_offset_, debug_attn_mask_] =
          at::_scaled_dot_product_flash_attention(
              input()[0].impl()->tensor().transpose(1, 2),
              input()[1].impl()->tensor().transpose(1, 2),
              input()[2].impl()->tensor().transpose(1, 2), args.dropout_p,
              args.is_causal, args.return_debug_mask, args.scale);
      output()[0].impl()->tensor() = output_.transpose(1, 2);
      output()[1].impl()->tensor() = logsumexp_;
      output()[2].impl()->tensor() = cum_seq_q_;
      output()[3].impl()->tensor() = cum_seq_k_;
      max->max_q = max_q_.as_int_unchecked();
      max->max_k = max_k_.as_int_unchecked();
      output()[4].impl()->tensor() = philox_seed_;
      output()[5].impl()->tensor() = philox_offset_;
    }
    [[nodiscard]] const char *name() const override {
      return "cs::compute::ScaledDotProductAttention::forward";
    }
  };

  Tensor output;
  const Tensor logsumexp;
  const Tensor cum_seq_q;
  const Tensor cum_seq_k;
  const Tensor philox_seed;
  const Tensor philox_offset;
  state->backward.query = query;
  state->backward.key = key;
  state->backward.value = value;
  state->backward.out = output;
  state->backward.logsumexp = logsumexp;
  state->backward.cum_seq_q = cum_seq_q;
  state->backward.cum_seq_k = cum_seq_k;
  state->backward.philox_seed = philox_seed;
  state->backward.philox_offset = philox_offset;
  scheduler.impl()->submit(Task{std::make_shared<Impl>(Impl{
      {output, logsumexp, cum_seq_q, cum_seq_k, philox_seed, philox_offset},
      {query, key, value},
      state->args,
      state->backward.max})});
  return output;
}

std::array<Tensor, 3> ScaledDotProductFlashAttention::backward(
    const Scheduler &scheduler, const std::shared_ptr<State> &state,
    const ReadOnlyTensor &grad_out) {
  struct Impl : Task::Impl {
    const State::Args args;
    const State::Backward::Max max;

    explicit Impl(
        std::vector<Tensor> output /* grad_query, grad_key, grad_value */,
        std::vector<ReadOnlyTensor>
            input /* grad_out[0], query[1], key[2], value[3], out[4],
                     logsumexp[5], cum_seq_q[6], cum_seq_k[7],
                     philox_seed[8], philox_offset[9] */
        ,
        const State::Args &args, const State::Backward::Max &max)
        : Task::Impl{std::move(output), std::move(input), kMain, kCompute},
          args{args},
          max{max} {}
    void operator()() const override {
      const auto &grad_out = input()[0];
      const auto &query = input()[1];
      const auto &key = input()[2];
      const auto &value = input()[3];
      const auto &out = input()[4];
      const auto &logsumexp = input()[5];
      const auto &cum_seq_q = input()[6];
      const auto &cum_seq_k = input()[7];
      const auto &philox_seed = input()[8];
      const auto &philox_offset = input()[9];
      auto [grad_query_, grad_key_, grad_value_] =
          at::_scaled_dot_product_flash_attention_backward(
              grad_out.impl()->tensor().transpose(1, 2),
              query.impl()->tensor().transpose(1, 2),
              key.impl()->tensor().transpose(1, 2),
              value.impl()->tensor().transpose(1, 2),
              out.impl()->tensor().transpose(1, 2), logsumexp.impl()->tensor(),
              cum_seq_q.impl()->tensor(), cum_seq_k.impl()->tensor(), max.max_q,
              max.max_k, args.dropout_p, args.is_causal,
              philox_seed.impl()->tensor(), philox_offset.impl()->tensor(),
              args.scale);
      output()[0].impl()->tensor() = grad_query_.transpose(1, 2);
      output()[1].impl()->tensor() = grad_key_.transpose(1, 2);
      output()[2].impl()->tensor() = grad_value_.transpose(1, 2);
    }
    [[nodiscard]] const char *name() const override {
      return "cs::compute::ScaledDotProductAttention::backward";
    }
  };

  Tensor grad_query;
  Tensor grad_key;
  Tensor grad_value;

  scheduler.impl()->submit(Task{std::make_shared<Impl>(Impl{
      {grad_query, grad_key, grad_value},
      {grad_out, state->backward.query, state->backward.key,
       state->backward.value, state->backward.out, state->backward.logsumexp,
       state->backward.cum_seq_q, state->backward.cum_seq_k,
       state->backward.philox_seed, state->backward.philox_offset},
      state->args,
      *state->backward.max})});

  // decrease counter
  state->backward.query.reset();
  state->backward.key.reset();
  state->backward.value.reset();
  state->backward.out.reset();
  state->backward.logsumexp.reset();
  state->backward.cum_seq_q.reset();
  state->backward.cum_seq_k.reset();
  state->backward.philox_seed.reset();
  state->backward.philox_offset.reset();
  return {grad_query, grad_key, grad_value};
}
}  // namespace cs::compute
