#include "compute/scaled_dot_product_attention.h"

#include <ATen/ops/_scaled_dot_product_flash_attention.h>
#include <ATen/ops/_scaled_dot_product_flash_attention_backward.h>

#include "internal_utils.h"
#include "logger.h"
#include "nvtx_helper.h"
#include "tensor_impl.h"
#include "threading/scheduler_impl.h"
#include "threading/task_compute.h"

namespace dllm::compute {
void ScaledDotProductFlashAttention::init(const Scheduler &scheduler,
                                          std::shared_ptr<State> &state,
                                          const Options &options) {
  state = std::make_shared<State>(
      State::Forward{}, State::Backward{},
      State::Args{options.dropout_p(), options.is_causal(),
                  options.return_debug_mask(), options.scale()});
}

void ScaledDotProductFlashAttention::forward(
    const Scheduler &scheduler, const std::shared_ptr<State> &state,
    Tensor &output, const ReadOnlyTensor &query, const ReadOnlyTensor &key,
    const ReadOnlyTensor &value) {
  Tensor output_;
  const Tensor logsumexp;
  const Tensor cum_seq_q;
  const Tensor cum_seq_k;
  const Tensor philox_seed;
  const Tensor philox_offset;
  auto task = TaskCompute{
      [args = state->args, query = query, key = key, value = value,
       output = output_, logsumexp = logsumexp, cum_seq_q = cum_seq_q,
       cum_seq_k = cum_seq_k, max = state->backward.max,
       philox_seed = philox_seed, philox_offset = philox_offset,
       queryFuture = utils::future(query), keyFuture = utils::future(key),
       valueFuture =
           utils::future(value)](const ContextCompute *context) mutable {
        DLLM_NVTX_RANGE_FN("dllm::compute::ScaledDotProductAttention::forward");
        {
          utils::FutureGuard queryGuard{queryFuture};
          utils::FutureGuard keyGuard{keyFuture};
          utils::FutureGuard valueGuard{valueFuture};
          auto [output_, logsumexp_, cum_seq_q_, cum_seq_k_, max_q_, max_k_,
                philox_seed_, philox_offset_, debug_attn_mask_] =
              at::_scaled_dot_product_flash_attention(
                  query.impl()->tensor().transpose(1, 2),
                  key.impl()->tensor().transpose(1, 2),
                  value.impl()->tensor().transpose(1, 2), args.dropout_p,
                  args.is_causal, args.return_debug_mask, args.scale);
          output.impl()->tensor() = output_.transpose(1, 2);
          logsumexp.impl()->tensor() = logsumexp_;
          cum_seq_q.impl()->tensor() = cum_seq_q_;
          cum_seq_k.impl()->tensor() = cum_seq_k_;
          max->max_q = max_q_.as_int_unchecked();
          max->max_k = max_k_.as_int_unchecked();
          philox_seed.impl()->tensor() = philox_seed_;
          philox_offset.impl()->tensor() = philox_offset_;
          query.reset();
          key.reset();
          value.reset();
          logsumexp.reset();
          cum_seq_q.reset();
          cum_seq_k.reset();
          philox_seed.reset();
          philox_offset.reset();
        }
        CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
      }};

  const TaskFuture future = task.get_future();
  utils::resetFuture(output_, future);
  utils::resetFuture(query, future);
  utils::resetFuture(key, future);
  utils::resetFuture(value, future);
  utils::resetFuture(logsumexp, future);
  utils::resetFuture(cum_seq_q, future);
  utils::resetFuture(cum_seq_k, future);
  utils::resetFuture(philox_seed, future);
  utils::resetFuture(philox_offset, future);
  state->backward.query = query;
  state->backward.key = key;
  state->backward.value = value;
  state->backward.out = output_;
  state->backward.logsumexp = logsumexp;
  state->backward.cum_seq_q = cum_seq_q;
  state->backward.cum_seq_k = cum_seq_k;
  state->backward.philox_seed = philox_seed;
  state->backward.philox_offset = philox_offset;
  // TODO(Jie): support different attention algorithm
  // size
  output.sizes() = query.sizes();
  output = output_;
  scheduler.impl()->submit(std::move(task));
}

void ScaledDotProductFlashAttention::backward(
    const Scheduler &scheduler, const std::shared_ptr<State> &state,
    Tensor &grad_query, Tensor &grad_key, Tensor &grad_value,
    const ReadOnlyTensor &grad_out) {
  Tensor grad_query_;
  Tensor grad_key_;
  Tensor grad_value_;
  auto task = TaskCompute{
      [args = state->args, grad_query = grad_query_, grad_key = grad_key_,
       grad_value = grad_value_, grad_out = grad_out,
       query = state->backward.query, key = state->backward.key,
       value = state->backward.value, out = state->backward.out,
       logsumexp = state->backward.logsumexp,
       cum_seq_q = state->backward.cum_seq_q,
       cum_seq_k = state->backward.cum_seq_k, max = *state->backward.max,
       philox_seed = state->backward.philox_seed,
       philox_offset = state->backward.philox_offset,
       grad_outFuture = utils::future(grad_out),
       outputFuture = utils::future(state->backward.out),
       logsumexpFuture = utils::future(state->backward.logsumexp),
       cum_seq_qFuture = utils::future(state->backward.cum_seq_q),
       cum_seq_kFuture = utils::future(state->backward.cum_seq_k),
       philox_seedFuture = utils::future(state->backward.philox_seed),
       philox_offsetFuture = utils::future(state->backward.philox_offset),
       queryFuture = utils::future(state->backward.query),
       keyFuture = utils::future(state->backward.key),
       valueFuture = utils::future(state->backward.value)](
          const ContextCompute *context) mutable {
        DLLM_NVTX_RANGE_FN(
            "dllm::compute::ScaledDotProductAttention::backward");
        {
          utils::FutureGuard grad_outGuard{grad_outFuture};
          utils::FutureGuard queryGuard{queryFuture};
          utils::FutureGuard keyGuard{keyFuture};
          utils::FutureGuard valueGuard{valueFuture};
          utils::FutureGuard outputGuard{outputFuture};
          utils::FutureGuard logsumexpGuard{logsumexpFuture};
          utils::FutureGuard cum_seq_qGuard{cum_seq_qFuture};
          utils::FutureGuard cum_seq_kGuard{cum_seq_kFuture};
          utils::FutureGuard philox_seedGuard{philox_seedFuture};
          utils::FutureGuard philox_offsetGuard{philox_offsetFuture};
          auto [grad_query_, grad_key_, grad_value_] =
              at::_scaled_dot_product_flash_attention_backward(
                  grad_out.impl()->tensor().transpose(1, 2),
                  query.impl()->tensor().transpose(1, 2),
                  key.impl()->tensor().transpose(1, 2),
                  value.impl()->tensor().transpose(1, 2),
                  out.impl()->tensor().transpose(1, 2),
                  logsumexp.impl()->tensor(), cum_seq_q.impl()->tensor(),
                  cum_seq_k.impl()->tensor(), max.max_q, max.max_k,
                  args.dropout_p, args.is_causal, philox_seed.impl()->tensor(),
                  philox_offset.impl()->tensor(), args.scale);
          grad_query.impl()->tensor() = grad_query_.transpose(1, 2);
          grad_key.impl()->tensor() = grad_key_.transpose(1, 2);
          grad_value.impl()->tensor() = grad_value_.transpose(1, 2);
        }
        CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
        grad_query.reset();
        grad_key.reset();
        grad_value.reset();
        grad_out.reset();
        query.reset();
        key.reset();
        value.reset();
        out.reset();
        logsumexp.reset();
        cum_seq_q.reset();
        cum_seq_k.reset();
        philox_seed.reset();
        philox_offset.reset();
      }};

  const TaskFuture future = task.get_future();
  utils::resetFuture(grad_query_, future);
  utils::resetFuture(grad_key_, future);
  utils::resetFuture(grad_value_, future);
  utils::resetFuture(grad_out, future);
  // size
  grad_query_.sizes() = state->backward.query.sizes();
  grad_key_.sizes() = state->backward.key.sizes();
  grad_value_.sizes() = state->backward.value.sizes();
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
  grad_query = grad_query_;
  grad_key = grad_key_;
  grad_value = grad_value_;
  scheduler.impl()->submit(std::move(task));
}
}  // namespace dllm::compute
