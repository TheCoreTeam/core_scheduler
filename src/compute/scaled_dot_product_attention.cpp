#include "compute/scaled_dot_product_attention.h"

#include <ATen/ops/_scaled_dot_product_flash_attention.h>
#include <ATen/ops/_scaled_dot_product_flash_attention_backward.h>

#include "internal_utils.h"
#include "logger.h"
#include "nvtx_helper.h"
#include "tensor_friend.h"

namespace dllm::compute {
TaskCompute ScaledDotProductFlashAttention::init(std::shared_ptr<State> &state,
                                                 const Options &options) {
  state = std::make_shared<State>(
      State::Forward{}, State::Backward{},
      State::Args{options.dropout_p(), options.is_causal(),
                  options.return_debug_mask(), options.scale()});
  return TaskCompute{[](const ContextCompute *) {}};
}

TaskCompute ScaledDotProductFlashAttention::forward(
    const std::shared_ptr<State> &state, const std::shared_ptr<Tensor> &output,
    const std::shared_ptr<const ReadOnlyTensor> &query,
    const std::shared_ptr<const ReadOnlyTensor> &key,
    const std::shared_ptr<const ReadOnlyTensor> &value) {
  const auto logsumexp = Tensor::create();
  const auto cum_seq_q = Tensor::create();
  const auto cum_seq_k = Tensor::create();
  const auto philox_seed = Tensor::create();
  const auto philox_offset = Tensor::create();
  auto task = TaskCompute{
      [args = state->args, query = query, key = key, value = value,
       output = output, logsumexp = logsumexp, cum_seq_q = cum_seq_q,
       cum_seq_k = cum_seq_k, max = state->backward.max,
       philox_seed = philox_seed, philox_offset = philox_offset,
       outputFuture = output->future(), logsumexpFuture = logsumexp->future(),
       cum_seq_qFuture = cum_seq_q->future(),
       cum_seq_kFuture = cum_seq_k->future(),
       philox_seedFuture = philox_seed->future(),
       philox_offsetFuture = philox_offset->future(),
       queryFuture = query->future(), keyFuture = key->future(),
       valueFuture = value->future()](const ContextCompute *context) mutable {
        DLLM_NVTX_RANGE_FN("dllm::compute::ScaledDotProductAttention::forward");
        {
          util::FutureGuard queryGuard{queryFuture};
          util::FutureGuard keyGuard{keyFuture};
          util::FutureGuard valueGuard{valueFuture};
          util::FutureGuard outputGuard{outputFuture};
          util::FutureGuard logsumexpGuard{logsumexpFuture};
          util::FutureGuard cum_seq_qGuard{cum_seq_qFuture};
          util::FutureGuard cum_seq_kGuard{cum_seq_kFuture};
          util::FutureGuard philox_seedGuard{philox_seedFuture};
          util::FutureGuard philox_offsetGuard{philox_offsetFuture};
          auto [output_, logsumexp_, cum_seq_q_, cum_seq_k_, max_q_, max_k_,
                philox_seed_, philox_offset_, debug_attn_mask_] =
              at::_scaled_dot_product_flash_attention(
                  DLLM_EXTRACT_TENSOR(query).transpose(1, 2),
                  DLLM_EXTRACT_TENSOR(key).transpose(1, 2),
                  DLLM_EXTRACT_TENSOR(value).transpose(1, 2), args.dropout_p,
                  args.is_causal, args.return_debug_mask, args.scale);
          DLLM_EXTRACT_TENSOR(output) = output_.transpose(1, 2);
          DLLM_EXTRACT_TENSOR(logsumexp) = logsumexp_;
          DLLM_EXTRACT_TENSOR(cum_seq_q) = cum_seq_q_;
          DLLM_EXTRACT_TENSOR(cum_seq_k) = cum_seq_k_;
          max->max_q = max_q_.as_int_unchecked();
          max->max_k = max_k_.as_int_unchecked();
          DLLM_EXTRACT_TENSOR(philox_seed) = philox_seed_;
          DLLM_EXTRACT_TENSOR(philox_offset) = philox_offset_;
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
  output->resetFuture(future);
  query->resetFuture(future);
  key->resetFuture(future);
  value->resetFuture(future);
  logsumexp->resetFuture(future);
  cum_seq_q->resetFuture(future);
  cum_seq_k->resetFuture(future);
  philox_seed->resetFuture(future);
  philox_offset->resetFuture(future);
  state->backward.query = query;
  state->backward.key = key;
  state->backward.value = value;
  state->backward.out = output;
  state->backward.logsumexp = logsumexp;
  state->backward.cum_seq_q = cum_seq_q;
  state->backward.cum_seq_k = cum_seq_k;
  state->backward.philox_seed = philox_seed;
  state->backward.philox_offset = philox_offset;
  // TODO(Jie): support different attention algorithm
  // size
  output->sizes() = query->sizes();
  return task;
}

TaskCompute ScaledDotProductFlashAttention::backward(
    const std::shared_ptr<State> &state,
    const std::shared_ptr<Tensor> &grad_query,
    const std::shared_ptr<Tensor> &grad_key,
    const std::shared_ptr<Tensor> &grad_value,
    const std::shared_ptr<const ReadOnlyTensor> &grad_out) {
  auto task = TaskCompute{
      [args = state->args, grad_query = grad_query, grad_key = grad_key,
       grad_value = grad_value, grad_out = grad_out,
       query = state->backward.query, key = state->backward.key,
       value = state->backward.value, out = state->backward.out,
       logsumexp = state->backward.logsumexp,
       cum_seq_q = state->backward.cum_seq_q,
       cum_seq_k = state->backward.cum_seq_k, max = *state->backward.max,
       philox_seed = state->backward.philox_seed,
       philox_offset = state->backward.philox_offset,
       grad_queryFuture = grad_query->future(),
       grad_keyFuture = grad_key->future(),
       grad_valueFuture = grad_value->future(),
       grad_outFuture = grad_out->future(),
       outputFuture = state->backward.out->future(),
       logsumexpFuture = state->backward.logsumexp->future(),
       cum_seq_qFuture = state->backward.cum_seq_q->future(),
       cum_seq_kFuture = state->backward.cum_seq_k->future(),
       philox_seedFuture = state->backward.philox_seed->future(),
       philox_offsetFuture = state->backward.philox_offset->future(),
       queryFuture = state->backward.query->future(),
       keyFuture = state->backward.key->future(),
       valueFuture = state->backward.value->future()](
          const ContextCompute *context) mutable {
        DLLM_NVTX_RANGE_FN(
            "dllm::compute::ScaledDotProductAttention::backward");
        {
          util::FutureGuard grad_queryGuard{grad_queryFuture};
          util::FutureGuard grad_keyGuard{grad_keyFuture};
          util::FutureGuard grad_valueGuard{grad_valueFuture};
          util::FutureGuard grad_outGuard{grad_outFuture};
          util::FutureGuard queryGuard{queryFuture};
          util::FutureGuard keyGuard{keyFuture};
          util::FutureGuard valueGuard{valueFuture};
          util::FutureGuard outputGuard{outputFuture};
          util::FutureGuard logsumexpGuard{logsumexpFuture};
          util::FutureGuard cum_seq_qGuard{cum_seq_qFuture};
          util::FutureGuard cum_seq_kGuard{cum_seq_kFuture};
          util::FutureGuard philox_seedGuard{philox_seedFuture};
          util::FutureGuard philox_offsetGuard{philox_offsetFuture};
          auto [grad_query_, grad_key_, grad_value_] =
              at::_scaled_dot_product_flash_attention_backward(
                  DLLM_EXTRACT_TENSOR(grad_out).transpose(1, 2),
                  DLLM_EXTRACT_TENSOR(query).transpose(1, 2),
                  DLLM_EXTRACT_TENSOR(key).transpose(1, 2),
                  DLLM_EXTRACT_TENSOR(value).transpose(1, 2),
                  DLLM_EXTRACT_TENSOR(out).transpose(1, 2),
                  DLLM_EXTRACT_TENSOR(logsumexp),
                  DLLM_EXTRACT_TENSOR(cum_seq_q),
                  DLLM_EXTRACT_TENSOR(cum_seq_k), max.max_q, max.max_k,
                  args.dropout_p, args.is_causal,
                  DLLM_EXTRACT_TENSOR(philox_seed),
                  DLLM_EXTRACT_TENSOR(philox_offset), args.scale);
          DLLM_EXTRACT_TENSOR(grad_query) = grad_query_.transpose(1, 2);
          DLLM_EXTRACT_TENSOR(grad_key) = grad_key_.transpose(1, 2);
          DLLM_EXTRACT_TENSOR(grad_value) = grad_value_.transpose(1, 2);
          grad_query.reset();
          grad_key.reset();
          grad_value.reset();
          grad_out.reset();
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
  grad_query->resetFuture(future);
  grad_key->resetFuture(future);
  grad_value->resetFuture(future);
  grad_out->resetFuture(future);
  state->backward.query->resetFuture(future);
  state->backward.key->resetFuture(future);
  state->backward.value->resetFuture(future);
  state->backward.out->resetFuture(future);
  state->backward.logsumexp->resetFuture(future);
  state->backward.cum_seq_q->resetFuture(future);
  state->backward.cum_seq_k->resetFuture(future);
  state->backward.philox_seed->resetFuture(future);
  state->backward.philox_offset->resetFuture(future);
  // size
  grad_query->sizes() = state->backward.query->sizes();
  grad_key->sizes() = state->backward.key->sizes();
  grad_value->sizes() = state->backward.value->sizes();
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
  return task;
}
}  // namespace dllm::compute
