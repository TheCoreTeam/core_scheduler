#include "compute/embedding.h"

#include <torch/nn/functional/embedding.h>

#include "internal_utils.h"
#include "logger.h"

namespace dllm::compute {
void Embedding::init(std::shared_ptr<ForwardState> &state,
                     int64_t num_embeddings, int64_t embedding_dim,
                     c10::optional<int64_t> padding_idx,
                     c10::optional<double> max_norm, double norm_type,
                     bool scale_grad_by_freq, bool sparse,
                     const c10::optional<at::Device> device,
                     const c10::optional<at::ScalarType> dtype) {
  auto weight = std::make_shared<Tensor>(at::normal(
      0, 1, {num_embeddings, embedding_dim}, {}, dtype, {}, device, {}));
  if (padding_idx != c10::nullopt) {
    if (*padding_idx > 0) {
      TORCH_CHECK(*padding_idx < weight->tensor().size(0),
                  "Padding_idx must be within num_embeddings");
    } else if (*padding_idx < 0) {
      TORCH_CHECK(*padding_idx >= -weight->tensor().size(0),
                  "Padding_idx must be within num_embedding");
      padding_idx = weight->tensor().size(0) + *padding_idx;
    }
  } else {
    padding_idx = -1;
  }

  TORCH_CHECK(max_norm == c10::nullopt)
  // if (max_norm != c10::nullopt) {
  //   input_ = input_.contiguous();
  //   _no_grad_embedding_renorm_(weight, input_, *max_norm, norm_type);
  // }

  state = std::make_shared<ForwardState>(std::move(weight), num_embeddings,
                                         padding_idx.value(), max_norm,
                                         norm_type, scale_grad_by_freq, sparse);
}

TaskCompute Embedding::forward(
    std::shared_ptr<BackwardState> &backwardState,
    const std::shared_ptr<const ForwardState> &forwardState,
    const std::shared_ptr<Tensor> &output,
    const std::shared_ptr<const Tensor> &indices) {
  auto task =
      TaskCompute{[state = forwardState, output = output, indices = indices,
                   outputFuture = output->future(),
                   weightFuture = indices->future().wFuture,
                   indicesFuture = forwardState->weight->future().wFuture](
                      const ContextCompute *context) mutable {
        {
          util::FutureGuard utputrGuard{outputFuture};
          util::FutureGuard weightGuard{weightFuture};
          util::FutureGuard indicesGuard{indicesFuture};

          output->tensor() = torch::embedding(
              state->weight->tensor(), indices->tensor(), state->padding_idx,
              state->scale_grad_by_freq, state->sparse);
        }
        CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
        state.reset();
        output.reset();
        indices.reset();
      }};
  const TaskFuture future = task.get_future();
  forwardState->weight->future().rFuture = future;
  indices->future().rFuture = future;
  output->future().wFuture = future;
  backwardState = std::make_shared<BackwardState>(
      forwardState->weight, indices, forwardState->num_weights,
      forwardState->padding_idx, forwardState->max_norm,
      forwardState->norm_type, forwardState->scale_grad_by_freq,
      forwardState->sparse);
  return task;
}

TaskCompute Embedding::backward(
    const std::shared_ptr<Tensor> &grad_weight,
    const std::shared_ptr<const BackwardState> &backwardState,
    const std::shared_ptr<const Tensor> &grad_output) {
  auto task =
      TaskCompute{[grad_weight = grad_weight, grad_output = grad_output,
                   indices = backwardState->indices, state = backwardState,
                   grad_weight_future = grad_weight->future(),
                   grad_output_future = grad_output->future().wFuture,
                   indicesFuture = backwardState->indices->future().wFuture,
                   weightFuture = backwardState->weight->future().wFuture](
                      const ContextCompute *context) mutable {
        util::FutureGuard grad_weightGuard{grad_weight_future};
        util::FutureGuard grad_output_Guard{grad_output_future};
        util::FutureGuard indicesGuard{indicesFuture};
        util::FutureGuard weighGuard{weightFuture};
        grad_weight->tensor() = torch::embedding_backward(
            grad_output->tensor(), indices->tensor(), state->num_weights,
            state->padding_idx, state->scale_grad_by_freq, state->sparse);
        CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
        grad_weight.reset();
        grad_output.reset();
        indices.reset();
        state.reset();
      }};
  const TaskFuture future = task.get_future();
  grad_weight->future().wFuture = future;
  grad_output->future().rFuture = future;
  backwardState->indices->future().rFuture = future;
  backwardState->weight->future().rFuture = future;
  return task;
}
}  // namespace dllm::compute
