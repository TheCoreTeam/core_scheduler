#include "compute/embedding.h"

#include <torch/nn/functional/embedding.h>

#include "internal_utils.h"
#include "logger.h"
#include "nvtx_helper.h"
#include "tensor_friend.h"

namespace dllm::compute {
TaskCompute Embedding::init(std::shared_ptr<State> &state,
                            const Options &options) {
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

  auto weight = Tensor::create();

  auto task =
      TaskCompute{[=, weight = weight](const ContextCompute *context) mutable {
        DLLM_NVTX_RANGE_FN("dllm::compute::Embedding::init");
        const auto weight_ = at::normal(
            0, 1, {options.num_embeddings(), options.embedding_dim()}, {},
            options.dtype(), {}, options.device(), {});
        DLLM_EXTRACT_TENSOR(weight) = weight_;
        // if (max_norm != c10::nullopt) {
        //   input_ = input_.contiguous();
        //   _no_grad_embedding_renorm_(weight, input_, *max_norm, norm_type);
        // }
        CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
        weight.reset();
      }};

  const TaskFuture future = task.get_future();
  weight->resetFuture(future);
  weight->sizes() = IntArray{options.num_embeddings(), options.embedding_dim()};

  state = std::make_shared<State>(
      State::Forward{std::move(weight)}, State::Backward{},
      State::Args{options.num_embeddings(), padding_idx, options.max_norm(),
                  options.norm_type(), options.scale_grad_by_freq(),
                  options.sparse()});
  return task;
}

TaskCompute Embedding::forward(
    const std::shared_ptr<State> &state, const std::shared_ptr<Tensor> &output,
    const std::shared_ptr<const ReadOnlyTensor> &indices) {
  auto task = TaskCompute{
      [padding_idx = state->args.padding_idx,
       scale_grad_by_freq = state->args.scale_grad_by_freq,
       sparse = state->args.sparse, weight = state->forward.weight,
       output = output, indices = indices, outputFuture = output->future(),
       weightFuture = state->forward.weight->future(),
       indicesFuture =
           indices->future()](const ContextCompute *context) mutable {
        DLLM_NVTX_RANGE_FN("dllm::compute::Embedding::forward");
        {
          util::FutureGuard outputrGuard{outputFuture};
          util::FutureGuard weightGuard{weightFuture};
          util::FutureGuard indicesGuard{indicesFuture};
          DLLM_EXTRACT_TENSOR(output) =
              torch::embedding(DLLM_EXTRACT_TENSOR(weight),
                               DLLM_EXTRACT_TENSOR(indices).view(
                                   {-1, DLLM_EXTRACT_TENSOR(indices).size(-1)}),
                               padding_idx, scale_grad_by_freq, sparse);
          weight.reset();
          output.reset();
          indices.reset();
        }
        CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
      }};
  const TaskFuture future = task.get_future();
  state->forward.weight->resetFuture(future);
  indices->resetFuture(future);
  output->resetFuture(future);
  state->backward.indices = indices;
  // size
  output->sizes() = [&]() {
    auto sizes = indices->sizes();
    sizes.push_back(state->forward.weight->size(1));
    return sizes;
  }();
  return task;
}

TaskCompute Embedding::backward(
    const std::shared_ptr<State> &state,
    const std::shared_ptr<const ReadOnlyTensor> &grad_output) {
  auto task = TaskCompute{
      [num_weights = state->args.num_weights,
       padding_idx = state->args.padding_idx,
       scale_grad_by_freq = state->args.scale_grad_by_freq,
       sparse = state->args.sparse, grad_weight = state->forward.grad_weight,
       grad_output = grad_output, indices = state->backward.indices,
       grad_weight_future = state->forward.grad_weight->future(),
       grad_output_future = grad_output->future(),
       indicesFuture = state->backward.indices->future(),
       weightFuture = state->forward.weight->future()](
          const ContextCompute *context) mutable {
        DLLM_NVTX_RANGE_FN("dllm::compute::Embedding::backward");
        {
          util::FutureGuard grad_weightGuard{grad_weight_future};
          util::FutureGuard grad_output_Guard{grad_output_future};
          util::FutureGuard indicesGuard{indicesFuture};
          util::FutureGuard weighGuard{weightFuture};
          if (DLLM_EXTRACT_TENSOR(grad_weight).defined()) {
            DLLM_EXTRACT_TENSOR(grad_weight) += torch::embedding_backward(
                DLLM_EXTRACT_TENSOR(grad_output), DLLM_EXTRACT_TENSOR(indices),
                num_weights, padding_idx, scale_grad_by_freq, sparse);
          } else /* accumulate grad */ {
            DLLM_EXTRACT_TENSOR(grad_weight) = torch::embedding_backward(
                DLLM_EXTRACT_TENSOR(grad_output), DLLM_EXTRACT_TENSOR(indices),
                num_weights, padding_idx, scale_grad_by_freq, sparse);
          }
          grad_weight.reset();
          grad_output.reset();
          indices.reset();
        }
        CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
      }};
  const TaskFuture future = task.get_future();
  state->forward.grad_weight->resetFuture(future);
  state->forward.grad_weight->resetFuture(future);
  state->forward.grad_weight->sizes() = state->forward.weight->sizes();
  grad_output->resetFuture(future);
  state->backward.indices->resetFuture(future);
  state->forward.weight->resetFuture(future);
  // decrease counter
  state->backward.indices.reset();
  return task;
}
}  // namespace dllm::compute
