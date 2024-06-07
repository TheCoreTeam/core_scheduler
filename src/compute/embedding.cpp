#include "compute/embedding.h"

#include <torch/nn/functional/embedding.h>

#include "internal_utils.h"
#include "logger.h"
#include "nvtx_helper.h"
#include "tensor_impl.h"
#include "threading/scheduler_impl.h"
#include "threading/task_compute.h"

namespace dllm::compute {
void Embedding::init(const Scheduler &scheduler, std::shared_ptr<State> &state,
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

  Tensor weight;

  auto task =
      TaskCompute{[=, weight = weight](const ContextCompute *context) mutable {
        DLLM_NVTX_RANGE_FN("dllm::compute::Embedding::init");
        const auto weight_ = at::normal(
            0, 1, {options.num_embeddings(), options.embedding_dim()}, {},
            options.dtype(), {}, options.device(), {});
        weight.impl()->tensor() = weight_;
        // if (max_norm != c10::nullopt) {
        //   input_ = input_.contiguous();
        //   _no_grad_embedding_renorm_(weight, input_, *max_norm, norm_type);
        // }
        CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
        weight.reset();
      }};

  const TaskFuture future = task.get_future();
  utils::resetFuture(weight, future);
  weight.sizes() = IntArray{options.num_embeddings(), options.embedding_dim()};

  state = std::make_shared<State>(
      State::Forward{std::move(weight)}, State::Backward{},
      State::Args{options.num_embeddings(), padding_idx, options.max_norm(),
                  options.norm_type(), options.scale_grad_by_freq(),
                  options.sparse()});
  scheduler.impl()->submit(std::move(task));
}

void Embedding::forward(const Scheduler &scheduler,
                        const std::shared_ptr<State> &state, Tensor &output,
                        const ReadOnlyTensor &indices) {
  output = Tensor{};
  auto task =
      TaskCompute{[padding_idx = state->args.padding_idx,
                   scale_grad_by_freq = state->args.scale_grad_by_freq,
                   sparse = state->args.sparse, weight = state->forward.weight,
                   output = output, indices = indices,
                   weightFuture = utils::future(state->forward.weight),
                   indicesFuture = utils::future(indices)](
                      const ContextCompute *context) mutable {
        DLLM_NVTX_RANGE_FN("dllm::compute::Embedding::forward");
        {
          utils::FutureGuard weightGuard{weightFuture};
          utils::FutureGuard indicesGuard{indicesFuture};
          output.impl()->tensor() =
              torch::embedding(weight.impl()->tensor(),
                               indices.impl()->tensor().view(
                                   {-1, indices.impl()->tensor().size(-1)}),
                               padding_idx, scale_grad_by_freq, sparse);
          weight.reset();
          output.reset();
          indices.reset();
        }
        CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
      }};
  const TaskFuture future = task.get_future();
  utils::resetFuture(state->forward.weight, future);
  utils::resetFuture(indices, future);
  utils::resetFuture(output, future);
  state->backward.indices = indices;
  // size
  output.sizes() = [&]() {
    auto sizes = indices.sizes();
    sizes.push_back(state->forward.weight.size(1));
    return sizes;
  }();
  scheduler.impl()->submit(std::move(task));
}

void Embedding::backward(const Scheduler &scheduler,
                         const std::shared_ptr<State> &state,
                         const ReadOnlyTensor &grad_output) {
  auto task = TaskCompute{
      [num_weights = state->args.num_weights,
       padding_idx = state->args.padding_idx,
       scale_grad_by_freq = state->args.scale_grad_by_freq,
       sparse = state->args.sparse, grad_weight = state->forward.grad_weight,
       grad_output = grad_output, indices = state->backward.indices,
       grad_weight_future = utils::future(state->forward.grad_weight),
       grad_output_future = utils::future(grad_output),
       indicesFuture = utils::future(state->backward.indices),
       weightFuture = utils::future(state->forward.weight)](
          const ContextCompute *context) mutable {
        DLLM_NVTX_RANGE_FN("dllm::compute::Embedding::backward");
        {
          utils::FutureGuard grad_weightGuard{grad_weight_future};
          utils::FutureGuard grad_output_Guard{grad_output_future};
          utils::FutureGuard indicesGuard{indicesFuture};
          utils::FutureGuard weighGuard{weightFuture};
          if (grad_weight.impl()->tensor().defined()) {
            grad_weight.impl()->tensor() += torch::embedding_backward(
                grad_output.impl()->tensor(), indices.impl()->tensor(),
                num_weights, padding_idx, scale_grad_by_freq, sparse);
          } else /* accumulate grad */ {
            grad_weight.impl()->tensor() = torch::embedding_backward(
                grad_output.impl()->tensor(), indices.impl()->tensor(),
                num_weights, padding_idx, scale_grad_by_freq, sparse);
          }
          grad_weight.reset();
          grad_output.reset();
          indices.reset();
        }
        CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
      }};
  const TaskFuture future = task.get_future();
  utils::resetFuture(state->forward.grad_weight, future);
  utils::resetFuture(state->forward.grad_weight, future);
  state->forward.grad_weight.sizes() = state->forward.weight.sizes();
  utils::resetFuture(grad_output, future);
  utils::resetFuture(state->backward.indices, future);
  utils::resetFuture(state->forward.weight, future);
  // decrease counter
  state->backward.indices.reset();
  scheduler.impl()->submit(std::move(task));
}
}  // namespace dllm::compute
