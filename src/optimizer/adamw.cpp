#include "optimizer/adamw.h"

#include <torch/csrc/autograd/generated/variable_factories.h>

#include "internal_utils.h"
#include "logger.h"
#include "module/module.h"
#include "nvtx_helper.h"
#include "tensor_friend.h"
#include "threading/task_compute.h"
#include "threading/thread_pool_compute.h"

namespace dllm::optimizer {
void stepKernel(cudaStream_t stream, const AdamW::State::Args &args,
                const std::shared_ptr<Tensor> &w,
                const std::shared_ptr<Tensor> &m,
                const std::shared_ptr<Tensor> &v,
                const std::shared_ptr<const ReadOnlyTensor> &dw);

void stepKernelAmsgrad(cudaStream_t stream, const AdamW::State::Args &args,
                       const std::shared_ptr<Tensor> &w,
                       const std::shared_ptr<Tensor> &m,
                       const std::shared_ptr<Tensor> &v,
                       const std::shared_ptr<Tensor> &vMax,
                       const std::shared_ptr<const ReadOnlyTensor> &dw);

void AdamW::init(ThreadPoolCompute &tp, const module::Module &module,
                 const double lr, const double beta1, const double beta2,
                 const double eps, const double weight_decay,
                 const bool amsgrad, const long t) {
  for (auto &kvState : module.named_states()) {
    for (auto &kvIncrement : kvState.value()->increments()) {
      std::shared_ptr<State> state;
      tp.submit(init(state, kvIncrement->parameter, lr, beta1, beta2, eps,
                     weight_decay, amsgrad, t));
      kvIncrement->optimizerState = state;
    }
  }
}

void AdamW::step(ThreadPoolCompute &tp, const module::Module &module) {
  for (auto &kvState : module.named_states()) {
    for (auto &kvIncrement : kvState.value()->increments()) {
      tp.submit(
          step(std::dynamic_pointer_cast<State>(kvIncrement->optimizerState),
               kvIncrement->parameter, kvIncrement->gradient));
      kvIncrement->gradient = Tensor::create();
    }
  }
}

TaskCompute AdamW::init(std::shared_ptr<State> &state,
                        const std::shared_ptr<const ReadOnlyTensor> &parameter,
                        const double lr, const double beta1, const double beta2,
                        const double eps, const double weight_decay,
                        const bool amsgrad, const long t) {
  auto m = Tensor::create();
  auto v = Tensor::create();
  m->sizes() = parameter->sizes();
  v->sizes() = parameter->sizes();
  TaskCompute task;
  if (amsgrad) {
    auto vMax = Tensor::create();
    vMax->sizes() = parameter->sizes();
    state = std::make_shared<State>(
        State::Tensors{m, v, vMax},
        State::Args{lr, beta1, beta2, eps, weight_decay, amsgrad, t});
    task = TaskCompute{[parameter = parameter, m = m, v = v, vMax = vMax,
                        parameterFuture = parameter->future()](
                           const ContextCompute *context) mutable {
      util::FutureGuard guard{parameterFuture};
      DLLM_EXTRACT_TENSOR(m) = at::zeros_like(DLLM_EXTRACT_TENSOR(parameter));
      DLLM_EXTRACT_TENSOR(v) = at::zeros_like(DLLM_EXTRACT_TENSOR(parameter));
      DLLM_EXTRACT_TENSOR(vMax) =
          at::zeros_like(DLLM_EXTRACT_TENSOR(parameter));
      CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
      parameter.reset();
      m.reset();
      v.reset();
      vMax.reset();
    }};
    const TaskFuture future = task.get_future();
    m->resetFuture(future);
    v->resetFuture(future);
    vMax->resetFuture(future);
  } else {
    state = std::make_shared<State>(
        State::Tensors{m, v},
        State::Args{lr, beta1, beta2, eps, weight_decay, amsgrad, t});
    task = TaskCompute{[parameter = parameter, m = m, v = v,
                        parameterFuture = parameter->future()](
                           const ContextCompute *context) mutable {
      util::FutureGuard guard{parameterFuture};
      DLLM_EXTRACT_TENSOR(m) = at::zeros_like(DLLM_EXTRACT_TENSOR(parameter));
      DLLM_EXTRACT_TENSOR(v) = at::zeros_like(DLLM_EXTRACT_TENSOR(parameter));
      CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
      parameter.reset();
      m.reset();
      v.reset();
    }};
    const TaskFuture future = task.get_future();
    m->resetFuture(future);
    v->resetFuture(future);
  }
  return task;
}

TaskCompute AdamW::step(const std::shared_ptr<State> &state,
                        const std::shared_ptr<Tensor> &w,
                        const std::shared_ptr<const ReadOnlyTensor> &dw) {
  state->args.t++;
  TaskCompute task;
  if (state->args.amsgrad) {
    const auto m = std::reinterpret_pointer_cast<Tensor>(state->tensors.m);
    const auto v = std::reinterpret_pointer_cast<Tensor>(state->tensors.v);
    const auto vMax =
        std::reinterpret_pointer_cast<Tensor>(state->tensors.vMax);
    task = TaskCompute{
        [args = state->args, m = m, v = v, vMax = vMax, w = w, dw = dw,
         wFuture = w->future(), mFuture = m->future(), vFuture = v->future(),
         vMaxFuture = vMax->future(),
         dwFuture = dw->future()](const ContextCompute *context) mutable {
          DLLM_NVTX_RANGE_FN("dllm::optimizer::AdamW<true>::step");
          util::FutureGuard wGuard{wFuture};
          util::FutureGuard mGuard{mFuture};
          util::FutureGuard vGuard{vFuture};
          util::FutureGuard vMaxGuard{vMaxFuture};
          util::FutureGuard dwGuard{dwFuture};
          stepKernelAmsgrad(context->cudaStream, args, w, m, v, vMax, dw);
          CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
          m.reset();
          v.reset();
          vMax.reset();
          w.reset();
          dw.reset();
        }};
    const TaskFuture future = task.get_future();
    w->resetFuture(future);
    m->resetFuture(future);
    v->resetFuture(future);
    vMax->resetFuture(future);
    dw->resetFuture(future);
  } else {
    const auto m = std::reinterpret_pointer_cast<Tensor>(state->tensors.m);
    const auto v = std::reinterpret_pointer_cast<Tensor>(state->tensors.v);
    task = TaskCompute{
        [args = state->args, m = m, v = v, w = w, dw = dw,
         wFuture = w->future(), mFuture = m->future(), vFuture = v->future(),
         dwFuture = dw->future()](const ContextCompute *context) mutable {
          DLLM_NVTX_RANGE_FN("dllm::optimizer::AdamW<true>::step");
          util::FutureGuard wGuard{wFuture};
          util::FutureGuard mGuard{mFuture};
          util::FutureGuard vGuard{vFuture};
          util::FutureGuard dwGuard{dwFuture};
          stepKernel(context->cudaStream, args, w, m, v, dw);
          CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
          m.reset();
          v.reset();
          w.reset();
          dw.reset();
        }};
    const TaskFuture future = task.get_future();
    w->resetFuture(future);
    m->resetFuture(future);
    v->resetFuture(future);
    dw->resetFuture(future);
  }
  return task;
}
}  // namespace dllm::optimizer
