#include "optimizer/adamw.h"

#include <torch/csrc/autograd/generated/variable_factories.h>

#include "internal_utils.h"
#include "logger.h"
#include "module/module.h"
#include "nvtx_helper.h"
#include "tensor_friend.h"
#include "threading/scheduler_impl.h"
#include "threading/task_compute.h"
#include "threading/thread_pool_compute.h"

namespace dllm::optimizer {
void stepKernel(cudaStream_t stream, const AdamW::State::Options &options,
                const std::shared_ptr<Tensor> &w,
                const std::shared_ptr<Tensor> &m,
                const std::shared_ptr<Tensor> &v,
                const std::shared_ptr<const ReadOnlyTensor> &dw);

void stepKernelAmsgrad(cudaStream_t stream,
                       const AdamW::State::Options &options,
                       const std::shared_ptr<Tensor> &w,
                       const std::shared_ptr<Tensor> &m,
                       const std::shared_ptr<Tensor> &v,
                       const std::shared_ptr<Tensor> &vMax,
                       const std::shared_ptr<const ReadOnlyTensor> &dw);

void AdamW::init(const Scheduler &scheduler, const module::Module &module,
                 const Options &options) {
  for (auto &kvState : module.named_states()) {
    for (auto &kvIncrement : kvState.value()->increments()) {
      std::shared_ptr<State> state;
      init(scheduler, state, kvIncrement->parameter, options);
      kvIncrement->optimizerState = state;
    }
  }
}

void AdamW::step(const Scheduler &scheduler, const module::Module &module) {
  for (auto &kvState : module.named_states()) {
    for (auto &kvIncrement : kvState.value()->increments()) {
      step(scheduler,
           std::dynamic_pointer_cast<State>(kvIncrement->optimizerState),
           kvIncrement->parameter, kvIncrement->gradient);
      kvIncrement->gradient = Tensor::create();
    }
  }
}

void AdamW::init(const Scheduler &scheduler, std::shared_ptr<State> &state,
                 const std::shared_ptr<const ReadOnlyTensor> &parameter,
                 const Options &options) {
  auto m = Tensor::create();
  auto v = Tensor::create();
  m->sizes() = parameter->sizes();
  v->sizes() = parameter->sizes();
  TaskCompute task;
  if (options.amsgrad()) {
    auto vMax = Tensor::create();
    vMax->sizes() = parameter->sizes();
    state = std::make_shared<State>(
        State::Tensors{m, v, vMax},
        State::Options{options.lr(), options.beta1(), options.beta2(),
                       options.eps(), options.weight_decay(), options.amsgrad(),
                       options.t()});
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
        State::Options{options.lr(), options.beta1(), options.beta2(),
                       options.eps(), options.weight_decay(), options.amsgrad(),
                       options.t()});
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
  scheduler.impl()->submit(std::move(task));
}

void AdamW::step(const Scheduler &scheduler,
                 const std::shared_ptr<State> &state,
                 const std::shared_ptr<Tensor> &w,
                 const std::shared_ptr<const ReadOnlyTensor> &dw) {
  state->options.t++;
  TaskCompute task;
  if (state->options.amsgrad) {
    const auto m = std::reinterpret_pointer_cast<Tensor>(state->tensors.m);
    const auto v = std::reinterpret_pointer_cast<Tensor>(state->tensors.v);
    const auto vMax =
        std::reinterpret_pointer_cast<Tensor>(state->tensors.vMax);
    task = TaskCompute{
        [options = state->options, m = m, v = v, vMax = vMax, w = w, dw = dw,
         wFuture = w->future(), mFuture = m->future(), vFuture = v->future(),
         vMaxFuture = vMax->future(),
         dwFuture = dw->future()](const ContextCompute *context) mutable {
          DLLM_NVTX_RANGE_FN("dllm::optimizer::AdamW<true>::step");
          util::FutureGuard wGuard{wFuture};
          util::FutureGuard mGuard{mFuture};
          util::FutureGuard vGuard{vFuture};
          util::FutureGuard vMaxGuard{vMaxFuture};
          util::FutureGuard dwGuard{dwFuture};
          if (DLLM_EXTRACT_TENSOR(dw).defined()) {
            stepKernelAmsgrad(context->cudaStream, options, w, m, v, vMax, dw);
          } else {
            DLLM_WARN_TRUE(false, "got non-defined gradient, skip the update");
          }
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
        [options = state->options, m = m, v = v, w = w, dw = dw,
         wFuture = w->future(), mFuture = m->future(), vFuture = v->future(),
         dwFuture = dw->future()](const ContextCompute *context) mutable {
          DLLM_NVTX_RANGE_FN("dllm::optimizer::AdamW<true>::step");
          util::FutureGuard wGuard{wFuture};
          util::FutureGuard mGuard{mFuture};
          util::FutureGuard vGuard{vFuture};
          util::FutureGuard dwGuard{dwFuture};
          if (DLLM_EXTRACT_TENSOR(dw).defined()) {
            stepKernel(context->cudaStream, options, w, m, v, dw);
          } else {
            DLLM_WARN_TRUE(false, "got non-defined gradient, skip the update");
          }
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
  scheduler.impl()->submit(std::move(task));
}
}  // namespace dllm::optimizer
