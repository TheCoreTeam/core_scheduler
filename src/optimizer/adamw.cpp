#include "optimizer/adamw.h"

#include <torch/csrc/autograd/generated/variable_factories.h>

#include "internal_utils.h"
#include "logger.h"
#include "module/module.h"
#include "nvtx_helper.h"
#include "tensor_impl.h"
#include "threading/scheduler_impl.h"
#include "threading/task_compute.h"
#include "threading/thread_pool_compute.h"

namespace dllm::optimizer {
void stepKernel(cudaStream_t stream, const AdamW::State::Options &options,
                Tensor &w, Tensor &m, Tensor &v, const ReadOnlyTensor &dw);

void stepKernelAmsgrad(cudaStream_t stream,
                       const AdamW::State::Options &options, Tensor &w,
                       Tensor &m, Tensor &v, Tensor &vMax,
                       const ReadOnlyTensor &dw);

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
      kvIncrement->gradient = Tensor{};
    }
  }
}

void AdamW::init(const Scheduler &scheduler, std::shared_ptr<State> &state,
                 const ReadOnlyTensor &parameter, const Options &options) {
  Tensor m;
  Tensor v;
  m.sizes() = parameter.sizes();
  v.sizes() = parameter.sizes();
  TaskCompute task;
  if (options.amsgrad()) {
    Tensor vMax;
    vMax.sizes() = parameter.sizes();
    state = std::make_shared<State>(
        State::Tensors{m, v, vMax},
        State::Options{options.lr(), options.beta1(), options.beta2(),
                       options.eps(), options.weight_decay(), options.amsgrad(),
                       options.t()});
    task = TaskCompute{[parameter = parameter, m = m, v = v, vMax = vMax,
                        parameterFuture = utils::future(parameter)](
                           const ContextCompute *context) mutable {
      utils::FutureGuard guard{parameterFuture};
      m.impl()->tensor() = at::zeros_like(parameter.impl()->tensor());
      v.impl()->tensor() = at::zeros_like(parameter.impl()->tensor());
      vMax.impl()->tensor() = at::zeros_like(parameter.impl()->tensor());
      CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
      parameter.reset();
      m.reset();
      v.reset();
      vMax.reset();
    }};
    const TaskFuture future = task.get_future();
    utils::resetFuture(m, future);
    utils::resetFuture(v, future);
    utils::resetFuture(vMax, future);
  } else {
    state = std::make_shared<State>(
        State::Tensors{m, v},
        State::Options{options.lr(), options.beta1(), options.beta2(),
                       options.eps(), options.weight_decay(), options.amsgrad(),
                       options.t()});
    task = TaskCompute{[parameter = parameter, m = m, v = v,
                        parameterFuture = utils::future(parameter)](
                           const ContextCompute *context) mutable {
      utils::FutureGuard guard{parameterFuture};
      m.impl()->tensor() = at::zeros_like(parameter.impl()->tensor());
      v.impl()->tensor() = at::zeros_like(parameter.impl()->tensor());
      CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
      parameter.reset();
      m.reset();
      v.reset();
    }};
    const TaskFuture future = task.get_future();
    utils::resetFuture(m, future);
    utils::resetFuture(v, future);
  }
  scheduler.impl()->submit(std::move(task));
}

void AdamW::step(const Scheduler &scheduler,
                 const std::shared_ptr<State> &state, Tensor &w,
                 const ReadOnlyTensor &dw) {
  state->options.t++;
  TaskCompute task;
  if (state->options.amsgrad) {
    const auto &m = state->tensors.m;
    const auto &v = state->tensors.v;
    const auto &vMax = state->tensors.vMax;
    task = TaskCompute{
        [options = state->options, m = m, v = v, vMax = vMax, w = w, dw = dw,
         wFuture = utils::future(w), mFuture = utils::future(m),
         vFuture = utils::future(v), vMaxFuture = utils::future(vMax),
         dwFuture = utils::future(dw)](const ContextCompute *context) mutable {
          DLLM_NVTX_RANGE_FN("dllm::optimizer::AdamW<true>::step");
          utils::FutureGuard wGuard{wFuture};
          utils::FutureGuard mGuard{mFuture};
          utils::FutureGuard vGuard{vFuture};
          utils::FutureGuard vMaxGuard{vMaxFuture};
          utils::FutureGuard dwGuard{dwFuture};
          if (dw.impl()->tensor().defined()) {
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
    utils::resetFuture(w, future);
    utils::resetFuture(m, future);
    utils::resetFuture(v, future);
    utils::resetFuture(vMax, future);
    utils::resetFuture(dw, future);
  } else {
    const auto &m = state->tensors.m;
    const auto &v = state->tensors.v;
    task =
        TaskCompute{[options = state->options, m = m, v = v, w = w, dw = dw,
                     wFuture = utils::future(w), mFuture = utils::future(m),
                     vFuture = utils::future(v), dwFuture = utils::future(dw)](
                        const ContextCompute *context) mutable {
          DLLM_NVTX_RANGE_FN("dllm::optimizer::AdamW<true>::step");
          utils::FutureGuard wGuard{wFuture};
          utils::FutureGuard mGuard{mFuture};
          utils::FutureGuard vGuard{vFuture};
          utils::FutureGuard dwGuard{dwFuture};
          if (dw.impl()->tensor().defined()) {
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
    utils::resetFuture(w, future);
    utils::resetFuture(m, future);
    utils::resetFuture(v, future);
    utils::resetFuture(dw, future);
  }
  scheduler.impl()->submit(std::move(task));
}
}  // namespace dllm::optimizer
