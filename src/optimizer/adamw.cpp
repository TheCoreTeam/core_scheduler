#include "optimizer/adamw.h"

#include <torch/csrc/autograd/generated/variable_factories.h>

#include "internal_utils.h"
#include "logger.h"
#include "tensor_friend.h"
#include "threading/task_compute.h"

namespace dllm::optimizer {
void stepKernel(cudaStream_t stream, const AdamW<false>::State::Args &args,
                const std::shared_ptr<Tensor> &w,
                const std::shared_ptr<Tensor> &m,
                const std::shared_ptr<Tensor> &v,
                const std::shared_ptr<const ReadOnlyTensor> &dw);

void stepKernel(cudaStream_t stream, const AdamW<true>::State::Args &args,
                const std::shared_ptr<Tensor> &w,
                const std::shared_ptr<Tensor> &m,
                const std::shared_ptr<Tensor> &v,
                const std::shared_ptr<Tensor> &vMax,
                const std::shared_ptr<const ReadOnlyTensor> &dw);

TaskCompute AdamW<false>::init(std::shared_ptr<State> &state,
                               const IntArrayRef &size, const double lr,
                               const double beta1, const double beta2,
                               const double eps, const double weight_decay,
                               const long t) {
  auto m = Tensor::create();
  auto v = Tensor::create();
  m->sizes() = size;
  v->sizes() = size;
  state = std::make_shared<State>(
      m, v, State::Args{lr, beta1, beta2, eps, weight_decay, t});
  auto task = TaskCompute{[](const ContextCompute *) {}};
  return task;
}

TaskCompute AdamW<false>::step(
    const std::shared_ptr<State> &state, const std::shared_ptr<Tensor> &w,
    const std::shared_ptr<const ReadOnlyTensor> &dw) {
  state->args.t++;
  // TODO(Jie): necessary check
  const auto m = std::reinterpret_pointer_cast<Tensor>(
      std::const_pointer_cast<ReadOnlyTensor>(state->m));
  const auto v = std::reinterpret_pointer_cast<Tensor>(
      std::const_pointer_cast<ReadOnlyTensor>(state->v));
  auto task = TaskCompute{
      [args = state->args, m = m, v = v, w = w, dw = dw, wFuture = w->future(),
       mFuture = m->future(), vFuture = v->future(),
       dwFuture = dw->future()](const ContextCompute *context) mutable {
        util::FutureGuard wGuard{wFuture};
        util::FutureGuard mGuard{mFuture};
        util::FutureGuard vGuard{vFuture};
        util::FutureGuard dwGuard{dwFuture};
        if (!DLLM_EXTRACT_TENSOR(m).defined()) {
          DLLM_EXTRACT_TENSOR(m) = torch::zeros_like(DLLM_EXTRACT_TENSOR(dw));
        }
        if (!DLLM_EXTRACT_TENSOR(v).defined()) {
          DLLM_EXTRACT_TENSOR(v) = torch::zeros_like(DLLM_EXTRACT_TENSOR(dw));
        }
        stepKernel(context->cudaStream, args, w, m, v, dw);
        CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
        w.reset();
        m.reset();
        v.reset();
        dw.reset();
      }};
  const TaskFuture future = task.get_future();
  w->resetFuture(future);
  m->resetFuture(future);
  v->resetFuture(future);
  dw->resetFuture(future);
  return task;
}

TaskCompute AdamW<true>::init(std::shared_ptr<State> &state,
                              const IntArrayRef &size, const double lr,
                              const double beta1, const double beta2,
                              const double eps, const double weight_decay,
                              const long t) {
  auto m = Tensor::create();
  auto v = Tensor::create();
  auto vMax = Tensor::create();
  m->sizes() = size;
  v->sizes() = size;
  vMax->sizes() = size;
  state = std::make_shared<State>(
      m, v, vMax, State::Args{lr, beta1, beta2, eps, weight_decay, t});
  auto task = TaskCompute{[](const ContextCompute *) {}};
  return task;
}

TaskCompute AdamW<true>::step(const std::shared_ptr<State> &state,
                              const std::shared_ptr<Tensor> &w,
                              const std::shared_ptr<const ReadOnlyTensor> &dw) {
  state->args.t++;
  // TODO(Jie): necessary check
  const auto m = std::reinterpret_pointer_cast<Tensor>(state->m);
  const auto v = std::reinterpret_pointer_cast<Tensor>(state->v);
  const auto vMax = std::reinterpret_pointer_cast<Tensor>(state->vMax);
  auto task = TaskCompute{
      [args = state->args, m = m, v = v, vMax = vMax, w = w, dw = dw,
       wFuture = w->future(), mFuture = m->future(), vFuture = v->future(),
       dwFuture = dw->future()](const ContextCompute *context) mutable {
        util::FutureGuard wGuard{wFuture};
        util::FutureGuard mGuard{mFuture};
        util::FutureGuard vGuard{vFuture};
        util::FutureGuard dwGuard{dwFuture};
        stepKernel(context->cudaStream, args, w, m, v, vMax, dw);
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
  return task;
}
}  // namespace dllm::optimizer
