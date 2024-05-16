#include "optimizer/adamw.h"

#include "memory/malloc_from_mem_internal.h"
#include "util.h"

namespace dllm::optimizer {
template <bool amsgrad>
void stepKernel(cudaStream_t stream,
                const typename AdamW<amsgrad>::State &state, Tensor1D &w,
                const Tensor1D &dw);

template <Dtype dtype, DeviceType deviceType>
TaskCompute AdamW<false>::init(std::shared_ptr<State> &state,
                               const State::Layout &layout, const double lr,
                               const double beta1, const double beta2,
                               const double eps, const double weight_decay,
                               const long t) {
  if (state != nullptr) {
    SPDLOG_LOGGER_CRITICAL(&logger(), "The state must be a null ptr");
  }
  state = std::make_shared<State>(
      std::make_shared<Tensor1D>(nullptr, layout, dtype, deviceType),
      std::make_shared<Tensor1D>(nullptr, layout, dtype, deviceType), lr, beta1,
      beta2, eps, weight_decay, t);
  const auto size = cute::size(layout);
  auto task = TaskCompute{
      [size = size, state = state](const ContextCompute *context) mutable {
        state->m->resetData(memory::mallocFromMemPool(size, dtype, context));
        state->v->resetData(memory::mallocFromMemPool(size, dtype, context));
        CHECK_CUDART(cudaMemsetAsync(state->m->data(), 0, toByte(dtype) * size,
                                     context->cudaStream));
        CHECK_CUDART(cudaMemsetAsync(state->v->data(), 0, toByte(dtype) * size,
                                     context->cudaStream));
        CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
        state.reset();
      }};
  const auto &future = *state->m->future = task.get_future();
  *state->v->future = future;
  return task;
}

template TaskCompute AdamW<false>::init<R_32F, CUDA>(
    std::shared_ptr<State> &state, const State::Layout &layout, double lr,
    double beta1, double beta2, double eps, double weight_decay, long t);

TaskCompute AdamW<false>::step(const std::shared_ptr<State> &state,
                               const std::shared_ptr<Tensor1D> &w,
                               const std::shared_ptr<const Tensor1D> &dw) {
  state->t++;
  // TODO(Jie): necessary check
  auto task = TaskCompute{
      [state = state, w = w, dw = dw, wFuture = *w->future,
       mFuture = *state->m->future, vFuture = *state->v->future,
       dwFuture = *dw->future](const ContextCompute *context) mutable {
        util::FutureGuard wGuard{wFuture};
        util::FutureGuard mGuard{mFuture};
        util::FutureGuard vGuard{vFuture};
        util::FutureGuard dwGuard{dwFuture};
        stepKernel<false>(context->cudaStream, *state, *w, *dw);
        CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
        state.reset();
        w.reset();
        dw.reset();
      }};
  const auto &future = *w->future = task.get_future();
  *state->m->future = future;
  *state->v->future = future;
  *dw->future = future;
  return task;
}

template <Dtype dtype, DeviceType deviceType>
TaskCompute AdamW<true>::init(std::shared_ptr<State> &state,
                              const State::Layout &layout, const double lr,
                              const double beta1, const double beta2,
                              const double eps, const double weight_decay,
                              const long t) {
  if (state != nullptr) {
    SPDLOG_LOGGER_CRITICAL(&logger(), "The state must be a null ptr");
  }
  auto task = TaskCompute{
      [layout = layout, state = state](const ContextCompute *context) mutable {
        const auto size = cute::size(layout);
        state->m->resetData(memory::mallocFromMemPool(size, dtype, context));
        state->v->resetData(memory::mallocFromMemPool(size, dtype, context));
        state->vMax->resetData(memory::mallocFromMemPool(size, dtype, context));
        CHECK_CUDART(cudaMemsetAsync(state->m->data(), 0,
                                     toByte(dtype) * cute::size(layout),
                                     context->cudaStream));
        CHECK_CUDART(cudaMemsetAsync(state->v->data(), 0,
                                     toByte(dtype) * cute::size(layout),
                                     context->cudaStream));
        CHECK_CUDART(cudaMemsetAsync(state->vMax->data(), 0,
                                     toByte(dtype) * cute::size(layout),
                                     context->cudaStream));
        CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
        state.reset();
      }};
  state = std::make_shared<State>(
      std::make_shared<Tensor1D>(nullptr, layout, dtype, deviceType),
      std::make_shared<Tensor1D>(nullptr, layout, dtype, deviceType),
      std::make_shared<Tensor1D>(nullptr, layout, dtype, deviceType), lr, beta1,
      beta2, eps, weight_decay, t);
  const auto &future = *state->m->future = task.get_future();
  *state->v->future = future;
  *state->vMax->future = future;
  return task;
}

TaskCompute AdamW<true>::step(const std::shared_ptr<State> &state,
                              const std::shared_ptr<Tensor1D> &w,
                              const std::shared_ptr<const Tensor1D> &dw) {
  state->t++;
  // TODO(Jie): necessary check
  auto task =
      TaskCompute{[state = state, w = w, dw = dw, wFuture = *w->future,
                   mFuture = *state->m->future, vFuture = *state->v->future,
                   vMaxFuture = *state->vMax->future, dwFuture = *dw->future](
                      const ContextCompute *context) mutable {
        util::FutureGuard{wFuture};
        util::FutureGuard{mFuture};
        util::FutureGuard{vMaxFuture};
        util::FutureGuard{dwFuture};
        stepKernel<true>(context->cudaStream, *state, *w, *dw);
        CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
        state.reset();
        w.reset();
        dw.reset();
      }};
  const auto &future = *w->future = task.get_future();
  *state->m->future = future;
  *state->v->future = future;
  *state->vMax->future = future;
  *dw->future = future;
  return task;
}
}  // namespace dllm::optimizer