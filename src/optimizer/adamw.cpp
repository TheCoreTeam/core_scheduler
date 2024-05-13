#include "optimizer/adamw.h"

#include "util.h"

namespace dllm::optimizer {
template <bool amsgrad>
void stepKernel(cudaStream_t stream, const Tensor1D &dw,
                const typename AdamW<amsgrad>::State &state);

TaskCompute AdamW<false>::step(const std::shared_ptr<const Tensor1D> &dw,
                               const State &state) {
  // TODO(Jie): necessary check
  auto task =
      TaskCompute{[=, wFuture = *state.w->future, mFuture = *state.m->future,
                   vFuture = *state.v->future,
                   dwFuture = *dw->future](const ContextCompute *context) {
        util::waitFutureIfValid(wFuture);
        util::waitFutureIfValid(mFuture);
        util::waitFutureIfValid(vFuture);
        util::waitFutureIfValid(dwFuture);
        stepKernel<false>(context->cudaStream, *dw, state);
        CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
      }};
  state.t++;
  const auto &future = *state.w->future = task.get_future();
  *state.m->future = future;
  *state.v->future = future;
  *dw->future = future;
  return task;
}

TaskCompute AdamW<true>::step(const std::shared_ptr<const Tensor1D> &dw,
                              const State &state) {
  // TODO(Jie): necessary check
  auto task =
      TaskCompute{[=, wFuture = *state.w->future, mFuture = *state.m->future,
                   vFuture = *state.v->future, vMaxFuture = *state.vMax->future,
                   dwFuture = *dw->future](const ContextCompute *context) {
        util::waitFutureIfValid(wFuture);
        util::waitFutureIfValid(mFuture);
        util::waitFutureIfValid(vFuture);
        util::waitFutureIfValid(vMaxFuture);
        util::waitFutureIfValid(dwFuture);
        stepKernel<true>(context->cudaStream, *dw, state);
        CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
      }};
  state.t++;
  const auto &future = *state.w->future = task.get_future();
  *state.m->future = future;
  *state.v->future = future;
  *state.vMax->future = future;
  *dw->future = future;
  return task;
}
}  // namespace dllm::optimizer
