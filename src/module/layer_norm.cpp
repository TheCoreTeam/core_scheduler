#include "module/layer_norm.h"

#include "threading/thread_pool_compute.h"

namespace dllm::module {
LayerNormImpl::LayerNormImpl(ThreadPoolCompute& tp,
                             const compute::LayerNorm::Options& options) {
  std::shared_ptr<compute::LayerNorm::State> state;
  DLLM_SUBMIT_TASK(tp, compute::LayerNorm::init(state, options));
  register_state("LayerNormState", state);
  state_ = state;
}

void LayerNormImpl::forward(
    ThreadPoolCompute& tp, const std::shared_ptr<Tensor>& output,
    const std::shared_ptr<const ReadOnlyTensor>& input) const {
  tp.submit(compute::LayerNorm::forward(state(), output, input));
}

std::shared_ptr<compute::LayerNorm::State> LayerNormImpl::state() const {
  return state_.lock();
}
}  // namespace dllm::module
