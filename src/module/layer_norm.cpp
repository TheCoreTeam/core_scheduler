#include "module/layer_norm.h"

#include "threading/scheduler.h"

namespace dllm::module {
LayerNormImpl::LayerNormImpl(const Scheduler& scheduler,
                             const Options& options) {
  const auto state = compute::LayerNorm::init(scheduler, options);
  register_state("LayerNormState", state);
  state_ = state;
}

Tensor LayerNormImpl::forward(const Scheduler& scheduler,
                              const ReadOnlyTensor& input) const {
  return compute::LayerNorm::forward(scheduler, state(), input);
}

Tensor LayerNormImpl::backward(const Scheduler& scheduler,
                               const ReadOnlyTensor& grad_output) const {
  return compute::LayerNorm::backward(scheduler, state(), grad_output);
}

std::shared_ptr<compute::LayerNorm::State> LayerNormImpl::state() const {
  return state_.lock();
}
}  // namespace dllm::module
