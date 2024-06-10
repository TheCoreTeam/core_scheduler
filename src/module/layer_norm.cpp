#include "module/layer_norm.h"

#include "threading/scheduler.h"

namespace dllm::module {
LayerNormImpl::LayerNormImpl(const Scheduler& scheduler,
                             const Options& options) {
  std::shared_ptr<compute::LayerNorm::State> state;
  compute::LayerNorm::init(scheduler, state, options);
  register_state("LayerNormState", state);
  state_ = state;
}

void LayerNormImpl::forward(const Scheduler& scheduler, Tensor& output,
                            const ReadOnlyTensor& input) const {
  compute::LayerNorm::forward(scheduler, state(), output, input);
}

void LayerNormImpl::backward(const Scheduler& scheduler, Tensor& grad_input,
                             const ReadOnlyTensor& grad_output) const {
  compute::LayerNorm::backward(scheduler, state(), grad_input, grad_output);
}

std::shared_ptr<compute::LayerNorm::State> LayerNormImpl::state() const {
  return state_.lock();
}
}  // namespace dllm::module
