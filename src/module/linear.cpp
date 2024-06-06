#include "module/linear.h"

#include "threading/thread_pool_compute.h"

namespace dllm::module {
LinearImpl::LinearImpl(ThreadPoolCompute& tp, const Options& options) {
  std::shared_ptr<compute::Linear::State> state;
  DLLM_SUBMIT_TASK(tp, compute::Linear::init(state, options));
  register_state("LinearState", state);
  state_ = state;
}

void LinearImpl::forward(
    ThreadPoolCompute& tp, const std::shared_ptr<Tensor>& output,
    const std::shared_ptr<const ReadOnlyTensor>& input) const {
  tp.submit(compute::Linear::forward(state(), output, input));
}

void LinearImpl::backward(
    ThreadPoolCompute& tp, const std::shared_ptr<Tensor>& grad_input,
    const std::shared_ptr<const ReadOnlyTensor>& grad_output) const {
  tp.submit(compute::Linear::backwardInput(state(), grad_input, grad_output));
  tp.submit(compute::Linear::backwardParameter(state(), grad_output));
}

void LinearImpl::backwardInputOnly(
    ThreadPoolCompute& tp, const std::shared_ptr<Tensor>& grad_input,
    const std::shared_ptr<const ReadOnlyTensor>& grad_output) const {
  tp.submit(compute::Linear::backwardInput(state(), grad_input, grad_output));
}

std::shared_ptr<compute::Linear::State> LinearImpl::state() const {
  return state_.lock();
}
}  // namespace dllm::module
