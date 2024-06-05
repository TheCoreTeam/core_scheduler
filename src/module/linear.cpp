#include "module/linear.h"

#include "threading/thread_pool_compute.h"

namespace dllm::module {
LinearImpl::LinearImpl(ThreadPoolCompute& tp, const int64_t in_futures,
                       const int64_t out_futures, const bool bias,
                       const c10::optional<at::Device> device,
                       const c10::optional<at::ScalarType> dtype) {
  std::shared_ptr<compute::Linear::State> state;
  DLLM_SUBMIT_TASK(tp, compute::Linear::init(state, in_futures, out_futures,
                                             bias, device, dtype));
  register_state("LinearState", state);
  state_ = state;
}

void LinearImpl::forward(
    ThreadPoolCompute& tp, const std::shared_ptr<Tensor>& output,
    const std::shared_ptr<const ReadOnlyTensor>& input) const {
  tp.submit(compute::Linear::forward(state(), output, input));
}

std::shared_ptr<compute::Linear::State> LinearImpl::state() const {
  return state_.lock();
}
}  // namespace dllm::module
