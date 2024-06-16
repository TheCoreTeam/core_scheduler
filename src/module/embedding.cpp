#include "module/embedding.h"

#include "threading/scheduler.h"

namespace dllm::module {
EmbeddingImpl::EmbeddingImpl(const Scheduler& scheduler,
                             const Options& options) {
  const auto state = compute::Embedding::init(scheduler, options);
  register_state("EmbeddingState", state);
  state_ = state;
}

Tensor EmbeddingImpl::forward(const Scheduler& scheduler,
                              const ReadOnlyTensor& input) const {
  return compute::Embedding::forward(scheduler, state(), input);
}

void EmbeddingImpl::backward(const Scheduler& scheduler,
                             const ReadOnlyTensor& grad_output) const {
  compute::Embedding::backward(scheduler, state(), grad_output);
}

std::shared_ptr<compute::Embedding::State> EmbeddingImpl::state() const {
  return state_.lock();
}
}  // namespace dllm::module
