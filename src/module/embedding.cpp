#include "module/embedding.h"

#include "threading/scheduler.h"

namespace dllm::module {
EmbeddingImpl::EmbeddingImpl(const Scheduler& scheduler,
                             const Options& options) {
  std::shared_ptr<compute::Embedding::State> state;
  compute::Embedding::init(scheduler, state, options);
  register_state("EmbeddingState", state);
  state_ = state;
}

void EmbeddingImpl::forward(const Scheduler& scheduler, Tensor& output,
                            const ReadOnlyTensor& input) const {
  compute::Embedding::forward(scheduler, state(), output, input);
}

void EmbeddingImpl::backward(const Scheduler& scheduler,
                             const ReadOnlyTensor& grad_output) const {
  compute::Embedding::backward(scheduler, state(), grad_output);
}

std::shared_ptr<compute::Embedding::State> EmbeddingImpl::state() const {
  return state_.lock();
}
}  // namespace dllm::module
