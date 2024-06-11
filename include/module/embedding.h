#pragma once
#include "compute/embedding.h"
#include "module/module.h"
#include "module/pimpl.h"

namespace dllm::module {
struct DLLM_API EmbeddingImpl : Module {
  using Options = compute::Embedding::Options;

  explicit EmbeddingImpl(const Scheduler &scheduler, const Options &options);

  void forward(const Scheduler &scheduler, Tensor &output,
               const ReadOnlyTensor &input) const;

  void backward(const Scheduler &scheduler,
                const ReadOnlyTensor &grad_output) const;

  std::shared_ptr<compute::Embedding::State> state() const;

 private:
  std::weak_ptr<compute::Embedding::State> state_;
};

DLLM_MODULE(Embedding);
}  // namespace dllm::module
