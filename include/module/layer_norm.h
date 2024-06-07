#pragma once
#include "compute/layer_norm.h"
#include "module/module.h"
#include "module/pimpl.h"

namespace dllm {
struct ThreadPoolCompute;
}

namespace dllm::module {
struct DLLM_API LayerNormImpl : Module {
  using Options = compute::LayerNorm::Options;

  explicit LayerNormImpl(const Scheduler &scheduler, const Options &options);

  void forward(const Scheduler &scheduler, Tensor &output,
               const ReadOnlyTensor &input) const;

  void backward(const Scheduler &scheduler, Tensor &grad_input,
                const ReadOnlyTensor &grad_output) const;

  std::shared_ptr<compute::LayerNorm::State> state() const;

 private:
  std::weak_ptr<compute::LayerNorm::State> state_;
};

DLLM_MODULE(LayerNorm);
}  // namespace dllm::module
