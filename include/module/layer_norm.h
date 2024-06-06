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

  explicit LayerNormImpl(ThreadPoolCompute &tp, const Options &options);

  void forward(ThreadPoolCompute &tp, const std::shared_ptr<Tensor> &output,
               const std::shared_ptr<const ReadOnlyTensor> &input) const;

  void backward(ThreadPoolCompute &tp,
                const std::shared_ptr<Tensor> &grad_input,
                const std::shared_ptr<const ReadOnlyTensor> &grad_output) const;

  std::shared_ptr<compute::LayerNorm::State> state() const;

 private:
  std::weak_ptr<compute::LayerNorm::State> state_;
};

DLLM_MODULE(LayerNorm);
}  // namespace dllm::module
