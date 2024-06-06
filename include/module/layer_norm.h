#pragma once
#include "compute/layer_norm.h"
#include "module/module.h"
#include "module/pimpl.h"

namespace dllm {
struct ThreadPoolCompute;
}

namespace dllm::module {
struct DLLM_API LayerNormImpl : Module {
  explicit LayerNormImpl(ThreadPoolCompute &tp,
                         const compute::LayerNorm::Options &options);

  /// Transforms the `input` tensor by multiplying with the `weight` and
  /// optionally adding the `bias`, if `with_bias` is true in the options.
  void forward(ThreadPoolCompute &tp, const std::shared_ptr<Tensor> &output,
               const std::shared_ptr<const ReadOnlyTensor> &input) const;

  std::shared_ptr<compute::LayerNorm::State> state() const;

 private:
  std::weak_ptr<compute::LayerNorm::State> state_;
};

DLLM_MODULE(LayerNorm);
}  // namespace dllm::module
