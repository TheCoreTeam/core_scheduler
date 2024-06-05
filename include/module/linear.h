#pragma once
#include "compute/linear.h"
#include "module/module.h"
#include "module/pimpl.h"

namespace dllm {
struct ThreadPoolCompute;
}

namespace dllm::module {
struct DLLM_API LinearImpl : Module {
  explicit LinearImpl(ThreadPoolCompute &tp, int64_t in_futures,
                      int64_t out_futures, bool bias = true,
                      c10::optional<at::Device> device = {},
                      c10::optional<at::ScalarType> dtype = {});

  /// Transforms the `input` tensor by multiplying with the `weight` and
  /// optionally adding the `bias`, if `with_bias` is true in the options.
  void forward(ThreadPoolCompute &tp, const std::shared_ptr<Tensor> &output,
               const std::shared_ptr<const ReadOnlyTensor> &input) const;

  std::shared_ptr<compute::Linear::State> state() const;

 private:
  std::weak_ptr<compute::Linear::State> state_;
};

DLLM_MODULE(Linear);
}  // namespace dllm::module
