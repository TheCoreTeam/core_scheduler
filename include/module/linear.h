#pragma once
#include "compute/linear.h"
#include "module/module.h"
#include "module/pimpl.h"

namespace dllm {
struct ThreadPoolCompute;
}

namespace dllm::module {
struct DLLM_API LinearImpl : Module {
  using Options = compute::Linear::Options;

  explicit LinearImpl(ThreadPoolCompute &tp, const Options &options);

  void forward(ThreadPoolCompute &tp, const std::shared_ptr<Tensor> &output,
               const std::shared_ptr<const ReadOnlyTensor> &input) const;

  void backward(ThreadPoolCompute &tp,
                const std::shared_ptr<Tensor> &grad_input,
                const std::shared_ptr<const ReadOnlyTensor> &grad_output) const;

  void backwardInputOnly(
      ThreadPoolCompute &tp, const std::shared_ptr<Tensor> &grad_input,
      const std::shared_ptr<const ReadOnlyTensor> &grad_output) const;

  std::shared_ptr<compute::Linear::State> state() const;

 private:
  std::weak_ptr<compute::Linear::State> state_;
};

DLLM_MODULE(Linear);
}  // namespace dllm::module
