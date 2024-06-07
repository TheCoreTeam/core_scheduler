#pragma once
#include "compute/linear.h"
#include "module/module.h"
#include "module/pimpl.h"

namespace dllm::module {
struct DLLM_API LinearImpl : Module {
  using Options = compute::Linear::Options;

  explicit LinearImpl(const Scheduler &scheduler, const Options &options);

  void forward(const Scheduler &scheduler,
               const std::shared_ptr<Tensor> &output,
               const std::shared_ptr<const ReadOnlyTensor> &input) const;

  void backward(const Scheduler &scheduler,
                const std::shared_ptr<Tensor> &grad_input,
                const std::shared_ptr<const ReadOnlyTensor> &grad_output) const;

  void backwardParameter(
      const Scheduler &scheduler,
      const std::shared_ptr<const ReadOnlyTensor> &grad_output) const;

  void backwardInput(
      const Scheduler &scheduler, const std::shared_ptr<Tensor> &grad_input,
      const std::shared_ptr<const ReadOnlyTensor> &grad_output) const;

  std::shared_ptr<compute::Linear::State> state() const;

 private:
  std::weak_ptr<compute::Linear::State> state_;
};

DLLM_MODULE(Linear);
}  // namespace dllm::module
