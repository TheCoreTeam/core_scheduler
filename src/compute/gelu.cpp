#include "compute/gelu.h"

#include <ATen/ops/gelu.h>
#include <ATen/ops/gelu_backward.h>

#include "tensor_impl.h"
#include "threading/scheduler_impl.h"
#include "threading/task_impl.h"

namespace dllm::compute {
std::shared_ptr<GeLU::State> GeLU::init(const Scheduler &scheduler) {
  return std::make_shared<State>();
}

Tensor GeLU::forward(const Scheduler &scheduler,
                     const std::shared_ptr<State> &state,
                     const ReadOnlyTensor &input) {
  struct Impl : Task::Impl {
    Impl(std::vector<Tensor> output /* output */,
         std::vector<ReadOnlyTensor> input /* input */)
        : Task::Impl{std::move(output), std::move(input), compute} {}
    void operator()() const override {
      output()[0].impl()->tensor() = at::gelu(input()[0].impl()->tensor());
    }
    [[nodiscard]] const char *name() const override {
      return "dllm::compute::GeLU::forward";
    }
  };

  Tensor output{};
  state->backward.input = input;
  scheduler.impl()->submit(
      Task{std::make_shared<Impl>(Impl{{output}, {input}})});
  return output;
}

Tensor GeLU::backward(const Scheduler &scheduler,
                      const std::shared_ptr<State> &state,
                      const ReadOnlyTensor &grad_output) {
  struct Impl : Task::Impl {
    Impl(std::vector<Tensor> output /* grad_input */,
         std::vector<ReadOnlyTensor> input /* grad_ouput, input */)
        : Task::Impl{std::move(output), std::move(input), compute} {}
    void operator()() const override {
      output()[0].impl()->tensor() = at::gelu_backward(
          input()[0].impl()->tensor(), input()[1].impl()->tensor());
    }
    [[nodiscard]] const char *name() const override {
      return "dllm::compute::GeLU::backward";
    }
  };

  Tensor grad_input{};
  // decrease counter
  scheduler.impl()->submit(Task{std::make_shared<Impl>(
      Impl{{grad_input}, {grad_output, state->backward.input}})});
  state->backward.input.reset();
  return grad_input;
}
}  // namespace dllm::compute
