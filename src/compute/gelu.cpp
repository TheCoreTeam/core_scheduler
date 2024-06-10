#include "compute/gelu.h"

#include <ATen/ops/gelu.h>
#include <ATen/ops/gelu_backward.h>

#include "tensor_impl.h"
#include "threading/scheduler_impl.h"
#include "threading/task_impl.h"

namespace dllm::compute {
void GeLU::init(const Scheduler &scheduler, std::shared_ptr<State> &state) {
  state = std::make_shared<State>();
}

void GeLU::forward(const Scheduler &scheduler,
                   const std::shared_ptr<State> &state, Tensor &output,
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

  Tensor output_{};
  state->backward.input = input;
  // size
  output_.sizes() = input.sizes();
  scheduler.impl()->submit(
      Task{std::make_shared<Impl>(Impl{{output_}, {input}})});
  output = output_;
}

void GeLU::backward(const Scheduler &scheduler,
                    const std::shared_ptr<State> &state, Tensor &grad_input,
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

  Tensor grad_input_{};
  // size
  grad_input_.sizes() = grad_output.sizes();
  // decrease counter
  scheduler.impl()->submit(Task{std::make_shared<Impl>(
      Impl{{grad_input_}, {grad_output, state->backward.input}})});
  state->backward.input.reset();
  grad_input = grad_input_;
}
}  // namespace dllm::compute
