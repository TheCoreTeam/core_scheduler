#include "optimizer/adamw.h"

#include <c10/cuda/CUDAStream.h>
#include <torch/csrc/autograd/generated/variable_factories.h>

#include "logger.h"
#include "module/module.h"
#include "tensor_impl.h"
#include "threading/scheduler.h"
#include "threading/scheduler_impl.h"
#include "threading/task_impl.h"

namespace dllm::optimizer {
void stepKernel(cudaStream_t stream, const AdamW::State::Options &options,
                const Tensor &w, const Tensor &m, const Tensor &v,
                const ReadOnlyTensor &dw);

void stepKernelAmsgrad(cudaStream_t stream,
                       const AdamW::State::Options &options, const Tensor &w,
                       const Tensor &m, const Tensor &v, const Tensor &vMax,
                       const ReadOnlyTensor &dw);

void AdamW::init(const Scheduler &scheduler, const module::Module &module,
                 const Options &options) {
  for (auto &kvState : module.named_states()) {
    for (auto &kvIncrement : kvState.value()->increments()) {
      std::shared_ptr<State> state;
      init(scheduler, state, kvIncrement->parameter, options);
      kvIncrement->optimizerState = state;
    }
  }
}

void AdamW::step(const Scheduler &scheduler, const module::Module &module) {
  for (auto &kvState : module.named_states()) {
    for (auto &kvIncrement : kvState.value()->increments()) {
      step(scheduler,
           std::dynamic_pointer_cast<State>(kvIncrement->optimizerState),
           kvIncrement->parameter, kvIncrement->gradient);
      kvIncrement->gradient = Tensor{};
    }
  }
}

void AdamW::init(const Scheduler &scheduler, std::shared_ptr<State> &state,
                 const ReadOnlyTensor &parameter, const Options &options) {
  Tensor m;
  Tensor v;
  if (options.amsgrad()) {
    struct Impl : Task::Impl {
      explicit Impl(std::vector<Tensor> output /* m, v, vMax */,
                    std::vector<ReadOnlyTensor> input /* parameter */)
          : Task::Impl{std::move(output), std::move(input), compute} {}
      void operator()() const override {
        output()[0].impl()->tensor() =
            at::zeros_like(input()[0].impl()->tensor());
        output()[1].impl()->tensor() =
            at::zeros_like(input()[0].impl()->tensor());
        output()[2].impl()->tensor() =
            at::zeros_like(input()[0].impl()->tensor());
      }
      [[nodiscard]] const char *name() const override {
        return "dllm::optimizer::AdamW::init";
      }
    };

    Tensor vMax;
    state = std::make_shared<State>(
        State::Tensors{m, v, vMax},
        State::Options{options.lr(), options.beta1(), options.beta2(),
                       options.eps(), options.weight_decay(), options.amsgrad(),
                       options.t()});

    scheduler.impl()->submit(
        Task{std::make_shared<Impl>(Impl{{m, v, vMax}, {parameter}})});
  } else {
    struct Impl : Task::Impl {
      explicit Impl(std::vector<Tensor> output /* m, v */,
                    std::vector<ReadOnlyTensor> input /* parameter */)
          : Task::Impl{std::move(output), std::move(input), compute} {}
      void operator()() const override {
        output()[0].impl()->tensor() =
            at::zeros_like(input()[0].impl()->tensor());
        output()[1].impl()->tensor() =
            at::zeros_like(input()[0].impl()->tensor());
      }
      [[nodiscard]] const char *name() const override {
        return "dllm::optimizer::AdamW::init";
      }
    };

    state = std::make_shared<State>(
        State::Tensors{m, v},
        State::Options{options.lr(), options.beta1(), options.beta2(),
                       options.eps(), options.weight_decay(), options.amsgrad(),
                       options.t()});

    scheduler.impl()->submit(
        Task{std::make_shared<Impl>(Impl{{m, v}, {parameter}})});
  }
}

void AdamW::step(const Scheduler &scheduler,
                 const std::shared_ptr<State> &state, Tensor &w,
                 const ReadOnlyTensor &dw) {
  state->options.t++;
  if (state->options.amsgrad) {
    struct Impl : Task::Impl {
      State::Options options;

      explicit Impl(std::vector<Tensor> output /* w, m, v, vMax */,
                    std::vector<ReadOnlyTensor> input /* w, m, v, vMax, dw */,
                    const State::Options &options)
          : Task::Impl{std::move(output), std::move(input), compute},
            options{options} {}
      void operator()() const override {
        const auto stream = c10::cuda::getCurrentCUDAStream();
        if (input()[4].impl()->tensor().defined()) {
          stepKernelAmsgrad(stream.stream(), options, output()[0], output()[1],
                            output()[2], output()[3], input()[4]);
        } else {
          DLLM_WARN_TRUE(false, "got non-defined gradient, skip the update");
        }
      }
      [[nodiscard]] const char *name() const override {
        return "dllm::optimizer::AdamW::step";
      }
    };

    const auto &m = state->tensors.m;
    const auto &v = state->tensors.v;
    const auto &vMax = state->tensors.vMax;
    scheduler.impl()->submit(Task{std::make_shared<Impl>(
        Impl{{w, m, v, vMax}, {w, m, v, vMax, dw}, state->options})});
  } else {
    struct Impl : Task::Impl {
      State::Options options;

      explicit Impl(std::vector<Tensor> output /* w, m, v */,
                    std::vector<ReadOnlyTensor> input /* w, m, v, dw */,
                    const State::Options &options)
          : Task::Impl{std::move(output), std::move(input), compute},
            options{options} {}
      void operator()() const override {
        const auto stream = c10::cuda::getCurrentCUDAStream();
        if (input()[3].impl()->tensor().defined()) {
          stepKernel(stream.stream(), options, output()[0], output()[1],
                     output()[2], input()[3]);
        } else {
          DLLM_WARN_TRUE(false, "got non-defined gradient, skip the update");
        }
      }
      [[nodiscard]] const char *name() const override {
        return "dllm::optimizer::AdamW::step";
      }
    };

    const auto &m = state->tensors.m;
    const auto &v = state->tensors.v;
    scheduler.impl()->submit(Task{std::make_shared<Impl>(
        Impl{{w, m, v}, {w, m, v, dw}, state->options})});
  }
}
}  // namespace dllm::optimizer
