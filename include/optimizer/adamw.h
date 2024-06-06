#pragma once
#include "module/state.h"
#include "tensor.h"
#include "threading/task_compute.h"
namespace dllm::module {
struct Module;
}

namespace dllm {
struct ThreadPoolCompute;
}

namespace dllm::optimizer {
struct AdamW {
  AdamW() = delete;

  struct State final : module::OptimizerState {
    struct Tensors {
      std::shared_ptr<Tensor> m;
      std::shared_ptr<Tensor> v;
      std::shared_ptr<Tensor> vMax = nullptr;
    } tensors;
    struct Options {
      const double lr = 1e-3;
      const double beta1 = 0.9;
      const double beta2 = 0.999;
      const double eps = 1e-8;
      const double weight_decay = 1e-2;
      const bool amsgrad = false;
      long t = 0;
    } options;

    State(const Tensors &tensors, const Options &options)
        : tensors{tensors}, options{options} {}
  };

  using Options = State::Options;

  static void init(ThreadPoolCompute &tp, const module::Module &module,
                   const Options &options);

  static void step(ThreadPoolCompute &tp, const module::Module &module);

  static TaskCompute init(
      std::shared_ptr<State> &state,
      const std::shared_ptr<const ReadOnlyTensor> &parameter,
      const Options &options);

  static TaskCompute step(const std::shared_ptr<State> &state,
                          const std::shared_ptr<Tensor> &w,
                          const std::shared_ptr<const ReadOnlyTensor> &dw);
};
}  // namespace dllm::optimizer
