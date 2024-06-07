#pragma once
#include <type_traits>

#include "arg.h"
#include "module/state.h"
#include "tensor.h"
#include "threading/scheduler.h"

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

  struct Options {
    DLLM_ARG(double, lr) = 1e-3;
    DLLM_ARG(double, beta1) = 0.9;
    DLLM_ARG(double, beta2) = 0.999;
    DLLM_ARG(double, eps) = 1e-8;
    DLLM_ARG(double, weight_decay) = 1e-2;
    DLLM_ARG(bool, amsgrad) = false;
    DLLM_ARG(long, t) = 0;
  };

  static void init(const Scheduler &scheduler, const module::Module &module,
                   const Options &options);

  template <typename Module, typename = std::enable_if_t<
                                 !std::is_base_of_v<module::Module, Module>>>
  static void init(const Scheduler &scheduler, const Module &module,
                   const Options &options) {
    init(scheduler, *module, options);
  }

  static void step(const Scheduler &scheduler, const module::Module &module);

  template <typename Module, typename = std::enable_if_t<
                                 !std::is_base_of_v<module::Module, Module>>>
  static void step(const Scheduler &scheduler, const Module &module) {
    step(scheduler, *module);
  }

  static void init(const Scheduler &scheduler, std::shared_ptr<State> &state,
                   const std::shared_ptr<const ReadOnlyTensor> &parameter,
                   const Options &options);

  static void step(const Scheduler &scheduler,
                   const std::shared_ptr<State> &state,
                   const std::shared_ptr<Tensor> &w,
                   const std::shared_ptr<const ReadOnlyTensor> &dw);
};
}  // namespace dllm::optimizer
