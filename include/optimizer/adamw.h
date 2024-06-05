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
    struct Args {
      const double lr;
      const double beta1;
      const double beta2;
      const double eps;
      const double weight_decay;
      const bool amsgrad;
      long t;
    } args;

    State(const Tensors &tensors, const Args &args)
        : tensors{tensors}, args{args} {}
  };

  static void init(ThreadPoolCompute &tp, const module::Module &module,
                   double lr = 1e-3, double beta1 = 0.9, double beta2 = 0.999,
                   double eps = 1e-8, double weight_decay = 1e-2,
                   bool amsgrad = false, long t = 0);

  static void step(ThreadPoolCompute &tp, const module::Module &module);

  static TaskCompute init(
      std::shared_ptr<State> &state,
      const std::shared_ptr<const ReadOnlyTensor> &parameter, double lr = 1e-3,
      double beta1 = 0.9, double beta2 = 0.999, double eps = 1e-8,
      double weight_decay = 1e-2, bool amsgrad = false, long t = 0);

  static TaskCompute step(const std::shared_ptr<State> &state,
                          const std::shared_ptr<Tensor> &w,
                          const std::shared_ptr<const ReadOnlyTensor> &dw);
};
}  // namespace dllm::optimizer
