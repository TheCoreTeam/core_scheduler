#pragma once
#include "tensor.h"
#include "threading/task_compute.h"

namespace dllm::optimizer {
template <bool amsgrad = false>
struct AdamW;

template <>
struct AdamW<false> {
  AdamW() = delete;

  struct State {
    std::shared_ptr<const ReadOnlyTensor> m;
    std::shared_ptr<const ReadOnlyTensor> v;
    struct Args {
      const double lr;
      const double beta1;
      const double beta2;
      const double eps;
      const double weight_decay;
      mutable long t;
    } args;
  };

  static TaskCompute init(std::shared_ptr<State> &state,
                          const IntArrayRef &size, double lr = 1e-3,
                          double beta1 = 0.9, double beta2 = 0.999,
                          double eps = 1e-8, double weight_decay = 1e-2,
                          long t = 0);

  static TaskCompute step(const std::shared_ptr<State> &state,
                          const std::shared_ptr<Tensor> &w,
                          const std::shared_ptr<const ReadOnlyTensor> &dw);
};

template <>
struct AdamW<true> {
  AdamW() = delete;

  struct State {
    std::shared_ptr<Tensor> m;
    std::shared_ptr<Tensor> v;
    std::shared_ptr<Tensor> vMax;
    struct Args {
      const double lr;
      const double beta1;
      const double beta2;
      const double eps;
      const double weight_decay;
      mutable long t;
    } args;
  };

  static TaskCompute init(std::shared_ptr<State> &state,
                          const IntArrayRef &size, double lr = 1e-3,
                          double beta1 = 0.9, double beta2 = 0.999,
                          double eps = 1e-8, double weight_decay = 1e-2,
                          long t = 0);

  static TaskCompute step(const std::shared_ptr<State> &state,
                          const std::shared_ptr<Tensor> &w,
                          const std::shared_ptr<const ReadOnlyTensor> &dw);
};
}  // namespace dllm::optimizer
