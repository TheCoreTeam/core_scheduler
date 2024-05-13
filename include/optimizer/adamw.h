#pragma once
#include "tensor.h"

namespace dllm::optimizer {
template <bool amsgrad = false>
struct AdamW;

template <>
struct AdamW<false> {
  struct State {
    const std::shared_ptr<Tensor1D> w;
    const std::shared_ptr<Tensor1D> m;
    const std::shared_ptr<Tensor1D> v;
    const double lr;
    const double beta1;
    const double beta2;
    const double eps;
    const double weight_decay;
    mutable long t;
  };

  static TaskCompute step(const std::shared_ptr<const Tensor1D> &dw,
                          const State &state);
};

template <>
struct AdamW<true> {
  struct State {
    const std::shared_ptr<Tensor1D> w;
    const std::shared_ptr<Tensor1D> m;
    const std::shared_ptr<Tensor1D> v;
    const std::shared_ptr<Tensor1D> vMax;
    const double lr;
    const double beta1;
    const double beta2;
    const double eps;
    const double weight_decay;
    mutable long t;
  };

  static TaskCompute step(const std::shared_ptr<const Tensor1D> &dw,
                          const State &state);
};
}  // namespace dllm::optimizer
