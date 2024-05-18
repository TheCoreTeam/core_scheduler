#pragma once
#include "tensor.h"

namespace dllm::optimizer {
template <bool amsgrad = false>
struct AdamW;

template <>
struct AdamW<false> {
  AdamW() = delete;

  struct State {
    using Layout = Tensor1D::Layout;
    std::shared_ptr<Tensor1D> m;
    std::shared_ptr<Tensor1D> v;
    const double lr{1e-3};
    const double beta1{0.9};
    const double beta2{0.999};
    const double eps{1e-8};
    const double weight_decay{1e-2};
    mutable long t{0};
  };

  template <Dtype dtype, DeviceType deviceType>
  static TaskCompute init(std::shared_ptr<State> &state,
                          const State::Layout &layout, double lr = 1e-3,
                          double beta1 = 0.9, double beta2 = 0.999,
                          double eps = 1e-8, double weight_decay = 1e-2,
                          long t = 0);

  static TaskCompute step(const std::shared_ptr<State> &state,
                          const std::shared_ptr<Tensor1D> &w,
                          const std::shared_ptr<const Tensor1D> &dw);
};

template <>
struct AdamW<true> {
  AdamW() = delete;

  struct State {
    using Layout = Tensor1D::Layout;
    std::shared_ptr<Tensor1D> m;
    std::shared_ptr<Tensor1D> v;
    std::shared_ptr<Tensor1D> vMax;
    const double lr;
    const double beta1;
    const double beta2;
    const double eps;
    const double weight_decay;
    mutable long t;
  };

  template <Dtype dtype, DeviceType deviceType>
  static TaskCompute init(std::shared_ptr<State> &state,
                          const State::Layout &layout, double lr = 1e-3,
                          double beta1 = 0.9, double beta2 = 0.999,
                          double eps = 1e-8, double weight_decay = 1e-2,
                          long t = 0);

  static TaskCompute step(const std::shared_ptr<State> &state,
                          const std::shared_ptr<Tensor1D> &w,
                          const std::shared_ptr<const Tensor1D> &dw);
};
}  // namespace dllm::optimizer
