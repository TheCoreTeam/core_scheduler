#include "tensor.h"

#include <ATen/ops/allclose.h>
#include <torch/cuda.h>

#include "nvtx_helper.h"
#include "tensor_impl.h"

namespace dllm {
template <typename T1, typename T2>
static bool allclose_impl(const T1 &t1_, const T2 &t2_, const double rtol,
                          const double atol, const bool equal_nan) {
  DLLM_NVTX_RANGE_FN("dllm::allclose");
  at::Tensor t1, t2;
  torch::cuda::synchronize();
  if constexpr (std::is_same_v<T1, at::Tensor>) {
    t1 = t1_;
  } else {
    t1_.wait();
    t1 = t1_.impl()->tensor();
  }
  if constexpr (std::is_same_v<T2, at::Tensor>) {
    t2 = t2_;
  } else {
    t2_.wait();
    t2 = t2_.impl()->tensor();
  }
  return at::allclose(t1, t2, rtol, atol, equal_nan);
}

void ReadOnlyTensor::wait() const {
  if (utils::future(*this).valid()) {
    utils::future(*this).wait();
  }
}

const std::shared_ptr<ReadOnlyTensor::Impl> &ReadOnlyTensor::impl() const {
  return impl_;
}
void ReadOnlyTensor::reset() { *this = ReadOnlyTensor{}; }

void Tensor::wait() const {
  if (utils::future(*this).valid()) {
    utils::future(*this).wait();
    try {
      utils::future(*this).get();
    } catch (const std::exception &) {
      std::rethrow_exception(std::current_exception());
    }
  }
}
Tensor::Tensor() : ReadOnlyTensor{} {}

int64_t ReadOnlyTensor::numel() const { return impl_->numel(); }

int64_t ReadOnlyTensor::size(const int64_t dim) const {
  return impl_->size(dim);
}

const IntArray &ReadOnlyTensor::sizes() const { return impl_->sizes(); }

IntArray &ReadOnlyTensor::sizes() { return impl_->sizes(); }

TensorOptions &ReadOnlyTensor::options() { return impl_->options(); }

ReadOnlyTensor::ReadOnlyTensor() : impl_{std::make_shared<Impl>()} {}

const TensorOptions &ReadOnlyTensor::options() const {
  return impl_->options();
}

}  // namespace dllm

namespace at {
bool allclose(const dllm::ReadOnlyTensor &t1, const at::Tensor &t2,
              const double rtol, const double atol, const bool equal_nan) {
  return dllm::allclose_impl(t1, t2, rtol, atol, equal_nan);
}
bool allclose(const dllm::ReadOnlyTensor &t1, const dllm::ReadOnlyTensor &t2,
              const double rtol, const double atol, const bool equal_nan) {
  return dllm::allclose_impl(t1, t2, rtol, atol, equal_nan);
}
bool allclose(const at::Tensor &t1, const dllm::ReadOnlyTensor &t2,
              const double rtol, const double atol, const bool equal_nan) {
  return dllm::allclose_impl(t1, t2, rtol, atol, equal_nan);
}
}  // namespace at
