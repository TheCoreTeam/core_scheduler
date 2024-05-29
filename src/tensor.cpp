#include "tensor.h"

#include <ATen/ops/allclose.h>

#include "nvtx_helper.h"
#include "tensor_friend.h"

namespace dllm {
template <typename T1, typename T2>
static bool allclose_impl(const T1 &t1_, const T2 &t2_, const double rtol,
                          const double atol, const bool equal_nan) {
  DLLM_NVTX_RANGE_FN("TensorFriend::allclose");
  at::Tensor t1, t2;
  torch::cuda::synchronize();
  if constexpr (std::is_same_v<T1, at::Tensor>) {
    t1 = t1_;
  } else {
    t1_->wait();
    t1 = DLLM_EXTRACT_TENSOR(t1_);
  }
  if constexpr (std::is_same_v<T2, at::Tensor>) {
    t2 = t2_;
  } else {
    t2_->wait();
    t2 = DLLM_EXTRACT_TENSOR(t2_);
  }
  return at::allclose(t1, t2, rtol, atol, equal_nan);
}
}  // namespace dllm

namespace at {
bool allclose(const std::shared_ptr<const ::dllm::ReadOnlyTensor> &t1,
              const at::Tensor &t2, const double rtol, const double atol,
              const bool equal_nan) {
  return dllm::allclose_impl(t1, t2, rtol, atol, equal_nan);
}
bool allclose(const std::shared_ptr<const ::dllm::ReadOnlyTensor> &t1,
              const std::shared_ptr<const ::dllm::ReadOnlyTensor> &t2,
              const double rtol, const double atol, const bool equal_nan) {
  return dllm::allclose_impl(t1, t2, rtol, atol, equal_nan);
}
bool allclose(const at::Tensor &t1,
              const std::shared_ptr<const ::dllm::ReadOnlyTensor> &t2,
              const double rtol, const double atol, const bool equal_nan) {
  return dllm::allclose_impl(t1, t2, rtol, atol, equal_nan);
}
}  // namespace at