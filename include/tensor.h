#pragma once
#include <ATen/core/TensorBody.h>

#include <memory>

namespace dllm {
using IntArrayRef = at::IntArrayRef;

using IntArray = c10::SmallVector<IntArrayRef::value_type>;

using TensorOptions = at::TensorOptions;

struct ReadOnlyTensor {
  ReadOnlyTensor();

  [[nodiscard]] TensorOptions options() const;

  [[nodiscard]] IntArrayRef sizes() const;

  [[nodiscard]] int64_t size(int64_t dim) const;

  [[nodiscard]] int64_t numel() const;

  void wait() const;

  struct Impl;

  [[nodiscard]] const std::shared_ptr<Impl> &impl() const;

  void reset();

 protected:
  std::shared_ptr<Impl> impl_;
};

struct Tensor : ReadOnlyTensor {
  Tensor() = default;

  void wait() const;
};

TORCH_API std::ostream &print(std::ostream &stream,
                              const ReadOnlyTensor &tensor, int64_t linesize);

static std::ostream &operator<<(std::ostream &out, const ReadOnlyTensor &t) {
  return print(out, t, 80);
}
}  // namespace dllm

namespace at {
bool allclose(const dllm::ReadOnlyTensor &t1, const at::Tensor &t2,
              double rtol = 1e-05, double atol = 1e-08, bool equal_nan = false);
bool allclose(const dllm::ReadOnlyTensor &t1, const dllm::ReadOnlyTensor &t2,
              double rtol = 1e-05, double atol = 1e-08, bool equal_nan = false);
bool allclose(const at::Tensor &t1, const dllm::ReadOnlyTensor &t2,
              double rtol = 1e-05, double atol = 1e-08, bool equal_nan = false);
}  // namespace at
