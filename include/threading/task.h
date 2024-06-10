#pragma once
#include <memory>
#include <vector>

#include "tensor.h"

namespace dllm {
struct Task {
  struct Impl;

  explicit Task(std::shared_ptr<Impl> impl);

  [[nodiscard]] const std::vector<ReadOnlyTensor> &input() const;

  [[nodiscard]] const std::vector<Tensor> &output() const;

  [[nodiscard]] const char *name() const;

  void operator()() const;

  [[nodiscard]] const std::shared_ptr<Impl> &impl() const;

  void reset();

  [[nodiscard]] bool valid() const;

 private:
  std::shared_ptr<Impl> impl_;
};
}  // namespace dllm
