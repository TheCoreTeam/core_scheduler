#pragma once
#include <latch>

#include "threading/task.h"

namespace dllm {
struct Task::Impl {
  enum Type {
    compute,
    memcpy,
    nccl,
  };

  virtual ~Impl() = default;

  Impl(std::vector<Tensor> output, std::vector<ReadOnlyTensor> input,
       const Type type)
      : output_{std::move(output)}, input_{std::move(input)}, type_{type} {}

  [[nodiscard]] auto &latch() const { return latch_; }

  [[nodiscard]] auto &input() const { return input_; }

  [[nodiscard]] auto &output() const { return output_; }

  [[nodiscard]] auto &type() const { return type_; }

  virtual void operator()() const = 0;

  [[nodiscard]] virtual const char *name() const = 0;

 protected:
  std::shared_ptr<std::latch> latch_{new std::latch{1}};

  std::vector<Tensor> output_;

  std::vector<ReadOnlyTensor> input_;

  Type type_;
};
}  // namespace dllm
