#pragma once
#include "data/dataset.h"

namespace dllm::data {
struct Dataset::Impl {
  virtual ~Impl() = default;
};
}  // namespace dllm::data
