#pragma once
// ReSharper disable once CppUnusedIncludeDirective
#include <c10/util/Exception.h>
#include <torch/ordered_dict.h>

#include "tensor.h"

namespace dllm {
template <typename Key, typename Value>
using OrderedDict = torch::OrderedDict<Key, Value>;
}

namespace dllm::module {
struct OptimizerState {
  virtual ~OptimizerState() = default;
};

struct State {
  struct Increment {
    Tensor &parameter;
    Tensor &gradient;
    std::shared_ptr<OptimizerState> &optimizerState;
  };

  virtual ~State() = default;

  [[nodiscard]] virtual OrderedDict<std::string, Tensor>

  parameters() const {
    return OrderedDict<std::string, Tensor>{};
  }

  [[nodiscard]] virtual OrderedDict<std::string, Increment> increments() {
    return OrderedDict<std::string, Increment>{};
  }
};
}  // namespace dllm::module
