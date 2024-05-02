#pragma once
#include "dtensor.h"
#include "task.h"

namespace dllm::communication {
template <Backend backend>
struct AllReduce {
  static Task run(const std::shared_ptr<const DTensor1D<backend>> &tensorSend,
                  const std::shared_ptr<DTensor1D<backend>> &tensorReceive,
                  Operation operation);

  static Task runInplace(const std::shared_ptr<DTensor1D<backend>> &tensor,
                         Operation operation);
};
}  // namespace dllm::communication
