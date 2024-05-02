#pragma once
#include <future>

#include "context.h"

namespace dllm {
using Task = std::packaged_task<void(const Context *)>;
using Future = std::future<void>;
}  // namespace dllm
