#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace c10 {
using Half = nv_half;
using BFloat16 = nv_bfloat16;

#if defined(__HIPCPU__)
#define C10_WARP_SIZE warpSize  // = 64 or 32 (Defined in hip_runtime.h)
#else
#define C10_WARP_SIZE 32
#endif

#define TORCH_INTERNAL_ASSERT(statement)                               \
  do {                                                                 \
    if (!(statement)) {                                                \
      SPDLOG_LOGGER_CRITICAL(&::dllm::logger(),                        \
                             "Assert (" #statement ") = true Failed"); \
    }                                                                  \
  } while (false)
}  // namespace c10

namespace at::cuda {
cudaStream_t getCurrentCUDAStream() { return nullptr; }
}  // namespace at::cuda
