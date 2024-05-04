#pragma once
#include <nccl.h>

#include "communication.h"

namespace dllm::util {
constexpr ncclRedOp_t toNcclRedOp(communication::Operation operation) {
  switch (operation) {
    case communication::SUM:
      return ncclSum;
    default:
      return static_cast<ncclRedOp_t>(0);
  }
}

inline ncclDataType_t toNcclDataType(Dtype dtype) {
  switch (dtype) {
    case R_64F:
      return ncclFloat64;
    case R_32F:
      return ncclFloat32;
    case R_16F:
      return ncclFloat16;
    case R_16BF:
      return ncclBfloat16;
    default:
      return static_cast<ncclDataType_t>(0);
  }
}
}  // namespace dllm::util
