#pragma once
#include <nccl.h>

#include "dtensor.h"

namespace dllm {
template <int N>
struct DTensor<N, communication::NCCL> : public Tensor<N> {
  int rank, lrank;
  ncclComm_t comm;
};

namespace communication {
constexpr ncclRedOp_t toNcclRedOp(Operation operation) {
  switch (operation) {
    case SUM:
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
}  // namespace communication
}  // namespace dllm
