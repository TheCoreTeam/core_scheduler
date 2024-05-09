#pragma once
#include "tensor.h"

namespace dllm::compute::Random {
__inline__ __attribute__((always_inline)) void skipahead(unsigned long long n,
                                                         curandState_t &state,
                                                         std::mutex &mutex) {
  std::lock_guard lockGuard{mutex};
  constexpr auto N = 5;
  auto _skipahead_inplace = [](const unsigned long long x, auto *state) {
    auto __curand_matvec_inplace = [](unsigned int *vector,
                                      unsigned int *matrix) {
      unsigned int result[N] = {0};
      for (int i = 0; i < N; i++) {
        for (int j = 0; j < 32; j++) {
          if (vector[i] & (1 << j)) {
            for (int k = 0; k < N; k++) {
              result[k] ^= matrix[N * (i * 32 + j) + k];
            }
          }
        }
      }
      for (int i = 0; i < N; i++) {
        vector[i] = result[i];
      }
    };
    unsigned long long p = x;
    int matrix_num = 0;
    while (p) {
      for (unsigned int t = 0; t < (p & PRECALC_BLOCK_MASK); t++) {
        __curand_matvec_inplace(state->v,
                                precalc_xorwow_offset_matrix_host[matrix_num]);
      }
      p >>= PRECALC_BLOCK_SIZE;
      matrix_num++;
    }
    state->d += 362437 * (unsigned int)x;
  };
  _skipahead_inplace(n, &state);
}

TaskCompute kaimingNorm(const std::shared_ptr<Tensor2D> &x);

TaskCompute gaussian(const std::shared_ptr<Tensor1D> &x);

TaskCompute uniform(const std::shared_ptr<Tensor1D> &x);
}  // namespace dllm::compute::Random
