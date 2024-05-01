#include "fc.h"
#include <Eigen/Dense>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

TEST(FcTests, TestForward) {
  cublasHandle_t handle;
  cublasCreate_v2(&handle);

  using T = float;
  const int m = 1024, n = 2048, k = 512, s = 2;
  auto shapeX = cute::make_shape(m, s, k);
  auto layoutX = cute::make_layout(shapeX, cute::GenRowMajor{});
  auto shapeW = cute::make_shape(n, k);
  auto layoutW = cute::make_layout(shapeW, cute::GenRowMajor{});
  auto shapeY = cute::make_shape(m, s, n);
  auto layoutY = cute::make_layout(shapeY, cute::GenRowMajor{});

  void *ptrX, *ptrW, *ptrY;
  cudaMalloc(&ptrX, sizeof(T) * size(layoutX));
  cudaMalloc(&ptrW, sizeof(T) * size(layoutW));
  cudaMalloc(&ptrY, sizeof(T) * size(layoutY));

  dllm::Tensor3D tensorX{ptrX, layoutX, dllm::toDtype<T>(), dllm::CUDA};
  dllm::Tensor2D tensorW{ptrW, layoutW, dllm::toDtype<T>(), dllm::CUDA};
  dllm::Tensor3D tensorY{ptrY, layoutY, dllm::toDtype<T>(), dllm::CUDA};

  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> hostX(m * s,
                                                                          k),
      hostW(n, k), hostY(m * s, n), refY;

  hostX.setRandom();
  hostW.setRandom();

  refY = hostX * hostW.transpose();

  cudaMemcpy(ptrX, hostX.data(), sizeof(T) * size(layoutX),
             cudaMemcpyHostToDevice);
  cudaMemcpy(ptrW, hostW.data(), sizeof(T) * size(layoutW),
             cudaMemcpyHostToDevice);

  dllm::FcNoBias::forward(handle, tensorY, tensorX, tensorW,
                          CUBLAS_COMPUTE_32F);

  cudaMemcpy(hostY.data(), ptrY, sizeof(T) * size(layoutY),
             cudaMemcpyDeviceToHost);

  // 检查结果
  for (int col = 0; col < refY.cols(); ++col) {
    for (int row = 0; row < refY.rows(); ++row) {
      ASSERT_NEAR(hostY(row, col), refY(row, col), 1e-4);
    }
  }

  cudaFree(ptrY);
  cudaFree(ptrW);
  cudaFree(ptrX);
  cublasDestroy_v2(handle);
}
