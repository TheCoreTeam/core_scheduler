#include "matmul.h"

#include "logger.h"

namespace dllm {
namespace {
cublasStatus_t cublasGemmExOneAlpheZeroBeta(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, const void *A, cudaDataType Atype, int lda,
    const void *B, cudaDataType Btype, int ldb, void *C, cudaDataType Ctype,
    int ldc, cublasComputeType_t computeType, cublasGemmAlgo_t algo) {
  switch (computeType) {
    case CUBLAS_COMPUTE_16F_PEDANTIC:
    case CUBLAS_COMPUTE_16F: {
      nv_half alpha = 1., beta = 0.;
      return cublasGemmEx(handle, transa, transb, m, n, k, &alpha, A, Atype,
                          lda, B, Btype, ldb, &beta, C, Ctype, ldc, computeType,
                          algo);
    }
    case CUBLAS_COMPUTE_32F_FAST_16F:
    case CUBLAS_COMPUTE_32F_FAST_16BF:
    case CUBLAS_COMPUTE_32F_FAST_TF32:
    case CUBLAS_COMPUTE_32F_PEDANTIC:
    case CUBLAS_COMPUTE_32F: {
      float alpha = 1., beta = 0.;
      return cublasGemmEx(handle, transa, transb, m, n, k, &alpha, A, Atype,
                          lda, B, Btype, ldb, &beta, C, Ctype, ldc, computeType,
                          algo);
    }
    case CUBLAS_COMPUTE_64F_PEDANTIC:
    case CUBLAS_COMPUTE_64F: {
      double alpha = 1., beta = 0.;
      return cublasGemmEx(handle, transa, transb, m, n, k, &alpha, A, Atype,
                          lda, B, Btype, ldb, &beta, C, Ctype, ldc, computeType,
                          algo);
    }
    default:
      return CUBLAS_STATUS_INVALID_VALUE;
  }
}
}  // namespace

void RowMajorNNMatmulNoBias(cublasHandle_t handle, const Tensor2D &A,
                            const Tensor2D &B, const Tensor2D &C,
                            cublasComputeType_t computeType) {
  // C (row) = A (row) * B (row) -> C (col) = B (col) @ A (col)
  const auto m = C.layout.shape<1>();
  const auto n = C.layout.shape<0>();
  const auto k = A.layout.shape<1>();
  const auto lda = A.layout.stride<0>();
  const auto ldb = B.layout.stride<0>();
  const auto ldc = C.layout.stride<0>();

  CHECK_CUBLAS(cublasGemmExOneAlpheZeroBeta(
      handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, B.data,
      toCudaDataType(B.dtype), ldb, A.data, toCudaDataType(A.dtype), lda,
      C.data, toCudaDataType(C.dtype), ldc, computeType, CUBLAS_GEMM_DEFAULT));
}

void RowMajorNTMatmulNoBias(cublasHandle_t handle, const Tensor2D &A,
                            const Tensor2D &B, const Tensor2D &C,
                            cublasComputeType_t computeType) {
  // C (row) = A (row) * B^T (row) -> C (col) = B^T (col) @ A (col)
  const auto m = C.layout.shape<1>();
  const auto n = C.layout.shape<0>();
  const auto k = A.layout.shape<1>();
  const auto lda = A.layout.stride<0>();
  const auto ldb = B.layout.stride<0>();
  const auto ldc = C.layout.stride<0>();

  CHECK_CUBLAS(cublasGemmExOneAlpheZeroBeta(
      handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, B.data,
      toCudaDataType(B.dtype), ldb, A.data, toCudaDataType(A.dtype), lda,
      C.data, toCudaDataType(C.dtype), ldc, computeType, CUBLAS_GEMM_DEFAULT));
}

void RowMajorTNMatmulNoBias(cublasHandle_t handle, const Tensor2D &A,
                            const Tensor2D &B, const Tensor2D &C,
                            cublasComputeType_t computeType) {
  // C (row) = A^T (row) * B (row) -> C (col) = B (col) @ A^T (col)
  const auto m = C.layout.shape<1>();
  const auto n = C.layout.shape<0>();
  const auto k = A.layout.shape<0>();
  const auto lda = A.layout.stride<0>();
  const auto ldb = B.layout.stride<0>();
  const auto ldc = C.layout.stride<0>();

  CHECK_CUBLAS(cublasGemmExOneAlpheZeroBeta(
      handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, B.data,
      toCudaDataType(B.dtype), ldb, A.data, toCudaDataType(A.dtype), lda,
      C.data, toCudaDataType(C.dtype), ldc, computeType, CUBLAS_GEMM_DEFAULT));
}
}  // namespace dllm
