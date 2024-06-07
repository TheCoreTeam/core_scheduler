#include <cuda_runtime.h>
#include <mpi.h>
#include <nccl.h>

#include <Eigen/Dense>
#include <iostream>
#include <vector>

#include "communication/nccl/all_reduce.h"
#include "compute/fc.h"
#include "compute/mse.h"
#include "compute/random.h"
#include "logger.h"
#include "optimizer/sgd.h"
#include "threading/thread_pool_compute.h"
#include "threading/thread_stream_nccl.h"
#include "util.h"

int main(int argc, char **argv) {
  // dtype: FP32
  // Because we use FcNoBias, we can only fit to a linear function without bias
  // f(x) = 2x
  // 1000 samples, batch size = 100, no shuffle
  // x (N x 1) -> FC1 -> f1 (N x 8) -> FC2 -> f2 (N x 1) -> MSE
  // FC1.w (8 x 1)
  // FC2.w (1 x 8)
  CHECK_MPI(MPI_Init(&argc, &argv));
  int worldSize;
  int mpiRank;
  CHECK_MPI(MPI_Comm_size(MPI_COMM_WORLD, &worldSize));
  CHECK_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank));
  ncclUniqueId id;
  if (mpiRank == 0) {
    CHECK_NCCL(ncclGetUniqueId(&id));
  }
  CHECK_MPI(MPI_Bcast(&id, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD));
  dllm::ThreadStreamNccl threadStreamNccl{id, worldSize, mpiRank, mpiRank};

  int deviceRank;
  using T = float;
  const double lr = 1e-3;
  auto f = [](auto x) {
    using Scalar = decltype(x)::Scalar;
    return static_cast<Scalar>(2) * x.array();
  };
  const dllm::TensorIndexType inputDim = 1;
  const dllm::TensorIndexType batchSize = 100;
  const dllm::TensorIndexType sampleNum = 1000 / worldSize;
  using Input = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
  using Output = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
  std::vector<std::tuple<Input, Output>> data;
  data.reserve(sampleNum / batchSize);
  for (int i = 0; i < sampleNum / batchSize; ++i) {
    Input x(batchSize, inputDim);
    x.setRandom();
    Output y = f(x);
    data.emplace_back(std::move(x), std::move(y));
  }

  std::vector<std::tuple<T *, T *>> dataDevice;
  dataDevice.reserve(data.size());
  for (auto &[x, y] : data) {
    T *input, *output;
    CHECK_CUDART(
        cudaMalloc(reinterpret_cast<void **>(&input), sizeof(T) * x.size()));
    CHECK_CUDART(cudaMemcpy(input, x.data(), sizeof(T) * x.size(),
                            cudaMemcpyHostToDevice));
    CHECK_CUDART(
        cudaMalloc(reinterpret_cast<void **>(&output), sizeof(T) * y.size()));
    CHECK_CUDART(cudaMemcpy(output, y.data(), sizeof(T) * y.size(),
                            cudaMemcpyHostToDevice));
    dataDevice.emplace_back(input, output);
  }

  using SharedTensor2D = std::shared_ptr<dllm::Tensor2D>;
  std::vector<std::tuple<SharedTensor2D, SharedTensor2D>> tensor;
  tensor.reserve(dataDevice.size());
  for (auto [ptrX, ptrY] : dataDevice) {
    auto shapeX = cute::make_shape(batchSize, inputDim);
    auto layoutX = cute::make_layout(shapeX, cute::GenRowMajor{});
    auto tensorX = std::make_shared<dllm::Tensor2D>(
        ptrX, layoutX, dllm::toDtype<T>(), dllm::CUDA);

    auto shapeY = cute::make_shape(batchSize, inputDim);
    auto layoutY = cute::make_layout(shapeY, cute::GenRowMajor{});
    auto tensorY = std::make_shared<dllm::Tensor2D>(
        ptrY, layoutY, dllm::toDtype<T>(), dllm::CUDA);

    tensor.emplace_back(std::move(tensorX), std::move(tensorY));
  }

  const dllm::TensorIndexType outDimW1 = 8;
  T *ptrW1;
  auto shapeW1 = cute::make_shape(outDimW1, inputDim);
  CHECK_CUDART(cudaMalloc(reinterpret_cast<void **>(&ptrW1),
                          sizeof(T) * cute::size(shapeW1)));
  auto layoutW1 = cute::make_layout(shapeW1, cute::GenRowMajor{});
  auto tensorW1 = std::make_shared<dllm::Tensor2D>(
      ptrW1, layoutW1, dllm::toDtype<T>(), dllm::CUDA);

  T *ptrDW1;
  auto shapeDW1 = cute::make_shape(outDimW1, inputDim);
  CHECK_CUDART(cudaMalloc(reinterpret_cast<void **>(&ptrDW1),
                          sizeof(T) * cute::size(shapeDW1)));
  auto layoutDW1 = cute::make_layout(shapeDW1, cute::GenRowMajor{});
  auto tensorDW1 = std::make_shared<dllm::Tensor2D>(
      ptrDW1, layoutDW1, dllm::toDtype<T>(), dllm::CUDA);

  T *ptrW1Out;
  auto shapeW1Out = cute::make_shape(batchSize, outDimW1);
  CHECK_CUDART(cudaMalloc(reinterpret_cast<void **>(&ptrW1Out),
                          sizeof(T) * cute::size(shapeW1Out)));
  auto layoutW1Out = cute::make_layout(shapeW1Out, cute::GenRowMajor{});
  auto tensorW1Out = std::make_shared<dllm::Tensor2D>(
      ptrW1Out, layoutW1Out, dllm::toDtype<T>(), dllm::CUDA);

  T *ptrDW1Out;
  auto shapeDW1Out = cute::make_shape(batchSize, outDimW1);
  CHECK_CUDART(cudaMalloc(reinterpret_cast<void **>(&ptrDW1Out),
                          sizeof(T) * cute::size(shapeDW1Out)));
  auto layoutDW1Out = cute::make_layout(shapeDW1Out, cute::GenRowMajor{});
  auto tensorDW1Out = std::make_shared<dllm::Tensor2D>(
      ptrDW1Out, layoutDW1Out, dllm::toDtype<T>(), dllm::CUDA);

  T *ptrW2;
  auto shapeW2 =
      cute::make_shape(static_cast<dllm::TensorIndexType>(1), outDimW1);
  CHECK_CUDART(cudaMalloc(reinterpret_cast<void **>(&ptrW2),
                          sizeof(T) * cute::size(shapeW2)));
  auto layoutW2 = cute::make_layout(shapeW2, cute::GenRowMajor{});
  auto tensorW2 = std::make_shared<dllm::Tensor2D>(
      ptrW2, layoutW2, dllm::toDtype<T>(), dllm::CUDA);

  T *ptrDW2;
  auto shapeDW2 =
      cute::make_shape(static_cast<dllm::TensorIndexType>(1), outDimW1);
  CHECK_CUDART(cudaMalloc(reinterpret_cast<void **>(&ptrDW2),
                          sizeof(T) * cute::size(shapeDW2)));
  auto layoutDW2 = cute::make_layout(shapeDW2, cute::GenRowMajor{});
  auto tensorDW2 = std::make_shared<dllm::Tensor2D>(
      ptrDW2, layoutDW2, dllm::toDtype<T>(), dllm::CUDA);

  T *ptrW2Out;
  auto shapeW2Out =
      cute::make_shape(batchSize, static_cast<dllm::TensorIndexType>(1));
  CHECK_CUDART(cudaMalloc(reinterpret_cast<void **>(&ptrW2Out),
                          sizeof(T) * cute::size(shapeW2Out)));
  auto layoutW2Out = cute::make_layout(shapeW2Out, cute::GenRowMajor{});
  auto tensorW2Out = std::make_shared<dllm::Tensor2D>(
      ptrW2Out, layoutW2Out, dllm::toDtype<T>(), dllm::CUDA);

  T *ptrDW2Out;
  auto shapeDW2Out =
      cute::make_shape(batchSize, static_cast<dllm::TensorIndexType>(1));
  CHECK_CUDART(cudaMalloc(reinterpret_cast<void **>(&ptrDW2Out),
                          sizeof(T) * cute::size(shapeDW2Out)));
  auto layoutDW2Out = cute::make_layout(shapeDW2Out, cute::GenRowMajor{});
  auto tensorDW2Out = std::make_shared<dllm::Tensor2D>(
      ptrDW2Out, layoutDW2Out, dllm::toDtype<T>(), dllm::CUDA);

  dllm::ThreadPoolCompute threadPool{0, 2};

  {
    auto task = dllm::compute::Random::kaimingNorm(tensorW1);
    threadPool.submit(std::move(task));
  }
  {
    auto task = dllm::compute::Random::kaimingNorm(tensorW2);
    threadPool.submit(std::move(task));
  }

  const int epoch = 10;
  for (int i = 0; i < epoch; ++i) {
    for (auto &[tensorX, tensorY] : tensor) {
      {
        auto task = dllm::compute::FcNoBias::forward(
            tensorW1Out, tensorX, tensorW1, CUBLAS_COMPUTE_32F_PEDANTIC);
        threadPool.submit(std::move(task));
      }
      {
        auto task = dllm::compute::FcNoBias::forward(
            tensorW2Out, tensorW1Out, tensorW2, CUBLAS_COMPUTE_32F_PEDANTIC);
        threadPool.submit(std::move(task));
      }
      {
        auto task = dllm::compute::Mse::backward(
            dllm::util::flatten<1>(tensorDW2Out),
            dllm::util::toConstSharedPtr(dllm::util::flatten<1>(tensorW2Out)),
            dllm::util::toConstSharedPtr(dllm::util::flatten<1>(tensorY)));
        threadPool.submit(std::move(task));
      }
      {
        auto task = dllm::compute::FcNoBias::backwardW(
            tensorDW2, dllm::util::toConstSharedPtr(tensorDW2Out),
            dllm::util::toConstSharedPtr(tensorW1Out),
            CUBLAS_COMPUTE_32F_PEDANTIC);
        threadPool.submit(std::move(task));
      }
      {
        auto task = dllm::compute::FcNoBias::backwardX(
            tensorDW1Out, dllm::util::toConstSharedPtr(tensorDW2Out),
            dllm::util::toConstSharedPtr(tensorW2),
            CUBLAS_COMPUTE_32F_PEDANTIC);
        threadPool.submit(std::move(task));
      }
      {
        auto task = dllm::compute::FcNoBias::backwardW(
            tensorDW1, dllm::util::toConstSharedPtr(tensorDW1Out),
            dllm::util::toConstSharedPtr(tensorW1Out),
            CUBLAS_COMPUTE_32F_PEDANTIC);
        threadPool.submit(std::move(task));
      }
      {
        auto task = dllm::communication::AllReduce<dllm::communication::NCCL>::
            runInplace(dllm::util::flatten<1>(tensorW2),
                       dllm::communication::SUM);
        threadStreamNccl.submit(std::move(task));
      }
      {
        auto task = dllm::communication::AllReduce<dllm::communication::NCCL>::
            runInplace(dllm::util::flatten<1>(tensorW1),
                       dllm::communication::SUM);
        threadStreamNccl.submit(std::move(task));
      }
      {
        auto task = dllm::optimizer::Sgd::step(
            dllm::util::flatten<1>(tensorW2),
            dllm::util::toConstSharedPtr(dllm::util::flatten<1>(tensorDW2)),
            lr);
        threadPool.submit(std::move(task));
      }
      {
        auto task = dllm::optimizer::Sgd::step(
            dllm::util::flatten<1>(tensorW1),
            dllm::util::toConstSharedPtr(dllm::util::flatten<1>(tensorDW1)),
            lr);
        threadPool.submit(std::move(task));
      }

      {
        dllm::utils::FutureGuard{tensorW2->future->rFuture};
        dllm::utils::FutureGuard{tensorW2->future->wFuture};
      }
      {
        dllm::utils::FutureGuard{tensorW1->future->rFuture};
        dllm::utils::FutureGuard{tensorW1->future->wFuture};
      }
    }
  }

  Input xTest(batchSize, inputDim);
  xTest.setRandom();
  Output yTest = f(xTest);
  Output yTestRef(batchSize, 1);

  auto &[tensorX, tensorY] = tensor[0];

  CHECK_CUDART(cudaMemcpy(tensorX->data(), xTest.data(),
                          sizeof(T) * xTest.size(), cudaMemcpyHostToDevice));
  CHECK_CUDART(cudaDeviceSynchronize());

  {
    auto task = dllm::compute::FcNoBias::forward(tensorW1Out, tensorX, tensorW1,
                                                 CUBLAS_COMPUTE_32F_PEDANTIC);
    threadPool.submit(std::move(task));
  }
  {
    auto task = dllm::compute::FcNoBias::forward(
        tensorW2Out, tensorW1Out, tensorW2, CUBLAS_COMPUTE_32F_PEDANTIC);
    threadPool.submit(std::move(task));
  }
  {
    dllm::utils::FutureGuard{tensorW2Out->future->wFuture};
    dllm::utils::FutureGuard{tensorW2Out->future->rFuture};
  }

  CHECK_CUDART(cudaMemcpy(yTestRef.data(), tensorW2Out->data(),
                          sizeof(T) * yTest.size(), cudaMemcpyDeviceToHost));
  CHECK_CUDART(cudaDeviceSynchronize());

  if (mpiRank == 0) {
    printf("Error: %.5e\n", (yTest - yTestRef).norm());
  }

  CHECK_CUDART(cudaFree(ptrW2Out));
  CHECK_CUDART(cudaFree(ptrW2));
  CHECK_CUDART(cudaFree(ptrDW1Out));
  CHECK_CUDART(cudaFree(ptrW1Out));
  CHECK_CUDART(cudaFree(ptrW1));
  for (auto [ptrX, ptrY] : dataDevice) {
    CHECK_CUDART(cudaFree(ptrX));
    CHECK_CUDART(cudaFree(ptrY));
  }
  CHECK_MPI(MPI_Finalize());
}
