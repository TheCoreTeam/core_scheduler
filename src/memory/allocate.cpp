#include "memory/allocate.h"

#include "util.h"

namespace dllm::memory {
template <int N>
  requires(N >= 1 && N <= 4)
TaskCudart allocateRowMajor(std::shared_ptr<Tensor<N>> &p,
                            const typename Tensor<N>::Shape &shape,
                            const Dtype &dtype, const DeviceType &deviceType) {
  const auto layout = cute::make_layout(shape, cute::GenRowMajor{});
  p = std::make_shared<Tensor<N>>(nullptr, layout, dtype, deviceType);
  const std::size_t sizeInByte = toByte(dtype) * cute::cosize(layout);
  auto task = TaskCudart{[sizeInByte = sizeInByte, p = p, future = *p->future](
                             const ContextCudart *context) mutable {
    util::FutureGuard rGuard{future.rFuture};
    util::FutureGuard wGuard{future.wFuture};
    p->resetData(std::shared_ptr<const void>{
        [&] {
          void *ptr;
          CHECK_CUDART(cudaMallocFromPoolAsync(
              &ptr, sizeInByte, context->memPool, context->cudaStream));
          CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
          return ptr;
        }(),
        [stream = context->cudaStream](void *ptr) {
          CHECK_CUDART(cudaFreeAsync(ptr, stream));
        }});
  }};

  p->future->wFuture = task.get_future();
  return task;
}

template TaskCudart allocateRowMajor<1>(std::shared_ptr<Tensor<1>> &p,
                                        const Tensor<1>::Shape &shape,
                                        const Dtype &dtype,
                                        const DeviceType &deviceType);
template TaskCudart allocateRowMajor<2>(std::shared_ptr<Tensor<2>> &p,
                                        const Tensor<2>::Shape &shape,
                                        const Dtype &dtype,
                                        const DeviceType &deviceType);
template TaskCudart allocateRowMajor<3>(std::shared_ptr<Tensor<3>> &p,
                                        const Tensor<3>::Shape &shape,
                                        const Dtype &dtype,
                                        const DeviceType &deviceType);
template TaskCudart allocateRowMajor<4>(std::shared_ptr<Tensor<4>> &p,
                                        const Tensor<4>::Shape &shape,
                                        const Dtype &dtype,
                                        const DeviceType &deviceType);
}  // namespace dllm::memory
