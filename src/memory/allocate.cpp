#include "memory/allocate.h"

namespace dllm::memory {
template <int N>
TaskCompute allocateRowMajor(
    std::shared_ptr<Tensor<N>> &p,
    const typename repeat_type<TensorIndexType, N, cute::Shape>::type &shape,
    const Dtype &dtype, const DeviceType &deviceType) {
  const auto layout = cute::make_layout(shape, cute::GenRowMajor{});
  p = std::make_shared<Tensor<N>>(nullptr, layout, dtype, deviceType);
  const std::size_t sizeInByte = toByte(dtype) * cute::cosize(layout);
  auto task = TaskCompute{
      [sizeInByte = sizeInByte, p = p](const ContextCompute *context) mutable {
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

template TaskCompute allocateRowMajor<1>(
    std::shared_ptr<Tensor<1>> &p,
    const typename repeat_type<TensorIndexType, 1, cute::Shape>::type &shape,
    const Dtype &dtype, const DeviceType &deviceType);
template TaskCompute allocateRowMajor<2>(
    std::shared_ptr<Tensor<2>> &p,
    const typename repeat_type<TensorIndexType, 2, cute::Shape>::type &shape,
    const Dtype &dtype, const DeviceType &deviceType);
template TaskCompute allocateRowMajor<3>(
    std::shared_ptr<Tensor<3>> &p,
    const typename repeat_type<TensorIndexType, 3, cute::Shape>::type &shape,
    const Dtype &dtype, const DeviceType &deviceType);
template TaskCompute allocateRowMajor<4>(
    std::shared_ptr<Tensor<4>> &p,
    const typename repeat_type<TensorIndexType, 4, cute::Shape>::type &shape,
    const Dtype &dtype, const DeviceType &deviceType);
}  // namespace dllm::memory
