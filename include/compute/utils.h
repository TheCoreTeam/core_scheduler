#pragma once
#include "tensor.h"
#include "threading/scheduler.h"

namespace dllm::compute::Utils {
void sum(const Scheduler &scheduler, Tensor &output,
         const ReadOnlyTensor &input, IntArrayRef dim,
         bool keep_dim = false,
         c10::optional<at::ScalarType> dtype = c10::nullopt);

inline void sum(const Scheduler &scheduler, Tensor &output,
                const ReadOnlyTensor &input, const int64_t dim,
                const bool keep_dim = false,
                const c10::optional<at::ScalarType> dtype = c10::nullopt) {
  return sum(scheduler, output, input, IntArrayRef{dim}, keep_dim, dtype);
}

void range(const Scheduler &scheduler, Tensor &tensor, const at::Scalar &start,
           const at::Scalar &end, TensorOptions options = {});

void arange(const Scheduler &scheduler, Tensor &tensor, const at::Scalar &start,
            const at::Scalar &end, TensorOptions options = {});

void arange(const Scheduler &scheduler, Tensor &tensor, const at::Scalar &start,
            const at::Scalar &end, const at::Scalar &step,
            TensorOptions options = {});

void randint(const Scheduler &scheduler, Tensor &tensor, int64_t low,
             int64_t high, IntArrayRef size,
             TensorOptions options = at::kLong);

void empty(const Scheduler &scheduler, Tensor &tensor, IntArrayRef size,
           TensorOptions options = {});

void empty_like(const Scheduler &scheduler, Tensor &dst,
                const ReadOnlyTensor &src);

void ones(const Scheduler &scheduler, Tensor &tensor, IntArrayRef size,
          TensorOptions options = {});

void ones_like(const Scheduler &scheduler, Tensor &dst,
               const ReadOnlyTensor &src);

void zeros(const Scheduler &scheduler, Tensor &tensor, IntArrayRef size,
           TensorOptions options = {});

void zeros_like(const Scheduler &scheduler, Tensor &dst,
                const ReadOnlyTensor &src);

void rand(const Scheduler &scheduler, Tensor &tensor, IntArrayRef size,
          TensorOptions options = {});

void rand_like(const Scheduler &scheduler, Tensor &dst,
               const ReadOnlyTensor &src);

void randn(const Scheduler &scheduler, Tensor &tensor, IntArrayRef size,
           TensorOptions options = {});

void randn_like(const Scheduler &scheduler, Tensor &dst,
                const ReadOnlyTensor &src);

void split(const Scheduler &scheduler, std::vector<Tensor> &output,
           const ReadOnlyTensor &src, const int64_t &split_size,
           const int64_t &dim);

void view(const Scheduler &scheduler, Tensor &output,
          const ReadOnlyTensor &input, IntArrayRef size);

void broadcast_to(const Scheduler &scheduler, Tensor &output,
                  const ReadOnlyTensor &input, IntArrayRef size);

void cat(const Scheduler &scheduler, Tensor &output,
         const std::vector<ReadOnlyTensor> &input, int64_t dim);
}  // namespace dllm::compute::Utils
