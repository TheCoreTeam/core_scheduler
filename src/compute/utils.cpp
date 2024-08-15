/*
 * Copyright (c) 2024 The Core team
 *
 * Licensed under the Apache License, Version 2.0;
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an 'AS IS' BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "compute/utils.h"

// #include <ATen/native/ForeachUtils.h>
#include <torch/csrc/autograd/generated/variable_factories.h>

#include "logger.h"
#include "tensor_impl.h"
#include "threading/scheduler_impl.h"
#include "threading/task_impl.h"

namespace cs::compute::Utils {
Tensor sum(const Scheduler &scheduler, const ReadOnlyTensor &input,
           const IntArrayRef dim, const bool keep_dim,
           const c10::optional<at::ScalarType> dtype) {
  struct Impl : Task::Impl {
    const IntArrayRef dim;
    const bool keep_dim;
    const c10::optional<at::ScalarType> dtype;

    explicit Impl(std::vector<Tensor> output /* tensor */,
                  std::vector<ReadOnlyTensor> input /* input */,
                  const IntArrayRef dim, const bool keep_dim,
                  const c10::optional<at::ScalarType> dtype)
        : Task::Impl{std::move(output), std::move(input), kMain, kCompute},
          dim{dim},
          keep_dim{keep_dim},
          dtype{dtype} {}
    void operator()() const override {
      output()[0].impl()->tensor() =
          at::sum(input()[0].impl()->tensor(), dim, keep_dim, dtype);
    }
    [[nodiscard]] const char *name() const override {
      return "cs::compute::Utils::sum";
    }
  };

  Tensor output{};
  CS_ASSERT_TRUE(!dim.empty(), "dim should not be empty");

  scheduler.impl()->submit(Task{
      std::make_shared<Impl>(Impl{{output}, {input}, dim, keep_dim, dtype})});
  return output;
}

Tensor range(const Scheduler &scheduler, const at::Scalar &start,
             const at::Scalar &end, const TensorOptions options) {
  struct Impl : Task::Impl {
    const at::Scalar start, end;
    const TensorOptions options;

    explicit Impl(std::vector<Tensor> output /* tensor */,
                  const at::Scalar &start, const at::Scalar &end,
                  const TensorOptions options)
        : Task::Impl{std::move(output), {}, kMain, kCompute},
          start{start},
          end{end},
          options{options} {}
    void operator()() const override {
      output()[0].impl()->tensor() = at::range(start, end, options);
    }
    [[nodiscard]] const char *name() const override {
      return "cs::compute::Utils::range";
    }
  };

  Tensor tensor{};
  scheduler.impl()->submit(
      Task{std::make_shared<Impl>(Impl{{tensor}, start, end, options})});
  return tensor;
}

Tensor arange(const Scheduler &scheduler, const at::Scalar &start,
              const at::Scalar &end, const TensorOptions options) {
  struct Impl : Task::Impl {
    const at::Scalar start, end;
    const TensorOptions options;

    explicit Impl(std::vector<Tensor> output /* tensor */,
                  const at::Scalar &start, const at::Scalar &end,
                  const TensorOptions options)
        : Task::Impl{std::move(output), {}, kMain, kCompute},
          start{start},
          end{end},
          options{options} {}
    void operator()() const override {
      output()[0].impl()->tensor() = at::arange(start, end, options);
    }
    [[nodiscard]] const char *name() const override {
      return "cs::compute::Utils::arange";
    }
  };

  Tensor tensor{};
  scheduler.impl()->submit(
      Task{std::make_shared<Impl>(Impl{{tensor}, start, end, options})});
  return tensor;
}

Tensor arange(const Scheduler &scheduler, const at::Scalar &start,
              const at::Scalar &end, const at::Scalar &step,
              const TensorOptions options) {
  struct Impl : Task::Impl {
    const at::Scalar start, end, step;
    const TensorOptions options;

    explicit Impl(std::vector<Tensor> output /* tensor */,
                  const at::Scalar &start, const at::Scalar &end,
                  const at::Scalar &step, const TensorOptions options)
        : Task::Impl{std::move(output), {}, kMain, kCompute},
          start{start},
          end{end},
          step{step},
          options{options} {}
    void operator()() const override {
      output()[0].impl()->tensor() = at::arange(start, end, step, options);
    }
    [[nodiscard]] const char *name() const override {
      return "cs::compute::Utils::arange";
    }
  };

  Tensor tensor{};
  scheduler.impl()->submit(
      Task{std::make_shared<Impl>(Impl{{tensor}, start, end, step, options})});
  return tensor;
}

Tensor randint(const Scheduler &scheduler, const int64_t low,
               const int64_t high, const IntArrayRef size,
               const TensorOptions options) {
  struct Impl : Task::Impl {
    const int64_t low, high;
    const IntArrayRef size;
    const TensorOptions options;

    explicit Impl(std::vector<Tensor> output /* tensor */, const int64_t low,
                  const int64_t high, const IntArrayRef size,
                  const TensorOptions &options)
        : Task::Impl{std::move(output), {}, kMain, kCompute},
          low{low},
          high{high},
          size{size},
          options{options} {}
    void operator()() const override {
      output()[0].impl()->tensor() = torch::randint(low, high, size, options);
    }
    [[nodiscard]] const char *name() const override {
      return "cs::compute::Utils::randint";
    }
  };

  Tensor tensor{};
  scheduler.impl()->submit(
      Task{std::make_shared<Impl>(Impl{{tensor}, low, high, size, options})});
  return tensor;
}

Tensor empty(const Scheduler &scheduler, const IntArrayRef size,
             const TensorOptions options) {
  struct Impl : Task::Impl {
    const IntArrayRef size;
    const TensorOptions options;

    explicit Impl(std::vector<Tensor> output /* tensor */,
                  const IntArrayRef size, const TensorOptions options)
        : Task::Impl{std::move(output), {}, kMain, kCompute},
          size{size},
          options{options} {}
    void operator()() const override {
      output()[0].impl()->tensor() = torch::empty(size, options);
    }
    [[nodiscard]] const char *name() const override {
      return "cs::compute::Utils::empty";
    }
  };

  Tensor tensor{};
  scheduler.impl()->submit(
      Task{std::make_shared<Impl>(Impl{{tensor}, size, options})});
  return tensor;
}

Tensor empty_like(const Scheduler &scheduler, const ReadOnlyTensor &src) {
  struct Impl : Task::Impl {
    explicit Impl(std::vector<Tensor> output /* tensor */,
                  std::vector<ReadOnlyTensor> input /* input */)
        : Task::Impl{std::move(output), std::move(input), kMain, kCompute} {}
    void operator()() const override {
      output()[0].impl()->tensor() =
          torch::empty_like(input()[0].impl()->tensor());
    }
    [[nodiscard]] const char *name() const override {
      return "cs::compute::Utils::empty_like";
    }
  };

  Tensor dst{};
  scheduler.impl()->submit(Task{std::make_shared<Impl>(Impl{{dst}, {src}})});
  return dst;
}

Tensor ones(const Scheduler &scheduler, const IntArrayRef size,
            const TensorOptions options) {
  struct Impl : Task::Impl {
    const IntArrayRef size;
    const TensorOptions options;

    explicit Impl(std::vector<Tensor> output /* tensor */,
                  const IntArrayRef size, const TensorOptions options)
        : Task::Impl{std::move(output), {}, kMain, kCompute},
          size{size},
          options{options} {}
    void operator()() const override {
      output()[0].impl()->tensor() = torch::ones(size, options);
    }
    [[nodiscard]] const char *name() const override {
      return "cs::compute::Utils::ones";
    }
  };

  Tensor tensor{};
  scheduler.impl()->submit(
      Task{std::make_shared<Impl>(Impl{{tensor}, size, options})});
  return tensor;
}

Tensor ones_like(const Scheduler &scheduler, const ReadOnlyTensor &src) {
  struct Impl : Task::Impl {
    explicit Impl(std::vector<Tensor> output /* tensor */,
                  std::vector<ReadOnlyTensor> input /* input */)
        : Task::Impl{std::move(output), std::move(input), kMain, kCompute} {}
    void operator()() const override {
      output()[0].impl()->tensor() =
          torch::ones_like(input()[0].impl()->tensor());
    }
    [[nodiscard]] const char *name() const override {
      return "cs::compute::Utils::ones_like";
    }
  };

  Tensor dst{};
  scheduler.impl()->submit(Task{std::make_shared<Impl>(Impl{{dst}, {src}})});
  return dst;
}

Tensor zeros(const Scheduler &scheduler, const IntArrayRef size,
             const TensorOptions options) {
  struct Impl : Task::Impl {
    const IntArrayRef size;
    const TensorOptions options;

    explicit Impl(std::vector<Tensor> output /* tensor */,
                  const IntArrayRef size, const TensorOptions options)
        : Task::Impl{std::move(output), {}, kMain, kCompute},
          size{size},
          options{options} {}
    void operator()() const override {
      output()[0].impl()->tensor() = torch::zeros(size, options);
    }
    [[nodiscard]] const char *name() const override {
      return "cs::compute::Utils::zeros";
    }
  };

  Tensor tensor{};
  scheduler.impl()->submit(
      Task{std::make_shared<Impl>(Impl{{tensor}, size, options})});
  return tensor;
}

Tensor zeros_like(const Scheduler &scheduler, const ReadOnlyTensor &src) {
  struct Impl : Task::Impl {
    explicit Impl(std::vector<Tensor> output /* tensor */,
                  std::vector<ReadOnlyTensor> input /* input */)
        : Task::Impl{std::move(output), std::move(input), kMain, kCompute} {}
    void operator()() const override {
      output()[0].impl()->tensor() =
          torch::zeros_like(input()[0].impl()->tensor());
    }
    [[nodiscard]] const char *name() const override {
      return "cs::compute::Utils::zeros_like";
    }
  };

  Tensor dst{};
  scheduler.impl()->submit(Task{std::make_shared<Impl>(Impl{{dst}, {src}})});
  return dst;
}

Tensor rand(const Scheduler &scheduler, const IntArrayRef size,
            const TensorOptions options) {
  struct Impl : Task::Impl {
    const IntArrayRef size;
    const TensorOptions options;

    explicit Impl(std::vector<Tensor> output /* tensor */,
                  const IntArrayRef size, const TensorOptions options)
        : Task::Impl{std::move(output), {}, kMain, kCompute},
          size{size},
          options{options} {}
    void operator()() const override {
      output()[0].impl()->tensor() = torch::rand(size, options);
    }
    [[nodiscard]] const char *name() const override {
      return "cs::compute::Utils::rand";
    }
  };

  Tensor tensor{};
  scheduler.impl()->submit(
      Task{std::make_shared<Impl>(Impl{{tensor}, size, options})});
  return tensor;
}

Tensor rand_like(const Scheduler &scheduler, const ReadOnlyTensor &src) {
  struct Impl : Task::Impl {
    explicit Impl(std::vector<Tensor> output /* tensor */,
                  std::vector<ReadOnlyTensor> input /* input */)
        : Task::Impl{std::move(output), std::move(input), kMain, kCompute} {}
    void operator()() const override {
      output()[0].impl()->tensor() =
          torch::rand_like(input()[0].impl()->tensor());
    }
    [[nodiscard]] const char *name() const override {
      return "cs::compute::Utils::rand_like";
    }
  };

  Tensor dst{};
  scheduler.impl()->submit(Task{std::make_shared<Impl>(Impl{{dst}, {src}})});
  return dst;
}

Tensor randn(const Scheduler &scheduler, const IntArrayRef size,
             const TensorOptions options) {
  struct Impl : Task::Impl {
    const IntArrayRef size;
    const TensorOptions options;

    explicit Impl(std::vector<Tensor> output /* tensor */,
                  const IntArrayRef size, const TensorOptions options)
        : Task::Impl{std::move(output), {}, kMain, kCompute},
          size{size},
          options{options} {}
    void operator()() const override {
      output()[0].impl()->tensor() = torch::randn(size, options);
    }
    [[nodiscard]] const char *name() const override {
      return "cs::compute::Utils::randn";
    }
  };

  Tensor tensor{};
  scheduler.impl()->submit(
      Task{std::make_shared<Impl>(Impl{{tensor}, size, options})});
  return tensor;
}

Tensor randn_like(const Scheduler &scheduler, const ReadOnlyTensor &src) {
  struct Impl : Task::Impl {
    explicit Impl(std::vector<Tensor> output /* tensor */,
                  std::vector<ReadOnlyTensor> input /* input */)
        : Task::Impl{std::move(output), std::move(input), kMain, kCompute} {}
    void operator()() const override {
      output()[0].impl()->tensor() =
          torch::randn_like(input()[0].impl()->tensor());
    }
    [[nodiscard]] const char *name() const override {
      return "cs::compute::Utils::randn_like";
    }
  };

  Tensor dst{};
  scheduler.impl()->submit(Task{std::make_shared<Impl>(Impl{{dst}, {src}})});
  return dst;
}

std::vector<Tensor> split(const Scheduler &scheduler, const ReadOnlyTensor &src,
                          const int64_t &split_size, const int64_t &dim) {
  struct Impl : Task::Impl {
    const int64_t split_size;
    const int64_t dim;

    explicit Impl(std::vector<Tensor> output /* tensor */,
                  std::vector<ReadOnlyTensor> input /* inputs */,
                  const int64_t split_size, const int64_t dim)
        : Task::Impl{std::move(output), std::move(input), kMain, kCompute},
          split_size{split_size},
          dim{dim} {}
    void operator()() const override {
      const auto v = input()[0].impl()->tensor().split(split_size, dim);
      for (std::size_t i = 0; i < v.size(); ++i) {
        output()[i].impl()->tensor() = v[i];
      }
    }
    [[nodiscard]] const char *name() const override {
      return "cs::compute::Utils::split";
    }
  };

  const auto input_size = src.sizes();
  const auto split_num =
      input_size[dim > 0 ? dim : input_size.size() + dim] / split_size;
  std::vector<Tensor> output;
  output.resize(split_num);
  at::SmallVector<Tensor> outputNew;
  outputNew.reserve(split_size);
  for (auto &p : output) {
    Tensor pNew;
    outputNew.push_back(pNew);
    p = std::move(pNew);
  }
  scheduler.impl()->submit(
      Task{std::make_shared<Impl>(Impl{output, {src}, split_size, dim})});
  return output;
}

Tensor view(const Scheduler &scheduler, const ReadOnlyTensor &input,
            const IntArrayRef size) {
  struct Impl : Task::Impl {
    const IntArrayRef size;

    explicit Impl(std::vector<Tensor> output /* tensor */,
                  std::vector<ReadOnlyTensor> input /* input */,
                  const IntArrayRef size)
        : Task::Impl{std::move(output), std::move(input), kMain, kCompute},
          size{size} {}
    void operator()() const override {
      output()[0].impl()->tensor() = input()[0].impl()->tensor().view(size);
    }
    [[nodiscard]] const char *name() const override {
      return "cs::compute::Utils::view";
    }
  };

  Tensor output{};

  scheduler.impl()->submit(
      Task{std::make_shared<Impl>(Impl{{output}, {input}, size})});
  return output;
}

Tensor as_strided(const Scheduler &scheduler, const ReadOnlyTensor &input,
                  const IntArrayRef size, const IntArrayRef stride,
                  const optional<int64_t> storage_offset) {
  struct Impl : Task::Impl {
    const IntArrayRef size;
    const IntArrayRef stride;
    const optional<int64_t> storage_offset;

    explicit Impl(std::vector<Tensor> output /* tensor */,
                  std::vector<ReadOnlyTensor> input /* input */,
                  const IntArrayRef size, const IntArrayRef stride,
                  const optional<int64_t> storage_offset)
        : Task::Impl{std::move(output), std::move(input), kMain, kCompute},
          size{size},
          stride{stride},
          storage_offset(storage_offset) {}
    void operator()() const override {
      output()[0].impl()->tensor() =
          input()[0].impl()->tensor().as_strided(size, stride, storage_offset);
    }
    [[nodiscard]] const char *name() const override {
      return "cs::compute::Utils::as_strided";
    }
  };

  Tensor output{};

  scheduler.impl()->submit(Task{std::make_shared<Impl>(
      Impl{{output}, {input}, size, stride, storage_offset})});
  return output;
}

Tensor broadcast_to(const Scheduler &scheduler, const ReadOnlyTensor &input,
                    const IntArrayRef size) {
  struct Impl : Task::Impl {
    const IntArrayRef size;

    explicit Impl(std::vector<Tensor> output /* tensor */,
                  std::vector<ReadOnlyTensor> input /* inputs */,
                  const IntArrayRef size)
        : Task::Impl{std::move(output), std::move(input), kMain, kCompute},
          size{size} {}
    void operator()() const override {
      output()[0].impl()->tensor() =
          input()[0].impl()->tensor().broadcast_to(size);
    }
    [[nodiscard]] const char *name() const override {
      return "cs::compute::Utils::broadcast_to";
    }
  };

  Tensor output{};
  scheduler.impl()->submit(
      Task{std::make_shared<Impl>(Impl{{output}, {input}, size})});
  return output;
}

Tensor cat(const Scheduler &scheduler, const std::vector<ReadOnlyTensor> &input,
           const int64_t dim) {
  struct Impl : Task::Impl {
    const int64_t dim;

    explicit Impl(std::vector<Tensor> output /* tensor */,
                  std::vector<ReadOnlyTensor> input /* inputs */,
                  const int64_t dim)
        : Task::Impl{std::move(output), std::move(input), kMain, kCompute},
          dim{dim} {}
    void operator()() const override {
      std::vector<at::Tensor> vInput;
      vInput.reserve(input().size());
      for (const auto &t : input()) {
        vInput.push_back(t.impl()->tensor());
      }
      output()[0].impl()->tensor() = at::cat(vInput, dim);
    }
    [[nodiscard]] const char *name() const override {
      return "cs::compute::Utils::cat";
    }
  };

  Tensor output{};

  scheduler.impl()->submit(
      Task{std::make_shared<Impl>(Impl{{output}, input, dim})});
  return output;
}

Tensor add(const Scheduler &scheduler, ReadOnlyTensor x, ReadOnlyTensor y) {
  Tensor output;
  struct Impl : Task::Impl {
    explicit Impl(std::vector<Tensor> output /* tensor */,
                  std::vector<ReadOnlyTensor> input /* A, B */)
        : Task::Impl{std::move(output), std::move(input), kMain, kCompute} {}
    void operator()() const override {
      output()[0].impl()->tensor() =
          at::add(input()[0].impl()->tensor(), input()[1].impl()->tensor());
    }
    [[nodiscard]] const char *name() const override {
      return "cs::compute::Utils::Add";
    }
  };
  scheduler.impl()->submit(
      Task{std::make_shared<Impl>(Impl{{output}, {x, y}})});
  return output;
}

void zero_(const Scheduler &scheduler, const Tensor &tensor) {
  struct Impl : Task::Impl {
    explicit Impl(const Tensor &tensor /* tensor */)
        : Task::Impl{{tensor}, {tensor}, kMain, kCompute} {}
    void operator()() const override {
      (void)output()[0].impl()->tensor().zero_();
    }
    [[nodiscard]] const char *name() const override {
      return "cs::compute::Utils::zero_";
    }
  };

  scheduler.impl()->submit(Task{std::make_shared<Impl>(Impl{tensor})});
}

void uniform_(const Scheduler &scheduler, const Tensor &tensor,
              const double from, const double to) {
  struct Impl : Task::Impl {
    const double from;
    const double to;

    explicit Impl(const Tensor &tensor /* tensor */, const double from,
                  const double to)
        : Task::Impl{{tensor}, {tensor}, kMain, kCompute}, from{from}, to{to} {}
    void operator()() const override {
      (void)output()[0].impl()->tensor().uniform_(from, to);
    }
    [[nodiscard]] const char *name() const override {
      return "cs::compute::Utils::uniform_";
    }
  };

  scheduler.impl()->submit(
      Task{std::make_shared<Impl>(Impl{tensor, from, to})});
}

void normal_(const Scheduler &scheduler, const Tensor &tensor,
             const double mean, const double std) {
  struct Impl : Task::Impl {
    const double mean;
    const double std;

    explicit Impl(const Tensor &tensor /* tensor */, const double mean,
                  const double std)
        : Task::Impl{{tensor}, {tensor}, kMain, kCompute},
          mean{mean},
          std{std} {}
    void operator()() const override {
      (void)output()[0].impl()->tensor().normal_(mean, std);
    }
    [[nodiscard]] const char *name() const override {
      return "cs::compute::Utils::normal_";
    }
  };

  scheduler.impl()->submit(
      Task{std::make_shared<Impl>(Impl{tensor, mean, std})});
}

Tensor clone(const Scheduler &scheduler, const Tensor &tensor) {
  struct Impl : Task::Impl {
    explicit Impl(const Tensor &result, const ReadOnlyTensor &input)
        : Task::Impl{{result}, {input}, kMain, kCompute} {}
    void operator()() const override {
      output()[0].impl()->tensor() = input()[0].impl()->tensor().clone();
    }
    [[nodiscard]] const char *name() const override {
      return "cs::compute::Utils::clone";
    }
  };

  Tensor result;
  scheduler.impl()->submit(Task{std::make_shared<Impl>(Impl{result, tensor})});
  return result;
}

Tensor linalg_vector_norm(const Scheduler &scheduler,
                          const ReadOnlyTensor &self, const double ord,
                          at::OptionalIntArrayRef dim, const bool keepdim,
                          const c10::optional<at::ScalarType> dtype) {
  Tensor result;
  if (dim.has_value()) {
    struct Impl : Task::Impl {
      const Scalar ord;
      const IntArray dim;
      const bool keepdim;
      const c10::optional<at::ScalarType> dtype;

      explicit Impl(const Tensor &result, const ReadOnlyTensor &input,
                    const double ord, const IntArray &dim, const bool keepdim,
                    const c10::optional<at::ScalarType> dtype)
          : Task::Impl{{result}, {input}, kMain, kCompute},
            ord{ord},
            dim{dim},
            keepdim{keepdim},
            dtype{dtype} {}
      void operator()() const override {
        output()[0].impl()->tensor() = at::linalg_vector_norm(
            input()[0].impl()->tensor(), ord, dim, keepdim, dtype);
      }
      [[nodiscard]] const char *name() const override {
        return "cs::compute::Utils::linalg_vector_norm";
      }
    };

    scheduler.impl()->submit(Task{std::make_shared<Impl>(
        Impl{result, self, ord, {dim->begin(), dim->end()}, keepdim, dtype})});
  } else {
    struct Impl : Task::Impl {
      const Scalar ord;
      const bool keepdim;
      const c10::optional<at::ScalarType> dtype;

      explicit Impl(const Tensor &result, const ReadOnlyTensor &input,
                    const double ord, const bool keepdim,
                    const c10::optional<at::ScalarType> dtype)
          : Task::Impl{{result}, {input}, kMain, kCompute},
            ord{ord},
            keepdim{keepdim},
            dtype{dtype} {}
      void operator()() const override {
        output()[0].impl()->tensor() = at::linalg_vector_norm(
            input()[0].impl()->tensor(), ord, c10::nullopt, keepdim, dtype);
      }
      [[nodiscard]] const char *name() const override {
        return "cs::compute::Utils::linalg_vector_norm";
      }
    };

    scheduler.impl()->submit(
        Task{std::make_shared<Impl>(Impl{result, self, ord, keepdim, dtype})});
  }
  return result;
}

Tensor clip_grad_norm_(const Scheduler &scheduler,
                       const std::vector<Tensor> &tensor, const double max_norm,
                       const double norm_type) {
  struct Impl : Task::Impl {
    const double max_norm;
    const double norm_type;

    explicit Impl(const Tensor &total_norm, const std::vector<Tensor> &result,
                  const double max_norm, const double norm_type)
        : Task::Impl{[&] {
                       std::vector output{result};
                       output.push_back(total_norm);
                       return output;
                     }(),
                     {result.begin(), result.end()},
                     kMain,
                     kCompute},
          max_norm{max_norm},
          norm_type{norm_type} {}
    void operator()() const override {
      if (output().empty()) {
        return;
      }
      std::vector<at::Tensor> tensors;
      for (auto &t : input()) {
        tensors.push_back(t.impl()->tensor().view(-1));
      }
      const auto total_norm =
          at::linalg_vector_norm(at::hstack(tensors), norm_type);
      const auto clip_coef =
          at::empty_like(total_norm, at::MemoryFormat::Preserve)
              .fill_(max_norm)
              .div_(total_norm.add(1e-6));
      const auto clip_coef_clamped = at::clamp(clip_coef, c10::nullopt, 1.0);
      at::_foreach_mul_(tensors, clip_coef_clamped);
      output()[output().size() - 1].impl()->tensor() = total_norm;
    }
    [[nodiscard]] const char *name() const override {
      return "cs::compute::Utils::clip_grad_norm_";
    }
  };

  Tensor total_norm;
  scheduler.impl()->submit(Task{
      std::make_shared<Impl>(Impl{total_norm, tensor, max_norm, norm_type})});
  return total_norm;
}
}  // namespace cs::compute::Utils
