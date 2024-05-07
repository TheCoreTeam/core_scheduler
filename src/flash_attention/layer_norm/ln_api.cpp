#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include "ATen/cuda/CUDAContext.h"
#include "ln.h"
#include "threading/context_compute.h"

using IntArrayRef1D = std::array<dllm::TensorIndexType, 1>;
using IntArrayRef2D = std::array<dllm::TensorIndexType, 2>;
using IntArrayRef3D = std::array<dllm::TensorIndexType, 2>;

/*

Supported Type combinations:

input  residual   compute   weights   output
============================================
fp32     fp32      fp32      fp32      fp32
fp16     fp32      fp32      fp32      fp16
fp16     fp16      fp32      fp32      fp16
bf16     fp32      fp32      fp32      bf16
bf16     bf16      fp32      fp32      bf16
fp16     fp16      fp32      fp16      fp16
bf16     bf16      fp32      bf16      bf16

Remarks:
Output type = Input type
Compute always in FP32

*/

namespace layer_norm {

// Create registries and provide runtime versions of config hash functions.

FwdRegistry FWD_FUNCS, PARALLEL_FWD_FUNCS;
BwdRegistry BWD_FUNCS, PARALLEL_BWD_FUNCS;

////////////////////////////////////////////////////////////////////////////////////////////////////

uint32_t get_type_id(torch::Dtype dtype) {
  if (dtype == torch::kFloat16) {
    return TypeId<fp16>::Value;
  } else if (dtype == torch::kBFloat16) {
    return TypeId<bf16>::Value;
  } else if (dtype == torch::kFloat32) {
    return TypeId<fp32>::Value;
  } else {
    TORCH_CHECK(false, "Type not supported: ", dtype);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

uint64_t get_key(torch::Dtype wtype, torch::Dtype itype, torch::Dtype rtype,
                 torch::Dtype otype, torch::Dtype ctype, uint64_t hidden_size) {
  using namespace layer_norm;
  uint64_t type_key = get_type_id(wtype) | (get_type_id(itype) << 2) |
                      (get_type_id(rtype) << 4) | (get_type_id(otype) << 6) |
                      (get_type_id(ctype) << 8);
  uint64_t launcher_key = (type_key << 32) | hidden_size;
  return launcher_key;
}

}  // namespace layer_norm

////////////////////////////////////////////////////////////////////////////////////////////////////

layer_norm::FwdFunction &get_fwd_launcher(
    torch::Dtype wtype, torch::Dtype itype, torch::Dtype rtype,
    torch::Dtype otype, torch::Dtype ctype, uint32_t hidden_size) {
  auto iter = layer_norm::FWD_FUNCS.find(
      layer_norm::get_key(wtype, itype, rtype, otype, ctype, hidden_size));
  if (iter != layer_norm::FWD_FUNCS.end()) {
    return iter->second;
  } else {
    TORCH_CHECK(false, "FWD: Unsupported hidden_size or types: ", hidden_size,
                wtype, itype, rtype, otype, ctype);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

layer_norm::BwdFunction &get_bwd_launcher(
    torch::Dtype wtype, torch::Dtype itype, torch::Dtype rtype,
    torch::Dtype otype, torch::Dtype ctype, uint32_t hidden_size) {
  auto iter = layer_norm::BWD_FUNCS.find(
      layer_norm::get_key(wtype, itype, rtype, otype, ctype, hidden_size));
  if (iter != layer_norm::BWD_FUNCS.end()) {
    return iter->second;
  } else {
    TORCH_CHECK(false, "BWD: Unsupported hidden_size or types: ", hidden_size,
                wtype, itype, rtype, otype, ctype);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

layer_norm::FwdFunction &get_parallel_fwd_launcher(
    torch::Dtype wtype, torch::Dtype itype, torch::Dtype rtype,
    torch::Dtype otype, torch::Dtype ctype, uint32_t hidden_size) {
  auto iter = layer_norm::PARALLEL_FWD_FUNCS.find(
      layer_norm::get_key(wtype, itype, rtype, otype, ctype, hidden_size));
  if (iter != layer_norm::PARALLEL_FWD_FUNCS.end()) {
    return iter->second;
  } else {
    TORCH_CHECK(false, "FWD: Unsupported hidden_size or types: ", hidden_size,
                wtype, itype, rtype, otype, ctype);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

layer_norm::BwdFunction &get_parallel_bwd_launcher(
    torch::Dtype wtype, torch::Dtype itype, torch::Dtype rtype,
    torch::Dtype otype, torch::Dtype ctype, uint32_t hidden_size) {
  auto iter = layer_norm::PARALLEL_BWD_FUNCS.find(
      layer_norm::get_key(wtype, itype, rtype, otype, ctype, hidden_size));
  if (iter != layer_norm::PARALLEL_BWD_FUNCS.end()) {
    return iter->second;
  } else {
    TORCH_CHECK(false, "BWD: Unsupported hidden_size or types: ", hidden_size,
                wtype, itype, rtype, otype, ctype);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

auto dropout_add_ln_fwd(
    const dllm::ContextCompute *context,
    const at::Tensor<2> &x0,                        // Input: BxSxhidden_size
    c10::optional<const at::Tensor<2>> &residual_,  // Residual: BxSxhidden_size
    const at::Tensor<1> &gamma,                     // hidden_size
    c10::optional<const at::Tensor<1>> &beta_,      // hidden_size
    c10::optional<const at::Tensor<1>> &rowscale_,  // BxS
    c10::optional<const at::Tensor<1>> &colscale_,  // hidden_size
    c10::optional<const at::Tensor<1>> &x0_subset_,  // BxS
    c10::optional<const at::Tensor<1>> &z_subset_,   // BxS
    const float dropout_p, const float epsilon, const float rowscale_const,
    // const int64_t z_numrows, c10::optional<at::Generator> gen_,
    const int64_t z_numrows, bool residual_in_fp32 = false,
    bool is_rms_norm = false) {
  auto itype = x0.scalar_type();
  auto rtype = residual_.has_value()
                   ? residual_.value().scalar_type()
                   : (residual_in_fp32 ? torch::kFloat32 : x0.scalar_type());
  auto wtype = gamma.scalar_type();
  auto otype = itype;
  auto ctype = torch::kFloat32;
  auto mtype = torch::kUInt8;

  TORCH_CHECK(x0.is_cuda());
  TORCH_CHECK(gamma.is_cuda());

  // TORCH_CHECK(x0.is_contiguous());
  // c10::IntArrayRef does not own the storage, so we need to construct a
  // vector. Otherwise just constructing IntArrayRef({blah}) will cause
  // uninitialized memory because blah is then deallocated.
  //  std::vector<int64_t> sizes_vec{
  //      !x0_subset_.has_value() ? x0.size(0) : x0_subset_.value().size(0),
  //      x0.size(1)};
  IntArrayRef2D sizes_vec{
      !x0_subset_.has_value() ? x0.size<0>() : x0_subset_.value().size<0>(),
      x0.size<1>()};
  // auto sizes = c10::IntArrayRef(sizes_vec);
  auto &sizes = sizes_vec;
  TORCH_CHECK(x0.dim() == 2);
  TORCH_CHECK(sizes.size() == 2);

  const int rows = sizes[0];
  const int cols = sizes[1];
  auto hidden_size = gamma.numel();
  TORCH_CHECK(hidden_size == cols);

  if (beta_.has_value()) {
    auto beta = beta_.value();
    TORCH_CHECK(beta.dtype == wtype);
    TORCH_CHECK(beta.is_cuda());
    // TORCH_CHECK(beta.is_contiguous());
    TORCH_CHECK(beta.sizes() == gamma.sizes());
  }

  if (residual_.has_value()) {
    auto residual = residual_.value();
    TORCH_CHECK(residual.is_cuda());
    // TORCH_CHECK(residual.is_contiguous());
    TORCH_CHECK(residual.sizes() == sizes);
  }

  if (rowscale_.has_value()) {
    auto rowscale = rowscale_.value();
    TORCH_CHECK(rowscale.is_cuda());
    // TORCH_CHECK(rowscale.is_contiguous());
    TORCH_CHECK(rowscale.sizes() == IntArrayRef1D{rows});
    TORCH_CHECK(rowscale.dtype == itype);
  }

  if (colscale_.has_value()) {
    auto colscale = colscale_.value();
    TORCH_CHECK(colscale.is_cuda());
    // TORCH_CHECK(colscale.is_contiguous());
    TORCH_CHECK(colscale.sizes() == IntArrayRef1D{cols});
    TORCH_CHECK(colscale.dtype == wtype);
  }

  if (x0_subset_.has_value()) {
    auto x0_subset = x0_subset_.value();
    TORCH_CHECK(x0_subset.is_cuda());
    // TORCH_CHECK(x0_subset.is_contiguous());
    TORCH_CHECK(x0_subset.sizes() == IntArrayRef1D{rows});
    TORCH_CHECK(x0_subset.dtype == torch::kInt32);

    TORCH_CHECK(z_subset_.has_value());
    auto z_subset = z_subset_.value();
    TORCH_CHECK(z_subset.is_cuda());
    // TORCH_CHECK(z_subset.is_contiguous());
    TORCH_CHECK(z_subset.sizes() == IntArrayRef1D{rows});
    TORCH_CHECK(z_subset.dtype == torch::kInt32);
  }

  TORCH_CHECK((hidden_size % 8 == 0) && (hidden_size <= 8192));
  TORCH_CHECK(epsilon >= 0.f);

  // Otherwise the kernel will be launched from cuda:0 device
  // Cast to char to avoid compiler warning about narrowing
  // at::cuda::CUDAGuard device_guard{(char)x0.get_device()};

  // auto opts = x0.options();

  bool save_x = residual_.has_value() || (dropout_p > 0.f) ||
                rowscale_.has_value() || colscale_.has_value() ||
                x0_subset_.has_value() || (itype != rtype);
  // at::Tensor x;
  // if (save_x) {
  //   x = torch::empty(sizes, opts.dtype(rtype));
  // }
  // at::Tensor dmask;
  // if (dropout_p > 0.f) {
  //   dmask = torch::empty(x0.sizes(), opts.dtype(mtype));
  // };
  // auto z = torch::empty(
  //     z_subset_.has_value() ? IntArrayRef2D{z_numrows, cols} : sizes,
  //     opts.dtype(otype));

  auto x = [&]() {
    if (save_x) {
      return torch::empty<dllm::CUDA>(sizes, rtype, context);
    }
    return decltype(torch::empty<dllm::CUDA>(sizes, rtype, context)){};
  }();
  auto dmask = [&]() {
    if (dropout_p > 0.f) {
      return torch::empty<dllm::CUDA>(x0.sizes(), mtype, context);
    };
    return decltype(torch::empty<dllm::CUDA>(x0.sizes(), mtype, context)){};
  }();
  auto z = torch::empty<dllm::CUDA>(
      z_subset_.has_value()
          ? IntArrayRef2D{static_cast<dllm::TensorIndexType>(z_numrows),
                          static_cast<dllm::TensorIndexType>(cols)}
          : sizes,
      otype, context);

  // auto mu = torch::empty({rows}, opts.dtype(ctype));
  // auto rsigma = torch::empty({rows}, opts.dtype(ctype));
  static_assert(
      std::is_same_v<std::remove_const_t<
                         std::remove_pointer_t<std::decay_t<const int *&>>>,
                     int>);
  auto mu = torch::empty<dllm::CUDA>(
      IntArrayRef1D{static_cast<dllm::TensorIndexType>(rows)}, ctype, context);
  auto rsigma = torch::empty<dllm::CUDA>(
      IntArrayRef1D{static_cast<dllm::TensorIndexType>(rows)}, ctype, context);

  layer_norm::LaunchParams<layer_norm::FwdParams> launch_params;

  launch_params.props = at::cuda::getCurrentDeviceProperties();
  // launch_params.stream = at::cuda::getCurrentCUDAStream().stream();
  launch_params.stream = context->cudaStream;
  TORCH_CHECK(dropout_p < 1.f);
  launch_params.params.dropout_keep_p = 1.f - dropout_p;
  // launch_params.params.residual =
  //     residual_.has_value() ? residual_.value().data_ptr() : nullptr;
  // launch_params.params.rowscale =
  //     rowscale_.has_value() ? rowscale_.value().data_ptr() : nullptr;
  // launch_params.params.colscale =
  //     colscale_.has_value() ? colscale_.value().data_ptr() : nullptr;
  // launch_params.params.x0_subset =
  //     x0_subset_.has_value() ? x0_subset_.value().data_ptr() : nullptr;
  // launch_params.params.z_subset =
  //     z_subset_.has_value() ? z_subset_.value().data_ptr() : nullptr;

  launch_params.params.residual =
      residual_.has_value() ? const_cast<void *>(residual_.value().data_ptr())
                            : nullptr;
  launch_params.params.rowscale =
      rowscale_.has_value() ? const_cast<void *>(rowscale_.value().data_ptr())
                            : nullptr;
  launch_params.params.colscale =
      colscale_.has_value() ? const_cast<void *>(colscale_.value().data_ptr())
                            : nullptr;
  launch_params.params.x0_subset =
      x0_subset_.has_value() ? const_cast<void *>(x0_subset_.value().data_ptr())
                             : nullptr;
  launch_params.params.z_subset =
      z_subset_.has_value() ? const_cast<void *>(z_subset_.value().data_ptr())
                            : nullptr;

  // auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
  //     gen_, at::cuda::detail::getDefaultCUDAGenerator());

  auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
  const int multiple =
      hidden_size <= 1536 ? 256 : (hidden_size <= 3072 ? 512 : 1024);
  // Request the kernel launcher.
  auto launcher = get_fwd_launcher(wtype, itype, rtype, otype, ctype,
                                   round_multiple(hidden_size, multiple));

  // Set the kernel runtime parameters.
  layer_norm::FwdParams &params = launch_params.params;
  params.rows = rows;
  params.cols = cols;
  // params.x0 = x0.data_ptr();
  params.x0 = const_cast<void *>(x0.data_ptr());
  params.x = save_x ? x->data_ptr() : nullptr;
  params.dmask = dropout_p > 0.f ? dmask->data_ptr() : nullptr;
  params.mu = mu->data_ptr();
  params.rs = rsigma->data_ptr();
  // params.gamma = gamma.data_ptr();
  params.gamma = const_cast<void *>(gamma.data_ptr());
  // params.beta = beta_.has_value() ? beta_.value().data_ptr() : nullptr;
  params.beta = beta_.has_value() ? const_cast<void *>(beta_.value().data_ptr())
                                  : nullptr;
  params.z = z->data_ptr();
  params.epsilon = epsilon;
  params.dropout_scale = 1.f / (1.f - dropout_p);
  params.inverse_cols = 1.f / float(params.cols);
  params.rowscale_const = rowscale_const;
  params.is_rms_norm = is_rms_norm;

  // Query the kernel-specific launch parameters.
  launcher(launch_params, true);

  // at::Tensor workspace, barrier;

  if (dropout_p > 0.f) {
    // number of times random will be generated per thread, to offset philox
    // counter in thc random state
    int64_t counter_offset = launch_params.elts_per_thread;

    // See Note [Acquire lock when using random generators]
    {
      // std::lock_guard<std::mutex> lock(gen->mutex_);
      // params.philox_args = gen->philox_cuda_state(counter_offset);
      params.philox_args = {context->curandSeed, context->curandOffset.load()};
      context->curandOffset += counter_offset;
    }
  }

  std::shared_ptr<dllm::Tensor<1>> workspace, barrier;
  if (launch_params.barrier_size > 0) {
    // TODO Any way to avoid this?
    // barrier =
    //     torch::zeros(launch_params.barrier_size, opts.dtype(torch::kInt32));
    // workspace =
    //     torch::empty(launch_params.workspace_bytes,
    //     opts.dtype(torch::kChar));
    // params.workspace = workspace.data_ptr();
    // params.barrier = barrier.data_ptr<int>();
    barrier = torch::empty<dllm::CUDA>(
        IntArrayRef1D{
            static_cast<dllm::TensorIndexType>(launch_params.barrier_size)},
        torch::kInt32, context);
    CHECK_CUDART(cudaMemsetAsync(barrier->data(), 0,
                                 sizeof(int) * launch_params.barrier_size,
                                 context->cudaStream));
    workspace = torch::empty<dllm::CUDA>(
        IntArrayRef1D{
            static_cast<dllm::TensorIndexType>(launch_params.workspace_bytes)},
        torch::kChar, context);
  }

  // Launch the kernel.
  launcher(launch_params, false);

  // return {z, x, dmask, mu, rsigma};
  return std::make_tuple(z, x, dmask, mu, rsigma);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

auto dropout_add_ln_bwd(
    const dllm::ContextCompute *context,
    const at::Tensor<2> &dz,                         // BxSxhidden_size
    c10::optional<const at::Tensor<3>> &dx_,         // BxSxhidden_size
    const at::Tensor<3> &x,                          // BxSxhidden_size
    c10::optional<const at::Tensor<2>> &x0_,         // BxSxhidden_size
    c10::optional<const at::Tensor<2>> &dmask_,      // BxSxhidden_size
    const at::Tensor<2> &mu,                         // BxS, FP32!
    const at::Tensor<2> &rsigma,                     // BxS, FP32!
    const at::Tensor<1> &gamma,                      // hidden_size
    c10::optional<const at::Tensor<1>> &rowscale_,   // BxS
    c10::optional<const at::Tensor<1>> &colscale_,   // hidden_size
    c10::optional<const at::Tensor<1>> &x0_subset_,  // BxS
    c10::optional<const at::Tensor<1>> &z_subset_,   // BxS
    const float dropout_p, const float rowscale_const, const int64_t x0_numrows,
    const bool has_residual, bool is_rms_norm = false) {
  auto itype = dz.scalar_type();
  auto rtype = x.scalar_type();
  auto wtype = gamma.scalar_type();
  auto otype = itype;
  auto ctype = torch::kFloat32;
  auto mtype = torch::kUInt8;

  if (dropout_p > 0.f) {
    TORCH_CHECK(dmask_.has_value());
  }

  TORCH_CHECK(dz.dtype == otype);
  TORCH_CHECK(mu.dtype == ctype);
  TORCH_CHECK(rsigma.dtype == ctype);

  TORCH_CHECK(x.is_cuda());
  TORCH_CHECK(dz.is_cuda());
  TORCH_CHECK(mu.is_cuda());
  TORCH_CHECK(rsigma.is_cuda());
  TORCH_CHECK(gamma.is_cuda());

  // TORCH_CHECK(x.is_contiguous());
  // TORCH_CHECK(dz.is_contiguous());

  auto sizes = x.sizes();
  TORCH_CHECK(sizes.size() == 2);
  auto rows = sizes[0];
  auto cols = sizes[1];
  TORCH_CHECK(dz.dim() == 2);
  TORCH_CHECK(dz.size<1>() == cols);
  auto hidden_size = gamma.numel();
  TORCH_CHECK(hidden_size == cols);

  // c10::IntArrayRef does not own the storage, so we need to construct a
  // vector. Otherwise just constructing IntArrayRef({blah}) will cause
  // uninitialized memory because blah is then deallocated.
  IntArrayRef2D x0_sizes_vec{!x0_subset_.has_value() ? rows : x0_numrows, cols};
  auto &x0_sizes = x0_sizes_vec;

  if (dx_.has_value()) {
    auto dx = dx_.value();
    TORCH_CHECK(dx.dtype == rtype);
    TORCH_CHECK(dx.is_cuda());
    // TORCH_CHECK(dx.is_contiguous());
    TORCH_CHECK(dx.sizes() == sizes);
  }

  if (dmask_.has_value()) {
    auto dmask = dmask_.value();
    TORCH_CHECK(dmask.dtype == mtype);
    TORCH_CHECK(dmask.is_cuda());
    // TORCH_CHECK(dmask.is_contiguous());
    TORCH_CHECK(dmask.sizes() == x0_sizes);
  }

  if (rowscale_.has_value()) {
    auto rowscale = rowscale_.value();
    TORCH_CHECK(rowscale.is_cuda());
    // TORCH_CHECK(rowscale.is_contiguous());
    TORCH_CHECK(rowscale.sizes() == IntArrayRef1D{rows});
    TORCH_CHECK(rowscale.dtype == itype);
  }

  if (colscale_.has_value()) {
    auto colscale = colscale_.value();
    TORCH_CHECK(colscale.is_cuda());
    // TORCH_CHECK(colscale.is_contiguous());
    TORCH_CHECK(colscale.sizes() == IntArrayRef1D{cols});
    TORCH_CHECK(colscale.dtype == wtype);

    TORCH_CHECK(x0_.has_value());
    auto x0 = x0_.value();
    TORCH_CHECK(x0.is_cuda());
    // TORCH_CHECK(x0.is_contiguous());
    TORCH_CHECK(x0.sizes() == x0_sizes);
    TORCH_CHECK(x0.dtype == itype);
  }

  if (x0_subset_.has_value()) {
    auto x0_subset = x0_subset_.value();
    TORCH_CHECK(x0_subset.is_cuda());
    // TORCH_CHECK(x0_subset.is_contiguous());
    TORCH_CHECK(x0_subset.sizes() == IntArrayRef1D{rows});
    TORCH_CHECK(x0_subset.dtype == torch::kInt32);

    TORCH_CHECK(z_subset_.has_value());
    auto z_subset = z_subset_.value();
    TORCH_CHECK(z_subset.is_cuda());
    // TORCH_CHECK(z_subset.is_contiguous());
    TORCH_CHECK(z_subset.sizes() == IntArrayRef1D{rows});
    TORCH_CHECK(z_subset.dtype == torch::kInt32);
  }

  TORCH_CHECK((hidden_size % 8 == 0) && (hidden_size <= 8192));

  TORCH_CHECK(mu.numel() == rows);
  TORCH_CHECK(mu.sizes() == rsigma.sizes());

  TORCH_CHECK(gamma.numel() == cols);

  // Otherwise the kernel will be launched from cuda:0 device
  // Cast to char to avoid compiler warning about narrowing
  // at::cuda::CUDAGuard device_guard{(char)dz.get_device()};

  // auto opts = x.options();

  // auto dx0 = torch::empty(x0_sizes, opts.dtype(itype));
  // at::Tensor dresidual;
  // if (has_residual) {
  //   dresidual = torch::empty_like(x, opts.dtype(rtype));
  // }
  // auto dgamma = torch::empty_like(gamma);
  // auto dbeta = torch::empty_like(gamma);
  // at::Tensor dcolscale;
  // if (colscale_.has_value()) {
  //   dcolscale = torch::empty_like(colscale_.value());
  // }

  auto dx0 = torch::empty<dllm::CUDA>(x0_sizes, itype, context);
  auto dresidual = [&]() {
    if (has_residual) {
      return torch::empty<dllm::CUDA>(x.layout, rtype, context);
    }
    return decltype(torch::empty<dllm::CUDA>(x.layout, rtype, context)){};
  }();
  auto dgamma = torch::empty_like<dllm::CUDA>(gamma, context);
  auto dbeta = torch::empty_like<dllm::CUDA>(gamma, context);
  auto dcolscale = [&]() {
    if (colscale_.has_value()) {
      return torch::empty_like<dllm::CUDA>(colscale_.value(), context);
    }
    return decltype(torch::empty_like<dllm::CUDA>(colscale_.value(),
                                                  context)){};
  }();

  layer_norm::LaunchParams<layer_norm::BwdParams> launch_params;
  // launch_params.stream = at::cuda::getCurrentCUDAStream().stream();
  launch_params.stream = context->cudaStream;
  launch_params.props = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(dropout_p < 1.f);
  launch_params.params.dropout_keep_p = 1.f - dropout_p;
  launch_params.params.dresidual =
      has_residual ? dresidual->data_ptr() : nullptr;
  launch_params.params.rowscale =
      rowscale_.has_value() ? const_cast<void *>(rowscale_.value().data_ptr())
                            : nullptr;
  launch_params.params.colscale =
      colscale_.has_value() ? const_cast<void *>(colscale_.value().data_ptr())
                            : nullptr;
  launch_params.params.x0_subset =
      x0_subset_.has_value() ? const_cast<void *>(x0_subset_.value().data_ptr())
                             : nullptr;
  launch_params.params.z_subset =
      z_subset_.has_value() ? const_cast<void *>(z_subset_.value().data_ptr())
                            : nullptr;

  auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
  const int multiple =
      hidden_size <= 1536 ? 256 : (hidden_size <= 3072 ? 512 : 1024);
  auto launcher = get_bwd_launcher(wtype, itype, rtype, otype, ctype,
                                   round_multiple(hidden_size, multiple));

  launcher(launch_params, true);

  // auto dgamma_part = torch::empty(
  //     {launch_params.params.ctas_per_col, hidden_size}, opts.dtype(ctype));
  // auto dbeta_part = torch::empty(
  //     {launch_params.params.ctas_per_col, hidden_size}, opts.dtype(ctype));
  // at::Tensor dcolscale_part;
  // if (colscale_.has_value()) {
  //   dcolscale_part = torch::empty(
  //       {launch_params.params.ctas_per_col, hidden_size}, opts.dtype(ctype));
  // }

  auto dgamma_part = torch::empty<dllm::CUDA>(
      IntArrayRef2D{launch_params.params.ctas_per_col, hidden_size}, ctype,
      context);
  auto dbeta_part = torch::empty<dllm::CUDA>(
      IntArrayRef2D{launch_params.params.ctas_per_col, hidden_size}, ctype,
      context);
  auto dcolscale_part = [&]() {
    if (colscale_.has_value()) {
      return torch::empty<dllm::CUDA>(
          IntArrayRef2D{launch_params.params.ctas_per_col, hidden_size}, ctype,
          context);
    }
    return decltype(torch::empty<dllm::CUDA>(
        IntArrayRef2D{launch_params.params.ctas_per_col, hidden_size}, ctype,
        context)){};
  }();

  layer_norm::BwdParams &params = launch_params.params;
  params.rows = rows;
  params.cols = cols;
  params.x = const_cast<void *>(x.data_ptr());
  params.x0 =
      x0_.has_value() ? const_cast<void *>(x0_.value().data_ptr()) : nullptr;
  params.dmask =
      dropout_p > 0.f ? const_cast<void *>(dmask_.value().data_ptr()) : nullptr;
  params.mu = const_cast<void *>(mu.data_ptr());
  params.rs = const_cast<void *>(rsigma.data_ptr());
  params.gamma = const_cast<void *>(gamma.data_ptr());
  params.dz = const_cast<void *>(dz.data_ptr());
  params.dx =
      dx_.has_value() ? const_cast<void *>(dx_.value().data_ptr()) : nullptr;
  params.dx0 = dx0->data_ptr();
  params.dbeta = dbeta->data_ptr();
  params.dgamma = dgamma->data_ptr();
  params.dcolscale = colscale_.has_value() ? dcolscale->data_ptr() : nullptr;
  params.dbeta_part = dbeta_part->data_ptr();
  params.dgamma_part = dgamma_part->data_ptr();
  params.dcolscale_part =
      colscale_.has_value() ? dcolscale_part->data_ptr() : nullptr;
  params.dropout_scale = 1.f / (1.f - dropout_p);
  params.inverse_cols = 1.f / float(params.cols);
  params.rowscale_const = rowscale_const;
  params.is_rms_norm = is_rms_norm;

  std::shared_ptr<dllm::Tensor<1>> workspace, barrier;
  if (launch_params.barrier_size > 0) {
    // TODO Any way to avoid this?
    // barrier =
    //     torch::zeros(launch_params.barrier_size, opts.dtype(torch::kInt32));
    // workspace =
    //     torch::empty(launch_params.workspace_bytes,
    //     opts.dtype(torch::kChar));
    // params.workspace = workspace.data_ptr();
    // params.barrier = barrier.data_ptr<int>();
    barrier = torch::empty<dllm::CUDA>(
        IntArrayRef1D{
            static_cast<dllm::TensorIndexType>(launch_params.barrier_size)},
        torch::kInt32, context);
    CHECK_CUDART(cudaMemsetAsync(barrier->data(), 0,
                                 sizeof(int) * launch_params.barrier_size,
                                 context->cudaStream));
    workspace = torch::empty<dllm::CUDA>(
        IntArrayRef1D{
            static_cast<dllm::TensorIndexType>(launch_params.workspace_bytes)},
        torch::kChar, context);
  }

  launcher(launch_params, false);

  // std::vector<at::Tensor> result = {dx0,   dresidual,   dgamma,
  //                                   dbeta, dgamma_part, dbeta_part};
  // if (colscale_.has_value()) {
  //   result.push_back(dcolscale);
  //   result.push_back(dcolscale_part);
  // }
  // return result;

  return std::make_tuple(dx0, dresidual, dgamma, dbeta, dgamma_part, dbeta_part,
                         dcolscale, dcolscale_part);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

auto dropout_add_ln_parallel_residual_fwd(
    const dllm::ContextCompute *context,
    const at::Tensor<2> &x0,                        // Input: BxSxhidden_size
    c10::optional<const at::Tensor<2>> &x1_,        // Input: BxSxhidden_size
    c10::optional<const at::Tensor<2>> &residual_,  // Residual: BxSxhidden_size
    const at::Tensor<1> &gamma0,                    // hidden_size
    c10::optional<const at::Tensor<1>> &beta0_,     // hidden_size
    c10::optional<const at::Tensor<1>> &gamma1_,    // hidden_size
    c10::optional<const at::Tensor<1>> &beta1_,     // hidden_size
    const float dropout_p, const float epsilon,
    // c10::optional<at::Generator> gen_, bool residual_in_fp32 = false,
    bool residual_in_fp32 = false, bool is_rms_norm = false) {
  auto itype = x0.scalar_type();
  auto rtype = residual_.has_value()
                   ? residual_.value().scalar_type()
                   : (residual_in_fp32 ? torch::kFloat32 : x0.scalar_type());
  auto wtype = gamma0.scalar_type();
  auto otype = itype;
  auto ctype = torch::kFloat32;
  auto mtype = torch::kUInt8;

  TORCH_CHECK(x0.is_cuda());
  TORCH_CHECK(gamma0.is_cuda());

  // TORCH_CHECK(x0.is_contiguous());
  const auto sizes = x0.sizes();
  TORCH_CHECK(x0.dim() == 2);

  const int rows = sizes[0];
  const int cols = sizes[1];
  auto hidden_size = gamma0.numel();
  TORCH_CHECK(hidden_size == cols);

  if (x1_.has_value()) {
    auto x1 = x1_.value();
    TORCH_CHECK(x1.is_cuda());
    // TORCH_CHECK(x1.is_contiguous());
    TORCH_CHECK(x1.sizes() == sizes);
  }

  if (residual_.has_value()) {
    auto residual = residual_.value();
    TORCH_CHECK(residual.is_cuda());
    // TORCH_CHECK(residual.is_contiguous());
    TORCH_CHECK(residual.sizes() == sizes);
  }

  if (beta0_.has_value()) {
    auto beta0 = beta0_.value();
    TORCH_CHECK(beta0.dtype == wtype);
    TORCH_CHECK(beta0.is_cuda());
    // TORCH_CHECK(beta0.is_contiguous());
    TORCH_CHECK(beta0.sizes() == gamma0.sizes());
  }

  if (gamma1_.has_value()) {
    auto gamma1 = gamma1_.value();
    TORCH_CHECK(gamma1.dtype == wtype);
    TORCH_CHECK(gamma1.is_cuda());
    // TORCH_CHECK(gamma1.is_contiguous());
    TORCH_CHECK(gamma1.sizes() == gamma0.sizes());
  }

  if (beta1_.has_value()) {
    auto beta1 = beta1_.value();
    TORCH_CHECK(beta1.dtype == wtype);
    TORCH_CHECK(beta1.is_cuda());
    // TORCH_CHECK(beta1.is_contiguous());
    TORCH_CHECK(beta1.sizes() == gamma0.sizes());
  }

  TORCH_CHECK((hidden_size % 8 == 0) && (hidden_size <= 8192));
  TORCH_CHECK(epsilon >= 0.f);

  // Otherwise the kernel will be launched from cuda:0 device
  // Cast to char to avoid compiler warning about narrowing
  // at::cuda::CUDAGuard device_guard{(char)x0.get_device()};

  // auto opts = x0.options();

  bool save_x = residual_.has_value() || x1_.has_value() || (dropout_p > 0.f) ||
                (itype != rtype);
  // at::Tensor x;
  // if (save_x) {
  //   x = torch::empty(sizes, opts.dtype(rtype));
  // }
  // at::Tensor dmask0, dmask1;
  // if (dropout_p > 0.f) {
  //   dmask0 = torch::empty(x0.sizes(), opts.dtype(mtype));
  //   if (x1_.has_value()) {
  //     dmask1 = torch::empty(x0.sizes(), opts.dtype(mtype));
  //   }
  // };
  // auto z0 = torch::empty(sizes, opts.dtype(otype));
  // at::Tensor z1;
  // if (gamma1_.has_value()) {
  //   z1 = torch::empty(sizes, opts.dtype(otype));
  // }

  auto x = [&]() {
    if (save_x) {
      return torch::empty<dllm::CUDA>(sizes, rtype, context);
    }
    return decltype(torch::empty<dllm::CUDA>(sizes, rtype, context)){};
  }();
  auto [dmask0, dmask1] = [&]() {
    auto dmask0 =
        decltype(torch::empty<dllm::CUDA>(x0.sizes(), mtype, context)){};
    auto dmask1 =
        decltype(torch::empty<dllm::CUDA>(x0.sizes(), mtype, context)){};
    if (dropout_p > 0.f) {
      dmask0 = torch::empty<dllm::CUDA>(x0.sizes(), mtype, context);
      if (x1_.has_value()) {
        dmask1 = torch::empty<dllm::CUDA>(x0.sizes(), mtype, context);
      }
    }
    return std::make_tuple(dmask0, dmask1);
  }();
  auto z0 = torch::empty<dllm::CUDA>(sizes, otype, context);
  auto z1 = [&]() {
    if (gamma1_.has_value()) {
      return torch::empty<dllm::CUDA>(sizes, otype, context);
    }
    return decltype(torch::empty<dllm::CUDA>(sizes, otype, context)){};
  }();

  // auto mu = torch::empty({rows}, opts.dtype(ctype));
  // auto rsigma = torch::empty({rows}, opts.dtype(ctype));
  auto mu = torch::empty<dllm::CUDA>(IntArrayRef1D{rows}, ctype, context);
  auto rsigma = torch::empty<dllm::CUDA>(IntArrayRef1D{rows}, ctype, context);

  layer_norm::LaunchParams<layer_norm::FwdParams> launch_params;

  launch_params.props = at::cuda::getCurrentDeviceProperties();
  // launch_params.stream = at::cuda::getCurrentCUDAStream().stream();
  launch_params.stream = context->cudaStream;
  TORCH_CHECK(dropout_p < 1.f);
  launch_params.params.dropout_keep_p = 1.f - dropout_p;
  launch_params.params.residual =
      residual_.has_value() ? const_cast<void *>(residual_.value().data_ptr())
                            : nullptr;

  // auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
  //     gen_, at::cuda::detail::getDefaultCUDAGenerator());

  auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
  const int multiple =
      hidden_size <= 1536 ? 256 : (hidden_size <= 3072 ? 512 : 1024);
  // Request the kernel launcher.
  auto launcher = get_parallel_fwd_launcher(
      wtype, itype, rtype, otype, ctype, round_multiple(hidden_size, multiple));

  // Set the kernel runtime parameters.
  layer_norm::FwdParams &params = launch_params.params;
  params.rows = rows;
  params.cols = cols;
  params.x0 = const_cast<void *>(x0.data_ptr());
  params.x1 =
      x1_.has_value() ? const_cast<void *>(x1_.value().data_ptr()) : nullptr;
  params.x = save_x ? x->data_ptr() : nullptr;
  params.dmask = dropout_p > 0.f ? dmask0->data_ptr() : nullptr;
  params.dmask1 =
      (dropout_p > 0.f && x1_.has_value()) ? dmask1->data_ptr() : nullptr;
  params.mu = mu->data_ptr();
  params.rs = rsigma->data_ptr();
  params.gamma = const_cast<void *>(gamma0.data_ptr());
  params.gamma1 = gamma1_.has_value()
                      ? const_cast<void *>(gamma1_.value().data_ptr())
                      : nullptr;
  params.beta = beta0_.has_value()
                    ? const_cast<void *>(beta0_.value().data_ptr())
                    : nullptr;
  params.beta1 = beta1_.has_value()
                     ? const_cast<void *>(beta1_.value().data_ptr())
                     : nullptr;
  params.z = z0->data_ptr();
  params.z1 = gamma1_.has_value() ? z1->data_ptr() : nullptr;
  params.epsilon = epsilon;
  params.dropout_scale = 1.f / (1.f - dropout_p);
  params.inverse_cols = 1.f / float(params.cols);
  params.is_rms_norm = is_rms_norm;

  // Query the kernel-specific launch parameters.
  launcher(launch_params, true);

  // at::Tensor workspace, barrier;

  if (dropout_p > 0.f) {
    // number of times random will be generated per thread, to offset philox
    // counter in thc random state
    int64_t counter_offset = 2 * launch_params.elts_per_thread;

    // See Note [Acquire lock when using random generators]
    {
      // std::lock_guard<std::mutex> lock(gen->mutex_);
      // params.philox_args = gen->philox_cuda_state(counter_offset);
      params.philox_args = {context->curandSeed, context->curandOffset.load()};
      context->curandOffset += counter_offset;
    }
  }

  std::shared_ptr<dllm::Tensor<1>> workspace, barrier;
  if (launch_params.barrier_size > 0) {
    // TODO Any way to avoid this?
    // barrier =
    //     torch::zeros(launch_params.barrier_size, opts.dtype(torch::kInt32));
    // workspace =
    //     torch::empty(launch_params.workspace_bytes,
    //     opts.dtype(torch::kChar));
    // params.workspace = workspace.data_ptr();
    // params.barrier = barrier.data_ptr<int>();
    barrier = torch::empty<dllm::CUDA>(
        IntArrayRef1D{
            static_cast<dllm::TensorIndexType>(launch_params.barrier_size)},
        torch::kInt32, context);
    CHECK_CUDART(cudaMemsetAsync(barrier->data(), 0,
                                 sizeof(int) * launch_params.barrier_size,
                                 context->cudaStream));
    workspace = torch::empty<dllm::CUDA>(
        IntArrayRef1D{
            static_cast<dllm::TensorIndexType>(launch_params.workspace_bytes)},
        torch::kChar, context);
  }

  // Launch the kernel.
  launcher(launch_params, false);

  // return {z0, z1, x, dmask0, dmask1, mu, rsigma};
  return std::make_tuple(z0, z1, x, dmask0, dmask1, mu, rsigma);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

auto dropout_add_ln_parallel_residual_bwd(
    const dllm::ContextCompute *context,
    const at::Tensor<2> &dz0,                     // BxSxhidden_size
    c10::optional<const at::Tensor<3>> &dz1_,     // BxSxhidden_size
    c10::optional<const at::Tensor<3>> &dx_,      // BxSxhidden_size
    const at::Tensor<3> &x,                       // BxSxhidden_size
    c10::optional<const at::Tensor<3>> &dmask0_,  // BxSxhidden_size
    c10::optional<const at::Tensor<3>> &dmask1_,  // BxSxhidden_size
    const at::Tensor<2> &mu,                      // BxS, FP32!
    const at::Tensor<2> &rsigma,                  // BxS, FP32!
    const at::Tensor<1> &gamma0,                  // hidden_size
    c10::optional<const at::Tensor<1>> &gamma1_,  // hidden_size
    const float dropout_p, const bool has_x1, const bool has_residual,
    bool is_rms_norm = false) {
  auto itype = dz0.scalar_type();
  auto rtype = x.scalar_type();
  auto wtype = gamma0.scalar_type();
  auto otype = itype;
  auto ctype = torch::kFloat32;
  auto mtype = torch::kUInt8;

  if (dropout_p > 0.f) {
    TORCH_CHECK(dmask0_.has_value());
  }

  TORCH_CHECK(dz0.dtype == otype);
  TORCH_CHECK(dz0.dtype == otype);
  TORCH_CHECK(mu.dtype == ctype);
  TORCH_CHECK(rsigma.dtype == ctype);

  TORCH_CHECK(x.is_cuda());
  TORCH_CHECK(dz0.is_cuda());
  TORCH_CHECK(mu.is_cuda());
  TORCH_CHECK(rsigma.is_cuda());
  TORCH_CHECK(gamma0.is_cuda());

  // TORCH_CHECK(x.is_contiguous());
  // TORCH_CHECK(dz0.is_contiguous());

  auto sizes = x.sizes();
  TORCH_CHECK(sizes.size() == 2);
  auto rows = sizes[0];
  auto cols = sizes[1];
  TORCH_CHECK(dz0.dim() == 2);
  TORCH_CHECK(dz0.size<1>() == cols);
  auto hidden_size = gamma0.numel();
  TORCH_CHECK(hidden_size == cols);

  if (dz1_.has_value()) {
    auto &dz1 = dz1_.value();
    TORCH_CHECK(dz1.dtype == otype);
    TORCH_CHECK(dz1.is_cuda());
    // TORCH_CHECK(dz1.is_contiguous());
    TORCH_CHECK(dz1.sizes() == sizes);

    TORCH_CHECK(gamma1_.has_value());
    auto &gamma1 = gamma1_.value();
    TORCH_CHECK(gamma1.dtype == wtype);
    TORCH_CHECK(gamma1.is_cuda());
    // TORCH_CHECK(gamma1.is_contiguous());
    TORCH_CHECK(gamma1.sizes() == gamma0.sizes());
  }

  if (dx_.has_value()) {
    auto dx = dx_.value();
    TORCH_CHECK(dx.dtype == rtype);
    TORCH_CHECK(dx.is_cuda());
    // TORCH_CHECK(dx.is_contiguous());
    TORCH_CHECK(dx.sizes() == sizes);
  }

  if (dmask0_.has_value()) {
    auto dmask0 = dmask0_.value();
    TORCH_CHECK(dmask0.dtype == mtype);
    TORCH_CHECK(dmask0.is_cuda());
    // TORCH_CHECK(dmask0.is_contiguous());
    TORCH_CHECK(dmask0.sizes() == sizes);

    if (has_x1) {
      TORCH_CHECK(dmask1_.has_value());
      auto dmask1 = dmask1_.value();
      TORCH_CHECK(dmask1.dtype == mtype);
      TORCH_CHECK(dmask1.is_cuda());
      // TORCH_CHECK(dmask1.is_contiguous());
      TORCH_CHECK(dmask1.sizes() == sizes);
    }
  }

  TORCH_CHECK((hidden_size % 8 == 0) && (hidden_size <= 8192));

  TORCH_CHECK(mu.numel() == rows);
  TORCH_CHECK(mu.sizes() == rsigma.sizes());

  // Otherwise the kernel will be launched from cuda:0 device
  // Cast to char to avoid compiler warning about narrowing
  // at::cuda::CUDAGuard device_guard{(char)dz0.get_device()};

  // auto opts = x.options();

  auto dx0 = torch::empty<dllm::CUDA>(sizes, itype, context);
  // at::Tensor dx1;
  // if (has_x1) {
  //   dx1 = torch::empty(sizes, opts.dtype(itype));
  // }
  // at::Tensor dresidual;
  // if (has_residual) {
  //   dresidual = torch::empty_like(x, opts.dtype(rtype));
  // }
  // auto dgamma0 = torch::empty_like(gamma0);
  // auto dbeta0 = torch::empty_like(gamma0);
  // at::Tensor dgamma1, dbeta1;
  // if (gamma1_.has_value()) {
  //   dgamma1 = torch::empty_like(gamma0);
  //   dbeta1 = torch::empty_like(gamma0);
  // }

  auto dx1 = [&]() {
    if (has_x1) {
      return torch::empty<dllm::CUDA>(sizes, itype, context);
    }
    return decltype(torch::empty<dllm::CUDA>(sizes, itype, context)){};
  }();
  auto dresidual = [&]() {
    if (has_residual) {
      return torch::empty<dllm::CUDA>(x.layout, rtype, context);
    }
    return decltype(torch::empty<dllm::CUDA>(x.layout, rtype, context)){};
  }();
  auto dgamma0 = torch::empty_like<dllm::CUDA>(gamma0, context);
  auto dbeta0 = torch::empty_like<dllm::CUDA>(gamma0, context);
  auto [dgamma1, dbeta1] = [&]() {
    if (gamma1_.has_value()) {
      auto dgamma1 = torch::empty_like<dllm::CUDA>(gamma0, context);
      auto dbeta1 = torch::empty_like<dllm::CUDA>(gamma0, context);
      return std::make_tuple(dgamma1, dbeta1);
    }
    auto dgamma1 = decltype(torch::empty_like<dllm::CUDA>(gamma0, context)){};
    auto dbeta1 = decltype(torch::empty_like<dllm::CUDA>(gamma0, context)){};
    return std::make_tuple(dgamma1, dbeta1);
  }();

  layer_norm::LaunchParams<layer_norm::BwdParams> launch_params;
  // launch_params.stream = at::cuda::getCurrentCUDAStream().stream();
  launch_params.stream = context->cudaStream;
  launch_params.props = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(dropout_p < 1.f);
  launch_params.params.dropout_keep_p = 1.f - dropout_p;
  launch_params.params.dresidual =
      has_residual ? dresidual->data_ptr() : nullptr;

  auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
  const int multiple =
      hidden_size <= 1536 ? 256 : (hidden_size <= 3072 ? 512 : 1024);
  auto launcher = get_parallel_bwd_launcher(
      wtype, itype, rtype, otype, ctype, round_multiple(hidden_size, multiple));

  launcher(launch_params, true);

  auto dgamma0_part = torch::empty<dllm::CUDA>(
      IntArrayRef2D{launch_params.params.ctas_per_col, hidden_size}, ctype,
      context);
  auto dbeta0_part = torch::empty<dllm::CUDA>(
      IntArrayRef2D{launch_params.params.ctas_per_col, hidden_size}, ctype,
      context);
  std::shared_ptr<at::Tensor<2>> dgamma1_part, dbeta1_part;
  if (gamma1_.has_value()) {
    dgamma1_part = torch::empty_like<dllm::CUDA>(dgamma0_part, context);
    CHECK_CUDART(cudaMemsetAsync(dgamma1_part->data(), 0, dgamma1_part->numel(),
                                 context->cudaStream));
    dbeta1_part = torch::empty_like<dllm::CUDA>(dbeta0_part, context);
    CHECK_CUDART(cudaMemsetAsync(dbeta1_part->data(), 0, dgamma1_part->numel(),
                                 context->cudaStream));
  }
  // at::Tensor workspace, barrier;

  layer_norm::BwdParams &params = launch_params.params;
  params.rows = rows;
  params.cols = cols;
  params.x = const_cast<void *>(x.data_ptr());
  params.dmask = dropout_p > 0.f
                     ? const_cast<void *>(dmask0_.value().data_ptr())
                     : nullptr;
  params.dmask1 = (dropout_p > 0.f && has_x1)
                      ? const_cast<void *>(dmask1_.value().data_ptr())
                      : nullptr;
  params.mu = const_cast<void *>(mu.data_ptr());
  params.rs = const_cast<void *>(rsigma.data_ptr());
  params.gamma = const_cast<void *>(gamma0.data_ptr());
  params.gamma1 = gamma1_.has_value()
                      ? const_cast<void *>(gamma1_.value().data_ptr())
                      : nullptr;
  params.dz = const_cast<void *>(dz0.data_ptr());
  params.dz1 =
      dz1_.has_value() ? const_cast<void *>(dz1_.value().data_ptr()) : nullptr;
  params.dx =
      dx_.has_value() ? const_cast<void *>(dx_.value().data_ptr()) : nullptr;
  params.dx0 = dx0->data_ptr();
  params.dx1 = has_x1 ? dx1->data_ptr() : nullptr;
  params.dbeta = dbeta0->data_ptr();
  params.dgamma = dgamma0->data_ptr();
  params.dbeta1 = gamma1_.has_value() ? dbeta1->data_ptr() : nullptr;
  params.dgamma1 = gamma1_.has_value() ? dgamma1->data_ptr() : nullptr;
  params.dbeta_part = dbeta0_part->data_ptr();
  params.dgamma_part = dgamma0_part->data_ptr();
  params.dbeta1_part = gamma1_.has_value() ? dbeta1_part->data_ptr() : nullptr;
  params.dgamma1_part =
      gamma1_.has_value() ? dgamma1_part->data_ptr() : nullptr;
  params.dropout_scale = 1.f / (1.f - dropout_p);
  params.inverse_cols = 1.f / float(params.cols);
  params.is_rms_norm = is_rms_norm;

  std::shared_ptr<dllm::Tensor<1>> workspace, barrier;
  if (launch_params.barrier_size > 0) {
    // TODO Any way to avoid this?
    // barrier =
    //     torch::zeros(launch_params.barrier_size, opts.dtype(torch::kInt32));
    // workspace =
    //     torch::empty(launch_params.workspace_bytes,
    //     opts.dtype(torch::kChar));
    // params.workspace = workspace.data_ptr();
    // params.barrier = barrier.data_ptr<int>();
    barrier = torch::empty<dllm::CUDA>(
        IntArrayRef1D{
            static_cast<dllm::TensorIndexType>(launch_params.barrier_size)},
        torch::kInt32, context);
    CHECK_CUDART(cudaMemsetAsync(barrier->data(), 0,
                                 sizeof(int) * launch_params.barrier_size,
                                 context->cudaStream));
    workspace = torch::empty<dllm::CUDA>(
        IntArrayRef1D{
            static_cast<dllm::TensorIndexType>(launch_params.workspace_bytes)},
        torch::kChar, context);
  }

  launcher(launch_params, false);

  // std::vector<at::Tensor> result = {
  //     dx0,    dx1,          dresidual,   dgamma0,      dbeta0,     dgamma1,
  //     dbeta1, dgamma0_part, dbeta0_part, dgamma1_part, dbeta1_part};
  // return result;

  return std::make_tuple(dx0, dx1, dresidual, dgamma0, dbeta0, dgamma1, dbeta1,
                         dgamma0_part, dbeta0_part, dgamma1_part, dbeta1_part);
}
