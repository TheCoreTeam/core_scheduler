#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include "ATen/cuda/CUDAContext.h"
#include "ln.h"
#include "logger.h"
#include "random/random_internal.h"
#include "threading/context_compute.h"
#include "threading/task_compute.h"
#include "util.h"

using IntArrayRef1D = std::array<dllm::TensorIndexType, 1>;
using IntArrayRef2D = std::array<dllm::TensorIndexType, 2>;
using IntArrayRef3D = std::array<dllm::TensorIndexType, 3>;

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

// FwdRegistry FWD_FUNCS, PARALLEL_FWD_FUNCS;
// BwdRegistry BWD_FUNCS, PARALLEL_BWD_FUNCS;

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

namespace {

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

}  // namespace

namespace dllm::flash_attn::LayerNorm {

void dropout_add_ln_fwd_no_dropout(
    at::Tensor<2> &z,            // Input: BxSxhidden_size
    at::Tensor<1> &mu,           // Input: FP32
    at::Tensor<1> &rsigma,       // Input: FP32
    const at::Tensor<2> &x0,     // Input: BxSxhidden_size
    const at::Tensor<1> &gamma,  // hidden_size  // weight
    const at::Tensor<1> &beta,   // hidden_size  // bias
    const float epsilon,         // epsilon
    const dllm::ContextCompute *context) {
  const float dropout_p = 0;       // 0.0
  const float rowscale_const = 1;  // 1.0
  const int64_t z_numrows = 0;     // 0
  bool residual_in_fp32 = false;   // Not sure
  bool is_rms_norm = false;        // false
  at::cuda::setCurrentCUDAStream(context->cudaStream);

  auto itype = x0.scalar_type();
  auto rtype = (x0.scalar_type());
  auto wtype = gamma.scalar_type();
  auto otype = itype;
  auto ctype = torch::kFloat32;
  auto mtype = torch::kUInt8;

  TORCH_CHECK(x0.is_cuda());
  TORCH_CHECK(gamma.is_cuda());

  IntArrayRef2D sizes_vec{x0.size<0>(), x0.size<1>()};
  auto &sizes = sizes_vec;
  TORCH_CHECK(x0.dim() == 2);
  TORCH_CHECK(sizes.size() == 2);

  const int rows = sizes[0];
  const int cols = sizes[1];
  auto hidden_size = gamma.numel();
  TORCH_CHECK(hidden_size == cols);

  TORCH_CHECK((hidden_size % 8 == 0) && (hidden_size <= 8192));
  TORCH_CHECK(epsilon >= 0.f);

  static_assert(
      std::is_same_v<std::remove_const_t<
                         std::remove_pointer_t<std::decay_t<const int *&>>>,
                     int>);

  layer_norm::LaunchParams<layer_norm::FwdParams> launch_params;

  launch_params.props = at::cuda::getCurrentDeviceProperties();
  launch_params.stream = context->cudaStream;
  TORCH_CHECK(dropout_p < 1.f);
  launch_params.params.dropout_keep_p = 1.f - dropout_p;

  launch_params.params.residual = nullptr;
  launch_params.params.rowscale = nullptr;
  launch_params.params.colscale = nullptr;
  launch_params.params.x0_subset = nullptr;
  launch_params.params.z_subset = nullptr;

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
  params.x0 = const_cast<void *>(x0.data_ptr());
  params.x = nullptr;
  params.dmask = nullptr;
  params.mu = mu.data_ptr();
  params.rs = rsigma.data_ptr();
  params.gamma = const_cast<void *>(gamma.data_ptr());
  params.beta = const_cast<void *>(beta.data_ptr());
  params.z = z.data_ptr();
  params.epsilon = epsilon;
  params.dropout_scale = 1.f / (1.f - dropout_p);
  params.inverse_cols = 1.f / float(params.cols);
  params.rowscale_const = rowscale_const;
  params.is_rms_norm = is_rms_norm;

  // Query the kernel-specific launch parameters.
  launcher(launch_params, true);

  std::shared_ptr<dllm::Tensor<1>> workspace, barrier;
  if (launch_params.barrier_size > 0) {
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
    params.barrier = barrier->data_ptr<int>();
    params.workspace = workspace->data_ptr();
  }

  // Launch the kernel.
  launcher(launch_params, false);
  CHECK_CUDART(cudaStreamSynchronize(launch_params.stream));

}

dllm::TaskCompute forward(
    std::shared_ptr<dllm::Tensor<2>> z,               // Input: BxSxhidden_size
    std::shared_ptr<dllm::Tensor<1>> mu,              // Input: FP32
    std::shared_ptr<dllm::Tensor<1>> rsigma,          // Input: FP32
    const std::shared_ptr<const dllm::Tensor<2>> x0,  // Input: BxSxhidden_size
    const std::shared_ptr<const dllm::Tensor<1>>
        gamma,  // hidden_size  // weight
    const std::shared_ptr<const dllm::Tensor<1>> beta,  // hidden_size  // bias
    const float epsilon                                 // epsilon
) {
  auto task = dllm::TaskCompute{
      [z = z, mu = mu, rsigma = rsigma, x0 = x0, gamma = gamma, beta = beta,
       epsilon = epsilon, zFuture = *z->future, muFuture = *mu->future,
       rsigmaFuture = *rsigma->future, x0Future = x0->future->wFuture,
       gammaFuture = gamma->future->wFuture,
       betaFuture =
           beta->future->wFuture](const dllm::ContextCompute *context) mutable {
        util::FutureGuard zRGuard{zFuture.rFuture};
        util::FutureGuard zWGuard{zFuture.wFuture};
        util::FutureGuard muRGuard{muFuture.rFuture};
        util::FutureGuard muWGuard{muFuture.wFuture};
        util::FutureGuard rsigmaRGuard{rsigmaFuture.rFuture};
        util::FutureGuard rsigmaWGuard{rsigmaFuture.wFuture};
        util::FutureGuard x0Guard{x0Future};
        util::FutureGuard gammaGuard{gammaFuture};
        util::FutureGuard betaGuard{betaFuture};
        dropout_add_ln_fwd_no_dropout(*z, *mu, *rsigma, *x0, *gamma, *beta,
                                      epsilon, context);
        CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));

        z.reset();
        mu.reset();
        rsigma.reset();
        x0.reset();
        gamma.reset();
        beta.reset();
      }};
  const TaskFuture future = task.get_future();
  z->future->wFuture = future;
  mu->future->wFuture = future;
  rsigma->future->wFuture = future;
  x0->future->rFuture = future;
  gamma->future->rFuture = future;
  beta->future->rFuture = future;
  return task;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void dropout_add_ln_bwd_no_dropout(at::Tensor<2> &dx0, at::Tensor<1> &dgamma,
                                   at::Tensor<1> &dbeta,
                                   const at::Tensor<2> &dz,  // BxSxhidden_size
                                   const at::Tensor<2> &x,   // BxSxhidden_size
                                   const at::Tensor<1> &mu,  // BxS, FP32!
                                   const at::Tensor<1> &rsigma,  // BxS, FP32!
                                   const at::Tensor<1> &gamma,   // hidden_size
                                   const dllm::ContextCompute *context) {
  at::cuda::setCurrentCUDAStream(context->cudaStream);

  const auto rowscale_const = 1;
  const auto dropout_p = 0.f;
  auto itype = dz.scalar_type();
  auto rtype = x.scalar_type();
  auto wtype = gamma.scalar_type();
  auto otype = itype;
  auto ctype = torch::kFloat32;
  auto mtype = torch::kUInt8;

  TORCH_CHECK(x.is_cuda());
  TORCH_CHECK(dz.is_cuda());
  TORCH_CHECK(mu.is_cuda());
  TORCH_CHECK(rsigma.is_cuda());
  TORCH_CHECK(gamma.is_cuda());

  auto sizes = x.sizes();
  TORCH_CHECK(sizes.size() == 2);
  auto rows = sizes[0];
  auto cols = sizes[1];
  TORCH_CHECK(dz.dim() == 2);
  //    TORCH_CHECK(dz.size(1) == cols);
  auto hidden_size = gamma.numel();
  TORCH_CHECK(hidden_size == cols);

  TORCH_CHECK((hidden_size % 8 == 0) && (hidden_size <= 8192));

  TORCH_CHECK(mu.numel() == rows);
  TORCH_CHECK(mu.sizes() == rsigma.sizes());

  TORCH_CHECK(gamma.numel() == cols);

  layer_norm::LaunchParams<layer_norm::BwdParams> launch_params;
  launch_params.props = at::cuda::getCurrentDeviceProperties();
  launch_params.stream = context->cudaStream;
  TORCH_CHECK(dropout_p < 1.f);
  launch_params.params.dropout_keep_p = 1.f - dropout_p;
  launch_params.params.dresidual = nullptr;
  launch_params.params.rowscale = nullptr;
  launch_params.params.colscale = nullptr;
  launch_params.params.x0_subset = nullptr;
  launch_params.params.z_subset = nullptr;

  auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
  const int multiple =
      hidden_size <= 1536 ? 256 : (hidden_size <= 3072 ? 512 : 1024);
  auto launcher = get_bwd_launcher(wtype, itype, rtype, otype, ctype,
                                   round_multiple(hidden_size, multiple));

  launcher(launch_params, true);

  auto dgamma_part = torch::empty<dllm::CUDA>(
      IntArrayRef2D{
          static_cast<dllm::TensorIndexType>(launch_params.params.ctas_per_col),
          static_cast<dllm::TensorIndexType>(hidden_size)},
      ctype, context);
  CHECK_CUDART(cudaMemsetAsync(dgamma_part->data(), 0,
                               sizeof(float) * launch_params.barrier_size,
                               context->cudaStream));

  auto dbeta_part = torch::empty<dllm::CUDA>(
      IntArrayRef2D{
          static_cast<dllm::TensorIndexType>(launch_params.params.ctas_per_col),
          static_cast<dllm::TensorIndexType>(hidden_size)},
      ctype, context);
  CHECK_CUDART(cudaMemsetAsync(dbeta_part->data(), 0,
                               sizeof(float) * launch_params.barrier_size,
                               context->cudaStream));

  std::shared_ptr<dllm::Tensor<1>> workspace, barrier;

  layer_norm::BwdParams &params = launch_params.params;
  params.rows = rows;
  params.cols = cols;
  params.x = const_cast<void *>(x.data_ptr());
  params.x0 = nullptr;
  params.dmask = nullptr;
  params.mu = const_cast<void *>(mu.data_ptr());
  params.rs = const_cast<void *>(rsigma.data_ptr());
  params.gamma = const_cast<void *>(gamma.data_ptr());
  params.dz = const_cast<void *>(dz.data_ptr());
  params.dx = nullptr;
  params.dx0 = dx0.data_ptr();
  params.dbeta = dbeta.data_ptr();
  params.dgamma = dgamma.data_ptr();
  params.dcolscale = nullptr;
  params.dbeta_part = dbeta_part->data();
  params.dgamma_part = dgamma_part->data();
  params.dcolscale_part = nullptr;
  params.dropout_scale = 1.f / (1.f - dropout_p);
  params.inverse_cols = 1.f / float(params.cols);
  params.rowscale_const = rowscale_const;
  params.is_rms_norm = false;

  if (launch_params.barrier_size > 0) {
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
    params.barrier = barrier->data_ptr<int>();
    params.workspace = workspace->data_ptr();
  }

  launcher(launch_params, false);

  CHECK_CUDART(cudaStreamSynchronize(launch_params.stream));
}

dllm::TaskCompute backward(std::shared_ptr<dllm::Tensor<2>> dx0,
                           std::shared_ptr<dllm::Tensor<1>> dgamma,
                           std::shared_ptr<dllm::Tensor<1>> dbeta,
                           const std::shared_ptr<const dllm::Tensor<2>> dz,
                           const std::shared_ptr<const dllm::Tensor<2>> x,
                           const std::shared_ptr<const dllm::Tensor<1>> mu,
                           const std::shared_ptr<const dllm::Tensor<1>> rsigma,
                           const std::shared_ptr<const dllm::Tensor<1>> gamma) {
  auto task = dllm::TaskCompute{
      [dz = dz, x = x, mu = mu, rsigma = rsigma, gamma = gamma, dx0 = dx0,
       dgamma = dgamma, dbeta = dbeta, dx0Future = *dx0->future,
       dgammaFuture = *dgamma->future, dbetaFuture = *dbeta->future,
       dzFuture = dz->future->wFuture, xFuture = x->future->wFuture,
       muFuture = mu->future->wFuture, rsigmaFuture = rsigma->future->wFuture,
       gammaFuture = gamma->future->wFuture](
          const dllm::ContextCompute *context) mutable {
        util::FutureGuard dx0RGuard{dx0Future.rFuture};
        util::FutureGuard dx0WGuard{dx0Future.wFuture};
        util::FutureGuard dgammaRGuard{dgammaFuture.rFuture};
        util::FutureGuard dgammaWGuard{dgammaFuture.wFuture};
        util::FutureGuard dbetaRGuard{dbetaFuture.rFuture};
        util::FutureGuard dbetaWGuard{dbetaFuture.wFuture};
        util::FutureGuard dzGuard{dzFuture};
        util::FutureGuard xGuard{xFuture};
        util::FutureGuard muGuard{muFuture};
        util::FutureGuard rsigmaGuard{rsigmaFuture};
        util::FutureGuard gammaGuard{gammaFuture};
        dropout_add_ln_bwd_no_dropout(*dx0, *dgamma, *dbeta, *dz, *x, *mu,
                                      *rsigma, *gamma, context);
        CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));

        dz.reset();
        x.reset();
        mu.reset();
        rsigma.reset();
        gamma.reset();
        dx0.reset();
        dgamma.reset();
        dbeta.reset();
      }};
  const TaskFuture future = task.get_future();
  dz->future->rFuture = future;
  x->future->rFuture = future;
  mu->future->rFuture = future;
  rsigma->future->rFuture = future;
  gamma->future->rFuture = future;
  dx0->future->wFuture = future;
  dgamma->future->wFuture = future;
  dbeta->future->wFuture = future;

  return task;
}

}  // namespace dllm::flash_attn::LayerNorm
