/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

// Include these 2 headers instead of torch/extension.h since we don't need all
// of the torch headers.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cutlass/numeric_types.h>
#include <torch/nn/functional.h>
#include <torch/python.h>

#include "flash.h"
#include "static_switch.h"
#include "util.h"

using IntArrayRef1D = std::array<dllm::TensorIndexType, 1>;
using IntArrayRef2D = std::array<dllm::TensorIndexType, 2>;
using IntArrayRef3D = std::array<dllm::TensorIndexType, 3>;
using IntArrayRef4D = std::array<dllm::TensorIndexType, 4>;
using IntArrayRef5D = std::array<dllm::TensorIndexType, 5>;

#define CHECK_DEVICE(x) TORCH_CHECK((x).is_cuda(), #x " must be on CUDA")
#define CHECK_SHAPE(x, ...)                 \
  TORCH_CHECK((x).sizes() == (__VA_ARGS__), \
              #x " must have shape (" #__VA_ARGS__ ")")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

void set_params_fprop(Flash_fwd_params &params,
                      // sizes
                      const size_t b, const size_t seqlen_q,
                      const size_t seqlen_k, const size_t seqlen_q_rounded,
                      const size_t seqlen_k_rounded, const size_t h,
                      const size_t h_k, const size_t d, const size_t d_rounded,
                      // device pointers
                      const at::Tensor<4> &q, const at::Tensor<4> &k,
                      const at::Tensor<4> &v, at::Tensor<4> &out,
                      void *cu_seqlens_q_d, void *cu_seqlens_k_d,
                      void *seqused_k, void *p_d, void *softmax_lse_d,
                      float p_dropout, float softmax_scale,
                      int window_size_left, int window_size_right,
                      bool seqlenq_ngroups_swapped = false) {
  // Reset the parameters
  params = {};

  params.is_bf16 = q.dtype == torch::kBFloat16;

  // Set the pointers and strides.
  // params.q_ptr = q.data_ptr();
  // params.k_ptr = k.data_ptr();
  // params.v_ptr = v.data_ptr();
  params.q_ptr = const_cast<void *>(q.data_ptr());
  params.k_ptr = const_cast<void *>(k.data_ptr());
  params.v_ptr = const_cast<void *>(v.data_ptr());
  // All stride are in elements, not bytes.
  // params.q_row_stride = q.stride(-3);
  // params.k_row_stride = k.stride(-3);
  // params.v_row_stride = v.stride(-3);
  // params.q_head_stride = q.stride(-2);
  // params.k_head_stride = k.stride(-2);
  // params.v_head_stride = v.stride(-2);
  params.q_row_stride = q.stride<-3>();
  params.k_row_stride = k.stride<-3>();
  params.v_row_stride = v.stride<-3>();
  params.q_head_stride = q.stride<-2>();
  params.k_head_stride = k.stride<-2>();
  params.v_head_stride = v.stride<-2>();

  params.o_ptr = out.data_ptr();
  // params.o_row_stride = out.stride(-3);
  // params.o_head_stride = out.stride(-2);
  params.o_row_stride = out.stride<-3>();
  params.o_head_stride = out.stride<-2>();

  if (cu_seqlens_q_d == nullptr) {
    // params.q_batch_stride = q.stride(0);
    // params.k_batch_stride = k.stride(0);
    // params.v_batch_stride = v.stride(0);
    // params.o_batch_stride = out.stride(0);
    params.q_batch_stride = q.stride<0>();
    params.k_batch_stride = k.stride<0>();
    params.v_batch_stride = v.stride<0>();
    params.o_batch_stride = out.stride<0>();
    if (seqlenq_ngroups_swapped) {
      params.q_batch_stride *= seqlen_q;
      params.o_batch_stride *= seqlen_q;
    }
  }

  params.cu_seqlens_q = static_cast<int *>(cu_seqlens_q_d);
  params.cu_seqlens_k = static_cast<int *>(cu_seqlens_k_d);
  params.seqused_k = static_cast<int *>(seqused_k);

  // P = softmax(QK^T)
  params.p_ptr = p_d;

  // Softmax sum
  params.softmax_lse_ptr = softmax_lse_d;

  // Set the dimensions.
  params.b = b;
  params.h = h;
  params.h_k = h_k;
  params.h_h_k_ratio = h / h_k;
  params.seqlen_q = seqlen_q;
  params.seqlen_k = seqlen_k;
  params.seqlen_q_rounded = seqlen_q_rounded;
  params.seqlen_k_rounded = seqlen_k_rounded;
  params.d = d;
  params.d_rounded = d_rounded;

  // Set the different scale values.
  params.scale_softmax = softmax_scale;
  params.scale_softmax_log2 = softmax_scale * M_LOG2E;

  // Set this to probability of keeping an element to simplify things.
  params.p_dropout = 1.f - p_dropout;
  // Convert p from float to int so we don't have to convert the random uint to
  // float to compare. [Minor] We want to round down since when we do the
  // comparison we use <= instead of < params.p_dropout_in_uint =
  // uint32_t(std::floor(params.p_dropout * 4294967295.0));
  // params.p_dropout_in_uint16_t = uint16_t(std::floor(params.p_dropout *
  // 65535.0));
  params.p_dropout_in_uint8_t = uint8_t(std::floor(params.p_dropout * 255.0));
  params.rp_dropout = 1.f / params.p_dropout;
  params.scale_softmax_rp_dropout = params.rp_dropout * params.scale_softmax;
  TORCH_CHECK(p_dropout < 1.f);
#ifdef FLASHATTENTION_DISABLE_DROPOUT
  TORCH_CHECK(p_dropout == 0.0f,
              "This flash attention build does not support dropout.");
#endif

  // Causal is the special case where window_size_right == 0 and
  // window_size_left < 0. Local is the more general case where
  // window_size_right >= 0 or window_size_left >= 0.
  params.is_causal = window_size_left < 0 && window_size_right == 0;

  if (window_size_left < 0 && window_size_right >= 0) {
    window_size_left = seqlen_k;
  }
  if (window_size_left >= 0 && window_size_right < 0) {
    window_size_right = seqlen_k;
  }
  params.window_size_left = window_size_left;
  params.window_size_right = window_size_right;

#ifdef FLASHATTENTION_DISABLE_LOCAL
  TORCH_CHECK(
      params.is_causal || (window_size_left < 0 && window_size_right < 0),
      "This flash attention build does not support local attention.");
#endif

  params.is_seqlens_k_cumulative = true;

#ifdef FLASHATTENTION_DISABLE_UNEVEN_K
  TORCH_CHECK(d == d_rounded,
              "This flash attention build does not support headdim not being a "
              "multiple of 32.");
#endif
}

void set_params_dgrad(Flash_bwd_params &params,
                      // sizes
                      const size_t b, const size_t seqlen_q,
                      const size_t seqlen_k, const size_t seqlen_q_rounded,
                      const size_t seqlen_k_rounded, const size_t h,
                      const size_t h_k, const size_t d, const size_t d_rounded,
                      // device pointers
                      const at::Tensor<4> &q, const at::Tensor<4> &k,
                      const at::Tensor<4> &v, at::Tensor<4> &out,
                      at::Tensor<4> &dout, at::Tensor<4> &dq, at::Tensor<4> &dk,
                      at::Tensor<4> &dv, void *cu_seqlens_q_d,
                      void *cu_seqlens_k_d, void *dq_accum_d, void *dk_accum_d,
                      void *dv_accum_d, void *softmax_lse_d,
                      void *dsoftmax_sum_d, float p_dropout,
                      float softmax_scale, int window_size_left,
                      int window_size_right, bool deterministic) {
  set_params_fprop(params, b, seqlen_q, seqlen_k, seqlen_q_rounded,
                   seqlen_k_rounded, h, h_k, d, d_rounded, q, k, v, out,
                   cu_seqlens_q_d, cu_seqlens_k_d, nullptr, nullptr,
                   softmax_lse_d, p_dropout, softmax_scale, window_size_left,
                   window_size_right);

  // Set the pointers and strides.
  params.do_ptr = dout.data_ptr();
  // params.do_row_stride = dout.stride(-3);
  // params.do_head_stride = dout.stride(-2);
  params.do_row_stride = dout.stride<-3>();
  params.do_head_stride = dout.stride<-2>();
  params.dq_ptr = dq.data_ptr();
  params.dk_ptr = dk.data_ptr();
  params.dv_ptr = dv.data_ptr();
  // params.dq_row_stride = dq.stride(-3);
  // params.dk_row_stride = dk.stride(-3);
  // params.dv_row_stride = dv.stride(-3);
  // params.dq_head_stride = dq.stride(-2);
  // params.dk_head_stride = dk.stride(-2);
  // params.dv_head_stride = dv.stride(-2);
  params.dq_row_stride = dq.stride<-3>();
  params.dk_row_stride = dk.stride<-3>();
  params.dv_row_stride = dv.stride<-3>();
  params.dq_head_stride = dq.stride<-2>();
  params.dk_head_stride = dk.stride<-2>();
  params.dv_head_stride = dv.stride<-2>();

  if (cu_seqlens_q_d == nullptr) {
    // params.do_batch_stride = dout.stride(0);
    // params.dq_batch_stride = dq.stride(0);
    // params.dk_batch_stride = dk.stride(0);
    // params.dv_batch_stride = dv.stride(0);
    params.do_batch_stride = dout.stride<0>();
    params.dq_batch_stride = dq.stride<0>();
    params.dk_batch_stride = dk.stride<0>();
    params.dv_batch_stride = dv.stride<0>();
  }

  params.dq_accum_ptr = dq_accum_d;
  params.dk_accum_ptr = dk_accum_d;
  params.dv_accum_ptr = dv_accum_d;

  // Softmax sum
  params.dsoftmax_sum = dsoftmax_sum_d;

  params.deterministic = deterministic;
}

void run_mha_fwd(Flash_fwd_params &params, cudaStream_t stream,
                 bool force_split_kernel = false) {
  FP16_SWITCH(!params.is_bf16, [&] {
    HEADDIM_SWITCH(params.d, [&] {
      if (params.num_splits <= 1 &&
          !force_split_kernel) {  // If we don't set it num_splits == 0
        run_mha_fwd_<elem_type, kHeadDim>(params, stream);
      } else {
        run_mha_fwd_splitkv_dispatch<elem_type, kHeadDim>(params, stream);
      }
    });
  });
}

// Find the number of splits that maximizes the occupancy. For example, if we
// have batch * n_heads = 48 and we have 108 SMs, having 2 splits (efficiency =
// 0.89) is better than having 3 splits (efficiency = 0.67). However, we also
// don't want too many splits as that would incur more HBM reads/writes. So we
// find the best efficiency, then find the smallest number of splits that gets
// 85% of the best efficiency.
inline int num_splits_heuristic(int batch_nheads_mblocks, int num_SMs,
                                int num_n_blocks, int max_splits) {
  // If we have enough to almost fill the SMs, then just use 1 split
  if (batch_nheads_mblocks >= 0.8f * num_SMs) {
    return 1;
  }
  max_splits = std::min({max_splits, num_SMs, num_n_blocks});
  float max_efficiency = 0.f;
  std::vector<float> efficiency;
  efficiency.reserve(max_splits);
  auto ceildiv = [](int a, int b) { return (a + b - 1) / b; };
  // Some splits are not eligible. For example, if we have 64 blocks and choose
  // 11 splits, we'll have 6 * 10 + 4 blocks. If we choose 12 splits, we'll have
  // 6 * 11 + (-2) blocks (i.e. it's 11 splits anyway). So we check if the
  // number of blocks per split is the same as the previous num_splits.
  auto is_split_eligible = [&ceildiv, &num_n_blocks](int num_splits) {
    return num_splits == 1 || ceildiv(num_n_blocks, num_splits) !=
                                  ceildiv(num_n_blocks, num_splits - 1);
  };
  for (int num_splits = 1; num_splits <= max_splits; num_splits++) {
    if (!is_split_eligible(num_splits)) {
      efficiency.push_back(0.f);
    } else {
      float n_waves = float(batch_nheads_mblocks * num_splits) / num_SMs;
      float eff = n_waves / ceil(n_waves);
      // printf("num_splits = %d, eff = %f\n", num_splits, eff);
      if (eff > max_efficiency) {
        max_efficiency = eff;
      }
      efficiency.push_back(eff);
    }
  }
  for (int num_splits = 1; num_splits <= max_splits; num_splits++) {
    if (!is_split_eligible(num_splits)) {
      continue;
    }
    if (efficiency[num_splits - 1] >= 0.85 * max_efficiency) {
      // printf("num_splits chosen = %d\n", num_splits);
      return num_splits;
    }
  }
  return 1;
}

void set_params_splitkv(const dllm::ContextCompute *context,
                        Flash_fwd_params &params, const int batch_size,
                        const int num_heads, const int head_size,
                        const int max_seqlen_k, const int max_seqlen_q,
                        const int head_size_rounded, const float p_dropout,
                        const int num_splits, cudaDeviceProp *dprops
                        // , struct c10::TensorOptions opts) {
) {
  // This needs to match with run_mha_fwd_splitkv_dispatch
  const int block_n = head_size <= 64 ? 256 : (head_size <= 128 ? 128 : 64);
  const int num_n_blocks = (max_seqlen_k + block_n - 1) / block_n;
  // Technically kBlockM = 64 only for the splitKV kernels, not the standard
  // kernel. In any case we don't expect seqlen_q to be larger than 64 for
  // inference.
  const int num_m_blocks = (max_seqlen_q + 64 - 1) / 64;
  params.num_splits = num_splits;
  if (p_dropout == 0.0f) {  // SplitKV is not implemented for dropout
    if (num_splits < 1) {
      // We multiply number of SMs by 2 to hard-code the fact that we're using
      // 128 threads per block.
      params.num_splits = num_splits_heuristic(
          batch_size * num_heads * num_m_blocks,
          dprops->multiProcessorCount * 2, num_n_blocks, 128);
    }
    if (params.num_splits > 1) {
      // at::Tensor softmax_lse_accum =
      //     torch::empty({params.num_splits, batch_size, num_heads,
      //     max_seqlen_q},
      //                  opts.dtype(at::kFloat));
      // at::Tensor out_accum =
      //     torch::empty({params.num_splits, batch_size, num_heads,
      //     max_seqlen_q,
      //                   head_size_rounded},
      //                  opts.dtype(at::kFloat));
      // params.softmax_lseaccum_ptr = softmax_lse_accum.data_ptr();
      // params.oaccum_ptr = out_accum.data_ptr();
      auto softmax_lse_accum = torch::empty<dllm::CUDA>(
          IntArrayRef5D{params.num_splits, batch_size, num_heads, max_seqlen_q},
          at::kFloat, context);
      auto out_accum = torch::empty<dllm::CUDA>(
          IntArrayRef5D{params.num_splits, batch_size, num_heads, max_seqlen_q,
                        head_size_rounded},
          at::kFloat, context);
      params.softmax_lseaccum_ptr = softmax_lse_accum->data_ptr();
      params.oaccum_ptr = out_accum->data_ptr();
    }
    TORCH_CHECK(params.num_splits <= 128, "num_splits > 128 not supported");
  }
}

void set_params_alibi(Flash_fwd_params &params,
                      c10::optional<at::Tensor<1>> &alibi_slopes_,
                      int batch_size, int num_heads) {
#ifdef FLASHATTENTION_DISABLE_ALIBI
  TORCH_CHECK(!alibi_slopes_.has_value(),
              "This flash attention build does not support alibi.");
  params.alibi_slopes_ptr = nullptr;
#else
  if (alibi_slopes_.has_value()) {
    auto alibi_slopes = alibi_slopes_.value();
    TORCH_CHECK(alibi_slopes.dtype == torch::kFloat32,
                "ALiBi slopes must have dtype fp32");
    CHECK_DEVICE(alibi_slopes);
    // TORCH_CHECK(alibi_slopes.stride(-1) == 1,
    TORCH_CHECK(alibi_slopes.stride<-1>() == 1,
                "ALiBi slopes tensor must have contiguous last dimension");
    TORCH_CHECK(alibi_slopes.sizes() == IntArrayRef1D({num_heads}));
    params.alibi_slopes_ptr = alibi_slopes.data_ptr();
    // params.alibi_slopes_batch_stride =
    //     alibi_slopes.dim() == 2 ? alibi_slopes.stride(0) : 0;
    params.alibi_slopes_batch_stride =
        alibi_slopes.dim() == 2 ? alibi_slopes.stride<0>() : 0;
  } else {
    params.alibi_slopes_ptr = nullptr;
  }
#endif
}

void set_params_alibi(Flash_fwd_params &params,
                      c10::optional<at::Tensor<2>> &alibi_slopes_,
                      int batch_size, int num_heads) {
#ifdef FLASHATTENTION_DISABLE_ALIBI
  TORCH_CHECK(!alibi_slopes_.has_value(),
              "This flash attention build does not support alibi.");
  params.alibi_slopes_ptr = nullptr;
#else
  if (alibi_slopes_.has_value()) {
    auto alibi_slopes = alibi_slopes_.value();
    TORCH_CHECK(alibi_slopes.dtype == torch::kFloat32,
                "ALiBi slopes must have dtype fp32");
    CHECK_DEVICE(alibi_slopes);
    // TORCH_CHECK(alibi_slopes.stride(-1) == 1,
    TORCH_CHECK(alibi_slopes.stride<-1>() == 1,
                "ALiBi slopes tensor must have contiguous last dimension");
    TORCH_CHECK(alibi_slopes.sizes() == IntArrayRef2D({batch_size, num_heads}));
    params.alibi_slopes_ptr = alibi_slopes.data_ptr();
    // params.alibi_slopes_batch_stride =
    //     alibi_slopes.dim() == 2 ? alibi_slopes.stride(0) : 0;
    params.alibi_slopes_batch_stride =
        alibi_slopes.dim() == 2 ? alibi_slopes.stride<0>() : 0;
  } else {
    params.alibi_slopes_ptr = nullptr;
  }
#endif
}

namespace dllm::compute::FlashAttention {
void forward(
    const dllm::ContextCompute *context,
    at::Tensor<1> &random_state,  // 2 seed
    at::Tensor<4> &out,  // batch_size x seqlen_q x num_heads x head_size
    at::Tensor<3> &softmax_lse,  // batch_size, num_heads, seqlen_q
    const at::Tensor<4> &q,  // batch_size x seqlen_q x num_heads x head_size
    const at::Tensor<4> &k,  // batch_size x seqlen_k x num_heads_k x head_size
    const at::Tensor<4> &v,  // batch_size x seqlen_k x num_heads_k x head_size
    const double p_dropout, const double softmax_scale) {
  bool is_causal = true;
  int window_size_left = -1;
  int window_size_right = -1;
  auto dprops = at::cuda::getCurrentDeviceProperties();
  // bool is_sm75 = dprops->major == 7 && dprops->minor == 5;
  bool is_sm8x = dprops->major == 8 && dprops->minor >= 0;
  bool is_sm90 = dprops->major == 9 && dprops->minor == 0;
  TORCH_CHECK(is_sm90 || is_sm8x,
              "FlashAttention only supports Ampere GPUs or newer.");
  // We will support Turing in the near future
  // TORCH_CHECK(is_sm90 || is_sm8x || is_sm75, "FlashAttention only supports
  // Turing GPUs or newer.");

  auto q_dtype = q.dtype;
  TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16,
              "FlashAttention only support fp16 and bf16 data type");
  if (q_dtype == torch::kBFloat16) {
    TORCH_CHECK(is_sm90 || is_sm8x,
                "bfloat16 is only supported on Ampere GPUs or newer");
  }
  TORCH_CHECK(k.dtype == q_dtype, "query and key must have the same dtype");
  TORCH_CHECK(v.dtype == q_dtype, "query and value must have the same dtype");

  CHECK_DEVICE(q);
  CHECK_DEVICE(k);
  CHECK_DEVICE(v);

  // TORCH_CHECK(q.stride(-1) == 1,
  //             "Input tensor must have contiguous last dimension");
  // TORCH_CHECK(k.stride(-1) == 1,
  //             "Input tensor must have contiguous last dimension");
  // TORCH_CHECK(v.stride(-1) == 1,
  //             "Input tensor must have contiguous last dimension");
  TORCH_CHECK(q.stride<-1>() == 1,
              "Input tensor must have contiguous last dimension");
  TORCH_CHECK(k.stride<-1>() == 1,
              "Input tensor must have contiguous last dimension");
  TORCH_CHECK(v.stride<-1>() == 1,
              "Input tensor must have contiguous last dimension");

  const auto sizes = q.sizes();

  const int batch_size = sizes[0];
  int seqlen_q = sizes[1];
  int num_heads = sizes[2];
  const int head_size_og = sizes[3];
  // TODO(Jie): we assume head_size_og is dividable by 8
  if (head_size_og % 8 != 0) {
    exit(1);
  }
  // TODO(Jie): we assume head_size_og is dividable by 8
  // const int seqlen_k = k.size(1);
  // const int num_heads_k = k.size(2);
  const int seqlen_k = k.size<1>();
  const int num_heads_k = k.size<2>();
  TORCH_CHECK(batch_size > 0, "batch size must be postive");
  TORCH_CHECK(
      head_size_og <= 256,
      "FlashAttention forward only supports head dimension at most 256");
  TORCH_CHECK(
      num_heads % num_heads_k == 0,
      "Number of heads in key/value must divide number of heads in query");

  if (window_size_left >= seqlen_k) {
    window_size_left = -1;
  }
  if (window_size_right >= seqlen_k) {
    window_size_right = -1;
  }

  // causal=true is the same as causal=false in this case
  if (seqlen_q == 1) {
    is_causal = false;
  }
  //  if (seqlen_q == 1 && !alibi_slopes_.has_value()) {
  //    is_causal = false;
  //  }
  if (is_causal) {
    window_size_right = 0;
  }

  // Faster to transpose q from (b, 1, (nheads_kv ngroups), d) to (b, ngroups,
  // nheads_kv, d) in this case H/t Daniel Haziza
  // TODO(Jie): assume seqlen_q != 1 now
  if (seqlen_q == 1) {
    exit(1);
  }
  // TODO(Jie): assume seqlen_q != 1 now

  //  const int seqlenq_ngroups_swapped =
  //      seqlen_q == 1 && num_heads > num_heads_k && window_size_left < 0 &&
  //      window_size_right < 0 && p_dropout == 0.f && head_size_og % 8 == 0 &&
  //      !alibi_slopes_.has_value();
  //  const int ngroups = num_heads / num_heads_k;
  //  if (seqlenq_ngroups_swapped) {
  //    q = q.reshape({batch_size, num_heads_k, ngroups, head_size_og})
  //            .transpose(1, 2);
  //    seqlen_q = ngroups;
  //    num_heads = num_heads_k;
  //  }

  CHECK_SHAPE(q, IntArrayRef4D{batch_size, seqlen_q, num_heads, head_size_og});
  CHECK_SHAPE(k,
              IntArrayRef4D{batch_size, seqlen_k, num_heads_k, head_size_og});
  CHECK_SHAPE(v,
              IntArrayRef4D{batch_size, seqlen_k, num_heads_k, head_size_og});

  // at::Tensor q_padded, k_padded, v_padded;
  // if (head_size_og % 8 != 0) {
  //   q_padded = torch::nn::functional::pad(
  //       q, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
  //   k_padded = torch::nn::functional::pad(
  //       k, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
  //   v_padded = torch::nn::functional::pad(
  //       v, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
  // } else {
  //   q_padded = q;
  //   k_padded = k;
  //   v_padded = v;
  // }

  // TODO(Jie): we assume head_size_og is dividable by 8
  if (head_size_og % 8 != 0) {
    exit(1);
  }
  //  auto [q_padded, k_padded, v_padded] = [&]() {
  //    auto q_padded = q;
  //    auto kcache_padded = k;
  //    auto vcache_padded = v;
  //    return std::make_tuple(q_padded, kcache_padded, vcache_padded);
  //  }();
  // TODO(Jie): we assume head_size_og is dividable by 8

  // at::Tensor out;
  // if (out_.has_value()) {
  //   out = out_.value();
  //   TORCH_CHECK(out.dtype() == q_dtype,
  //               "Output must have the same dtype as inputs");
  //   CHECK_DEVICE(out);
  //   TORCH_CHECK(out.stride(-1) == 1,
  //               "Output tensor must have contiguous last dimension");
  //   CHECK_SHAPE(out, batch_size, sizes[1], sizes[2], head_size_og);
  //   if (seqlenq_ngroups_swapped) {
  //     out = out.reshape({batch_size, num_heads_k, ngroups, head_size_og})
  //               .transpose(1, 2);
  //   }
  //   if (head_size_og % 8 != 0) {
  //     out = torch::empty_like(q_padded);
  //   }
  // } else {
  //   out = torch::empty_like(q_padded);
  // }

  auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
  const int head_size = round_multiple(head_size_og, 8);
  const int head_size_rounded = round_multiple(head_size, 32);
  const int seqlen_q_rounded = round_multiple(seqlen_q, 128);
  const int seqlen_k_rounded = round_multiple(seqlen_k, 128);

  // Otherwise the kernel will be launched from cuda:0 device
  // Cast to char to avoid compiler warning about narrowing
  // at::cuda::CUDAGuard device_guard{(char)q.get_device()};

  // auto opts = q.options();

  // auto softmax_lse =
  //     torch::empty({batch_size, num_heads, seqlen_q},
  //     opts.dtype(at::kFloat));
  //  auto softmax_lse = torch::empty<dllm::CUDA>(
  //      IntArrayRef3D{batch_size, num_heads, seqlen_q}, at::kFloat, context);
  // at::Tensor p;
  // Only return softmax if there's dropout to reduce compilation time
  // if (return_softmax) {
  //   TORCH_CHECK(p_dropout > 0.0f,
  //               "return_softmax is only supported when p_dropout > 0.0");
  //   p = torch::empty(
  //       {batch_size, num_heads, seqlen_q_rounded, seqlen_k_rounded}, opts);
  // }
  //  auto p = [&]() {
  //    // Only return softmax if there's dropout to reduce compilation time
  //    if (return_softmax) {
  //      TORCH_CHECK(p_dropout > 0.0f,
  //                  "return_softmax is only supported when p_dropout > 0.0");
  //      return torch::empty<dllm::CUDA>(
  //          IntArrayRef4D{batch_size, num_heads, seqlen_q_rounded,
  //                        seqlen_k_rounded},
  //          q.dtype, context);
  //    }
  //    return decltype(torch::empty<dllm::CUDA>(
  //        IntArrayRef4D{batch_size, num_heads, seqlen_q_rounded,
  //                      seqlen_k_rounded},
  //        q.dtype, context)){};
  //  }();

  Flash_fwd_params params;
  set_params_fprop(params, batch_size, seqlen_q, seqlen_k, seqlen_q_rounded,
                   seqlen_k_rounded, num_heads, num_heads_k, head_size,
                   head_size_rounded, q, k, v, out,
                   /*cu_seqlens_q_d=*/nullptr,
                   /*cu_seqlens_k_d=*/nullptr,
                   /*seqused_k=*/nullptr, nullptr, softmax_lse.data_ptr(),
                   p_dropout, softmax_scale, window_size_left,
                   window_size_right);

  // set_params_splitkv(params, batch_size, num_heads, head_size, seqlen_k,
  //                    seqlen_q, head_size_rounded, p_dropout, /*num_splits*/
  //                    0, dprops, opts);
  set_params_splitkv(context, params, batch_size, num_heads, head_size,
                     seqlen_k, seqlen_q, head_size_rounded, p_dropout,
                     /*num_splits*/ 0, dprops);

  // number of times random will be generated per thread, to offset philox
  // counter in thc random state We use a custom RNG that increases the offset
  // by batch_size * nheads * 32.
  int64_t counter_offset = params.b * params.h * 32;
  // auto options =
  //     torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
  // auto rng_state = torch::empty({2}, options.dtype(torch::kInt64));
  // Forward kernel will populate memory with the seed and offset.
  params.rng_state = reinterpret_cast<uint64_t *>(random_state.data_ptr());

  if (p_dropout > 0.0) {
    // auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
    //     gen_, at::cuda::detail::getDefaultCUDAGenerator());
    // See Note [Acquire lock when using random generators]
    // std::lock_guard<std::mutex> lock(gen->mutex_);
    // params.philox_args = gen->philox_cuda_state(counter_offset);
    params.philox_args = {context->curandSeed, context->curandOffset.load()};
    context->curandOffset += counter_offset;
  }

  // set_params_alibi(params, alibi_slopes_, batch_size, num_heads);
  params.alibi_slopes_ptr = nullptr;

  // TODO(Jie): We assume seqlen_k > 0 is true
  if (!(seqlen_k > 0)) {
    exit(1);
  }
  // TODO(Jie): We assume seqlen_k > 0 is true
  if (seqlen_k > 0) {
    // auto stream = at::cuda::getCurrentCUDAStream().stream();
    auto stream = context->cudaStream;
    run_mha_fwd(params, stream);
  } else {
    // If seqlen_k == 0, then we have an empty tensor. We need to set the output
    // to 0.
    // out.zero_();
    // softmax_lse.fill_(std::numeric_limits<float>::infinity());
  }

  // auto out_padded = out;
  //  if (head_size_og % 8 != 0) {
  //    out = out.index(
  //        {"...", torch::indexing::Slice(torch::indexing::None,
  //        head_size_og)});
  //    if (out_.has_value()) {
  //      // out_.value().copy_(out);
  //      CHECK_CUDART(
  //          cudaMemcpyAsync(out_.value()->data(), out->data(),
  //                          dllm::toByte(out->dtype) *
  //                          cute::size(out->layout), cudaMemcpyDeviceToDevice,
  //                          context->cudaStream));
  //    }
  //  }

  //  if (seqlenq_ngroups_swapped) {
  //    out = out.transpose(1, 2).reshape(
  //        {batch_size, 1, num_heads_k * seqlen_q, head_size_og});
  //    out_padded = out_padded.transpose(1, 2).reshape(
  //        {batch_size, 1, num_heads_k * seqlen_q, head_size_og});
  //    q_padded = q_padded.transpose(1, 2).reshape(
  //        {batch_size, 1, num_heads_k * seqlen_q, head_size_og});
  //    softmax_lse = softmax_lse.reshape({batch_size, num_heads_k * seqlen_q,
  //    1});
  //  }
  return;
}

TaskCompute forward(const std::shared_ptr<dllm::Tensor<1>> &random_state,
                    const std::shared_ptr<dllm::Tensor<4>> &out,
                    const std::shared_ptr<dllm::Tensor<3>> &softmax_lse,
                    const std::shared_ptr<const dllm::Tensor<4>> &q,
                    const std::shared_ptr<const dllm::Tensor<4>> &k,
                    const std::shared_ptr<const dllm::Tensor<4>> &v,
                    const double drouput_p, const double softmax_scale) {
  if (!(out->layout.shape() == q->layout.shape()) ||
      !(k->layout.shape() == v->layout.shape())) {
  }
  auto task = TaskCompute{
      [=, futureRandom = *random_state->future, futureOut = *out->future,
       futureSoftmax = *softmax_lse->future, futureQ = *q->future,
       futureK = *k->future,
       futureV = *v->future](const ContextCompute *context) {
        util::waitFutureIfValid(futureRandom);
        util::waitFutureIfValid(futureOut);
        util::waitFutureIfValid(futureSoftmax);
        util::waitFutureIfValid(futureQ);
        util::waitFutureIfValid(futureK);
        util::waitFutureIfValid(futureV);
        forward(context, *random_state, *out, *softmax_lse, *q, *k, *v,
                drouput_p, softmax_scale);
        CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
      }};
  const auto &future = *random_state->future = task.get_future();
  *out->future = future;
  *softmax_lse->future = future;
  *q->future = future;
  *k->future = future;
  *v->future = future;
  return task;
}
}  // namespace dllm::compute::FlashAttention

// auto mha_fwd(
//     const dllm::ContextCompute *context,
//     at::Tensor<4> &q,        // batch_size x seqlen_q x num_heads x head_size
//     const at::Tensor<4> &k,  // batch_size x seqlen_k x num_heads_k x
//     head_size const at::Tensor<4> &v,  // batch_size x seqlen_k x num_heads_k
//     x head_size c10::optional<std::shared_ptr<at::Tensor<4>>>
//         &out_,  // batch_size x seqlen_q x num_heads x head_size
//     c10::optional<at::Tensor<1>>
//         &alibi_slopes_,  // num_heads or batch_size x num_heads
//     const float p_dropout, const float softmax_scale, bool is_causal,
//     int window_size_left, int window_size_right, const bool return_softmax) {
//   auto dprops = at::cuda::getCurrentDeviceProperties();
//   // bool is_sm75 = dprops->major == 7 && dprops->minor == 5;
//   bool is_sm8x = dprops->major == 8 && dprops->minor >= 0;
//   bool is_sm90 = dprops->major == 9 && dprops->minor == 0;
//   TORCH_CHECK(is_sm90 || is_sm8x,
//               "FlashAttention only supports Ampere GPUs or newer.");
//   // We will support Turing in the near future
//   // TORCH_CHECK(is_sm90 || is_sm8x || is_sm75, "FlashAttention only supports
//   // Turing GPUs or newer.");
//
//   auto q_dtype = q.dtype;
//   TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16,
//               "FlashAttention only support fp16 and bf16 data type");
//   if (q_dtype == torch::kBFloat16) {
//     TORCH_CHECK(is_sm90 || is_sm8x,
//                 "bfloat16 is only supported on Ampere GPUs or newer");
//   }
//   TORCH_CHECK(k.dtype == q_dtype, "query and key must have the same dtype");
//   TORCH_CHECK(v.dtype == q_dtype, "query and value must have the same
//   dtype");
//
//   CHECK_DEVICE(q);
//   CHECK_DEVICE(k);
//   CHECK_DEVICE(v);
//
//   // TORCH_CHECK(q.stride(-1) == 1,
//   //             "Input tensor must have contiguous last dimension");
//   // TORCH_CHECK(k.stride(-1) == 1,
//   //             "Input tensor must have contiguous last dimension");
//   // TORCH_CHECK(v.stride(-1) == 1,
//   //             "Input tensor must have contiguous last dimension");
//   TORCH_CHECK(q.stride<-1>() == 1,
//               "Input tensor must have contiguous last dimension");
//   TORCH_CHECK(k.stride<-1>() == 1,
//               "Input tensor must have contiguous last dimension");
//   TORCH_CHECK(v.stride<-1>() == 1,
//               "Input tensor must have contiguous last dimension");
//
//   const auto sizes = q.sizes();
//
//   const int batch_size = sizes[0];
//   int seqlen_q = sizes[1];
//   int num_heads = sizes[2];
//   const int head_size_og = sizes[3];
//   // TODO(Jie): we assume head_size_og is dividable by 8
//   if (head_size_og % 8 != 0) {
//     exit(1);
//   }
//   // TODO(Jie): we assume head_size_og is dividable by 8
//   // const int seqlen_k = k.size(1);
//   // const int num_heads_k = k.size(2);
//   const int seqlen_k = k.size<1>();
//   const int num_heads_k = k.size<2>();
//   TORCH_CHECK(batch_size > 0, "batch size must be postive");
//   TORCH_CHECK(
//       head_size_og <= 256,
//       "FlashAttention forward only supports head dimension at most 256");
//   TORCH_CHECK(
//       num_heads % num_heads_k == 0,
//       "Number of heads in key/value must divide number of heads in query");
//
//   if (window_size_left >= seqlen_k) {
//     window_size_left = -1;
//   }
//   if (window_size_right >= seqlen_k) {
//     window_size_right = -1;
//   }
//
//   // causal=true is the same as causal=false in this case
//   if (seqlen_q == 1 && !alibi_slopes_.has_value()) {
//     is_causal = false;
//   }
//   if (is_causal) {
//     window_size_right = 0;
//   }
//
//   // Faster to transpose q from (b, 1, (nheads_kv ngroups), d) to (b,
//   ngroups,
//   // nheads_kv, d) in this case H/t Daniel Haziza
//   // TODO(Jie): assume seqlen_q != 1 now
//   if (seqlen_q == 1) {
//     exit(1);
//   }
//   // TODO(Jie): assume seqlen_q != 1 now
//   const int seqlenq_ngroups_swapped =
//       seqlen_q == 1 && num_heads > num_heads_k && window_size_left < 0 &&
//       window_size_right < 0 && p_dropout == 0.f && head_size_og % 8 == 0 &&
//       !alibi_slopes_.has_value();
//   const int ngroups = num_heads / num_heads_k;
//   if (seqlenq_ngroups_swapped) {
//     q = q.reshape({batch_size, num_heads_k, ngroups, head_size_og})
//             .transpose(1, 2);
//     seqlen_q = ngroups;
//     num_heads = num_heads_k;
//   }
//
//   CHECK_SHAPE(q, IntArrayRef4D{batch_size, seqlen_q, num_heads,
//   head_size_og}); CHECK_SHAPE(k,
//               IntArrayRef4D{batch_size, seqlen_k, num_heads_k,
//               head_size_og});
//   CHECK_SHAPE(v,
//               IntArrayRef4D{batch_size, seqlen_k, num_heads_k,
//               head_size_og});
//
//   // at::Tensor q_padded, k_padded, v_padded;
//   // if (head_size_og % 8 != 0) {
//   //   q_padded = torch::nn::functional::pad(
//   //       q, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og %
//   8}));
//   //   k_padded = torch::nn::functional::pad(
//   //       k, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og %
//   8}));
//   //   v_padded = torch::nn::functional::pad(
//   //       v, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og %
//   8}));
//   // } else {
//   //   q_padded = q;
//   //   k_padded = k;
//   //   v_padded = v;
//   // }
//   auto [q_padded, k_padded, v_padded] = [&]() {
//     if (head_size_og % 8 != 0) {
//       auto q_padded = torch::nn::functional::pad(
//           q, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og %
//           8}));
//       auto kcache_padded = torch::nn::functional::pad(
//           k, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og %
//           8}));
//       auto vcache_padded = torch::nn::functional::pad(
//           v, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og %
//           8}));
//       return std::make_tuple(q_padded, kcache_padded, vcache_padded);
//     } else {
//       auto q_padded = q;
//       auto kcache_padded = k;
//       auto vcache_padded = v;
//       return std::make_tuple(q_padded, kcache_padded, vcache_padded);
//     }
//   }();
//
//   // at::Tensor out;
//   // if (out_.has_value()) {
//   //   out = out_.value();
//   //   TORCH_CHECK(out.dtype() == q_dtype,
//   //               "Output must have the same dtype as inputs");
//   //   CHECK_DEVICE(out);
//   //   TORCH_CHECK(out.stride(-1) == 1,
//   //               "Output tensor must have contiguous last dimension");
//   //   CHECK_SHAPE(out, batch_size, sizes[1], sizes[2], head_size_og);
//   //   if (seqlenq_ngroups_swapped) {
//   //     out = out.reshape({batch_size, num_heads_k, ngroups, head_size_og})
//   //               .transpose(1, 2);
//   //   }
//   //   if (head_size_og % 8 != 0) {
//   //     out = torch::empty_like(q_padded);
//   //   }
//   // } else {
//   //   out = torch::empty_like(q_padded);
//   // }
//   auto out = [&]() {
//     if (out_.has_value()) {
//       auto out = out_.value();
//       TORCH_CHECK(out->dtype == q_dtype,
//                   "Output must have the same dtype as inputs");
//       CHECK_DEVICE(*out);
//       TORCH_CHECK(out->stride<-1>() == 1,
//                   "Output tensor must have contiguous last dimension");
//       CHECK_SHAPE(*out,
//                   IntArrayRef4D{batch_size, sizes[1], sizes[2],
//                   head_size_og});
//       if (seqlenq_ngroups_swapped) {
//         return out.reshape({batch_size, num_heads_k, ngroups, head_size_og})
//             .transpose(1, 2);
//       }
//       if (head_size_og % 8 != 0) {
//         return torch::empty_like<dllm::CUDA>(q_padded, context);
//       }
//       return out;
//     } else {
//       return torch::empty_like<dllm::CUDA>(q_padded, context);
//     }
//   }();
//
//   auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
//   const int head_size = round_multiple(head_size_og, 8);
//   const int head_size_rounded = round_multiple(head_size, 32);
//   const int seqlen_q_rounded = round_multiple(seqlen_q, 128);
//   const int seqlen_k_rounded = round_multiple(seqlen_k, 128);
//
//   // Otherwise the kernel will be launched from cuda:0 device
//   // Cast to char to avoid compiler warning about narrowing
//   // at::cuda::CUDAGuard device_guard{(char)q.get_device()};
//
//   // auto opts = q.options();
//
//   // auto softmax_lse =
//   //     torch::empty({batch_size, num_heads, seqlen_q},
//   //     opts.dtype(at::kFloat));
//   auto softmax_lse = torch::empty<dllm::CUDA>(
//       IntArrayRef3D{batch_size, num_heads, seqlen_q}, at::kFloat, context);
//   // at::Tensor p;
//   // Only return softmax if there's dropout to reduce compilation time
//   // if (return_softmax) {
//   //   TORCH_CHECK(p_dropout > 0.0f,
//   //               "return_softmax is only supported when p_dropout > 0.0");
//   //   p = torch::empty(
//   //       {batch_size, num_heads, seqlen_q_rounded, seqlen_k_rounded},
//   opts);
//   // }
//   auto p = [&]() {
//     // Only return softmax if there's dropout to reduce compilation time
//     if (return_softmax) {
//       TORCH_CHECK(p_dropout > 0.0f,
//                   "return_softmax is only supported when p_dropout > 0.0");
//       return torch::empty<dllm::CUDA>(
//           IntArrayRef4D{batch_size, num_heads, seqlen_q_rounded,
//                         seqlen_k_rounded},
//           q.dtype, context);
//     }
//     return decltype(torch::empty<dllm::CUDA>(
//         IntArrayRef4D{batch_size, num_heads, seqlen_q_rounded,
//                       seqlen_k_rounded},
//         q.dtype, context)){};
//   }();
//
//   Flash_fwd_params params;
//   set_params_fprop(params, batch_size, seqlen_q, seqlen_k, seqlen_q_rounded,
//                    seqlen_k_rounded, num_heads, num_heads_k, head_size,
//                    head_size_rounded, q_padded, k_padded, v_padded, out,
//                    /*cu_seqlens_q_d=*/nullptr,
//                    /*cu_seqlens_k_d=*/nullptr,
//                    /*seqused_k=*/nullptr,
//                    return_softmax ? p->data_ptr() : nullptr,
//                    softmax_lse->data_ptr(), p_dropout, softmax_scale,
//                    window_size_left, window_size_right);
//
//   // set_params_splitkv(params, batch_size, num_heads, head_size, seqlen_k,
//   //                    seqlen_q, head_size_rounded, p_dropout,
//   /*num_splits*/
//   //                    0, dprops, opts);
//   set_params_splitkv(context, params, batch_size, num_heads, head_size,
//                      seqlen_k, seqlen_q, head_size_rounded, p_dropout,
//                      /*num_splits*/ 0, dprops);
//
//   // number of times random will be generated per thread, to offset philox
//   // counter in thc random state We use a custom RNG that increases the
//   offset
//   // by batch_size * nheads * 32.
//   int64_t counter_offset = params.b * params.h * 32;
//   // auto options =
//   //     torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
//   // auto rng_state = torch::empty({2}, options.dtype(torch::kInt64));
//   auto rng_state =
//       torch::empty<dllm::CUDA>(IntArrayRef1D{2}, torch::kInt64, context);
//   // Forward kernel will populate memory with the seed and offset.
//   params.rng_state = reinterpret_cast<uint64_t *>(rng_state->data_ptr());
//
//   if (p_dropout > 0.0) {
//     // auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
//     //     gen_, at::cuda::detail::getDefaultCUDAGenerator());
//     // See Note [Acquire lock when using random generators]
//     // std::lock_guard<std::mutex> lock(gen->mutex_);
//     // params.philox_args = gen->philox_cuda_state(counter_offset);
//     params.philox_args = {context->curandSeed, context->curandOffset.load()};
//     context->curandOffset += counter_offset;
//   }
//
//   set_params_alibi(params, alibi_slopes_, batch_size, num_heads);
//
//   // TODO(Jie): We assume seqlen_k > 0 is true
//   if (!(seqlen_k > 0)) {
//     exit(1);
//   }
//   // TODO(Jie): We assume seqlen_k > 0 is true
//   if (seqlen_k > 0) {
//     // auto stream = at::cuda::getCurrentCUDAStream().stream();
//     auto stream = context->cudaStream;
//     run_mha_fwd(params, stream);
//   } else {
//     // If seqlen_k == 0, then we have an empty tensor. We need to set the
//     output
//     // to 0.
//     out.zero_();
//     softmax_lse.fill_(std::numeric_limits<float>::infinity());
//   }
//
//   at::Tensor out_padded = out;
//   if (head_size_og % 8 != 0) {
//     out = out.index(
//         {"...", torch::indexing::Slice(torch::indexing::None,
//         head_size_og)});
//     if (out_.has_value()) {
//       // out_.value().copy_(out);
//       CHECK_CUDART(
//           cudaMemcpyAsync(out_.value()->data(), out->data(),
//                           dllm::toByte(out->dtype) * cute::size(out->layout),
//                           cudaMemcpyDeviceToDevice, context->cudaStream));
//     }
//   }
//
//   if (seqlenq_ngroups_swapped) {
//     out = out.transpose(1, 2).reshape(
//         {batch_size, 1, num_heads_k * seqlen_q, head_size_og});
//     out_padded = out_padded.transpose(1, 2).reshape(
//         {batch_size, 1, num_heads_k * seqlen_q, head_size_og});
//     q_padded = q_padded.transpose(1, 2).reshape(
//         {batch_size, 1, num_heads_k * seqlen_q, head_size_og});
//     softmax_lse = softmax_lse.reshape({batch_size, num_heads_k * seqlen_q,
//     1});
//   }
//   return std::make_tuple(out, q_padded, k_padded, v_padded, out_padded,
//                          softmax_lse, p, rng_state);
// }

// auto mha_varlen_fwd(
//     const dllm::ContextCompute *context,
//     at::Tensor<3>
//         &q,  // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b}
//         s_i
//     const at::Tensor<3>
//         &k,  // total_k x num_heads_k x head_size, total_k :=
//              // \sum_{i=0}^{b} s_i or num_blocks x page_block_size
//              // x num_heads_k x head_size if there's a block_table.
//     const at::Tensor<3>
//         &v,  // total_k x num_heads_k x head_size, total_k :=
//              // \sum_{i=0}^{b} s_i or num_blocks x page_block_size
//              // x num_heads_k x head_size if there's a block_table.
//     c10::optional<std::shared_ptr<at::Tensor<3>>> &
//         out_,  // total_q x num_heads x head_size, total_k := \sum_{i=0}^{b}
//         s_i
//     const at::Tensor<1> &cu_seqlens_q,  // b+1
//     const at::Tensor<1> &cu_seqlens_k,  // b+1
//     c10::optional<at::Tensor<1>>
//         &seqused_k,  // b. If given, only this many elements of each batch
//                      // element's keys are used.
//     c10::optional<std::shared_ptr<at::Tensor<2>>>
//         &block_table_,  // batch_size x max_num_blocks_per_seq
//     c10::optional<at::Tensor<1>> &alibi_slopes_,  // num_heads or b x
//     num_heads int max_seqlen_q, const int max_seqlen_k, const float
//     p_dropout, const float softmax_scale, const bool zero_tensors, bool
//     is_causal, int window_size_left, int window_size_right, const bool
//     return_softmax) {
//   auto dprops = at::cuda::getCurrentDeviceProperties();
//   // bool is_sm75 = dprops->major == 7 && dprops->minor == 5;
//   bool is_sm8x = dprops->major == 8 && dprops->minor >= 0;
//   bool is_sm90 = dprops->major == 9 && dprops->minor == 0;
//   TORCH_CHECK(is_sm90 || is_sm8x,
//               "FlashAttention only supports Ampere GPUs or newer.");
//   // We will support Turing in the near future
//   // TORCH_CHECK(is_sm90 || is_sm8x || is_sm75, "FlashAttention only supports
//   // Turing GPUs or newer.");
//
//   auto q_dtype = q.dtype;
//   TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16,
//               "FlashAttention only support fp16 and bf16 data type");
//   if (q_dtype == torch::kBFloat16) {
//     TORCH_CHECK(is_sm90 || is_sm8x,
//                 "bfloat16 is only supported on Ampere GPUs or newer");
//   }
//   TORCH_CHECK(k.dtype == q_dtype, "query and key must have the same dtype");
//   TORCH_CHECK(v.dtype == q_dtype, "query and value must have the same
//   dtype"); TORCH_CHECK(cu_seqlens_q.dtype == torch::kInt32,
//               "cu_seqlens_q must have dtype int32");
//   TORCH_CHECK(cu_seqlens_k.dtype == torch::kInt32,
//               "cu_seqlens_k must have dtype int32");
//
//   CHECK_DEVICE(q);
//   CHECK_DEVICE(k);
//   CHECK_DEVICE(v);
//   CHECK_DEVICE(cu_seqlens_q);
//   CHECK_DEVICE(cu_seqlens_k);
//
//   // at::Tensor block_table;
//   const bool paged_KV = block_table_.has_value();
//   // if (paged_KV) {
//   //   block_table = block_table_.value();
//   //   CHECK_DEVICE(block_table);
//   //   TORCH_CHECK(block_table.dtype() == torch::kInt32,
//   //               "block_table must have dtype torch.int32");
//   //   TORCH_CHECK(block_table.stride(-1) == 1,
//   //               "block_table must have contiguous last dimension");
//   // }
//   auto block_table = [&]() {
//     if (paged_KV) {
//       auto block_table = block_table_.value();
//       CHECK_DEVICE(*block_table);
//       TORCH_CHECK(block_table->dtype == torch::kInt32,
//                   "block_table must have dtype torch.int32");
//       TORCH_CHECK(block_table->stride<-1>() == 1,
//                   "block_table must have contiguous last dimension");
//       return block_table;
//     }
//     return std::remove_reference_t<decltype(block_table_.value())>{};
//   }();
//
//   // TORCH_CHECK(q.stride(-1) == 1,
//   //             "Input tensor must have contiguous last dimension");
//   // TORCH_CHECK(k.stride(-1) == 1,
//   //             "Input tensor must have contiguous last dimension");
//   // TORCH_CHECK(v.stride(-1) == 1,
//   //             "Input tensor must have contiguous last dimension");
//   TORCH_CHECK(q.stride<1>() == 1,
//               "Input tensor must have contiguous last dimension");
//   TORCH_CHECK(k.stride<1>() == 1,
//               "Input tensor must have contiguous last dimension");
//   TORCH_CHECK(v.stride<1>() == 1,
//               "Input tensor must have contiguous last dimension");
//   // CHECK_CONTIGUOUS(cu_seqlens_q);
//   // CHECK_CONTIGUOUS(cu_seqlens_k);
//
//   const auto sizes = q.sizes();
//
//   const int batch_size = cu_seqlens_q.numel() - 1;
//   int num_heads = sizes[1];
//   const int head_size_og = sizes[2];
//   // const int num_heads_k = paged_KV ? k.size(2) : k.size(1);
//   const int num_heads_k = paged_KV ? k.size<2>() : k.size<1>();
//
//   // const int max_num_blocks_per_seq = !paged_KV ? 0 : block_table.size(1);
//   const int max_num_blocks_per_seq = !paged_KV ? 0 : block_table->size<1>();
//   // const int num_blocks = !paged_KV ? 0 : k.size(0);
//   const int num_blocks = !paged_KV ? 0 : k.size<0>();
//   // const int page_block_size = !paged_KV ? 1 : k.size(1);
//   const int page_block_size = !paged_KV ? 1 : k.size<1>();
//   TORCH_CHECK(!paged_KV || page_block_size % 256 == 0,
//               "Paged KV cache block size must be divisible by 256");
//
//   if (max_seqlen_q == 1 && !alibi_slopes_.has_value()) {
//     is_causal = false;
//   }  // causal=true is the same as causal=false in this case
//   if (is_causal) {
//     window_size_right = 0;
//   }
//
//   auto cu_seqlens_q_d = cu_seqlens_q.data_ptr();
//
//   // Faster to transpose q from (b, 1, (nheads_kv ngroups), d) to (b,
//   ngroups,
//   // nheads_kv, d) in this case H/t Daniel Haziza
//   const int seqlenq_ngroups_swapped =
//       max_seqlen_q == 1 && num_heads > num_heads_k && window_size_left < 0 &&
//       window_size_right < 0 && p_dropout == 0.f && head_size_og % 8 == 0 &&
//       !alibi_slopes_.has_value();
//   const int ngroups = num_heads / num_heads_k;
//   if (seqlenq_ngroups_swapped) {
//     q = q.reshape({batch_size, num_heads_k, ngroups, head_size_og})
//             .transpose(1, 2)
//             .reshape({batch_size * ngroups, num_heads_k, head_size_og});
//     max_seqlen_q = ngroups;
//     num_heads = num_heads_k;
//     cu_seqlens_q_d = nullptr;
//   }
//
//   const int total_q = q.sizes()[0];
//
//   TORCH_CHECK(batch_size > 0, "batch size must be positive");
//   TORCH_CHECK(
//       head_size_og <= 256,
//       "FlashAttention forward only supports head dimension at most 256");
//   TORCH_CHECK(
//       num_heads % num_heads_k == 0,
//       "Number of heads in key/value must divide number of heads in query");
//
//   if (window_size_left >= max_seqlen_k) {
//     window_size_left = -1;
//   }
//   if (window_size_right >= max_seqlen_k) {
//     window_size_right = -1;
//   }
//
//   // CHECK_SHAPE(q, total_q, num_heads, head_size_og);
//   CHECK_SHAPE(q, IntArrayRef3D{total_q, num_heads, head_size_og});
//   if (!paged_KV) {
//     // const int total_k = k.size(0);
//     // CHECK_SHAPE(k, total_k, num_heads_k, head_size_og);
//     // CHECK_SHAPE(v, total_k, num_heads_k, head_size_og);
//     const int total_k = k.size<0>();
//     CHECK_SHAPE(k, IntArrayRef3D{total_k, num_heads_k, head_size_og});
//     CHECK_SHAPE(v, IntArrayRef3D{total_k, num_heads_k, head_size_og});
//   } else {
//     // CHECK_SHAPE(k, num_blocks, page_block_size, num_heads_k,
//     head_size_og);
//     // CHECK_SHAPE(v, num_blocks, page_block_size, num_heads_k,
//     head_size_og);
//     // CHECK_SHAPE(block_table, batch_size, max_num_blocks_per_seq);
//     CHECK_SHAPE(k, IntArrayRef4D{num_blocks, page_block_size, num_heads_k,
//                                  head_size_og});
//     CHECK_SHAPE(v, IntArrayRef4D{num_blocks, page_block_size, num_heads_k,
//                                  head_size_og});
//     CHECK_SHAPE(block_table, IntArrayRef2D{batch_size,
//     max_num_blocks_per_seq});
//   }
//
//   CHECK_SHAPE(cu_seqlens_q, IntArrayRef1D{batch_size + 1});
//   CHECK_SHAPE(cu_seqlens_k, IntArrayRef1D{batch_size + 1});
//   if (seqused_k.has_value()) {
//     auto seqused_k_ = seqused_k.value();
//     TORCH_CHECK(seqused_k_.dtype == torch::kInt32,
//                 "seqused_k must have dtype int32");
//     TORCH_CHECK(seqused_k_.is_cuda(), "seqused_k must be on CUDA device");
//     // TORCH_CHECK(seqused_k_.is_contiguous(), "seqused_k must be
//     contiguous"); CHECK_SHAPE(seqused_k_, IntArrayRef1D{batch_size});
//   }
//
//   // at::Tensor q_padded, k_padded, v_padded;
//   // if (head_size_og % 8 != 0) {
//   //   q_padded = torch::nn::functional::pad(
//   //       q, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og %
//   8}));
//   //   k_padded = torch::nn::functional::pad(
//   //       k, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og %
//   8}));
//   //   v_padded = torch::nn::functional::pad(
//   //       v, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og %
//   8}));
//   // } else {
//   //   q_padded = q;
//   //   k_padded = k;
//   //   v_padded = v;
//   // }
//   auto [q_padded, kcache_padded, vcache_padded] = [&]() {
//     if (head_size_og % 8 != 0) {
//       auto q_padded = torch::nn::functional::pad(
//           q, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og %
//           8}));
//       auto kcache_padded = torch::nn::functional::pad(
//           kcache,
//           torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
//       auto vcache_padded = torch::nn::functional::pad(
//           vcache,
//           torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
//       return std::make_tuple(q_padded, kcache_padded, vcache_padded);
//     } else {
//       auto q_padded = q;
//       auto kcache_padded = kcache;
//       auto vcache_padded = vcache;
//       return std::make_tuple(q_padded, kcache_padded, vcache_padded);
//     }
//   }();
//
//   // at::Tensor out;
//   // if (out_.has_value()) {
//   //   out = out_.value();
//   //   TORCH_CHECK(out.dtype() == q_dtype,
//   //               "Output must have the same dtype as inputs");
//   //   CHECK_DEVICE(out);
//   //   TORCH_CHECK(out.stride(-1) == 1,
//   //               "Output tensor must have contiguous last dimension");
//   //   CHECK_SHAPE(out, total_q, num_heads, head_size_og);
//   //   CHECK_SHAPE(out, sizes[0], sizes[1], head_size_og);
//   //   if (seqlenq_ngroups_swapped) {
//   //     out = out.reshape({batch_size, num_heads_k, ngroups, head_size_og})
//   //               .transpose(1, 2)
//   //               .reshape({batch_size * ngroups, num_heads_k,
//   head_size_og});
//   //   }
//   //   if (head_size_og % 8 != 0) {
//   //     out = torch::empty_like(q_padded);
//   //   }
//   // } else {
//   //   out = torch::empty_like(q_padded);
//   // }
//   auto out = [&]() {
//     if (out_.has_value()) {
//       auto out = out_.value();
//       TORCH_CHECK(out->dtype == q_dtype,
//                   "Output must have the same dtype as inputs");
//       CHECK_DEVICE(*out);
//       TORCH_CHECK(out->stride<-1>() == 1,
//                   "Output tensor must have contiguous last dimension");
//       CHECK_SHAPE(*out, IntArrayRef3D{total_q, num_heads, head_size_og});
//       CHECK_SHAPE(*out, IntArrayRef3D{sizes[0], sizes[1], head_size_og});
//       if (seqlenq_ngroups_swapped) {
//         return out.reshape({batch_size, num_heads_k, ngroups, head_size_og})
//             .transpose(1, 2)
//             .reshape({batch_size * ngroups, num_heads_k, head_size_og});
//       }
//       if (head_size_og % 8 != 0) {
//         return torch::empty_like<dllm::CUDA>(q_padded, context);
//       }
//       return out;
//     } else {
//       return torch::empty_like<dllm::CUDA>(q_padded, context);
//     }
//   }();
//
//   auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
//   const int head_size = round_multiple(head_size_og, 8);
//   const int head_size_rounded = round_multiple(head_size, 32);
//   const int seqlen_q_rounded = round_multiple(max_seqlen_q, 128);
//   const int seqlen_k_rounded = round_multiple(max_seqlen_k, 128);
//
//   // Otherwise the kernel will be launched from cuda:0 device
//   // Cast to char to avoid compiler warning about narrowing
//   // at::cuda::CUDAGuard device_guard{(char)q.get_device()};
//
//   // auto opts = q.options();
//
//   // auto softmax_lse = torch::empty({batch_size, num_heads, max_seqlen_q},
//   //                                 opts.dtype(at::kFloat));
//   auto softmax_lse = torch::empty<dllm::CUDA>(
//       IntArrayRef3D{batch_size, num_heads, max_seqlen_q}, at::kFloat,
//       context);
//   // at::Tensor p;
//   // Only return softmax if there's dropout to reduce compilation time
//   // if (return_softmax) {
//   //   TORCH_CHECK(p_dropout > 0.0f,
//   //               "return_softmax is only supported when p_dropout > 0.0");
//   //   p = torch::empty(
//   //       {batch_size, num_heads, seqlen_q_rounded, seqlen_k_rounded},
//   opts);
//   // }
//   auto p = [&]() {
//     if (return_softmax) {
//       TORCH_CHECK(p_dropout > 0.0f,
//                   "return_softmax is only supported when p_dropout > 0.0");
//       return torch::empty<dllm::CUDA>(
//           IntArrayRef4D{batch_size, num_heads, seqlen_q_rounded,
//                         seqlen_k_rounded},
//           q.dtype, context);
//     }
//     return decltype(torch::empty<dllm::CUDA>(
//         IntArrayRef4D{batch_size, num_heads, seqlen_q_rounded,
//                       seqlen_k_rounded},
//         q.dtype, context)){};
//   }();
//
//   if (zero_tensors) {
//     out.zero_();
//     softmax_lse.fill_(-std::numeric_limits<float>::infinity());
//     if (return_softmax) {
//       // p.zero_();
//       CHECK_CUDART(cudaMemsetAsync(p->data(), 0,
//                                    dllm::toByte(p->dtype) * p->numel(),
//                                    context->cudaStream));
//     }
//   }
//
//   Flash_fwd_params params;
//   set_params_fprop(
//       params, batch_size, max_seqlen_q, max_seqlen_k, seqlen_q_rounded,
//       seqlen_k_rounded, num_heads, num_heads_k, head_size, head_size_rounded,
//       q_padded, k_padded, v_padded, out, cu_seqlens_q_d,
//       cu_seqlens_k.data_ptr(),
//       seqused_k.has_value() ? seqused_k.value().data_ptr() : nullptr,
//       return_softmax ? p->data_ptr() : nullptr, softmax_lse->data_ptr(),
//       p_dropout, softmax_scale, window_size_left, window_size_right,
//       seqlenq_ngroups_swapped);
//
//   if (paged_KV) {
//     // params.block_table = block_table.data_ptr<int>();
//     // params.block_table_batch_stride = block_table.stride(0);
//     params.block_table = block_table->data_ptr<int>();
//     params.block_table_batch_stride = block_table->stride<0>();
//     params.k_batch_stride = k_padded.stride(0);
//     params.v_batch_stride = v_padded.stride(0);
//   }
//   params.page_block_size = page_block_size;
//   if (seqlenq_ngroups_swapped) {
//     // Only apply split-k for decoding
//     // set_params_splitkv(params, batch_size, num_heads, head_size,
//     // max_seqlen_k,
//     //                    max_seqlen_q, head_size_rounded, p_dropout,
//     //                    /*num_splits*/ 0, dprops, opts);
//     set_params_splitkv(context, params, batch_size, num_heads, head_size,
//                        max_seqlen_k, max_seqlen_q, head_size_rounded,
//                        p_dropout,
//                        /*num_splits*/ 0, dprops);
//   }
//
//   // number of times random will be generated per thread, to offset philox
//   // counter in thc random state We use a custom RNG that increases the
//   offset
//   // by batch_size * nheads * 32.
//   int64_t counter_offset = params.b * params.h * 32;
//   // auto options =
//   //     torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
//   // auto rng_state = torch::empty({2}, options.dtype(torch::kInt64));
//   auto rng_state =
//       torch::empty<dllm::CUDA>(IntArrayRef2D{2}, torch::kInt64, context);
//   // Forward kernel will populate memory with the seed and offset.
//   params.rng_state = reinterpret_cast<uint64_t *>(rng_state->data_ptr());
//
//   if (p_dropout > 0.0) {
//     // auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
//     //     gen_, at::cuda::detail::getDefaultCUDAGenerator());
//     // See Note [Acquire lock when using random generators]
//     // std::lock_guard<std::mutex> lock(gen->mutex_);
//     // params.philox_args = gen->philox_cuda_state(counter_offset);
//     params.philox_args = {context->curandSeed, context->curandOffset.load()};
//     context->curandOffset += counter_offset;
//   }
//
//   set_params_alibi(params, alibi_slopes_, batch_size, num_heads);
//
//   if (max_seqlen_k > 0) {
//     // auto stream = at::cuda::getCurrentCUDAStream().stream();
//     auto stream = context->cudaStream;
//     run_mha_fwd(params, stream, paged_KV);
//   } else {
//     // If seqlen_k == 0, then we have an empty tensor. We need to set the
//     output
//     // to 0.
//     out.zero_();
//     softmax_lse.fill_(std::numeric_limits<float>::infinity());
//   }
//
//   at::Tensor out_padded = out;
//   if (head_size_og % 8 != 0) {
//     out = out.index(
//         {"...", torch::indexing::Slice(torch::indexing::None,
//         head_size_og)});
//     if (out_.has_value()) {
//       // out_.value().copy_(out);
//       CHECK_CUDART(
//           cudaMemcpyAsync(out_.value()->data(), out->data(),
//                           dllm::toByte(out->dtype) * cute::size(out->layout),
//                           cudaMemcpyDeviceToDevice, context->cudaStream));
//     }
//   }
//
//   if (seqlenq_ngroups_swapped) {
//     int64_t size_before[] = {batch_size, max_seqlen_q, num_heads_k,
//                              head_size_og};
//     int64_t size_after[] = {batch_size, num_heads_k * max_seqlen_q,
//                             head_size_og};
//     out = out.reshape(size_before).transpose(1, 2).reshape(size_after);
//     out_padded =
//         out_padded.reshape(size_before).transpose(1, 2).reshape(size_after);
//     q_padded =
//         q_padded.reshape(size_before).transpose(1, 2).reshape(size_after);
//     softmax_lse =
//         softmax_lse.reshape({batch_size, num_heads_k * max_seqlen_q, 1});
//   }
//
//   return std::make_tuple(out, q_padded, k_padded, v_padded, out_padded,
//                          softmax_lse, p, rng_state);
// }

void run_mha_bwd(Flash_bwd_params &params, cudaStream_t stream) {
  FP16_SWITCH(!params.is_bf16, [&] {
    HEADDIM_SWITCH(params.d,
                   [&] { run_mha_bwd_<elem_type, kHeadDim>(params, stream); });
  });
}

namespace dllm::compute::FlashAttention {
void backward(
    const dllm::ContextCompute *context,
    at::Tensor<4> &dout,  // batch_size x seqlen_q x num_heads, x head_size_og
    const std::shared_ptr<at::Tensor<4>>
        &dq,  // batch_size x seqlen_q x num_heads x head_size
    const std::shared_ptr<at::Tensor<4>>
        &dk,  // batch_size x seqlen_k x num_heads_k x head_size
    const std::shared_ptr<at::Tensor<4>>
        &dv,  // batch_size x seqlen_k x num_heads_k x head_size
    at::Tensor<1> &rng_state,
    at::Tensor<4> &out,  // batch_size x seqlen_q x num_heads x head_size
    at::Tensor<3> &softmax_lse,  // b x h x seqlen_q
    const at::Tensor<4> &q,  // batch_size x seqlen_q x num_heads x head_size
    const at::Tensor<4> &k,  // batch_size x seqlen_k x num_heads_k x head_size
    const at::Tensor<4> &v,  // batch_size x seqlen_k x num_heads_k x head_size
    const double p_dropout,  // probability to drop
    const double softmax_scale, const bool is_causal = true) {
  const bool deterministic = false;
  int window_size_left = -1;
  int window_size_right = -1;
#ifdef FLASHATTENTION_DISABLE_BACKWARD
  TORCH_CHECK(false, "This flash attention build does not support backward.");
#endif
  if (is_causal) {
    window_size_right = 0;
  }
  auto dprops = at::cuda::getCurrentDeviceProperties();
  // bool is_sm75 = dprops->major == 7 && dprops->minor == 5;
  bool is_sm8x = dprops->major == 8 && dprops->minor >= 0;
  bool is_sm80 = dprops->major == 8 && dprops->minor == 0;
  bool is_sm90 = dprops->major == 9 && dprops->minor == 0;
  TORCH_CHECK(is_sm90 || is_sm8x,
              "FlashAttention only supports Ampere GPUs or newer.");
  // We will support Turing in the near future
  // TORCH_CHECK(is_sm90 || is_sm8x || is_sm75, "FlashAttention only supports
  // Turing GPUs or newer.");

  bool is_dropout = p_dropout > 0.0;
  // auto stream = at::cuda::getCurrentCUDAStream().stream();
  auto stream = context->cudaStream;

  auto q_dtype = q.dtype;
  TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16,
              "FlashAttention only support fp16 and bf16 data type");
  if (q_dtype == torch::kBFloat16) {
    TORCH_CHECK(is_sm90 || is_sm8x,
                "bfloat16 is only supported on Ampere GPUs or newer");
  }
  TORCH_CHECK(k.dtype == q_dtype, "query and key must have the same dtype");
  TORCH_CHECK(v.dtype == q_dtype, "query and value must have the same dtype");
  TORCH_CHECK(out.dtype == q_dtype, "query and out must have the same dtype");
  TORCH_CHECK(dout.dtype == q_dtype, "query and dout must have the same dtype");

  CHECK_DEVICE(q);
  CHECK_DEVICE(k);
  CHECK_DEVICE(v);
  CHECK_DEVICE(out);
  CHECK_DEVICE(dout);
  CHECK_DEVICE(softmax_lse);

  // TORCH_CHECK(q.stride(-1) == 1,
  //             "Input tensor must have contiguous last dimension");
  // TORCH_CHECK(k.stride(-1) == 1,
  //             "Input tensor must have contiguous last dimension");
  // TORCH_CHECK(v.stride(-1) == 1,
  //             "Input tensor must have contiguous last dimension");
  // TORCH_CHECK(out.stride(-1) == 1,
  //             "out tensor must have contiguous last dimension");
  // TORCH_CHECK(dout.stride(-1) == 1,
  //             "dout tensor must have contiguous last dimension");
  TORCH_CHECK(q.stride<-1>() == 1,
              "Input tensor must have contiguous last dimension");
  TORCH_CHECK(k.stride<-1>() == 1,
              "Input tensor must have contiguous last dimension");
  TORCH_CHECK(v.stride<-1>() == 1,
              "Input tensor must have contiguous last dimension");
  TORCH_CHECK(out.stride<-1>() == 1,
              "out tensor must have contiguous last dimension");
  TORCH_CHECK(dout.stride<-1>() == 1,
              "dout tensor must have contiguous last dimension");

  const auto sizes = q.sizes();

  const int batch_size = sizes[0];
  const int seqlen_q = sizes[1];
  const int num_heads = sizes[2];
  // const int head_size_og = dout.size(3);
  const int head_size_og = dout.size<3>();
  const int head_size = sizes[3];
  const int seqlen_k = k.size<1>();
  const int num_heads_k = k.size<2>();
  // const int seqlen_k = k.size(1);
  // const int num_heads_k = k.size(2);
  TORCH_CHECK(batch_size > 0, "batch size must be positive");
  TORCH_CHECK(head_size % 8 == 0, "head_size should be a multiple of 8");
  TORCH_CHECK(
      head_size <= 256,
      "FlashAttention backward only supports head dimension at most 256");
  if (head_size > 192 && (head_size <= 224 || is_dropout)) {
    TORCH_CHECK(
        is_sm80 || is_sm90,
        "FlashAttention backward for head dim 256 with dropout, or head dim "
        "224 with/without dropout requires A100/A800 or H100/H800");
  }
  TORCH_CHECK(
      num_heads % num_heads_k == 0,
      "Number of heads in key/value must divide number of heads in query");

  auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
  const int head_size_rounded = round_multiple(head_size, 32);
  const int seqlen_q_rounded = round_multiple(seqlen_q, 128);
  const int seqlen_k_rounded = round_multiple(seqlen_k, 128);

  TORCH_CHECK(head_size == round_multiple(head_size_og, 8),
              "head_size must be head_size_og rounded to a multiple of 8");

  if (window_size_left >= seqlen_k) {
    window_size_left = -1;
  }
  if (window_size_right >= seqlen_k) {
    window_size_right = -1;
  }

  // CHECK_SHAPE(q, batch_size, seqlen_q, num_heads, head_size);
  // CHECK_SHAPE(k, batch_size, seqlen_k, num_heads_k, head_size);
  // CHECK_SHAPE(v, batch_size, seqlen_k, num_heads_k, head_size);
  // CHECK_SHAPE(out, batch_size, seqlen_q, num_heads, head_size);
  // CHECK_SHAPE(dout, batch_size, seqlen_q, num_heads, head_size_og);
  CHECK_SHAPE(q, IntArrayRef4D{batch_size, seqlen_q, num_heads, head_size});
  CHECK_SHAPE(k, IntArrayRef4D{batch_size, seqlen_k, num_heads_k, head_size});
  CHECK_SHAPE(v, IntArrayRef4D{batch_size, seqlen_k, num_heads_k, head_size});
  CHECK_SHAPE(out, IntArrayRef4D{batch_size, seqlen_q, num_heads, head_size});
  CHECK_SHAPE(dout,
              IntArrayRef4D{batch_size, seqlen_q, num_heads, head_size_og});

  // at::Tensor dq, dk, dv;
  // if (dq_.has_value()) {
  //   dq = dq_.value();
  //   TORCH_CHECK(dq.dtype() == q_dtype, "dq must have the same dtype as q");
  //   CHECK_DEVICE(dq);
  //   TORCH_CHECK(dq.stride(-1) == 1, "dq must have contiguous last
  //   dimension"); CHECK_SHAPE(dq, batch_size, seqlen_q, num_heads,
  //   head_size);
  // } else {
  //   dq = torch::empty_like(q);
  // }
  // if (dk_.has_value()) {
  //   dk = dk_.value();
  //   TORCH_CHECK(dk.dtype() == q_dtype, "dk must have the same dtype as q");
  //   CHECK_DEVICE(dk);
  //   TORCH_CHECK(dk.stride(-1) == 1, "dk must have contiguous last
  //   dimension"); CHECK_SHAPE(dk, batch_size, seqlen_k, num_heads_k,
  //   head_size);
  // } else {
  //   dk = torch::empty_like(k);
  // }
  // if (dv_.has_value()) {
  //   dv = dv_.value();
  //   TORCH_CHECK(dv.dtype() == q_dtype, "dv must have the same dtype as q");
  //   CHECK_DEVICE(dv);
  //   TORCH_CHECK(dv.stride(-1) == 1, "dv must have contiguous last
  //   dimension"); CHECK_SHAPE(dv, batch_size, seqlen_k, num_heads_k,
  //   head_size);
  // } else {
  //   dv = torch::empty_like(v);
  // }
  TORCH_CHECK(dq->dtype == q_dtype, "dk must have the same dtype as q");
  CHECK_DEVICE(*dq);
  TORCH_CHECK(dq->stride<-1>() == 1, "dk must have contiguous last dimension");
  CHECK_SHAPE(*dq, IntArrayRef4D{batch_size, seqlen_k, num_heads_k, head_size});
  TORCH_CHECK(dk->dtype == q_dtype, "dk must have the same dtype as q");
  CHECK_DEVICE(*dk);
  TORCH_CHECK(dk->stride<-1>() == 1, "dk must have contiguous last dimension");
  CHECK_SHAPE(*dk, IntArrayRef4D{batch_size, seqlen_k, num_heads_k, head_size});
  TORCH_CHECK(dv->dtype == q_dtype, "dk must have the same dtype as q");
  CHECK_DEVICE(*dv);
  TORCH_CHECK(dv->stride<-1>() == 1, "dk must have contiguous last dimension");
  CHECK_SHAPE(*dv, IntArrayRef4D{batch_size, seqlen_k, num_heads_k, head_size});

  // at::Tensor dout_padded;
  // if (head_size_og % 8 != 0) {
  //   dout_padded = torch::nn::functional::pad(
  //       dout, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og %
  //       8}));
  // } else {
  //   dout_padded = dout;
  // }
  //  auto dout_padded = [&]() {
  //    if (head_size_og % 8 != 0) {
  //      return torch::nn::functional::pad(
  //          dout,
  //          torch::nn::functional::PadFuncOptions({0, 8 - head_size_og %
  //          8}));
  //    } else {
  //      return dout;
  //    }
  //  }();

  // bool loop = seqlen_k > blocksize_c;
  // TODO: change later, for now set to true for simplicity
  bool loop = true;

  // Otherwise the kernel will be launched from cuda:0 device
  // Cast to char to avoid compiler warning about narrowing
  // at::cuda::CUDAGuard device_guard{(char)q.get_device()};

  // auto opts = q.options();
  // auto softmax_d = torch::empty({batch_size, num_heads, seqlen_q_rounded},
  //                               opts.dtype(at::kFloat));
  auto softmax_d = torch::empty<dllm::CUDA>(
      IntArrayRef3D{batch_size, num_heads, seqlen_q_rounded}, at::kFloat,
      context);
  // at::Tensor dq_accum;
  // at::Tensor dk_accum, dv_accum;
  // if (loop) {
  //   if (!deterministic) {
  //     dq_accum = torch::empty(
  //         {batch_size, seqlen_q_rounded, num_heads, head_size_rounded},
  //         opts.dtype(at::kFloat));
  //   } else {
  //     const int nsplits =
  //         (dprops->multiProcessorCount + batch_size * num_heads - 1) /
  //         (batch_size * num_heads);
  //     dq_accum = torch::zeros(
  //         {nsplits, batch_size, seqlen_q_rounded, num_heads,
  //         head_size_rounded}, opts.dtype(at::kFloat));
  //   }
  //   // dk_accum = torch::empty({batch_size, num_heads_k, seqlen_k_rounded,
  //   // head_size_rounded}, opts.dtype(at::kFloat)); dv_accum =
  //   // torch::empty({batch_size, num_heads_k, seqlen_k_rounded,
  //   // head_size_rounded}, opts.dtype(at::kFloat));
  // }
  auto dq_accum = [&]() {
    if (loop) {
      // TODO(Jie): We assume deterministic is false
      if (deterministic != false) {
        exit(1);
      }
      // TODO(Jie): We assume deterministic is false
      if (!deterministic) {
        return torch::empty<dllm::CUDA>(
            IntArrayRef4D{batch_size, seqlen_q_rounded, num_heads,
                          head_size_rounded},
            at::kFloat, context);
      }
      //      else {
      //        const int nsplits =
      //            (dprops->multiProcessorCount + batch_size * num_heads - 1)
      //            / (batch_size * num_heads);
      //        auto p = torch::empty<dllm::CUDA>(
      //            IntArrayRef5D{nsplits, batch_size, seqlen_q_rounded,
      //            num_heads,
      //                          head_size_rounded},
      //            at::kFloat, context);
      //        CHECK_CUDART(cudaMemsetAsync(
      //            p->data(), 0, dllm::toByte(p->dtype) *
      //            cute::size(p->layout), context->cudaStream));
      //        return p;
      //      }
      // dk_accum = torch::empty({batch_size, num_heads_k, seqlen_k_rounded,
      // head_size_rounded}, opts.dtype(at::kFloat)); dv_accum =
      // torch::empty({batch_size, num_heads_k, seqlen_k_rounded,
      // head_size_rounded}, opts.dtype(at::kFloat));
    }
    throw;
  }();

  // at::Tensor dk_expanded, dv_expanded;
  // if (num_heads_k != num_heads) {  // MQA / GQA
  //   dk_expanded =
  //       torch::empty({batch_size, seqlen_k, num_heads, head_size}, opts);
  //   dv_expanded =
  //       torch::empty({batch_size, seqlen_k, num_heads, head_size}, opts);
  // } else {
  //   dk_expanded = dk;
  //   dv_expanded = dv;
  // }
  auto [dk_expanded, dv_expanded] = [&]() {
    if (num_heads_k != num_heads) {  // MQA / GQA
      throw;
      auto dk_expanded = torch::empty<dllm::CUDA>(
          IntArrayRef4D{batch_size, seqlen_k, num_heads, head_size}, q.dtype,
          context);
      auto dv_expanded = torch::empty<dllm::CUDA>(
          IntArrayRef4D{batch_size, seqlen_k, num_heads, head_size}, q.dtype,
          context);
      return std::make_tuple(dk_expanded, dv_expanded);
    } else {
      auto dk_expanded = dk;
      auto dv_expanded = dv;
      return std::make_tuple(dk_expanded, dv_expanded);
    }
  }();

  Flash_bwd_params params;

  set_params_dgrad(params, batch_size, seqlen_q, seqlen_k, seqlen_q_rounded,
                   seqlen_k_rounded, num_heads, num_heads_k, head_size,
                   head_size_rounded, q, k, v, out, dout, *dq, *dk_expanded,
                   *dv_expanded, nullptr, nullptr,
                   loop ? dq_accum->data_ptr() : nullptr,
                   // loop ? dk_accum.data_ptr() : nullptr,
                   // loop ? dv_accum.data_ptr() : nullptr,
                   nullptr, nullptr, softmax_lse.data_ptr(),
                   softmax_d->data_ptr(), p_dropout, softmax_scale,
                   window_size_left, window_size_right, deterministic);
  params.dq_accum_split_stride = !deterministic ? 0 : dq_accum->stride<0>();

  auto launch = &run_mha_bwd;

  // auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
  //     gen_, at::cuda::detail::getDefaultCUDAGenerator());

  // We use a custom RNG that increases the offset by batch_size * nheads
  // * 32. int64_t counter_offset = params.b * params.h * 32;

  params.rng_state = reinterpret_cast<uint64_t *>(rng_state.data_ptr());

  //  if (rng_state.has_value()) {
  //    params.rng_state =
  //        reinterpret_cast<uint64_t *>(rng_state.value().data_ptr());
  //  } else if (is_dropout) {
  //    // See Note [Acquire lock when using random generators]
  //    // std::lock_guard<std::mutex> lock(gen->mutex_);
  //    // params.philox_args = gen->philox_cuda_state(counter_offset);
  //    params.philox_args = {context->curandSeed,
  //    context->curandOffset.load()}; context->curandOffset +=
  //    counter_offset; auto seeds =
  //    at::cuda::philox::unpack(params.philox_args); params.rng_state[0] =
  //    std::get<0>(seeds); params.rng_state[1] = std::get<1>(seeds);
  //  }

  // set_params_alibi(params, alibi_slopes_, batch_size, num_heads);
  params.alibi_slopes_ptr = nullptr;

  if (seqlen_q > 0) {
    launch(params, stream);
  } else {
    // If seqlen_q == 0, then we have an empty tensor. We need to set the
    // output to 0. dk_expanded.zero_(); dv_expanded.zero_();
    // softmax_d.zero_();
    CHECK_CUDART(cudaMemsetAsync(
        dk_expanded->data(), 0,
        dllm::toByte(dk_expanded->dtype) * cute::size(dk_expanded->layout),
        context->cudaStream));
    CHECK_CUDART(cudaMemsetAsync(
        dv_expanded->data(), 0,
        dllm::toByte(dv_expanded->dtype) * cute::size(dv_expanded->layout),
        context->cudaStream));
    CHECK_CUDART(cudaMemsetAsync(
        softmax_d->data(), 0,
        dllm::toByte(softmax_d->dtype) * cute::size(softmax_d->layout),
        context->cudaStream));
  }

  // TODO(Jie): ----
  // For MQA/GQA we need to sum dK and dV across the groups
  //  if (num_heads_k != num_heads) {
  //    at::sum_out(dk,
  //                at::reshape(dk_expanded, {batch_size, seqlen_k,
  //                num_heads_k,
  //                                          num_heads / num_heads_k,
  //                                          head_size}),
  //                {3});
  //    at::sum_out(dv,
  //                at::reshape(dv_expanded, {batch_size, seqlen_k,
  //                num_heads_k,
  //                                          num_heads / num_heads_k,
  //                                          head_size}),
  //                {3});
  //  }
  // TODO(Jie): ----

  //  if (head_size_og % 8 != 0) {
  //    dq = dq.index(
  //        {"...", torch::indexing::Slice(torch::indexing::None,
  //        head_size_og)});
  //    dk = dk.index(
  //        {"...", torch::indexing::Slice(torch::indexing::None,
  //        head_size_og)});
  //    dv = dv.index(
  //        {"...", torch::indexing::Slice(torch::indexing::None,
  //        head_size_og)});
  //  }

  //  return std::make_tuple(dq, dk, dv, softmax_d);
}

TaskCompute backward(const std::shared_ptr<dllm::Tensor<4>> &dq,
                     const std::shared_ptr<dllm::Tensor<4>> &dk,
                     const std::shared_ptr<dllm::Tensor<4>> &dv,
                     const std::shared_ptr<dllm::Tensor<4>> &dout,
                     const std::shared_ptr<dllm::Tensor<1>> &random_state,
                     const std::shared_ptr<dllm::Tensor<4>> &out,
                     const std::shared_ptr<dllm::Tensor<3>> &softmax_lse,
                     const std::shared_ptr<const dllm::Tensor<4>> &q,
                     const std::shared_ptr<const dllm::Tensor<4>> &k,
                     const std::shared_ptr<const dllm::Tensor<4>> &v,
                     const double drouput_p, const double softmax_scale) {
  if (!(out->layout.shape() == q->layout.shape()) ||
      !(k->layout.shape() == v->layout.shape())) {
  }
  auto task = TaskCompute{
      [=, futureDq = *dq->future, futureDk = *dk->future,
       futureDv = *dv->future, futureDout = *dout->future,
       futureOut = *out->future, futureSofmax = *softmax_lse->future,
       futureQ = *q->future, futureK = *k->future,
       futureV = *v->future](const ContextCompute *context) {
        util::waitFutureIfValid(futureDout);
        util::waitFutureIfValid(futureDq);
        util::waitFutureIfValid(futureDk);
        util::waitFutureIfValid(futureDv);
        util::waitFutureIfValid(futureOut);
        util::waitFutureIfValid(futureSofmax);
        util::waitFutureIfValid(futureQ);
        util::waitFutureIfValid(futureK);
        util::waitFutureIfValid(futureV);
        backward(context, *dout, dq, dk, dv, *random_state, *out, *softmax_lse,
                 *q, *k, *v, drouput_p, softmax_scale);
        CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
      }};
  const auto &future = *dq->future = task.get_future();
  *dk->future = future;
  *dv->future = future;
  *dout->future = future;
  *random_state->future = future;
  *out->future = future;
  *softmax_lse->future = future;
  *q->future = future;
  *k->future = future;
  *v->future = future;
  return task;
}
}  // namespace dllm::compute::FlashAttention

// auto mha_bwd(
//     const dllm::ContextCompute *context,
//     const at::Tensor<4>
//         &dout,  // batch_size x seqlen_q x num_heads, x head_size_og
//     const at::Tensor<4> &q,  // batch_size x seqlen_q x num_heads x head_size
//     const at::Tensor<4> &k,  // batch_size x seqlen_k x num_heads_k x
//     head_size const at::Tensor<4> &v,  // batch_size x seqlen_k x num_heads_k
//     x head_size const at::Tensor<4> &out,  // batch_size x seqlen_q x
//     num_heads x head_size const at::Tensor<3> &softmax_lse,  // b x h x
//     seqlen_q c10::optional<std::shared_ptr<at::Tensor<4>>>
//         &dq_,  // batch_size x seqlen_q x num_heads x head_size
//     c10::optional<std::shared_ptr<at::Tensor<4>>>
//         &dk_,  // batch_size x seqlen_k x num_heads_k x head_size
//     c10::optional<std::shared_ptr<at::Tensor<4>>>
//         &dv_,  // batch_size x seqlen_k x num_heads_k x head_size
//     c10::optional<at::Tensor<1>>
//         &alibi_slopes_,     // num_heads or batch_size x num_heads
//     const float p_dropout,  // probability to drop
//     const float softmax_scale, const bool is_causal, int window_size_left,
//     int window_size_right, const bool deterministic,
//     c10::optional<at::Tensor<2>> &rng_state) {
// #ifdef FLASHATTENTION_DISABLE_BACKWARD
//   TORCH_CHECK(false, "This flash attention build does not support
//   backward.");
// #endif
//   if (is_causal) {
//     window_size_right = 0;
//   }
//   auto dprops = at::cuda::getCurrentDeviceProperties();
//   // bool is_sm75 = dprops->major == 7 && dprops->minor == 5;
//   bool is_sm8x = dprops->major == 8 && dprops->minor >= 0;
//   bool is_sm80 = dprops->major == 8 && dprops->minor == 0;
//   bool is_sm90 = dprops->major == 9 && dprops->minor == 0;
//   TORCH_CHECK(is_sm90 || is_sm8x,
//               "FlashAttention only supports Ampere GPUs or newer.");
//   // We will support Turing in the near future
//   // TORCH_CHECK(is_sm90 || is_sm8x || is_sm75, "FlashAttention only supports
//   // Turing GPUs or newer.");
//
//   bool is_dropout = p_dropout > 0.0;
//   // auto stream = at::cuda::getCurrentCUDAStream().stream();
//   auto stream = context->cudaStream;
//
//   auto q_dtype = q.dtype;
//   TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16,
//               "FlashAttention only support fp16 and bf16 data type");
//   if (q_dtype == torch::kBFloat16) {
//     TORCH_CHECK(is_sm90 || is_sm8x,
//                 "bfloat16 is only supported on Ampere GPUs or newer");
//   }
//   TORCH_CHECK(k.dtype == q_dtype, "query and key must have the same dtype");
//   TORCH_CHECK(v.dtype == q_dtype, "query and value must have the same
//   dtype"); TORCH_CHECK(out.dtype == q_dtype, "query and out must have the
//   same dtype"); TORCH_CHECK(dout.dtype == q_dtype, "query and dout must have
//   the same dtype");
//
//   CHECK_DEVICE(q);
//   CHECK_DEVICE(k);
//   CHECK_DEVICE(v);
//   CHECK_DEVICE(out);
//   CHECK_DEVICE(dout);
//   CHECK_DEVICE(softmax_lse);
//
//   // TORCH_CHECK(q.stride(-1) == 1,
//   //             "Input tensor must have contiguous last dimension");
//   // TORCH_CHECK(k.stride(-1) == 1,
//   //             "Input tensor must have contiguous last dimension");
//   // TORCH_CHECK(v.stride(-1) == 1,
//   //             "Input tensor must have contiguous last dimension");
//   // TORCH_CHECK(out.stride(-1) == 1,
//   //             "out tensor must have contiguous last dimension");
//   // TORCH_CHECK(dout.stride(-1) == 1,
//   //             "dout tensor must have contiguous last dimension");
//   TORCH_CHECK(q.stride<-1>() == 1,
//               "Input tensor must have contiguous last dimension");
//   TORCH_CHECK(k.stride<-1>() == 1,
//               "Input tensor must have contiguous last dimension");
//   TORCH_CHECK(v.stride<-1>() == 1,
//               "Input tensor must have contiguous last dimension");
//   TORCH_CHECK(out.stride<-1>() == 1,
//               "out tensor must have contiguous last dimension");
//   TORCH_CHECK(dout.stride<-1>() == 1,
//               "dout tensor must have contiguous last dimension");
//
//   const auto sizes = q.sizes();
//
//   const int batch_size = sizes[0];
//   const int seqlen_q = sizes[1];
//   const int num_heads = sizes[2];
//   // const int head_size_og = dout.size(3);
//   const int head_size_og = dout.size<3>();
//   const int head_size = sizes[3];
//   const int seqlen_k = k.size<1>();
//   const int num_heads_k = k.size<2>();
//   // const int seqlen_k = k.size(1);
//   // const int num_heads_k = k.size(2);
//   TORCH_CHECK(batch_size > 0, "batch size must be positive");
//   TORCH_CHECK(head_size % 8 == 0, "head_size should be a multiple of 8");
//   TORCH_CHECK(
//       head_size <= 256,
//       "FlashAttention backward only supports head dimension at most 256");
//   if (head_size > 192 && (head_size <= 224 || is_dropout)) {
//     TORCH_CHECK(
//         is_sm80 || is_sm90,
//         "FlashAttention backward for head dim 256 with dropout, or head dim "
//         "224 with/without dropout requires A100/A800 or H100/H800");
//   }
//   TORCH_CHECK(
//       num_heads % num_heads_k == 0,
//       "Number of heads in key/value must divide number of heads in query");
//
//   auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
//   const int head_size_rounded = round_multiple(head_size, 32);
//   const int seqlen_q_rounded = round_multiple(seqlen_q, 128);
//   const int seqlen_k_rounded = round_multiple(seqlen_k, 128);
//
//   TORCH_CHECK(head_size == round_multiple(head_size_og, 8),
//               "head_size must be head_size_og rounded to a multiple of 8");
//
//   if (window_size_left >= seqlen_k) {
//     window_size_left = -1;
//   }
//   if (window_size_right >= seqlen_k) {
//     window_size_right = -1;
//   }
//
//   // CHECK_SHAPE(q, batch_size, seqlen_q, num_heads, head_size);
//   // CHECK_SHAPE(k, batch_size, seqlen_k, num_heads_k, head_size);
//   // CHECK_SHAPE(v, batch_size, seqlen_k, num_heads_k, head_size);
//   // CHECK_SHAPE(out, batch_size, seqlen_q, num_heads, head_size);
//   // CHECK_SHAPE(dout, batch_size, seqlen_q, num_heads, head_size_og);
//   CHECK_SHAPE(q, IntArrayRef4D{batch_size, seqlen_q, num_heads, head_size});
//   CHECK_SHAPE(k, IntArrayRef4D{batch_size, seqlen_k, num_heads_k,
//   head_size}); CHECK_SHAPE(v, IntArrayRef4D{batch_size, seqlen_k,
//   num_heads_k, head_size}); CHECK_SHAPE(out, IntArrayRef4D{batch_size,
//   seqlen_q, num_heads, head_size}); CHECK_SHAPE(dout,
//               IntArrayRef4D{batch_size, seqlen_q, num_heads, head_size_og});
//
//   // at::Tensor dq, dk, dv;
//   // if (dq_.has_value()) {
//   //   dq = dq_.value();
//   //   TORCH_CHECK(dq.dtype() == q_dtype, "dq must have the same dtype as
//   q");
//   //   CHECK_DEVICE(dq);
//   //   TORCH_CHECK(dq.stride(-1) == 1, "dq must have contiguous last
//   //   dimension"); CHECK_SHAPE(dq, batch_size, seqlen_q, num_heads,
//   head_size);
//   // } else {
//   //   dq = torch::empty_like(q);
//   // }
//   // if (dk_.has_value()) {
//   //   dk = dk_.value();
//   //   TORCH_CHECK(dk.dtype() == q_dtype, "dk must have the same dtype as
//   q");
//   //   CHECK_DEVICE(dk);
//   //   TORCH_CHECK(dk.stride(-1) == 1, "dk must have contiguous last
//   //   dimension"); CHECK_SHAPE(dk, batch_size, seqlen_k, num_heads_k,
//   //   head_size);
//   // } else {
//   //   dk = torch::empty_like(k);
//   // }
//   // if (dv_.has_value()) {
//   //   dv = dv_.value();
//   //   TORCH_CHECK(dv.dtype() == q_dtype, "dv must have the same dtype as
//   q");
//   //   CHECK_DEVICE(dv);
//   //   TORCH_CHECK(dv.stride(-1) == 1, "dv must have contiguous last
//   //   dimension"); CHECK_SHAPE(dv, batch_size, seqlen_k, num_heads_k,
//   //   head_size);
//   // } else {
//   //   dv = torch::empty_like(v);
//   // }
//   auto [dq, dk, dv] = [&]() {
//     std::shared_ptr<at::Tensor<4>> dq, dk, dv;
//     if (dq_.has_value()) {
//       dq = dq_.value();
//       TORCH_CHECK(dq->dtype == q_dtype, "dq must have the same dtype as q");
//       CHECK_DEVICE(*dq);
//       TORCH_CHECK(dq->stride<-1>() == 1,
//                   "dq must have contiguous last dimension");
//       CHECK_SHAPE(*dq,
//                   IntArrayRef4D{batch_size, seqlen_q, num_heads, head_size});
//     } else {
//       dq = torch::empty_like<dllm::CUDA>(q, context);
//     }
//     if (dk_.has_value()) {
//       dk = dk_.value();
//       TORCH_CHECK(dk->dtype == q_dtype, "dk must have the same dtype as q");
//       CHECK_DEVICE(*dk);
//       TORCH_CHECK(dk->stride<-1>() == 1,
//                   "dk must have contiguous last dimension");
//       CHECK_SHAPE(*dk,
//                   IntArrayRef4D{batch_size, seqlen_k, num_heads_k,
//                   head_size});
//     } else {
//       dk = torch::empty_like<dllm::CUDA>(k, context);
//     }
//     if (dv_.has_value()) {
//       dv = dv_.value();
//       TORCH_CHECK(dv->dtype == q_dtype, "dv must have the same dtype as q");
//       CHECK_DEVICE(*dv);
//       TORCH_CHECK(dv->stride<-1>() == 1,
//                   "dv must have contiguous last dimension");
//       CHECK_SHAPE(*dv,
//                   IntArrayRef4D{batch_size, seqlen_k, num_heads_k,
//                   head_size});
//     } else {
//       dv = torch::empty_like<dllm::CUDA>(v, context);
//     }
//     return std::make_tuple(dq, dk, dv);
//   }();
//
//   // at::Tensor dout_padded;
//   // if (head_size_og % 8 != 0) {
//   //   dout_padded = torch::nn::functional::pad(
//   //       dout, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og %
//   //       8}));
//   // } else {
//   //   dout_padded = dout;
//   // }
//   auto dout_padded = [&]() {
//     if (head_size_og % 8 != 0) {
//       return torch::nn::functional::pad(
//           dout,
//           torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
//     } else {
//       return dout;
//     }
//   }();
//
//   // bool loop = seqlen_k > blocksize_c;
//   // TODO: change later, for now set to true for simplicity
//   bool loop = true;
//
//   // Otherwise the kernel will be launched from cuda:0 device
//   // Cast to char to avoid compiler warning about narrowing
//   // at::cuda::CUDAGuard device_guard{(char)q.get_device()};
//
//   // auto opts = q.options();
//   // auto softmax_d = torch::empty({batch_size, num_heads, seqlen_q_rounded},
//   //                               opts.dtype(at::kFloat));
//   auto softmax_d = torch::empty<dllm::CUDA>(
//       IntArrayRef3D{batch_size, num_heads, seqlen_q_rounded}, at::kFloat,
//       context);
//   // at::Tensor dq_accum;
//   // at::Tensor dk_accum, dv_accum;
//   // if (loop) {
//   //   if (!deterministic) {
//   //     dq_accum = torch::empty(
//   //         {batch_size, seqlen_q_rounded, num_heads, head_size_rounded},
//   //         opts.dtype(at::kFloat));
//   //   } else {
//   //     const int nsplits =
//   //         (dprops->multiProcessorCount + batch_size * num_heads - 1) /
//   //         (batch_size * num_heads);
//   //     dq_accum = torch::zeros(
//   //         {nsplits, batch_size, seqlen_q_rounded, num_heads,
//   //         head_size_rounded}, opts.dtype(at::kFloat));
//   //   }
//   //   // dk_accum = torch::empty({batch_size, num_heads_k, seqlen_k_rounded,
//   //   // head_size_rounded}, opts.dtype(at::kFloat)); dv_accum =
//   //   // torch::empty({batch_size, num_heads_k, seqlen_k_rounded,
//   //   // head_size_rounded}, opts.dtype(at::kFloat));
//   // }
//   auto dq_accum = [&]() {
//     if (loop) {
//       // TODO(Jie): We assume deterministic is false
//       if (deterministic != false) {
//         exit(1);
//       }
//       // TODO(Jie): We assume deterministic is false
//       if (!deterministic) {
//         return torch::empty<dllm::CUDA>(
//             IntArrayRef4D{batch_size, seqlen_q_rounded, num_heads,
//                           head_size_rounded},
//             at::kFloat, context);
//       } else {
//         const int nsplits =
//             (dprops->multiProcessorCount + batch_size * num_heads - 1) /
//             (batch_size * num_heads);
//         auto p = torch::empty<dllm::CUDA>(
//             IntArrayRef5D{nsplits, batch_size, seqlen_q_rounded, num_heads,
//                           head_size_rounded},
//             at::kFloat, context);
//         CHECK_CUDART(cudaMemsetAsync(
//             p->data(), 0, dllm::toByte(p->dtype) * cute::size(p->layout),
//             context->cudaStream));
//         return p;
//       }
//       // dk_accum = torch::empty({batch_size, num_heads_k, seqlen_k_rounded,
//       // head_size_rounded}, opts.dtype(at::kFloat)); dv_accum =
//       // torch::empty({batch_size, num_heads_k, seqlen_k_rounded,
//       // head_size_rounded}, opts.dtype(at::kFloat));
//     }
//   }();
//
//   // at::Tensor dk_expanded, dv_expanded;
//   // if (num_heads_k != num_heads) {  // MQA / GQA
//   //   dk_expanded =
//   //       torch::empty({batch_size, seqlen_k, num_heads, head_size}, opts);
//   //   dv_expanded =
//   //       torch::empty({batch_size, seqlen_k, num_heads, head_size}, opts);
//   // } else {
//   //   dk_expanded = dk;
//   //   dv_expanded = dv;
//   // }
//   auto [dk_expanded, dv_expanded] = [&]() {
//     if (num_heads_k != num_heads) {  // MQA / GQA
//       auto dk_expanded = torch::empty<dllm::CUDA>(
//           IntArrayRef4D{batch_size, seqlen_k, num_heads, head_size}, q.dtype,
//           context);
//       auto dv_expanded = torch::empty<dllm::CUDA>(
//           IntArrayRef4D{batch_size, seqlen_k, num_heads, head_size}, q.dtype,
//           context);
//       return std::make_tuple(dk_expanded, dv_expanded);
//     } else {
//       auto dk_expanded = dk;
//       auto dv_expanded = dv;
//       return std::make_tuple(dk_expanded, dv_expanded);
//     }
//   }();
//
//   Flash_bwd_params params;
//
//   set_params_dgrad(params, batch_size, seqlen_q, seqlen_k, seqlen_q_rounded,
//                    seqlen_k_rounded, num_heads, num_heads_k, head_size,
//                    head_size_rounded, q, k, v, out, dout_padded, dq,
//                    dk_expanded, dv_expanded, nullptr, nullptr,
//                    loop ? dq_accum.data_ptr() : nullptr,
//                    // loop ? dk_accum.data_ptr() : nullptr,
//                    // loop ? dv_accum.data_ptr() : nullptr,
//                    nullptr, nullptr, softmax_lse.data_ptr(),
//                    softmax_d->data_ptr(), p_dropout, softmax_scale,
//                    window_size_left, window_size_right, deterministic);
//   params.dq_accum_split_stride = !deterministic ? 0 : dq_accum.stride(0);
//
//   auto launch = &run_mha_bwd;
//
//   // auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
//   //     gen_, at::cuda::detail::getDefaultCUDAGenerator());
//
//   // We use a custom RNG that increases the offset by batch_size * nheads
//   * 32. int64_t counter_offset = params.b * params.h * 32;
//
//   if (rng_state.has_value()) {
//     params.rng_state =
//         reinterpret_cast<uint64_t *>(rng_state.value().data_ptr());
//   } else if (is_dropout) {
//     // See Note [Acquire lock when using random generators]
//     // std::lock_guard<std::mutex> lock(gen->mutex_);
//     // params.philox_args = gen->philox_cuda_state(counter_offset);
//     params.philox_args = {context->curandSeed, context->curandOffset.load()};
//     context->curandOffset += counter_offset;
//     auto seeds = at::cuda::philox::unpack(params.philox_args);
//     params.rng_state[0] = std::get<0>(seeds);
//     params.rng_state[1] = std::get<1>(seeds);
//   }
//
//   set_params_alibi(params, alibi_slopes_, batch_size, num_heads);
//
//   if (seqlen_q > 0) {
//     launch(params, stream);
//   } else {
//     // If seqlen_q == 0, then we have an empty tensor. We need to set the
//     output
//     // to 0.
//     // dk_expanded.zero_();
//     // dv_expanded.zero_();
//     // softmax_d.zero_();
//     CHECK_CUDART(cudaMemsetAsync(
//         dk_expanded->data(), 0,
//         dllm::toByte(dk_expanded->dtype) * cute::size(dk_expanded->layout),
//         context->cudaStream));
//     CHECK_CUDART(cudaMemsetAsync(
//         dv_expanded->data(), 0,
//         dllm::toByte(dv_expanded->dtype) * cute::size(dv_expanded->layout),
//         context->cudaStream));
//     CHECK_CUDART(cudaMemsetAsync(
//         softmax_d->data(), 0,
//         dllm::toByte(softmax_d->dtype) * cute::size(softmax_d->layout),
//         context->cudaStream));
//   }
//
//   // For MQA/GQA we need to sum dK and dV across the groups
//   if (num_heads_k != num_heads) {
//     at::sum_out(dk,
//                 at::reshape(dk_expanded, {batch_size, seqlen_k, num_heads_k,
//                                           num_heads / num_heads_k,
//                                           head_size}),
//                 {3});
//     at::sum_out(dv,
//                 at::reshape(dv_expanded, {batch_size, seqlen_k, num_heads_k,
//                                           num_heads / num_heads_k,
//                                           head_size}),
//                 {3});
//   }
//   if (head_size_og % 8 != 0) {
//     dq = dq.index(
//         {"...", torch::indexing::Slice(torch::indexing::None,
//         head_size_og)});
//     dk = dk.index(
//         {"...", torch::indexing::Slice(torch::indexing::None,
//         head_size_og)});
//     dv = dv.index(
//         {"...", torch::indexing::Slice(torch::indexing::None,
//         head_size_og)});
//   }
//
//   return std::make_tuple(dq, dk, dv, softmax_d);
// }

// auto mha_varlen_bwd(
//     const dllm::ContextCompute *context,
//     const std::shared_ptr<at::Tensor<3>>
//         &dout,  // total_q x num_heads, x head_size
//     const at::Tensor<3>
//         &q,  // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b}
//         s_i
//     const at::Tensor<3>
//         &k,  // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b}
//         s_i
//     const at::Tensor<3>
//         &v,  // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b}
//         s_i
//     at::Tensor<3> &out,                // total_q x num_heads x head_size
//     const at::Tensor<3> &softmax_lse,  // b x h x s   softmax logsumexp
//     c10::optional<std::shared_ptr<at::Tensor<3>>>
//         &dq_,  // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b}
//         s_i
//     c10::optional<std::shared_ptr<at::Tensor<3>>>
//         &dk_,  // total_k x num_heads_k x head_size,
//                // total_k := \sum_{i=0}^{b} s_i
//     c10::optional<std::shared_ptr<at::Tensor<3>>>
//         &dv_,                           // total_k x num_heads_k x head_size,
//                                         // total_k := \sum_{i=0}^{b} s_i
//     const at::Tensor<1> &cu_seqlens_q,  // b+1
//     const at::Tensor<1> &cu_seqlens_k,  // b+1
//     c10::optional<at::Tensor<1>> &alibi_slopes_,  // num_heads or b x
//     num_heads const int max_seqlen_q, const int max_seqlen_k,  // max
//     sequence length to choose the kernel const float p_dropout,   //
//     probability to drop const float softmax_scale, const bool zero_tensors,
//     const bool is_causal, int window_size_left, int window_size_right, const
//     bool deterministic, c10::optional<at::Tensor<1>> &rng_state) {
// #ifdef FLASHATTENTION_DISABLE_BACKWARD
//   TORCH_CHECK(false, "This flash attention build does not support
//   backward.");
// #endif
//
//   if (is_causal) {
//     window_size_right = 0;
//   }
//   auto dprops = at::cuda::getCurrentDeviceProperties();
//   // bool is_sm75 = dprops->major == 7 && dprops->minor == 5;
//   bool is_sm8x = dprops->major == 8 && dprops->minor >= 0;
//   bool is_sm80 = dprops->major == 8 && dprops->minor == 0;
//   bool is_sm90 = dprops->major == 9 && dprops->minor == 0;
//   TORCH_CHECK(is_sm90 || is_sm8x,
//               "FlashAttention only supports Ampere GPUs or newer.");
//   // We will support Turing in the near future
//   // TORCH_CHECK(is_sm90 || is_sm8x || is_sm75, "FlashAttention only supports
//   // Turing GPUs or newer.");
//   bool is_dropout = p_dropout > 0.0;
//   // auto stream = at::cuda::getCurrentCUDAStream().stream();
//   auto stream = context->cudaStream;
//
//   auto q_dtype = q.dtype;
//   TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16,
//               "FlashAttention only support fp16 and bf16 data type");
//   if (q_dtype == torch::kBFloat16) {
//     TORCH_CHECK(is_sm90 || is_sm8x,
//                 "bfloat16 is only supported on Ampere GPUs or newer");
//   }
//   TORCH_CHECK(k.dtype == q_dtype, "query and key must have the same dtype");
//   TORCH_CHECK(v.dtype == q_dtype, "query and value must have the same
//   dtype"); TORCH_CHECK(out.dtype == q_dtype, "query and out must have the
//   same dtype"); TORCH_CHECK(dout->dtype == q_dtype,
//               "query and dout must have the same dtype");
//   TORCH_CHECK(cu_seqlens_q.dtype == torch::kInt32,
//               "cu_seqlens_q must have dtype int32");
//   TORCH_CHECK(cu_seqlens_k.dtype == torch::kInt32,
//               "cu_seqlens_k must have dtype int32");
//
//   CHECK_DEVICE(q);
//   CHECK_DEVICE(k);
//   CHECK_DEVICE(v);
//   CHECK_DEVICE(out);
//   CHECK_DEVICE(*dout);
//   CHECK_DEVICE(softmax_lse);
//   CHECK_DEVICE(cu_seqlens_q);
//   CHECK_DEVICE(cu_seqlens_k);
//
//   // TORCH_CHECK(q.stride(-1) == 1,
//   //             "Input tensor must have contiguous last dimension");
//   // TORCH_CHECK(k.stride(-1) == 1,
//   //             "Input tensor must have contiguous last dimension");
//   // TORCH_CHECK(v.stride(-1) == 1,
//   //             "Input tensor must have contiguous last dimension");
//   // TORCH_CHECK(out.stride(-1) == 1,
//   //             "out tensor must have contiguous last dimension");
//   // TORCH_CHECK(dout.stride(-1) == 1,
//   //             "dout tensor must have contiguous last dimension");
//   TORCH_CHECK(q.stride<-1>() == 1,
//               "Input tensor must have contiguous last dimension");
//   TORCH_CHECK(k.stride<-1>() == 1,
//               "Input tensor must have contiguous last dimension");
//   TORCH_CHECK(v.stride<-1>() == 1,
//               "Input tensor must have contiguous last dimension");
//   TORCH_CHECK(out.stride<-1>() == 1,
//               "out tensor must have contiguous last dimension");
//   TORCH_CHECK(dout->stride<-1>() == 1,
//               "dout tensor must have contiguous last dimension");
//   // CHECK_CONTIGUOUS(cu_seqlens_q);
//   // CHECK_CONTIGUOUS(cu_seqlens_k);
//
//   const auto sizes = q.sizes();
//
//   const int total_q = sizes[0];
//   const int batch_size = cu_seqlens_q.numel() - 1;
//   const int num_heads = sizes[1];
//   // const int head_size_og = dout.size(2);
//   const int head_size_og = dout->size<2>();
//   const int head_size = sizes[2];
//   // const int total_k = k.size(0);
//   // const int num_heads_k = k.size(1);
//   const int total_k = k.size<0>();
//   const int num_heads_k = k.size<1>();
//   TORCH_CHECK(batch_size > 0, "batch size must be positive");
//   TORCH_CHECK(head_size % 8 == 0, "head_size should be a multiple of 8");
//   TORCH_CHECK(
//       head_size <= 256,
//       "FlashAttention backward only supports head dimension at most 256");
//   if (head_size > 192 && (head_size <= 224 || is_dropout)) {
//     TORCH_CHECK(
//         is_sm80 || is_sm90,
//         "FlashAttention backward for head dim 256 with dropout, or head dim "
//         "224 with/without dropout requires A100/A800 or H100/H800");
//   }
//   TORCH_CHECK(
//       num_heads % num_heads_k == 0,
//       "Number of heads in key/value must divide number of heads in query");
//
//   auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
//   const int head_size_rounded = round_multiple(head_size, 32);
//   const int seqlen_q_rounded = round_multiple(max_seqlen_q, 128);
//   const int seqlen_k_rounded = round_multiple(max_seqlen_k, 128);
//
//   TORCH_CHECK(head_size == round_multiple(head_size_og, 8),
//               "head_size must be head_size_og rounded to a multiple of 8");
//
//   if (window_size_left >= max_seqlen_k) {
//     window_size_left = -1;
//   }
//   if (window_size_right >= max_seqlen_k) {
//     window_size_right = -1;
//   }
//
//   // CHECK_SHAPE(q, total_q, num_heads, head_size);
//   // CHECK_SHAPE(k, total_k, num_heads_k, head_size);
//   // CHECK_SHAPE(v, total_k, num_heads_k, head_size);
//   // CHECK_SHAPE(out, total_q, num_heads, head_size);
//   // CHECK_SHAPE(dout, total_q, num_heads, head_size_og);
//   // CHECK_SHAPE(cu_seqlens_q, batch_size + 1);
//   // CHECK_SHAPE(cu_seqlens_k, batch_size + 1);
//   CHECK_SHAPE(q, IntArrayRef3D{total_q, num_heads, head_size});
//   CHECK_SHAPE(k, IntArrayRef3D{total_k, num_heads_k, head_size});
//   CHECK_SHAPE(v, IntArrayRef3D{total_k, num_heads_k, head_size});
//   CHECK_SHAPE(out, IntArrayRef3D{total_q, num_heads, head_size});
//   CHECK_SHAPE(*dout, IntArrayRef3D{total_q, num_heads, head_size_og});
//   CHECK_SHAPE(cu_seqlens_q, IntArrayRef1D{batch_size + 1});
//   CHECK_SHAPE(cu_seqlens_k, IntArrayRef1D{batch_size + 1});
//
//   // at::Tensor dq, dk, dv;
//   // if (dq_.has_value()) {
//   //   dq = dq_.value();
//   //   TORCH_CHECK(dq.dtype() == q_dtype, "dq must have the same dtype as
//   q");
//   //   CHECK_DEVICE(dq);
//   //   TORCH_CHECK(dq.stride(-1) == 1, "dq must have contiguous last
//   //   dimension"); CHECK_SHAPE(dq, total_q, num_heads, head_size);
//   // } else {
//   //   dq = torch::empty_like(q);
//   // }
//   // if (dk_.has_value()) {
//   //   dk = dk_.value();
//   //   TORCH_CHECK(dk.dtype() == q_dtype, "dk must have the same dtype as
//   q");
//   //   CHECK_DEVICE(dk);
//   //   TORCH_CHECK(dk.stride(-1) == 1, "dk must have contiguous last
//   //   dimension"); CHECK_SHAPE(dk, total_k, num_heads_k, head_size);
//   // } else {
//   //   dk = torch::empty_like(k);
//   // }
//   // if (dv_.has_value()) {
//   //   dv = dv_.value();
//   //   TORCH_CHECK(dv.dtype() == q_dtype, "dv must have the same dtype as
//   q");
//   //   CHECK_DEVICE(dv);
//   //   TORCH_CHECK(dv.stride(-1) == 1, "dv must have contiguous last
//   //   dimension"); CHECK_SHAPE(dv, total_k, num_heads_k, head_size);
//   // } else {
//   //   dv = torch::empty_like(v);
//   // }
//   auto [dq, dk, dv] = [&]() {
//     std::shared_ptr<at::Tensor<3>> dq, dk, dv;
//     if (dq_.has_value()) {
//       dq = dq_.value();
//       TORCH_CHECK(dq->dtype == q_dtype, "dq must have the same dtype as q");
//       CHECK_DEVICE(*dq);
//       TORCH_CHECK(dq->stride<-1>() == 1,
//                   "dq must have contiguous last dimension");
//       CHECK_SHAPE(*dq, IntArrayRef3D{total_q, num_heads, head_size});
//     } else {
//       dq = torch::empty_like<dllm::CUDA>(q, context);
//     }
//     if (dk_.has_value()) {
//       dk = dk_.value();
//       TORCH_CHECK(dk->dtype == q_dtype, "dk must have the same dtype as q");
//       CHECK_DEVICE(*dk);
//       TORCH_CHECK(dk->stride<-1>() == 1,
//                   "dk must have contiguous last dimension");
//       CHECK_SHAPE(*dk, IntArrayRef3D{total_k, num_heads_k, head_size});
//     } else {
//       dk = torch::empty_like<dllm::CUDA>(k, context);
//     }
//     if (dv_.has_value()) {
//       dv = dv_.value();
//       TORCH_CHECK(dv->dtype == q_dtype, "dv must have the same dtype as q");
//       CHECK_DEVICE(*dv);
//       TORCH_CHECK(dv->stride<-1>() == 1,
//                   "dv must have contiguous last dimension");
//       CHECK_SHAPE(*dv, IntArrayRef3D{total_k, num_heads_k, head_size});
//     } else {
//       dv = torch::empty_like<dllm::CUDA>(v, context);
//     }
//     return std::make_tuple(dq, dk, dv);
//   }();
//
//   // at::Tensor dout_padded;
//   // if (head_size_og % 8 != 0) {
//   //   dout_padded = torch::nn::functional::pad(
//   //       dout, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og %
//   //       8}));
//   // } else {
//   //   dout_padded = dout;
//   // }
//   auto dout_padded = [&]() {
//     if (head_size_og % 8 != 0) {
//       return torch::nn::functional::pad(
//           dout,
//           torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
//     } else {
//       return dout;
//     }
//   }();
//
//   // bool loop = max_seqlen_k > blocksize_c;
//   // TODO: change later, for now set to true for simplicity
//   bool loop = true;
//
//   // Otherwise the kernel will be launched from cuda:0 device
//   // Cast to char to avoid compiler warning about narrowing
//   // at::cuda::CUDAGuard device_guard{(char)q.get_device()};
//
//   // auto opts = q.options();
//   // auto softmax_d = torch::empty({batch_size, num_heads, seqlen_q_rounded},
//   //                               opts.dtype(at::kFloat));
//   auto softmax_d = torch::empty<dllm::CUDA>(
//       IntArrayRef3D{batch_size, num_heads, seqlen_q_rounded}, at::kFloat,
//       context);
//
//   // at::Tensor dq_accum;
//   // if (loop) {
//   //   // We don't want to allocate dq_accum of size (batch,
//   seqlen_q_rounded,
//   //   // num_heads, head_size_rounded) because that would be too large if
//   there
//   //   is
//   //   // a very long sequence and the rest of the sequences are short.
//   Instead,
//   //   we
//   //   // allocate dq_accum of size (total_q + 128 * batch, num_heads,
//   //   // head_size_rounded). Note that 128 is the max block size on the
//   //   seqlen_q
//   //   // dimension. For dQ, the i-th sequence is stored in indices from
//   //   // cu_seqlens[i] + 128 * i to cu_seqlens[i + 1] * 128 * i - 1. This
//   //   ensures
//   //   // that the i-th sequence and (i + 1)-th sequence will be at least 128
//   //   // apart. It's ok for us to do atomicAdds up to 128 rows beyond what
//   //   we're
//   //   // normally allowed to do. So we won't have to do any bound checking,
//   and
//   //   // performance should stay the same.
//   //   if (!deterministic) {
//   //     dq_accum = torch::empty(
//   //         {total_q + 128 * batch_size, num_heads, head_size_rounded},
//   //         opts.dtype(at::kFloat));
//   //   } else {
//   //     const int nsplits =
//   //         (dprops->multiProcessorCount + batch_size * num_heads - 1) /
//   //         (batch_size * num_heads);
//   //     dq_accum = torch::zeros(
//   //         {nsplits, total_q + 128 * batch_size, num_heads,
//   //         head_size_rounded}, opts.dtype(at::kFloat));
//   //   }
//   // }
//   auto dq_accum = [&]() {
//     if (loop) {
//       // We don't want to allocate dq_accum of size (batch, seqlen_q_rounded,
//       // num_heads, head_size_rounded) because that would be too large if
//       there
//       // is a very long sequence and the rest of the sequences are short.
//       // Instead, we allocate dq_accum of size (total_q + 128 * batch,
//       // num_heads, head_size_rounded). Note that 128 is the max block size
//       on
//       // the seqlen_q dimension. For dQ, the i-th sequence is stored in
//       indices
//       // from cu_seqlens[i] + 128 * i to cu_seqlens[i + 1] * 128 * i - 1.
//       This
//       // ensures that the i-th sequence and (i + 1)-th sequence will be at
//       least
//       // 128 apart. It's ok for us to do atomicAdds up to 128 rows beyond
//       what
//       // we're normally allowed to do. So we won't have to do any bound
//       // checking, and performance should stay the same.
//       if (!deterministic) {
//         return torch::empty<dllm::CUDA>(
//             IntArrayRef3D{total_q + 128 * batch_size, num_heads,
//                           head_size_rounded},
//             at::kFloat, context);
//       } else {
//         const int nsplits =
//             (dprops->multiProcessorCount + batch_size * num_heads - 1) /
//             (batch_size * num_heads);
//         auto p = torch::empty<dllm::CUDA>(
//             IntArrayRef4D{nsplits, total_q + 128 * batch_size, num_heads,
//                           head_size_rounded},
//             at::kFloat, context);
//         CHECK_CUDART(cudaMemsetAsync(
//             p->data(), 0, dllm::toByte(p->dtype) * cute::size(p->layout),
//             context->cudaStream));
//         return p;
//       }
//     }
//   }();
//
//   // at::Tensor dk_expanded, dv_expanded;
//   // if (num_heads_k != num_heads) {  // MQA / GQA
//   //   dk_expanded = torch::empty({total_k, num_heads, head_size}, opts);
//   //   dv_expanded = torch::empty({total_k, num_heads, head_size}, opts);
//   // } else {
//   //   dk_expanded = dk;
//   //   dv_expanded = dv;
//   // }
//   auto [dk_expanded, dv_expanded] = [&]() {
//     std::shared_ptr<at::Tensor<3>> dk_expanded, dv_expanded;
//     if (num_heads_k != num_heads) {  // MQA / GQA
//       dk_expanded = torch::empty<dllm::CUDA>(
//           IntArrayRef3D{total_k, num_heads, head_size}, q.dtype, context);
//       dv_expanded = torch::empty<dllm::CUDA>(
//           IntArrayRef3D{total_k, num_heads, head_size}, q.dtype, context);
//     } else {
//       dk_expanded = dk;
//       dv_expanded = dv;
//     }
//     return std::make_tuple(dk_expanded, dv_expanded);
//   }();
//
//   if (zero_tensors) {
//     // dq.zero_();
//     // dk_expanded.zero_();
//     // dv_expanded.zero_();
//     // softmax_d.zero_();
//     CHECK_CUDART(cudaMemsetAsync(
//         dq->data(), 0, dllm::toByte(dq->dtype) * cute::size(dq->layout),
//         context->cudaStream));
//     CHECK_CUDART(cudaMemsetAsync(
//         dk_expanded->data(), 0,
//         dllm::toByte(dk_expanded->dtype) * cute::size(dk_expanded->layout),
//         context->cudaStream));
//     CHECK_CUDART(cudaMemsetAsync(
//         dv_expanded->data(), 0,
//         dllm::toByte(dv_expanded->dtype) * cute::size(dv_expanded->layout),
//         context->cudaStream));
//     CHECK_CUDART(cudaMemsetAsync(
//         softmax_d->data(), 0,
//         dllm::toByte(softmax_d->dtype) * cute::size(softmax_d->layout),
//         context->cudaStream));
//   }
//
//   Flash_bwd_params params;
//
//   set_params_dgrad(
//       params, batch_size, max_seqlen_q, max_seqlen_k, seqlen_q_rounded,
//       seqlen_k_rounded, num_heads, num_heads_k, head_size, head_size_rounded,
//       q, k, v, out, dout_padded, dq, dk_expanded, dv_expanded,
//       cu_seqlens_q.data_ptr(), cu_seqlens_k.data_ptr(),
//       loop ? dq_accum.data_ptr() : nullptr, nullptr, nullptr,
//       softmax_lse.data_ptr(), softmax_d->data_ptr(), p_dropout,
//       softmax_scale, window_size_left, window_size_right, deterministic);
//   params.dq_accum_split_stride = !deterministic ? 0 : dq_accum.stride(0);
//
//   auto launch = &run_mha_bwd;
//
//   // auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
//   //     gen_, at::cuda::detail::getDefaultCUDAGenerator());
//
//   // We use a custom RNG that increases the offset by batch_size * nheads
//   * 32. int64_t counter_offset = params.b * params.h * 32;
//
//   if (rng_state.has_value()) {
//     params.rng_state =
//         reinterpret_cast<uint64_t *>(rng_state.value().data_ptr());
//   } else if (is_dropout) {
//     // See Note [Acquire lock when using random generators]
//     // std::lock_guard<std::mutex> lock(gen->mutex_);
//     // params.philox_args = gen->philox_cuda_state(counter_offset);
//     params.philox_args = {context->curandSeed, context->curandOffset.load()};
//     context->curandOffset += counter_offset;
//     auto seeds = at::cuda::philox::unpack(params.philox_args);
//     params.rng_state[0] = std::get<0>(seeds);
//     params.rng_state[1] = std::get<1>(seeds);
//   }
//
//   set_params_alibi(params, alibi_slopes_, batch_size, num_heads);
//
//   if (max_seqlen_q > 0) {
//     launch(params, stream);
//   } else {
//     // If seqlen_q == 0, then we have an empty tensor. We need to set the
//     output
//     // to 0.
//     // dk_expanded.zero_();
//     // dv_expanded.zero_();
//     // softmax_d.zero_();
//     CHECK_CUDART(cudaMemsetAsync(
//         dk_expanded->data(), 0,
//         dllm::toByte(dk_expanded->dtype) * cute::size(dk_expanded->layout),
//         context->cudaStream));
//     CHECK_CUDART(cudaMemsetAsync(
//         dv_expanded->data(), 0,
//         dllm::toByte(dv_expanded->dtype) * cute::size(dv_expanded->layout),
//         context->cudaStream));
//     CHECK_CUDART(cudaMemsetAsync(
//         softmax_d->data(), 0,
//         dllm::toByte(softmax_d->dtype) * cute::size(softmax_d->layout),
//         context->cudaStream));
//   }
//
//   // For MQA/GQA we need to sum dK and dV across the groups
//   if (num_heads_k != num_heads) {
//     at::sum_out(dk,
//                 at::reshape(dk_expanded, {total_k, num_heads_k,
//                                           num_heads / num_heads_k,
//                                           head_size}),
//                 {2});
//     at::sum_out(dv,
//                 at::reshape(dv_expanded, {total_k, num_heads_k,
//                                           num_heads / num_heads_k,
//                                           head_size}),
//                 {2});
//   }
//   if (head_size_og % 8 != 0) {
//     dq = dq.index(
//         {"...", torch::indexing::Slice(torch::indexing::None,
//         head_size_og)});
//     dk = dk.index(
//         {"...", torch::indexing::Slice(torch::indexing::None,
//         head_size_og)});
//     dv = dv.index(
//         {"...", torch::indexing::Slice(torch::indexing::None,
//         head_size_og)});
//   }
//
//   return std::make_tuple(dq, dk, dv, softmax_d);
// }
//
// auto mha_fwd_kvcache(
//     const dllm::ContextCompute *context,
//     at::Tensor<4> &q,  // batch_size x seqlen_q x num_heads x head_size
//     const at::Tensor<4>
//         &kcache,  // batch_size_c x seqlen_k x num_heads_k x head_size or
//                   // num_blocks x page_block_size x num_heads_k x head_size
//                   if
//                   // there's a block_table.
//     const at::Tensor<4>
//         &vcache,  // batch_size_c x seqlen_k x num_heads_k x head_size or
//                   // num_blocks x page_block_size x num_heads_k x head_size
//                   if
//                   // there's a block_table.
//     c10::optional<std::shared_ptr<const at::Tensor<4>>>
//         &k_,  // batch_size x seqlen_knew x num_heads_k x head_size
//     c10::optional<std::shared_ptr<const at::Tensor<4>>>
//         &v_,  // batch_size x seqlen_knew x num_heads_k x head_size
//     c10::optional<const at::Tensor<1>> &seqlens_k_,  // batch_size
//     c10::optional<const at::Tensor<2>>
//         &rotary_cos_,  // seqlen_ro x (rotary_dim / 2)
//     c10::optional<const at::Tensor<2>>
//         &rotary_sin_,  // seqlen_ro x (rotary_dim / 2)
//     c10::optional<const at::Tensor<1>>
//         &cache_batch_idx_,  // indices to index into the KV cache
//     c10::optional<std::shared_ptr<at::Tensor<2>>>
//         &block_table_,  // batch_size x max_num_blocks_per_seq
//     c10::optional<at::Tensor<1>>
//         &alibi_slopes_,  // num_heads or batch_size x num_heads
//     c10::optional<std::shared_ptr<at::Tensor<4>>>
//         &out_,  // batch_size x seqlen_q x num_heads x head_size
//     const float softmax_scale, bool is_causal, int window_size_left,
//     int window_size_right,
//     bool is_rotary_interleaved,  // if true, rotary combines indices 0 & 1,
//     else
//                                  // indices 0 & rotary_dim / 2
//     int num_splits) {
//   auto dprops = at::cuda::getCurrentDeviceProperties();
//   // bool is_sm75 = dprops->major == 7 && dprops->minor == 5;
//   bool is_sm8x = dprops->major == 8 && dprops->minor >= 0;
//   bool is_sm90 = dprops->major == 9 && dprops->minor == 0;
//   TORCH_CHECK(is_sm90 || is_sm8x,
//               "FlashAttention only supports Ampere GPUs or newer.");
//   // We will support Turing in the near future
//   // TORCH_CHECK(is_sm90 || is_sm8x || is_sm75, "FlashAttention only supports
//   // Turing GPUs or newer.");
//
//   auto q_dtype = q.dtype;
//   TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16,
//               "FlashAttention only support fp16 and bf16 data type");
//   if (q_dtype == torch::kBFloat16) {
//     TORCH_CHECK(is_sm90 || is_sm8x,
//                 "bfloat16 is only supported on Ampere GPUs or newer");
//   }
//   TORCH_CHECK(kcache.dtype == q_dtype,
//               "query and key must have the same dtype");
//   TORCH_CHECK(vcache.dtype == q_dtype,
//               "query and value must have the same dtype");
//
//   CHECK_DEVICE(q);
//   CHECK_DEVICE(kcache);
//   CHECK_DEVICE(vcache);
//
//   // TORCH_CHECK(q.stride(-1) == 1,
//   //             "Input tensor must have contiguous last dimension");
//   // TORCH_CHECK(kcache.stride(-1) == 1,
//   //             "Input tensor must have contiguous last dimension");
//   // TORCH_CHECK(vcache.stride(-1) == 1,
//   //             "Input tensor must have contiguous last dimension");
//   TORCH_CHECK(q.stride<-1>() == 1,
//               "Input tensor must have contiguous last dimension");
//   TORCH_CHECK(kcache.stride<-1>() == 1,
//               "Input tensor must have contiguous last dimension");
//   TORCH_CHECK(vcache.stride<-1>() == 1,
//               "Input tensor must have contiguous last dimension");
//
//   // at::Tensor block_table;
//   const bool paged_KV = block_table_.has_value();
//   // if (paged_KV) {
//   //   TORCH_CHECK(!cache_batch_idx_.has_value(),
//   //               "Paged KVcache does not support cache_batch_idx");
//   //   block_table = block_table_.value();
//   //   CHECK_DEVICE(block_table);
//   //   TORCH_CHECK(block_table.dtype() == torch::kInt32,
//   //               "block_table must have dtype torch.int32");
//   //   TORCH_CHECK(block_table.stride(-1) == 1,
//   //               "block_table must have contiguous last dimension");
//   // }
//   auto block_table = [&]() {
//     if (paged_KV) {
//       TORCH_CHECK(!cache_batch_idx_.has_value(),
//                   "Paged KVcache does not support cache_batch_idx");
//       auto block_table = block_table_.value();
//       CHECK_DEVICE(*block_table);
//       TORCH_CHECK(block_table->dtype == torch::kInt32,
//                   "block_table must have dtype torch.int32");
//       TORCH_CHECK(block_table->stride<-1>() == 1,
//                   "block_table must have contiguous last dimension");
//       return block_table;
//     }
//     return std::remove_reference_t<decltype(block_table_.value())>{};
//   };
//
//   const auto sizes = q.sizes();
//
//   const int batch_size = sizes[0];
//   int seqlen_q = sizes[1];
//   int num_heads = sizes[2];
//   const int head_size_og = sizes[3];
//
//   // const int max_num_blocks_per_seq = !paged_KV ? 0 : block_table.size(1);
//   const int max_num_blocks_per_seq = !paged_KV ? 0 : block_table->size<1>();
//   // const int num_blocks = !paged_KV ? 0 : kcache.size(0);
//   // const int page_block_size = !paged_KV ? 1 : kcache.size(1);
//   const int num_blocks = !paged_KV ? 0 : kcache.size<0>();
//   const int page_block_size = !paged_KV ? 1 : kcache.size<1>();
//   TORCH_CHECK(!paged_KV || page_block_size % 256 == 0,
//               "Paged KV cache block size must be divisible by 256");
//   // const int seqlen_k =
//   //     !paged_KV ? kcache.size(1) : max_num_blocks_per_seq *
//   page_block_size;
//   // const int num_heads_k = kcache.size(2);
//   // const int batch_size_c = !paged_KV ? kcache.size(0) : batch_size;
//   const int seqlen_k =
//       !paged_KV ? kcache.size<1>() : max_num_blocks_per_seq *
//       page_block_size;
//   const int num_heads_k = kcache.size<2>();
//   const int batch_size_c = !paged_KV ? kcache.size<0>() : batch_size;
//   TORCH_CHECK(batch_size > 0, "batch size must be postive");
//   TORCH_CHECK(
//       head_size_og <= 256,
//       "FlashAttention forward only supports head dimension at most 256");
//   TORCH_CHECK(
//       num_heads % num_heads_k == 0,
//       "Number of heads in key/value must divide number of heads in query");
//
//   // causal=true is the same as causal=false in this case
//   if (seqlen_q == 1 && !alibi_slopes_.has_value()) {
//     is_causal = false;
//   }
//   if (is_causal) {
//     window_size_right = 0;
//   }
//
//   // Faster to transpose q from (b, 1, (nheads_kv ngroups), d) to (b,
//   ngroups,
//   // nheads_kv, d) in this case H/t Daniel Haziza
//   const int seqlenq_ngroups_swapped =
//       seqlen_q == 1 && num_heads > num_heads_k && window_size_left < 0 &&
//       window_size_right < 0 && head_size_og % 8 == 0 &&
//       !alibi_slopes_.has_value();
//   if (seqlenq_ngroups_swapped) {
//     const int ngroups = num_heads / num_heads_k;
//     q = q.reshape({batch_size, num_heads_k, ngroups, head_size_og})
//             .transpose(1, 2);
//     seqlen_q = ngroups;
//     num_heads = num_heads_k;
//   }
//
//   if (window_size_left >= seqlen_k) {
//     window_size_left = -1;
//   }
//   if (window_size_right >= seqlen_k) {
//     window_size_right = -1;
//   }
//
//   // CHECK_SHAPE(q, batch_size, seqlen_q, num_heads, head_size_og);
//   // if (!paged_KV) {
//   //   CHECK_SHAPE(kcache, batch_size_c, seqlen_k, num_heads_k,
//   head_size_og);
//   //   CHECK_SHAPE(vcache, batch_size_c, seqlen_k, num_heads_k,
//   head_size_og);
//   // } else {
//   //   CHECK_SHAPE(kcache, num_blocks, page_block_size, num_heads_k,
//   //   head_size_og); CHECK_SHAPE(vcache, num_blocks, page_block_size,
//   //   num_heads_k, head_size_og); CHECK_SHAPE(block_table, batch_size,
//   //   max_num_blocks_per_seq);
//   // }
//   CHECK_SHAPE(q, IntArrayRef4D{batch_size, seqlen_q, num_heads,
//   head_size_og}); if (!paged_KV) {
//     CHECK_SHAPE(kcache, IntArrayRef4D{batch_size_c, seqlen_k, num_heads_k,
//                                       head_size_og});
//     CHECK_SHAPE(vcache, IntArrayRef4D{batch_size_c, seqlen_k, num_heads_k,
//                                       head_size_og});
//   } else {
//     CHECK_SHAPE(kcache, IntArrayRef4D{num_blocks, page_block_size,
//     num_heads_k,
//                                       head_size_og});
//     CHECK_SHAPE(vcache, IntArrayRef4D{num_blocks, page_block_size,
//     num_heads_k,
//                                       head_size_og});
//     CHECK_SHAPE(*block_table,
//                 IntArrayRef2D{batch_size, max_num_blocks_per_seq});
//   }
//
//   auto [q_padded, kcache_padded, vcache_padded] = [&]() {
//     if (head_size_og % 8 != 0) {
//       auto q_padded = torch::nn::functional::pad(
//           q, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og %
//           8}));
//       auto kcache_padded = torch::nn::functional::pad(
//           kcache,
//           torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
//       auto vcache_padded = torch::nn::functional::pad(
//           vcache,
//           torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
//       return std::make_tuple(q_padded, kcache_padded, vcache_padded);
//     } else {
//       auto q_padded = q;
//       auto kcache_padded = kcache;
//       auto vcache_padded = vcache;
//       return std::make_tuple(q_padded, kcache_padded, vcache_padded);
//     }
//   }();
//   // at::Tensor q_padded, kcache_padded, vcache_padded;
//   // if (head_size_og % 8 != 0) {
//   //   q_padded = torch::nn::functional::pad(
//   //       q, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og %
//   8}));
//   //   kcache_padded = torch::nn::functional::pad(
//   //       kcache,
//   //       torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
//   //   vcache_padded = torch::nn::functional::pad(
//   //       vcache,
//   //       torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
//   // } else {
//   //   q_padded = q;
//   //   kcache_padded = kcache;
//   //   vcache_padded = vcache;
//   // }
//
//   // at::Tensor out;
//   // if (out_.has_value()) {
//   //   out = out_.value();
//   //   TORCH_CHECK(out.dtype() == q_dtype,
//   //               "Output must have the same dtype as inputs");
//   //   CHECK_DEVICE(out);
//   //   TORCH_CHECK(out.stride(-1) == 1,
//   //               "Output tensor must have contiguous last dimension");
//   //   CHECK_SHAPE(out, batch_size, seqlen_q, num_heads, head_size_og);
//   //   if (head_size_og % 8 != 0) {
//   //     out = torch::empty_like(q_padded);
//   //   }
//   // } else {
//   //   out = torch::empty_like(q_padded);
//   // }
//   auto out = [&]() {
//     if (out_.has_value()) {
//       auto out = out_.value();
//       TORCH_CHECK(out->dtype == q_dtype,
//                   "Output must have the same dtype as inputs");
//       CHECK_DEVICE(*out);
//       TORCH_CHECK(out->stride<-1>() == 1,
//                   "Output tensor must have contiguous last dimension");
//       CHECK_SHAPE(*out,
//                   IntArrayRef4D{batch_size, seqlen_q, num_heads,
//                   head_size_og});
//       if (head_size_og % 8 != 0) {
//         return torch::empty_like<dllm::CUDA>(q_padded, context);
//       }
//       return out;
//     } else {
//       return torch::empty_like<dllm::CUDA>(q_padded, context);
//     }
//   }();
//
//   auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
//   const int head_size = round_multiple(head_size_og, 8);
//   const int head_size_rounded = round_multiple(head_size, 32);
//   const int seqlen_q_rounded = round_multiple(seqlen_q, 128);
//   const int seqlen_k_rounded = round_multiple(seqlen_k, 128);
//
//   // Otherwise the kernel will be launched from cuda:0 device
//   // Cast to char to avoid compiler warning about narrowing
//   // at::cuda::CUDAGuard device_guard{(char)q.get_device()};
//
//   // auto opts = q.options();
//
//   // auto softmax_lse =
//   //     torch::empty({batch_size, num_heads, seqlen_q},
//   //     opts.dtype(at::kFloat));
//   auto softmax_lse = torch::empty<dllm::CUDA>(
//       IntArrayRef3D{batch_size, num_heads, seqlen_q}, at::kFloat, context);
//
//   Flash_fwd_params params;
//   set_params_fprop(
//       params, batch_size, seqlen_q, seqlen_k, seqlen_q_rounded,
//       seqlen_k_rounded, num_heads, num_heads_k, head_size, head_size_rounded,
//       q_padded, kcache_padded, vcache_padded, out,
//       /*cu_seqlens_q_d=*/nullptr,
//       /*cu_seqlens_k_d=*/nullptr,
//       /*seqused_k=*/nullptr,
//       /*p_ptr=*/nullptr, softmax_lse->data_ptr(),
//       /*p_dropout=*/0.f, softmax_scale, window_size_left, window_size_right);
//
//   // at::Tensor k, v, k_padded, v_padded;
//   // if (k_.has_value()) {
//   //   TORCH_CHECK(v_.has_value(),
//   //               "If key is supplied, value must also be passed in");
//   //   TORCH_CHECK(seqlens_k_.has_value(),
//   //               "If key is supplied, seqlens_k must also be passed in");
//   //   TORCH_CHECK(seqlen_q <= seqlen_k,
//   //               "If key is supplied, it must have seqlen <= the seqlen of
//   the
//   //               " "KV cache");
//   //   k = k_.value();
//   //   v = v_.value();
//   //   TORCH_CHECK(k.dtype() == q_dtype, "Key must have the same dtype as
//   //   query"); TORCH_CHECK(v.dtype() == q_dtype,
//   //               "Value must have the same dtype as query");
//   //   CHECK_DEVICE(k);
//   //   CHECK_DEVICE(v);
//   //   TORCH_CHECK(k.stride(-1) == 1,
//   //               "Key tensor must have contiguous last dimension");
//   //   TORCH_CHECK(v.stride(-1) == 1,
//   //               "Value tensor must have contiguous last dimension");
//   //   int seqlen_knew = k.size(1);
//   //   CHECK_SHAPE(k, batch_size, seqlen_knew, num_heads_k, head_size_og);
//   //   CHECK_SHAPE(v, batch_size, seqlen_knew, num_heads_k, head_size_og);
//   //   if (head_size_og % 8 != 0) {
//   //     k_padded = torch::nn::functional::pad(
//   //         k, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og %
//   //         8}));
//   //     v_padded = torch::nn::functional::pad(
//   //         v, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og %
//   //         8}));
//   //   } else {
//   //     k_padded = k;
//   //     v_padded = v;
//   //   }
//   //   params.seqlen_knew = seqlen_knew;
//   //   params.knew_ptr = k_padded.data_ptr();
//   //   params.vnew_ptr = v_padded.data_ptr();
//   //   // All stride are in elements, not bytes.
//   //   params.knew_batch_stride = k_padded.stride(0);
//   //   params.vnew_batch_stride = v_padded.stride(0);
//   //   params.knew_row_stride = k_padded.stride(-3);
//   //   params.vnew_row_stride = v_padded.stride(-3);
//   //   params.knew_head_stride = k_padded.stride(-2);
//   //   params.vnew_head_stride = v_padded.stride(-2);
//   // }
//   auto [k, v, k_padded, v_padded] = [&]() {
//     if (k_.has_value()) {
//       TORCH_CHECK(v_.has_value(),
//                   "If key is supplied, value must also be passed in");
//       TORCH_CHECK(seqlens_k_.has_value(),
//                   "If key is supplied, seqlens_k must also be passed in");
//       TORCH_CHECK(
//           seqlen_q <= seqlen_k,
//           "If key is supplied, it must have seqlen <= the seqlen of the "
//           "KV cache");
//       auto k = k_.value();
//       auto v = v_.value();
//       TORCH_CHECK(k->dtype == q_dtype, "Key must have the same dtype as
//       query"); TORCH_CHECK(v->dtype == q_dtype,
//                   "Value must have the same dtype as query");
//       CHECK_DEVICE(*k);
//       CHECK_DEVICE(*v);
//       TORCH_CHECK(k->stride<-1>() == 1,
//                   "Key tensor must have contiguous last dimension");
//       TORCH_CHECK(v->stride<-1>() == 1,
//                   "Value tensor must have contiguous last dimension");
//       int seqlen_knew = k->size<1>();
//       CHECK_SHAPE(*k, IntArrayRef4D{batch_size, seqlen_knew, num_heads_k,
//                                     head_size_og});
//       CHECK_SHAPE(*v, IntArrayRef4D{batch_size, seqlen_knew, num_heads_k,
//                                     head_size_og});
//       auto [k_padded, v_padded] = [&]() {
//         if (head_size_og % 8 != 0) {
//           auto k_padded = torch::nn::functional::pad(
//               k,
//               torch::nn::functional::PadFuncOptions({0, 8 - head_size_og %
//               8}));
//           auto v_padded = torch::nn::functional::pad(
//               v,
//               torch::nn::functional::PadFuncOptions({0, 8 - head_size_og %
//               8}));
//           return std::make_tuple(k_padded, v_padded);
//         } else {
//           auto k_padded = k;
//           auto v_padded = v;
//           return std::make_tuple(k_padded, v_padded);
//         }
//       }();
//
//       params.seqlen_knew = seqlen_knew;
//       params.knew_ptr = k_padded.data_ptr();
//       params.vnew_ptr = v_padded.data_ptr();
//       // All stride are in elements, not bytes.
//       params.knew_batch_stride = k_padded.stride(0);
//       params.vnew_batch_stride = v_padded.stride(0);
//       params.knew_row_stride = k_padded.stride(-3);
//       params.vnew_row_stride = v_padded.stride(-3);
//       params.knew_head_stride = k_padded.stride(-2);
//       params.vnew_head_stride = v_padded.stride(-2);
//       return std::make_tuple(k, v, k_padded, v_padded);
//     }
//     return std::make_tuple(std::remove_reference_t<decltype(k_.value())>{},
//                            std::remove_reference_t<decltype(v_.value())>{},
//                            std::remove_reference_t<decltype(k_.value())>{},
//                            std::remove_reference_t<decltype(v_.value())>{});
//   }();
//
//   if (seqlens_k_.has_value()) {
//     auto seqlens_k = seqlens_k_.value();
//     TORCH_CHECK(seqlens_k.dtype == torch::kInt32,
//                 "seqlens_k must have dtype int32");
//     CHECK_DEVICE(seqlens_k);
//     // CHECK_CONTIGUOUS(seqlens_k);
//     // CHECK_SHAPE(seqlens_k, batch_size);
//     CHECK_SHAPE(seqlens_k, IntArrayRef1D{batch_size});
//     params.cu_seqlens_k = static_cast<int *>(seqlens_k.data_ptr());
//   }
//   params.is_seqlens_k_cumulative = !(seqlens_k_.has_value());
//
//   if (rotary_cos_.has_value()) {
//     TORCH_CHECK(k_.has_value(),
//                 "If rotary cos/sin are provided, new key / value to be "
//                 "appended to KV cache must also be provided");
//     auto rotary_cos = rotary_cos_.value();
//     CHECK_DEVICE(rotary_cos);
//     // params.rotary_dim = rotary_cos.size(1) * 2;
//     params.rotary_dim = rotary_cos.size<1>() * 2;
//     TORCH_CHECK(params.rotary_dim <= head_size,
//                 "rotary_dim must be <= headdim");
//     TORCH_CHECK(
//         params.rotary_dim % 16 == 0,
//         "Only rotary dimensions divisible by 16 are currently supported");
//     // const int seqlen_ro = rotary_cos.size(0);
//     const int seqlen_ro = rotary_cos.size<0>();
//     TORCH_CHECK(seqlen_ro >= seqlen_k,
//                 "cos/sin seqlen must be at least the seqlen of KV cache");
//     // CHECK_SHAPE(rotary_cos, seqlen_ro, params.rotary_dim / 2);
//     CHECK_SHAPE(rotary_cos, IntArrayRef2D{seqlen_ro, params.rotary_dim / 2});
//     // CHECK_CONTIGUOUS(rotary_cos);
//     TORCH_CHECK(rotary_cos.scalar_type() == q_dtype,
//                 "rotary_cos must have the same dtype as query");
//
//     TORCH_CHECK(rotary_sin_.has_value(),
//                 "If rotary cos is provided, rotary sin must also be
//                 provided");
//     auto rotary_sin = rotary_sin_.value();
//     CHECK_DEVICE(rotary_sin);
//     CHECK_SHAPE(rotary_sin, IntArrayRef2D{seqlen_ro, params.rotary_dim / 2});
//     // CHECK_CONTIGUOUS(rotary_sin);
//     TORCH_CHECK(rotary_sin.scalar_type() == q_dtype,
//                 "rotary_cos must have the same dtype as query");
//     params.rotary_cos_ptr = rotary_cos.data_ptr();
//     params.rotary_sin_ptr = rotary_sin.data_ptr();
//     params.is_rotary_interleaved = is_rotary_interleaved;
//   } else {
//     params.rotary_dim = 0;
//   }
//
//   if (cache_batch_idx_.has_value()) {
//     auto cache_batch_idx = cache_batch_idx_.value();
//     CHECK_DEVICE(cache_batch_idx);
//     // CHECK_CONTIGUOUS(cache_batch_idx);
//     TORCH_CHECK(cache_batch_idx.scalar_type() == torch::kInt32,
//                 "cache_batch_idx must have dtype int32");
//     params.cache_batch_idx =
//         reinterpret_cast<int *>(cache_batch_idx.data_ptr());
//   }
//
//   // set_params_splitkv(params, batch_size, num_heads, head_size, seqlen_k,
//   //                    seqlen_q, head_size_rounded, /*dropout*/ 0.f,
//   //                    num_splits, dprops, opts);
//   set_params_splitkv(context, params, batch_size, num_heads, head_size,
//                      seqlen_k, seqlen_q, head_size_rounded, /*dropout*/ 0.f,
//                      num_splits, dprops);
//
//   if (paged_KV) {
//     // params.block_table = block_table.data_ptr<int>();
//     // params.block_table_batch_stride = block_table.stride(0);
//     params.block_table = block_table->data_ptr<int>();
//     params.block_table_batch_stride = block_table->stride<0>();
//   }
//   params.page_block_size = page_block_size;
//
//   set_params_alibi(params, alibi_slopes_, batch_size, num_heads);
//
//   // auto stream = at::cuda::getCurrentCUDAStream().stream();
//   auto stream = context->cudaStream;
//   // Only split kernel supports appending to KV cache, or indexing to the
//   cache
//   // with cache_batch_idx, or paged KV cache
//   run_mha_fwd(params, stream, /*force_split_kernel=*/k_.has_value() ||
//                                   cache_batch_idx_.has_value() || paged_KV);
//
//   if (head_size_og % 8 != 0) {
//     out = out.index(
//         {"...", torch::indexing::Slice(torch::indexing::None,
//         head_size_og)});
//     if (out_.has_value()) {
//       // out_.value().copy_(out);
//       CHECK_CUDART(
//           cudaMemcpyAsync(out_.value()->data(), out->data(),
//                           dllm::toByte(out->dtype) * cute::size(out->layout),
//                           cudaMemcpyDeviceToDevice, context->cudaStream));
//     }
//     if (k_.has_value()) {
//       // It's expensive to copy the KV cache here for the case where head
//       size
//       // not divisible by 8, but we don't expect to get this case in
//       practice.
//       // This is just so that the code works for that case.
//       kcache.copy_(kcache_padded.index(
//           {"...",
//            torch::indexing::Slice(torch::indexing::None, head_size_og)}));
//       vcache.copy_(vcache_padded.index(
//           {"...",
//            torch::indexing::Slice(torch::indexing::None, head_size_og)}));
//     }
//   }
//
//   if (seqlenq_ngroups_swapped) {
//     out = out.transpose(1, 2).reshape(
//         {batch_size, 1, num_heads_k * seqlen_q, head_size_og});
//     softmax_lse = softmax_lse.reshape({batch_size, num_heads_k * seqlen_q,
//     1});
//   }
//   return std::make_tuple(out, softmax_lse);
// }
