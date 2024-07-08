# Copyright (c) 2024 The Core Team
#
# Licensed under the Apache License, Version 2.0
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

from triton import heuristics, jit
from triton import language as tl
from triton import next_power_of_2


def cross_entropy_loss_forward(
    output: torch.Tensor, 
    target: torch.Tensor, 
    reduction: str = 'mean', 
    label_smoothing: float = 0.0,
    epsilon: float = 1e-10,
):
    
    # make sure we can use triton
    assert (target.dtype == torch.int64), "Indices are expected to be of type long."
    # make kernel
    device, dtype = output.device, output.dtype
    n_cols = output.shape[-1]
    # run the kernel
    result = torch.empty_like(target, dtype=dtype, device=device)
    neg_logprobs = torch.empty_like(output, dtype=dtype, device=device)
    grid = lambda opt: (output.numel() // n_cols, )
    _forward[grid](output, neg_logprobs, target, result, n_cols)
    # save for backward
    save_for_backward = (neg_logprobs, target)
    return result, save_for_backward


def cross_entropy_loss_backward(save_for_backward: list, dneg_logprobs=None):
    # Load saved tensors
    neg_logprobs, target = save_for_backward
    # Assuming 'dneg_logprobs' should be initialized here if not provided
    if dneg_logprobs is None:
        # Initialize gradients; assuming a typical scenario where each sample's gradient contributes equally
        dneg_logprobs = torch.ones_like(neg_logprobs) / neg_logprobs.shape[0]
    # Run the kernel
    # 'neg_logprobs' will be modified in place to become our gradient:
    n_cols = neg_logprobs.shape[-1]
    grid = lambda opt: (neg_logprobs.numel() // n_cols, )
    _backward[grid](neg_logprobs, target, dneg_logprobs, n_cols)
    return neg_logprobs



def num_warps(N):
    if N < 2048:
        return 4
    elif N < 8192:
        return 8
    return 16


@heuristics({'num_warps': lambda nargs: num_warps(nargs['N'])})
@heuristics({'BLOCK': lambda nargs: next_power_of_2(nargs['N'])})
@jit
def _forward(LOGITS, PROBS, IDX, LOSS, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    idx = tl.load(IDX + row)
    # pointers to logit and probs
    LOGITS = LOGITS + row * N + cols
    WRIT_PROBS = PROBS + row * N + cols
    READ_PROBS = PROBS + row * N + idx
    # write-back negative log-probs
    logits = tl.load(LOGITS, mask=cols < N, other=-float('inf'))
    logits = logits.to(tl.float32)
    logits = logits - tl.max(logits, 0)
    probs = tl.log(tl.sum(tl.exp(logits), 0)) - logits
    tl.store(WRIT_PROBS, probs, mask=cols < N)
    # There is a bug in the compiler, which fails to insert a barrier here.
    # We add it explicitly for now. Will be fixed soon.
    tl.debug_barrier()
    # write-back loss
    probs = tl.load(READ_PROBS)
    tl.store(LOSS + row, probs)


@heuristics({'num_warps': lambda nargs: num_warps(nargs['N'])})
@heuristics({'BLOCK': lambda nargs: next_power_of_2(nargs['N'])})
@jit
def _backward(PROBS, IDX, DPROBS, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    idx = tl.load(IDX + row)
    # pointers to probs
    PROBS = PROBS + row * N + cols
    # We know d(-log(p[i])/dlogit[k] = -id_mat[i,k] + p[k]
    # and we have -log(p[k]) stored in PROBS, so this is easy
    probs = -tl.load(PROBS, mask=cols < N, other=float('inf'))
    probs = tl.exp(probs.to(tl.float32))
    delta = cols == idx
    # write result in-place in PROBS
    dout = tl.load(DPROBS + row)
    din = (probs - delta) * dout
    tl.store(PROBS, din.to(PROBS.dtype.element_ty), mask=cols < N)