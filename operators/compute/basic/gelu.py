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

import math
import torch
import triton
import triton.language as tl


def gelu_forward(*args, backend="naive", **kwargs):
    if backend == "naive":
        return gelu_forward_naive(*args, **kwargs)
    elif backend == "torchjit":
        return gelu_forward_torchjit(*args, **kwargs)
    elif backend == "triton":
        return gelu_forward_triton(*args, **kwargs)

def gelu_backward(*args, backend="naive", **kwargs):
    if backend == "naive":
        return gelu_backward_naive(*args, **kwargs)
    elif backend == "torchjit":
        return gelu_backward_torchjit(*args, **kwargs)
    elif backend == "triton":
        return gelu_backward_triton(*args, **kwargs)



def gelu_forward_naive(input: torch.Tensor):
    cdf = 0.5 * (1.0 + torch.tanh(math.sqrt(2 / math.pi) * (input + 0.044715 * torch.pow(input, 3))))
    save_for_backward = (input)
    return input * cdf, save_for_backward

def gelu_backward_naive(grad_output: torch.Tensor, save_for_backward: tuple):
    input = save_for_backward
    cdf = 0.5 * (1.0 + torch.tanh(math.sqrt(2 / math.pi) * (input + 0.044715 * torch.pow(input, 3))))
    sigmoid_dash = 0.5 * math.sqrt(2 / math.pi) * (1 + 3 * 0.044715 * torch.pow(input, 2)) * (1 - torch.tanh(math.sqrt(2 / math.pi) * (input + 0.044715 * torch.pow(input, 3))) ** 2)
    grad_input = grad_output * (cdf + input * sigmoid_dash)
    return grad_input


def gelu_forward_torchjit(input: torch.Tensor):
    @torch.jit.script
    def compute(input: torch.Tensor):
        cdf = 0.5 * (1.0 + torch.tanh(math.sqrt(2 / math.pi) * (input + 0.044715 * torch.pow(input, 3))))
        return input * cdf
    save_for_backward = (input)
    return compute(input), save_for_backward

def gelu_backward_torchjit(grad_output: torch.Tensor, save_for_backward: tuple):
    input = save_for_backward
    @torch.jit.script
    def compute(grad_output: torch.Tensor, input: torch.Tensor):
        cdf = 0.5 * (1.0 + torch.tanh(math.sqrt(2 / math.pi) * (input + 0.044715 * torch.pow(input, 3))))
        sigmoid_dash = 0.5 * math.sqrt(2 / math.pi) * (1 + 3 * 0.044715 * torch.pow(input, 2)) * (1 - torch.tanh(math.sqrt(2 / math.pi) * (input + 0.044715 * torch.pow(input, 3))) ** 2)
        grad_input = grad_output * (cdf + input * sigmoid_dash)
        return grad_input
    return compute(grad_output, input)


@triton.jit
def _gelu_forward_kernel(X, Y, num_elements, **meta):
    idx = tl.program_id(0) * tl.num_threads(0) + tl.thread_id(0)
    if idx < num_elements:
        x = X[idx]
        # Direct calculation of GELU
        cdf = 0.5 * (1.0 + tl.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * tl.pow(x, 3))))
        Y[idx] = x * cdf

@triton.jit
def _gelu_backward_kernel(grad_Y, X, grad_X, num_elements, **meta):
    idx = tl.program_id(0) * tl.num_threads(0) + tl.thread_id(0)
    if idx < num_elements:
        x = X[idx]
        grad_y = grad_Y[idx]
        cdf = 0.5 * (1.0 + tl.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * tl.pow(x, 3))))
        tanh_out = tl.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * tl.pow(x, 3)))
        sigmoid_dash = 0.5 * math.sqrt(2 / math.pi) * (1 + 3 * 0.044715 * tl.pow(x, 2)) * (1 - tanh_out ** 2)
        grad_x = grad_y * (cdf + x * sigmoid_dash)
        grad_X[idx] = grad_x

def gelu_forward_triton(input: torch.Tensor):
    num_elements = input.numel()
    output = torch.empty_like(input)
    grid = (triton.cdiv(num_elements, 1024),)
    _gelu_forward_kernel[grid](input, output, num_elements)
    save_for_backward = (input)
    return output, save_for_backward

def gelu_backward_triton(grad_output: torch.Tensor, save_for_backward: tuple):
    input = save_for_backward
    num_elements = input.numel()
    grad_input = torch.empty_like(input)
    grid = (triton.cdiv(num_elements, 1024),)
    _gelu_backward_kernel[grid](grad_output, input, grad_input, num_elements)
    return grad_input


