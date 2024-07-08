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
from typing import Tuple, Union, List

from . import init

_shape_t = Union[int, List[int], torch.Size]

def layernorm_init(
    normalized_shape: _shape_t, 
    eps: float = 1e-5, 
    elementwise_affine: bool = True,
    bias: bool = True, 
    device: torch.device = 'cuda:0', 
    dtype=torch.float32, 
    training=True,
):
    if len(normalized_shape) != 1:
        raise NotImplementedError("Currently layernorm only support normalized_dim=-1.")
    if elementwise_affine:
        is_bias = bias
        weight = torch.ones(normalized_shape, device=device, dtype=dtype)
        bias = torch.zeros(normalized_shape, device=device, dtype=dtype) if is_bias else None
        if training:
            weight.grad = torch.zeros_like(weight, device=weight.device, dtype=weight.dtype)
            bias.grad = torch.zeros_like(bias, device=bias.device, dtype=bias.dtype) if is_bias else None
    else:
        raise NotImplementedError("Currently layernorm only support elementwise_affine=True")
    init_info = (normalized_shape[0], eps, elementwise_affine)
    return weight, bias, init_info

def layernorm_forward(*args, backend="naive", **kwargs):
    if backend == "naive":
        return layernorm_forward_naive(*args, **kwargs)
    elif backend == "torchjit":
        return layernorm_forward_torchjit(*args, **kwargs)
    # elif backend == "triton":
    #     return layernorm_forward_triton(*args, **kwargs)

def layernorm_backward(*args, backend="naive", **kwargs):
    if backend == "naive":
        return layernorm_backward_naive(*args, **kwargs)
    elif backend == "torchjit":
        return layernorm_backward_torchjit(*args, **kwargs)
    # elif backend == "triton":
    #     return layernorm_backward_triton(*args, **kwargs)



def layernorm_forward_naive(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, init_info: tuple):
    normalized_shape, eps, elementwise_affine = init_info
    last_dims = -1
    mean = input.mean(dim=last_dims, keepdim=True)
    var = input.var(dim=last_dims, keepdim=True, unbiased=False)
    output_buffer = (input - mean) / torch.sqrt(var + eps)
    if elementwise_affine:
        output = output_buffer * weight + bias
    save_for_backward = (output, output_buffer, normalized_shape, mean, var, eps, elementwise_affine)
    return output, save_for_backward

def layernorm_backward_naive(grad_output: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, save_for_backward: tuple):
    output, output_buffer, normalized_shape, mean, var, eps, elementwise_affine = save_for_backward
    # Compute the normalization dimensions based on normalized_shape
    norm_dim = -1  # This computes it as the last dimensions
    non_norm_dims = tuple(range(len(grad_output.shape)-1))  # This computes it as the last dimensions

    if elementwise_affine:
        # Compute gradients for weight and bias
        grad_weight = (grad_output * output_buffer).sum(dim=non_norm_dims, keepdim=True)
        weight.grad += grad_weight.squeeze()
        grad_bias = grad_output.sum(dim=non_norm_dims, keepdim=True)
        bias.grad += grad_bias.squeeze()
        # Prepare grad_output for input gradient calculation
        grad_output = grad_output * weight

    # Gradient with respect to the input
    std_inv = 1 / torch.sqrt(var + eps)
    grad_input = grad_output * std_inv

    # Additional corrections for the mean and variance contributions
    grad_mean = (-std_inv * grad_output).sum(dim=norm_dim, keepdim=True)
    grad_var = (-0.5 * std_inv.pow(3) * (output - mean) * grad_output).sum(dim=norm_dim, keepdim=True)
    grad_input += (grad_var * 2 / torch.tensor(normalized_shape).prod() * (output - mean)) + \
                  (grad_mean / torch.tensor(normalized_shape).prod())

    return grad_input, weight if elementwise_affine else None, bias if elementwise_affine else None



def layernorm_forward_torchjit(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, init_info: tuple):
    normalized_shape, eps, elementwise_affine = init_info
    @torch.jit.script
    def compute(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, elementwise_affine: bool, eps: float):
        mean = input.mean(dim=-1, keepdim=True)
        var = input.var(dim=-1, keepdim=True, unbiased=False)
        output = (input - mean) / torch.sqrt(var + eps)
        output = output * weight + bias
        return output
    save_for_backward = (input, normalized_shape, eps, elementwise_affine)
    return compute(input, weight, bias, elementwise_affine, eps), save_for_backward

def layernorm_backward_torchjit(grad_output: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, save_for_backward: tuple):
    input, normalized_shape, eps, elementwise_affine = save_for_backward
    # Compute the normalization dimensions based on normalized_shape
    @torch.jit.script
    def compute(grad_output: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, normalized_shape: int, eps: float, elementwise_affine: bool):
        norm_dim = -1  # This computes it as the last dimensions
        # non_norm_dims = range(len(grad_output.shape)-1)  # This computes it as the last dimensions
        non_norm_dims = (i for i in range(len(grad_output.shape)-1))  # This computes it as the last dimensions
        mean = input.mean(dim=norm_dim, keepdim=True)
        var = input.var(dim=norm_dim, keepdim=True, unbiased=False)
        output_buffer = (input - mean) / torch.sqrt(var + eps)
        output = output_buffer * weight + bias
        grad_weight = (grad_output * output_buffer).sum(dim=non_norm_dims, keepdim=True)
        # weight.grad += grad_weight.squeeze()
        grad_bias = grad_output.sum(dim=non_norm_dims, keepdim=True)
        # bias.grad += grad_bias.squeeze()
        # Prepare grad_output for input gradient calculation
        grad_output = grad_output * weight
        # Gradient with respect to the input
        std_inv = 1 / torch.sqrt(var + eps)
        grad_input = grad_output * std_inv
        # Additional corrections for the mean and variance contributions
        grad_mean = (-std_inv * grad_output).sum(dim=norm_dim, keepdim=True)
        grad_var = (-0.5 * std_inv.pow(3) * (output - mean) * grad_output).sum(dim=norm_dim, keepdim=True)
        grad_input += (grad_var * 2 / torch.tensor(normalized_shape).prod() * (output - mean)) + \
                    (grad_mean / torch.tensor(normalized_shape).prod())
        return grad_input, grad_weight.squeeze(), grad_bias.squeeze()
    grad_input, weight.grad, bias.grad = compute(grad_output, input, weight, bias, normalized_shape, eps, elementwise_affine)
    return grad_input, weight, bias
