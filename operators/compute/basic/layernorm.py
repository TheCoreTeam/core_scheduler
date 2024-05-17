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
    device=None, 
    dtype=None
):
    if elementwise_affine:
        weight = torch.ones(normalized_shape, device=device, dtype=dtype)
        bias = torch.zeros(normalized_shape, device=device, dtype=dtype) if bias else None
    else:
        weight, bias = None, None
    init_info = (normalized_shape, eps, elementwise_affine)
    return weight, bias, init_info

def layernorm_forward(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, init_info: tuple):
    normalized_shape, eps, elementwise_affine = init_info
    last_dims = tuple(range(-len(normalized_shape), 0))
    mean = input.mean(dim=last_dims, keepdim=True)
    var = input.var(dim=last_dims, keepdim=True, unbiased=False)
    output_buffer = (input - mean) / torch.sqrt(var + eps)
    if elementwise_affine:
        output = output_buffer * weight + bias
    save_for_backward = (output, output_buffer, normalized_shape, weight, bias, mean, var, eps, elementwise_affine)
    return output, save_for_backward

def layernorm_backward(grad_output: torch.Tensor, save_for_backward: tuple):
    output, output_buffer, normalized_shape, weight, bias, mean, var, eps, elementwise_affine = save_for_backward
    # Compute the normalization dimensions based on normalized_shape
    norm_dims = tuple(range(-len(normalized_shape), 0))  # This computes it as the last dimensions
    non_norm_dims = tuple(range(len(grad_output)-len(normalized_shape)+1))  # This computes it as the last dimensions

    if elementwise_affine:
        # Compute gradients for weight and bias
        grad_weight = (grad_output * output_buffer).sum(dim=non_norm_dims, keepdim=True)
        grad_bias = grad_output.sum(dim=non_norm_dims, keepdim=True)
        # Prepare grad_output for input gradient calculation
        grad_output = grad_output * weight

    # Gradient with respect to the input
    std_inv = 1 / torch.sqrt(var + eps)
    grad_input = grad_output * std_inv

    # Additional corrections for the mean and variance contributions
    grad_mean = (-std_inv * grad_output).sum(dim=norm_dims, keepdim=True)
    grad_var = (-0.5 * std_inv.pow(3) * (output - mean) * grad_output).sum(dim=norm_dims, keepdim=True)
    grad_input += (grad_var * 2 / torch.tensor(normalized_shape).prod() * (output - mean)) + \
                  (grad_mean / torch.tensor(normalized_shape).prod())

    return grad_input, grad_weight.squeeze() if elementwise_affine else None, grad_bias.squeeze() if elementwise_affine else None

