import torch
from typing import Tuple
import math

from . import init

def linear_init(
    input_size: int, 
    output_size: int, 
    device: torch.device = 'cuda:0',
    dtype=torch.float32,
):
    weight = torch.randn(output_size, input_size).to(device=device, dtype=dtype)
    weight = init.kaiming_uniform_(weight, a=math.sqrt(5))
    init_info = None
    return weight, init_info

def linear_forward(input: torch.Tensor, weight: torch.Tensor, init_info: tuple):
    if input.dim() < 2:
        raise ValueError("Input tensor must be at least 2D")
    save_for_backward = (weight, input)
    return input @ weight.t(), save_for_backward

def linear_backward(grad_output: torch.Tensor, save_for_backward: Tuple[torch.Tensor]):
    weight, input = save_for_backward
    grad_input = grad_output @ weight
    assert grad_input.shape == input.shape, "Gradient input shape mismatch"
    if input.dim() > 2:
        # Flatten input to [batch_size, features] if necessary
        input = input.reshape(-1, input.size(-1))
        grad_output = grad_output.reshape(-1, grad_output.size(-1))
    grad_weight = grad_output.t() @ input
    assert grad_weight.shape == weight.shape, "Gradient weight shape mismatch"
    return grad_input, grad_weight
