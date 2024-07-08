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
from typing import Tuple
import math

from . import init

def linear_init(
    input_size: int, 
    output_size: int, 
    device: torch.device = 'cuda:0',
    dtype=torch.float32,
    training=True,
):
    weight = torch.randn(output_size, input_size).to(device=device, dtype=dtype)
    init.kaiming_uniform_(weight, a=math.sqrt(5))
    if training:
        weight.grad = torch.zeros_like(weight, device=weight.device, dtype=weight.dtype)
    init_info = None
    return weight, init_info

def linear_forward(input: torch.Tensor, weight: torch.Tensor, init_info: tuple):
    if input.dim() < 2:
        raise ValueError("Input tensor must be at least 2D")
    save_for_backward = (input)
    return input @ weight.t(), save_for_backward

def linear_backward(grad_output: torch.Tensor, weight: torch.Tensor, save_for_backward: Tuple[torch.Tensor]):
    input = save_for_backward
    grad_input = grad_output @ weight
    assert grad_input.shape == input.shape, "Gradient input shape mismatch"
    if input.dim() > 2:
        # Flatten input to [batch_size, features] if necessary
        input = input.reshape(-1, input.size(-1))
        grad_output = grad_output.reshape(-1, grad_output.size(-1))
    grad_weight = grad_output.t() @ input
    assert grad_weight.shape == weight.shape, "Gradient weight shape mismatch"
    weight.grad += grad_weight
    return grad_input, weight
