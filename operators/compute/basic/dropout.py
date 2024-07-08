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
torch.nn.Dropout

def dropout_forward(input: torch.Tensor, p: float = 0.5, training: bool = True):
    if not training or p==0:
        return input, (None, p)  # During evaluation, we do not apply dropout

    # Create a dropout mask using the Bernoulli distribution
    mask = (torch.rand_like(input) > p).to(device=input.device, dtype=input.dtype)
    output = input * mask / (1 - p)
    save_for_backward = (mask, p)

    return output, save_for_backward


def dropout_backward(grad_output: torch.Tensor, save_for_backward: Tuple):
    mask, p = save_for_backward
    if p==0:
        return grad_output
    # Apply the mask and scale by the same factor as during the forward pass
    return grad_output * mask / (1 - p)
