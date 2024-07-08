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

def silu_forward(input: torch.Tensor):
    output = input * torch.sigmoid(input)
    save_for_backward = (input)
    return output, save_for_backward

def silu_backward(grad_output: torch.Tensor, save_for_backward: torch.Tensor):
    input = save_for_backward
    sigmoid_i = torch.sigmoid(input)
    grad_input = grad_output * (sigmoid_i * (1 + input * (1 - sigmoid_i)))
    return grad_input
