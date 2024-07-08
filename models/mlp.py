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
from operators.compute import linear
from operators.compute import relu


class MLP():

    def __init__(
        self, 
        input_size: int = 784, 
        hidden_size: int = 1024, 
        output_size: int = 10
    ):
        self.weight1 = linear.linear_init(input_size, hidden_size)
        self.weight2 = linear.linear_init(hidden_size, output_size)
    
    def forward(self, input: torch.Tensor):
        output1, save_for_linear1_backward = linear.linear_forward(self.weight1, input)
        output2, save_for_relu_backward = relu.relu_forward(output1)
        output, save_for_linear2_backward = linear.linear_forward(self.weight2, output2)
        return output, (save_for_linear1_backward, save_for_relu_backward, save_for_linear2_backward)
    
    def backward(self, grad_output: torch.Tensor, save_for_backward):
        save_for_relu_backward, save_for_linear_backward = save_for_backward
        grad_output, grad_weight2 = linear.linear_backward(grad_output, save_for_linear_backward)
        grad_output = relu.relu_backward(grad_output, save_for_relu_backward)
        grad_input, grad_weight1 = linear.linear_backward(grad_output, save_for_relu_backward)
        grad_params = (grad_weight1, grad_weight2)
        return grad_params, grad_input
