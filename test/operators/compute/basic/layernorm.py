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

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')))
import torch

from operators.compute import (
    layernorm,
)

if __name__=="__main__":
    B, T, d = 2, 1024, 512
    device = torch.device("cuda:0")
    dtype = torch.float16
    tol = 1e-2

    # forward check
    normalized_shape = [d]
    weight, bias, init_info = layernorm.layernorm_init(normalized_shape, device=device, dtype=dtype)
    input = torch.randn(B, T, d).to(device=device, dtype=dtype)
    input1 = input.detach().clone().requires_grad_()
    weight1 = weight.detach().clone().requires_grad_()
    bias1 = bias.detach().clone().requires_grad_()
    output1 = torch.nn.functional.layer_norm(input1, normalized_shape, weight1, bias1)   # baseline
    input2 = input.clone()
    output2, save_for_backward = layernorm.layernorm_forward(input2, weight, bias, init_info)   # ours
    are_close = torch.allclose(output1, output2, atol=tol)
    print("Forward results are close?", are_close)
    print("output: \n", output1[...,:10], "\n", output2[...,:10])

    # backward check
    grad_output = torch.randn_like(output1).to(device=device, dtype=dtype)
    grad_input1, grad_weight1, grad_bias1 = torch.autograd.grad(
        outputs=output1, inputs=[input1, weight1, bias1], grad_outputs=grad_output, create_graph=True)   # baseline
    grad_input2, weight2, bias2 = layernorm.layernorm_backward(grad_output, weight, bias, save_for_backward)    # ours
    are_close = torch.allclose(grad_input1, grad_input2, atol=tol)
    print("Backward grad_input are close?", are_close)
    print("grad_input: \n", grad_input1[...,:10], "\n", grad_input2[...,:10])
    are_close = torch.allclose(grad_weight1, weight2.grad, atol=tol)
    print("Backward grad_weight are close?", are_close)
    print("grad_weight: \n", grad_weight1[...,:10], "\n", weight2.grad[...,:10])
    are_close = torch.allclose(grad_bias1, bias2.grad, atol=tol)
    print("Backward grad_bias are close?", are_close)
    print("grad_bias: \n", grad_bias1[...,:10], "\n", bias2.grad[...,:10])
