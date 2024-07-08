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
    embedding,
)

if __name__=="__main__":
    B, T, vocab, d = 2, 1024, 50257, 768
    device = torch.device("cuda:0")
    dtype = torch.float64
    tol = 1e-4

    # forward check
    weight, init_info = embedding.embedding_init(vocab, d, device=device, dtype=dtype)
    input = torch.randint(low=0, high=vocab, size=(B, T)).to(device=device, dtype=torch.long)
    input1 = input.detach().clone()     # embedding do not accept input that requires gradient computing
    weight1 = weight.detach().clone().requires_grad_()
    output1 = torch.nn.functional.embedding(input1, weight1)   # baseline
    output2, save_for_backward = embedding.embedding_forward(input, weight, init_info)   # ours
    are_close = torch.allclose(output1, output2, atol=tol)
    print("Forward results are close?", are_close)

    # backward check
    grad_output = torch.randn_like(output1).to(device=device, dtype=dtype)
    grad_weight1 = torch.autograd.grad(
        outputs=output1, inputs=[weight1], grad_outputs=grad_output, create_graph=True)[0]   # baseline
    grad_input2, weight = embedding.embedding_backward(grad_output, weight, save_for_backward)    # ours
    are_close = torch.allclose(grad_weight1, weight.grad, atol=tol)
    print("Backward grad_weight are close?", are_close)
