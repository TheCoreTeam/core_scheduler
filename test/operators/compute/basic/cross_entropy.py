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
from typing import List

from operators.compute import (
    cross_entropy
)

if __name__=="__main__":
    # Test data
    torch.manual_seed(0)
    tol = 1e-6
    B = 64
    C = 10
    output = torch.randn(B, C) * 1e3
    target = torch.randint(0, C, (B,))

    # Compute custom cross-entropy loss
    output1 = output.clone()
    loss, save_for_backward = cross_entropy.cross_entropy_loss_forward(output1, target)
    grad_custom = cross_entropy.cross_entropy_loss_backward(save_for_backward)

    # Compute PyTorch cross-entropy loss for comparison
    output2 = output.clone()
    output2.requires_grad = True
    criterion = torch.nn.CrossEntropyLoss()
    loss_pytorch = criterion(output2, target)
    loss_pytorch.backward()

    print("Compare losses: ", (loss - loss_pytorch).abs().max())
    print("Compare gradients: ", (output2.grad - grad_custom).abs().max())
    # are_close = torch.allclose(loss, loss_pytorch, atol=tol)
    # print("Losses are close?", are_close)    
    # are_close = torch.allclose(output2.grad, grad_custom, atol=tol)
    # print("Losses are close?", are_close)