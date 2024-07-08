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
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from typing import List
from tqdm.auto import tqdm
from dataclasses import dataclass
from collections import OrderedDict
import math
import torch
from flash_attn import flash_attn_func
from flash_attn.flash_attn_interface import _flash_attn_forward as flash_attn_forward
from flash_attn.flash_attn_interface import _flash_attn_backward as flash_attn_backward

from models.gpt2.single_device.gpt2 import GPTConfig, GPT2Model


if __name__=="__main__":

    # Initialize the GPT configuration
    config = GPTConfig()
    config.device = torch.device("cuda:0")
    config.dtype = torch.bfloat16

    # Create two instances of the GPT model
    manual_model = GPT2Model(config)
    autograd_model = GPT2Model(config)

    # Recursive function to set requires_grad for all parameters
    def set_requires_grad(parameters):
        if isinstance(parameters, torch.Tensor):
            parameters.requires_grad_(True)
        elif isinstance(parameters, dict):
            for param in parameters.values():
                set_requires_grad(param)
        elif isinstance(parameters, list):
            for param in parameters:
                set_requires_grad(param)

    # Set requires_grad for all parameters in the autograd model
    set_requires_grad(autograd_model.get_parameters())

    # Set manual_model parameters to autograd_model
    manual_model.set_parameters(autograd_model.get_parameters())

    # Generate random input_ids and labels for testing
    num_sample = 1024
    batch_size = 2
    sequence_length = 512  # Must be less than or equal to block_size
    input_ids = torch.randint(low=0, high=config.vocab_size, size=(batch_size, sequence_length)).to(config.device)
    labels = torch.randint(low=0, high=config.vocab_size, size=(batch_size, sequence_length)).to(config.device)

    # Run the forward pass for both models
    manual_logits, manual_loss, manual_save_for_backward = manual_model.forward(input_ids, labels)
    autograd_logits, autograd_loss, _ = autograd_model.forward(input_ids, labels)

    # Check if logits are the same
    # print(manual_logits, "\n", autograd_logits)
    print("Logits close:", torch.allclose(manual_logits, autograd_logits, atol=1e-2))
    print("Loss close:", torch.allclose(manual_loss, autograd_loss, atol=1e-2))

    # Compute the gradients for manual backward implementation
    manual_model.backward(manual_save_for_backward)

    # Compute the gradients using autograd
    autograd_loss.backward()

    # Recursive function to get gradients
    def get_grad(param):
        if isinstance(param, torch.Tensor):
            return param.grad
        elif isinstance(param, dict):
            return {k: get_grad(v) for k, v in param.items()}
        elif isinstance(param, list):
            return [get_grad(p) for p in param]
        else:
            raise TypeError("Unsupported parameter type")

    # Get gradients from autograd model
    autograd_grads = get_grad(autograd_model.get_parameters())

    # Get gradients from manual model
    manual_grads = get_grad(manual_model.get_parameters())

    # Recursive function to compare gradients
    def compare_grads(manual_grad, autograd_grad, name=""):
        if isinstance(manual_grad, torch.Tensor):
            if manual_grad is None or autograd_grad is None:
                return manual_grad is None and autograd_grad is None
            return torch.allclose(manual_grad, autograd_grad, atol=1e-2)
        elif isinstance(manual_grad, dict):
            return all(compare_grads(manual_grad[k], autograd_grad[k], name + "." + k) for k in manual_grad)
        elif isinstance(manual_grad, list):
            return all(compare_grads(mg, ag, name + f"[{i}]") for i, (mg, ag) in enumerate(zip(manual_grad, autograd_grad)))
        else:
            raise TypeError("Unsupported gradient type")

    # Compare the gradients of each parameter
    grads_match = compare_grads(manual_grads, autograd_grads)
    print("Gradients match:", grads_match)

    print(manual_model.weight_pe.grad)
    print(autograd_model.weight_pe.grad)
    # print(manual_model.h[5].attn.weight_attn.grad)
    # print(autograd_model.h[5].attn.weight_attn.grad)

    # Clear gradients for the next iteration
    def clear_grads(parameters):
        if isinstance(parameters, torch.Tensor):
            if parameters.grad is not None:
                parameters.grad = None
        elif isinstance(parameters, dict):
            for param in parameters.values():
                clear_grads(param)
        elif isinstance(parameters, list):
            for param in parameters:
                clear_grads(param)

    # Clear gradients for autograd model
    clear_grads(autograd_model.get_parameters())
