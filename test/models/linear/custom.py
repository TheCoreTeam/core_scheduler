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
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from typing import List
from tqdm.auto import tqdm
from collections import OrderedDict
import torch
import torchvision
import torchvision.transforms as transforms

from utils import MemoryTest, TimeTest, set_seed
from operators.compute import (
    linear,
    gelu,
    relu,
    cross_entropy,
)
from operators.optim import sgd

set_seed(42)

class MLP():

    def __init__(
        self, 
        input_size: int = 784, 
        output_size: int = 10,
        device: torch.device = 'cuda:0',
        dtype: torch.dtype = torch.float32,
    ):
        self.weight1, self.init_info1 = linear.linear_init(input_size, output_size, device=device, dtype=dtype)
    
        self._init_parameters()

    def _init_parameters(self):
        for k, p in self.get_parameters().items():
            torch.nn.init.normal_(p, mean=0, std=0.2)

    def forward(self, input: torch.Tensor, label: torch.Tensor):
        output, save_for_linear1_backward = linear.linear_forward(input, self.weight1, self.init_info1)
        loss_value, save_for_loss_backward = cross_entropy.cross_entropy_loss_forward(output, label)
        return loss_value, (save_for_linear1_backward, save_for_loss_backward)
    
    def backward(self, save_for_backward):
        save_for_linear1_backward, save_for_loss_backward = save_for_backward
        grad_output = cross_entropy.cross_entropy_loss_backward(save_for_loss_backward)
        _, self.weight1.grad = linear.linear_backward(grad_output, save_for_linear1_backward)
        return None
    
    def get_parameters(self):
        parameters = OrderedDict({
            'weight1': self.weight1,
        })
        return parameters
    
    def set_parameters(self, parameters: OrderedDict):
        self.weight1 = parameters["weight1"]




if __name__ == '__main__':
    # Hyperparameters
    batch_size = 64     # global batch size
    epochs = 3
    lr = 1e-4
    momentum = 0.9

    # Data settings
    feature_dim = 784
    classes = 10
    
    # MNIST Dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: torch.flatten(x))  # Flatten the images
    ])
    trainset = torchvision.datasets.MNIST(root='../../dataset/', train=True, download=False, transform=transform)
    dataloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    # GPU settings
    device = torch.device("cuda:0")
    dtype = torch.float64

    # Create the model
    model = MLP(input_size=feature_dim, output_size=classes, device=device, dtype=dtype)
    
    # Create the optimizer
    optimizer = sgd.SGD(model.get_parameters(), lr=lr, momentum=momentum)
    
    # Training loop
    for epoch in range(epochs):
        print(f"Epoch {epoch}")
        global_loss_value = 0
        with MemoryTest(enable=True, tqdm=False, master_process=True) as mem_test:
            with TimeTest(enable=True, tqdm=False, master_process=True) as time_test:
                for idx, (input, label) in enumerate(tqdm(dataloader)):
                    # Put the input and label on GPU
                    input, label = input.to(device, dtype), label.to(device)

                    # Forward pass
                    loss_value, save_for_backward = model.forward(input, label)
                    global_loss_value += loss_value.item()
                    
                    # Backward pass
                    model.backward(save_for_backward)

                    # Update the model parameters with SGD
                    model.set_parameters(optimizer.step(model.get_parameters()))
                    
                print(f"Loss: {global_loss_value / len(dataloader)}")