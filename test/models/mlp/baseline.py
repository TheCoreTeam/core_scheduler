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
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

from utils import MemoryTest, TimeTest, set_seed
set_seed(42)

class MLP(torch.nn.Module):

    def __init__(
        self, 
        input_size: int = 784, 
        hidden_size: int = 1024, 
        output_size: int = 10,
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        super().__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size, bias=False)
        self.fc2 = torch.nn.Linear(hidden_size, output_size, bias=False)
        self.loss = torch.nn.CrossEntropyLoss()
        
        # self._init_parameters()

    def _init_parameters(self):
        self.fc1.weight.data = torch.randn(self.hidden_size, self.input_size)
        self.fc2.weight.data = torch.randn(self.output_size, self.hidden_size)

    def forward(self, input: torch.Tensor, label: torch.Tensor):
        output = self.fc1(input)
        output = torch.nn.functional.relu(output)
        output = self.fc2(output)
        # loss_value = torch.nn.functional.cross_entropy(output, label)
        loss_value = self.loss(output, label)
        return loss_value


if __name__ == '__main__':
    # Hyperparameters
    batch_size = 64     # global batch size
    epochs = 3
    lr = 1e-4
    momentum = 0

    # Data settings
    feature_dimension = 784
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
    dtype = torch.float32

    # Create the model
    model = MLP(input_size=feature_dimension, hidden_size=8192, output_size=classes).to(device, dtype)
    
    # Create the optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    
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
                    loss_value = model.forward(input, label)
                    global_loss_value += loss_value.item()
                    
                    # Backward pass
                    loss_value.backward()

                    # Update the model parameters with SGD
                    optimizer.step()
                    optimizer.zero_grad()
                
                print(f"Loss: {global_loss_value / len(dataloader)}")
