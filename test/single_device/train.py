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
from tqdm.auto import tqdm
import torch
from torch.utils.data import DataLoader, Dataset

from utils import MemoryTest, TimeTest, set_seed
from dataset.load import load_dataset, collate_fn
from models.gpt2.single_device.gpt2 import GPTConfig, GPT2Model
from operators.optim import sgd

set_seed(42)

if __name__=="__main__":
    # Hyperparameters
    sequence_length = 1024  # Must be less than or equal to block_size
    epochs = 5
    batch_size = 2
    lr = 1e-3
    momentum = 0.9

    # Initialize the GPT configuration
    config = GPTConfig()
    config.device = torch.device("cuda:0")
    config.dtype = torch.bfloat16

    # Initialize the DataLoader for each split
    train_dataset, validation_dataset, test_dataset = load_dataset("data/")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # Create GPT model
    model = GPT2Model(config)
    
    # Create the optimizer
    optimizer = sgd.SGD(model.get_parameters(), lr=lr, momentum=momentum)
    
    # Training loop
    for epoch in range(epochs):
        print(f"Epoch {epoch}")
        with MemoryTest(enable=True, tqdm=False, master_process=True) as mem_test:
            with TimeTest(enable=True, tqdm=False, master_process=True) as time_test:
                global_loss_value = 0
                for idx, batch in enumerate(tqdm(train_loader)):
                    # Put the input and label on GPU
                    input, label = batch['input_ids'], batch['labels']
                    input, label = input.to(config.device), label.to(config.device)

                    # Forward pass
                    _, loss_value, save_for_backward = model.forward(input, label)
                    global_loss_value += loss_value.item()
                    
                    # Backward pass
                    model.backward(save_for_backward)

                    # Update the model parameters with SGD
                    model.set_parameters(optimizer.step(model.get_parameters()))
                    optimizer.zero_grad(model.get_parameters())
                
                print(f"Loss: {global_loss_value / len(train_loader)}")
