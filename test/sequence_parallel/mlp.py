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
import torch
import datasets
from transformers import AutoTokenizer

from operators.compute import (
    linear,
    relu,
    cross_entropy,
)
from operators.optim import sgd

from operators.communication import (
    all_scatter,
    all_gather,
    all_reduce,
    identity,
)


class MLP():
    """
    Sequence Parallel: https://arxiv.org/pdf/1909.08053
    """

    def __init__(
        self, 
        input_size: int = 784, 
        hidden_size: int = 1024, 
        output_size: int = 10,
        devices: List[torch.device] = ['cuda:0', 'cuda:1'],
    ):
        self.A1 = linear.linear_init(input_size, hidden_size).to(devices[0])
        self.A2 = linear.linear_init(input_size, hidden_size).to(devices[1])
        self.B1 = linear.linear_init(hidden_size, output_size).to(devices[0])
        self.B2 = linear.linear_init(hidden_size, output_size).to(devices[1])
    
    def forward(self, input: torch.Tensor, label: torch.Tensor):
        # Master process scatters the input data
        inputs = all_scatter(input, self.devices, dim=1)    # (B, L/N, D)
        outputs = []
        
        #*** Process 0 ***
        output, save_for_linearA1_backward = linear.linear_forward(inputs[0], self.A1)
        output, save_for_relu1_backward = relu.relu_forward(output)
        output, save_for_linearB1_backward = linear.linear_forward(output, self.B1)
        outputs.append(output)

        #*** Process 1 ***
        output, save_for_linearA2_backward = linear.linear_forward(inputs[1], self.A2)
        output, save_for_relu2_backward = relu.relu_forward(output)
        output, save_for_linearB2_backward = linear.linear_forward(output, self.B2)
        outputs.append(output)

        outputs = all_gather(outputs)

        
        #*** Process 0 / 1 ***
        loss_value, save_for_loss_backward = cross_entropy.cross_entropy_loss_forward(outputs[0], label)
        return loss_value, (
            save_for_linearA1_backward, save_for_relu1_backward, save_for_linearB1_backward,
            save_for_linearA2_backward, save_for_relu2_backward, save_for_linearB2_backward,
            save_for_loss_backward
        )
    
    def backward(self, save_for_backward):
        save_for_linearA1_backward, save_for_relu1_backward, save_for_linearB1_backward, \
            save_for_linearA2_backward, save_for_relu2_backward, save_for_linearB2_backward, \
            save_for_loss_backward = save_for_backward
        
        grad_output = cross_entropy.cross_entropy_loss_backward(save_for_loss_backward)
        grad_outputs = identity(grad_output, self.devices)
        grad_inputs = []

        #*** Process 0 ***
        grad_output, grad_B1 = linear.linear_backward(grad_outputs[0], save_for_linearB1_backward)
        grad_output = relu.relu_backward(grad_output, save_for_relu1_backward)
        grad_input, grad_A1 = linear.linear_backward(grad_output, save_for_linearA1_backward)
        grad_inputs.append(grad_input)

        #*** Process 1 ***
        grad_output, grad_B2 = linear.linear_backward(grad_outputs[1], save_for_linearB2_backward)
        grad_output = relu.relu_backward(grad_output, save_for_relu2_backward)
        grad_input, grad_A2 = linear.linear_backward(grad_output, save_for_linearA2_backward)
        grad_inputs.append(grad_input)

        grad_params = (grad_A1, grad_A2, grad_B1, grad_B2)
        grad_input = all_reduce(grad_inputs)
        return grad_params, grad_input
    
    def get_model_parameters(self):
        return [self.A1, self.A2, self.B1, self.B2]
    
    def set_model_parameters(self, parameters: List[torch.Tensor]):
        self.A1, self.A2, self.B1, self.B2 = parameters



if __name__ == '__main__':
    # Hyperparameters
    batch_size = 64     # global batch size
    epochs = 20
    lr = 1e-4
    momentum = 0.9

    # Data settings
    load_from_disk = False
    squence_length = 512
    feature_dimension = 128
    classes = 2
    
    # IMDB Dataset
    if not load_from_disk:
        trainset = datasets.load_dataset('imdb', split='train')
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        def tokenize_function(examples):
            return tokenizer(
                examples["text"], 
                padding="max_length", 
                truncation=True, 
                max_length=squence_length,
                return_tensors="pt"
            )
        tokenized_datasets = trainset.map(tokenize_function, batched=True)
        tokenized_datasets.save_to_disk('../../../dataset/tokenized_imdb/')
    else:
        tokenized_datasets = datasets.load_from_disk('../../../dataset/tokenized_imdb/')
    dataloader = torch.utils.data.DataLoader(tokenized_datasets, batch_size=batch_size, shuffle=True)


    # DDP settings
    devices_map = ['cuda:0', 'cuda:1']
    devices = [torch.device(device) for device in devices_map]

    # Create the model
    model = MLP(input_size=feature_dimension, hidden_size=1024, output_size=classes, devices=devices)
    
    # Create the optimizer
    optimizer = sgd.SGD(model.get_model_parameters(), lr=lr, momentum=momentum)
    
    # Training loop
    for epoch in range(epochs):
        print(f"Epoch {epoch}")
        global_loss_value = 0
        for idx, (input, label) in enumerate(tqdm(dataloader)):
            # Forward pass
            loss_value, save_for_backward = model.forward(input, label)
            global_loss_value += loss_value.item()
            
            # Backward pass
            grad_params, _ = model.backward(save_for_backward)
            
            # Update the model parameters with SGD
            model.set_model_parameters(optimizer.step(model.get_model_parameters(), grad_params))
        
        print(f"Loss: {global_loss_value / len(dataloader)}")
