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
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from utils import MemoryTest, TimeTest, set_seed
from dataset.load import load_dataset, collate_fn

set_seed(42)

# GPT-2 Model

class GPTConfig:
    block_size: int = 4096
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False
    flash_attn_available = True
    device = torch.device("cuda:0")


class GPT2Model(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.device = config.device
        self.transformer = nn.ModuleDict(dict(
            # use nn.Embedding to create word embeddings and position embeddings
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # self._init_parameters()
        # print("self.weight_te: ", self.transformer.wte.weight.data[0,:10])
        # print("self.weight_lm_head: ", self.lm_head.weight.data[0,:10])

    def _init_parameters(self):
        # for n, p in self.named_parameters():
        #     torch.nn.init.normal_(p, mean=0, std=0.2)
        set_seed(42)
        self.transformer.wte.weight.data = torch.randn(self.config.vocab_size, self.config.n_embd)
        self.lm_head.weight.data = torch.randn(self.config.vocab_size, self.config.n_embd)

    def forward(self, input_ids, labels=None):
        # input_ids: (B, T), labels: (B, T)
        device = input_ids.device
        b, t = input_ids.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(input_ids) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        # x = tok_emb + pos_emb
        x = tok_emb
        # print("x: ", x[0,0,:10])

        if labels is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            # output: (B, T, vocab)
            # print("logits: ", logits[0,0,:10])
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-1)
            # print("loss: ", loss)
            # exit()
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss



if __name__=="__main__":
    # Hyperparameters
    sequence_length = 1024  # Must be less than or equal to block_size
    epochs = 2
    batch_size = 2
    lr = 1e-3
    momentum = 0

    # Initialize the GPT configuration
    config = GPTConfig()

    # Generate some data to CPU memory
    train_dataset, validation_dataset, test_dataset = load_dataset("data/")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # Create GPT model
    model = GPT2Model(config).to(device=torch.device("cuda:0"), dtype=torch.float32)
    
    # Create the optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    
    # Training loop
    for epoch in range(epochs):
        print(f"Epoch {epoch}")
        with MemoryTest(enable=True, tqdm=False, master_process=True) as mem_test:
            with TimeTest(enable=True, tqdm=False, master_process=True) as time_test:
                global_loss_value = 0
                for idx, batch in enumerate(tqdm(train_loader)):
                    # Put the input and label on GPU
                    input, label = batch['input_ids'], batch['labels']
                    input, label = input.to(torch.device("cuda:0")), label.to(torch.device("cuda:0"))

                    # Forward pass
                    logits, loss_value = model.forward(input, label)
                    global_loss_value += loss_value.item()
                    
                    # Backward pass
                    loss_value.backward()
                    # grad = torch.autograd.grad(loss_value, inputs=[logits], retain_graph=True)[0]
                    # print("grad_output: ", grad[0,0,:10])
                    # exit()

                    # Update the model parameters with SGD
                    optimizer.step()
                    optimizer.zero_grad()
                
                print(f"Loss: {global_loss_value / len(train_loader)}")

