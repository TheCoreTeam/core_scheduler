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
import math
from collections import OrderedDict
import torch
from torch.utils.data import DataLoader, Dataset

from flash_attn.flash_attn_interface import _flash_attn_forward as flash_attn_forward
from flash_attn.flash_attn_interface import _flash_attn_backward as flash_attn_backward

from operators.compute import (
    linear,
    embedding,
    layernorm,
    dropout,
    gelu,
    cross_entropy,
)
from models.utils import flatten_parameters

from utils import MemoryTest, TimeTest, set_seed
from dataset.load import load_dataset, collate_fn
from models.gpt2.single_device.gpt2 import GPTConfig, GPT2Model
from operators.optim import sgd

set_seed(42)


class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False
    flash_attn_available = True
    device = torch.device("cuda")
    dtype = torch.float32



class CausalSelfAttention():

    def __init__(self, config: GPTConfig):
        assert config.n_embd % config.n_head == 0
        self.training = True
        self.flash_attn_available = config.flash_attn_available
        # key, query, value projections for all heads, but in a batch
        self.weight_attn, self.init_info_attn = linear.linear_init(
            config.n_embd, 3*config.n_embd, device=config.device, dtype=config.dtype)
        # output projection
        self.weight_proj, self.init_info_proj = linear.linear_init(
            config.n_embd, config.n_embd, device=config.device, dtype=config.dtype)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

    def forward(self, input: torch.Tensor):
        B, T, C = input.size() # batch size, sequence length, embedding dimensionality (n_embd)
        
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        output, save_for_linear_attn_backward = linear.linear_forward(input, self.weight_attn, self.init_info_attn)
        output1_shape = output.shape
        q, k, v  = output.split(self.n_embd, dim=-1)
        k = k.view(B, T, self.n_head, C // self.n_head) # (B, T, nh, hs)
        q = q.view(B, T, self.n_head, C // self.n_head) # (B, T, nh, hs)
        v = v.view(B, T, self.n_head, C // self.n_head) # (B, T, nh, hs)
        # flash attention
        scale = 1.0 / math.sqrt(k.size(-1))
        output, q, k, v, out_padded, softmax_lse, _, rng_state = flash_attn_forward(
            q, k, v, 
            dropout_p=self.dropout, softmax_scale=scale, causal=True,
            window_size=(-1,-1), alibi_slopes=None, return_softmax=False
        )   # output: (B, T, nh, hs)
        save_for_flash_attn_backward = (
            q, k, v, out_padded, softmax_lse,
            self.dropout, scale, True, (-1,-1), None, False, rng_state,
            output1_shape, output.shape,
        )
        output = output.view(B, T, C)
        # output projection
        output, save_for_linear_proj_backward = linear.linear_forward(input, self.weight_proj, self.init_info_proj)
        output, save_for_dropout_backward = dropout.dropout_forward(output, self.dropout, self.training)
        return output, (
            save_for_linear_attn_backward, 
            save_for_flash_attn_backward, 
            save_for_linear_proj_backward,
            save_for_dropout_backward
        )
    
    def backward(self, grad_output: torch.Tensor, save_for_backward):
        save_for_linear_attn_backward, \
            save_for_flash_attn_backward, \
            save_for_linear_proj_backward, \
            save_for_dropout_backward = save_for_backward
        grad_output = dropout.dropout_backward(grad_output, save_for_dropout_backward)
        grad_output, self.weight_proj = linear.linear_backward(grad_output, self.weight_proj, save_for_linear_proj_backward)
        q, k, v, out_padded, softmax_lse, \
            dropout_p, softmax_scale, causal, window_size, \
            alibi_slopes, deterministic, rng_state, \
            output1_shape, output2_shape = save_for_flash_attn_backward
        grad_output = grad_output.view(output2_shape)
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        dq, dk, dv, _ = flash_attn_backward(
            grad_output, q, k, v, out_padded, softmax_lse, 
            dq, dk, dv, dropout_p, softmax_scale, causal, window_size,
            alibi_slopes, deterministic, rng_state)
        dq = dq[..., : grad_output.shape[-1]]  # We could have padded the head dimension
        dk = dk[..., : grad_output.shape[-1]]
        dv = dv[..., : grad_output.shape[-1]]
        grad_output = torch.cat((dq, dk, dv), dim=-1).view(output1_shape)
        grad_input, self.weight_attn = linear.linear_backward(grad_output, self.weight_attn, save_for_linear_attn_backward)
        return grad_input

    def get_parameters(self):
        return OrderedDict({
            "weight_attn": self.weight_attn,
            "weight_proj": self.weight_proj
        })
    
    def set_parameters(self, parameters: OrderedDict):
        self.weight_attn = parameters["weight_attn"]
        self.weight_proj = parameters["weight_proj"]


class MLP():

    def __init__(self, config: GPTConfig):
        self.training = True
        self.weight1, self.init_info1 = linear.linear_init(
            config.n_embd, 4*config.n_embd, device=config.device, dtype=config.dtype)
        self.weight2, self.init_info2 = linear.linear_init(
            4*config.n_embd, config.n_embd, device=config.device, dtype=config.dtype)
        self.dropout = config.dropout
    
    def forward(self, input: torch.Tensor):
        output, save_for_linear1_backward = linear.linear_forward(input, self.weight1, self.init_info1)
        output, save_for_gelu_backward = gelu.gelu_forward(output)
        output, save_for_linear2_backward = linear.linear_forward(output, self.weight2, self.init_info2)
        output, save_for_dropout_backward = dropout.dropout_forward(output, self.dropout, self.training)
        return output, (
            save_for_linear1_backward, 
            save_for_gelu_backward, 
            save_for_linear2_backward,
            save_for_dropout_backward
        )
    
    def backward(self, grad_output: torch.Tensor, save_for_backward):
        save_for_linear1_backward, \
            save_for_gelu_backward, \
            save_for_linear2_backward, \
            save_for_dropout_backward = save_for_backward
        grad_output = dropout.dropout_backward(grad_output, save_for_dropout_backward)
        grad_output, self.weight2 = linear.linear_backward(grad_output, self.weight2, save_for_linear2_backward)
        grad_output = gelu.gelu_backward(grad_output, save_for_gelu_backward)
        grad_input, self.weight1 = linear.linear_backward(grad_output, self.weight1, save_for_linear1_backward)
        return grad_input
    
    def get_parameters(self):
        return OrderedDict({
            "weight1": self.weight1,
            "weight2": self.weight2
        })
    
    def set_parameters(self, parameters: OrderedDict):
        self.weight1 = parameters["weight1"]
        self.weight2 = parameters["weight2"]


class Block():

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.weight_ln1, self.bias_ln1, self.init_info_ln1 = layernorm.layernorm_init(
            [config.n_embd], device=config.device, dtype=config.dtype)
        self.attn = CausalSelfAttention(config)
        self.weight_ln2, self.bias_ln2, self.init_info_ln2 = layernorm.layernorm_init(
            [config.n_embd], device=config.device, dtype=config.dtype)
        self.mlp = MLP(config)

    def forward(self, input: torch.Tensor):
        output1, save_for_ln1_backward = layernorm.layernorm_forward(
            input, self.weight_ln1, self.bias_ln1, self.init_info_ln1)
        output1, save_for_attn_backward = self.attn.forward(output1)
        output1 = input + output1
        output2, save_for_ln2_backward = layernorm.layernorm_forward(
            output1, self.weight_ln2, self.bias_ln2, self.init_info_ln2)
        output2, save_for_mlp_backward = self.mlp.forward(output2)
        output2 = output1 + output2
        return output2, (
            save_for_ln1_backward, 
            save_for_attn_backward,
            save_for_ln2_backward,
            save_for_mlp_backward,
        )
    
    def backward(self, grad_output: torch.Tensor, save_for_backward):
        save_for_ln1_backward, \
            save_for_attn_backward, \
            save_for_ln2_backward, \
            save_for_mlp_backward = save_for_backward
        grad_output_mlp = grad_output  # Gradient from the output directly goes into the residual connection
        grad_output = self.mlp.backward(grad_output_mlp, save_for_mlp_backward)
        grad_output, self.weight_ln2, self.bias_ln2 = layernorm.layernorm_backward(grad_output, self.weight_ln2, self.bias_ln2, save_for_ln2_backward)
        grad_output_attn = grad_output + grad_output_mlp
        grad_output = self.attn.backward(grad_output_attn, save_for_attn_backward)
        grad_output, self.weight_ln1, self.bias_ln1 = layernorm.layernorm_backward(grad_output, self.weight_ln1, self.bias_ln1, save_for_ln1_backward)
        grad_input = grad_output + grad_output_attn
        return grad_input
    
    def get_parameters(self):
        parameters = OrderedDict({
            "weight_ln1": self.weight_ln1,
            "bias_ln1": self.bias_ln1,
            "weight_ln2": self.weight_ln2,
            "bias_ln2": self.bias_ln2,
        })
        parameters = flatten_parameters(parameters, self.attn, "attn")
        parameters = flatten_parameters(parameters, self.mlp, "mlp")
        return parameters
    
    def set_parameters(self, parameters: OrderedDict):
        self.weight_ln1 = parameters["weight_ln1"]
        self.bias_ln1 = parameters["bias_ln1"]
        self.weight_ln2 = parameters["weight_ln2"]
        self.bias_ln2 = parameters["bias_ln2"]
        # Set parameters for attention and MLP
        attn_params = OrderedDict({k.split('.', 1)[1]: v for k, v in parameters.items() if k.startswith("attn")})
        mlp_params = OrderedDict({k.split('.', 1)[1]: v for k, v in parameters.items() if k.startswith("mlp")})
        self.attn.set_parameters(attn_params)
        self.mlp.set_parameters(mlp_params)


class GPT2Model():
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.device = config.device
        self.dropout = config.dropout
        self.training = True
        
        # self.weight_te, self.init_info_te = embedding.embedding_init(config.vocab_size, config.n_embd, device=config.device, dtype=config.dtype)
        # self.weight_pe, self.init_info_pe = embedding.embedding_init(config.block_size, config.n_embd, device=config.device, dtype=config.dtype)
        self.weight_te, self.init_info_te = linear.linear_init(config.vocab_size, config.n_embd, device=config.device, dtype=config.dtype)
        self.weight_pe, self.init_info_pe = linear.linear_init(config.block_size, config.n_embd, device=config.device, dtype=config.dtype)
        self.h = [Block(config) for _ in range(config.n_layer)]
        self.weight_lnf, self.bias_lnf, self.init_info_lnf = layernorm.layernorm_init([config.n_embd], device=config.device, dtype=config.dtype)
        
        self.weight_lm_head, self.init_info_lm_head = linear.linear_init(
            config.n_embd, config.vocab_size, device=config.device, dtype=config.dtype)

        # self._init_parameters()

    def _init_parameters(self):
        for k, p in self.get_parameters().items():
            torch.nn.init.normal_(p, mean=0, std=0.2)

    def forward(self, input_ids, labels=None):
        # input_ids: (B, T), labels: (B, T)
        if labels is None and self.training:
            raise ValueError("In training mode, the labels should not be None.")
        device = input_ids.device
        b, t = input_ids.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        # tok_emb, save_for_te_backward = embedding.embedding_forward(input_ids, self.weight_te, self.init_info_te) # token embeddings of shape (b, t, n_embd)
        # pos_emb, save_for_pe_backward = embedding.embedding_forward(pos, self.weight_pe, self.init_info_pe) # position embeddings of shape (t, n_embd)
        input_ids = input_ids.unsqueeze(-1).repeat_interleave(self.config.vocab_size, dim=-1).to(self.config.dtype)
        pos = pos.unsqueeze(-1).repeat_interleave(self.config.block_size, dim=-1).to(self.config.dtype)
        tok_emb, save_for_te_backward = linear.linear_forward(input_ids, self.weight_te, self.init_info_te) # token embeddings of shape (b, t, n_embd)
        pos_emb, save_for_pe_backward = linear.linear_forward(pos, self.weight_pe, self.init_info_pe) # position embeddings of shape (t, n_embd)
        x, save_for_dropout_backward = dropout.dropout_forward(tok_emb + pos_emb, self.dropout, self.training)
        save_for_h_backward = []
        for block in self.h:
            x, save_for_backward = block.forward(x)
            save_for_h_backward.append(save_for_backward)
        x, save_for_lnf_backward = layernorm.layernorm_forward(x, self.weight_lnf, self.bias_lnf, self.init_info_lnf)

        if self.training:
            # if we are given some desired targets also calculate the loss
            logits, save_for_lm_head_backward = linear.linear_forward(x, self.weight_lm_head, self.init_info_lm_head)
            # output: (B, T, vocab)
            loss, save_for_loss_backward = cross_entropy.cross_entropy_loss_forward(logits, labels)
        else:
            # # inference-time mini-optimization: only forward the lm_head on the very last position
            # logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            # loss = None
            raise NotImplementedError("The inference mode development is still in progress.")

        return logits, loss, (
            save_for_te_backward,
            save_for_pe_backward,
            save_for_dropout_backward,
            save_for_h_backward,
            save_for_lnf_backward,
            save_for_lm_head_backward,
            save_for_loss_backward,
        )

    def backward(self, save_for_backward):
        save_for_te_backward, \
            save_for_pe_backward, \
            save_for_dropout_backward, \
            save_for_h_backward, \
            save_for_lnf_backward, \
            save_for_lm_head_backward, \
            save_for_loss_backward, = save_for_backward
        grad_output = cross_entropy.cross_entropy_loss_backward(save_for_loss_backward)
        grad_output, self.weight_lm_head = linear.linear_backward(grad_output, self.weight_lm_head, save_for_lm_head_backward)
        grad_output, self.weight_lnf, self.bias_lnf = layernorm.layernorm_backward(grad_output, self.weight_lnf, self.bias_lnf, save_for_lnf_backward)
        for block, save_for_block_backward in reversed(list(zip(self.h, save_for_h_backward))):
            grad_output = block.backward(grad_output, save_for_block_backward)
        grad_output = dropout.dropout_backward(grad_output, save_for_dropout_backward)
        # _, self.weight_pe = embedding.embedding_backward(grad_output.sum(0), self.weight_pe, save_for_pe_backward)
        # _, self.weight_te = embedding.embedding_backward(grad_output, self.weight_te, save_for_te_backward)
        _, self.weight_pe = linear.linear_backward(grad_output.sum(0), self.weight_pe, save_for_pe_backward)
        _, self.weight_te = linear.linear_backward(grad_output, self.weight_te, save_for_te_backward)
        return None

    def get_parameters(self):
        parameters = OrderedDict({
            'weight_te': self.weight_te,
            'weight_pe': self.weight_pe,
            'weight_lnf': self.weight_lnf,
            'bias_lnf': self.bias_lnf,
            'weight_lm_head': self.weight_lm_head
        })
        for layer, block in enumerate(self.h):
            parameters = flatten_parameters(parameters, block, f"h.{layer}")
        return parameters
    
    def set_parameters(self, parameters: OrderedDict):
        self.weight_te = parameters["weight_te"]
        self.weight_pe = parameters["weight_pe"]
        self.weight_lnf = parameters["weight_lnf"]
        self.bias_lnf = parameters["bias_lnf"]
        self.weight_lm_head = parameters["weight_lm_head"]
        # Set parameters for each block
        for layer in range(len(self.h)):
            layer_params = OrderedDict({k.split(f"h.{layer}.", 1)[1]: v for k, v in parameters.items() if k.startswith(f"h.{layer}.")})
            self.h[layer].set_parameters(layer_params)


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
