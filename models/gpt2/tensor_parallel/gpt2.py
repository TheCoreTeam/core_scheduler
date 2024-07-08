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

import math
from collections import OrderedDict
import torch
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
from operators.communication import Dist
from models.utils import flatten_parameters

import models.gpt2.single_device.gpt2 as single_device_gpt2

# @dataclass
class GPTConfig(single_device_gpt2.GPTConfig):
    block_size: int = 4096
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False
    flash_attn_available = True
    device = torch.device("cuda")
    dtype = torch.float32



class CausalSelfAttention(single_device_gpt2.CausalSelfAttention):

    def __init__(self, config: GPTConfig, dist: Dist):
        assert config.n_embd % config.n_head == 0
        self.config = config
        self.training = True
        self.flash_attn_available = config.flash_attn_available
        device = dist.get_rank()
        self.n_head = config.n_head
        self.n_embd = get_tp_dim(config.n_embd, dist)
        self.dropout = config.dropout
        # key, query, value projections for all heads, but in a batch
        self.weight_attn, self.init_info_attn = linear.linear_init(
            config.n_embd, 3*self.n_embd, device=device, dtype=config.dtype)
        # output projection
        self.weight_proj, self.init_info_proj = linear.linear_init(
            self.n_embd, config.n_embd, device=device, dtype=config.dtype)
        self.dist = dist
    
    def forward(self, input: torch.Tensor):
        B, T, C = input.size() # batch size, sequence length, embedding dimensionality (n_embd)
        
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        output, save_for_linear_attn_backward = linear.linear_forward(input, self.weight_attn, self.init_info_attn)
        output1_shape = output.shape
        q, k, v  = output.split(self.n_embd, dim=-1)
        k = k.view(B, T, self.n_head, self.n_embd // self.n_head) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, self.n_embd // self.n_head) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, self.n_embd // self.n_head) # (B, nh, T, hs)
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
        output = output.view(B, T, self.n_embd)
        # output projection
        output, save_for_linear_proj_backward = linear.linear_forward(output, self.weight_proj, self.init_info_proj)
        self.dist.all_reduce(output)    # for tensor parallel, must before dropout layer
        output, save_for_dropout_backward = dropout.dropout_forward(output, self.dropout, self.training)
        return output, (
            save_for_linear_attn_backward, 
            save_for_flash_attn_backward, 
            save_for_linear_proj_backward,
            save_for_dropout_backward
        )

    def backward(self, grad_output: torch.Tensor, save_for_backward):
        grad_input = super().backward(grad_output, save_for_backward)
        self.dist.all_reduce(grad_input)    # for tensor parallel
        return grad_input
    

class MLP(single_device_gpt2.MLP):

    def __init__(self, config: GPTConfig, dist: Dist):
        self.config = config
        self.training = True
        device = dist.get_rank()
        self.n_embd = get_tp_dim(config.n_embd, dist)
        self.dropout = config.dropout
        self.weight1, self.init_info1 = linear.linear_init(
            config.n_embd, 4*self.n_embd, device=device, dtype=config.dtype)
        self.weight2, self.init_info2 = linear.linear_init(
            4*self.n_embd, config.n_embd, device=device, dtype=config.dtype)
        self.dist = dist
    
    def forward(self, input: torch.Tensor):
        output, save_for_linear1_backward = linear.linear_forward(input, self.weight1, self.init_info1)
        output, save_for_gelu_backward = gelu.gelu_forward(output)
        output, save_for_linear2_backward = linear.linear_forward(output, self.weight2, self.init_info2)
        
        self.dist.all_reduce(output)    # for tensor parallel, must before dropout layer
        
        output, save_for_dropout_backward = dropout.dropout_forward(output, self.dropout, self.training)
        return output, (
            save_for_linear1_backward, 
            save_for_gelu_backward, 
            save_for_linear2_backward,
            save_for_dropout_backward
        )

    def backward(self, grad_output: torch.Tensor, save_for_backward):
        grad_input = super().backward(grad_output, save_for_backward)
        self.dist.all_reduce(grad_input)    # for tensor parallel
        return grad_input
    

class Block(single_device_gpt2.Block):

    def __init__(self, config: GPTConfig, dist: Dist):
        device = dist.get_rank()
        self.weight_ln1, self.bias_ln1, self.init_info_ln1 = layernorm.layernorm_init(
            [config.n_embd], device=device, dtype=config.dtype)
        self.attn = CausalSelfAttention(config, dist)
        self.weight_ln2, self.bias_ln2, self.init_info_ln2 = layernorm.layernorm_init(
            [config.n_embd], device=device, dtype=config.dtype)
        self.mlp = MLP(config, dist)
        self.dist = dist
    

class GPT2Model(single_device_gpt2.GPT2Model):
    
    def __init__(self, config: GPTConfig, dist: Dist):
        self.config = config
        self.device = dist.get_rank()
        self.dropout = config.dropout
        self.training = True
        
        self.weight_te, self.init_info_te = embedding.embedding_init(config.vocab_size, config.n_embd, device=self.device, dtype=config.dtype)
        self.weight_pe, self.init_info_pe = embedding.embedding_init(config.block_size, config.n_embd, device=self.device, dtype=config.dtype)
        self.h = [Block(config, dist) for _ in range(config.n_layer)]
        self.weight_lnf, self.bias_lnf, self.init_info_lnf = layernorm.layernorm_init([config.n_embd], device=self.device, dtype=config.dtype)
        self.weight_lm_head, self.init_info_lm_head = linear.linear_init(
            config.n_embd, config.vocab_size, device=self.device, dtype=config.dtype)

        # self._init_parameters()
        self.dist = dist

    def _init_parameters(self):
        for k, p in self.get_parameters().items():
            torch.nn.init.normal_(p, mean=0, std=0.2)

    def backward(self, save_for_backward):
        grad_input = super().backward(save_for_backward)
        return grad_input
    


def get_tp_dim(dim, dist: Dist):
    dim_ = dim // dist.get_world_size()
    if dim_ * dist.get_world_size() != dim:
        raise ValueError("The last dimension of the parameter should be divisible by the world size")
    return dim_

def gather_tensors(partitioned_tensor, dist: Dist) -> torch.Tensor:
    # communication complexity: g
    partitioned_tensors_list = [
        torch.zeros_like(partitioned_tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(partitioned_tensors_list, partitioned_tensor)
    full_tensor = torch.cat(partitioned_tensors_list, dim=-1)
    return full_tensor
