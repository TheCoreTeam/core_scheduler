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
        self.training = True
        self.flash_attn_available = config.flash_attn_available
        device = dist.get_rank()
        # key, query, value projections for all heads, but in a batch
        self.weight_attn, self.init_info_attn = linear.linear_init(
            config.n_embd, 3*config.n_embd, device=device, dtype=config.dtype)
        # output projection
        self.weight_proj, self.init_info_proj = linear.linear_init(
            config.n_embd, config.n_embd, device=device, dtype=config.dtype)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.dist = dist
        self.require_backward_grad_sync =False
    
    def backward(self, grad_output: torch.Tensor, save_for_backward):
        grad_input = super().backward(grad_output, save_for_backward)
        if self.require_backward_grad_sync:
            self.grad_sych()
            self.require_backward_grad_sync = False
        return grad_input
    
    def grad_sych(self):
        self.dist.all_reduce(self.weight_attn.grad, op=self.dist.ReduceOp.SUM)
        self.dist.all_reduce(self.weight_proj.grad, op=self.dist.ReduceOp.SUM)


class MLP(single_device_gpt2.MLP):

    def __init__(self, config: GPTConfig, dist: Dist):
        self.training = True
        device = dist.get_rank()
        self.weight1, self.init_info1 = linear.linear_init(
            config.n_embd, 4*config.n_embd, device=device, dtype=config.dtype)
        self.weight2, self.init_info2 = linear.linear_init(
            4*config.n_embd, config.n_embd, device=device, dtype=config.dtype)
        self.dropout = config.dropout
        self.dist = dist
        self.require_backward_grad_sync =False
    
    def backward(self, grad_output: torch.Tensor, save_for_backward):
        grad_input = super().backward(grad_output, save_for_backward)
        if self.require_backward_grad_sync:
            self.grad_sych()
            self.require_backward_grad_sync = False
        return grad_input
    
    def grad_sych(self):
        self.dist.all_reduce(self.weight1.grad, op=self.dist.ReduceOp.SUM)
        self.dist.all_reduce(self.weight2.grad, op=self.dist.ReduceOp.SUM)


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
        self.require_backward_grad_sync =False
    
    def backward(self, grad_output: torch.Tensor, save_for_backward):
        if self.require_backward_grad_sync:
            self.attn.require_backward_grad_sync = True
            self.mlp.require_backward_grad_sync = True
        grad_input = super().backward(grad_output, save_for_backward)
        if self.require_backward_grad_sync:
            self.grad_sych()
            self.require_backward_grad_sync  = False
        return grad_input
    
    def grad_sych(self):
        self.dist.all_reduce(self.weight_ln1.grad, op=self.dist.ReduceOp.SUM)
        self.dist.all_reduce(self.bias_ln1.grad, op=self.dist.ReduceOp.SUM)
        self.dist.all_reduce(self.weight_ln2.grad, op=self.dist.ReduceOp.SUM)
        self.dist.all_reduce(self.bias_ln2.grad, op=self.dist.ReduceOp.SUM)



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
        self.require_backward_grad_sync =False

    def _init_parameters(self):
        for k, p in self.get_parameters().items():
            torch.nn.init.normal_(p, mean=0, std=0.2)

    def backward(self, save_for_backward):
        if self.require_backward_grad_sync:
            for block in self.h:
                block.require_backward_grad_sync = True
        grad_input = super().backward(save_for_backward)
        if self.require_backward_grad_sync:
            self.grad_sych()
            self.require_backward_grad_sync = False
        return grad_input
    
    def grad_sych(self):
        self.dist.all_reduce(self.weight_te.grad, op=self.dist.ReduceOp.SUM)
        self.dist.all_reduce(self.weight_pe.grad, op=self.dist.ReduceOp.SUM)
        self.dist.all_reduce(self.weight_lnf.grad, op=self.dist.ReduceOp.SUM)
        self.dist.all_reduce(self.bias_lnf.grad, op=self.dist.ReduceOp.SUM)
        self.dist.all_reduce(self.weight_lm_head.grad, op=self.dist.ReduceOp.SUM)
    
