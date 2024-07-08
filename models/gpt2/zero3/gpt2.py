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
        self.n_embd = get_zero_dim(config.n_embd, dist)
        self.dropout = config.dropout
        # key, query, value projections for all heads, but in a batch
        self.weight_attn, self.init_info_attn = linear.linear_init(
            self.n_embd, 3*config.n_embd, device=device, dtype=config.dtype)
        # output projection
        self.weight_proj, self.init_info_proj = linear.linear_init(
            self.n_embd, config.n_embd, device=device, dtype=config.dtype)
        self.dist = dist
        self.require_backward_grad_sync =False
    
    def forward(self, input: torch.Tensor):
        B, T, C = input.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # Get the full parameter buffer
        weight_attn = gather_tensors(self.weight_attn.data, self.dist)
        weight_proj = gather_tensors(self.weight_proj.data, self.dist)
        
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        output, save_for_linear_attn_backward = linear.linear_forward(input, weight_attn, self.init_info_attn)
        output1_shape = output.shape
        q, k, v  = output.split(self.config.n_embd, dim=-1)
        k = k.view(B, T, self.n_head, C // self.n_head) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head) # (B, nh, T, hs)
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
        output, save_for_linear_proj_backward = linear.linear_forward(output, weight_proj, self.init_info_proj)
        output, save_for_dropout_backward = dropout.dropout_forward(output, self.dropout, self.training)
        return output, (
            save_for_linear_attn_backward, 
            save_for_flash_attn_backward, 
            save_for_linear_proj_backward,
            save_for_dropout_backward
        )
    
    def backward(self, grad_output: torch.Tensor, save_for_backward):
        
        # Get the full parameter and grad buffer
        weight_attn = gather_tensors(self.weight_attn.data, self.dist)
        weight_proj = gather_tensors(self.weight_proj.data, self.dist)
        weight_attn.grad = gather_tensors(self.weight_attn.grad, self.dist)
        weight_proj.grad = gather_tensors(self.weight_proj.grad, self.dist)
        
        save_for_linear_attn_backward, \
            save_for_flash_attn_backward, \
            save_for_linear_proj_backward, \
            save_for_dropout_backward = save_for_backward
        grad_output = dropout.dropout_backward(grad_output, save_for_dropout_backward)
        grad_output, weight_proj = linear.linear_backward(grad_output, weight_proj, save_for_linear_proj_backward)
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
        grad_input, self.weight_attn = linear.linear_backward(grad_output, weight_attn, save_for_linear_attn_backward)

        if self.require_backward_grad_sync:
            self.grad_sych()
            self.require_backward_grad_sync = False
        
        # partition tensor
        self.weight_proj.data = partition_tensor(weight_proj.data, self.dist)
        self.weight_proj.grad = partition_tensor(weight_proj.grad, self.dist)
        self.weight_attn.data = partition_tensor(weight_attn.data, self.dist)
        self.weight_attn.grad = partition_tensor(weight_attn.grad, self.dist)
        
        return grad_input
    
    def grad_sych(self):
        self.dist.all_reduce(self.weight_attn.grad, op=self.dist.ReduceOp.SUM)
        self.dist.all_reduce(self.weight_proj.grad, op=self.dist.ReduceOp.SUM)


class MLP(single_device_gpt2.MLP):

    def __init__(self, config: GPTConfig, dist: Dist):
        self.config = config
        self.training = True
        device = dist.get_rank()
        self.n_embd = get_zero_dim(config.n_embd, dist)
        self.weight1, self.init_info1 = linear.linear_init(
            self.n_embd, 4*config.n_embd, device=device, dtype=config.dtype)
        self.weight2, self.init_info2 = linear.linear_init(
            4*self.n_embd, config.n_embd, device=device, dtype=config.dtype)
        self.dropout = config.dropout
        self.dist = dist
        self.require_backward_grad_sync =False
    
    def forward(self, input: torch.Tensor):
        # Get the full parameter buffer
        weight1 = gather_tensors(self.weight1.data, self.dist)
        weight2 = gather_tensors(self.weight2.data, self.dist)
        
        output, save_for_linear1_backward = linear.linear_forward(input, weight1, self.init_info1)
        output, save_for_gelu_backward = gelu.gelu_forward(output)
        output, save_for_linear2_backward = linear.linear_forward(output, weight2, self.init_info2)
        output, save_for_dropout_backward = dropout.dropout_forward(output, self.dropout, self.training)
        return output, (
            save_for_linear1_backward, 
            save_for_gelu_backward, 
            save_for_linear2_backward,
            save_for_dropout_backward
        )
    
    def backward(self, grad_output: torch.Tensor, save_for_backward):
        # Get the full parameter buffer
        weight1 = gather_tensors(self.weight1.data, self.dist)
        weight2 = gather_tensors(self.weight2.data, self.dist)
        weight1.grad = gather_tensors(self.weight1.grad, self.dist)
        weight2.grad = gather_tensors(self.weight2.grad, self.dist)

        save_for_linear1_backward, \
            save_for_gelu_backward, \
            save_for_linear2_backward, \
            save_for_dropout_backward = save_for_backward
        grad_output = dropout.dropout_backward(grad_output, save_for_dropout_backward)
        grad_output, self.weight2 = linear.linear_backward(grad_output, weight2, save_for_linear2_backward)
        grad_output = gelu.gelu_backward(grad_output, save_for_gelu_backward)
        grad_input, self.weight1 = linear.linear_backward(grad_output, weight1, save_for_linear1_backward)

        if self.require_backward_grad_sync:
            self.grad_sych()
            self.require_backward_grad_sync = False
        
        # partition tensor
        self.weight1.data = partition_tensor(weight1.data, self.dist)
        self.weight1.grad = partition_tensor(weight1.grad, self.dist)
        self.weight2.data = partition_tensor(weight2.data, self.dist)
        self.weight2.grad = partition_tensor(weight2.grad, self.dist)
        
        return grad_input
    
    def grad_sych(self):
        self.dist.all_reduce(self.weight1.grad, op=self.dist.ReduceOp.SUM)
        self.dist.all_reduce(self.weight2.grad, op=self.dist.ReduceOp.SUM)


class Block(single_device_gpt2.Block):

    def __init__(self, config: GPTConfig, dist: Dist):
        self.config = config
        device = dist.get_rank()
        self.n_embd = get_zero_dim(config.n_embd, dist)
        self.weight_ln1, self.bias_ln1, self.init_info_ln1 = layernorm.layernorm_init(
            [self.n_embd], device=device, dtype=config.dtype)
        self.attn = CausalSelfAttention(config, dist)
        self.weight_ln2, self.bias_ln2, self.init_info_ln2 = layernorm.layernorm_init(
            [self.n_embd], device=device, dtype=config.dtype)
        self.mlp = MLP(config, dist)
        self.dist = dist
        self.require_backward_grad_sync =False
    
    def forward(self, input: torch.Tensor):
        # Get the full parameter buffer
        weight_ln1 = gather_tensors(self.weight_ln1.data, self.dist)
        bias_ln1 = gather_tensors(self.bias_ln1.data, self.dist)
        weight_ln2 = gather_tensors(self.weight_ln2.data, self.dist)
        bias_ln2 = gather_tensors(self.bias_ln2.data, self.dist)

        output1, save_for_ln1_backward = layernorm.layernorm_forward(
            input, weight_ln1, bias_ln1, self.init_info_ln1)
        output1, save_for_attn_backward = self.attn.forward(output1)
        output1 = input + output1
        output2, save_for_ln2_backward = layernorm.layernorm_forward(
            output1, weight_ln2, bias_ln2, self.init_info_ln2)
        output2, save_for_mlp_backward = self.mlp.forward(output2)
        output2 = output1 + output2
        return output2, (
            save_for_ln1_backward, 
            save_for_attn_backward,
            save_for_ln2_backward,
            save_for_mlp_backward,
        )
    
    def backward(self, grad_output: torch.Tensor, save_for_backward):
        if self.require_backward_grad_sync:
            self.attn.require_backward_grad_sync = True
            self.mlp.require_backward_grad_sync = True
        
        # Get the full parameter buffer
        weight_ln1 = gather_tensors(self.weight_ln1.data, self.dist)
        bias_ln1 = gather_tensors(self.bias_ln1.data, self.dist)
        weight_ln2 = gather_tensors(self.weight_ln2.data, self.dist)
        bias_ln2 = gather_tensors(self.bias_ln2.data, self.dist)
        weight_ln1.grad = gather_tensors(self.weight_ln1.grad, self.dist)
        bias_ln1.grad = gather_tensors(self.bias_ln1.grad, self.dist)
        weight_ln2.grad = gather_tensors(self.weight_ln2.grad, self.dist)
        bias_ln2.grad = gather_tensors(self.bias_ln2.grad, self.dist)
    
        save_for_ln1_backward, \
            save_for_attn_backward, \
            save_for_ln2_backward, \
            save_for_mlp_backward = save_for_backward
        grad_output_mlp = grad_output  # Gradient from the output directly goes into the residual connection
        grad_output = self.mlp.backward(grad_output_mlp, save_for_mlp_backward)
        grad_output, self.weight_ln2, self.bias_ln2 = layernorm.layernorm_backward(grad_output, weight_ln2, bias_ln2, save_for_ln2_backward)
        grad_output_attn = grad_output + grad_output_mlp
        grad_output = self.attn.backward(grad_output_attn, save_for_attn_backward)
        grad_output, self.weight_ln1, self.bias_ln1 = layernorm.layernorm_backward(grad_output, weight_ln1, bias_ln1, save_for_ln1_backward)
        grad_input = grad_output + grad_output_attn

        if self.require_backward_grad_sync:
            self.grad_sych()
            self.require_backward_grad_sync  = False
        
        # partition tensor
        self.weight_ln1.data = partition_tensor(weight_ln1.data, self.dist)
        self.bias_ln1.data = partition_tensor(bias_ln1.data, self.dist)
        self.weight_ln2.data = partition_tensor(weight_ln2.data, self.dist)
        self.bias_ln2.data = partition_tensor(bias_ln2.data, self.dist)
        self.weight_ln1.grad = partition_tensor(weight_ln1.grad, self.dist)
        self.bias_ln1.grad = partition_tensor(bias_ln1.grad, self.dist)
        self.weight_ln2.grad = partition_tensor(weight_ln2.grad, self.dist)
        self.bias_ln2.grad = partition_tensor(bias_ln2.grad, self.dist)
        
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
        self.n_embd = get_zero_dim(config.n_embd, dist)
        self.weight_te, self.init_info_te = embedding.embedding_init(
            config.vocab_size, self.n_embd, device=self.device, dtype=config.dtype)
        self.weight_pe, self.init_info_pe = embedding.embedding_init(
            config.block_size, self.n_embd, device=self.device, dtype=config.dtype)
        self.h = [Block(config, dist) for _ in range(config.n_layer)]
        self.weight_lnf, self.bias_lnf, self.init_info_lnf = layernorm.layernorm_init(
            [self.n_embd], device=self.device, dtype=config.dtype)
        self.weight_lm_head, self.init_info_lm_head = linear.linear_init(
            self.n_embd, config.vocab_size, device=self.device, dtype=config.dtype)

        # self._init_parameters()
        self.dist = dist
        self.require_backward_grad_sync =False

    def _init_parameters(self):
        for k, p in self.get_parameters().items():
            torch.nn.init.normal_(p, mean=0, std=0.2)

    def forward(self, input_ids, labels=None):
        # Get the full parameter buffer
        weight_te = gather_tensors(self.weight_te.data, self.dist)
        weight_pe = gather_tensors(self.weight_pe.data, self.dist)
        weight_lnf = gather_tensors(self.weight_lnf.data, self.dist)
        bias_lnf = gather_tensors(self.bias_lnf.data, self.dist)
        weight_lm_head = gather_tensors(self.weight_lm_head.data, self.dist)

        # input_ids: (B, T), labels: (B, T)
        if labels is None and self.training:
            raise ValueError("In training mode, the labels should not be None.")
        device = input_ids.device
        b, t = input_ids.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb, save_for_te_backward = embedding.embedding_forward(input_ids, weight_te, self.init_info_te) # token embeddings of shape (b, t, n_embd)
        pos_emb, save_for_pe_backward = embedding.embedding_forward(pos, weight_pe, self.init_info_pe) # position embeddings of shape (t, n_embd)
        x, save_for_dropout_backward = dropout.dropout_forward(tok_emb + pos_emb, self.dropout, self.training)
        save_for_h_backward = []
        for block in self.h:
            x, save_for_backward = block.forward(x)
            save_for_h_backward.append(save_for_backward)
        x, save_for_lnf_backward = layernorm.layernorm_forward(x, weight_lnf, bias_lnf, self.init_info_lnf)

        if self.training:
            # if we are given some desired targets also calculate the loss
            logits, save_for_lm_head_backward = linear.linear_forward(x, weight_lm_head, self.init_info_lm_head)
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
        if self.require_backward_grad_sync:
            for block in self.h:
                block.require_backward_grad_sync = True
        
        # Get the full parameter buffer
        weight_te = gather_tensors(self.weight_te.data, self.dist)
        weight_pe = gather_tensors(self.weight_pe.data, self.dist)
        weight_lnf = gather_tensors(self.weight_lnf.data, self.dist)
        bias_lnf = gather_tensors(self.bias_lnf.data, self.dist)
        weight_lm_head = gather_tensors(self.weight_lm_head.data, self.dist)
        weight_te.grad = gather_tensors(self.weight_te.grad, self.dist)
        weight_pe.grad = gather_tensors(self.weight_pe.grad, self.dist)
        weight_lnf.grad = gather_tensors(self.weight_lnf.grad, self.dist)
        bias_lnf.grad = gather_tensors(self.bias_lnf.grad, self.dist)
        weight_lm_head.grad = gather_tensors(self.weight_lm_head.grad, self.dist)

        save_for_te_backward, \
            save_for_pe_backward, \
            save_for_dropout_backward, \
            save_for_h_backward, \
            save_for_lnf_backward, \
            save_for_lm_head_backward, \
            save_for_loss_backward, = save_for_backward
        grad_output = cross_entropy.cross_entropy_loss_backward(save_for_loss_backward)
        grad_output, self.weight_lm_head = linear.linear_backward(grad_output, weight_lm_head, save_for_lm_head_backward)
        grad_output, self.weight_lnf, self.bias_lnf = layernorm.layernorm_backward(grad_output, weight_lnf, bias_lnf, save_for_lnf_backward)
        for block, save_for_block_backward in reversed(list(zip(self.h, save_for_h_backward))):
            grad_output = block.backward(grad_output, save_for_block_backward)
        grad_output = dropout.dropout_backward(grad_output, save_for_dropout_backward)
        _, self.weight_pe = embedding.embedding_backward(grad_output.sum(0), weight_pe, save_for_pe_backward)
        _, self.weight_te = embedding.embedding_backward(grad_output, weight_te, save_for_te_backward)

        if self.require_backward_grad_sync:
            self.grad_sych()
            self.require_backward_grad_sync = False
        
        # partition tensor
        self.weight_te.data = partition_tensor(weight_te.data, self.dist)
        self.weight_pe.data = partition_tensor(weight_pe.data, self.dist)
        self.weight_lnf.data = partition_tensor(weight_lnf.data, self.dist)
        self.bias_lnf.data = partition_tensor(bias_lnf.data, self.dist)
        self.weight_lm_head.data = partition_tensor(weight_lm_head.data, self.dist)
        self.weight_te.grad = partition_tensor(weight_te.grad, self.dist)
        self.weight_pe.grad = partition_tensor(weight_pe.grad, self.dist)
        self.weight_lnf.grad = partition_tensor(weight_lnf.grad, self.dist)
        self.bias_lnf.grad = partition_tensor(bias_lnf.grad, self.dist)
        self.weight_lm_head.grad = partition_tensor(weight_lm_head.grad, self.dist)
        
        return None
    
    def grad_sych(self):
        self.dist.all_reduce(self.weight_te.grad, op=self.dist.ReduceOp.SUM)
        self.dist.all_reduce(self.weight_pe.grad, op=self.dist.ReduceOp.SUM)
        self.dist.all_reduce(self.weight_lnf.grad, op=self.dist.ReduceOp.SUM)
        self.dist.all_reduce(self.bias_lnf.grad, op=self.dist.ReduceOp.SUM)
        self.dist.all_reduce(self.weight_lm_head.grad, op=self.dist.ReduceOp.SUM)
    


def get_zero_dim(dim, dist: Dist):
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

def partition_tensor(full_tensor, dist: Dist) -> torch.Tensor:
    # communication complexity: 0
    t_shape = list(full_tensor.shape)
    local_size = t_shape[-1] // dist.get_world_size()
    local_start = local_size * dist.get_rank()
    local_end = local_start + local_size
    partition_tensor = full_tensor[...,local_start:local_end].contiguous()
    return partition_tensor
