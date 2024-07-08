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

import torch
from collections import OrderedDict
import gc

from operators.communication import Dist
from .sgd import SGD


class DDPSGD(SGD):
    def __init__(self, parameters, lr=1e-3, momentum=0, dampening=0, 
                 weight_decay=0, nesterov=False, maximize=False, dist: Dist=None):
        if not dist:
            raise ValueError("Distributed context is required")
        self.dist = dist
        self.lr = lr
        self.momentum = momentum
        self.dampening = dampening
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.maximize = maximize
        if self.momentum != 0:
            # Initialize velocity for each parameter
            self.velocities = OrderedDict({k: torch.zeros_like(p, device=p.device) for k, p in parameters.items()})

    def step(self, parameters):
        for name, param in parameters.items():
            grad = param.grad
            if grad is None:
                continue
            grad = self.sync_grads(grad)
            
            if self.weight_decay != 0:
                grad.add_(param.data, alpha=self.weight_decay)

            if self.maximize:
                grad = -grad

            if self.momentum != 0:
                v = self.velocities[name]
                v.mul_(self.momentum).add_(grad, alpha=1 - self.dampening)
                
                if self.nesterov:
                    grad.add_(v, alpha=self.momentum)
                else:
                    grad = v

            param.data.add_(grad, alpha=-self.lr)
            
        return parameters

    def sync_grads(self, grad):
        # Implement an efficient parameter synchronization mechanism
        self.dist.all_reduce(grad)  # communication complexity: 2g
        return grad



class Zero1SGD(SGD):
    """
    Zero1SGD is a distributed optimizer that uses the Zero1 communication primitive to synchronize gradients across all workers.
    This implementation assumes that all optimizer states can be split at the last dimension.
    """
    def __init__(self, parameters, lr=1e-3, momentum=0, dampening=0, 
                 weight_decay=0, nesterov=False, maximize=False, dist: Dist=None):
        if not dist:
            raise ValueError("Distributed context is required")
        self.dist = dist
        self.lr = lr
        self.momentum = momentum
        self.dampening = dampening
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.maximize = maximize

        # Initialize velocity for the local partition
        if self.momentum != 0:
            self.velocities = OrderedDict()
            self.param_part_table = OrderedDict()
            for name, param in parameters.items():
                p_shape = list(param.shape)
                local_size = p_shape[-1] // dist.get_world_size()
                if local_size * dist.get_world_size() != param.shape[-1]:
                    raise ValueError("The last dimension of the parameter should be divisible by the world size")
                local_start = local_size * self.dist.get_rank()
                local_end = local_start + local_size
                self.velocities[name] = torch.zeros_like(param[...,local_start:local_end])
                self.param_part_table[name] = (local_start, local_end)
        
    def step(self, parameters):
        for (name, param), (name, (local_start, local_end)) in zip(parameters.items(), self.param_part_table.items()):
            if param.grad is None:
                continue
            param = self.sync_grads(param, local_start, local_end)
            partitioned_grad = param.grad[...,local_start:local_end]

            if self.weight_decay != 0:
                partitioned_grad.add_(param.data[...,local_start:local_end], alpha=self.weight_decay)

            if self.maximize:
                partitioned_grad = -partitioned_grad

            if self.momentum != 0:
                v = self.velocities[name]
                v.mul_(self.momentum).add_(partitioned_grad, alpha=1 - self.dampening)
                if self.nesterov:
                    partitioned_grad.add_(v, alpha=self.momentum)
                else:
                    partitioned_grad = v

            # p_grad = param.grad
            param = self.gather_grads(param, partitioned_grad)
            param.data.add_(param.grad, alpha=-self.lr)

        return parameters

    # def sync_grads(self, param, local_start, local_end):      # communication complexity: g
    #     #TODO: make the sync more efficient
    #     local_size = local_end - local_start
    #     for rank in range(self.dist.get_world_size()):
    #         local_start = local_size * rank
    #         local_end = local_start + local_size
    #         partitioned_grad = param.grad[...,local_start:local_end].contiguous()
    #         self.dist.reduce(partitioned_grad, dst=rank)
    #         param.grad[...,local_start:local_end] = partitioned_grad
    #     return param

    def sync_grads(self, param, local_start, local_end):
        local_size = local_end - local_start
        input_list = [
            param.grad[..., i * local_size: (i + 1) * local_size].contiguous() for i in range(self.dist.get_world_size())]
        output = torch.zeros_like(input_list[self.dist.get_rank()])
        self.dist.reduce_scatter(output, input_list, op=self.dist.ReduceOp.SUM)
        param.grad[..., local_start:local_end] = output
        return param

    def gather_grads(self, param, partitioned_grad):      # communication complexity: g
        partitioned_grads_list = [
            torch.zeros_like(partitioned_grad) for _ in range(self.dist.get_world_size())]
        self.dist.all_gather(partitioned_grads_list, partitioned_grad)
        param.grad = torch.cat(partitioned_grads_list, dim=-1)
        return param


class Zero3SGD(SGD):
    def __init__(self, *args, dist: Dist=None, **kwargs):
        super().__init__(*args, **kwargs)