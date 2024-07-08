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

import os
from contextlib import contextmanager
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist


class Dist():
    """
        Distributed package front-end (Currently only support pytorch distribute package)
    """

    # Ops
    ReduceOp = dist.ReduceOp

    def __init__(self, backend="nccl", rank=None, world_size=None):
        if rank is None:
            rank = int(os.environ.get("RANK", 0))
        if world_size is None:
            world_size = int(os.environ.get("WORLD_SIZE", 1))
        dist.init_process_group(
            backend=backend,
            world_size=world_size,
            rank=rank
        )
    
    def is_initialized(self):
        return dist.is_initialized()
    
    def get_rank(self):
        return dist.get_rank()
    
    def get_world_size(self):
        return dist.get_world_size()
    
    @contextmanager
    def torch_distributed_zero_first(self):
        rank = self.get_rank()
        if rank not in [-1, 0]:
            dist.barrier()
        yield
        if rank == 0:
            dist.barrier()
    
    def barrier(self):
        return dist.barrier()
    
    def destroy(self):
        return dist.destroy_process_group()

    # Operators

    def broadcast(self, tensor, src, *args, **kwargs):
        return dist.broadcast(tensor, src, *args, **kwargs)
    
    def reduce(self, tensor, dst, *args, **kwargs):
        return dist.reduce(tensor, dst, *args, **kwargs)
    
    def scatter(self, tensor, scatter_list, *args, **kwargs):
        return dist.scatter(tensor, scatter_list, *args, **kwargs)
    
    def all_reduce(self, tensor, op=ReduceOp.SUM, *args, **kwargs):
        return dist.all_reduce(tensor, op=op, *args, **kwargs)
    
    def all_gather(self, tensor_list, tensor, *args, **kwargs):
        return dist.all_gather(tensor_list, tensor, *args, **kwargs)
    
    def ring_reduce(self, tensor, dst=None, *args, **kwargs):
        if dst is None:
            dst = (self.get_rank() + 1) % self.get_world_size()
        return dist.reduce(tensor, dst, *args, **kwargs)
    
    def ring_all_reduce(self, tensor, op=dist.ReduceOp.SUM, *args, **kwargs):
        return self.all_reduce(tensor, op=op, *args, **kwargs)
    
    def reduce_scatter(self, output, input_list, op=dist.ReduceOp.SUM, *args, **kwargs):
        return dist.reduce_scatter(output, input_list, op=op, *args, **kwargs)
    
    def all_to_all(self, output_tensor_list, input_tensor_list, *args, **kwargs):
        return dist.all_to_all(output_tensor_list, input_tensor_list, *args, **kwargs)
    
    # Data sampler

    def get_sampler(self, dataset, *args, **kwargs):
        return DistributedSampler(dataset, *args, **kwargs)