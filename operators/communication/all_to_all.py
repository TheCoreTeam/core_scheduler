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
from typing import List

def all_to_all(tensors: List[torch.Tensor], dim=0) -> List[List[torch.Tensor]]:
    """
    Simulate an all-to-all operation where each process sends and receives a segment of data to and from every other process.
    Each tensor in the input list represents the complete data held by one process, which needs to be divided equally among all processes.
    """
    if not tensors:
        raise ValueError("The tensor list cannot be empty.")

    num_processes = len(tensors)
    # Split each tensor into parts, one for each process
    segments = [torch.chunk(tensor, num_processes, dim=dim) for tensor in tensors]

    # Distribute segments to each process
    result = [[] for _ in range(num_processes)]
    for i in range(num_processes):
        for j in range(num_processes):
            result[j].append(segments[i][j].to(tensors[j].device))

    return result