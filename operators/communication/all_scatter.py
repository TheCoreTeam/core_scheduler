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


def all_scatter(root_tensor: torch.Tensor, devices: List[torch.device], dim: int = 0) -> List[torch.Tensor]:
    """
    Simulate an all-scatter operation where a root tensor is split and scattered across different processes.
    Each process receives a chunk of the root tensor.
    """
    if not devices:
        raise ValueError("The devices list cannot be empty.")

    # Split the root tensor into as many parts as there are devices
    # Assumes the first dimension of the root tensor is divisible by the number of devices
    num_chunks = len(devices)
    if root_tensor.size(0) % num_chunks != 0:
        raise ValueError("The root tensor's first dimension must be divisible by the number of devices.")
    
    chunks = torch.chunk(root_tensor, num_chunks, dim=dim)

    # Scatter the chunks to each device
    scattered = [chunk.to(device) for chunk, device in zip(chunks, devices)]

    return scattered
