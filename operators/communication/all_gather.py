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


def all_gather(tensors: List[torch.Tensor]) -> List[torch.Tensor]:
    """
    Simulate an all-gather operation where each process gathers tensors from all others.
    This example assumes all tensors have the same shape and dtype.
    """
    if not tensors:
        raise ValueError("The tensor list cannot be empty.")
    
    # Initialize a list that will contain the gathered result for each device
    gathered = []
    for _ in tensors:
        gathered.append(torch.cat([tensor.to('cpu') for tensor in tensors], dim=0))

    # Return the list of gathered tensors, each placed back on its original device
    return [g.to(tensor.device) for g, tensor in zip(gathered, tensors)]
