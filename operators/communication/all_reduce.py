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


def all_reduce(tensors: List[torch.Tensor]) -> List[torch.Tensor]:
    """ 
    Simulate an all-reduce operation on a list of tensors across hypothetical processes.
    Each tensor in the list is from a different process.
    This implementation is not optimized for performance.
    """
    if not tensors:
        raise ValueError("The tensor list cannot be empty.")

    first_tensor = tensors[0]
    for tensor in tensors[1:]:
        if tensor.shape != first_tensor.shape:
            raise ValueError("All tensors must have the same shape.")
        if tensor.dtype != first_tensor.dtype:
            raise ValueError("All tensors must have the same data type.")

    result = torch.zeros_like(first_tensor).to('cpu')
    for tensor in tensors:
        result += tensor.to('cpu')

    # Move the result to the original devices and return
    return [result.to(tensor.device) for tensor in tensors]
