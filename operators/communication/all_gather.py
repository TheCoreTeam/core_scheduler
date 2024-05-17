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
