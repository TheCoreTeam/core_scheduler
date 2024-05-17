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