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
