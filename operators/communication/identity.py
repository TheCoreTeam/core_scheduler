import torch
from typing import List


def identity(root_tensor: torch.Tensor, devices: List[torch.device]) -> List[torch.Tensor]:
    if not devices:
        raise ValueError("The devices list cannot be empty.")

    # Send the root tensor to all specified devices
    tensors_on_devices = [root_tensor.to(device) for device in devices]

    return tensors_on_devices
