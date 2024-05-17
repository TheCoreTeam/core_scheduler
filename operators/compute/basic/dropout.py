import torch
from typing import Tuple
torch.nn.Dropout

def dropout_forward(input: torch.Tensor, p: float = 0.5, training: bool = True):
    if not training:
        return input, None  # During evaluation, we do not apply dropout

    # Create a dropout mask using the Bernoulli distribution
    mask = (torch.rand_like(input) > p).to(device=input.device, dtype=input.dtype)
    output = input * mask / (1 - p)
    save_for_backward = (mask, p)

    return output, save_for_backward


def dropout_backward(grad_output: torch.Tensor, save_for_backward: Tuple):
    mask, p = save_for_backward
    # Apply the mask and scale by the same factor as during the forward pass
    return grad_output * mask / (1 - p)
