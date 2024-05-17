import torch

def relu_forward(input: torch.Tensor):
    save_for_backward = input
    return torch.maximum(input, torch.tensor(0.0, device=input.device)), save_for_backward

def relu_backward(grad_output: torch.Tensor, save_for_backward: torch.Tensor):
    input = save_for_backward
    return grad_output * (input > 0).float()