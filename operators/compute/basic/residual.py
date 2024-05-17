import torch

def residual_forward(orig_input: torch.Tensor, input: torch.Tensor):
    save_for_backward = input
    return orig_input + input, save_for_backward

def residual_backward(grad_output: torch.Tensor, save_for_backward: torch.Tensor):
    orig_input, input = save_for_backward
    grad_orig_input = grad_output * torch.ones_like(orig_input).to(orig_input.device, orig_input.dtype)
    grad_input = grad_output * torch.ones_like(input).to(input.device, input.dtype)
    return grad_orig_input, grad_input