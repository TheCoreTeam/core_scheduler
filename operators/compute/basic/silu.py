import math
import torch

def silu_forward(input: torch.Tensor):
    output = input * torch.sigmoid(input)
    save_for_backward = (input)
    return output, save_for_backward

def silu_backward(grad_output: torch.Tensor, save_for_backward: torch.Tensor):
    input = save_for_backward
    sigmoid_i = torch.sigmoid(input)
    grad_input = grad_output * (sigmoid_i * (1 + input * (1 - sigmoid_i)))
    return grad_input
