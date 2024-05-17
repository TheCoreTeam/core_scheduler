import math
import torch

def gelu_forward(input: torch.Tensor):
    cdf = 0.5 * (1.0 + torch.tanh(math.sqrt(2 / math.pi) * (input + 0.044715 * torch.pow(input, 3))))
    save_for_backward = (input, cdf)
    return input * cdf, save_for_backward

def gelu_backward(grad_output: torch.Tensor, save_for_backward: torch.Tensor):
    input, cdf = save_for_backward
    sigmoid = cdf
    sigmoid_dash = 0.5 * math.sqrt(2 / math.pi) * (1 + 3 * 0.044715 * torch.pow(input, 2)) * (1 - torch.tanh(math.sqrt(2 / math.pi) * (input + 0.044715 * torch.pow(input, 3))) ** 2)
    grad_input = grad_output * (sigmoid + input * sigmoid_dash)
    return grad_input
