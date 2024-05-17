import torch
from typing import List


def softmax(x, dim=1):
    """Safe softmax function that handles potential numerical instability."""
    max_val = torch.max(x, dim=dim, keepdim=True)[0]
    exps = torch.exp(x - max_val)
    return exps / exps.sum(dim=dim, keepdim=True)


def cross_entropy_loss_forward(
    output: torch.Tensor, 
    target: torch.Tensor, 
    reduction: str = 'mean', 
    label_smoothing: float = 0.0,
    epsilon: float = 1e-10,
):
    # Check if the output shape and target shape are aligned
    if output.dim() < 2:
        raise ValueError("Output tensor must have at least two dimensions (N, C, ...)")
    if target.dim() != output.dim() - 1:
        raise ValueError("Target tensor must have exactly one dimension less than output tensor")
    
    # Flatten the tensors to (N, C) for computing the loss
    C = output.shape[-1]
    output_shape = output.shape
    output = output.view(-1, C)
    target = target.view(-1)

    p = softmax(output)

    if label_smoothing > 0:
        num_classes = p.size(1)
        smoothed_labels = (1 - label_smoothing) * torch.nn.functional.one_hot(target, num_classes=num_classes).float() + \
                          label_smoothing / num_classes
        log_likelihood = -torch.sum(smoothed_labels * torch.log(p + epsilon), dim=1)
    else:
        # Reshape target to match output dimensions for advanced indexing
        target = target.unsqueeze(1)  # Adds a class dimension to target
        log_likelihood = -torch.log(torch.gather(p, 1, target) + epsilon).squeeze(1)  # Remove extra class dimension after gather
    
    if reduction == "mean":
        loss = torch.mean(log_likelihood)
    elif reduction == "sum":
        loss = torch.sum(log_likelihood)
    save_for_backward = (p, target, reduction, label_smoothing, output_shape)
    return loss, save_for_backward


def cross_entropy_loss_backward(save_for_backward: list):
    p, target, reduction, label_smoothing, original_shape = save_for_backward
    grad_output = p.clone()

    if label_smoothing > 0:
        num_classes = p.size(1)
        smoothed_labels = (1 - label_smoothing) * torch.nn.functional.one_hot(target.squeeze(1), num_classes=num_classes).float() + \
                          label_smoothing / num_classes
        grad_output -= smoothed_labels
    else:
        grad_output.scatter_(1, target, grad_output.gather(1, target) - 1)  # scatter subtraction along class dimension
    
    if reduction == "mean":
        grad_output /= p.size(0)
    elif reduction == "sum":
        pass
    return grad_output.view(original_shape)


