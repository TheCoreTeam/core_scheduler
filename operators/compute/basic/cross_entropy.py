# Copyright (c) 2024 The Core Team
#
# Licensed under the Apache License, Version 2.0
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from typing import List


def cross_entropy_loss_forward(*args, backend="naive", **kwargs):
    if backend == "naive":
        return cross_entropy_loss_forward_naive(*args, **kwargs)
    elif backend == "torchjit":
        return cross_entropy_loss_forward_torchjit(*args, **kwargs)
    # elif backend == "triton":
    #     return cross_entropy_loss_forward_triton(*args, **kwargs)

def cross_entropy_loss_backward(*args, backend="naive", **kwargs):
    if backend == "naive":
        return cross_entropy_loss_backward_naive(*args, **kwargs)
    elif backend == "torchjit":
        return cross_entropy_loss_backward_torchjit(*args, **kwargs)
    # elif backend == "triton":
    #     return cross_entropy_loss_backward_triton(*args, **kwargs)



def log_softmax(x, dim=-1):
    x_max = torch.max(x, dim=dim, keepdim=True)[0]
    x_stable = x - x_max
    exps = torch.exp(x_stable)
    sum_exps = torch.sum(exps, dim=dim, keepdim=True)
    log_sum_exps = torch.log(sum_exps)
    log_softmax = x_stable - log_sum_exps
    return log_softmax


def cross_entropy_loss_forward_naive(
    output: torch.Tensor, 
    target: torch.Tensor, 
    reduction: str = 'mean', 
    label_smoothing: float = 0.0,
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

    log_p = log_softmax(output, dim=1)

    if label_smoothing > 0:
        num_classes = log_p.size(1)
        smoothed_labels = (1 - label_smoothing) * torch.nn.functional.one_hot(target, num_classes=num_classes).to(dtype=output.dtype) + \
                          label_smoothing / num_classes
        log_likelihood = -torch.sum(smoothed_labels * log_p, dim=1)
    else:
        # Reshape target to match output dimensions for advanced indexing
        target = target.unsqueeze(1)  # Adds a class dimension to target
        log_likelihood = -log_p.gather(1, target).squeeze(1)  # Remove extra class dimension after gather
    
    if reduction == "mean":
        loss = torch.mean(log_likelihood)
    elif reduction == "sum":
        loss = torch.sum(log_likelihood)

    # Prepare save_for_backward if using this in a custom backward pass
    save_for_backward = (log_p, target, reduction, label_smoothing, output_shape)
    return loss, save_for_backward


def cross_entropy_loss_backward_naive(save_for_backward: tuple):
    log_p, target, reduction, label_smoothing, original_shape = save_for_backward
    grad_output = torch.exp(log_p).clone()

    if label_smoothing > 0:
        num_classes = log_p.size(1)
        smoothed_labels = (1 - label_smoothing) * torch.nn.functional.one_hot(target.squeeze(1), num_classes=num_classes).to(dtype=grad_output.dtype) + \
                          label_smoothing / num_classes
        grad_output -= smoothed_labels
    else:
        grad_output.scatter_(1, target, grad_output.gather(1, target) - 1)  # scatter subtraction along class dimension
    
    if reduction == "mean":
        grad_output /= log_p.size(0)
    elif reduction == "sum":
        pass
    return grad_output.view(original_shape)



def cross_entropy_loss_forward_torchjit(
    output: torch.Tensor, 
    target: torch.Tensor, 
    reduction: str = 'mean', 
    label_smoothing: float = 0.0,
):
    # Check if the output shape and target shape are aligned
    if output.dim() < 2:
        raise ValueError("Output tensor must have at least two dimensions (N, C, ...)")
    if target.dim() != output.dim() - 1:
        raise ValueError("Target tensor must have exactly one dimension less than output tensor")
    @torch.jit.script
    def log_softmax(x, dim: int):
        x_max = torch.max(x, dim=dim, keepdim=True)[0]
        x_stable = x - x_max
        exps = torch.exp(x_stable)
        sum_exps = torch.sum(exps, dim=dim, keepdim=True)
        log_sum_exps = torch.log(sum_exps)
        log_softmax = x_stable - log_sum_exps
        return log_softmax
    @torch.jit.script
    def compute(output: torch.Tensor, target: torch.Tensor, reduction: str = 'mean', label_smoothing: float = 0.0):
        # Flatten the tensors to (N, C) for computing the loss
        C = output.shape[-1]
        output_shape = output.shape
        output = output.view(-1, C)
        target = target.view(-1)
        log_p = log_softmax(output, dim=1)
        if label_smoothing > 0:
            num_classes = log_p.size(1)
            smoothed_labels = (1 - label_smoothing) * torch.nn.functional.one_hot(target, num_classes=num_classes).to(dtype=output.dtype) + \
                            label_smoothing / num_classes
            log_likelihood = -torch.sum(smoothed_labels * log_p, dim=1)
        else:
            # Reshape target to match output dimensions for advanced indexing
            target = target.unsqueeze(1)  # Adds a class dimension to target
            log_likelihood = -log_p.gather(1, target).squeeze(1)  # Remove extra class dimension after gather
        if reduction == "mean":
            loss = torch.mean(log_likelihood)
        elif reduction == "sum":
            loss = torch.sum(log_likelihood)
        else:
            raise NotImplementedError
        return loss, log_p, target, output_shape
    loss, log_p, target, output_shape = compute(output, target, reduction, label_smoothing)
    # Prepare save_for_backward if using this in a custom backward pass
    save_for_backward = (log_p, target, reduction, label_smoothing, output_shape)
    return loss, save_for_backward


def cross_entropy_loss_backward_torchjit(save_for_backward: tuple):
    log_p, target, reduction, label_smoothing, original_shape = save_for_backward
    @torch.jit.script
    def compute(log_p: torch.Tensor, target: torch.Tensor, reduction: str, label_smoothing: float):
        grad_output = torch.exp(log_p).clone()
        if label_smoothing > 0:
            num_classes = log_p.size(1)
            smoothed_labels = (1 - label_smoothing) * torch.nn.functional.one_hot(target.squeeze(1), num_classes=num_classes).to(dtype=grad_output.dtype) + \
                            label_smoothing / num_classes
            grad_output -= smoothed_labels
        else:
            grad_output.scatter_(1, target, grad_output.gather(1, target) - 1)  # scatter subtraction along class dimension
        if reduction == "mean":
            grad_output /= log_p.size(0)
        elif reduction == "sum":
            pass
        return grad_output
    grad_output = compute(log_p, target, reduction, label_smoothing)
    return grad_output.view(original_shape)

