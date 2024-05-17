import torch
from typing import Tuple, Optional
import math

from . import init

def embedding_init(
    num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = -1,
    max_norm: Optional[float] = None, norm_type: float = 2., scale_grad_by_freq: bool = False,
    sparse: bool = False, device: torch.device = 'cuda:0', dtype=torch.float32
) -> torch.Tensor:
    if padding_idx is not None:
        if padding_idx > 0:
            assert padding_idx < num_embeddings, 'Padding_idx must be within num_embeddings'
        elif padding_idx < 0:
            assert padding_idx >= -num_embeddings, 'Padding_idx must be within num_embeddings'
            padding_idx = num_embeddings + padding_idx
    weight = torch.zeros((num_embeddings, embedding_dim)).to(device=device, dtype=dtype)
    weight = init.normal_(weight)
    weight = _fill_padding_idx_with_zero(weight, padding_idx)
    init_info = (max_norm, norm_type, scale_grad_by_freq, sparse)
    return weight, init_info

def embedding_forward(input: torch.Tensor, weight: torch.Tensor, init_info: Tuple):
    # if input.dim() < 2:
    #     raise ValueError("Input tensor must be at least 2D")
    max_norm, norm_type, scale_grad_by_freq, sparse = init_info
    if max_norm is not None:
        # Note [embedding_renorm contiguous]
        # `embedding_renorm_` will call .contiguous() on input anyways, so we
        # call it here and take advantage of the improved locality in the
        # `embedding` call below too.
        input = input.contiguous()
        # Note [embedding_renorm set_grad_enabled]
        # XXX: equivalent to
        # with torch.no_grad():
        #   torch.embedding_renorm_
        # remove once script supports set_grad_enabled
        _no_grad_embedding_renorm_(weight, input, max_norm, norm_type)
    save_for_backward = (weight, input)
    input_size = list(input.size())
    output = weight.index_select(0, input.view(-1)).view((*input_size, weight.size(-1)))
    return output, save_for_backward

def embedding_backward(grad_output: torch.Tensor, save_for_backward: Tuple[torch.Tensor, torch.Tensor]):
    weight, input = save_for_backward
    grad_weight = torch.zeros_like(weight)
    grad_input = None
    # for i in range(input.numel()):
    #     index = input.view(-1)[i]
    #     grad_weight[index] += grad_output.view(-1, weight.size(1))[i]
    input_flat = input.view(-1)
    grad_output_flat = grad_output.view(-1, weight.size(1))
    grad_weight.index_add_(0, input_flat, grad_output_flat)
    assert grad_weight.shape == weight.shape, "Gradient weight shape mismatch"
    return grad_input, grad_weight


def _fill_padding_idx_with_zero(weight: torch.Tensor, padding_idx) -> None:
    if padding_idx is not None:
        weight[padding_idx].fill_(0)
    return weight

def _no_grad_embedding_renorm_(weight: torch.Tensor, input: torch.Tensor, max_norm: float, norm_type: float) -> Tuple[torch.Tensor, torch.Tensor]:
    torch.embedding_renorm_(weight, input, max_norm, norm_type)

