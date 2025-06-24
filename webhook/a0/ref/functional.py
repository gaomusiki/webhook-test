from typing import Tuple, Optional

import torch
import torch.nn.functional as F
    
from typing import Tuple, Optional

import torch
import torch.nn.functional as F

from .utils import generate_mask


def matmul_with_importance(
    input: torch.Tensor,
    weight: torch.Tensor,
    probs: torch.Tensor,
    grad_output: Optional[torch.Tensor] = None,
    num_heads: int = 1,
    top_p: float = 1.0,
    top_k: Optional[int] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """matmul input and weight and return output (with optional grad_input, grad_weight whenever grad_output is given) 
    where only the important elements of the input tensor can be computed and gathered to the output tensor
    decided by the importance probability tensor, tuned by top_p and top_k
    
    Args:
        input (torch.Tensor): input tensor in the range of [-1, 1], with shape: [batch_size, seq_len, hidden_size]
        weight (torch.Tensor): weight tensor in the range of [-1, 1], with shape: [hidden_size, embed_size]
        probs (torch.Tensor): probability tensor in the range of [0, 1], with shape: [batch_size, seq_len]
        grad_output (Optional[torch.Tensor], optional): gradient for the output tensor, with shape: [t, hidden_size]. Defaults to None.
        num_heads (int): number of heads to split hidden_size
        top_p (float, [0., 1.]): only the elements with the probability equal or higher than top_p are important ones
        top_k (int, [1, ..., seq_len], optional): only the elements with the top_k highest probability are important ones
    
    Returns:
        output (torch.Tensor): output tensor, with shape: [t, num_heads, embed_size]
        grad_input (torch.Tensor, optional): gradient for the input tensor if grad_output is given, otherwise None
        grad_weight (torch.Tensor, optional): gradient for the weight tensor if grad_output is given, otherwise None
    """
    # raise NotImplementedError("TODO: Assignment0 - Task1")
    
    # get the shape from (b, s, h)
    _, _, h = input.shape
    
    # generate the importance mask
    mask = generate_mask(probs, top_p, top_k)
    
    # get the total important seq len
    t = mask.sum()
    
    # require grad for input and weight if grad_output is given
    if grad_output is not None:
        assert grad_output.shape[0] == t, f"grad_output should have shape {(t, h)}, got {grad_output.shape}"
        input.requires_grad_(True)
        weight.requires_grad_(True)
        
    # gather the important elements in input and split into multi-heads, 
    # with shape: (b, s, h) => (t*h, ) => (t, h) => (t, nh, hd)
    input_ = input.masked_select(
        mask.unsqueeze(-1).expand(-1, -1, h).bool() # shape: (b, s) => (b, s, 1) => (b, s, h), dtype: long -> bool
    ).reshape(-1, h).view(t, num_heads, -1)

    # reshape weight from (h, e) to (nh, hd, e)
    weight_ = weight.view(num_heads, -1, weight.shape[-1])
    
    # matmul input and weight to get output, with shape: (t, nh, e)
    
    #---- method1: use @ operator ----#
    # output = (
    #     input_.transpose(0, 1) # shape: (t, nh, hd) => (nh, t, hd)
    #     @ weight_
    # ).transpose(0, 1) # shape: (nh, t, e) => (t, nh, e)
    
    #---- method2: use bmm function ----#
    # output = torch.bmm(
    #     input_.transpose(0, 1), # shape: (t, nh, hd) => (nh, t, hd)
    #     weight_
    # ).transpose(0, 1) # shape: (nh, t, e) => (t, nh, e)
    
    #---- method3: use einsum function ----#
    output = torch.einsum('thd,hde->the', input_, weight_)
    
    # run backward to get grad_input and grad_weight if grad_output is given
    if grad_output is not None:
        output.backward(grad_output)
        grad_input, grad_weight = input.grad.clone(), weight.grad.clone()
        input.grad, weight.grad = None, None
        input.requires_grad_(False)
        weight.requires_grad_(False)
    else:
        grad_input, grad_weight = None, None
    
    # return output and grad_input, grad_weight
    return output, grad_input, grad_weight