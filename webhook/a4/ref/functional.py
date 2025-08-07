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


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    input: torch.Tensor, 
    cos: torch.Tensor, 
    sin: torch.Tensor, 
) -> torch.Tensor:
    """Applies rotary positional embedding to the input tensor.
    
    Args:
        input(torch.Tensor): input tensor, with shape: [batch_size, seq_len, num_heads, head_dim]
        cos(torch.Tensor): cos basis tensor, with shape: [seq_len, head_dim]
        sin(torch.Tensor): sin basis tensor, with shape: [seq_len, head_dim]
    
    Returns:
        output(torch.Tensor): embedded output tensor, with shape: [batch_size, seq_len, num_heads, head_dim]
    """
    # raise NotImplementedError("TODO: Assignment1 - Task3")
    
    cos = cos.unsqueeze(1).unsqueeze(0)  # shape: (1, s, 1, hd)
    sin = sin.unsqueeze(1).unsqueeze(0)  # shape: (1, s, 1, hd)

    # apply rotary pos emb to get the embedded output with shape: (1, s, 1, hd)
    output = (input * cos) + (rotate_half(input) * sin)
    return output


def safe_softmax(
    a: torch.Tensor, 
    dim: int = -1,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Safely applies softmax to the input tensor.
    where the all -inf rows will be set to all-zero rows.
    """
    all_neg_inf = (a == float('-inf')).all(dim=dim, keepdim=True)
    
    sm = F.softmax(a, dim=dim, dtype=dtype)
    
    sm = torch.where(all_neg_inf, torch.zeros_like(sm), sm)

    return sm


def safe_subtract(
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """Safely subtracts two tensors.
    where the subtraction results of two -inf will be set to -inf.
    """
    eq = (a == b) & (a == float('-inf'))
    
    sub = a - b
    sub = torch.where(eq, torch.fill(sub, float('-inf')), sub)
    
    return sub


def correct_attn_lse(
    lse1: torch.Tensor,
    lse2: torch.Tensor,
) -> torch.Tensor:
    """Corrects the log sum exp tensor for online attention.
    
    Args:
        lse1(torch.Tensor): log sum exp tensor, with shape: [batch_size, num_heads, seq_len]
        lse2(torch.Tensor): log sum exp tensor, with shape: [batch_size, num_heads, seq_len]
    
    Returns:
        lse(torch.Tensor): corrected log sum exp tensor, with shape: [batch_size, num_heads, seq_len]
    """
    min_lse = torch.min(lse1, lse2).to(torch.float32)
    max_lse = torch.max(lse1, lse2).to(torch.float32)
    
    # formula: lse = log(exp(lse1) + exp(lse2))
    #              = lse1 + log(1 + exp(lse2 - lse1))
    #              = max_lse + log(1 + exp(min_lse - max_lse))
    #              = max_lse + log1p(exp(min_lse - max_lse))
    #              = max_lse + softplus(min_lse - max_lse)
    lse = max_lse + F.softplus(safe_subtract(min_lse, max_lse))
    
    return lse.to(lse1.dtype)


def correct_attn_output(
    o1: torch.Tensor,
    lse1: torch.Tensor,
    o2: torch.Tensor,
    lse2: torch.Tensor,
    lse: torch.Tensor,
) -> torch.Tensor:
    """Corrects the output tensor for online attention.
    
    Args:
        o1(torch.Tensor): local output tensor o1, with shape: [batch_size, seq_len, num_heads, head_dim]
        lse1(torch.Tensor): local lse for o1, with shape: [batch_size, num_heads, seq_len]
        o2(torch.Tensor): local output tensor o2, with shape: [batch_size, seq_len, num_heads, head_dim]
        lse2(torch.Tensor): local lse for o2, with shape: [batch_size, num_heads, seq_len]
        lse(torch.Tensor): global lse, with shape: [batch_size, num_heads, seq_len]
    
    Returns:
        o(torch.Tensor): corrected global output tensor, with shape: [batch_size, seq_len, num_heads, head_dim]
    """
    # formula: lsei_ = exp(lsei - lse)
    # shape: [b, h, s] -> [b, s, h] -> [b, s, h, 1]
    lse1_, lse2_ = [
        safe_subtract(lsei, lse).exp().transpose(-1, -2).unsqueeze(-1).to(torch.float32)
        for lsei in [lse1, lse2]
    ]
    
    o = lse1_ * o1 + lse2_ * o2
    
    return o.to(o1.dtype)
