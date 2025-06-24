from typing import Optional

import torch
import torch.nn.functional as F


def generate_mask(
    probs: torch.Tensor, 
    top_p: float = 1.0, 
    top_k: Optional[int] = None
) -> torch.Tensor:
    # get the shape from (b, s)
    _, s = probs.shape
    
    # get the top_p mask with shape: (b, s)
    top_p_mask = probs >= top_p
    
    # get the top_k mask with shape: (b, s)
    if top_k is not None:
        top_k_indices = probs.topk(k=top_k, dim=-1).indices # shape: (b, top_k)
        top_k_mask = F.one_hot(top_k_indices, num_classes=s).sum(dim=1) # shape: (b, top_k, s) -> (b, s)
    else:
        top_k_mask = torch.ones_like(top_p_mask)
        
    # get the important elements mask with shape: (b, s)
    mask = top_p_mask & top_k_mask.to(torch.bool)
    
    return mask