from typing import Optional, Sequence, List

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
        top_k_mask = torch.ones_like(top_p_mask, dtype=torch.bool)
        
    # get the important elements mask with shape: (b, s)
    mask = top_p_mask & top_k_mask
    
    return mask


def safe_clone(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if x is None:
        return None
    else:
        return x.clone()
    

def construct_offline_attn_args(
    b: int,
    sq: int,
    skv: int,
    hq: int,
    hkv: int,
    hd: int,
    qkv_pack_format: str = "q_k_v_packed",
    qkv_layout: str = "bshd",
    seqlens_q: Optional[List[int]] = None,
    seqlens_kv: Optional[List[int]] = None,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    seed: int = 42,
) -> Sequence[Optional[torch.Tensor]]:
    torch.manual_seed(seed)
    q = torch.randn((b, sq, hq, hd), dtype=dtype, device=device)
    k = torch.randn((b, skv, hkv, hd), dtype=dtype, device=device)
    v = torch.randn((b, skv, hkv, hd), dtype=dtype, device=device)
    
    if qkv_layout == "thd":
        assert seqlens_q is not None, "THD layout requires cu_seqlens_q"
        assert seqlens_kv is not None, "THD layout requires cu_seqlens_kv"
        
        cu_seqlens_q, cu_seqlens_kv =[
            torch.concat([
                torch.zeros(1, dtype=torch.int32, device=device),
                torch.tensor(x, dtype=torch.int32, device=device).cumsum(dim=0)
            ], dim=0)
            for x in (seqlens_q, seqlens_kv)
        ]
        
        assert cu_seqlens_q[-1] == b*sq, f"cu_seqlens_q[-1]({cu_seqlens_q[-1]}) == b*sq({b*sq})"
        assert cu_seqlens_kv[-1] == b*skv, f"cu_seqlens_kv[-1]({cu_seqlens_kv[-1]}) == b*skv({b*skv})"
        
        q, k, v = [
            x.view(-1, *x.shape[-2:]).contiguous() 
            for x in (q, k, v)
        ]
    else:
        assert seqlens_q is None, "QKV layout does not require cu_seqlens_q"
        assert seqlens_kv is None, "QKV layout does not require cu_seqlens_kv"
        cu_seqlens_q, cu_seqlens_kv = None, None
        
        if qkv_layout == "sbhd":
            q, k, v = [
                x.transpose(0, 1).contiguous() 
                for x in (q, k, v)
            ]
    
    if qkv_pack_format == "qkv_packed":
        assert sq == skv, "QKV pack format requires sq == skv"
        q = torch.concat((q, k, v), dim=-2)
        k, v = None, None
    elif qkv_pack_format == "q_kv_packed":
        k = torch.concat((k, v), dim=-2)
        v = None
    
    return q, k, v, cu_seqlens_q, cu_seqlens_kv


def construct_online_attn_args(
    b: int,
    sq: int,
    skv: int,
    hq: int,
    hkv: int,
    hd: int,
    bq: int,
    bkv: int,
    bqi: int,
    bkvi: int,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    seed: int = 42,
) -> Sequence[torch.Tensor]:
    nbq = (sq + bq - 1) // bq
    nbk = (skv + bkv - 1) // bkv
    assert bqi < nbq, f"bqi({bqi}) >= nbq({nbq})"
    assert bkvi < nbk, f"bkvi({bkvi}) >= nbk({nbk})"
    
    torch.manual_seed(seed)
    q = torch.randn((b, sq, hq, hd), dtype=dtype, device=device)
    k = torch.randn((b, skv, hkv, hd), dtype=dtype, device=device)
    v = torch.randn((b, skv, hkv, hd), dtype=dtype, device=device)
    global_o = torch.randn_like(q)
    global_lse = torch.rand((b, hq, sq), dtype=torch.float32, device=device)
    
    q = F.pad(q, pad=(0, 0, 0, 0, 0, nbq*bq - sq), mode="constant", value=0)
    k = F.pad(k, pad=(0, 0, 0, 0, 0, nbk*bkv - skv), mode="constant", value=0)
    v = F.pad(v, pad=(0, 0, 0, 0, 0, nbk*bkv - skv), mode="constant", value=0)
    
    q = q[:, bqi*bq:(bqi+1)*bq, :, :]
    k = k[:, bkvi*bkv:(bkvi+1)*bkv, :, :]
    v = v[:, bkvi*bkv:(bkvi+1)*bkv, :, :]
    
    return q, k, v, global_o, global_lse