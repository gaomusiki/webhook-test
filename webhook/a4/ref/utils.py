from typing import Optional, Sequence, List, Dict, Any, Type

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


def construct_kvcache_args(
    b: int,
    nh: int,
    hd: int,
    qkv_layout: str,
    ops: List[Dict[str, Any]],
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
    seed: int = 42,
) -> List[Sequence[Optional[torch.Tensor]]]:
    input_tensors = []
    
    for i, op in enumerate(ops):
        if op['op'] in ("set", "append"):
            s, seqlens = op['s'], op['seqlens']
            
            torch.manual_seed(seed + i)
            k = torch.randn(b, s, nh, hd, dtype=dtype, device=device)
            v = torch.randn_like(k)
            cu_seqlens = None
            
            if qkv_layout == "bshd":
                pass
            elif qkv_layout == "sbhd":
                k, v = [x.transpose(0, 1) for x in (k, v)]
            elif qkv_layout == "thd":
                assert b == 1, "b should be equal to 1 when qkv_layout is THD"
                assert seqlens is not None, "seqlens must be given when qkv_layout is THD"
                k, v = [x.squeeze(0) for x in (k, v)]
                cu_seqlens = torch.concat([
                    torch.zeros(1, dtype=torch.int32, device=device),
                    torch.tensor(seqlens, dtype=torch.int32, device=device).cumsum(dim=0)
                ], dim=0).to(torch.int32)
                assert cu_seqlens[-1] == (t := b * s), f"The sum of seqlens ({cu_seqlens[-1]}) != length ({t})"
            else:
                raise ValueError(f"Unsupported qkv_layout: {qkv_layout}")

            input_tensors.append((k, v, cu_seqlens))
        else:
            input_tensors.append(None)
        
    return input_tensors
           

def construct_decoder_args(
    kv_cache_class: Type["TransformerKVCache"], # type: ignore
    config: "TransformerConfig", # type: ignore
    b: int,
    s: int,
    seqlens: Optional[List[int]] = None,
    past_seqlen_kv: int = 0,
    past_seqlens: Optional[List[int]] = None,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
) -> Sequence[Optional[torch.Tensor]]:
    torch.manual_seed(config.init_base_seed)
    input = torch.randn(b, s, config.hidden_size, dtype=dtype, device=device)
    input_ids = torch.randint(0, config.vocab_size, (b, s), dtype=torch.int32, device=device)
    
    if seqlens is not None:
        assert config.qkv_layout.value == "thd", "if using varlen attn, the qkv_layout must be THD"
        assert b == 1, "b should be equal to 1 if using varlen attn"
        
        cu_seqlens = torch.concat([
                torch.zeros(1, dtype=torch.int32, device=device),
                torch.tensor(seqlens, dtype=torch.int32, device=device).cumsum(dim=0)
        ], dim=0).to(torch.int32)
        assert cu_seqlens[-1] == (t:=b*s), f"The sum of seqlens ({cu_seqlens[-1]}) != b*s ({t})"
    else:
        cu_seqlens = None
    
    if past_seqlen_kv > 0:
        if config.qkv_layout.value == "thd":
            assert past_seqlens is not None, "past_seqlens must be given when qkv_layout is THD and past_seqlen_kv > 0"
        kv_cache = kv_cache_class(
            qkv_layout=config.qkv_layout,
            num_layers=config.num_layers,
        )
        
        for layer_idx in range(config.num_layers):
            torch.manual_seed(config.init_base_seed + layer_idx)
            past_k = torch.randn(
                b, past_seqlen_kv, config.num_kv_head, config.head_dim, 
                dtype=config.param_dtype, device=config.param_device
            )
            past_v = torch.randn_like(past_k)
            past_cu_seqlens = None
            
            if config.qkv_layout.value == "bshd":
                pass
            elif config.qkv_layout.value == "sbhd":
                past_k, past_v = [x.transpose(0, 1) for x in (past_k, past_v)]
            elif config.qkv_layout.value == "thd":
                past_k, past_v = [x.squeeze(0) for x in (past_k, past_v)]
                past_cu_seqlens = torch.concat([
                    torch.zeros(1, dtype=torch.int32, device=device),
                    torch.tensor(past_seqlens, dtype=torch.int32, device=device).cumsum(dim=0)
                ], dim=0).to(torch.int32)
                assert past_cu_seqlens[-1] == (t := len(past_k)), \
                    f"The sum of past seqlens ({past_cu_seqlens[-1]}) != past length ({t})"
            else:
                raise ValueError(f"Unsupported qkv_layout: {config.qkv_layout}")
            
            kv_cache.set(layer_idx, past_k, past_v, cu_seqlens=past_cu_seqlens)
    else:
        kv_cache = None
    
    return input, input_ids, cu_seqlens, kv_cache
