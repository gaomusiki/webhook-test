from typing import Optional, Tuple
from enum import Enum

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# from assignment1 implementations
from .norm import GroupRMSNorm

from ..functional import (
    safe_softmax,
    correct_attn_lse, 
    correct_attn_output
)


class AttnQKVPackFormat(Enum):
    QKV = "qkv_packed"
    Q_KV = "q_kv_packed"
    Q_K_V = "q_k_v_packed"


class AttnQKVLayout(Enum):
    BSHD = "bshd"
    SBHD = "sbhd"
    THD = "thd"


class OfflineSlidingWindowAttn(nn.Module):
    """Offline Sliding-Window Attention module
    This is a generalized variant of standard self-attention equipped with the sliding-window trick \
        to make use of spatial locality in language for computational efficiency, \
        with applying other methods to improve stability.
    """
    def __init__(
        self,
        head_dim: int,
        num_q_head: int,
        num_kv_head: int,
        qkv_pack_format: AttnQKVPackFormat = AttnQKVPackFormat.Q_K_V,
        qkv_layout: AttnQKVLayout = AttnQKVLayout.BSHD,
        window_size: Optional[int] = None,
        causal: bool = False,
        softmax_dropout_rate: float = 0.0,
        softmax_dropout_seed: int = 42,
        softmax_scale: Optional[float] = None,
        softmax_cap: Optional[float] = None,
        softmax_temp: float = 1.0,
        softmax_clip_range: Tuple[float, float] = (0., 1.),
        apply_qk_norm: bool = False,
        group_size: Optional[int] = None,
        eps: float = 1e-5,
        init_range: tuple = (-1.0, 1.0),
        init_seed: int = 42,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ):
        """Initialize Offline Sliding-Window Attention module
        
        Args:
            head_dim(int): head dimension size
            num_q_head(int): number of query heads
            num_kv_head(int): number of key/value heads
            qkv_pack_format(AttnQKVPackFormat, default = "q_k_v_packed"): qkv packed format
            qkv_layout(AttnQKVLayout, default = "bshd"): qkv shape layout
            window_size(int, default = None): window size
            causal(bool, default = False): if True, then apply causal masking as a prior to only allow unidirectional self-attention, otherwise bidirectional
            softmax_dropout_rate(float, default = 0.0): dropout probability for the softmax probs
            softmax_dropout_seed(int, default = 42): random seed for softmax drooput
            softmax_scale(float, default = None): softmax scale factor, if None, then applying the standard value: 1/√d
            softmax_cap(float, default = None): softmax capping to control the magnitude of the logits, if None, then NO capping is applied
            softmax_temp(float, default = 1.0): softmax temperature to control the sharpness of the distribution, only apply when softmax_cap is None
            softmax_clip_range(float, default = (0.0, 1.0): the range for softmax clipping to prevent the outliers from growing further
            apply_qk_norm(bool, default = False): if True, then apply qk norm
            group_size(int, optional, default = None): group size to split hidden size of query / key for GroupRMSNorm, if None, then set it to `head_dim`, if applying qk norm
            eps(float, default = 1e-5): epsilon for GroupRMSNorm, if applying qk norm
            init_range(tuple, default = (-1.0, 1.0)): the range of the initialization uniform distribution for GroupRMSNorm, if applying qk norm
            init_seed(int, default = 42): initialization seed for GroupRMSNorm, if applying qk norm
            dtype(torch.dtype, default = torch.float32): parameter dtype for GroupRMSNorm, if applying qk norm
            device(str, default = "cpu"): parameter device for GroupRMSNorm, if applying qk norm
        """
        super().__init__()
        # raise NotImplementedError("Assignment3 - Task1")
        
        assert group_size is None or head_dim % group_size == 0, f"The head dimension ({head_dim}) must be divisible by the group size ({group_size})"
        assert num_q_head % num_kv_head == 0, f"The number of query heads ({num_q_head}) must be divisible by the number of key/value heads ({num_kv_head})"
        assert softmax_temp > 0., "The softmax temperature must be greater than 0"
        assert softmax_cap is None or softmax_cap > 0., "The softmax capping must be greater than 0 if given"
        assert softmax_clip_range[0] < softmax_clip_range[1], "The softmax clip range must be a valid range (l,r), s.t. l < r"
        
        self.head_dim = head_dim
        self.num_q_head = num_q_head
        self.num_kv_head = num_kv_head
        
        self.qkv_pack_format = qkv_pack_format
        self.qkv_layout = qkv_layout
        
        self.window_size = window_size
        self.causal = causal
        
        self.softmax_dropout_rate = softmax_dropout_rate
        self.softmax_dropout_seed = softmax_dropout_seed
        
        self.softmax_scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(head_dim)
        self.softmax_cap = softmax_cap
        self.softmax_temp = softmax_temp
        self.softmax_clip_range = softmax_clip_range
        
        self.apply_qk_norm = apply_qk_norm
        self.group_size = group_size if group_size is not None else head_dim
        self.eps = eps
        self.init_range = init_range
        self.init_seed = init_seed
        self.dtype = dtype
        self.device = device
        
        self.q_hidden_size = self.num_q_head * self.head_dim
        self.k_hidden_size = self.num_kv_head * self.head_dim
        self.kv_repeat_times = self.num_q_head // self.num_kv_head
        
        self.softmax_clip_range_len = self.softmax_clip_range[1] - self.softmax_clip_range[0]
        self.softmax_clip_range_min = self.softmax_clip_range[0]
        
        # init dropout layer
        self.softmax_dropout = nn.Dropout(softmax_dropout_rate)
        
        # init q,k,v,o reshape funtion
        # i.e. q, k, v from the original shape to "bshd", while o from "bshd" to the original shape
        if self.qkv_layout is AttnQKVLayout.BSHD:
            # (b, s, h, d) -> (b, s, h, d)
            self.qkv_reshape_func = lambda q, k, v: (q, k, v)
            # (b, s, h, d) -> (b, s, h, d)
            self.o_reshape_func = lambda o: o
        elif self.qkv_layout is AttnQKVLayout.SBHD:
            # (s, b, h, d) -> (b, s, h, d)
            self.qkv_reshape_func = lambda q, k, v: [
                x.transpose(0, 1) for x in (q, k, v)
            ]
            # (b, s, h, d) -> (s, b, h, d)
            self.o_reshape_func = lambda o: o.transpose(0, 1)
        elif self.qkv_layout is AttnQKVLayout.THD:
            # (t, h, d) -> (1, t, h, d)
            self.qkv_reshape_func = lambda q, k, v: [
                x.unsqueeze(0) for x in (q, k, v)
            ]
            # (1, t, h, d) -> (t, h, d)
            self.o_reshape_func = lambda o: o.squeeze(0)
        
        # init q,k,v,o s/nh transpose function
        # i.e. q, k, v from "bshd" to "bhsd", while o from "bhsd" to "bshd"
        self.qkv_trans_func = lambda q, k, v: [
            x.transpose(1, 2) for x in (q, k, v)
        ]
        self.o_trans_func = lambda o: o.transpose(1, 2)
        
        # init q,k norm layers and qk norm function
        if self.apply_qk_norm:
            self.q_norm_layer = GroupRMSNorm(
                hidden_size=self.q_hidden_size,
                group_size=self.group_size,
                eps=self.eps,
                init_range=self.init_range,
                init_seed=self.init_seed,
                dtype=self.dtype,
                device=self.device,
            )
            self.k_norm_layer = GroupRMSNorm(
                hidden_size=self.k_hidden_size,
                group_size=self.group_size,
                eps=self.eps,
                init_range=self.init_range,
                init_seed=self.init_seed,
                dtype=self.dtype,
                device=self.device,
            )
            self.qk_norm_func = lambda q, k: [ # assuming q,k already have shape: (b, s, h, d)
                norm_layer(x.view(*x.shape[:2], -1)).view(*x.shape[:2], num_head, self.head_dim)
                for x, norm_layer, num_head in zip(
                    (q, k), 
                    (self.q_norm_layer, self.k_norm_layer), 
                    (self.num_q_head, self.num_kv_head)
                )
            ]
        else:
            self.qk_norm_func = lambda q, k: (q, k)
        
        # init qkv split function
        if self.qkv_pack_format is AttnQKVPackFormat.QKV:
            self.qkv_split_func = lambda qkv, _k, _v: (
                torch.split(
                    qkv,
                    split_size_or_sections=[self.num_q_head, self.num_kv_head, self.num_kv_head],
                    dim=-2, # nh dim
                )
            )
        elif self.qkv_pack_format is AttnQKVPackFormat.Q_KV:
            self.qkv_split_func = lambda q, kv, _v: (
                q, 
                *torch.split(
                    kv, 
                    split_size_or_sections=self.num_kv_head,
                    dim=-2 # nh dim
                )
            )
        elif self.qkv_pack_format is AttnQKVPackFormat.Q_K_V:
            self.qkv_split_func = lambda q, k, v: (q, k, v)
            
        # init kv repeat function
        if self.kv_repeat_times == 1:
            self.kv_repeat_func = lambda k, v: (k, v)
        else:
            self.kv_repeat_func = lambda k, v: [
                x.repeat_interleave(repeats=self.kv_repeat_times, dim=-2)
                for x in [k, v]
            ]
    
        # init attn forward function
        if self.qkv_layout is AttnQKVLayout.THD:
            self.attn_fwd_func = self._varlen_attn_fwd_func
        else:
            self.attn_fwd_func = self._non_varlen_attn_fwd_func
    
        # init softmax function to the safe one
        # since certain row of attn logits may be all -inf
        # self.softmax_func = lambda x: F.softmax(x, dim=-1, dtype=torch.float32)
        self.softmax_func = lambda x: safe_softmax(x, dim=-1, dtype=torch.float32)
    
    def forward(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor] = None,
        v: Optional[torch.Tensor] = None,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_k: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """The forward pass of Offline Sliding-Window Attention module
        
        Args:
            q(torch.Tensor): query tensor, or query-key-value packed tensor if the qkv_pack_format is "qkv_packed"
            k(Optional[torch.Tensor], default = None): key tensor, or key-value packed tensor if the qkv_pack_format is "q_kv_packed", or None if qkv_pack_format is "qkv_packed"
            v(Optional[torch.Tensor], default = None): value tensor if the qkv_pack_format is "q_k_v_packed", otherwise None
            cu_seqlens_q(Optional[torch.Tensor], default = None): cumulative sequence lengths for query tensor, with shape: [batch_size + 1, ]
            cu_seqlens_k(Optional[torch.Tensor], default = None): cumulative sequence lengths for key tensor, with shape: [batch_size + 1, ]
        Returns:
            torch.Tensor: output tensor o, with the same shape as q
        """
        # raise NotImplementedError("Assignment3 - Task1")
        
        # split q,k,v
        q, k, v = self.qkv_split_func(q, k, v)
        
        # reshape q,k,v
        q, k, v = self.qkv_reshape_func(q, k, v)
        
        # normalize q,k
        q, k = self.qk_norm_func(q, k)
        
        # repeat k,v
        k, v = self.kv_repeat_func(k, v)
        
        # apply attn forward
        o = self.attn_fwd_func(q, k, v, cu_seqlens_q, cu_seqlens_k)
        
        # reshape o
        o = self.o_reshape_func(o)
        
        return o.to(dtype=q.dtype, device=q.device)
    
    def reset_parameters(self):
        """Initialize the optional q, k norm parameters of Offline Sliding-Window Attention module"""
        # raise NotImplementedError("Assignment3 - Task1")
        
        if self.apply_qk_norm:
            self.q_norm_layer.reset_parameters()
            self.k_norm_layer.reset_parameters()
    
    def _attn_fwd_func(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """non-varlen attn forward function
        
        Args:
            q(torch.Tensor): query tensor, with shape: [batch_size, seq_len_q, num_head, head_dim]
            k(torch.Tensor): key tensor, with shape: [batch_size, seq_len_kv, num_head, head_dim]
            v(torch.Tensor): value tensor, with shape: [batch_size, seq_len_kv, num_head, head_dim]
            
        Returns:
            torch.Tensor: output tensor o, with shape: [batch_size, seq_len_q, num_head, head_dim]
            torch.Tensor: softmax lse, with shape: [batch_size, num_head, seq_len_q]
        """
        # transpose q,k,v from "bshd" to "bhsd"
        q, k, v = self.qkv_trans_func(q, k, v)
        
        # compute logits = q @ k.T * softmax_scale, with shape: (b, h, sq, skv)
        attn_logits = q @ k.transpose(-2, -1) * self.softmax_scale
        
        # adjust logits magnitude
        if self.softmax_cap is not None: # apply softmax capping
            attn_logits = F.tanh(attn_logits / self.softmax_cap) * self.softmax_cap
        else: # apply softmax temperature
            attn_logits /= self.softmax_temp
        
        # generate attn mask, with shape: (1, 1, sq, skv)
        attn_mask = self._generate_attn_mask(q, k)
        
        # apply attn mask to logits
        attn_logits += attn_mask
        
        # compute lse, with shape: (b, h, sq)
        attn_lse = attn_logits.logsumexp(dim=-1)
        
        # apply softmax to attn weights, with shape: (b, h, sq, skv)
        attn_weights = self.softmax_func(attn_logits).to(q.dtype)
        
        # apply softmax clipping to prevent outlier
        attn_weights = torch.clip(
            self.softmax_clip_range_len * attn_weights + self.softmax_clip_range_min,
            min=0.0, max=1.0
        )
        
        # apply softmax dropout
        torch.manual_seed(self.softmax_dropout_seed)
        attn_weights = self.softmax_dropout(attn_weights)
        
        # compute o = att_weights @ v, with shape: (b, h, sq, d)
        o = attn_weights @ v
        
        # transpose o from "bhsd" to "bshd"
        o = self.o_trans_func(o)
        
        return o, attn_lse
    
    def _non_varlen_attn_fwd_func(
        self,
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor,
        *args,
    ) -> torch.Tensor:
        """non-varlen attn forward function
        
        Args:
            q(torch.Tensor): query tensor, with shape: [batch_size, seq_len_q, num_head, head_dim]
            k(torch.Tensor): key tensor, with shape: [batch_size, seq_len_kv, num_head, head_dim]
            v(torch.Tensor): value tensor, with shape: [batch_size, seq_len_kv, num_head, head_dim]
            *args: to absorb the cu_seqlens kwargs which are not used in non-varlen attn
            
        Returns:
            torch.Tensor: output tensor o, with shape: [batch_size, seq_len_q, num_head, head_dim]
        """
        o, _ = self._attn_fwd_func(q, k, v)
        
        return o
    
    def _varlen_attn_fwd_func(
        self,
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
    ) -> torch.Tensor:
        """varlen attn forward function
        
        Args:
            q(torch.Tensor): query tensor, with shape: [1, total_seq_len_q, num_head, head_dim]
            k(torch.Tensor): key tensor, with shape: [1, total_seq_len_kv, num_head, head_dim]
            v(torch.Tensor): value tensor, with shape: [1, total_seq_len_kv, num_head, head_dim]
            cu_seqlens_q(torch.Tensor): cumulative sequence lengths for query tensor, with shape: [batch_size + 1, ]
            cu_seqlens_k(torch.Tensor): cumulative sequence lengths for key tensor, with shape: [batch_size + 1, ]
            
        Returns:
            torch.Tensor: output tensor o, with shape: [1, total_seq_len_q, num_head, head_dim]
        """
        # get batch size
        b = cu_seqlens_q.shape[0] - 1
        
        # init output buffer, with shape: (1, t, h, d)
        o = torch.zeros_like(q)
        
        # apply attn fwd for each seq in the batch
        for bi in range(b):
            # compute the [start_idx, end_idx) of q, kv for the i-th seq in the batch
            sq_si, sq_ei = cu_seqlens_q[bi], cu_seqlens_q[bi + 1]
            skv_si, skv_ei = cu_seqlens_k[bi], cu_seqlens_k[bi + 1]
            
            o[:, sq_si: sq_ei, ...].add_(
                self._attn_fwd_func(
                    q[:, sq_si: sq_ei, ...],
                    k[:, skv_si: skv_ei, ...],
                    v[:, skv_si: skv_ei, ...],
                )[0]
            )
            
        return o
        
    def _generate_attn_mask(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> torch.Tensor:
        """Generate attention mask
        
        Args:
            q(torch.Tensor): query tensor, with shape: [batch_size, num_heads, seq_len_q,  head_dim]
            k(torch.Tensor): key tensor, with shape: [batch_size, num_heads, seq_len_kv, head_dim]
            
        Returns:
            torch.Tensor: attention mask, with shape: [1, 1, seq_len_q, seq_len_kv]
        """        
        # get seqlen of q, kv from shape "bhsd"
        sq, skv = q.shape[-2], k.shape[-2]
        
        # get the [start, end) range of sq, skv
        # which aligns the bottom right part of the full square attention matrix
        # following the flash-attn's settings
        s = max(sq, skv)
        sq_start, sq_end = s - sq, s
        skv_start, skv_end = s - skv, s
        
        # set the window size, if None, then set to infinity window
        w = self.window_size if self.window_size is not None else s
        
        # init attn mask, with shape: [sq, skv]
        attn_mask = torch.zeros((sq, skv), dtype=q.dtype)

        # init q row-index and k col-index
        qi = torch.arange(sq_start, sq_end).view(-1, 1)  # [sq, 1]
        kj = torch.arange(skv_start, skv_end).view(1, -1)  # [1, skv]

        # compute [lb, ub) of kj for each qi
        # non causal: [i-w, i] | causal: [i-w, i+w]
        # where lb >= skv_start, ub <= skv_end
        lb = torch.clamp(
            qi - w,
            min=skv_start,
        )
        ub = torch.clamp(
            qi + w + 1,
            max=skv_end,
        ) if not self.causal else (qi + 1)

        # compute the bool mask
        # where 'True' means the position to be masked out
        bool_mask = (kj < lb) | (kj >= ub)

        # fill the attn mask
        # where '0' means the position to keep,
        # while '-inf' means the position to be masked out
        attn_mask.masked_fill_(
            bool_mask,
            float("-inf")
        )
        
        # return with shape: (1, 1, sq, skv) to broadcast
        return attn_mask.unsqueeze(0).unsqueeze(0).to(q.device)
    

class OnlineSlidingWindowAttn(OfflineSlidingWindowAttn):
    """Online Sliding-Window Attention module
    This is a online version of Offline Sliding-Window Attention module \
        which only apply attention on a block of q, k, v in "bshd" layout and "q_k_v_packed" format \
            and update the global o with the local block of o using lse
    """
    def __init__(
        self,
        seqlen_q: int,
        seqlen_kv: int,
        block_size_q: int,
        block_size_kv: int,
        head_dim: int,
        num_q_head: int,
        num_kv_head: int,
        window_size: Optional[int] = None,
        causal: bool = False,
        softmax_scale: Optional[float] = None,
        softmax_cap: Optional[float] = None,
        softmax_temp: float = 1.0,
        apply_qk_norm: bool = False,
        group_size: Optional[int] = None,
        eps: float = 1e-5,
        init_range: tuple = (-1.0, 1.0),
        init_seed: int = 42,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ):
        """Initialize Online Sliding-Window Attention module
        
        Args:
            seqlen_q(int): the sequence length of q
            seqlen_kv(int): the sequence length of kv
            block_size_q(int): the block size of q
            block_size_kv(int): the block size of kv
            head_dim(int): head dimension size
            num_q_head(int): number of query heads
            num_kv_head(int): number of key/value heads
            window_size(int, default = None): window size
            causal(bool, default = False): if True, then apply causal masking as a prior to only allow unidirectional self-attention, otherwise bidirectional
            softmax_scale(float, default = None): softmax scale factor, if None, then applying the standard value: 1/√d
            softmax_cap(float, default = None): softmax capping to control the magnitude of the logits, if None, then NO capping is applied
            softmax_temp(float, default = 1.0): softmax temperature to control the sharpness of the distribution, only apply when softmax_cap is None
            apply_qk_norm(bool, default = False): if True, then apply qk norm
            group_size(int, optional, default = None): group size to split hidden size of query / key for GroupRMSNorm, if None, then set it to `head_dim`, if applying qk norm
            eps(float, default = 1e-5): epsilon for GroupRMSNorm, if applying qk norm
            init_range(tuple, default = (-1.0, 1.0)): the range of the initialization uniform distribution for GroupRMSNorm, if applying qk norm
            init_seed(int, default = 42): initialization seed for GroupRMSNorm, if applying qk norm
            dtype(torch.dtype, default = torch.float32): parameter dtype for GroupRMSNorm, if applying qk norm
            device(str, default = "cpu"): parameter device for GroupRMSNorm, if applying qk norm
        """
        super().__init__(
            head_dim=head_dim,
            num_q_head=num_q_head,
            num_kv_head=num_kv_head,
            window_size=window_size,
            causal=causal,
            softmax_scale=softmax_scale,
            softmax_cap=softmax_cap,
            softmax_temp=softmax_temp,
            apply_qk_norm=apply_qk_norm,
            group_size=group_size,
            eps=eps,
            init_range=init_range,
            init_seed=init_seed,
            dtype=dtype,
            device=device,
        )
        # raise NotImplementedError("Assignment3 - Task2")
        
        # init seqlens
        self.seqlen_q = seqlen_q
        self.seqlen_kv = seqlen_kv
        self.seqlen_q_padded = self._compute_padded_seqlen(seqlen_q, block_size_q)
        self.seqlen_kv_padded = self._compute_padded_seqlen(seqlen_kv, block_size_kv)
        
        # init block attrs
        self.block_size_q = block_size_q
        self.block_size_kv = block_size_kv
        self.block_idx_q, self.block_idx_kv = None, None
        self.block_start_q, self.block_end_q = None, None
        self.block_start_kv, self.block_end_kv = None, None
        self.get_block_se_q_func = lambda: self._get_block_start_end(
            block_size=self.block_size_q,
            block_idx=self.block_idx_q,
            max_seqlen=self.seqlen_q
        )
        self.get_block_se_kv_func = lambda: self._get_block_start_end(
            block_size=self.block_size_kv,
            block_idx=self.block_idx_kv,
            max_seqlen=self.seqlen_kv
        ) 
    
        # init local o and lse attrs
        self.local_o = None
        self.local_lse = None
        
        # pre-generate the global sliding window attention mask
        self.global_attn_mask = super()._generate_attn_mask( # shape: [1, 1, sq_padded, skv_padded]
            torch.empty(1, 1, self.seqlen_q, 1), # fake q with shape "bhsd"
            torch.empty(1, 1, self.seqlen_kv, 1), # fake k with shape "bhsd"
        )
        self.global_attn_mask = F.pad( # pad the padding boundary to shape: [1, 1, sq_padded, skv_padded]
            self.global_attn_mask,
            pad=(0, self.seqlen_kv_padded - self.seqlen_kv, 0, self.seqlen_q_padded - self.seqlen_q),
            mode="constant",
            value=float("-inf")
        )
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        global_o: torch.Tensor,
        global_lse: torch.Tensor,
        block_idx_q: int,
        block_idx_kv: int,
    ) -> None:
        """The forward pass of Offline Sliding-Window Attention module
        
        Args:
            q(torch.Tensor): query tensor, with shape: [batch_size, block_size_q, num_q_head, head_dim]
            k(torch.Tensor): key tensor, with shape: [batch_size, block_size_kv, num_kv_head, head_dim]
            v(torch.Tensor): value tensor, with shape: [batch_size, block_size_kv, num_kv_head, head_dim]
            global_o(torch.Tensor): global output tensor to be updated inplace, with shape: [batch_size, seqlen_q, num_q_head, head_dim]
            global_lse(torch.Tensor): global lse tensor to be updated inplace, with shape: [batch_size, num_q_head, seqlen_q]
            block_idx_q(int): the block index of q
            block_idx_kv(int): the block index of kv
        """
        # raise NotImplementedError("Assignment3 - Task2")
        
        self.block_idx_q = block_idx_q
        self.block_idx_kv = block_idx_kv
        self.block_start_q, self.block_end_q, self.block_end_padded_q = self.get_block_se_q_func()
        self.block_start_kv, self.block_end_kv, self.block_end_padded_kv = self.get_block_se_kv_func()
        
        # apply the forward for this block of q, k, v
        # to get the local o with shape [b, bq, hq, hd]
        # and the local lse with shape [b, hq, bq]
        super().forward(q, k, v)
        
        # update global o and lse inplace
        self._update_global_o_lse(
            global_o=global_o[:, self.block_start_q:self.block_end_q, ...], # [b, bq', hq, hd]
            global_lse=(
                global_lse[..., self.block_start_q:self.block_end_q]
                if global_lse is not None else None
            ), # [b, hq, bq']
            local_o=self.local_o[:, :(self.block_end_q - self.block_start_q), ...], # [b, bq', hq, hd]
            local_lse=self.local_lse[..., :(self.block_end_q - self.block_start_q)], # [b, hq, bq']
        )
    
    def _non_varlen_attn_fwd_func(
        self,
        q: torch.Tensor, 
        k: torch.Tensor,
        v: torch.Tensor,
        *args,
    ) -> torch.Tensor:
        """non-varlen attn forward function
        
        Args:
            q(torch.Tensor): query tensor, with shape: [batch_size, seq_len_q, num_head, head_dim]
            k(torch.Tensor): key tensor, with shape: [batch_size, seq_len_kv, num_head, head_dim]
            v(torch.Tensor): value tensor, with shape: [batch_size, seq_len_kv, num_head, head_dim]
            *args: to absorb the cu_seqlens kwargs which are not used in non-varlen attn
            
        Returns:
            torch.Tensor: output tensor o, with shape: [batch_size, seq_len_q, num_head, head_dim]
        """
        o, lse = self._attn_fwd_func(q, k, v)
        
        self.local_o = o
        self.local_lse = lse
        
        return o
    
    def _generate_attn_mask(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> torch.Tensor:
        """Generate attention mask
        
        Args:
            q(torch.Tensor): query tensor, with shape: [batch_size, block_size_q, num_head, head_dim]
            k(torch.Tensor): key tensor, with shape: [batch_size, block_size_kv, num_head, head_dim]
            
        Returns:
            torch.Tensor: attention mask, with shape: [1, 1, block_size_q, block_size_kv]
        """
        
        # get the right global attention mask
        global_attn_mask = self.global_attn_mask.to(dtype=q.dtype, device=q.device)
        
        # extract the local block of the attention mask, with shape: [1, 1, bq, bkv]
        local_attn_mask = global_attn_mask[
            :, :, 
            self.block_start_q:self.block_end_padded_q,
            self.block_start_kv:self.block_end_padded_kv,
        ]
        
        return local_attn_mask
    
    def _update_global_o_lse(
        self,
        global_o: torch.Tensor,
        global_lse: Optional[torch.Tensor],
        local_o: torch.Tensor,
        local_lse: torch.Tensor,
    ) -> None:
        """Update the global o and lse inplace with the local ones

        Args:
            global_o (torch.Tensor): the global o to be updated inplace, with shape: [batch_size, block_size_q_, num_q_head, head_dim]
            global_lse (torch.Tensor, optional): the global lse to be updated inplace, with shape: [batch_size, num_q_head, block_size_q_]
            local_o (torch.Tensor): the local o to update global o, with shape: [batch_size, block_size_q_, num_q_head, head_dim]
            local_lse (torch.Tensor): the local lse to update global lse, with shape: [batch_size, num_q_head, block_size_q_]
        """
        # correct global lse
        new_global_lse = correct_attn_lse(
            lse1=global_lse,
            lse2=local_lse,
        )
        
        # correct and update global output
        global_o.copy_(
            correct_attn_output(
                o1=global_o,
                lse1=global_lse,
                o2=local_o,
                lse2=local_lse,
                lse=new_global_lse,
            )
        )
        
        # update global lse
        global_lse.copy_(new_global_lse)
    
    def _get_block_start_end(
        self,
        block_size: int,
        block_idx: int,
        max_seqlen: int,
    ) -> Tuple[int, int, int]:
        block_start = block_idx * block_size
        block_end_padded = (block_idx + 1) * block_size
        block_end = min(
            block_end_padded,
            max_seqlen,
        )
        
        return block_start, block_end, block_end_padded
    
    def _compute_padded_seqlen(
        self,
        seqlen: int,
        block_size: int,
    ) -> int:
        num_blocks = (seqlen + block_size - 1) // block_size
        seqlen_padded = num_blocks * block_size
        return seqlen_padded
    