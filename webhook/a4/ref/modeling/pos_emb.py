from typing import Tuple

import numpy as np

import torch
import torch.nn as nn

from ..functional import apply_rotary_pos_emb


class NTKAwareRoPE(nn.Module):
    """NTK-aware RoPE module
    This is a series variants of the RoPE modules based on NTK theory to enhance its extrapolation ability.
    """
    
    def __init__(
        self, 
        dim: int, 
        max_seq_len: int,
        base: int = 10000,
        ratio: int = 1,
        dynamic: bool = False,
        dtype: torch.dtype = torch.float32,
        device: str = 'cpu',
    ) -> None:
        """Initialize NTK-aware RoPE Module
        
        Args:
            dim (int): The dimension of the RoPE
            max_seq_len (int): The maximum sequence length used in training
            base (int, optional): The base of the NTK. Defaults to 10000.
            ratio (int, optional): The ratio of the NTK. Defaults to 1.
            dynamic (bool, optional): Whether to use dynamic mode. Defaults to False.
            dtype (torch.dtype, optional): The dtype of the RoPE. Defaults to torch.float32.
            device (str, optional): The device of the RoPE. Defaults to 'cpu'.
        """
        super().__init__()
        # raise NotImplementedError("TODO: Assignment1 - Task3")
        
        # build ntk inv freq
        inv_freq = NTKAwareRoPE._build_inv_freq(
            dim=dim, 
            base=base,
            k=ratio,
            dtype=dtype,
            device=device,
        )
        # register inv freq
        self._register_inv_freq(inv_freq=inv_freq)
        
        # build cos/sin cached
        cos_cached, sin_cached = NTKAwareRoPE._build_cos_sin_cached(
            seq_len=max_seq_len * ratio,
            inv_freq=inv_freq,
            dtype=dtype, 
            device=device,
        )
        # register cos/sin cache
        self._register_cos_sin_cache(
            cos_cached=cos_cached, 
            sin_cached=sin_cached
        )
        
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.ratio = ratio
        self.dynamic = dynamic
        self.dtype = dtype
        self.device = device
        
        self.extended_seq_len = self.max_seq_len * self.ratio
        
    def forward(self, input: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """The forward pass of the NTK-aware RoPE module
        
        Args:
            input(torch.Tensor): input tensor, with shape: [batch_size, seq_len, num_heads, head_dim]
            offset(int, optional): The offset of the starting position index of the input tensor. Defaults to 0.
        
        Returns:
            output(torch.Tensor): embedded output tensor, with shape: [batch_size, seq_len, num_heads, head_dim]
        """
        # raise NotImplementedError("TODO: Assignment1 - Task3")
        
        # get the cached cos, sin
        cos, sin = self._get_cos_sin_cached(x=input, offset=offset)
        
        # apply rotary pos emb
        output = apply_rotary_pos_emb(
            input=input.to(dtype=self.dtype, device=self.device),
            cos=cos,
            sin=sin,
        ).to(dtype=input.dtype, device=input.device)
        
        return output
        
    def _get_cos_sin_cached(
        self,
        x: torch.Tensor,
        offset: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = x.shape[1] # shape: (b, s, nh, hd)
        start_idx, end_idx = offset, offset + seq_len

        if end_idx > self.extended_seq_len:
            # give the too long seq a temperary embedding
            # and if in dynamic mode, register the new embedding cache and ntk ratio
            cos_cached, sin_cached = self._get_new_cos_sin_cached(end_idx)
        else:
            cos_cached, sin_cached = self.cos_cached, self.sin_cached
    
        return (
            cos_cached[start_idx:end_idx, ...],
            sin_cached[start_idx:end_idx, ...],
        )
        
    def _get_new_cos_sin_cached(
        self,
        new_seq_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Deal with the input of too long seq_len.
        """
        # recompute the new scaling ratio and extended seqlen
        new_ratio = self._recompute_ratio(new_seq_len)
        new_extended_seq_len = self.max_seq_len * new_ratio
        
        # rebuild inv freq
        new_inv_freq = NTKAwareRoPE._build_inv_freq(
            dim=self.dim, 
            base=self.base, 
            k=new_ratio, 
            dtype=self.dtype, 
            device=self.device,
        )
        
        # rebuild cos/sin cache 
        new_cos_cached, new_sin_cached = NTKAwareRoPE._build_cos_sin_cached(
            seq_len=new_extended_seq_len,
            inv_freq=new_inv_freq, 
            dtype=self.dtype, 
            device=self.device,
        )
        
        # if in dynamic mode, dynamically set the new properties
        if self.dynamic:
            self.ratio = new_ratio
            self.extended_seq_len = new_extended_seq_len
  
            # re-register new inv freq
            self._register_inv_freq(inv_freq=new_inv_freq)
            # re-register new cos/sin cache
            self._register_cos_sin_cache(cos_cached=new_cos_cached, sin_cached=new_sin_cached)
        
        return new_cos_cached, new_sin_cached
    
    def _recompute_ratio(self, new_seq_len: int) -> int:
        new_ratio = int(np.ceil(new_seq_len / self.max_seq_len))
        if new_ratio % 2 == 1: # to keep ntk ratio even
            new_ratio += 1
        
        return new_ratio
    
    @staticmethod
    def _build_inv_freq(
        dim: int,
        base: int,
        k: int,
        dtype: torch.dtype = torch.float32,
        device: str = 'cpu', 
    ) -> torch.Tensor:
        new_base = float(base) * k ** (dim / (dim-2)) # base change formula, here type cast to avoid cython type error
        inv_freq = 1.0 / (new_base ** (torch.arange(0, dim, 2).to(device=device, dtype=dtype) / dim))
        
        return inv_freq
    
    def _register_inv_freq(self, inv_freq: torch.Tensor) -> None:
        # self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.inv_freq = inv_freq
    
    @staticmethod
    def _build_cos_sin_cached(
        seq_len: int,
        inv_freq: torch.Tensor, 
        dtype: torch.dtype = torch.float32,
        device: str = 'cpu', 
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=device, dtype=dtype)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos_cached, sin_cached = emb.cos(), emb.sin() # shape: (s, hd)
        
        return cos_cached, sin_cached
    
    def _register_cos_sin_cache(
        self, 
        cos_cached: torch.Tensor, 
        sin_cached: torch.Tensor, 
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.register_buffer("cos_cached", cos_cached.to(dtype=dtype), persistent=False)
        self.register_buffer("sin_cached", sin_cached.to(dtype=dtype), persistent=False)
    
    