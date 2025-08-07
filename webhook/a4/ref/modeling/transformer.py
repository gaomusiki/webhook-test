from typing import Optional, Tuple, Sequence
from dataclasses import dataclass, field
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import ProcessGroup

# from assignment1 implementations
from .vocab_emb import ParallelVocabEmbedding
from .pos_emb import NTKAwareRoPE
from .norm import GroupRMSNorm

# from assignment2 implementations
from .mlp import (
    MLPActivationType,
    DenseMLPWithLoRA,
    SparseMLPWithLoRA,
)

# from assignment3 implementations
from .attention import (
    AttnQKVPackFormat,
    AttnQKVLayout,
    OfflineSlidingWindowAttn,
    OnlineSlidingWindowAttn,
)


@dataclass(frozen=True, repr=False)
class TransformerConfig:
    """Transformer Configurations Dataclass
        NOTE: some parameters are tagged as "required", indicating they MUST be set to some values except `None` during initialization,
        while some parameters are tagged as "fixed", indicating they can NOT be set during initialization and remain their own default values.
    """
    
    # common transformer configurations
    num_layers: int = field(default=None, metadata={"required": True})
    hidden_size: int = field(default=None, metadata={"required": True})
    ffh_size: int = field(default=None, metadata={"required": True})
    max_seq_len: int = field(default=None, metadata={"required": True})
    param_dtype: torch.dtype = torch.float32
    param_device: str = "cpu"
    init_base_seed: int = 42
    
    # fixed distributed configurations
    rank: int = field(default=0, metadata={"fixed": True})
    world_size: int = field(default=1, metadata={"fixed": True})
    process_group: Optional[ProcessGroup] = field(default=None, metadata={"fixed": True})
    
    # vocab embedding configurations
    vocab_size: int = field(default=None, metadata={"required": True})
    vocab_init_mean: float = 0.0
    vocab_init_std: float = 1.0
    
    # positional embedding configurations
    rope_base: int = 10000
    rope_ratio: int = 1
    rope_dynamic: bool = False
    
    # normalization configurations
    group_size: Optional[int] = None
    eps: float = 1e-5
    norm_init_range: tuple = (-1.0, 1.0)
    
    # projection configurations
    proj_init_seed: int = 42
    proj_init_mean: float = 0.0
    proj_init_std: float = 1.0
    lm_head_tied: bool = False
    
    # attention configurations
    online_attn_block_size: Optional[int] = None # NOTE: if None, then use offline mode, otherwise use online mode
    head_dim: int = field(default=None, metadata={"required": True})
    num_q_head: int = field(default=None, metadata={"required": True})
    num_kv_head: int = field(default=None, metadata={"required": True})
    qkv_pack_format: AttnQKVPackFormat = AttnQKVPackFormat.Q_K_V
    qkv_layout: AttnQKVLayout = AttnQKVLayout.BSHD
    window_size: Optional[int] = None
    causal: bool = False
    softmax_dropout_rate: float = 0.0
    softmax_dropout_seed: int = 42
    softmax_scale: Optional[float] = None
    softmax_cap: Optional[float] = None
    softmax_temp: float = 1.0
    softmax_clip_range: Tuple[float, float] = (0., 1.)
    apply_qk_norm: bool = False
    qk_norm_group_size: Optional[int] = None # NOTE: the other configurations of qk norm are the same as the ones of normalization above
    
    # dense mlp configurations
    activation_type: MLPActivationType = MLPActivationType.SILU
    lora_rank: int = 0
    lora_alpha: Optional[float] = None
    lora_dropout_rate: float = 0.0
    lora_dropout_seed: int = 42
    lora_init_base_seed: int = 42
    
    # sparse mlp configurations (optional)
    num_experts: Optional[int] = None # NOTE: if None, then use dense mlp, otherwise use sparse mlp
    moe_topk: int = 1
    gate_init_mean: float = 0.0
    gate_init_std: float = 1.0
    
    def __post_init__(self):
        """Post-initialization method for TransformerConfig, \
            ensuring all required fields are set and no fixed fields are modified.
        """
        missing_fields = []
        modified_fixed_fields = []
        
        for field_name, field_def in self.__dataclass_fields__.items():
            if field_def.metadata.get("required", False) and field_def.metadata.get("fixed", False):
                raise AttributeError(f"Field {field_name} cannot have set both 'required' and 'fixed' metadata to `True` at the same time.")
            if field_def.metadata.get("required", False) and getattr(self, field_name) is None:
                missing_fields.append(field_name)
            if field_def.metadata.get("fixed", False) and getattr(self, field_name) != field_def.default:
                modified_fixed_fields.append(field_name)
        
        if missing_fields or modified_fixed_fields:
            error_msg = "TransformerConfig initialization failed due to: \n"
            if missing_fields:
                error_msg += f"Missing required fields: {', '.join(missing_fields)}\n"
            if modified_fixed_fields:
                error_msg += f"Modified fixed fields: {', '.join(modified_fixed_fields)}\n"
            
            raise ValueError(error_msg)

    def __repr__(self) -> str:
        """Customized __repr__ method for TransformerConfig, \
            displaying all fields with their values in alphabetical order.
        """
        repr_str = f"{'*'*20}   TransformerConfig   {'*'*20}\n"
        title_len = len(repr_str)
        
        field_names = sorted(self.__dataclass_fields__.keys())
        for field_name in field_names:
            repr_str += f"{field_name}: {getattr(self, field_name)}\n"
        
        repr_str += f"{'*' * title_len}\n"
        
        return repr_str


class TransformerDecoderKVCache(nn.Module):
    """Transformer KV cache module
    This is a simple module to manage cached past key-value pairs for each transformer decoder layer \
        tradeoff memory footprint for avoiding redundant computation during inference.
    """
    def __init__(
        self,
        qkv_layout: AttnQKVLayout = AttnQKVLayout.BSHD,
        num_layers: int = 1,
    ):
        """Initialize Transformer KV cache module
        
        Args:
            qkv_layout (AttnQKVLayout, optional): Layout of the q, k, v tensors. Defaults to AttnQKVLayout.BSHD.
            num_layers (int, optional): Number of transformer layers. Defaults to 1.
        """
        # raise NotImplementedError("TODO: Assignment4 - Task1")
        
        self.num_layers = num_layers
        
        # init layout and get seq_dim
        self.qkv_layout = qkv_layout
        if self.qkv_layout == AttnQKVLayout.BSHD:
            self.seq_dim = 1
        elif self.qkv_layout == AttnQKVLayout.SBHD:
            self.seq_dim = 0
        elif self.qkv_layout == AttnQKVLayout.THD:
            self.seq_dim = 0
        else:
            raise ValueError(f"Unsupported qkv_layout: {self.qkv_layout}")

        # init empty kv cache
        self._kv_cache = None
        self.reset()

    def has(self, layer_idx: int) -> bool:
        """Check if cached past key-value pairs exist for a specific layer
        
        Args:
            layer_idx (int): Layer index

        Returns:
            bool: True if cached past key-value pairs exist for the layer, False otherwise
        """
        # raise NotImplementedError("TODO: Assignment4 - Task1")
        
        try:
            self.get(layer_idx)
            return True
        except Exception:
            return False

    def get(
        self, 
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Get cached past key-value pairs with their optional cumulative sequence lengths for a specific layer
        
        Args:
            layer_idx (int): Layer index

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]: (k, v, optional cu_seqlens)
            
        Raises:
            KeyError: If cached past key-value pairs do not exist for the layer
        """
        # raise NotImplementedError("TODO: Assignment4 - Task1")
        
        self._check_layer_idx(layer_idx)
        
        past_k, past_v, cu_seqlens = self._kv_cache[layer_idx]
        
        if past_k is None or past_v is None:
            raise KeyError(f"Cache for layer {layer_idx} does not exist.")
        
        return past_k, past_v, cu_seqlens

    def set(
        self,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
    ) -> None:
        """Set cached past key-value pairs with their optional cumulative sequence lengths for a specific layer
        
        Args:
            layer_idx (int): Layer index
            k (torch.Tensor): Key tensor to set
            v (torch.Tensor): Value tensor to set
            cu_seqlens (Optional[torch.Tensor], optional): Cumulative sequence lengths for the key-value pairs to set. Defaults to None.
            NOTE: The `cu_seqlens` must be provided if the `qkv_layout` is AttnQKVLayout.THD
        """
        # raise NotImplementedError("TODO: Assignment4 - Task1")
        
        self._check_layer_idx(layer_idx)
        self._check_cu_seqlens(cu_seqlens)
            
        self._kv_cache[layer_idx] = (k, v, cu_seqlens)

    def append(
        self,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
    ) -> None:
        """Dynamically append current cached past key-value pairs with their optional cumulative sequence lengths to the existing ones for a specific layer
        
        Args:
            layer_idx (int): Layer index
            k (torch.Tensor): Key tensor to append
            v (torch.Tensor): Value tensor to append
            cu_seqlens (Optional[torch.Tensor], optional): Cumulative sequence lengths for the key-value pairs to append. Defaults to None.
            NOTE: The `cu_seqlens` must be provided if the `qkv_layout` is AttnQKVLayout.THD, \
                and all of the pass-in arguments should be consistent with the existing ones.
        """
        # raise NotImplementedError("TODO: Assignment4 - Task1")
        
        self._check_layer_idx(layer_idx)
        self._check_cu_seqlens(cu_seqlens)
        
        past_k, past_v, past_cu_seqlens = self._kv_cache[layer_idx]
        
        if past_k is None or past_v is None:
            self._kv_cache[layer_idx] = (k, v, cu_seqlens)
        else:
            if self.qkv_layout is AttnQKVLayout.THD:
                if cu_seqlens is None or past_cu_seqlens is None:
                    raise ValueError(f"The cu_seqlens must be provided if the qkv_layout is AttnQKVLayout.THD, \
                        but got cu_seqlens={cu_seqlens} and past_cu_seqlens={past_cu_seqlens}.")
                elif cu_seqlens.shape[0] != past_cu_seqlens.shape[0]:
                    raise ValueError(f"The cu_seqlens must have the same inner batch size as the past cu_seqlens, \
                        but got len(cu_seqlens)={cu_seqlens.shape[0]} and len(past_cu_seqlens)={past_cu_seqlens.shape[0]}.")
                self._kv_cache[layer_idx] = (
                    self._append_varlen_x(past_k, past_cu_seqlens, k, cu_seqlens),
                    self._append_varlen_x(past_v, past_cu_seqlens, v, cu_seqlens),
                    past_cu_seqlens + cu_seqlens,
                )
            else:
                self._kv_cache[layer_idx] = (
                    torch.cat((past_k, k), dim=self.seq_dim),
                    torch.cat((past_v, v), dim=self.seq_dim),
                    past_cu_seqlens,
                )
    
    def reset(self):
        """Clear the cache memory and reset to the initial state
        """
        # raise NotImplementedError("TODO: Assignment4 - Task1")
        
        self._kv_cache = defaultdict(lambda: (None, None, None))
    
    def _append_varlen_x(
        self,
        past_x: torch.Tensor,
        past_cus: torch.Tensor,
        cur_x: torch.Tensor,
        cur_cus: torch.Tensor,
    ) -> torch.Tensor:
        """Append current varlen tensor x to the past one indexed by cumulative sequence lengths
        
        Args:
            past_x (torch.Tensor): past varlen tensor x, with shape: [total_seq_len, ...]
            past_cus (torch.Tensor): past cumulative sequence lengths, with shape: [batch_size + 1, ]
            cur_x (torch.Tensor): current varlen tensor x, with shape: [total_seq_len, ...]
            cur_cus (torch.Tensor): current cumulative sequence lengths, with shape: [batch_size + 1, ]
        
        Returns:
            torch.Tensor: appended varlen tensor x
        """
        batch_size = past_cus.shape[0] - 1
        
        x = torch.cat([
            torch.cat(
                (
                    past_x[past_cus[bi]:past_cus[bi+1]],
                    cur_x[cur_cus[bi]:cur_cus[bi+1]],
                ), dim=self.seq_dim)
            for bi in range(batch_size)
        ], dim=self.seq_dim)
        
        return x

    def _check_layer_idx(
        self,
        layer_idx: int,
    ) -> None:
        assert 0 <= layer_idx < self.num_layers, \
            f"Layer index must be less than {self.num_layers} and greater than or equal to 0, but got {layer_idx}."
        
    def _check_cu_seqlens(
        self,
        cu_seqlens: Optional[torch.Tensor],
    ) -> None:
        assert self.qkv_layout is AttnQKVLayout.THD or cu_seqlens is None, \
            "The cu_seqlens must be None if the qkv_layout is not AttnQKVLayout.THD."


class TransformerDecoderLayer(nn.Module):
    """Transformer Decoder Layer module
    This is a variant of transformer decoder layer, consisting of two sub-layers: \
            one offline / online self-attention layer, along with qkv projection, ntk-aware rope and out projection, \
            and one dense / sparse feed-forward mlp layer, supporting LoRA adaption intrinsically, \
        which are concatenated sequentially with residual connections and group rms normalization.
    """
    
    def __init__(
        self,
        config: TransformerConfig,
        layer_idx: int = 0,
    ):
        """Initialize Transformer Decoder Layer module
        
        Args:
            config (TransformerConfig): transformer configuration
            layer_idx (int): layer index, in the range of [0, num_layers). Defaults to 0.
        """
        super().__init__()
        # raise NotImplementedError("TODO: Assignment4 - Task2")
        
        self.config = config
        self.layer_idx = layer_idx
        assert 0 <= self.layer_idx < self.config.num_layers, \
            f"Layer index must be less than {self.config.num_layers} and greater than or equal to 0, but got {self.layer_idx}."
        
        self.use_online_attn = self.config.online_attn_block_size is not None
        self.use_sparse_mlp = self.config.num_experts is not None
        
        # init seeds
        self.init_base_seed = self.config.init_base_seed + self.layer_idx
        self.proj_init_seed = self.config.proj_init_seed + self.layer_idx
        self.softmax_dropout_seed = self.config.softmax_dropout_seed + self.layer_idx
        self.lora_init_base_seed = self.config.lora_init_base_seed + self.layer_idx
        self.lora_dropout_seed = self.config.lora_dropout_seed + self.layer_idx
        self.attn_pre_norm_init_seed = self.init_base_seed + 1
        self.attn_init_base_seed = self.init_base_seed + 2
        self.mlp_pre_norm_init_seed = self.init_base_seed + 3
        self.mlp_init_base_seed = self.init_base_seed + 4
        
        # init attn pre-norm
        self.attn_pre_norm = GroupRMSNorm(
            hidden_size=self.config.hidden_size,
            group_size=self.config.group_size,
            eps=self.config.eps,
            init_range=self.config.norm_init_range,
            init_seed=self.attn_pre_norm_init_seed,
            dtype=self.config.param_dtype,
            device=self.config.param_device,
        )
        
        # init attn qkv projection
        self.qkv_head_list = [self.config.num_q_head, self.config.num_kv_head, self.config.num_kv_head]
        self.qkv_hz_list = [self.config.head_dim * nh for nh in self.qkv_head_list]
        self.qkv_proj = nn.Parameter(
            torch.empty(
                self.config.hidden_size,
                sum(self.qkv_hz_list), # hidden_size_q + hidden_size_k + hidden_size_v
                dtype=self.config.param_dtype,
                device=self.config.param_device,
            )
        )
        
        # init rope
        self.rope = NTKAwareRoPE(
            dim=self.config.head_dim,
            max_seq_len=self.config.max_seq_len,
            base=self.config.rope_base,
            ratio=self.config.rope_ratio,
            dynamic=self.config.rope_dynamic,
            dtype=self.config.param_dtype,
            device=self.config.param_device,
        )
        
        # init attn layer
        if self.use_online_attn:
            assert self.config.qkv_pack_format is AttnQKVPackFormat.Q_K_V, \
                f"Online sliding window attention only supports QKV packing format Q_K_V, but got {self.config.qkv_pack_format}"
            assert self.config.qkv_layout is AttnQKVLayout.BSHD, \
                f"Online sliding window attention only supports QKV layout BSHD, but got {self.config.qkv_layout}"
                
            self.num_attn_blocks = (self.config.max_seq_len + self.config.online_attn_block_size - 1) // self.config.online_attn_block_size
            self.pad_seq_len = self.num_attn_blocks * self.config.online_attn_block_size - self.config.max_seq_len
            
            self.attn = OnlineSlidingWindowAttn(
                seqlen_q=self.config.max_seq_len, # NOTE: only use online mode in training, thus no single token for q
                seqlen_kv=self.config.max_seq_len,
                block_size_q=self.config.online_attn_block_size,
                block_size_kv=self.config.online_attn_block_size,
                head_dim=self.config.head_dim,
                num_q_head=self.config.num_q_head,
                num_kv_head=self.config.num_kv_head,
                window_size=self.config.window_size,
                causal=self.config.causal,
                softmax_scale=self.config.softmax_scale,
                softmax_cap=self.config.softmax_cap,
                softmax_temp=self.config.softmax_temp,
                apply_qk_norm=self.config.apply_qk_norm,
                group_size=self.config.qk_norm_group_size,
                eps=self.config.eps,
                init_range=self.config.norm_init_range,
                init_seed=self.attn_init_base_seed,
                dtype=self.config.param_dtype,
                device=self.config.param_device,
            )
        else:
            self.attn = OfflineSlidingWindowAttn(
                head_dim=self.config.head_dim,
                num_q_head=self.config.num_q_head,
                num_kv_head=self.config.num_kv_head,
                qkv_pack_format=self.config.qkv_pack_format,
                qkv_layout=self.config.qkv_layout,
                window_size=self.config.window_size,
                causal=self.config.causal,
                softmax_dropout_rate=self.config.softmax_dropout_rate,
                softmax_dropout_seed=self.softmax_dropout_seed,
                softmax_scale=self.config.softmax_scale,
                softmax_cap=self.config.softmax_cap,
                softmax_temp=self.config.softmax_temp,
                softmax_clip_range=self.config.softmax_clip_range,
                apply_qk_norm=self.config.apply_qk_norm,
                group_size=self.config.qk_norm_group_size,
                eps=self.config.eps,
                init_range=self.config.norm_init_range,
                init_seed=self.attn_init_base_seed,
                dtype=self.config.param_dtype,
                device=self.config.param_device,
            )
    
        # init attn out projection
        self.out_proj = nn.Parameter(
            torch.empty(
                self.qkv_hz_list[0], # hidden_size_q
                self.config.hidden_size,
                dtype=self.config.param_dtype,
                device=self.config.param_device,
            )
        )
        
        # init mlp pre-norm
        self.mlp_pre_norm = GroupRMSNorm(
            hidden_size=self.config.hidden_size,
            group_size=self.config.group_size,
            eps=self.config.eps,
            init_range=self.config.norm_init_range,
            init_seed=self.mlp_pre_norm_init_seed,
            dtype=self.config.param_dtype,
            device=self.config.param_device,
        )
    
        # init mlp layer
        if self.use_sparse_mlp:
            self.mlp = SparseMLPWithLoRA(
                hidden_size=self.config.hidden_size,
                ffh_size=self.config.ffh_size,
                activation_type=self.config.activation_type,
                num_experts=self.config.num_experts,
                moe_topk=self.config.moe_topk,
                rank=self.config.rank,
                world_size=self.config.world_size,
                process_group=self.config.process_group,
                init_mean=self.config.gate_init_mean,
                init_std=self.config.gate_init_std,
                init_base_seed=self.mlp_init_base_seed,
                lora_rank=self.config.lora_rank,
                lora_alpha=self.config.lora_alpha,
                lora_dropout_rate=self.config.lora_dropout_rate,
                lora_dropout_seed=self.lora_dropout_seed,
                lora_init_base_seed=self.lora_init_base_seed,
                dtype=self.config.param_dtype,
                device=self.config.param_device,
            )
        else:
            self.mlp = DenseMLPWithLoRA(
                hidden_size=self.config.hidden_size,
                ffh_size=self.config.ffh_size,
                activation_type=self.config.activation_type,
                init_base_seed=self.mlp_init_base_seed,
                lora_rank=self.config.lora_rank,
                lora_alpha=self.config.lora_alpha,
                lora_dropout_rate=self.config.lora_dropout_rate,
                lora_dropout_seed=self.lora_dropout_seed,
                lora_init_base_seed=self.lora_init_base_seed,
                dtype=self.config.param_dtype,
                device=self.config.param_device,
            )

        # reset parameters
        self.reset_parameters()
    
    def forward(
        self,
        input: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
        kv_cache: Optional[TransformerDecoderKVCache] = None,
    ) -> torch.Tensor:
        """The forward pass of Transformer Decoder Layer module
        
        Args:
            input(torch.Tensor): input hidden states tensor, with shape: [batch_size, seq_len, hidden_size]
            cu_seqlens(torch.Tensor, optional): cumulative sequence lengths for input tensor, with shape: [inner_batch_size + 1, ]
            kv_cache(Optional[TransformerDecoderKVCache], default = None): transformer kv cache, to retrieve / update past key and value during inference, \
                if None, then no kv cache (i.e. during training)
            NOTE: if `cu_seqlens` is not None, then the `batch_size` in the shape of `input` is ensured to be `1` to remain the 3-dim shape, \
                while the real `batch_size` is inferred from `cu_seqlens` (i.e. `inner_batch_size`) since the inner sequences are concatenated along the `seqlen` dim.
        Returns:
            torch.Tensor: output hidden states tensor, with the same shape as input
        """
        # raise NotImplementedError("TODO: Assignment4 - Task2")
        
        # preprocess input
        input_dtype, input_device = input.dtype, input.device
        input = input.to(dtype=self.config.param_dtype, device=self.config.param_device)
        
        # --------------------------------------------
        # apply attn layer with residual and pre-norm
        # --------------------------------------------
        output = self._apply_attn(
            self.attn_pre_norm(input), #  # apply attn pre-norm onto attn input
            cu_seqlens,
            kv_cache,
        ) + input # apply attn input residual onto attn output
        
        # --------------------------------------------
        # apply mlp layer with residual and pre-norm
        # --------------------------------------------
        output = self.mlp(
            self.mlp_pre_norm(output), # apply mlp pre-norm onto mlp input (i.e. attn output)
        ) + output # apply mlp input (i.e. attn output) residual onto mlp output

        # postprocess output
        output = output.to(dtype=input_dtype, device=input_device)
        
        return output
    
    def reset_parameters(self):
        """Initialize learnable parameters for Transformer Decoder Layer module"""
        # raise NotImplementedError("TODO: Assignment4 - Task2")
        
        # attn pre-norm
        self.attn_pre_norm.reset_parameters()
        
        # attn qkv projection
        torch.manual_seed(self.proj_init_seed + 1)
        nn.init.normal_(self.qkv_proj, self.config.proj_init_mean, self.config.proj_init_std)
        
        # attn out projection
        torch.manual_seed(self.proj_init_seed + 2)
        nn.init.normal_(self.out_proj, self.config.proj_init_mean, self.config.proj_init_std)
        
        # mlp pre-norm
        self.mlp_pre_norm.reset_parameters()
        
        # mlp
        self.mlp.reset_parameters()
    
    def _apply_attn(
        self,
        input: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
        kv_cache: Optional[TransformerDecoderKVCache] = None,
    ) -> torch.Tensor:
        """The sub forward pass to apply attention
        
        Args:
            input(torch.Tensor): input hidden states tensor, with shape: [batch_size, seq_len, hidden_size]
            cu_seqlens(torch.Tensor, optional): cumulative sequence lengths for input tensor, with shape: [inner_batch_size + 1, ]
            kv_cache(Optional[TransformerDecoderKVCache], default = None): transformer kv cache, to retrieve / update past key and value during inference, \
                if None, then no kv cache (i.e. during training)
        Returns:
            torch.Tensor: output hidden states tensor, with the same shape as input
        """
        
        # apply qkv projection to get packed q,k,v, shape: [b, s, h] -> [b, s, hd*(nhq + 2*nhkv)]
        qkv = input @ self.qkv_proj
        
        # split qkv to q, k, v with shape: [b, s, nhq*hd], [b, s, nhkv*hd], [b, s, nhkv*hd]
        q, k, v = qkv.split(self.qkv_hz_list, dim=-1)
        
        # split q, k, v by heads to shape: [b, s, nhq, hd], [b, s, nhkv, hd], [b, s, nhkv, hd]
        q, k, v = [
            x.view(*x.shape[:-1], nh, self.config.head_dim)
            for x, nh in zip((q, k, v), self.qkv_head_list)
        ]
        
        # apply attn layer to get attn output, with shape: [b, s, nhq, hd]
        if self.use_online_attn:
            o = self._loop_apply_online_attn(q, k, v)
        else:
            o = self._apply_offline_attn(q, k, v, cu_seqlens, kv_cache)
            
        # apply out proj, shape: [b, s, nhq, hd] -> [b, s, nhq*hd] -> [b, s, h]
        o = o.contiguous().view(*o.shape[:-2], -1) @ self.out_proj
        
        return o
    
    def _apply_offline_attn(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
        kv_cache: Optional[TransformerDecoderKVCache] = None,
    ) -> torch.Tensor:
        """The sub forward pass to apply offline attention
        
        Args:
            q(torch.Tensor): query tensor, with shape: [b, s, nhq, hd]
            k(torch.Tensor): key tensor, with shape: [b, s, nhkv, hd]
            v(torch.Tensor): value tensor, with shape: [b, s, nhkv, hd]
            cu_seqlens(torch.Tensor, optional): cumulative sequence lengths for input tensor, with shape: [inner_batch_size + 1, ]
            kv_cache(Optional[TransformerDecoderKVCache], default = None): transformer kv cache, to retrieve / update past key and value during inference, \
                if None, then no kv cache (i.e. during training)
        Returns:
            torch.Tensor: output hidden states tensor, with the same shape as q
        """
        
        # apply rope to q, k with layout "bshd"
        q, k = self._apply_rope(q, k, cu_seqlens, kv_cache)
        
        # transform qkv layout from layout "bshd" to the specified one
        q, k, v = self._trans_qkv_layout(q, k, v)
        
        # update k,v and (cu_seqlens_q, cu_seqlens_k) from past cache if it exists
        cu_seqlens_q = cu_seqlens
        if kv_cache is not None:
            # update current kv to kv cache
            kv_cache.append(self.layer_idx, k, v, cu_seqlens_q)
            # retrieve updated kv from kv cache
            k, v, cu_seqlens_k = kv_cache.get(self.layer_idx)
        else:
            cu_seqlens_k = cu_seqlens
        
        # apply attn layer to get attn output, with shape: [b, s, nhq, hd]
        o = self._trans_o_layout( # transform output layout from the specified one to "bshd"
            self.attn(
            *self._trans_qkv_pack_format(q, k, v), # transform qkv pack format from q_k_v to specified one
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
        ))
        
        return o
            
    def _loop_apply_online_attn(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """The sub forward pass to apply online attention in loop
        
        Args:
            q(torch.Tensor): query tensor, with shape: [b, sq, nhq, hd]
            k(torch.Tensor): key tensor, with shape: [b, sk, nhkv, hd]
            v(torch.Tensor): value tensor, with shape: [b, sv, nhkv, hd]
        Returns:
            torch.Tensor: output hidden states tensor, with the same shape as q
        """
        # init global output and lse buffer
        global_o = torch.zeros_like(q)
        global_lse = torch.zeros( # shape: (b, nhq, sq)
            (q.shape[0], q.shape[-2], q.shape[1]),
            dtype=torch.float32, 
            device=q.device
        ).fill_(float("-inf"))
        
        # pad q,k,v to multiple of block size
        q, k, v = [
            F.pad(
                x, pad=(0, 0, 0, 0, 0, self.pad_seq_len),
                mode="constant", value=0,
            ) for x in (q, k, v)
        ]
        
        # loop over each (bqi, bkvj) block pair
        bz = self.config.online_attn_block_size # alias for block size
        for bqi in range(self.num_attn_blocks):
            for bkvj in range(self.num_attn_blocks):
                q_ = q[:, bqi*bz:(bqi+1)*bz, :, :]
                k_ = k[:, bkvj*bz:(bkvj+1)*bz, :, :]
                v_ = v[:, bkvj*bz:(bkvj+1)*bz, :, :]
                self.attn(
                    q=q_,
                    k=k_,
                    v=v_,
                    global_o=global_o,
                    global_lse=global_lse,
                    block_idx_q=bqi,
                    block_idx_kv=bkvj,
                )
                
        return global_o
    
    def _apply_rope(
        self, 
        q: torch.Tensor,
        k: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
        kv_cache: Optional[TransformerDecoderKVCache] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rope to q, k
        
        Args:
            q(torch.Tensor): query tensor, with shape: [b, sq, nhq, hd]
            k(torch.Tensor): key tensor, with shape: [b, sk, nhkv, hd]
            cu_seqlens(Optional[torch.Tensor], default = None): current sequence length tensor, with shape: [batch_size + 1, ]
            kv_cache(Optional[TransformerDecoderKVCache], default = None): transformer kv cache, to retrieve / update past key and value during inference, \
                if None, then no kv cache (i.e. during training)
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: rope-embedded q, k
        """
        
        # get starting pos idx offset(s)
        if cu_seqlens is not None:
            if kv_cache is not None and kv_cache.has(self.layer_idx):
                past_k, _, cu_seqlens_k = kv_cache.get(self.layer_idx)
                assert cu_seqlens_k is not None, "cu_seqlens_k is not found in kv_cache"
                offsets = torch.diff(cu_seqlens_k)
            else:
                offsets = [0] * (cu_seqlens.shape[0] - 1)
        else:
            if kv_cache is not None and kv_cache.has(self.layer_idx):
                past_k, _, cu_seqlens_k = kv_cache.get(self.layer_idx)
                offset = past_k.shape[kv_cache.seq_dim]
            else:
                offset = 0
                
        # apply rope to q, k for each (inner) sequence
        q, k = [
            self.rope(x, offset) if cu_seqlens is None else (
                torch.cat([
                    self.rope(
                        x[:, cu_seqlens[bi]:cu_seqlens[bi+1], ...],
                        offsets[bi],
                    ) for bi in range(cu_seqlens.shape[0] - 1)
                ], dim=1)
            )
            for x in (q, k)
        ]
        
        return q, k
    
    def _trans_qkv_layout(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> Sequence[torch.Tensor]:
        if self.config.qkv_layout == AttnQKVLayout.BSHD:  # "bshd" -> "bshd"
            self.seq_dim = 1  # lazy init self.seq_dim
        elif self.config.qkv_layout == AttnQKVLayout.SBHD:  # "bshd" -> "sbhd"
            q, k, v = [x.transpose(0, 1) for x in (q, k, v)]
            self.seq_dim = 0
        elif self.config.qkv_layout == AttnQKVLayout.THD:  # "bshd" -> "thd"
            q, k, v = [x.squeeze(0) for x in (q, k, v)]
            self.seq_dim = 0
        else:
            raise ValueError(f"Unsupported qkv_layout: {self.config.qkv_layout}")
                
        return q, k, v
                
    def _trans_qkv_pack_format(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> Sequence[Optional[torch.Tensor]]:
        if self.config.qkv_pack_format == AttnQKVPackFormat.Q_K_V:
            pass
        elif self.config.qkv_pack_format == AttnQKVPackFormat.Q_KV:
            k, v = torch.cat([k, v], dim=-2), None
        elif self.config.qkv_pack_format == AttnQKVPackFormat.QKV:
            assert q.shape[self.seq_dim] == k.shape[self.seq_dim] == v.shape[self.seq_dim], \
                f"q, k, v must have the same sequence length if qkv pack format is AttnQKVPackFormat.QKV, " \
                f"but got q: {q.shape[self.seq_dim]}, k: {k.shape[self.seq_dim]}, v: {v.shape[self.seq_dim]}"
            
            q = torch.cat([q, k, v], dim=-2)
            k, v = None, None
        else:
            raise ValueError(f"Unsupported qkv_pack_format: {self.config.qkv_pack_format}")
            
        return q, k, v
    
    def _trans_o_layout(
        self,
        o: torch.Tensor,
    ) -> torch.Tensor:
        if self.config.qkv_layout == AttnQKVLayout.BSHD:  # "bshd" -> "bshd"
            pass
        elif self.config.qkv_layout == AttnQKVLayout.SBHD:  # "sbhd" -> "bshd"
            o = o.transpose(0, 1)
        elif self.config.qkv_layout == AttnQKVLayout.THD:  # "thd" -> "bshd"
            o = o.unsqueeze(0)
        else:
            raise ValueError(f"Unsupported qkv_layout: {self.config.qkv_layout}")
        
        return o
    

class TransformerDecoderBlock(nn.Module):
    """Transformer Decoder Block module
    
    This is a standard decoder-only transformer block for language modeling, \
        which mainly consists of a sequence of transformer decoder layers, \
        transforming the hidden states of input token ids initialized from vocab embedding, \
        and finally returning the vocab logits with a lm head projection.
    """
    
    def __init__(
        self,
        config: TransformerConfig,
    ):
        """Initialize Transformer Decoder Block module
        
        Args:
            config (TransformerConfig): transformer configuration
        """
        super().__init__()
        # raise NotImplementedError("TODO: Assignment4 - Task3")

        self.config = config
        
        # init seeds
        self.vocab_init_seed = self.config.init_base_seed
        self.lm_head_init_seed = self.config.proj_init_seed
        
        # init vocab embedding
        self.vocab_emb = ParallelVocabEmbedding(
            vocab_size=self.config.vocab_size,
            emb_size=self.config.hidden_size,
            rank=self.config.rank,
            world_size=self.config.world_size,
            process_group=self.config.process_group,
            init_mean=self.config.vocab_init_mean,
            init_std=self.config.vocab_init_std,
            init_base_seed=self.vocab_init_seed,
            dtype=self.config.param_dtype,
            device=self.config.param_device,
        )
        
        # init transformer decoder layers
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(
                config=self.config,
                layer_idx=layer_idx,
            )
            for layer_idx in range(self.config.num_layers)
        ])
        
        # init transformer kv cache, used only during inference
        self.kv_cache = TransformerDecoderKVCache(
            qkv_layout=self.config.qkv_layout,
            num_layers=self.config.num_layers,
        )
        
        # init final norm
        self.final_norm = GroupRMSNorm(
            hidden_size=self.config.hidden_size,
            group_size=self.config.group_size,
            eps=self.config.eps,
            init_range=self.config.norm_init_range,
            init_seed=self.config.init_base_seed,
            dtype=self.config.param_dtype,
            device=self.config.param_device,
        )
        
        # init lm head
        self.lm_head = nn.Linear(
            in_features=self.config.hidden_size,
            out_features=self.config.vocab_size,
            bias=False,
            dtype=self.config.param_dtype,
            device=self.config.param_device,
        )
        
        # reset parameters
        self.reset_parameters()
    
    def forward(
        self,
        input_ids: torch.LongTensor,
        cu_seqlens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """The forward pass of Transformer Decoder Block module
        
        Args:
            input_ids(torch.LongTensor): the vocab ids for the input, with shape: [batch_size, seq_len]
            cu_seqlens(torch.Tensor, optional): cumulative sequence lengths, with shape: [inner_batch_size + 1, ]
            NOTE: if `cu_seqlens` is not None, then the `batch_size` in the shape of `input_ids` is ensured to be `1` to remain the 2-dim shape, \
                while the real `batch_size` is inferred from `cu_seqlens` (i.e. `inner_batch_size`) since the inner sequences are concatenated along the `seqlen` dim.
        Returns:
            torch.Tensor: output tensor as vocab logits, with shape: [batch_size, seq_len, vocab_size]
        """
        # raise NotImplementedError("TODO: Assignment4 - Task3")
        
        # preprocess input ids
        input_device = input_ids.device
        input_ids = input_ids.to(device=self.config.param_device)
        
        # apply vocab embedding to get initial hidden states, shape: (b, s) -> (b, s, h)
        hidden_states = self.vocab_emb(input_ids)
        
        # apply each transformer decoder layer to hidden states, shape: (b, s, h) -> (b, s, h)
        for layer in self.layers:
            hidden_states = layer(
                input=hidden_states,
                cu_seqlens=cu_seqlens,
                kv_cache=self.kv_cache if not self.training else None,
            )
            
        # apply final norm to hidden states, shape: (b, s, h) -> (b, s, h)
        hidden_states = self.final_norm(hidden_states)
            
        # apply lm head onto final hidden states to get vocab logits, shape: (b, s, h) -> (b, s, v)
        vocab_logits = self.lm_head(hidden_states)
        
        # postprocess vocab logits
        vocab_logits = vocab_logits.to(device=input_device)
        
        return vocab_logits
        
    def get_kv_cache(self) -> TransformerDecoderKVCache:
        """Get the TransformerDecoderKVCache object managing the kv cache memory"""
        # raise NotImplementedError("TODO: Assignment4 - Task3")
        
        return self.kv_cache
    
    def reset_kv_cache(self):
        """Clear the cache memory and reset to the initial state"""
        # raise NotImplementedError("TODO: Assignment4 - Task3")
        
        self.kv_cache.reset()
       
    def reset_parameters(self):
        """Initialize learnable parameters for Transformer Decoder Block module"""
        # raise NotImplementedError("TODO: Assignment4 - Task3")
        
        # vocab embedding
        self.vocab_emb.reset_parameters()
        
        # transformer decoder layers
        for layer in self.layers:
            layer.reset_parameters()
            
        # final norm
        self.final_norm.reset_parameters()
            
        # lm head
        if self.config.lm_head_tied: # share the lm head with vocab embedding
            self.lm_head.weight = self.vocab_emb.weight
        else:
            torch.manual_seed(self.lm_head_init_seed)
            self.lm_head.weight.data.normal_(mean=self.config.proj_init_mean, std=self.config.proj_init_std)
     
    def num_parameters(self, learnable_only: bool = False, unit: str = "1") -> float:
        """Compute the number of (learnable) parameters in the Llama Model module
        
        Args:
            learnable_only(bool, optional): whether to count only learnable parameters or not, default to False
            unit(str, optional): unit of the number of parameters, default to '1' for "1", \
                other options include 'K' for "1 thousand", 'M' for "1 million", 'B' for "1 billion"
        Returns:
            float: the number of (learnable) parameters in the Llama Model module in the specified unit
        """
        # raise NotImplementedError("TODO: Assignment4 - Task3")
        
        params = sum(p.numel() for p in self.parameters() if not learnable_only or p.requires_grad)
        
        if unit == "1":
            pass
        elif unit == "B":
            params /= 1000**3
        elif unit == "M":
            params /= 1000**2
        elif unit == "K":
            params /= 1000
        else:
            raise ValueError(f"Unsupported unit: {unit}")
            
        return params
    
    def num_memory_footprint(self, unit: str = "B") -> float:
        """Compute the theoretical memory footprint of the Llama Model module's parameters
        
        Args:
            unit(str, optional): unit of the memory footprint, default to 'B' for "1 byte", \
                other options include 'KB' for "1 kilobyte", 'MB' for "1 megabyte", 'GB' for "1 gigabyte"
                
        Returns:
            float: the theoretical memory footprint of the Llama Model module's parameters in the specified unit
        """
        # raise NotImplementedError("TODO: Assignment4 - Task3")
        
        mems = sum(p.element_size() * p.numel() for p in self.parameters())
        
        if unit == "B":
            pass
        elif unit == "GB":
            mems /= 1024**3
        elif unit == "MB":
            mems /= 1024**2
        elif unit == "KB":
            mems /= 1024
        else:
            raise ValueError(f"Unsupported unit: {unit}")
            
        return mems