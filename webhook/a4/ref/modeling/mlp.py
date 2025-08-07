from enum import Enum
from typing import Optional
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import ProcessGroup


class MLPActivationType(Enum):
    RELU = "relu"
    GELU = "gelu"
    SILU = "silu"
    SIGMOID = "sigmoid"
    BILINEAR = "bilinear"


class DenseMLPWithLoRA(nn.Module):
    """Dense MLP module with LoRA adapters
    This is a GLU-style dense MLP layer with LoRA adapters.
    """
    
    def __init__(self,
        hidden_size: int,
        ffh_size: int,
        activation_type: MLPActivationType = MLPActivationType.SILU,
        init_base_seed: int = 42,
        lora_rank: int = 0,
        lora_alpha: Optional[float] = None,
        lora_dropout_rate: float = 0.0,
        lora_dropout_seed: int = 42,
        lora_init_base_seed: int = 42,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ):
        """Initialize Dense MLP module with LoRA adapters
        Args:
            hidden_size(int): hidden dimension size
            ffh_size(int): hidden dimension size
            activation_type(MLPActivationType, default = "silu"): activation type
            init_base_seed(int, default = 42): seed for base weight initialization
            lora_rank(int, default = 0): lora rank, if 0, then no lora to apply
            lora_alpha(Optional[float], default = None): lora alpha, if None, then set to lora_rank
            lora_dropout_rate(float, default = 0.0): lora dropout rate
            lora_dropout_seed(int, default = 42): lora dropout seed
            lora_init_base_seed(int, default = 42): seed for lora weight initialization
            dtype(torch.dtype, default = torch.float32): parameter dtype
            device(str, default = "cpu"): parameter device
        """
        super().__init__()
        # raise NotImplementedError("Assignment2 - Task1")
        
        self.hidden_size = hidden_size
        self.ffh_size = ffh_size
        self.activation_type = activation_type
        self.init_base_seed = init_base_seed
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout_rate = lora_dropout_rate
        self.lora_dropout_seed = lora_dropout_seed
        self.lora_init_base_seed = lora_init_base_seed
        self.dtype = dtype
        self.device = device
        
        # init base weight initialization function
        if self.activation_type in (MLPActivationType.SIGMOID, MLPActivationType.BILINEAR):
            # apply xavier initialization from a normal distribution
            self.weight_init_func = nn.init.xavier_normal_
        else:
            # apply kaiming initialization from a normal distribution
            self.weight_init_func = partial(
                nn.init.kaiming_normal_,
                mode="fan_in",
                nonlinearity="relu"
            )
        
        # init weight, including up/down/gate projection and optional lora weight A/B
        self.up_proj = nn.Parameter(
            torch.empty(
                self.hidden_size,
                self.ffh_size,
                dtype=self.dtype,
                device=self.device,
            )
        )
        self.gate_proj = nn.Parameter(
            torch.empty(
                self.hidden_size,
                self.ffh_size,
                dtype=self.dtype,
                device=self.device,
            )
        )
        self.down_proj = nn.Parameter(
            torch.empty(
                self.ffh_size,
                self.hidden_size,
                dtype=self.dtype,
                device=self.device,
            )
        )
        if self.lora_rank > 0:
            # init lora weight initialization function
            if self.activation_type in (MLPActivationType.SIGMOID, MLPActivationType.BILINEAR):
                # apply xavier initialization from a uniform distribution
                self.lora_weight_init_func = nn.init.xavier_uniform_
            else:
                # apply kaiming initialization from a uniform distribution
                self.lora_weight_init_func = partial(
                    nn.init.kaiming_uniform_,
                    mode="fan_in",
                    nonlinearity="relu"
                )
                
            # init lora scaling factor
            if self.lora_alpha is None:
                self.lora_alpha = self.lora_rank
            self.lora_scaling_factor = self.lora_alpha / self.lora_rank
            
            # init lora dropout layer
            self.lora_dropout = nn.Dropout(self.lora_dropout_rate)
            
            # init lora weight A/B
            self.lora_weight_A = nn.Parameter(
                torch.empty(
                    self.hidden_size,
                    self.lora_rank,
                    dtype=self.dtype,
                    device=self.device,
                )
            )
            self.lora_weight_B = nn.Parameter(
                torch.empty(
                    self.lora_rank,
                    self.hidden_size,
                    dtype=self.dtype,
                    device=self.device,
                )
            )
        self.reset_parameters()
        
        # init activation func
        if self.activation_type == MLPActivationType.SIGMOID:
            self.activation_func = F.sigmoid
        elif self.activation_type == MLPActivationType.BILINEAR:
            self.activation_func = lambda x: x
        elif self.activation_type == MLPActivationType.RELU:
            self.activation_func = F.relu
        elif self.activation_type == MLPActivationType.GELU:
            self.activation_func = F.gelu
        elif self.activation_type == MLPActivationType.SILU:
            self.activation_func = F.silu
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """The forward pass of the Dense MLP module with LoRA adapters
        
        Args:
            input(torch.Tensor): input tensor, with shape: [batch_size, seq_len, hidden_size]
            
        Returns:
            output(torch.Tensor): output tensor, with shape: [batch_size, seq_len, hidden_size]
        """
        # raise NotImplementedError("Assignment2 - Task1")
        
        input_dtype, input_device = input.dtype, input.device
        input = input.to(dtype=self.dtype, device=self.device)
        
        # apply up projection, with shape: (b, s, h) -> (b, s, ffh)
        output = input @ self.up_proj
        # apply gate projection, with shape: (b, s, h) -> (b, s, ffh)
        gate = input @ self.gate_proj
        # apply glu
        output = self.activation_func(gate) * output
        # apply down projection, with shape: (b, s, ffh) -> (b, s, h)
        output = output @ self.down_proj
        # apply lora optionally
        if self.lora_rank > 0:
            torch.manual_seed(self.lora_dropout_seed)
            output.add_(
                self.lora_dropout(
                    self.lora_scaling_factor * (input @ self.lora_weight_A @ self.lora_weight_B)
                )
            )
            
        return output.to(dtype=input_dtype, device=input_device)
    
    def reset_parameters(self):
        """Initialize the weights of the Dense MLP module with LoRA adapters
        from a normal distribution (or a uniform distribution for lora weights)
        """
        # raise NotImplementedError("Assignment2 - Task1")
        
        torch.manual_seed(self.init_base_seed + 1); self.weight_init_func(self.up_proj.t())
        torch.manual_seed(self.init_base_seed + 2); self.weight_init_func(self.gate_proj.t())
        torch.manual_seed(self.init_base_seed + 3); self.weight_init_func(self.down_proj.t())
        if self.lora_rank > 0:
            torch.manual_seed(self.lora_init_base_seed + 1); self.lora_weight_init_func(self.lora_weight_A.t())
            torch.manual_seed(self.lora_init_base_seed + 2); self.lora_weight_init_func(self.lora_weight_B.t())

    
class SparseMLPWithLoRA(nn.Module):
    """Sparse MLP module with LoRA adapters
    This is a GLU-style sparse MLP layer with LoRA adapters, \
        where the sparcity is implemented as Mixture of Experts (MoE), \
            and each expert is a dense MLP with LoRA adapters.
    """
    
    def __init__(self,
        hidden_size: int,
        ffh_size: int,
        activation_type: MLPActivationType = MLPActivationType.SILU,
        num_experts: int = 1,
        moe_topk: int = 1,
        rank: int = 0,
        world_size: int = 1,
        process_group: Optional[ProcessGroup] = None,
        init_mean: float = 0.0,
        init_std: float = 1.0,
        init_base_seed: int = 42,
        lora_rank: int = 0,
        lora_alpha: Optional[float] = None,
        lora_dropout_rate: float = 0.0,
        lora_dropout_seed: int = 42,
        lora_init_base_seed: int = 42,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ):
        """Initialize Sparse MLP module with LoRA adapters
        
        Args:
            hidden_size(int): hidden dimension size
            ffh_size(int): hidden dimension size
            activation_type(MLPActivationType, default = MLPActivationType.SILU): activation type
            num_experts(int, default = 1): number of (global) experts, which can deduce expert_size = ffh_size // num_experts
            moe_topk(int, default = 1): topk-routing for MoE to control the sparcity
            rank(int, default = 0): rank
            world_size(int, default = 1): world size
            process_group(Optional[ProcessGroup], default = None): the process group (which will not be used for this simpler module yet)
            init_mean(float, default = 0.0): mean for the initialization
            init_std(float, default = 1.0): std for the initialization
            init_base_seed(int, default = 42): seed for the initialization
            lora_rank(int, default = 0): lora rank
            lora_alpha(Optional[float], default = None): lora alpha
            lora_dropout_rate(float, default = 0.0): lora dropout rate
            lora_dropout_seed(int, default = 42): lora dropout seed
            lora_init_base_seed(int, default = 42): seed for lora weight initialization
            dtype(torch.dtype, default = torch.float32): parameter dtype
            device(str, default = "cpu"): parameter device
        """
        super().__init__()
        # raise NotImplementedError("Assignment2 - Task2")
        
        assert ffh_size % num_experts == 0, "ffh_size must be divisible by num_experts"
        
        self.hidden_size = hidden_size
        self.ffh_size = ffh_size
        self.activation_type = activation_type
        self.num_global_experts = num_experts
        self.moe_topk = moe_topk
        self.rank = rank
        self.world_size = world_size
        self.process_group = process_group
        self.init_mean = init_mean
        self.init_std = init_std
        self.init_base_seed = init_base_seed
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout_rate = lora_dropout_rate
        self.lora_dropout_seed = lora_dropout_seed
        self.lora_init_base_seed = lora_init_base_seed
        self.dtype = dtype
        self.device = device
        
        self.expert_size = self.ffh_size // self.num_global_experts
        self.num_local_experts = self.num_global_experts // self.world_size
        self.start_expert_idx, self.end_expert_idx = self.rank * self.num_local_experts, (self.rank + 1) * self.num_local_experts
        
        # init local experts weights
        self.experts = nn.ModuleList([
            DenseMLPWithLoRA(
                hidden_size=self.hidden_size,
                ffh_size=self.expert_size,
                activation_type=self.activation_type,
                init_base_seed=self.init_base_seed + expert_idx,
                lora_rank=self.lora_rank,
                lora_alpha=self.lora_alpha,
                lora_dropout_rate=self.lora_dropout_rate,
                lora_dropout_seed=self.lora_dropout_seed + expert_idx,
                lora_init_base_seed=self.lora_init_base_seed + expert_idx,
                dtype=self.dtype,
                device=self.device,
            )
            for expert_idx in range(self.start_expert_idx, self.end_expert_idx)
        ])
        
        # init global gating weights
        self.gating = nn.Parameter(
            torch.empty(
                self.hidden_size,
                self.num_global_experts,
                dtype=torch.float32,
                device=self.device,
            )
        )
        self.reset_parameters()
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """The forward pass of the Sparse MLP module with LoRA adapters
        
        Args:
            input(torch.Tensor): input tensor, with shape: [batch_size, seq_len, hidden_size]
            
        Returns:
            output(torch.Tensor): output tensor, with shape: [batch_size, seq_len, hidden_size]
        """
        # raise NotImplementedError("Assignment2 - Task2")
        
        b, s, h = input.shape
        input_dtype, input_device = input.dtype, input.device
        input = input.to(dtype=self.dtype, device=self.device)
        
        # flatten input tensor along batch and sequence dimensions, with shape: (b, s, h) -> (b*s, h)
        input = input.view(-1, h)
        
        # init global output buffer, with shape: (b*s, h)
        output = torch.zeros_like(input)
        
        # get routing logits, with shape: (b*s, nge)
        route_logits = input.to(torch.float32) @ self.gating
        
        # get routing probabilities, with shape: (b*s, nge)
        route_probs = F.softmax(route_logits, dim=-1)
        
        # get routing weights / idxs, both with shape: (b*s, k)
        route_weights, route_idxs = torch.topk(route_probs, k=self.moe_topk, dim=-1)
        route_weights /= route_weights.sum(dim=-1, keepdim=True)
        
        # get routing mask, with shape: (b*s, k) -> (b*s, k, nge) -> (b*s, nge)
        route_mask = F.one_hot(route_idxs, num_classes=self.num_global_experts).sum(dim=-2)
        
        # apply for each local expert and get the weighted output
        for expert_idx in range(self.start_expert_idx, self.end_expert_idx):
            # get local expert mlp layer, with shape: (h, e) + (e, h)
            local_expert = self.experts[expert_idx - self.start_expert_idx]
            
            # get token idxs that are routed to this expert, with shape: (b*s, nge) -> (b*s,) -> (gs, 1)
            expert_route_idxs = route_mask.select(dim=1, index=expert_idx).nonzero()
            if expert_route_idxs.numel() == 0: # no tokens are routed to this expert, so just skip it
                continue
            # expand the idxs for later gather / scatter
            expert_route_idxs_k = expert_route_idxs.expand(-1, self.moe_topk)
            expert_route_idxs_h = expert_route_idxs.expand(-1, h)
            
            # get routing weights with its idxs for this local expert
            expert_route_weights, expert_route_weights_idxs = [ # with shape: (b*s, k) -> (gs, k)
                x.gather(dim=0, index=expert_route_idxs_k)
                for x in [route_weights, route_idxs]
            ]
            expert_route_weights_idxs = ( # with shape: (gs, k) -> (gs,) -> (gs, 1)
                expert_route_weights_idxs == expert_idx
            ).nonzero().select(dim=1, index=1).unsqueeze(-1)
            expert_route_weights = expert_route_weights.gather( # with shape: (gs, k) -> (gs, 1)
                dim=1, index=expert_route_weights_idxs
            )
            
            # get the input tokens that are routed to this expert, with shape: (b*s, h) -> (gs, h)
            expert_input = input.gather(dim=0, index=expert_route_idxs_h)
            
            # apply the local expert mlp on the tokens that are routed to this expert, with shape: (gs, h) -> (gs, h)
            expert_output = expert_route_weights * local_expert(expert_input)
            
            # scatter the local expert output to the global output buffer, with shape: (gs, h) -> (b*s, h)
            output.scatter_add_(dim=0, index=expert_route_idxs_h, src=expert_output.to(output.dtype))
        
        return output.to(dtype=input_dtype, device=input_device).view(b, s, h)
        
    def reset_parameters(self):
        """Initialize the weights of each local expert from its own distribution \
            and the gating layer from a normal distribution
        """
        # raise NotImplementedError("Assignment2 - Task2")
        
        # init local expert weights
        for expert_idx in range(self.num_local_experts):
            self.experts[expert_idx].reset_parameters()
        
        # init global gating weights
        torch.manual_seed(self.init_base_seed)
        nn.init.normal_(self.gating, self.init_mean, self.init_std)