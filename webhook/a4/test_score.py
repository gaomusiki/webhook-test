import sys
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

from typing import List, Optional, Sequence, Dict, Any
from dataclasses import asdict

import pytest

import torch
import torch.nn.functional as F
from torch.testing import assert_close
torch.use_deterministic_algorithms(True)

from test_utils import (
    ResultCapture, score_results,
    check_if_io_meta_is_match,
    check_if_param_reset_is_fine,
)

# get student repo path from env
student_repo_path = os.getenv("STUDENT_REPO_PATH", None)
if student_repo_path is None:
    print("env variable `STUDENT_REPO_PATH` is not set")
    sys.exit(1)
sys.path.insert(0, os.path.abspath(os.path.join(student_repo_path)))

# import reference module
from ref.modeling import (
    MLPActivationType as MLPActivationTypeRef,
    AttnQKVLayout as AttnQKVLayoutRef,
    AttnQKVPackFormat as AttnQKVPackFormatRef,
    TransformerConfig as TransformerConfigRef,
    TransformerDecoderKVCache as TransformerDecoderKVCacheRef,
    TransformerDecoderLayer as TransformerDecoderLayerRef,
    TransformerDecoderBlock as TransformerDecoderBlockRef,
)

# import student's source module
from src.modeling import (
    MLPActivationType,
    AttnQKVLayout,
    AttnQKVPackFormat,
    TransformerConfig,
    TransformerDecoderKVCache,
    TransformerDecoderLayer,
    TransformerDecoderBlock,
)

# constants for all score test cases
ATOL = 1e-2
RTOL = 5e-2
SEED = 142
TIMEOUT = 20

# mapping from ref mlp_activation_type to student mlp_activation_type
mlp_activation_type_ref_to_student = {
    MLPActivationTypeRef.SIGMOID: MLPActivationType.SIGMOID,
    MLPActivationTypeRef.SILU: MLPActivationType.SILU,
    MLPActivationTypeRef.GELU: MLPActivationType.GELU,
    MLPActivationTypeRef.BILINEAR: MLPActivationType.BILINEAR,
    MLPActivationTypeRef.RELU: MLPActivationType.RELU,
}

# mapping from ref attn_qkv_layout to student attn_qkv_layout
attn_qkv_layout_ref_to_student = {
    AttnQKVLayoutRef.BSHD: AttnQKVLayout.BSHD,
    AttnQKVLayoutRef.SBHD: AttnQKVLayout.SBHD,
    AttnQKVLayoutRef.THD: AttnQKVLayout.THD,
}

# mapping from ref attn_qkv_pack_format to student attn_qkv_pack_format
attn_qkv_pack_format_ref_to_student = {
    AttnQKVPackFormatRef.QKV: AttnQKVPackFormat.QKV,
    AttnQKVPackFormatRef.Q_KV: AttnQKVPackFormat.Q_KV,
    AttnQKVPackFormatRef.Q_K_V: AttnQKVPackFormat.Q_K_V,
}

# configs for each score test case
score_test_cases = {
    "task1": {
        "case1": {
            "score": 10,
            
            "b": 4,
            "nh": 8,
            "hd": 128,
            "qkv_layout": AttnQKVLayoutRef.BSHD,
            "num_layers": 3,
            "ops": [
                {
                    "op": "append",
                    "layer_idx": 2,
                    "s": 4096,
                    "seqlens": None,
                },
                {
                    "op": "set",
                    "layer_idx": 1,
                    "s": 2048,
                    "seqlens": None,
                },
                {
                    "op": "has",
                    "layer_idx": 0,
                },
                {
                    "op": "append",
                    "layer_idx": 0,
                    "s": 1024,
                    "seqlens": None,
                },
                {
                    "op": "has",
                    "layer_idx": 2,
                },
                {
                    "op": "get",
                    "layer_idx": 2,
                },
                {
                    "op": "append",
                    "layer_idx": 1,
                    "s": 1024,
                    "seqlens": None,
                },
                {
                    "op": "has",
                    "layer_idx": 1,
                },
                {
                    "op": "get",
                    "layer_idx": 1,
                },
                {
                    "op": "set",
                    "layer_idx": 0,
                    "s": 4096,
                    "seqlens": None,
                },
                {
                    "op": "has",
                    "layer_idx": 0,
                },
                {
                    "op": "get",
                    "layer_idx": 0,
                },
                {
                    "op": "reset",
                },
                {
                    "op": "has",
                    "layer_idx": 1,
                },
                {
                    "op": "set",
                    "layer_idx": 2,
                    "s": 2048,
                    "seqlens": None,
                },
                {
                    "op": "has",
                    "layer_idx": 2,
                },
                {
                    "op": "get",
                    "layer_idx": 2,
                },
            ],
            "kv_dtype": torch.float32,
            "kv_device": "cuda",
        },
        "case2": {
            "score": 10,
            
            "b": 1,
            "nh": 1,
            "hd": 4,
            "qkv_layout": AttnQKVLayoutRef.THD,
            "num_layers": 5,
            "ops": [
                {
                    "op": "reset",
                },
                {
                    "op": "has",
                    "layer_idx": 4,
                },
                {
                    "op": "append",
                    "layer_idx": 0,
                    "s": 1024,
                    "seqlens": [512, 250, 262],
                },
                {
                    "op": "set",
                    "layer_idx": 2,
                    "s": 3072,
                    "seqlens": [1024, 1024, 1024],
                },
                {
                    "op": "append",
                    "layer_idx": 1,
                    "s": 2048,
                    "seqlens": [1024, 500, 524],
                },
                {
                    "op": "set",
                    "layer_idx": 4,
                    "s": 4096,
                    "seqlens": [1024, 2048, 1024],
                },
                {
                    "op": "append",
                    "layer_idx": 3,
                    "s": 4096,
                    "seqlens": [1024, 3000, 72],
                },
                {
                    "op": "has",
                    "layer_idx": 2,
                },
                {
                    "op": "get",
                    "layer_idx": 2,
                },
                {
                    "op": "has",
                    "layer_idx": 1,
                },
                {
                    "op": "get",
                    "layer_idx": 1,
                },
                {
                    "op": "append",
                    "layer_idx": 0,
                    "s": 3,
                    "seqlens": [1, 1, 1],
                },
                {
                    "op": "has",
                    "layer_idx": 0,
                },
                {
                    "op": "get",
                    "layer_idx": 0,
                },
                {
                    "op": "append",
                    "layer_idx": 3,
                    "s": 7,
                    "seqlens": [3, 2, 2],
                },
                {
                    "op": "has",
                    "layer_idx": 3,
                },
                {
                    "op": "get",
                    "layer_idx": 3,
                },
                {
                    "op": "set",
                    "layer_idx": 2,
                    "s": 2048,
                    "seqlens": [1024, 1000, 24],
                },
                {
                    "op": "has",
                    "layer_idx": 2,
                },
                {
                    "op": "get",
                    "layer_idx": 2,
                },
                {
                    "op": "reset",
                },
                {
                    "op": "append",
                    "layer_idx": 0,
                    "s": 128,
                    "seqlens": [64, 32, 32],
                },
                {
                    "op": "has",
                    "layer_idx": 1,
                },
                {
                    "op": "has",
                    "layer_idx": 2,
                },
                {
                    "op": "has",
                    "layer_idx": 3,
                },
                {
                    "op": "has",
                    "layer_idx": 4,
                },
                {
                    "op": "has",
                    "layer_idx": 0,
                },
                {
                    "op": "get",
                    "layer_idx": 0,
                },
            ],
            "kv_dtype": torch.bfloat16,
            "kv_device": "cuda",
        },
    },
    "task2": {
        "case1": {
            "score": 10,
            
            "b": 4,
            "s": 2048,
            "seqlens": None,
            "past_seqlen_kv": 0,
            "past_seqlens": None,
            "activation_dtype": torch.bfloat16,
            "activation_device": "cuda",
            "config": TransformerConfigRef(
                num_layers=1,
                hidden_size=1024,
                ffh_size=2048,
                max_seq_len=8192,
                param_dtype=torch.bfloat16,
                param_device="cuda",
                init_base_seed=SEED + 1,
                
                vocab_size=10000,
                vocab_init_mean=0.1,
                vocab_init_std=1.1,
                
                rope_base=10000,
                rope_ratio=1,
                rope_dynamic=False,
                
                group_size=128,
                eps=1e-5,
                norm_init_range=(-1.1, 1.1),
                
                proj_init_seed=SEED + 1,
                proj_init_mean=0.1,
                proj_init_std=1.1,
                lm_head_tied=False,
                
                online_attn_block_size=None,
                head_dim=128,
                num_q_head=8,
                num_kv_head=2,
                qkv_pack_format=AttnQKVPackFormatRef.Q_KV,
                qkv_layout=AttnQKVLayoutRef.BSHD,
                window_size=None,
                causal=True,
                softmax_dropout_rate=0.,
                softmax_dropout_seed=SEED + 1,
                softmax_scale=None,
                softmax_cap=None,
                softmax_temp=1.,
                softmax_clip_range=(0., 1.),
                apply_qk_norm=False,
                qk_norm_group_size=None,
                
                activation_type=MLPActivationTypeRef.SILU,
                lora_rank=0,
                lora_alpha=None,
                lora_dropout_rate=0.,
                lora_dropout_seed=SEED + 1,
                lora_init_base_seed=SEED + 1,
                
                num_experts=None,
                moe_topk=1,
                gate_init_mean=0.1,
                gate_init_std=1.1,
            ),
        },
        "case2": {
            "score": 10,
            
            "b": 2,
            "s": 4096,
            "seqlens": None,
            "past_seqlen_kv": 0,
            "past_seqlens": None,
            "activation_dtype": torch.bfloat16,
            "activation_device": "cpu",
            "config": TransformerConfigRef(
                num_layers=1,
                hidden_size=2048,
                ffh_size=2048,
                max_seq_len=8192,
                param_dtype=torch.float32,
                param_device="cuda",
                init_base_seed=SEED + 2,
                
                vocab_size=10000,
                vocab_init_mean=0.2,
                vocab_init_std=1.2,
                
                rope_base=20000,
                rope_ratio=1,
                rope_dynamic=False,
                
                group_size=None,
                eps=1e-5,
                norm_init_range=(-1.2, 1.2),
                
                proj_init_seed=SEED + 2,
                proj_init_mean=0.2,
                proj_init_std=1.2,
                lm_head_tied=False,
                
                online_attn_block_size=None,
                head_dim=128,
                num_q_head=16,
                num_kv_head=1,
                qkv_pack_format=AttnQKVPackFormatRef.QKV,
                qkv_layout=AttnQKVLayoutRef.SBHD,
                window_size=None,
                causal=True,
                softmax_dropout_rate=0.,
                softmax_dropout_seed=SEED + 2,
                softmax_scale=None,
                softmax_cap=None,
                softmax_temp=0.8,
                softmax_clip_range=(0., 1.),
                apply_qk_norm=True,
                qk_norm_group_size=None,
                
                activation_type=MLPActivationTypeRef.RELU,
                lora_rank=0,
                lora_alpha=None,
                lora_dropout_rate=0.,
                lora_dropout_seed=SEED + 2,
                lora_init_base_seed=SEED + 2,
                
                num_experts=None,
                moe_topk=1,
                gate_init_mean=0.2,
                gate_init_std=1.2,
            ),
        },
        "case3": {
            "score": 10,
            
            "b": 2,
            "s": 2048,
            "seqlens": None,
            "past_seqlen_kv": 0,
            "past_seqlens": None,
            "activation_dtype": torch.float32,
            "activation_device": "cuda",
            "config": TransformerConfigRef(
                num_layers=1,
                hidden_size=1024,
                ffh_size=2048,
                max_seq_len=2048,
                param_dtype=torch.float32,
                param_device="cuda",
                init_base_seed=SEED + 3,
                
                vocab_size=10000,
                vocab_init_mean=0.3,
                vocab_init_std=1.3,
                
                rope_base=30000,
                rope_ratio=1,
                rope_dynamic=False,
                
                group_size=None,
                eps=1e-5,
                norm_init_range=(-1.3, 1.3),
                
                proj_init_seed=SEED + 3,
                proj_init_mean=0.3,
                proj_init_std=1.3,
                lm_head_tied=True,
                
                online_attn_block_size=128,
                head_dim=64,
                num_q_head=8,
                num_kv_head=8,
                qkv_pack_format=AttnQKVPackFormatRef.Q_K_V,
                qkv_layout=AttnQKVLayoutRef.BSHD,
                window_size=512,
                causal=False,
                softmax_dropout_rate=0.,
                softmax_dropout_seed=SEED + 3,
                softmax_scale=None,
                softmax_cap=50,
                softmax_temp=1.0,
                softmax_clip_range=(-0.03, 1.03),
                apply_qk_norm=False,
                qk_norm_group_size=None,
                
                activation_type=MLPActivationTypeRef.GELU,
                lora_rank=0,
                lora_alpha=None,
                lora_dropout_rate=0.,
                lora_dropout_seed=SEED + 3,
                lora_init_base_seed=SEED + 3,
                
                num_experts=4,
                moe_topk=1,
                gate_init_mean=0.3,
                gate_init_std=1.3,
            ),
        },
        "case4": {
            "score": 10,
            
            "b": 4,
            "s": 1,
            "seqlens": None,
            "past_seqlen_kv": 2048,
            "past_seqlens": None,
            "activation_dtype": torch.bfloat16,
            "activation_device": "cuda",
            "config": TransformerConfigRef(
                num_layers=1,
                hidden_size=1024,
                ffh_size=2048,
                max_seq_len=8192,
                param_dtype=torch.float32,
                param_device="cuda",
                init_base_seed=SEED + 4,
                
                vocab_size=10000,
                vocab_init_mean=0.4,
                vocab_init_std=1.4,
                
                rope_base=40000,
                rope_ratio=1,
                rope_dynamic=False,
                
                group_size=None,
                eps=1e-5,
                norm_init_range=(-1.4, 1.4),
                
                proj_init_seed=SEED + 4,
                proj_init_mean=0.4,
                proj_init_std=1.4,
                lm_head_tied=False,
                
                online_attn_block_size=None,
                head_dim=64,
                num_q_head=16,
                num_kv_head=4,
                qkv_pack_format=AttnQKVPackFormatRef.Q_KV,
                qkv_layout=AttnQKVLayoutRef.SBHD,
                window_size=None,
                causal=True,
                softmax_dropout_rate=0.,
                softmax_dropout_seed=SEED + 4,
                softmax_scale=None,
                softmax_cap=None,
                softmax_temp=1.0,
                softmax_clip_range=(-0., 1.),
                apply_qk_norm=True,
                qk_norm_group_size=None,
                
                activation_type=MLPActivationTypeRef.BILINEAR,
                lora_rank=0,
                lora_alpha=None,
                lora_dropout_rate=0.,
                lora_dropout_seed=SEED + 4,
                lora_init_base_seed=SEED + 4,
                
                num_experts=None,
                moe_topk=1,
                gate_init_mean=0.4,
                gate_init_std=1.4,
            ),
        },
        "case5": {
            "score": 10,
            
            "b": 1,
            "s": 4,
            "seqlens": [1, 1, 1, 1],
            "past_seqlen_kv": 3072,
            "past_seqlens": [1000, 512, 1024, 536],
            "activation_dtype": torch.bfloat16,
            "activation_device": "cuda",
            "config": TransformerConfigRef(
                num_layers=1,
                hidden_size=1024,
                ffh_size=2048,
                max_seq_len=2048,
                param_dtype=torch.float32,
                param_device="cuda",
                init_base_seed=SEED + 5,
                
                vocab_size=10000,
                vocab_init_mean=0.5,
                vocab_init_std=1.5,
                
                rope_base=50000,
                rope_ratio=1,
                rope_dynamic=False,
                
                group_size=None,
                eps=1e-5,
                norm_init_range=(-1.5, 1.5),
                
                proj_init_seed=SEED + 5,
                proj_init_mean=0.5,
                proj_init_std=1.5,
                lm_head_tied=True,
                
                online_attn_block_size=None,
                head_dim=128,
                num_q_head=8,
                num_kv_head=1,
                qkv_pack_format=AttnQKVPackFormatRef.Q_K_V,
                qkv_layout=AttnQKVLayoutRef.THD,
                window_size=None,
                causal=True,
                softmax_dropout_rate=0.,
                softmax_dropout_seed=SEED + 5,
                softmax_scale=None,
                softmax_cap=None,
                softmax_temp=1.,
                softmax_clip_range=(0., 1.),
                apply_qk_norm=False,
                qk_norm_group_size=None,
                
                activation_type=MLPActivationTypeRef.SILU,
                lora_rank=0,
                lora_alpha=None,
                lora_dropout_rate=0.,
                lora_dropout_seed=SEED + 5,
                lora_init_base_seed=SEED + 5,
                
                num_experts=None,
                moe_topk=1,
                gate_init_mean=0.5,
                gate_init_std=1.5,
            ),
        },
    },
    "task3": {
        "case1": {
            "score": 10,
            
            "b": 4,
            "s": 2048,
            "seqlens": None,
            "past_seqlen_kv": 0,
            "past_seqlens": None,
            "activation_dtype": torch.bfloat16,
            "activation_device": "cuda",
            "config": TransformerConfigRef(
                num_layers=4,
                hidden_size=1024,
                ffh_size=2048,
                max_seq_len=8192,
                param_dtype=torch.bfloat16,
                param_device="cuda",
                init_base_seed=SEED + 1,
                
                vocab_size=1000,
                vocab_init_mean=0.1,
                vocab_init_std=1.1,
                
                rope_base=10000,
                rope_ratio=1,
                rope_dynamic=False,
                
                group_size=128,
                eps=1e-5,
                norm_init_range=(-1.1, 1.1),
                
                proj_init_seed=SEED + 1,
                proj_init_mean=0.1,
                proj_init_std=1.1,
                lm_head_tied=True,
                
                online_attn_block_size=None,
                head_dim=128,
                num_q_head=8,
                num_kv_head=1,
                qkv_pack_format=AttnQKVPackFormatRef.QKV,
                qkv_layout=AttnQKVLayoutRef.SBHD,
                window_size=128,
                causal=False,
                softmax_dropout_rate=0.,
                softmax_dropout_seed=SEED + 1,
                softmax_scale=None,
                softmax_cap=None,
                softmax_temp=0.8,
                softmax_clip_range=(0., 1.),
                apply_qk_norm=False,
                qk_norm_group_size=None,
                
                activation_type=MLPActivationTypeRef.RELU,
                lora_rank=0,
                lora_alpha=None,
                lora_dropout_rate=0.,
                lora_dropout_seed=SEED + 1,
                lora_init_base_seed=SEED + 1,
                
                num_experts=2,
                moe_topk=1,
                gate_init_mean=0.1,
                gate_init_std=1.1,
            ),
        },
        "case2": {
            "score": 10,
            
            "b": 2,
            "s": 1024,
            "seqlens": None,
            "past_seqlen_kv": 0,
            "past_seqlens": None,
            "activation_dtype": torch.bfloat16,
            "activation_device": "cpu",
            "config": TransformerConfigRef(
                num_layers=4,
                hidden_size=1024,
                ffh_size=1024,
                max_seq_len=1024,
                param_dtype=torch.float32,
                param_device="cuda",
                init_base_seed=SEED + 2,
                
                vocab_size=1000,
                vocab_init_mean=0.2,
                vocab_init_std=1.2,
                
                rope_base=20000,
                rope_ratio=1,
                rope_dynamic=False,
                
                group_size=512,
                eps=1e-5,
                norm_init_range=(-1.2, 1.2),
                
                proj_init_seed=SEED + 2,
                proj_init_mean=0.2,
                proj_init_std=1.2,
                lm_head_tied=True,
                
                online_attn_block_size=512,
                head_dim=64,
                num_q_head=16,
                num_kv_head=1,
                qkv_pack_format=AttnQKVPackFormatRef.Q_K_V,
                qkv_layout=AttnQKVLayoutRef.BSHD,
                window_size=128,
                causal=True,
                softmax_dropout_rate=0.,
                softmax_dropout_seed=SEED + 2,
                softmax_scale=None,
                softmax_cap=50,
                softmax_temp=1.,
                softmax_clip_range=(0., 1.),
                apply_qk_norm=True,
                qk_norm_group_size=64,
                
                activation_type=MLPActivationTypeRef.GELU,
                lora_rank=0,
                lora_alpha=None,
                lora_dropout_rate=0.,
                lora_dropout_seed=SEED + 2,
                lora_init_base_seed=SEED + 2,
                
                num_experts=4,
                moe_topk=1,
                gate_init_mean=0.2,
                gate_init_std=1.2,
            ),
        },
        "case3": {
            "score": 10,
            
            "b": 1,
            "s": 4,
            "seqlens": [1, 1, 1, 1],
            "past_seqlen_kv": 3072,
            "past_seqlens": [1024, 1036, 500, 512],
            "activation_dtype": torch.float32,
            "activation_device": "cuda",
            "config": TransformerConfigRef(
                num_layers=4,
                hidden_size=1024,
                ffh_size=1024,
                max_seq_len=2048,
                param_dtype=torch.float32,
                param_device="cuda",
                init_base_seed=SEED + 3,
                
                vocab_size=1000,
                vocab_init_mean=0.3,
                vocab_init_std=1.3,
                
                rope_base=30000,
                rope_ratio=1,
                rope_dynamic=False,
                
                group_size=None,
                eps=1e-5,
                norm_init_range=(-1.3, 1.3),
                
                proj_init_seed=SEED + 3,
                proj_init_mean=0.3,
                proj_init_std=1.3,
                lm_head_tied=True,
                
                online_attn_block_size=None,
                head_dim=64,
                num_q_head=8,
                num_kv_head=2,
                qkv_pack_format=AttnQKVPackFormatRef.Q_KV,
                qkv_layout=AttnQKVLayoutRef.THD,
                window_size=None,
                causal=True,
                softmax_dropout_rate=0.,
                softmax_dropout_seed=SEED + 3,
                softmax_scale=None,
                softmax_cap=None,
                softmax_temp=1.,
                softmax_clip_range=(-0.03, 1.03),
                apply_qk_norm=False,
                qk_norm_group_size=None,
                
                activation_type=MLPActivationTypeRef.SILU,
                lora_rank=0,
                lora_alpha=None,
                lora_dropout_rate=0.,
                lora_dropout_seed=SEED + 3,
                lora_init_base_seed=SEED + 3,
                
                num_experts=None,
                moe_topk=1,
                gate_init_mean=0.3,
                gate_init_std=1.3,
            ),
        },
    },
}


def construct_kvcache_args(
    b: int,
    nh: int,
    hd: int,
    qkv_layout: AttnQKVLayoutRef,
    ops: List[Dict[str, Any]],
    dtype: torch.dtype = torch.float32,
    device: str = "cuda",
    seed: int = SEED,
) -> List[Sequence[Optional[torch.Tensor]]]:
    input_tensors = []
    
    for i, op in enumerate(ops):
        if op['op'] in ("set", "append"):
            s, seqlens = op['s'], op['seqlens']
            
            torch.manual_seed(seed + i)
            k = torch.randn(b, s, nh, hd, dtype=dtype, device=device)
            v = torch.randn_like(k)
            cu_seqlens = None
            
            match qkv_layout:
                case AttnQKVLayoutRef.SBHD:
                    k, v = [x.transpose(0, 1) for x in (k, v)]
                case AttnQKVLayoutRef.THD:
                    assert b == 1, "b should be equal to 1 when qkv_layout is THD"
                    assert seqlens is not None, "seqlens must be given when qkv_layout is THD"
                    k, v = [x.squeeze(0) for x in (k, v)]
                    cu_seqlens = torch.concat([
                            torch.zeros(1, dtype=torch.int32, device=device),
                            torch.tensor(seqlens, dtype=torch.int32, device=device).cumsum(dim=0)
                    ], dim=0).to(torch.int32)
                    assert cu_seqlens[-1] == (t:=b*s), f"The sum of seqlens ({cu_seqlens[-1]}) != length ({t})"
            input_tensors.append((k, v, cu_seqlens))
        else:
            input_tensors.append(None)
        
    return input_tensors
           

def construct_decoder_args(
    config: TransformerConfigRef,
    b: int,
    s: int,
    seqlens: Optional[List[int]] = None,
    past_seqlen_kv: int = 0,
    past_seqlens: Optional[List[int]] = None,
    dtype: torch.dtype = torch.float32,
    device: str = "cuda",
) -> Sequence[Optional[torch.Tensor]]:
    torch.manual_seed(config.init_base_seed)
    input = torch.randn(b, s, config.hidden_size, dtype=dtype, device=device)
    input_ids = torch.randint(0, config.vocab_size, (b, s), dtype=torch.int32, device=device)
    
    if seqlens is not None:
        assert config.qkv_layout is AttnQKVLayoutRef.THD, "if using varlen attn, the qkv_layout must be THD"
        assert b == 1, "b should be equal to 1 if using varlen attn"
        
        cu_seqlens = torch.concat([
                torch.zeros(1, dtype=torch.int32, device=device),
                torch.tensor(seqlens, dtype=torch.int32, device=device).cumsum(dim=0)
        ], dim=0).to(torch.int32)
        assert cu_seqlens[-1] == (t:=b*s), f"The sum of seqlens ({cu_seqlens[-1]}) != b*s ({t})"
    else:
        cu_seqlens = None
    
    if past_seqlen_kv > 0:
        if config.qkv_layout is AttnQKVLayoutRef.THD:
            assert past_seqlens is not None, "past_seqlens must be given when qkv_layout is THD and past_seqlen_kv > 0"
        kv_cache_ref = TransformerDecoderKVCacheRef(
            qkv_layout=config.qkv_layout,
            num_layers=config.num_layers,
        )
        kv_cache = TransformerDecoderKVCache(
            qkv_layout=attn_qkv_layout_ref_to_student[config.qkv_layout],
            num_layers=config.num_layers,
        )
        torch.manual_seed(config.init_base_seed)
        past_k = torch.randn(
            b, past_seqlen_kv, config.num_kv_head, config.head_dim, 
            dtype=config.param_dtype, device=config.param_device
        )
        past_v = torch.randn_like(past_k)
        past_cu_seqlens = None
        
        match config.qkv_layout:
            case AttnQKVLayoutRef.SBHD:
                past_k, past_v = [x.transpose(0, 1) for x in (past_k, past_v)]
            case AttnQKVLayoutRef.THD:
                past_k, past_v = [x.squeeze(0) for x in (past_k, past_v)]
                past_cu_seqlens = torch.concat([
                        torch.zeros(1, dtype=torch.int32, device=device),
                        torch.tensor(past_seqlens, dtype=torch.int32, device=device).cumsum(dim=0)
                ], dim=0).to(torch.int32)
                assert past_cu_seqlens[-1] == (t:=len(past_k)), f"The sum of past seqlens ({cu_seqlens[-1]}) != past length ({t})"
        
        kv_cache_ref.set(0, past_k.clone(), past_v.clone(), cu_seqlens=safe_clone(past_cu_seqlens))
        kv_cache.set(0, past_k, past_v, cu_seqlens=past_cu_seqlens)
    else:
        kv_cache_ref, kv_cache = None, None
    
    return input, input_ids, cu_seqlens, kv_cache_ref, kv_cache


def safe_clone(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if x is None:
        return None
    else:
        return x.clone()


@pytest.mark.parametrize(
    "case_key, case_config",
    score_test_cases["task1"].items(),
)
def test_task1(case_key, case_config):
    # set hyper parameters
    b, nh, hd = case_config["b"], case_config["nh"], case_config["hd"]
    qkv_layout, num_layers = case_config["qkv_layout"], case_config["num_layers"]
    ops = case_config["ops"]
    atol, rtol = case_config.pop("atol", ATOL), case_config.pop("rtol", RTOL)
    kv_dtype, kv_device = case_config["kv_dtype"], case_config["kv_device"]
    seed = case_config.pop("seed", SEED)
    
    # construct the input tensors
    input_tensors = construct_kvcache_args(
        b, nh, hd, qkv_layout, 
        ops, 
        kv_dtype, kv_device, 
        seed,
    )
    
    # instantiate the reference module
    kv_cache_ref = TransformerDecoderKVCacheRef(
        qkv_layout=qkv_layout,
        num_layers=num_layers,
    )
    
    # instantiate the student's module
    kv_cache = TransformerDecoderKVCache(
        qkv_layout=attn_qkv_layout_ref_to_student[qkv_layout],
        num_layers=num_layers,
    )
    
    # apply each operation to check if the output for each operation is correct
    for i, (op, input_tensor) in enumerate(zip(ops, input_tensors)):
        try:
            match op['op']:
                case "reset":
                    kv_cache_ref.reset()
                    kv_cache.reset()
                case "has":
                    layer_idx = op['layer_idx']
                    assert kv_cache.has(layer_idx) == kv_cache_ref.has(layer_idx)
                case "get":
                    layer_idx = op['layer_idx']
                    k, v, cu_seqlens = kv_cache.get(layer_idx)
                    k_ref, v_ref, cu_seqlens_ref = kv_cache_ref.get(layer_idx)
                    assert_close(k, k_ref, atol=atol, rtol=rtol)
                    assert_close(v, v_ref, atol=atol, rtol=rtol)
                    if cu_seqlens_ref is not None:
                        assert_close(cu_seqlens, cu_seqlens_ref, atol=atol, rtol=rtol)
                    else:
                        assert cu_seqlens is None
                case "set":
                    layer_idx = op['layer_idx']
                    k, v, cu_seqlens = input_tensor
                    kv_cache.set(layer_idx, k, v, cu_seqlens=cu_seqlens)
                    kv_cache_ref.set(layer_idx, k, v, cu_seqlens=cu_seqlens)
                case "append":
                    layer_idx = op['layer_idx']
                    k, v, cu_seqlens = input_tensor
                    kv_cache.append(layer_idx, k, v, cu_seqlens=cu_seqlens)
                    kv_cache_ref.append(layer_idx, k, v, cu_seqlens=cu_seqlens)
                case _:
                    raise ValueError(f"Unknown operation: {op['op']}")
        except Exception as e:
            assert False, f"The {i+1}-th operation `{op['op']}` failed due to the error: {e}"

        
@pytest.mark.timeout(TIMEOUT)
@pytest.mark.parametrize(
    "case_key, case_config",
    score_test_cases["task2"].items(),
)
def test_task2(case_key, case_config):
    # set hyper parameters
    b, s, seqlens = case_config["b"], case_config["s"], case_config["seqlens"]
    past_seqlen_kv, past_seqlens = case_config["past_seqlen_kv"], case_config["past_seqlens"]
    config_ref: TransformerConfigRef = case_config["config"]
    activation_dtype, activation_device = case_config["activation_dtype"], case_config["activation_device"]
    atol, rtol = case_config.pop("atol", ATOL), case_config.pop("rtol", RTOL)
    if config_ref.qkv_layout is AttnQKVLayoutRef.THD:
        assert seqlens is not None, "seqlens must be given when qkv_layout is THD"
        if past_seqlen_kv > 0:
            assert past_seqlens is not None, "past_seqlens must be given when qkv_layout is THD and past_seqlen_kv > 0"
    
    # construct the input tensor
    input, _, cu_seqlens, kv_cache_ref, kv_cache = construct_decoder_args(
        config_ref,
        b, s, seqlens,
        past_seqlen_kv, past_seqlens,
        dtype=activation_dtype,
        device=activation_device,
    )
    
    # instantiate the reference module
    layer_ref = TransformerDecoderLayerRef(config_ref)
    
    # apply the forward pass to get the reference output tensor
    output_ref = layer_ref(input.clone(), cu_seqlens=safe_clone(cu_seqlens), kv_cache=kv_cache_ref)
    
    # instantiate the student's module
    config_dict = asdict(config_ref)
    config_dict["qkv_layout"] = attn_qkv_layout_ref_to_student[config_ref.qkv_layout]
    config_dict["qkv_pack_format"] = attn_qkv_pack_format_ref_to_student[config_ref.qkv_pack_format]
    config_dict["activation_type"] = mlp_activation_type_ref_to_student[config_ref.activation_type]
    config: TransformerConfig = TransformerConfig(**config_dict)
    layer = TransformerDecoderLayer(config)
    
    # apply the forward pass to get student's output tensor
    output = layer(input, cu_seqlens=cu_seqlens, kv_cache=kv_cache)
    
    # check if the output tensor is correct
    assert_close(output, output_ref, atol=atol, rtol=rtol)


@pytest.mark.timeout(TIMEOUT)
@pytest.mark.parametrize(
    "case_key, case_config",
    score_test_cases["task3"].items(),
)
def test_task3(case_key, case_config):
    # set hyper parameters
    b, s, seqlens = case_config["b"], case_config["s"], case_config["seqlens"]
    past_seqlen_kv, past_seqlens = case_config["past_seqlen_kv"], case_config["past_seqlens"]
    config_ref: TransformerConfigRef = case_config["config"]
    activation_dtype, activation_device = case_config["activation_dtype"], case_config["activation_device"]
    atol, rtol = case_config.pop("atol", ATOL), case_config.pop("rtol", RTOL)
    if config_ref.qkv_layout is AttnQKVLayoutRef.THD:
        assert seqlens is not None, "seqlens must be given when qkv_layout is THD"
        if past_seqlen_kv > 0:
            assert past_seqlens is not None, "past_seqlens must be given when qkv_layout is THD and past_seqlen_kv > 0"
    
    # construct the input tensor
    _, input_ids, cu_seqlens, _, _ = construct_decoder_args(
        config_ref,
        b, s, seqlens,
        past_seqlen_kv, past_seqlens,
        dtype=activation_dtype,
        device=activation_device,
    )
    
    # instantiate the reference module
    block_ref = TransformerDecoderBlockRef(config_ref)
    
    # apply the forward pass to get the reference output logits tensor
    logits_ref = block_ref(input_ids.clone(), cu_seqlens=safe_clone(cu_seqlens))
    
    # instantiate the student's module
    config_dict = asdict(config_ref)
    config_dict["qkv_layout"] = attn_qkv_layout_ref_to_student[config_ref.qkv_layout]
    config_dict["qkv_pack_format"] = attn_qkv_pack_format_ref_to_student[config_ref.qkv_pack_format]
    config_dict["activation_type"] = mlp_activation_type_ref_to_student[config_ref.activation_type]
    config: TransformerConfig = TransformerConfig(**config_dict)
    block = TransformerDecoderBlock(config)
    
    # apply the forward pass to get student's output logits tensor
    logits = block(input_ids, cu_seqlens=cu_seqlens)
    
    # check if the output logits tensor is correct
    assert_close(logits, logits_ref, atol=atol, rtol=rtol)
 

def main():
    capture = ResultCapture()
    
    pytest.main(
        [
            '--quiet',
            '--tb=no', 
            '--disable-warnings',
            os.path.abspath(__file__) # to filter to only execute this test file
        ], 
        plugins=[capture]
    )
    
    score_results(
        capture=capture,
        score_test_cases=score_test_cases,
        student_repo_path=student_repo_path,
    )


if __name__ == "__main__":
    main()