# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""
Ascend NPU implementation of GatedDeltaNet using TE Plugin system.

This module provides NPU-optimized implementations through Transformer Engine Plugin,
which automatically selects the best kernel (AscendC/Triton/PyTorch) for the platform.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from megatron.core.utils import deprecate_inference_params, nvtx_range_pop, nvtx_range_push
from megatron.core.tensor_parallel import get_cuda_rng_tracker
from megatron.core.fp8_utils import get_fp8_align_size
from megatron.core.transformer.spec_utils import build_module

try:
    from transformer_engine.plugin.core.manager import OpManager
    HAVE_TE_PLUGIN = True
except ImportError:
    HAVE_TE_PLUGIN = False
    OpManager = None

try:
    from causal_conv1d import causal_conv1d_fn
except ImportError:
    causal_conv1d_fn = None


# L2 normalization is now handled by TE-FL kernel internally
# No need for manual preprocessing here


def gated_delta_net_init(
    self,
    config,
    submodules,
    layer_number=None,
    bias=False,
    conv_bias=False,
    conv_init=None,
    use_qk_l2norm=True,
    A_init_range=(1, 16),
    pg_collection=None,
):
    """
    NPU-optimized __init__ for GatedDeltaNet that bypasses FLA requirement check.

    Instead of checking for FLA in __init__, we rely on TE Plugin system to provide
    the gated_delta_net_forward op at runtime, which will use NPU kernels.
    """
    # Call parent MegatronModule.__init__
    from megatron.core.transformer.module import MegatronModule
    MegatronModule.__init__(self, config)

    # Attributes from arguments
    self.layer_number = layer_number
    self.bias = bias
    self.conv_bias = conv_bias
    self.conv_init = conv_init
    assert A_init_range[0] >= 0 and A_init_range[1] >= A_init_range[0]
    self.A_init_range = A_init_range
    self.use_qk_l2norm = use_qk_l2norm
    assert pg_collection is not None, "pg_collection must be provided for GatedDeltaNet"
    self.pg_collection = pg_collection
    self.tp_size = self.pg_collection.tp.size()
    self.sp_size = self.tp_size if config.sequence_parallel else 1

    # Attributes from config
    self.config = config
    self.hidden_size = config.hidden_size
    self.act_fn = config.activation_func
    self.activation = self.act_fn.__name__
    self.conv_kernel_dim = config.linear_conv_kernel_dim
    self.key_head_dim = config.linear_key_head_dim
    self.value_head_dim = config.linear_value_head_dim
    self.num_key_heads = config.linear_num_key_heads
    self.num_value_heads = config.linear_num_value_heads
    self.qk_dim = self.key_head_dim * self.num_key_heads
    self.v_dim = self.value_head_dim * self.num_value_heads

    # Input projection (hidden_states -> q, k, v, gate, beta, alpha)
    self.in_proj_dim = self.qk_dim * 2 + self.v_dim * 2 + self.num_value_heads * 2
    if self.config.fp8:
        fp8_align_size = get_fp8_align_size(self.config.fp8_recipe)
        assert self.in_proj_dim % fp8_align_size == 0, (
            "For FP8, the innermost dimension of the GDN layer "
            "input projection output tensor must be a multiple of 16."
        )
    self.in_proj = build_module(
        submodules.in_proj,
        self.hidden_size,
        self.in_proj_dim,
        config=self.config,
        init_method=self.config.init_method,
        gather_output=False,
        bias=bias,
        skip_bias_add=False,
        is_expert=False,
        tp_comm_buffer_name="fc1",
        tp_group=self.pg_collection.tp,
    )

    # Conv1d for QKV
    self.conv_dim = self.qk_dim * 2 + self.v_dim
    self.conv_dim_local_tp = self.conv_dim // self.tp_size

    # weight shape: [conv_dim, 1, d_conv]
    # bias shape: [conv_dim]
    self.conv1d = nn.Conv1d(
        in_channels=self.conv_dim_local_tp,
        out_channels=self.conv_dim_local_tp,
        bias=conv_bias,
        kernel_size=self.conv_kernel_dim,
        groups=self.conv_dim_local_tp,
        padding=self.conv_kernel_dim - 1,
        device=torch.cuda.current_device(),
        dtype=config.params_dtype,
    )
    setattr(self.conv1d.weight, "tensor_model_parallel", True)
    setattr(self.conv1d.weight, "partition_dim", 0)
    if conv_bias:
        setattr(self.conv1d.bias, "tensor_model_parallel", True)
        setattr(self.conv1d.bias, "partition_dim", 0)

    # Time step projection (discretization)
    self.num_v_heads_local_tp = self.num_value_heads // self.tp_size
    # dt_bias parameter
    self.dt_bias = nn.Parameter(
        torch.empty(
            self.num_v_heads_local_tp,
            dtype=config.params_dtype,
            device=torch.cuda.current_device(),
        )
    )
    setattr(self.dt_bias, "tensor_model_parallel", True)
    setattr(self.dt_bias, "partition_dim", 0)
    # A_log parameter
    self.A_log = nn.Parameter(
        torch.empty(
            self.num_v_heads_local_tp,
            dtype=config.params_dtype,
            device=torch.cuda.current_device(),
        )
    )
    setattr(self.A_log, "tensor_model_parallel", True)
    setattr(self.A_log, "partition_dim", 0)

    # Output layernorm before projection
    self.out_norm = build_module(
        submodules.out_norm,
        config=self.config,
        hidden_size=self.value_head_dim,
        eps=self.config.layernorm_epsilon,
    )

    self.out_proj = build_module(
        submodules.out_proj,
        self.v_dim,
        self.hidden_size,
        config=self.config,
        init_method=self.config.output_layer_init_method,
        bias=bias,
        input_is_parallel=True,
        skip_bias_add=True,
        is_expert=False,
        tp_comm_buffer_name="fc2",
        tp_group=self.pg_collection.tp,
    )

    # Call reset_parameters
    self.reset_parameters()


def reset_parameters(self):
    """Reset the parameters for NPU GatedDeltaNet."""
    if self.config.perform_initialization:
        with get_cuda_rng_tracker().fork():
            # conv1d.weight
            if self.conv_init is not None:
                nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)
            # dt_bias
            torch.ones(
                self.num_v_heads_local_tp,
                out=self.dt_bias.data,
                dtype=self.config.params_dtype,
                device=torch.cuda.current_device(),
            )
            # A_log
            A = torch.empty(
                self.num_v_heads_local_tp,
                dtype=self.config.params_dtype,
                device=torch.cuda.current_device(),
            ).uniform_(*self.A_init_range)
            self.A_log.data.copy_(torch.log(A))


def _apply_gated_norm(self, x, gate):
    """
    Apply gated normalization to the output.

    This method is called by forward() and must be defined as part of the
    NPU override implementation since it's not automatically inherited.
    """
    # Output Norm
    x_dtype = x.dtype
    x = x.reshape(-1, x.shape[-1])
    y = self.out_norm(x)
    # Output gate
    gate = gate.reshape(-1, gate.shape[-1])
    y = y * self.act_fn(gate.float())
    y = y.to(x_dtype)
    return y


def gated_delta_net_forward(
    self,
    hidden_states,
    attention_mask,
    key_value_states=None,
    inference_context=None,
    attention_bias=None,
    packed_seq_params=None,
    sequence_len_offset=None,
    *,
    inference_params=None,
    **kwargs,
):
    """
    Ascend NPU optimized forward pass for GatedDeltaNet.

    This implementation uses TE Plugin to automatically select the best kernel:
    - AscendC kernel for native NPU acceleration
    - Triton kernel fallback
    - PyTorch fallback for deterministic mode
    """
    # Call parent's preprocessing logic
    inference_context = deprecate_inference_params(inference_context, inference_params)

    seq_len, batch, _ = hidden_states.shape
    seq_len = seq_len * self.sp_size

    if inference_context is not None:
        assert (
            inference_context.is_static_batching()
        ), "GDN does not currently support dynamic inference batching."
        assert not self.config.sequence_parallel
        raise NotImplementedError("GDN does not support inference for now.")

    if packed_seq_params is not None:
        raise NotImplementedError("GDN does not support packed sequence for now.")

    # Input projection
    nvtx_range_push(suffix="in_proj")
    qkvzba, _ = self.in_proj(hidden_states)
    nvtx_range_pop(suffix="in_proj")

    # Transpose: s b x --> b s x
    qkvzba = qkvzba.transpose(0, 1)

    # Split into q, k, v, gate, beta, alpha
    qkv, gate, beta, alpha = torch.split(
        qkvzba,
        [
            (self.qk_dim * 2 + self.v_dim) // self.tp_size,
            self.v_dim // self.tp_size,
            self.num_value_heads // self.tp_size,
            self.num_value_heads // self.tp_size,
        ],
        dim=-1,
    )
    gate = gate.reshape(batch, seq_len, -1, self.value_head_dim)
    beta = beta.reshape(batch, seq_len, -1)
    alpha = alpha.reshape(batch, seq_len, -1)

    # Convolution on qkv
    qkv = qkv.transpose(1, 2).contiguous()  # b, s, d -> b, d, s
    nvtx_range_push(suffix="conv1d")

    # Use NPU-compatible convolution
    if (causal_conv1d_fn is None) or self.config.deterministic_mode:
        qkv = self.act_fn(self.conv1d(qkv)[..., :seq_len])
    else:
        assert self.activation in ["silu", "swish"]
        qkv = causal_conv1d_fn(
            x=qkv,
            weight=self.conv1d.weight.squeeze(1),
            bias=self.conv1d.bias,
            activation=self.activation,
        )
    nvtx_range_pop(suffix="conv1d")

    # Split qkv
    qkv = qkv.transpose(1, 2)
    query, key, value = torch.split(
        qkv,
        [self.qk_dim // self.tp_size, self.qk_dim // self.tp_size, self.v_dim // self.tp_size],
        dim=-1,
    )
    query = query.reshape(batch, seq_len, -1, self.key_head_dim)
    key = key.reshape(batch, seq_len, -1, self.key_head_dim)
    value = value.reshape(batch, seq_len, -1, self.value_head_dim)

    # Apply L2 normalization if requested
    # IMPORTANT: We must apply L2 norm here since we're using torch_chunk_gated_delta_rule
    # with use_qk_l2norm_in_kernel=False (the kernel's L2 norm would apply it again)
    if self.use_qk_l2norm:
        # Use PyTorch native L2 norm to avoid Triton kernel issues on NPU
        def l2norm_torch(x, dim=-1, eps=1e-6):
            norm = torch.norm(x, p=2, dim=dim, keepdim=True).clamp(min=eps)
            return x / norm
        query = l2norm_torch(query.contiguous(), dim=-1, eps=1e-6)
        key = l2norm_torch(key.contiguous(), dim=-1, eps=1e-6)

    if self.num_value_heads // self.num_key_heads > 1:
        query = query.repeat_interleave(self.num_value_heads // self.num_key_heads, dim=2)
        key = key.repeat_interleave(self.num_value_heads // self.num_key_heads, dim=2)

    # Make contiguous
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    gate = gate.contiguous()
    beta = beta.contiguous()
    alpha = alpha.contiguous()

    # Calculate g and beta
    nvtx_range_push(suffix="g_and_beta")
    g = -self.A_log.exp() * F.softplus(alpha.float() + self.dt_bias)
    beta = beta.sigmoid()
    nvtx_range_pop(suffix="g_and_beta")

    nvtx_range_push(suffix="gated_delta_rule")

    # Use TE Plugin for NPU-optimized implementation
    # TE Plugin will try to use AscendC kernel first, then fallback to PyTorch if needed
    if HAVE_TE_PLUGIN and not self.config.deterministic_mode:
        op_manager = OpManager()
        core_attn_out, last_recurrent_state = op_manager.call(
            "gated_delta_net_forward",
            query=query,
            key=key,
            value=value,
            g=g,
            beta=beta,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm=False,  # L2 norm already applied above
            chunk_size=64,
        )
    else:
        # Fallback to PyTorch implementation for deterministic mode
        from megatron.core.ssm.gated_delta_net import torch_chunk_gated_delta_rule
        core_attn_out, last_recurrent_state = torch_chunk_gated_delta_rule(
            query, key, value, g=g, beta=beta,
            initial_state=None, output_final_state=False,
            use_qk_l2norm_in_kernel=False,  # L2 norm already applied above
        )

    nvtx_range_pop(suffix="gated_delta_rule")

    # RMSNorm with gating
    nvtx_range_push(suffix="gated_norm")
    norm_out = self._apply_gated_norm(core_attn_out, gate)
    nvtx_range_pop(suffix="gated_norm")

    # Transpose back: b s x --> s b x
    norm_out = norm_out.reshape(batch, seq_len, -1)
    norm_out = norm_out.transpose(0, 1).contiguous()

    # Output projection
    nvtx_range_push(suffix="out_proj")
    out, out_bias = self.out_proj(norm_out)
    nvtx_range_pop(suffix="out_proj")

    return out, out_bias
