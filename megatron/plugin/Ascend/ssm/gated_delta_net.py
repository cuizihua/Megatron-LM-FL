# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""
Ascend NPU implementation of GatedDeltaNet using TE Plugin system.

This module provides NPU-optimized implementations through Transformer Engine Plugin,
which automatically selects the best kernel (AscendC/Triton/PyTorch) for the platform.
"""

import torch
import torch.nn.functional as F

from megatron.core.utils import deprecate_inference_params, nvtx_range_pop, nvtx_range_push

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


def _l2norm_torch(x, eps=1e-6):
    """PyTorch-native L2 normalization to avoid NaN on NPU backward."""
    norm = torch.norm(x, p=2, dim=-1, keepdim=True).clamp(min=eps)
    return x / norm


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

    # Apply L2 norm using NPU-safe implementation
    if self.use_qk_l2norm:
        query = _l2norm_torch(query.contiguous())
        key = _l2norm_torch(key.contiguous())

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
            use_qk_l2norm=False,  # Already applied externally
        )
    else:
        # Fallback to PyTorch implementation for deterministic mode
        from megatron.core.ssm.gated_delta_net import torch_chunk_gated_delta_rule
        core_attn_out, last_recurrent_state = torch_chunk_gated_delta_rule(
            query, key, value, g=g, beta=beta,
            initial_state=None, output_final_state=False,
            use_qk_l2norm_in_kernel=False,
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
