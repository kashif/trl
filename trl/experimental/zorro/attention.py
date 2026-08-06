# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# The shared-prefix attention pattern implemented here is derived from ZoRRo Train (Zero Redundancy Rollouts),
# Copyright 2025 Snowflake Inc., released under the Apache License 2.0.

from collections.abc import Callable

import torch
from transformers.masking_utils import (
    ALL_MASK_ATTENTION_FUNCTIONS,
    and_masks,
    causal_mask_function,
    flash_attention_mask,
)


def zorro_mask_function(segment_ids: torch.Tensor, branch_ids: torch.Tensor) -> Callable:
    """
    Mask function for a batch packed by [`pack_shared_prefix_groups`].

    A query attends to a key when both belong to the same group *and* the key is either shared prefix or part of the
    query's own branch. Combined with [`~transformers.masking_utils.causal_mask_function`], this reproduces the
    attention pattern each rollout would see if it were laid out on its own: it reads the whole prompt and its own
    tokens, and never sees a sibling rollout.

    Args:
        segment_ids (`torch.Tensor`):
            Group index of each token, of shape `(batch_size, seq_len)`.
        branch_ids (`torch.Tensor`):
            Branch index of each token, of shape `(batch_size, seq_len)`, with `-1` on shared-prefix tokens.

    Returns:
        `Callable`: A mask function following the [`~transformers.masking_utils`] convention.
    """

    def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
        same_group = segment_ids[batch_idx, q_idx] == segment_ids[batch_idx, kv_idx]
        kv_is_shared_prefix = branch_ids[batch_idx, kv_idx] == -1
        same_branch = branch_ids[batch_idx, kv_idx] == branch_ids[batch_idx, q_idx]
        return same_group & (kv_is_shared_prefix | same_branch)

    return inner_mask


def build_zorro_mask(
    segment_ids: torch.Tensor,
    branch_ids: torch.Tensor,
    attn_implementation: str,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor | None:
    """
    Build the attention mask for a packed batch, in the format the given attention implementation expects.

    Dispatch goes through [`~transformers.masking_utils.ALL_MASK_ATTENTION_FUNCTIONS`], so `"sdpa"` gets a 4D boolean
    mask, `"eager"` a 4D float mask and `"flex_attention"` a `BlockMask` that lets flex skip the empty blocks — which
    is where the shared-prefix layout turns into an actual speedup. The resulting mask is passed straight to the model
    as `attention_mask`; already-prepared masks are forwarded untouched by
    [`~transformers.masking_utils.create_causal_mask`], so no model code has to be patched.

    FlashAttention takes no arbitrary mask and is handled by a dedicated attention implementation instead.

    Args:
        segment_ids (`torch.Tensor`):
            Group index of each token, of shape `(batch_size, seq_len)`.
        branch_ids (`torch.Tensor`):
            Branch index of each token, of shape `(batch_size, seq_len)`, with `-1` on shared-prefix tokens.
        attn_implementation (`str`):
            Attention implementation the model runs with, e.g. `"sdpa"`, `"eager"` or `"flex_attention"`.
        dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
            Dtype of the mask, only used by the `"eager"` implementation.

    Returns:
        `torch.Tensor` or `BlockMask` or `None`: The mask to pass to the model as `attention_mask`.
    """
    mask_interface = ALL_MASK_ATTENTION_FUNCTIONS.get(attn_implementation)
    # FlashAttention (whether requested directly or resolved from a Hub kernel, in which case it is not registered
    # here at all) builds no mask. Returning `None` would silently drop the isolation between sibling rollouts, so
    # refuse loudly instead.
    if mask_interface is None or mask_interface is flash_attention_mask:
        raise ValueError(
            f"ZoRRo has no mask builder for `attn_implementation='{attn_implementation}'`: FlashAttention takes no "
            "attention mask, so the shared-prefix pattern has to be expressed with a dedicated attention "
            "implementation instead."
        )

    batch_size, seq_len = segment_ids.shape
    mask_function = and_masks(causal_mask_function, zorro_mask_function(segment_ids, branch_ids))
    return mask_interface(
        batch_size=batch_size,
        q_length=seq_len,
        kv_length=seq_len,
        mask_function=mask_function,
        attention_mask=None,
        # Without this, sdpa is free to drop the mask and fall back to `is_causal=True`, which would let sibling
        # rollouts attend to each other.
        allow_is_causal_skip=False,
        dtype=dtype,
        device=segment_ids.device,
    )
