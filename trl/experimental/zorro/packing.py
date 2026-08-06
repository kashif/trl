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
# The shared-prefix packing scheme implemented here is derived from ZoRRo Train (Zero Redundancy Rollouts),
# Copyright 2025 Snowflake Inc., released under the Apache License 2.0.

from dataclasses import dataclass

import torch


@dataclass
class ZoRRoLayout:
    """
    Packed layout produced by [`pack_shared_prefix_groups`].

    A group of rollouts sampled from the same prompt is packed as `[shared_prefix, tail_0, tail_1, ..., tail_G]`, so
    the prompt is stored and attended to once instead of `G` times. The layout carries everything needed to (1) build
    the attention mask that keeps each tail causally isolated from its siblings, and (2) map packed positions back to
    the tokens they predict.

    Attributes:
        input_ids (`torch.Tensor`):
            Packed token ids of shape `(num_tokens,)`.
        position_ids (`torch.Tensor`):
            Position ids of shape `(num_tokens,)`. The shared prefix spans `0..prefix_len-1` and every tail of that
            group restarts at `prefix_len`, so rotary embeddings match the unpacked sequence.
        segment_ids (`torch.Tensor`):
            Group index of shape `(num_tokens,)`. Tokens from different groups never attend to each other.
        branch_ids (`torch.Tensor`):
            Branch index within the group, of shape `(num_tokens,)`. Shared-prefix tokens are `-1`; tail tokens carry
            the index of the rollout they belong to. A token attends to a key when the key is shared prefix (`branch_id
            == -1`) or belongs to the same branch.
        gather_idx (`torch.Tensor`):
            Packed positions of shape `(num_targets,)` whose hidden state predicts `target_ids`.
        target_ids (`torch.Tensor`):
            Predicted token ids of shape `(num_targets,)`.
        target_sample_idx (`torch.Tensor`):
            Index of shape `(num_targets,)` into the input sample list, for gathering per-sample fields such as
            advantages.
        target_token_idx (`torch.Tensor`):
            Index of shape `(num_targets,)` into the sample's own token sequence, for gathering per-token fields such
            as old log probabilities.
        sample_positions (`list[list[int]]`):
            Packed position of every token of every sample, in token order. Sharing shows up here as several samples
            listing the same positions for their prefix.
        num_tokens_unpacked (`int`):
            Number of tokens the same batch would occupy without prefix sharing, used to report the dedup ratio.
    """

    input_ids: torch.Tensor
    position_ids: torch.Tensor
    segment_ids: torch.Tensor
    branch_ids: torch.Tensor
    gather_idx: torch.Tensor
    target_ids: torch.Tensor
    target_sample_idx: torch.Tensor
    target_token_idx: torch.Tensor
    sample_positions: list[list[int]]
    num_tokens_unpacked: int

    @property
    def num_tokens(self) -> int:
        """Number of tokens actually packed."""
        return self.input_ids.numel()

    @property
    def dedup_ratio(self) -> float:
        """Fraction of tokens removed by prefix sharing. `0.0` when nothing was shared."""
        if self.num_tokens_unpacked == 0:
            return 0.0
        return 1.0 - self.num_tokens / self.num_tokens_unpacked


def _common_prefix_length(sequences: list[list[int]], limit: int) -> int:
    """Length of the longest common prefix of `sequences`, capped at `limit`."""
    limit = min(limit, min(len(sequence) for sequence in sequences))
    reference = sequences[0]
    for position in range(limit):
        token = reference[position]
        if any(sequence[position] != token for sequence in sequences[1:]):
            return position
    return limit


def _leading_context_length(completion_mask: list[int]) -> int:
    """Number of leading tokens that are context (not trained) in `completion_mask`."""
    for position, trained in enumerate(completion_mask):
        if trained:
            return position
    return len(completion_mask)


def pack_shared_prefix_groups(
    input_ids: list[list[int]],
    completion_mask: list[list[int]],
    group_ids: list[int],
    min_group_size: int = 2,
    min_prefix_length: int = 0,
) -> ZoRRoLayout:
    """
    Pack rollouts into a single sequence, storing each group's shared prompt once.

    Samples are grouped by `group_ids`; within a group, the longest common prefix of the token ids is emitted once and
    followed by each sample's remaining tokens. The prefix is capped so that it only ever covers context tokens (a
    token trained for one sample but not another is never shared) and always leaves at least one token per tail.

    Groups that are too small, or whose common prefix is too short to pay for itself, fall back to plain packing: each
    sample becomes its own group, which is exactly the layout the non-ZoRRo path produces.

    Args:
        input_ids (`list[list[int]]`):
            Token ids of each sample, prompt and completion concatenated.
        completion_mask (`list[list[int]]`):
            Per-token mask of each sample, `1` for tokens to train on and `0` for context.
        group_ids (`list[int]`):
            Group each sample belongs to. Samples sharing a group are expected to share a prompt.
        min_group_size (`int`, *optional*, defaults to `2`):
            Minimum number of samples in a group before prefix sharing is attempted.
        min_prefix_length (`int`, *optional*, defaults to `0`):
            Minimum shared prefix length before prefix sharing is attempted. Sharing a short prefix costs more in
            gather overhead than it saves in attention.

    Returns:
        [`ZoRRoLayout`]: The packed layout.

    Example:
    ```python
    >>> layout = pack_shared_prefix_groups(
    ...     input_ids=[[1, 2, 3], [1, 2, 4]],
    ...     completion_mask=[[0, 0, 1], [0, 0, 1]],
    ...     group_ids=[0, 0],
    ... )
    >>> layout.input_ids.tolist()  # prompt [1, 2] stored once, followed by both completions
    [1, 2, 3, 4]

    >>> layout.branch_ids.tolist()
    [-1, -1, 0, 1]
    ```
    """
    if not (len(input_ids) == len(completion_mask) == len(group_ids)):
        raise ValueError(
            f"`input_ids` ({len(input_ids)}), `completion_mask` ({len(completion_mask)}) and `group_ids` "
            f"({len(group_ids)}) must have the same length."
        )

    # Bucket by group id, keeping first-seen order so the packed layout is deterministic.
    buckets: dict[int, list[int]] = {}
    for sample_idx, group_id in enumerate(group_ids):
        buckets.setdefault(group_id, []).append(sample_idx)

    packed_ids: list[int] = []
    packed_positions: list[int] = []
    packed_segments: list[int] = []
    packed_branches: list[int] = []
    # Packed position of every token of every sample, needed to resolve the gather indices below.
    sample_positions: list[list[int]] = [[] for _ in input_ids]

    segment = 0
    for members in buckets.values():
        prefix_len = 0
        if len(members) >= min_group_size:
            # The prefix must be context for every member and must leave at least one token in each tail.
            limit = min(_leading_context_length(completion_mask[i]) for i in members)
            limit = min(limit, min(len(input_ids[i]) for i in members) - 1)
            prefix_len = _common_prefix_length([input_ids[i] for i in members], limit)
        if prefix_len < min_prefix_length:
            prefix_len = 0

        if prefix_len == 0:
            # Nothing worth sharing: emit each member as its own single-branch group.
            for sample_idx in members:
                start = len(packed_ids)
                tokens = input_ids[sample_idx]
                packed_ids.extend(tokens)
                packed_positions.extend(range(len(tokens)))
                packed_segments.extend([segment] * len(tokens))
                packed_branches.extend([0] * len(tokens))
                sample_positions[sample_idx] = list(range(start, start + len(tokens)))
                segment += 1
            continue

        prefix_start = len(packed_ids)
        packed_ids.extend(input_ids[members[0]][:prefix_len])
        packed_positions.extend(range(prefix_len))
        packed_segments.extend([segment] * prefix_len)
        packed_branches.extend([-1] * prefix_len)

        for branch, sample_idx in enumerate(members):
            tokens = input_ids[sample_idx]
            tail_start = len(packed_ids)
            tail = tokens[prefix_len:]
            packed_ids.extend(tail)
            packed_positions.extend(range(prefix_len, len(tokens)))
            packed_segments.extend([segment] * len(tail))
            packed_branches.extend([branch] * len(tail))
            sample_positions[sample_idx] = list(range(prefix_start, prefix_start + prefix_len)) + list(
                range(tail_start, tail_start + len(tail))
            )
        segment += 1

    # Targets: every trained token is predicted by the hidden state of the token before it. Resolving that position
    # through `sample_positions` is what keeps a shared prefix correct -- its final hidden state predicts the first
    # token of *every* branch, which a global shift of the packed row would get wrong for all but the first branch.
    gather_idx: list[int] = []
    target_ids: list[int] = []
    target_sample_idx: list[int] = []
    target_token_idx: list[int] = []
    for sample_idx, (tokens, mask) in enumerate(zip(input_ids, completion_mask, strict=True)):
        for token_idx in range(1, len(tokens)):
            if mask[token_idx]:
                gather_idx.append(sample_positions[sample_idx][token_idx - 1])
                target_ids.append(tokens[token_idx])
                target_sample_idx.append(sample_idx)
                target_token_idx.append(token_idx)

    return ZoRRoLayout(
        input_ids=torch.tensor(packed_ids, dtype=torch.long),
        position_ids=torch.tensor(packed_positions, dtype=torch.long),
        segment_ids=torch.tensor(packed_segments, dtype=torch.long),
        branch_ids=torch.tensor(packed_branches, dtype=torch.long),
        gather_idx=torch.tensor(gather_idx, dtype=torch.long),
        target_ids=torch.tensor(target_ids, dtype=torch.long),
        target_sample_idx=torch.tensor(target_sample_idx, dtype=torch.long),
        target_token_idx=torch.tensor(target_token_idx, dtype=torch.long),
        sample_positions=sample_positions,
        num_tokens_unpacked=sum(len(tokens) for tokens in input_ids),
    )
