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

import pytest

from trl.experimental.zorro import pack_shared_prefix_groups

from ..testing_utils import TrlTestCase


def reference_targets(input_ids, completion_mask):
    """Targets the unpacked path would train on: `(sample_idx, token_idx, predicted_token, predicting_token)`."""
    targets = []
    for sample_idx, (tokens, mask) in enumerate(zip(input_ids, completion_mask, strict=True)):
        for token_idx in range(1, len(tokens)):
            if mask[token_idx]:
                targets.append((sample_idx, token_idx, tokens[token_idx], tokens[token_idx - 1]))
    return targets


class TestPackSharedPrefixGroups(TrlTestCase):
    def test_shared_prefix_is_packed_once(self):
        layout = pack_shared_prefix_groups(
            input_ids=[[1, 2, 3, 10], [1, 2, 3, 20], [1, 2, 3, 30]],
            completion_mask=[[0, 0, 0, 1]] * 3,
            group_ids=[7, 7, 7],
        )

        assert layout.input_ids.tolist() == [1, 2, 3, 10, 20, 30]
        assert layout.position_ids.tolist() == [0, 1, 2, 3, 3, 3]
        assert layout.segment_ids.tolist() == [0, 0, 0, 0, 0, 0]
        assert layout.branch_ids.tolist() == [-1, -1, -1, 0, 1, 2]
        assert layout.num_tokens == 6
        assert layout.num_tokens_unpacked == 12
        assert layout.dedup_ratio == 0.5

    def test_shared_prefix_final_position_predicts_every_branch(self):
        # The regression this whole scheme hinges on: the last shared-prefix position predicts the first token of
        # *every* branch. A global shift of the packed row would pair it with branch 0 only.
        layout = pack_shared_prefix_groups(
            input_ids=[[1, 2, 10], [1, 2, 20], [1, 2, 30]],
            completion_mask=[[0, 0, 1]] * 3,
            group_ids=[0, 0, 0],
        )

        assert layout.gather_idx.tolist() == [1, 1, 1]  # the shared prefix's last position, three times
        assert layout.target_ids.tolist() == [10, 20, 30]

    def test_targets_match_the_unpacked_path(self):
        input_ids = [[1, 2, 3, 40, 41], [1, 2, 3, 50], [1, 2, 9, 60, 61, 62], [7, 8, 70]]
        completion_mask = [[0, 0, 0, 1, 1], [0, 0, 0, 1], [0, 0, 0, 1, 1, 1], [0, 0, 1]]
        group_ids = [0, 0, 0, 1]

        layout = pack_shared_prefix_groups(input_ids, completion_mask, group_ids)

        expected = reference_targets(input_ids, completion_mask)
        assert layout.target_sample_idx.tolist() == [target[0] for target in expected]
        assert layout.target_token_idx.tolist() == [target[1] for target in expected]
        assert layout.target_ids.tolist() == [target[2] for target in expected]
        # Every gathered position must hold the token that precedes the target in the *unpacked* sample.
        gathered = layout.input_ids[layout.gather_idx].tolist()
        assert gathered == [target[3] for target in expected]

    def test_partial_prefix_overlap_is_shared(self):
        # Prompts diverge at position 2; only the common part is shared.
        layout = pack_shared_prefix_groups(
            input_ids=[[1, 2, 3, 10], [1, 2, 4, 20]],
            completion_mask=[[0, 0, 0, 1], [0, 0, 0, 1]],
            group_ids=[0, 0],
        )

        assert layout.input_ids.tolist() == [1, 2, 3, 10, 4, 20]
        assert layout.branch_ids.tolist() == [-1, -1, 0, 0, 1, 1]
        assert layout.position_ids.tolist() == [0, 1, 2, 3, 2, 3]

    def test_prefix_never_covers_trained_tokens(self):
        # Token 2 is identical in both samples but trained in the second, so it can't live in the shared prefix.
        layout = pack_shared_prefix_groups(
            input_ids=[[1, 2, 3, 10], [1, 2, 3, 20]],
            completion_mask=[[0, 0, 0, 1], [0, 0, 1, 1]],
            group_ids=[0, 0],
        )

        assert layout.branch_ids.tolist() == [-1, -1, 0, 0, 1, 1]
        assert layout.input_ids.tolist() == [1, 2, 3, 10, 3, 20]

    def test_prefix_leaves_at_least_one_token_per_branch(self):
        # The second sample is a strict prefix of the first; sharing all of it would leave it with an empty tail.
        layout = pack_shared_prefix_groups(
            input_ids=[[1, 2, 3, 10], [1, 2, 3]],
            completion_mask=[[0, 0, 0, 1], [0, 0, 1]],
            group_ids=[0, 0],
        )

        assert layout.branch_ids.tolist() == [-1, -1, 0, 0, 1]
        assert layout.input_ids.tolist() == [1, 2, 3, 10, 3]

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"min_group_size": 2},  # groups of one
            {"min_group_size": 4},  # group too small to bother
            {"min_prefix_length": 8},  # prefix too short to pay for itself
        ],
    )
    def test_falls_back_to_plain_packing(self, kwargs):
        input_ids = [[1, 2, 10], [1, 2, 20]]
        completion_mask = [[0, 0, 1], [0, 0, 1]]
        group_ids = [0, 1] if kwargs.get("min_group_size") == 2 else [0, 0]

        layout = pack_shared_prefix_groups(input_ids, completion_mask, group_ids, **kwargs)

        assert layout.input_ids.tolist() == [1, 2, 10, 1, 2, 20]
        assert layout.branch_ids.tolist() == [0, 0, 0, 0, 0, 0]
        assert layout.segment_ids.tolist() == [0, 0, 0, 1, 1, 1]
        assert layout.position_ids.tolist() == [0, 1, 2, 0, 1, 2]
        assert layout.dedup_ratio == 0.0
        assert layout.target_ids.tolist() == [10, 20]
        assert layout.gather_idx.tolist() == [1, 4]

    def test_groups_do_not_share_across_segments(self):
        # Identical prompts in different groups stay separate: grouping is by `group_ids`, not by content.
        layout = pack_shared_prefix_groups(
            input_ids=[[1, 2, 10], [1, 2, 20]],
            completion_mask=[[0, 0, 1], [0, 0, 1]],
            group_ids=[0, 1],
        )

        assert layout.segment_ids.tolist() == [0, 0, 0, 1, 1, 1]
        assert layout.num_tokens == layout.num_tokens_unpacked

    def test_interleaved_groups_are_bucketed(self):
        layout = pack_shared_prefix_groups(
            input_ids=[[1, 2, 10], [3, 4, 30], [1, 2, 20]],
            completion_mask=[[0, 0, 1]] * 3,
            group_ids=[0, 1, 0],
        )

        assert layout.input_ids.tolist() == [1, 2, 10, 20, 3, 4, 30]
        assert layout.branch_ids.tolist() == [-1, -1, 0, 1, 0, 0, 0]
        assert layout.segment_ids.tolist() == [0, 0, 0, 0, 1, 1, 1]
        # Targets stay in input-sample order regardless of how the tokens were bucketed.
        assert layout.target_sample_idx.tolist() == [0, 1, 2]
        assert layout.input_ids[layout.gather_idx].tolist() == [2, 4, 2]

    def test_mismatched_input_lengths_raise(self):
        with pytest.raises(ValueError, match="must have the same length"):
            pack_shared_prefix_groups(input_ids=[[1, 2]], completion_mask=[[0, 1], [0, 1]], group_ids=[0, 0])
