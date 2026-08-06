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
import torch
from transformers import AutoModelForCausalLM

from trl.experimental.zorro import build_zorro_mask, pack_shared_prefix_groups
from trl.trainer.utils import patch_chunked_lm_head

from ..testing_utils import TrlTestCase


def make_group(num_rollouts, prompt_len, completion_len, group_id, first_token=0):
    """A rollout group sharing one prompt, in the shape [`pack_shared_prefix_groups`] consumes."""
    prompt = list(range(first_token, first_token + prompt_len))
    input_ids, completion_mask, group_ids = [], [], []
    for rollout in range(num_rollouts):
        completion = [1000 + 100 * group_id + 10 * rollout + i for i in range(completion_len)]
        input_ids.append(prompt + completion)
        completion_mask.append([0] * prompt_len + [1] * completion_len)
        group_ids.append(group_id)
    return input_ids, completion_mask, group_ids


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

    def test_sample_positions_recover_every_sample(self):
        input_ids = [[1, 2, 3, 40, 41], [1, 2, 3, 50], [7, 8, 70]]
        layout = pack_shared_prefix_groups(
            input_ids=input_ids,
            completion_mask=[[0, 0, 0, 1, 1], [0, 0, 0, 1], [0, 0, 1]],
            group_ids=[0, 0, 1],
        )

        for tokens, positions in zip(input_ids, layout.sample_positions, strict=True):
            assert layout.input_ids[positions].tolist() == tokens
            assert layout.position_ids[positions].tolist() == list(range(len(tokens)))

    def test_dedup_drops_duplicate_prompts(self):
        # Ported from the ZoRRo reference test suite (`tests/zorro_train/test_dedup.py::TestTokenSavings`): a batch of
        # `num_groups` prompts rolled out `num_rollouts` times each keeps one prompt block per group plus every
        # rollout's own completion.
        num_groups, num_rollouts, prompt_len, completion_len = 2, 3, 16, 4
        input_ids, completion_mask, group_ids = [], [], []
        for group_id in range(num_groups):
            group = make_group(num_rollouts, prompt_len, completion_len, group_id, first_token=100 * group_id)
            input_ids += group[0]
            completion_mask += group[1]
            group_ids += group[2]

        layout = pack_shared_prefix_groups(input_ids, completion_mask, group_ids)

        assert layout.num_tokens == num_groups * prompt_len + num_groups * num_rollouts * completion_len
        assert layout.num_tokens < layout.num_tokens_unpacked
        assert layout.target_ids.numel() == num_groups * num_rollouts * completion_len

    def test_mismatched_input_lengths_raise(self):
        with pytest.raises(ValueError, match="must have the same length"):
            pack_shared_prefix_groups(input_ids=[[1, 2]], completion_mask=[[0, 1], [0, 1]], group_ids=[0, 0])


class TestBuildZorroMask(TrlTestCase):
    @staticmethod
    def expected_mask(layout):
        """Union of the per-rollout causal masks: each sample sees its own tokens, causally, and nothing else."""
        mask = torch.zeros(layout.num_tokens, layout.num_tokens, dtype=torch.bool)
        for positions in layout.sample_positions:
            for query_rank, query in enumerate(positions):
                mask[query, positions[: query_rank + 1]] = True
        return mask

    def test_sdpa_mask_matches_per_rollout_causal_masks(self):
        input_ids, completion_mask, group_ids = make_group(3, prompt_len=5, completion_len=3, group_id=0)
        layout = pack_shared_prefix_groups(input_ids, completion_mask, group_ids)

        mask = build_zorro_mask(layout.segment_ids.unsqueeze(0), layout.branch_ids.unsqueeze(0), "sdpa")

        assert mask.shape == (1, 1, layout.num_tokens, layout.num_tokens)
        assert torch.equal(mask[0, 0], self.expected_mask(layout))

    def test_siblings_never_attend_to_each_other(self):
        input_ids, completion_mask, group_ids = make_group(2, prompt_len=3, completion_len=2, group_id=0)
        layout = pack_shared_prefix_groups(input_ids, completion_mask, group_ids)

        mask = build_zorro_mask(layout.segment_ids.unsqueeze(0), layout.branch_ids.unsqueeze(0), "sdpa")[0, 0]

        branches = layout.branch_ids
        first, second = (branches == 0).nonzero().flatten(), (branches == 1).nonzero().flatten()
        assert not mask[second][:, first].any()
        assert not mask[first][:, second].any()
        # ... while both still read the whole shared prefix.
        prefix = (branches == -1).nonzero().flatten()
        assert mask[second][:, prefix].all()

    def test_groups_are_isolated(self):
        input_ids, completion_mask, group_ids = make_group(2, prompt_len=3, completion_len=2, group_id=0)
        other = make_group(2, prompt_len=3, completion_len=2, group_id=1, first_token=50)
        layout = pack_shared_prefix_groups(input_ids + other[0], completion_mask + other[1], group_ids + other[2])

        mask = build_zorro_mask(layout.segment_ids.unsqueeze(0), layout.branch_ids.unsqueeze(0), "sdpa")[0, 0]

        first, second = (layout.segment_ids == 0).nonzero().flatten(), (layout.segment_ids == 1).nonzero().flatten()
        assert not mask[second][:, first].any()

    def test_eager_mask_is_additive(self):
        input_ids, completion_mask, group_ids = make_group(2, prompt_len=3, completion_len=2, group_id=0)
        layout = pack_shared_prefix_groups(input_ids, completion_mask, group_ids)

        mask = build_zorro_mask(
            layout.segment_ids.unsqueeze(0), layout.branch_ids.unsqueeze(0), "eager", dtype=torch.float32
        )

        allowed = self.expected_mask(layout)
        assert mask.dtype == torch.float32
        assert torch.equal(mask[0, 0] == 0, allowed)
        assert (mask[0, 0][~allowed] == torch.finfo(torch.float32).min).all()

    def test_unsupported_attn_implementation_raises(self):
        layout = pack_shared_prefix_groups([[1, 2, 3]], [[0, 0, 1]], [0])
        with pytest.raises(ValueError, match="no mask builder"):
            build_zorro_mask(layout.segment_ids.unsqueeze(0), layout.branch_ids.unsqueeze(0), "flash_attention_2")


class TestZorroEquivalence(TrlTestCase):
    """Packed rollouts must train on exactly what the unpacked ones would."""

    MODEL_ID = "trl-internal-testing/tiny-Qwen3ForCausalLM"
    CHUNK_SIZE = 16
    TEMPERATURE = 0.7

    def make_model(self):
        model = AutoModelForCausalLM.from_pretrained(self.MODEL_ID, attn_implementation="sdpa", dtype=torch.float32)
        patch_chunked_lm_head(model, chunk_size=self.CHUNK_SIZE, temperature=self.TEMPERATURE)
        return model

    @staticmethod
    def unpacked_logprobs(model, input_ids, completion_mask):
        """Run every sample on its own, the way training would without packing."""
        logprobs = []
        for tokens, mask in zip(input_ids, completion_mask, strict=True):
            ids = torch.tensor([tokens])
            outputs = model(input_ids=ids, labels=ids, completion_mask=torch.tensor([mask]), use_cache=False)
            logprobs.append(outputs["log_probs"][0][torch.tensor(mask[1:]).bool()])
        return torch.cat(logprobs)

    @staticmethod
    def packed_logprobs(model, layout):
        """Run the whole group as one packed row with the ZoRRo mask."""
        segment_ids, branch_ids = layout.segment_ids.unsqueeze(0), layout.branch_ids.unsqueeze(0)
        outputs = model(
            input_ids=layout.input_ids.unsqueeze(0),
            position_ids=layout.position_ids.unsqueeze(0),
            attention_mask=build_zorro_mask(segment_ids, branch_ids, "sdpa"),
            gather_idx=layout.gather_idx.unsqueeze(0),
            target_ids=layout.target_ids.unsqueeze(0),
            use_cache=False,
        )
        return outputs["log_probs"][0]

    def test_logprobs_match_the_unpacked_forward(self):
        input_ids, completion_mask, group_ids = make_group(4, prompt_len=12, completion_len=6, group_id=0)
        layout = pack_shared_prefix_groups(input_ids, completion_mask, group_ids)
        model = self.make_model()

        assert layout.dedup_ratio > 0.4  # the packing must actually be doing something

        with torch.no_grad():
            expected = self.unpacked_logprobs(model, input_ids, completion_mask)
            actual = self.packed_logprobs(model, layout)

        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)

    def test_logprobs_match_with_mixed_groups_and_ragged_lengths(self):
        first = make_group(3, prompt_len=9, completion_len=5, group_id=0)
        second = make_group(2, prompt_len=7, completion_len=3, group_id=1, first_token=40)
        lone = ([[60, 61, 62, 63]], [[0, 0, 1, 1]], [2])
        input_ids = first[0] + second[0] + lone[0]
        completion_mask = first[1] + second[1] + lone[1]
        group_ids = first[2] + second[2] + lone[2]
        # Ragged tails within a group: truncate one rollout of the first group.
        input_ids[1] = input_ids[1][:-2]
        completion_mask[1] = completion_mask[1][:-2]

        layout = pack_shared_prefix_groups(input_ids, completion_mask, group_ids)
        model = self.make_model()

        with torch.no_grad():
            expected = self.unpacked_logprobs(model, input_ids, completion_mask)
            actual = self.packed_logprobs(model, layout)

        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)

    def test_gradients_match_the_unpacked_forward(self):
        input_ids, completion_mask, group_ids = make_group(3, prompt_len=8, completion_len=4, group_id=0)
        layout = pack_shared_prefix_groups(input_ids, completion_mask, group_ids)

        unpacked_model = self.make_model()
        self.unpacked_logprobs(unpacked_model, input_ids, completion_mask).sum().backward()

        packed_model = self.make_model()
        self.packed_logprobs(packed_model, layout).sum().backward()

        unpacked_grads = dict(unpacked_model.named_parameters())
        for name, param in packed_model.named_parameters():
            reference = unpacked_grads[name].grad
            assert (param.grad is None) == (reference is None), name
            if param.grad is not None:
                torch.testing.assert_close(param.grad, reference, atol=1e-4, rtol=1e-4, msg=f"grad mismatch: {name}")
