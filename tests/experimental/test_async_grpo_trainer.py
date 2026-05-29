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

import itertools
import queue

import numpy as np
import pytest
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers.utils import is_renderers_available

from trl.experimental.async_grpo import AsyncGRPOConfig, AsyncGRPOTrainer
from trl.experimental.async_grpo.async_rollout_worker import RolloutSample

from ..testing_utils import TrlTestCase, require_vllm


require_renderers = pytest.mark.skipif(not is_renderers_available(), reason="test requires renderers")


def dummy_reward_func(completions, **kwargs):
    return [float(hash(c[0]["content"]) % 100) / 100.0 for c in completions]


class _StubRolloutWorker:
    """Minimal rollout worker stub for testing the trainer in isolation."""

    def __init__(self, tokenizer, dataset, num_generations: int = 8, samples_per_weight_sync: int = 10):
        self.rollout_buffer = queue.Queue()
        self._samples_per_weight_sync = samples_per_weight_sync
        self._model_version = 0
        self._sample_iter = self._make_sample_iter(tokenizer, dataset, num_generations)

    def _make_sample_iter(self, tokenizer, dataset, num_generations):
        for row in itertools.cycle(dataset):
            completions = [
                [{"role": "assistant", "content": f"{row['completion'][0]['content']} {idx}"}]
                for idx in range(num_generations)
            ]
            prompt_completions = [row["prompt"] + completion for completion in completions]
            prompt_ids = tokenizer.apply_chat_template(
                row["prompt"], tokenize=True, add_generation_prompt=True, return_dict=False
            )
            prompt_completion_ids = tokenizer.apply_chat_template(
                prompt_completions, tokenize=True, add_generation_prompt=False, return_dict=False
            )
            rewards = np.array(dummy_reward_func(completions))
            advantages = (rewards - rewards.mean()) / rewards.std()
            for idx in range(num_generations):
                completion_ids = prompt_completion_ids[idx][len(prompt_ids) :]
                yield RolloutSample(
                    prompt=row["prompt"],
                    completion=completions[idx],
                    input_ids=prompt_ids + completion_ids,
                    completion_mask=[0] * len(prompt_ids) + [1] * len(completion_ids),
                    old_log_probs=[0.0] * len(prompt_ids) + [-0.5] * len(completion_ids),
                    advantage=float(advantages[idx]),
                    model_version=self._model_version,
                    metrics={"reward": float(rewards[idx]), "reward_std": float(rewards.std())},
                )

    def _fill_queue(self):
        for _ in range(self._samples_per_weight_sync):
            self.rollout_buffer.put(next(self._sample_iter))

    def start(self):
        self._fill_queue()

    def update_model_version(self, version):
        self._model_version = version
        self._fill_queue()

    def stop(self):
        pass

    def pause(self):
        pass

    def resume(self):
        pass

    def send_weights(self, iterator):
        pass

    def check_health(self, stale_after_s):
        pass


class TestAsyncGRPOTrainer(TrlTestCase):
    def test_init_minimal(self):
        # Test that AsyncGRPOTrainer can be instantiated with only model, reward_model and train_dataset
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_completion", split="train")
        AsyncGRPOTrainer(
            model=model_id,
            reward_funcs=dummy_reward_func,
            train_dataset=dataset,
            rollout_worker=_StubRolloutWorker(AutoTokenizer.from_pretrained(model_id), dataset, num_generations=3),
        )

    def test_train(self):
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_completion", split="train")

        training_args = AsyncGRPOConfig(
            output_dir=self.tmp_dir,
            learning_rate=0.1,  # use higher lr because gradients are tiny and default lr can stall updates
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=8,  # reduce the completion length to reduce memory usage
            vllm_server_timeout=5.0,  # short timeout so test fails fast if queue runs dry
            report_to="none",
        )
        trainer = AsyncGRPOTrainer(
            model=model_id,
            reward_funcs=dummy_reward_func,  # unused: the stub pre-computes rewards, but the trainer requires this argument
            args=training_args,
            train_dataset=dataset,
            rollout_worker=_StubRolloutWorker(AutoTokenizer.from_pretrained(model_id), dataset, num_generations=3),
        )

        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check that the params have changed
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            assert not torch.equal(param, new_param), f"Parameter {n} has not changed."


CALC_TOOL = {
    "type": "function",
    "function": {
        "name": "calc",
        "description": "Compute an arithmetic expression.",
        "parameters": {
            "type": "object",
            "properties": {"expr": {"type": "string"}},
            "required": ["expr"],
        },
    },
}
ASSISTANT_TOOL_CALL = {
    "role": "assistant",
    "content": "",
    "tool_calls": [{"type": "function", "function": {"name": "calc", "arguments": {"expr": "2+2"}}}],
}
TOOL_MESSAGE = {"role": "tool", "name": "calc", "content": "4"}


class _SuffixStub:
    """Carries only the attributes ``AsyncRolloutWorker._get_tool_suffix_ids`` reads, so the real
    (apply_chat_template dummy-diff) method can be exercised without standing up a vLLM server."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.chat_template = None
        self.chat_template_kwargs = {}


@require_vllm
@require_renderers
class TestAsyncRolloutWorkerRenderer(TrlTestCase):
    """The worker now bridges tool turns with a per-family renderer. These cover the silent failures of
    the apply_chat_template dummy-diff path that motivated the switch.
    """

    def test_glm_observation_token_not_doubled(self):
        # GLM has no end-of-turn token; the inference engine stops on the next turn's <|observation|>.
        # The dummy-diff suffix *also* starts with <|observation|>, so naive stitching doubles it. The
        # renderer's bridge extends the engine-stopped stream verbatim and emits exactly one.
        from trl.experimental.async_grpo.async_rollout_worker import AsyncRolloutWorker

        tokenizer = AutoTokenizer.from_pretrained("zai-org/GLM-4.5", trust_remote_code=True)
        renderer = tokenizer.get_renderer("glm-4.5")
        observation_id = tokenizer.convert_tokens_to_ids("<|observation|>")

        prompt = [{"role": "user", "content": "What's 2+2?"}]
        prompt_ids = renderer.render_ids(prompt, tools=[CALC_TOOL], add_generation_prompt=True)
        full = renderer.render_ids(prompt + [ASSISTANT_TOOL_CALL], tools=[CALC_TOOL])
        # Simulate the engine stopping on <|observation|> at the end of the sampled tool-call turn.
        turn_ids = list(full[len(prompt_ids) :]) + [observation_id]

        bridged = renderer.bridge_to_next_turn(
            previous_prompt_ids=prompt_ids,
            previous_completion_ids=turn_ids,
            new_messages=[TOOL_MESSAGE],
            tools=[CALC_TOOL],
        )
        # This is exactly how the worker assembles the next prompt.
        suffix_ids = list(bridged.token_ids)[len(prompt_ids) + len(turn_ids) :]
        sequence = prompt_ids + turn_ids + suffix_ids
        assert sequence.count(observation_id) == 1, "renderer bridge must not duplicate <|observation|>"

        # The fallback dummy-diff suffix begins with <|observation|>, which would double it against the
        # engine-stopped turn — the silent corruption the renderer path avoids.
        dummy_suffix = AsyncRolloutWorker._get_tool_suffix_ids(_SuffixStub(tokenizer), [TOOL_MESSAGE])
        assert dummy_suffix[0] == observation_id

    def test_minimax_bare_tool_message_not_tokenizable_but_renderer_resolves(self):
        # Building a tool bridge by tokenizing a bare tool message — an optimization one might reach for —
        # does not generalize: MiniMax's template raises on a lone tool message. The worker sidesteps this
        # by resolving a per-family renderer, which bridges without that assumption.
        tokenizer = AutoTokenizer.from_pretrained("MiniMaxAI/MiniMax-M2", trust_remote_code=True)
        with pytest.raises(Exception):
            tokenizer.apply_chat_template([TOOL_MESSAGE], add_generation_prompt=True, tokenize=True, return_dict=False)
        assert type(tokenizer.get_renderer(strict=True)).__name__ == "MiniMaxM2Renderer"
