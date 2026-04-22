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

from trl.experimental.async_grpo import AsyncGRPOConfig, AsyncGRPOTrainer
from trl.experimental.async_grpo.async_grpo_trainer import _KondoGateState
from trl.experimental.async_grpo.async_rollout_worker import RolloutSample

from ..testing_utils import TrlTestCase


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

    @pytest.mark.parametrize(
        "extra_args",
        [
            {},
            {"use_delight": True},
            {
                "use_delight": True,
                "use_kondo_gate": True,
                "kondo_gate_rate": 0.8,
                "kondo_gate_warmup": 1,
                "kondo_gate_history_size": 16,
            },
        ],
    )
    def test_training(self, extra_args):
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
            **extra_args,
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

        # Check that at least one parameter has changed. With the Kondo gate, some steps may be skipped
        # entirely, so we don't require *every* parameter to have moved.
        changed = [
            n
            for n, param in previous_trainable_params.items()
            if not torch.equal(param, trainer.model.get_parameter(n))
        ]
        assert changed, "No parameters changed during training."


class TestKondoGateState:
    def test_warmup_never_gates(self):
        gate = _KondoGateState(rate=0.5, temperature=1.0, history_size=8, warmup=4, device=torch.device("cpu"), seed=0)
        for i in range(3):
            should, prob, lam = gate.decide(torch.tensor(float(i)))
            assert should is True
            assert prob == 1.0
            assert lam == float("-inf")

    def test_rate_one_is_noop(self):
        gate = _KondoGateState(rate=1.0, temperature=1.0, history_size=8, warmup=1, device=torch.device("cpu"), seed=0)
        for i in range(10):
            should, prob, lam = gate.decide(torch.tensor(float(i)))
            assert should is True
            assert prob == 1.0

    def test_sharp_temperature_separates_high_low_delight(self):
        gate = _KondoGateState(
            rate=0.5, temperature=0.01, history_size=16, warmup=2, device=torch.device("cpu"), seed=0
        )
        # Fill the warmup buffer with two values — λ lands at 0.5.
        gate.decide(torch.tensor(0.0))
        gate.decide(torch.tensor(1.0))
        # High delight → prob ≈ 1; low delight → prob ≈ 0.
        _, high_prob, _ = gate.decide(torch.tensor(100.0))
        _, low_prob, _ = gate.decide(torch.tensor(-100.0))
        assert high_prob > 0.99
        assert low_prob < 0.01

    def test_seeded_generator_is_deterministic_across_ranks(self):
        # Two independently-constructed gates with the same seed must produce the same Bernoulli draws
        # given the same inputs -- required to keep DP collectives in sync.
        gate_a = _KondoGateState(
            rate=0.5, temperature=1.0, history_size=16, warmup=2, device=torch.device("cpu"), seed=42
        )
        gate_b = _KondoGateState(
            rate=0.5, temperature=1.0, history_size=16, warmup=2, device=torch.device("cpu"), seed=42
        )
        decisions_a, decisions_b = [], []
        for i in range(30):
            val = torch.tensor(float(i % 5) - 2.0)  # mixed positive / negative delights
            decisions_a.append(gate_a.decide(val)[0])
            decisions_b.append(gate_b.decide(val)[0])
        assert decisions_a == decisions_b

    def test_buffer_wraps_at_history_size(self):
        gate = _KondoGateState(rate=0.5, temperature=1.0, history_size=4, warmup=1, device=torch.device("cpu"), seed=0)
        for i in range(10):
            gate.decide(torch.tensor(float(i)))
        # After 10 appends with history_size=4, only the last 4 delights remain: {6, 7, 8, 9}.
        assert gate._count == 4
        assert set(gate._buffer.tolist()) == {6.0, 7.0, 8.0, 9.0}
