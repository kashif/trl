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

# /// script
# dependencies = [
#     "trl[vllm]",
#     "peft",
#     "openenv-ortools-env @ git+https://github.com/huggingface/OpenEnv.git#subdirectory=envs/ortools_env",
# ]
# ///

"""
LoRA AsyncGRPO on hard operations research problems with the OpenEnv OR-Tools environment.

Same task as `examples/grpo_ortools`: the model gets a problem, a Python sandbox (`run_python`) and submits a solution
(level 1) or a `solve(instance)` algorithm (level 2) through a tool call. Only a submission is rewarded, so the run also
teaches the model to call tools. See [LLMs Can Design Near-Optimal OR Algorithms](https://huggingface.co/papers/2608.27296).

The adapter is synced to vLLM as a directory, so the server and the trainer can share one GPU. Start the environment
server (from an OpenEnv checkout), the vLLM server and the trainer:

```bash
MAX_CONCURRENT_ENVS=64 PYTHONPATH=src:envs uvicorn ortools_env.server.app:app --port 8001

VLLM_SERVER_DEV_MODE=1 VLLM_ALLOW_RUNTIME_LORA_UPDATING=1 \
vllm serve Qwen/Qwen3-4B \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.3 \
    --logprobs-mode processed_logprobs \
    --enable-lora --max-lora-rank 32 --max-loras 6

PYTHONPATH=src:envs python -m ortools_env.build_dataset --problem tardiness --sizes n40 --level 2 --num-rows 64 \
    --num-eval-instances 2 --out tardiness_n40_l2.jsonl

python examples/async_grpo_ortools/async_grpo_ortools.py --dataset tardiness_n40_l2.jsonl
```
"""

import argparse
import functools
import json
import os

from datasets import Dataset
from openenv.core.env_server.mcp_types import CallToolAction
from ortools_env import ORToolsEnv
from peft import LoraConfig

from trl.experimental.async_grpo import AsyncGRPOConfig, AsyncGRPOTrainer


# Module level, because the rollout worker pickles it into a child process.
class ORToolsToolEnv:
    def __init__(self, env_url: str):
        self.client = ORToolsEnv(base_url=env_url).sync()
        self.reward = -1.0

    def reset(self, **kwargs) -> str:
        # An episode that never submits scores like an infeasible submission.
        self.reward = -1.0
        # Columns that are absent from a row come back as None; let the environment use its defaults.
        result = self.client.reset(**{k: v for k, v in kwargs.items() if v is not None})
        return result.observation.metadata["prompt"]

    def get_reward(self) -> float:
        return self.reward

    def _submit(self, tool_name: str, **arguments) -> str:
        result = self.client.step(CallToolAction(tool_name=tool_name, arguments=arguments))
        if result.observation.error is not None:
            # Rejected before scoring (e.g. a solution passed as a string), so the episode goes on.
            raise ValueError(result.observation.error.message)
        # The episode ends at the first accepted submission; anything else (e.g. the wrong submit tool for the
        # level) leaves it open with no reward yet.
        if result.done:
            self.reward = result.reward
        return result.observation.result["content"][0]["text"]

    def run_python(self, code: str) -> str:
        """
        Run a Python script in the sandbox and return its output. The instance is bound to the global `instance`.

        Args:
            code: The Python source to run.

        Returns:
            The script output and the remaining compute budget.
        """
        return self.client.call_tool("run_python", code=code)

    def check_solution(self, solution: dict) -> str:
        """
        Check a candidate solution for the instance (level 1) without submitting it. Counts as a Python call.

        Args:
            solution: The solution as a JSON object (not a string), in the format given in the problem statement.

        Returns:
            The objective and the naive objective, or why the solution is infeasible.
        """
        return self.client.call_tool("check_solution", solution=solution)

    def test_algorithm(self, code: str) -> str:
        """
        Run `solve` on the example instance (level 2) without submitting it. Counts as a Python call.

        Args:
            code: The Python source defining `solve`.

        Returns:
            The objective and the naive objective, or why `solve` failed.
        """
        return self.client.call_tool("test_algorithm", code=code)

    def submit_solution(self, solution: dict) -> str:
        """
        Submit the final solution for the instance (level 1). Ends the episode.

        Args:
            solution: The solution as a JSON object (not a string), in the format given in the problem statement.

        Returns:
            The objective, the naive and reference objectives and the score.
        """
        return self._submit("submit_solution", solution=solution)

    def submit_algorithm(self, code: str) -> str:
        """
        Submit Python code defining `solve(instance) -> solution` (level 2). Ends the episode.

        Args:
            code: The Python source defining `solve`.

        Returns:
            The score on each evaluation instance and their mean.
        """
        return self._submit("submit_algorithm", code=code)


def parse_args():
    parser = argparse.ArgumentParser(description="LoRA AsyncGRPO with the OpenEnv OR-Tools environment.")
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--env-url", default="http://localhost:8001")
    parser.add_argument("--dataset", required=True, help="JSONL from `ortools_env.build_dataset`.")
    parser.add_argument("--allow-ortools", action="store_true")
    parser.add_argument("--compute-budget", type=float, default=60.0, help="Sandbox seconds per episode.")
    parser.add_argument("--max-tool-calls", type=int, default=4, help="Python and check calls per episode.")
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--max-completion-length", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument(
        "--max-inflight-tasks", type=int, default=-1, help="Concurrent rollouts; lower it when vLLM shares the GPU."
    )
    parser.add_argument("--save-steps", type=int, default=10)
    parser.add_argument("--output-dir", default="async_grpo_ortools")
    parser.add_argument("--report-to", default="none")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.dataset) as f:
        rows = [json.loads(line) for line in f]
    for row in rows:
        row["allow_ortools"] = args.allow_ortools
        row["compute_budget_s"] = args.compute_budget
        row["max_python_calls"] = args.max_tool_calls
    # No `prompt` column: the task prompt comes from the environment's `reset`.
    keys = sorted({key for row in rows for key in row})
    dataset = Dataset.from_dict({key: [row.get(key) for row in rows] for key in keys})

    config = AsyncGRPOConfig(
        output_dir=args.output_dir,
        vllm_server_base_url=f"http://localhost:{os.environ.get('VLLM_PORT', 8000)}",
        dtype="bfloat16",
        chat_template_kwargs={"enable_thinking": False},
        learning_rate=args.learning_rate,
        lr_scheduler_type="constant",
        num_generations=args.num_generations,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        max_completion_length=args.max_completion_length,
        max_tool_calling_iterations=args.max_tool_calls + 1,
        max_steps=args.max_steps,
        max_inflight_tasks=args.max_inflight_tasks,
        save_steps=args.save_steps,
        logging_steps=1,
        log_completions=True,
        num_completions_to_print=1,
        report_to=args.report_to,
    )
    trainer = AsyncGRPOTrainer(
        model=args.model,
        args=config,
        train_dataset=dataset,
        # A plain LoRA config, so vLLM can serve the adapter directly.
        peft_config=LoraConfig(r=32, lora_alpha=64, target_modules="all-linear"),
        environment_factory=functools.partial(ORToolsToolEnv, args.env_url),
    )
    trainer.train()

    trainer.end()


if __name__ == "__main__":
    main()
