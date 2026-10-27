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
#     "openenv-ortools-env @ git+https://github.com/huggingface/OpenEnv.git#subdirectory=envs/ortools_env",
# ]
# ///

"""
GRPO training on hard operations research problems with the OpenEnv OR-Tools environment.

The model gets a problem (job shop, RCPSP, vehicle routing or weighted tardiness), a Python sandbox (`run_python`) and
submits a solution (level 1) or a `solve(instance)` algorithm (level 2). The reward is the OR-Tools reference objective
divided by the model's, so 1.0 matches the reference and above 1.0 beats it. See
[LLMs Can Design Near-Optimal OR Algorithms](https://huggingface.co/papers/2608.27296) for the setup.

Start the environment server (from an OpenEnv checkout):

```sh
MAX_CONCURRENT_ENVS=64 PYTHONPATH=src:envs uvicorn ortools_env.server.app:app --port 8001
```

Optionally precompute references with a long time limit (otherwise they are computed at submission time):

```sh
PYTHONPATH=src:envs python -m ortools_env.build_dataset --problem tardiness --sizes n40 n100 --num-rows 512 \
    --reference-time-limit 30 --out tardiness_l1.jsonl
```

Train:

```sh
python examples/grpo_ortools/grpo_ortools.py --env-url http://localhost:8001 --dataset tardiness_l1.jsonl
```
"""

import argparse
import json
import random

from datasets import Dataset
from openenv.core.env_server.mcp_types import CallToolAction
from ortools_env import ORToolsEnv

from trl import GRPOConfig, GRPOTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="GRPO training with the OpenEnv OR-Tools environment.")
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--env-url", default="http://localhost:8001")
    parser.add_argument("--dataset", default=None, help="JSONL from `ortools_env.build_dataset`.")
    parser.add_argument("--problem", default="tardiness", help="Problem to generate when no dataset is given.")
    parser.add_argument("--size", default="n10", help="Size to generate when no dataset is given.")
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--dataset-size", type=int, default=512)
    parser.add_argument("--allow-ortools", action="store_true")
    parser.add_argument("--compute-budget", type=float, default=60.0, help="Sandbox seconds per episode.")
    parser.add_argument("--max-tool-calls", type=int, default=4, help="Python and check calls per episode.")
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--per-device-train-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-completion-length", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--use-peft", action="store_true", help="Train LoRA adapters instead of the full model.")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--use-vllm", action="store_true")
    parser.add_argument("--output-dir", default="grpo_ortools")
    parser.add_argument("--report-to", default="none")
    return parser.parse_args()


def reward_func(environments, **kwargs):
    return [env.reward for env in environments]


def main():
    args = parse_args()

    if args.dataset is not None:
        with open(args.dataset) as f:
            rows = [json.loads(line) for line in f]
    else:
        rng = random.Random(0)
        rows = [
            {"problem": args.problem, "size": args.size, "level": args.level, "instance_seed": rng.randrange(2**31)}
            for _ in range(args.dataset_size)
        ]
    for row in rows:
        row["allow_ortools"] = args.allow_ortools
        row["compute_budget_s"] = args.compute_budget
        row["max_python_calls"] = args.max_tool_calls
    # The task prompt comes from the environment's `reset`, which is appended to this (empty) user message.
    keys = sorted({key for row in rows for key in row})
    columns = {key: [row.get(key) for row in rows] for key in keys}
    dataset = Dataset.from_dict({"prompt": [[{"role": "user", "content": ""}]] * len(rows), **columns})

    class ORToolsToolEnv:
        def __init__(self):
            self.client = ORToolsEnv(base_url=args.env_url).sync()
            self.reward = -1.0

        def reset(self, prompt=None, **kwargs) -> str:
            # An episode that never submits scores like an infeasible submission.
            self.reward = -1.0
            # Columns that are absent from a row come back as None; let the environment use its defaults.
            result = self.client.reset(**{k: v for k, v in kwargs.items() if v is not None})
            return result.observation.metadata["prompt"]

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

    peft_config = None
    if args.use_peft:
        from peft import LoraConfig

        peft_config = LoraConfig(r=32, lora_alpha=64, target_modules="all-linear", task_type="CAUSAL_LM")

    trainer = GRPOTrainer(
        model=args.model,
        train_dataset=dataset,
        reward_funcs=reward_func,
        peft_config=peft_config,
        args=GRPOConfig(
            output_dir=args.output_dir,
            chat_template_kwargs={"enable_thinking": args.enable_thinking},
            learning_rate=args.learning_rate,
            lr_scheduler_type="constant",
            temperature=args.temperature,
            num_generations=args.num_generations,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_completion_length=args.max_completion_length,
            max_tool_calling_iterations=args.max_tool_calls + 1,
            max_steps=args.max_steps,
            use_vllm=args.use_vllm,
            vllm_mode="colocate",
            gradient_checkpointing=True,
            bf16=True,
            logging_steps=1,
            log_completions=True,
            num_completions_to_print=1,
            report_to=args.report_to,
        ),
        environment_factory=ORToolsToolEnv,
    )
    trainer.train()


if __name__ == "__main__":
    main()
