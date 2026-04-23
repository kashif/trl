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

from dataclasses import dataclass, field

from trl.trainer.base_config import _BaseConfig


@dataclass
class AsyncGRPOConfig(_BaseConfig):
    # docstyle-ignore
    r"""
    Configuration class for the [`AsyncGRPOTrainer`].

    This class includes only the parameters that are specific to asynchronous GRPO training. For a full list of
    training arguments, please refer to the [`~transformers.TrainingArguments`] documentation. Note that default values
    in this class may differ from those in [`~transformers.TrainingArguments`].

    Parameters:
        > Parameters that control generation

        num_generations (`int`, *optional*, defaults to `8`):
            Number of generations per prompt to sample.
        max_completion_length (`int`, *optional*, defaults to `2048`):
            Maximum number of tokens to generate per completion.
        temperature (`float`, *optional*, defaults to `1.0`):
            Temperature for sampling. The higher the temperature, the more random the completions.
        chat_template_kwargs (`dict[str, Any]`, *optional*):
            Additional keyword arguments to pass to the `apply_chat_template` function when generating completions.
        max_tool_calling_iterations (`int`, *optional*):
            Maximum number of tool-calling turns when training an agent. If `None`, there is no limit and generation
            stops when the model generates a response turn with no tool calls or when the total response length reaches
            `max_completion_length`.

        > Parameters that control the vLLM server

        vllm_server_base_url (`str`, *optional*, defaults to `"http://localhost:8000"`):
            Base URL of the vLLM server used for generation (e.g., `"http://localhost:8000"`).
        vllm_server_timeout (`float`, *optional*, defaults to `240.0`):
            Total timeout duration in seconds to wait for the vLLM server to be ready.
        request_timeout (`int`, *optional*, defaults to `600`):
            Timeout in seconds for individual HTTP requests to the vLLM server.

        > Parameters that control the training

        epsilon (`float`, *optional*, defaults to `0.2`):
            Lower-bound epsilon value for clipping.
        epsilon_high (`float`, *optional*, defaults to `0.2`):
            Upper-bound epsilon value for clipping.
        use_delight (`bool`, *optional*, defaults to `False`):
            Whether to gate per-token policy-gradient terms with the Delightful Policy Gradient sigmoid of
            delight = advantage × surprisal, as introduced in the [Delightful Policy Gradient
            paper](https://huggingface.co/papers/2603.14608). Temperature η is fixed at 1.
        use_kondo_gate (`bool`, *optional*, defaults to `False`):
            Whether to enable the per-sample Kondo gate introduced in the [Does This Gradient Spark Joy?
            paper](https://huggingface.co/papers/2603.20526). When enabled, each training step draws a Bernoulli
            gate on the sample-level delight; gated-out steps skip the full forward and backward pass. The
            surprisal used for screening is the pre-computed `old_log_probs` from the rollout (§3.2 of the paper
            shows approximate delight is sufficient for screening).
        kondo_gate_rate (`float`, *optional*, defaults to `1.0`):
            Target fraction ρ of training steps that receive a backward pass. `1.0` is a no-op even when the gate
            is enabled. Smaller values keep only the highest-delight samples.
        kondo_gate_temperature (`float`, *optional*, defaults to `1.0`):
            Temperature η in the Kondo gate Bernoulli probability σ((χ − λ) / η).
        kondo_gate_history_size (`int`, *optional*, defaults to `1024`):
            Size of the ring buffer of past sample delights used to compute the adaptive threshold
            λ = quantile_{1−ρ}.
        kondo_gate_warmup (`int`, *optional*, defaults to `32`):
            Never gate until the history contains at least this many sample delights.

        > Parameters that control the async rollout pipeline

        max_inflight_tasks (`int`, *optional*, defaults to `-1`):
            Maximum number of concurrent generation tasks sent to the vLLM server. Defaults to `-1` (auto), which
            sets it to `max_staleness * per_device_train_batch_size * gradient_accumulation_steps * num_processes`.
            If using tool-use environments, you may want to set this manually based on how many parallel environments
            you can run.
        max_staleness (`int`, *optional*, defaults to `4`):
            Maximum number of weight update steps a rollout sample can lag behind the current model version before
            being discarded.
        queue_maxsize (`int`, *optional*, defaults to `1024`):
            Maximum number of rollout samples to buffer in the rollout queue.
        weight_sync_steps (`int`, *optional*, defaults to `1`):
            Number of training steps between weight synchronizations to the vLLM server.

        > Parameters that control the logging

        log_completions (`bool`, *optional*, defaults to `False`):
            Whether to log a sample of (prompt, completion) pairs every `logging_steps` steps.
        num_completions_to_print (`int`, *optional*, defaults to `3`):
            Number of completions to print when `log_completions=True`.

    > [!NOTE]
    > These parameters have default values different from [`~transformers.TrainingArguments`]:
    > - `logging_steps`: Defaults to `10` instead of `500`.
    > - `gradient_checkpointing`: Defaults to `True` instead of `False`.
    > - `bf16`: Defaults to `True` if `fp16` is not set, instead of `False`.
    > - `learning_rate`: Defaults to `1e-6` instead of `5e-5`.
    """

    # Parameters whose default values are overridden from TrainingArguments
    learning_rate: float = field(
        default=1e-6,
        metadata={"help": "The initial learning rate for AdamW."},
    )
    logging_steps: float = field(
        default=1,
        metadata={
            "help": "Log every X update steps. Should be an integer or a float in range `[0,1)`. If smaller than 1, "
            "will be interpreted as ratio of total training steps."
        },
    )

    # Parameters that control generation
    num_generations: int = field(
        default=8,
        metadata={"help": "Number of generations per prompt to sample."},
    )
    max_completion_length: int = field(
        default=2048,
        metadata={"help": "Maximum number of tokens to generate per completion."},
    )
    temperature: float = field(
        default=1.0,
        metadata={"help": "Temperature for sampling. The higher the temperature, the more random the completions."},
    )
    chat_template_kwargs: dict | None = field(
        default=None,
        metadata={
            "help": "Additional keyword arguments to pass to the `apply_chat_template` function when generating "
            "completions."
        },
    )
    max_tool_calling_iterations: int | None = field(
        default=None,
        metadata={
            "help": "Maximum number of tool-calling turns when training an agent. If `None`, there is no limit and "
            "generation stops when the model generates a response turn with no tool calls or when the total response "
            "length reaches `max_completion_length`."
        },
    )

    # Parameters that control the vLLM server
    vllm_server_base_url: str = field(
        default="http://localhost:8000",
        metadata={"help": "Base URL of the vLLM server used for generation (e.g., 'http://localhost:8000')."},
    )
    vllm_server_timeout: float = field(
        default=240.0,
        metadata={
            "help": "Total timeout duration in seconds to wait for the vLLM server to be ready. If the server is not "
            "up after the timeout, a `TimeoutError` is raised."
        },
    )
    request_timeout: int = field(
        default=600,
        metadata={"help": "Timeout in seconds for individual HTTP requests to the vLLM server."},
    )

    # Parameters that control the training
    epsilon: float = field(
        default=0.2,
        metadata={"help": "Lower-bound epsilon value for clipping."},
    )
    epsilon_high: float = field(
        default=0.2,
        metadata={"help": "Upper-bound epsilon value for clipping."},
    )
    use_delight: bool = field(
        default=False,
        metadata={
            "help": "Whether to gate per-token policy-gradient terms with the Delightful Policy Gradient sigmoid "
            "of delight = advantage × surprisal (https://huggingface.co/papers/2603.14608). Temperature η is fixed "
            "at 1."
        },
    )
    use_kondo_gate: bool = field(
        default=False,
        metadata={
            "help": "Whether to enable the per-sample Kondo gate (https://huggingface.co/papers/2603.20526). When "
            "enabled, each training step draws a Bernoulli gate on the sample-level delight; gated-out steps skip "
            "the full forward and backward pass. Uses `old_log_probs` from the rollout as the surprisal proxy."
        },
    )
    kondo_gate_rate: float = field(
        default=1.0,
        metadata={
            "help": "Target fraction ρ of training steps that receive a backward pass. 1.0 is a no-op. Smaller "
            "values keep only the highest-delight samples."
        },
    )
    kondo_gate_temperature: float = field(
        default=1.0,
        metadata={"help": "Temperature η in the Kondo gate Bernoulli probability σ((χ − λ) / η)."},
    )
    kondo_gate_history_size: int = field(
        default=1024,
        metadata={"help": "Size of the ring buffer of past sample delights used to compute λ = quantile_{1−ρ}."},
    )
    kondo_gate_warmup: int = field(
        default=32,
        metadata={"help": "Never gate until the history contains at least this many sample delights."},
    )

    # Parameters that control the async rollout pipeline
    max_inflight_tasks: int = field(
        default=-1,
        metadata={
            "help": "Maximum number of concurrent generation tasks sent to the vLLM server. Defaults to -1 (auto), "
            "which sets it to `max_staleness * per_device_train_batch_size * gradient_accumulation_steps * "
            "num_processes`. Generating more samples than this is wasteful since they will be discarded as stale "
            "before the trainer can consume them. If using tool-use environments, you may want to set this manually "
            "based on how many parallel environments you can run."
        },
    )
    max_staleness: int = field(
        default=4,
        metadata={
            "help": "Maximum number of weight update steps a rollout sample can lag behind the current model version "
            "before being discarded."
        },
    )
    queue_maxsize: int = field(
        default=1024,
        metadata={"help": "Maximum number of rollout samples to buffer in the rollout queue."},
    )
    weight_sync_steps: int = field(
        default=1,
        metadata={"help": "Number of training steps between weight synchronizations to the vLLM server."},
    )

    # Parameters that control the logging
    log_completions: bool = field(
        default=False,
        metadata={
            "help": "Whether to log a sample of (prompt, completion) pairs every `logging_steps` steps. If `rich` is "
            "installed, it prints the sample. If `wandb` logging is enabled, it logs it to `wandb`."
        },
    )
    num_completions_to_print: int = field(
        default=3,
        metadata={"help": "Number of completions to print when `log_completions=True`."},
    )

    def __post_init__(self):
        super().__post_init__()

        if self.use_kondo_gate:
            if not (0.0 < self.kondo_gate_rate <= 1.0):
                raise ValueError(f"`kondo_gate_rate` must be in (0, 1], got {self.kondo_gate_rate}")
            if self.kondo_gate_temperature <= 0.0:
                raise ValueError(f"`kondo_gate_temperature` must be > 0, got {self.kondo_gate_temperature}")
            if self.kondo_gate_warmup <= 0:
                raise ValueError(f"`kondo_gate_warmup` must be > 0, got {self.kondo_gate_warmup}")
            if self.kondo_gate_history_size < self.kondo_gate_warmup:
                raise ValueError(
                    f"`kondo_gate_history_size` ({self.kondo_gate_history_size}) must be >= "
                    f"`kondo_gate_warmup` ({self.kondo_gate_warmup})."
                )

        # Accelerator config: required for the async IterableDataset-backed dataloader to work correctly.
        # split_batches=True and dispatch_batches=True ensure that the main process drives the dataloader
        # and batches are broadcast to other processes rather than each process pulling independently.
        if not hasattr(self, "accelerator_config") or self.accelerator_config is None:
            self.accelerator_config = {"split_batches": True, "dispatch_batches": True}
        elif isinstance(self.accelerator_config, dict):
            self.accelerator_config["split_batches"] = True
            self.accelerator_config["dispatch_batches"] = True
        else:
            self.accelerator_config.split_batches = True
            self.accelerator_config.dispatch_batches = True
